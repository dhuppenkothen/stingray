from __future__ import division, absolute_import, print_function

import numpy as np
import scipy
import scipy.stats
import scipy.fftpack
import scipy.optimize

from stingray.lightcurve import Lightcurve
from stingray.utils import rebin_data, simon, rebin_data_log
from stingray.exceptions import StingrayError
from stingray.gti import cross_two_gtis, bin_intervals_from_gtis, check_gtis
import copy

__all__ = ["Crossspectrum", "AveragedCrossspectrum", "coherence", "time_lag"]


def coherence(lc1, lc2):
    """
    Estimate coherence function of two light curves.
    For details on the definition of the coherence, see [vaughan-1996].

    Parameters
    ----------
    lc1: :class:`stingray.Lightcurve` object
        The first light curve data for the channel of interest.

    lc2: :class:`stingray.Lightcurve` object
        The light curve data for reference band

    Returns
    -------
    coh : ``np.ndarray``
        The array of coherence versus frequency

    References
    ----------
    .. [vaughan-1996] http://iopscience.iop.org/article/10.1086/310430/pdf

    """

    if not isinstance(lc1, Lightcurve):
        raise TypeError("lc1 must be a lightcurve.Lightcurve object")

    if not isinstance(lc2, Lightcurve):
        raise TypeError("lc2 must be a lightcurve.Lightcurve object")

    cs = Crossspectrum(lc1, lc2, norm='none')

    return cs.coherence()


def time_lag(lc1, lc2):
    """
    Estimate the time lag of two light curves.
    Calculate time lag and uncertainty.

        Equation from Bendat & Piersol, 2011 [bendat-2011]_.

        Returns
        -------
        lag : np.ndarray
            The time lag

        lag_err : np.ndarray
            The uncertainty in the time lag

        References
        ----------

        .. [bendat-2011] https://www.wiley.com/en-us/Random+Data%3A+Analysis+and+Measurement+Procedures%2C+4th+Edition-p-9780470248775

    """

    if not isinstance(lc1, Lightcurve):
        raise TypeError("lc1 must be a lightcurve.Lightcurve object")

    if not isinstance(lc2, Lightcurve):
        raise TypeError("lc2 must be a lightcurve.Lightcurve object")

    cs = Crossspectrum(lc1, lc2, norm='none')
    lag = cs.time_lag()

    return lag


class Crossspectrum(object):
    """
    Make a cross spectrum from a (binned) light curve.
    You can also make an empty :class:`Crossspectrum` object to populate with your
    own Fourier-transformed data (this can sometimes be useful when making
    binned power spectra).

    Parameters
    ----------
    lc1: :class:`stingray.Lightcurve` object, optional, default ``None``
        The first light curve data for the channel/band of interest.

    lc2: :class:`stingray.Lightcurve` object, optional, default ``None``
        The light curve data for the reference band.

    norm: {``frac``, ``abs``, ``leahy``, ``none``}, default ``none``
        The normalization of the (real part of the) cross spectrum.

    power_type: string, optional, default ``real`` Parameter to choose among
    complete, real part and magnitude of the cross spectrum.
    
    fullspec: boolean, optional, default ``False``
    Parameter to either only keep the positive frequencies, or all of them.

    Other Parameters
    ----------------
    gti: 2-d float array
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care!

    Attributes
    ----------
    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples

    power: numpy.ndarray
        The array of cross spectra (complex numbers)

    power_err: numpy.ndarray
        The uncertainties of ``power``.
        An approximation for each bin given by ``power_err= power/sqrt(m)``.
        Where ``m`` is the number of power averaged in each bin (by frequency
        binning, or averaging more than one spectra). Note that for a single
        realization (``m=1``) the error is equal to the power.

    df: float
        The frequency resolution

    m: int
        The number of averaged cross-spectra amplitudes in each bin.

    n: int
        The number of data points/time bins in one segment of the light
        curves.

    nphots1: float
        The total number of photons in light curve 1

    nphots2: float
        The total number of photons in light curve 2
    """
    def __init__(self, lc1=None, lc2=None, norm='none', gti=None,
                 power_type="real", fullspec=False):
        if isinstance(norm, str) is False:
            raise TypeError("norm must be a string")

        if norm.lower() not in ["frac", "abs", "leahy", "none"]:
            raise ValueError("norm must be 'frac', 'abs', 'leahy', or 'none'!")

        self.norm = norm.lower()

        # check if input data is a Lightcurve object, if not make one or
        # make an empty Crossspectrum object if lc1 == ``None`` or lc2 == ``None``
        if lc1 is None or lc2 is None:
            if lc1 is not None or lc2 is not None:
                raise TypeError("You can't do a cross spectrum with just one "
                                "light curve!")
            else:
                self.freq = None
                self.power = None
                self.power_err = None
                self.df = None
                self.nphots1 = None
                self.nphots2 = None
                self.m = 1
                self.n = None
                return
        self.gti = gti
        self.lc1 = lc1
        self.lc2 = lc2
        self.power_type = power_type
        self.fullspec = fullspec

        self._make_crossspectrum(lc1, lc2, fullspec)

        # These are needed to calculate coherence
        self._make_auxil_pds(lc1, lc2)

    def _make_auxil_pds(self, lc1, lc2):
        """
        Helper method to create the power spectrum of both light curves
        independently.

        Parameters
        ----------
        lc1, lc2 : :class:`stingray.Lightcurve` objects
            Two light curves used for computing the cross spectrum.
        """
        if lc1 is not lc2 and isinstance(lc1, Lightcurve):
            self.pds1 = Crossspectrum(lc1, lc1, norm='none')
            self.pds2 = Crossspectrum(lc2, lc2, norm='none')

    def _make_crossspectrum(self, lc1, lc2, fullspec=False):
        """
        Auxiliary method computing the normalized cross spectrum from two
        light curves. This includes checking for the presence of and
        applying Good Time Intervals, computing the unnormalized Fourier
        cross-amplitude, and then renormalizing using the required
        normalization. Also computes an uncertainty estimate on the cross
        spectral powers.

        Parameters
        ----------
        lc1, lc2 : :class:`stingray.Lightcurve` objects
            Two light curves used for computing the cross spectrum.

        fullspec: boolean, default ``False``
            Full frequency array (True) or just positive (False)

        """

        # make sure the inputs work!
        if not isinstance(lc1, Lightcurve):
            raise TypeError("lc1 must be a lightcurve.Lightcurve object")

        if not isinstance(lc2, Lightcurve):
            raise TypeError("lc2 must be a lightcurve.Lightcurve object")

        if self.lc2.mjdref != self.lc1.mjdref:
            raise ValueError("MJDref is different in the two light curves")

        # Then check that GTIs make sense
        if self.gti is None:
            self.gti = cross_two_gtis(lc1.gti, lc2.gti)

        check_gtis(self.gti)

        if self.gti.shape[0] != 1:
            raise TypeError("Non-averaged Cross Spectra need "
                            "a single Good Time Interval")

        lc1 = lc1.split_by_gti()[0]
        lc2 = lc2.split_by_gti()[0]

        # total number of photons is the sum of the
        # counts in the light curve
        self.nphots1 = np.float64(np.sum(lc1.counts))
        self.nphots2 = np.float64(np.sum(lc2.counts))

        self.meancounts1 = lc1.meancounts
        self.meancounts2 = lc2.meancounts

        # the number of data points in the light curve

        if lc1.n != lc2.n:
            raise StingrayError("Light curves do not have same number "
                                "of time bins per segment.")

        # If dt differs slightly, its propagated error must not be more than
        # 1/100th of the bin
        if not np.isclose(lc1.dt, lc2.dt, rtol=0.01 * lc1.dt / lc1.tseg):
            raise StingrayError("Light curves do not have same time binning "
                                "dt.")

        # In case a small difference exists, ignore it
        lc1.dt = lc2.dt

        self.n = lc1.n

        # the frequency resolution
        self.df = 1.0 / lc1.tseg

        # the number of averaged periodograms in the final output
        # This should *always* be 1 here
        self.m = 1

        # make the actual Fourier transform and compute cross spectrum
        self.freq, self.unnorm_power = self._fourier_cross(lc1, lc2, fullspec)

        # If co-spectrum is desired, normalize here. Otherwise, get raw back
        # with the imaginary part still intact.
        self.power = self._normalize_crossspectrum(self.unnorm_power, lc1.tseg)

        if lc1.err_dist.lower() != lc2.err_dist.lower():
            simon("Your lightcurves have different statistics."
                  "The errors in the Crossspectrum will be incorrect.")
        elif lc1.err_dist.lower() != "poisson":
            simon("Looks like your lightcurve statistic is not poisson."
                  "The errors in the Powerspectrum will be incorrect.")

        if self.__class__.__name__ in ['Powerspectrum',
                                       'AveragedPowerspectrum']:
            self.power_err = self.power / np.sqrt(self.m)
        elif self.__class__.__name__ in ['Crossspectrum',
                                         'AveragedCrossspectrum']:
            # This is clearly a wild approximation.
            simon("Errorbars on cross spectra are not thoroughly tested. "
                  "Please report any inconsistencies.")
            unnorm_power_err = np.sqrt(2) / np.sqrt(self.m)  # Leahy-like
            unnorm_power_err /= (2 / np.sqrt(self.nphots1 * self.nphots2))
            unnorm_power_err += np.zeros_like(self.power)

            self.power_err = \
                self._normalize_crossspectrum(unnorm_power_err, lc1.tseg)
        else:
            self.power_err = np.zeros(len(self.power))

    def _fourier_cross(self, lc1, lc2, fullspec=False):
        """
        Fourier transform the two light curves, then compute the cross spectrum.
        Computed as CS = lc1 x lc2* (where lc2 is the one that gets
        complex-conjugated). The user has the option to either get just the
        positive frequencies or the full spectrum.

        Parameters
        ----------
        lc1: :class:`stingray.Lightcurve` object
            One light curve to be Fourier transformed. Ths is the band of
            interest or channel of interest.

        lc2: :class:`stingray.Lightcurve` object
            Another light curve to be Fourier transformed.
            This is the reference band.

        fullspec: boolean. Default is False. Whole array of frequencies (True)
        or only positive (False).

        Returns
        -------
        fr: numpy.ndarray
            The squared absolute value of the Fourier amplitudes

        """
        fourier_1 = scipy.fftpack.fft(lc1.counts)  # do Fourier transform 1
        fourier_2 = scipy.fftpack.fft(lc2.counts)  # do Fourier transform 2

        freqs = scipy.fftpack.fftfreq(lc1.n, lc1.dt)

        if fullspec is  True:
            cross = np.multiply(fourier_1, np.conj(fourier_2))
            return freqs, cross

        else:
            cross = np.multiply(fourier_1[freqs > 0], np.conj(fourier_2[freqs > 0]))
            return freqs[freqs > 0], cross

    def rebin(self, df=None, f=None, method="mean"):
        """
        Rebin the cross spectrum to a new frequency resolution ``df``.

        Parameters
        ----------
        df: float
            The new frequency resolution

        Other Parameters
        ----------------
        f: float
            the rebin factor. If specified, it substitutes df with ``f*self.df``

        Returns
        -------
        bin_cs = :class:`Crossspectrum` (or one of its subclasses) object
            The newly binned cross spectrum or power spectrum.
            Note: this object will be of the same type as the object
            that called this method. For example, if this method is called
            from :class:`AveragedPowerspectrum`, it will return an object of class
            :class:`AveragedPowerspectrum`, too.
        """

        if f is None and df is None:
            raise ValueError('You need to specify at least one between f and '
                             'df')
        elif f is not None:
            df = f * self.df

        # rebin cross spectrum to new resolution
        binfreq, bincs, binerr, step_size = \
            rebin_data(self.freq, self.power, df, self.power_err,
                       method=method, dx=self.df)

        # make an empty cross spectrum object
        # note: syntax deliberate to work with subclass Powerspectrum
        bin_cs = copy.copy(self)

        # store the binned periodogram in the new object
        bin_cs.freq = binfreq
        bin_cs.power = bincs
        bin_cs.df = df
        bin_cs.n = self.n
        bin_cs.norm = self.norm
        bin_cs.nphots1 = self.nphots1
        bin_cs.power_err = binerr

        if hasattr(self, 'unnorm_power'):
            _, binpower_unnorm, _, _ = \
                rebin_data(self.freq, self.unnorm_power, df,
                           method=method, dx=self.df)

            bin_cs.unnorm_power = binpower_unnorm

        if hasattr(self, 'cs_all'):
            cs_all = []
            for c in self.cs_all:
                cs_all.append(c.rebin(df=df, f=f, method=method))
            bin_cs.cs_all = cs_all
        if hasattr(self, 'pds1'):
            bin_cs.pds1 = self.pds1.rebin(df=df, f=f, method=method)
        if hasattr(self, 'pds2'):
            bin_cs.pds2 = self.pds2.rebin(df=df, f=f, method=method)

        try:
            bin_cs.nphots2 = self.nphots2
        except AttributeError:
            if self.type == 'powerspectrum':
                pass
            else:
                raise AttributeError(
                    'Spectrum has no attribute named nphots2.')

        bin_cs.m = np.rint(step_size * self.m)

        return bin_cs

    def _normalize_crossspectrum(self, unnorm_power, tseg):
        """
        Normalize the real part of the cross spectrum to Leahy, absolute rms^2,
        fractional rms^2 normalization, or not at all.

        Parameters
        ----------
        unnorm_power: numpy.ndarray
            The unnormalized cross spectrum.

        tseg: int
            The length of the Fourier segment, in seconds.

        Returns
        -------
        power: numpy.nd.array
            The normalized co-spectrum (real part of the cross spectrum). For
            'none' normalization, imaginary part is returned as well.
        """

        # The "effective" counts/bin is the geometrical mean of the counts/bin
        # of the two light curves. Same goes for counts/second in meanrate.

        log_nphots1 = np.log(self.nphots1)
        log_nphots2 = np.log(self.nphots2)

        actual_nphots = np.float64(np.sqrt(np.exp(log_nphots1 + log_nphots2)))
        actual_mean = np.sqrt(self.meancounts1 * self.meancounts2)

        meanrate = np.sqrt((self.nphots1 * self.nphots2) / tseg)

        assert actual_mean > 0.0, \
            "Mean count rate is <= 0. Something went wrong."

        if self.power_type == "all":
            c_num = unnorm_power
        elif self.power_type == "real":
            c_num = unnorm_power.real
        elif self.power_type == "absolute":
            c_num = np.absolute(unnorm_power)
        else:
            raise ValueError("`power_type` not recognized!")

        if self.norm.lower() == 'leahy':
            power = c_num * 2. / actual_nphots

        elif self.norm.lower() == 'frac':
            c = c_num / np.float(self.n ** 2.)
            power = c * 2. * tseg / (actual_mean ** 2.0)

        elif self.norm.lower() == 'abs':
            power = c_num * 2. * meanrate / actual_nphots

        elif self.norm.lower() == 'none':
            power = unnorm_power
        return power

    def rebin_log(self, f=0.01):
        """
        Logarithmic rebin of the periodogram.
        The new frequency depends on the previous frequency
        modified by a factor f:

        .. math::

            d\\nu_j = d\\nu_{j-1} (1+f)

        Parameters
        ----------
        f: float, optional, default ``0.01``
            parameter that steers the frequency resolution


        Returns
        -------
        new_spec : :class:`Crossspectrum` (or one of its subclasses) object
            The newly binned cross spectrum or power spectrum.
            Note: this object will be of the same type as the object
            that called this method. For example, if this method is called
            from :class:`AveragedPowerspectrum`, it will return an object of class
        """

        binfreq, binpower, binpower_err, nsamples = \
            rebin_data_log(self.freq, self.power, f,
                           y_err=self.power_err, dx=self.df)

        # the frequency resolution
        df = np.diff(binfreq)

        # shift the lower bin edges to the middle of the bin and drop the
        # last right bin edge
        binfreq = binfreq[:-1] + df / 2

        new_spec = copy.copy(self)
        new_spec.freq = binfreq
        new_spec.power = binpower
        new_spec.power_err = binpower_err
        new_spec.m = nsamples * self.m

        if hasattr(self, 'unnorm_power'):
            _, binpower_unnorm, _, _ = \
                rebin_data_log(self.freq, self.unnorm_power, f, dx=self.df)

            new_spec.unnorm_power = binpower_unnorm

        if hasattr(self, 'pds1'):
            new_spec.pds1 = self.pds1.rebin_log(f)
        if hasattr(self, 'pds2'):
            new_spec.pds2 = self.pds2.rebin_log(f)

        if hasattr(self, 'cs_all'):
            cs_all = []
            for c in self.cs_all:
                cs_all.append(c.rebin_log(f))
            new_spec.cs_all = cs_all

        return new_spec

    def coherence(self):
        """ Compute Coherence function of the cross spectrum.

        Coherence is defined in Vaughan and Nowak, 1996 [vaughan-1996].
        It is a Fourier frequency dependent measure of the linear correlation
        between time series measured simultaneously in two energy channels.

        Returns
        -------
        coh : numpy.ndarray
            Coherence function

        References
        ----------
        .. [vaughan-1996] http://iopscience.iop.org/article/10.1086/310430/pdf

        """
        # this computes the averaged power spectrum, but using the
        # cross spectrum code to avoid circular imports

        return self.unnorm_power.real / (self.pds1.power.real *
                                         self.pds2.power.real)

    def _phase_lag(self):
        """Return the fourier phase lag of the cross spectrum."""
        return np.angle(self.unnorm_power)

    def time_lag(self):
        """
        Calculate the fourier time lag of the cross spectrum. The time lag is
        calculate using the center of the frequency bins.
        """
        if self.__class__ in [Crossspectrum, AveragedCrossspectrum]:
            ph_lag = self._phase_lag()

            return ph_lag / (2 * np.pi * self.freq)
        else:
            raise AttributeError("Object has no attribute named 'time_lag' !")

    def plot(self, labels=None, axis=None, title=None, marker='-', save=False,
             filename=None):
        """
        Plot the amplitude of the cross spectrum vs. the frequency using ``matplotlib``.

        Parameters
        ----------
        labels : iterable, default ``None``
            A list of tuple with ``xlabel`` and ``ylabel`` as strings.

        axis : list, tuple, string, default ``None``
            Parameter to set axis properties of the ``matplotlib`` figure. For example
            it can be a list like ``[xmin, xmax, ymin, ymax]`` or any other
            acceptable argument for the``matplotlib.pyplot.axis()`` method.

        title : str, default ``None``
            The title of the plot.

        marker : str, default '-'
            Line style and color of the plot. Line styles and colors are
            combined in a single format string, as in ``'bo'`` for blue
            circles. See ``matplotlib.pyplot.plot`` for more options.

        save : boolean, optional, default ``False``
            If ``True``, save the figure with specified filename.

        filename : str
            File name of the image to save. Depends on the boolean ``save``.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib required for plot()")

        fig = plt.figure('crossspectrum')
        fig = plt.plot(self.freq, np.abs(self.power), marker, color='b',
                       label='Amplitude')
        fig = plt.plot(self.freq, np.abs(self.power.real), marker, color='r',
                       alpha=0.5, label='Real Part')
        fig = plt.plot(self.freq, np.abs(self.power.imag), marker, color='g',
                       alpha=0.5, label='Imaginary Part')

        if labels is not None:
            try:
                plt.xlabel(labels[0])
                plt.ylabel(labels[1])
            except TypeError:
                utils.simon("``labels`` must be either a list or tuple with "
                            "x and y labels.")
                raise
            except IndexError:
                utils.simon("``labels`` must have two labels for x and y "
                            "axes.")
                # Not raising here because in case of len(labels)==1, only
                # x-axis will be labelled.
        plt.legend(loc='best')
        if axis is not None:
            plt.axis(axis)

        if title is not None:
            plt.title(title)

        if save:
            if filename is None:
                plt.savefig('spec.png')
            else:
                plt.savefig(filename)
        else:
            plt.show(block=False)


class AveragedCrossspectrum(Crossspectrum):
    """
    Make an averaged cross spectrum from a light curve by segmenting two
    light curves, Fourier-transforming each segment and then averaging the
    resulting cross spectra.

    Parameters
    ----------
    lc1: :class:`stingray.Lightcurve` object OR iterable of :class:`stingray.Lightcurve` objects
        A light curve from which to compute the cross spectrum. In some cases, this would
        be the light curve of the wavelength/energy/frequency band of interest.

    lc2: :class:`stingray.Lightcurve` object OR iterable of :class:`stingray.Lightcurve` objects
        A second light curve to use in the cross spectrum. In some cases, this would be
        the wavelength/energy/frequency reference band to compare the band of interest with.

    segment_size: float
        The size of each segment to average. Note that if the total
        duration of each :class:`Lightcurve` object in ``lc1`` or ``lc2`` is not an
        integer multiple of the ``segment_size``, then any fraction left-over
        at the end of the time series will be lost. Otherwise you introduce
        artifacts.

    norm: {``frac``, ``abs``, ``leahy``, ``none``}, default ``none``
        The normalization of the (real part of the) cross spectrum.

    Other Parameters
    ----------------
    gti: 2-d float array
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care!

    power_type: string, optional, default ``real`` Parameter to choose among
    complete, real part and magnitude of the cross spectrum.

    fullspec: boolean, optional, default ``False`` Parameter to either
    return the full array of frequencies (True) or just the positive (False)

    Attributes
    ----------
    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples

    power: numpy.ndarray
        The array of cross spectra

    power_err: numpy.ndarray
        The uncertainties of ``power``.
        An approximation for each bin given by ``power_err= power/sqrt(m)``.
        Where ``m`` is the number of power averaged in each bin (by frequency
        binning, or averaging powerspectrum). Note that for a single
        realization (``m=1``) the error is equal to the power.

    df: float
        The frequency resolution

    m: int
        The number of averaged cross spectra

    n: int
        The number of time bins per segment of light curve

    nphots1: float
        The total number of photons in the first (interest) light curve

    nphots2: float
        The total number of photons in the second (reference) light curve

    gti: 2-d float array
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        They are calculated by taking the common GTI between the
        two light curves

    """

    def __init__(self, lc1=None, lc2=None, segment_size=None,
                 norm='none', gti=None, power_type="real",
                 fullspec=False):

        self.type = "crossspectrum"

        if segment_size is None and lc1 is not None:
            raise ValueError("segment_size must be specified")
        if segment_size is not None and not np.isfinite(segment_size):
            raise ValueError("segment_size must be finite!")


        self.segment_size = segment_size
        self.power_type = power_type
        self.fullspec = fullspec

        Crossspectrum.__init__(self, lc1, lc2, norm, gti=gti,
                               power_type=power_type, fullspec=fullspec)
        return

    def _make_auxil_pds(self, lc1, lc2):
        """
        Helper method to create the power spectrum of both light curves independently.

        Parameters
        ----------
        lc1, lc2 : :class:`stingray.Lightcurve` objects
            Two light curves used for computing the cross spectrum.
        """
        # A way to say that this is actually not a power spectrum
        if lc1 is not lc2 and isinstance(lc1, Lightcurve):
            self.pds1 = AveragedCrossspectrum(lc1, lc1,
                                              segment_size=self.segment_size,
                                              norm='none', gti=lc1.gti, power_type=self.power_type,
                                              fullspec=self.fullspec)
            self.pds2 = AveragedCrossspectrum(lc2, lc2,
                                              segment_size=self.segment_size,
                                              norm='none', gti=lc2.gti, power_type=self.power_type,
                                              fullspec=self.fullspec)

    def _make_segment_spectrum(self, lc1, lc2, segment_size):
        """
        Split the light curves into segments of size ``segment_size``, and calculate a cross spectrum for
        each.

        Parameters
        ----------
        lc1, lc2 : :class:`stingray.Lightcurve` objects
            Two light curves used for computing the cross spectrum.

        segment_size : ``numpy.float``
            Size of each light curve segment to use for averaging.

        Returns
        -------
        cs_all : list of :class:`Crossspectrum`` objects
            A list of cross spectra calculated independently from each light curve segment

        nphots1_all, nphots2_all : ``numpy.ndarray` for each of ``lc1`` and ``lc2``
            Two lists containing the number of photons for all segments calculated from ``lc1`` and ``lc2``.

        """

        # TODO: need to update this for making cross spectra.
        assert isinstance(lc1, Lightcurve)
        assert isinstance(lc2, Lightcurve)

        if lc1.tseg != lc2.tseg:
            simon("Lightcurves do not have same tseg. This means that the data"
                  "from the two channels are not completely in sync. This "
                  "might or might not be an issue. Keep an eye on it.")

        # If dt differs slightly, its propagated error must not be more than
        # 1/100th of the bin
        if not np.isclose(lc1.dt, lc2.dt, rtol=0.01 * lc1.dt / lc1.tseg):
            raise ValueError("Light curves do not have same time binning dt.")

        # In case a small difference exists, ignore it
        lc1.dt = lc2.dt
        if self.gti is None:
            self.gti = cross_two_gtis(lc1.gti, lc2.gti)
            lc1.gti = lc2.gti = self.gti
            lc1._apply_gtis()
            lc2._apply_gtis()

        check_gtis(self.gti)

        cs_all = []
        nphots1_all = []
        nphots2_all = []

        start_inds, end_inds = \
            bin_intervals_from_gtis(self.gti, segment_size, lc1.time,
                                    dt=lc1.dt)

        for start_ind, end_ind in zip(start_inds, end_inds):
            time_1 = lc1.time[start_ind:end_ind]
            counts_1 = lc1.counts[start_ind:end_ind]
            counts_1_err = lc1.counts_err[start_ind:end_ind]
            time_2 = lc2.time[start_ind:end_ind]
            counts_2 = lc2.counts[start_ind:end_ind]
            counts_2_err = lc2.counts_err[start_ind:end_ind]
            gti1 = np.array([[time_1[0] - lc1.dt / 2,
                              time_1[-1] + lc1.dt / 2]])
            gti2 = np.array([[time_2[0] - lc2.dt / 2,
                              time_2[-1] + lc2.dt / 2]])
            lc1_seg = Lightcurve(time_1, counts_1, err=counts_1_err,
                                 err_dist=lc1.err_dist,
                                 gti=gti1,
                                 dt=lc1.dt)
            lc2_seg = Lightcurve(time_2, counts_2, err=counts_2_err,
                                 err_dist=lc2.err_dist,
                                 gti=gti2,
                                 dt=lc2.dt)
            cs_seg = Crossspectrum(lc1_seg, lc2_seg, norm=self.norm, power_type=self.power_type,
                                 fullspec=self.fullspec)
            cs_all.append(cs_seg)
            nphots1_all.append(np.sum(lc1_seg.counts))
            nphots2_all.append(np.sum(lc2_seg.counts))

        return cs_all, nphots1_all, nphots2_all

    def _make_crossspectrum(self, lc1, lc2, fullspec=False):
        """
        Auxiliary method computing the normalized cross spectrum from two light curves.
        This includes checking for the presence of and applying Good Time Intervals, computing the
        unnormalized Fourier cross-amplitude, and then renormalizing using the required normalization.
        Also computes an uncertainty estimate on the cross spectral powers.

        Parameters
        ----------
        lc1, lc2 : :class:`stingray.Lightcurve` objects
            Two light curves used for computing the cross spectrum.

        fullspec: boolean, default ``False``, all frequencies returned (True),
        or just positive (False)

        """

        # chop light curves into segments
        if isinstance(lc1, Lightcurve) and \
                isinstance(lc2, Lightcurve):

            if self.type == "crossspectrum":
                self.cs_all, nphots1_all, nphots2_all = \
                    self._make_segment_spectrum(lc1, lc2, self.segment_size)

            elif self.type == "powerspectrum":
                self.cs_all, nphots1_all = \
                    self._make_segment_spectrum(lc1, self.segment_size)

            else:
                raise ValueError("Type of spectrum not recognized!")

        else:
            self.cs_all, nphots1_all, nphots2_all = [], [], []

            for lc1_seg, lc2_seg in zip(lc1, lc2):

                if self.type == "crossspectrum":
                    cs_sep, nphots1_sep, nphots2_sep = \
                        self._make_segment_spectrum(lc1_seg, lc2_seg,
                                                    self.segment_size)
                    nphots2_all.append(nphots2_sep)
                elif self.type == "powerspectrum":
                    cs_sep, nphots1_sep = \
                        self._make_segment_spectrum(lc1_seg, self.segment_size)

                else:
                    raise ValueError("Type of spectrum not recognized!")

                self.cs_all.append(cs_sep)
                nphots1_all.append(nphots1_sep)

            self.cs_all = np.hstack(self.cs_all)
            nphots1_all = np.hstack(nphots1_all)

            if self.type == "crossspectrum":
                nphots2_all = np.hstack(nphots2_all)

        m = len(self.cs_all)
        nphots1 = np.mean(nphots1_all)

        power_avg = np.zeros_like(self.cs_all[0].power)
        power_err_avg = np.zeros_like(self.cs_all[0].power_err)
        unnorm_power_avg = np.zeros_like(self.cs_all[0].unnorm_power)
        for cs in self.cs_all:
            power_avg += cs.power
            unnorm_power_avg += cs.unnorm_power
            power_err_avg += (cs.power_err) ** 2

        power_avg /= np.float(m)
        power_err_avg = np.sqrt(power_err_avg) / m
        unnorm_power_avg /= np.float(m)

        self.freq = self.cs_all[0].freq
        self.power = power_avg
        self.unnorm_power = unnorm_power_avg
        self.m = m
        self.power_err = power_err_avg
        self.df = self.cs_all[0].df
        self.n = self.cs_all[0].n
        self.nphots1 = nphots1

        if self.type == "crossspectrum":
            self.nphots1 = nphots1
            nphots2 = np.mean(nphots2_all)

            self.nphots2 = nphots2

    def coherence(self):
        """Averaged Coherence function.

        Coherence is defined in Vaughan and Nowak, 1996 [vaughan-1996].
        It is a Fourier frequency dependent measure of the linear correlation
        between time series measured simultaneously in two energy channels.

        Compute an averaged Coherence function of cross spectrum by computing
        coherence function of each segment and averaging them. The return type
        is a tuple with first element as the coherence function and the second
        element as the corresponding uncertainty associated with it.

        Note : The uncertainty in coherence function is strictly valid for Gaussian \
               statistics only.

        Returns
        -------
        (coh, uncertainty) : tuple of np.ndarray
            Tuple comprising the coherence function and uncertainty.

        References
        ----------
        .. [vaughan-1996] http://iopscience.iop.org/article/10.1086/310430/pdf

        """
        if np.any(self.m < 50):
            simon("Number of segments used in averaging is "
                  "significantly low. The result might not follow the "
                  "expected statistical distributions.")

        # Calculate average coherence
        unnorm_power_avg = self.unnorm_power

        num = np.absolute(unnorm_power_avg) ** 2

        # The normalization was 'none'!
        unnorm_powers_avg_1 = self.pds1.power.real
        unnorm_powers_avg_2 = self.pds2.power.real

        coh = num / (unnorm_powers_avg_1 * unnorm_powers_avg_2)

        # Calculate uncertainty
        uncertainty = \
            (2 ** 0.5 * coh * (1 - coh)) / (np.abs(coh) * self.m ** 0.5)

        return (coh, uncertainty)

    def time_lag(self):
        """Calculate time lag and uncertainty.

        Equation from Bendat & Piersol, 2011 [bendat-2011]__.

        Returns
        -------
        lag : np.ndarray
            The time lag

        lag_err : np.ndarray
            The uncertainty in the time lag
        """
        lag = super(AveragedCrossspectrum, self).time_lag()
        coh, uncert = self.coherence()
        dum = (1. - coh) / (2. * coh)
        lag_err = np.sqrt(dum / self.m) / (2 * np.pi * self.freq)

        return lag, lag_err
