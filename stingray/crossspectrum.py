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

__all__ = ["Crossspectrum", "AveragedCrossspectrum", "coherence"]


def coherence(lc1, lc2):
    """
    Estimate coherence function of two light curves.

    Parameters
    ----------
    lc1: lightcurve.Lightcurve object
        The first light curve data for the channel of interest.

    lc2: lightcurve.Lightcurve object
        The light curve data for reference band

    Returns
    -------
    coh : np.ndarray
        Coherence function
    """

    if not isinstance(lc1, Lightcurve):
        raise TypeError("lc1 must be a lightcurve.Lightcurve object")

    if not isinstance(lc2, Lightcurve):
        raise TypeError("lc2 must be a lightcurve.Lightcurve object")

    cs = Crossspectrum(lc1, lc2, norm='none')

    return cs.coherence()


class Crossspectrum(object):
    def __init__(self, lc1=None, lc2=None, norm='none', gti=None):
        """
        Make a cross spectrum from a (binned) light curve.
        You can also make an empty Crossspectrum object to populate with your
        own fourier-transformed data (this can sometimes be useful when making
        binned periodograms).

        Parameters
        ----------
        lc1: lightcurve.Lightcurve object, optional, default None
            The first light curve data for the channel/band of interest.

        lc2: lightcurve.Lightcurve object, optional, default None
            The light curve data for the reference band.

        norm: {'frac', 'abs', 'leahy', 'none'}, default 'none'
            The normalization of the (real part of the) cross spectrum.

        Other Parameters
        ----------------
        gti: 2-d float array
            [[gti0_0, gti0_1], [gti1_0, gti1_1], ...] -- Good Time intervals.
            This choice overrides the GTIs in the single light curves. Use with
            care!

        Attributes
        ----------
        freq: numpy.ndarray
            The array of mid-bin frequencies that the Fourier transform samples

        power: numpy.ndarray
            The array of cross spectra (complex numbers)

        power_err: numpy.ndarray
            The uncertainties of `power`.
            An approximation for each bin given by "power_err= power/Sqrt(m)".
            Where `m` is the number of power averaged in each bin (by frequency
            binning, or averaging more than one spectra). Note that for a single
            realization (m=1) the error is equal to the power.

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

        if isinstance(norm, str) is False:
            raise TypeError("norm must be a string")

        if norm.lower() not in ["frac", "abs", "leahy", "none"]:
            raise ValueError("norm must be 'frac', 'abs', 'leahy', or 'none'!")

        self.norm = norm.lower()

        # check if input data is a Lightcurve object, if not make one or
        # make an empty Crossspectrum object if lc1 == None or lc2 == None
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

        self._make_crossspectrum(lc1, lc2)
        # These are needed to calculate coherence
        self._make_auxil_pds(lc1, lc2)

    def _make_auxil_pds(self, lc1, lc2):
        if lc1 is not lc2 and isinstance(lc1, Lightcurve):
            self.pds1 = Crossspectrum(lc1, lc1, norm='none')
            self.pds2 = Crossspectrum(lc2, lc2, norm='none')

    def _make_crossspectrum(self, lc1, lc2):

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
        self.freq, self.unnorm_power = self._fourier_cross(lc1, lc2)

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

    def _fourier_cross(self, lc1, lc2):
        """
        Fourier transform the two light curves, then compute the cross spectrum.
        Computed as CS = lc1 x lc2* (where lc2 is the one that gets
        complex-conjugated)

        Parameters
        ----------
        lc1: lightcurve.Lightcurve object
            One light curve to be Fourier transformed. Ths is the band of
            interest or channel of interest.

        lc2: lightcurve.Lightcurve object
            Another light curve to be Fourier transformed.
            This is the reference band.

        Returns
        -------
        fr: numpy.ndarray
            The squared absolute value of the Fourier amplitudes

        """
        fourier_1 = scipy.fftpack.fft(lc1.counts)  # do Fourier transform 1
        fourier_2 = scipy.fftpack.fft(lc2.counts)  # do Fourier transform 2

        freqs = scipy.fftpack.fftfreq(lc1.n, lc1.dt)
        cross = fourier_1[freqs > 0] * np.conj(fourier_2[freqs > 0])

        return freqs[freqs > 0], cross

    def rebin(self, df=None, f=None, method="mean"):
        """
        Rebin the cross spectrum to a new frequency resolution df.

        Parameters
        ----------
        df: float
            The new frequency resolution

        Other Parameters
        ----------------
        f: float
            the rebin factor. If specified, it substitutes df with f*self.df

        Returns
        -------
        bin_cs = Crossspectrum (or one of its subclasses) object
            The newly binned cross spectrum or power spectrum.
            Note: this object will be of the same type as the object
            that called this method. For example, if this method is called
            from `AveragedPowerspectrum`, it will return an object of class
            `AveragedPowerspectrum`, too.
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
                raise AttributeError('Spectrum has no attribute named nphots2.')

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
        # of the two light curves

        log_nphots1 = np.log(self.nphots1)
        log_nphots2 = np.log(self.nphots2)

        actual_nphots = np.float64(np.sqrt(np.exp(log_nphots1 + log_nphots2)))
        actual_mean = np.sqrt(self.meancounts1 * self.meancounts2)

        assert actual_mean > 0.0, \
            "Mean count rate is <= 0. Something went wrong."

        if self.norm.lower() == 'leahy':
            c = unnorm_power.real
            power = c * 2. / actual_nphots

        elif self.norm.lower() == 'frac':
            c = unnorm_power.real / np.float(self.n ** 2.)
            power = c * 2. * tseg / (actual_mean ** 2.0)

        elif self.norm.lower() == 'abs':
            c = unnorm_power.real / np.float(self.n ** 2.)
            power = c * (2. * tseg)

        elif self.norm.lower() == 'none':
            power = unnorm_power

        else:
            raise Exception("Normalization not recognized!")

        return power

    def rebin_log(self, f=0.01):
        """
        Logarithmic rebin of the periodogram.
        The new frequency depends on the previous frequency
        modified by a factor f:

        dnu_j = dnu_{j-1}*(1+f)

        Parameters
        ----------
        f: float, optional, default 0.01
            parameter that steers the frequency resolution


        Returns
        -------
        new_spec : Crossspectrum (or one of its subclasses) object
            The newly binned cross spectrum or power spectrum.
            Note: this object will be of the same type as the object
            that called this method. For example, if this method is called
            from `AveragedPowerspectrum`, it will return an object of class
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
        """
        Compute Coherence function of the cross spectrum. Coherence is a
        Fourier frequency dependent measure of the linear correlation
        between time series measured simultaneously in two energy channels.

        Returns
        -------
        coh : numpy.ndarray
            Coherence function

        References
        ----------
        .. [1] http://iopscience.iop.org/article/10.1086/310430/pdf

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


class AveragedCrossspectrum(Crossspectrum):
    def __init__(self, lc1=None, lc2=None, segment_size=None,
                 norm='none', gti=None):
        """
        Make an averaged cross spectrum from a light curve by segmenting two
        light curves, Fourier-transforming each segment and then averaging the
        resulting cross spectra.

        Parameters
        ----------
        lc1: lightcurve.Lightcurve object OR
            iterable of lightcurve.Lightcurve objects
            One light curve data to be Fourier-transformed. This is the band
            of interest or channel of interest.

        lc2: lightcurve.Lightcurve object OR
            iterable of lightcurve.Lightcurve objects
            Second light curve data to be Fourier-transformed. This is the
            reference band.

        segment_size: float
            The size of each segment to average. Note that if the total
            duration of each Lightcurve object in lc1 or lc2 is not an
            integer multiple of the segment_size, then any fraction left-over
            at the end of the time series will be lost. Otherwise you introduce
            artefacts.

        norm: {'frac', 'abs', 'leahy', 'none'}, default 'none'
            The normalization of the (real part of the) cross spectrum.

        Other Parameters
        ----------------
        gti: 2-d float array
            [[gti0_0, gti0_1], [gti1_0, gti1_1], ...] -- Good Time intervals.
            This choice overrides the GTIs in the single light curves. Use with
            care!

        Attributes
        ----------
        freq: numpy.ndarray
            The array of mid-bin frequencies that the Fourier transform samples

        power: numpy.ndarray
            The array of cross spectra

        power_err: numpy.ndarray
            The uncertainties of `power`.
            An approximation for each bin given by "power_err= power/Sqrt(m)".
            Where `m` is the number of power averaged in each bin (by frequency
            binning, or averaging powerspectrum). Note that for a single
            realization (m=1) the error is equal to the power.

        df: float
            The frequency resolution

        m: int
            The number of averaged cross spectra

        n: int
            The number of time bins per segment of light curve?

        nphots1: float
            The total number of photons in the first (interest) light curve

        nphots2: float
            The total number of photons in the second (reference) light curve

        gti: 2-d float array
            [[gti0_0, gti0_1], [gti1_0, gti1_1], ...] -- Good Time intervals.
            They are calculated by taking the common GTI between the
            two light curves

        """
        self.type = "crossspectrum"

        if segment_size is None and lc1 is not None:
            raise ValueError("segment_size must be specified")
        if segment_size is not None and not np.isfinite(segment_size):
            raise ValueError("segment_size must be finite!")

        self.segment_size = segment_size

        Crossspectrum.__init__(self, lc1, lc2, norm, gti=gti)

        return

    def _make_auxil_pds(self, lc1, lc2):
        # A way to say that this is actually not a power spectrum
        if lc1 is not lc2 and isinstance(lc1, Lightcurve):
            self.pds1 = AveragedCrossspectrum(lc1, lc1,
                                              segment_size=self.segment_size,
                                              norm='none', gti=lc1.gti)
            self.pds2 = AveragedCrossspectrum(lc2, lc2,
                                              segment_size=self.segment_size,
                                              norm='none', gti=lc2.gti)

    def _make_segment_spectrum(self, lc1, lc2, segment_size):

        # TODO: need to update this for making cross spectra.
        assert isinstance(lc1, Lightcurve)
        assert isinstance(lc2, Lightcurve)

        if lc1.tseg != lc2.tseg:
            raise ValueError("Lightcurves do not have same tseg.")

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
            cs_seg = Crossspectrum(lc1_seg, lc2_seg, norm=self.norm)
            cs_all.append(cs_seg)
            nphots1_all.append(np.sum(lc1_seg.counts))
            nphots2_all.append(np.sum(lc2_seg.counts))
        return cs_all, nphots1_all, nphots2_all

    def _make_crossspectrum(self, lc1, lc2):

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
            # TODO: should be using izip from iterables if lc1 or lc2 could
            # be long
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
        """
        Compute an averaged Coherence function of cross spectrum by computing
        coherence function of each segment and averaging them. The return type
        is a tuple with first element as the coherence function and the second
        element as the corresponding uncertainty[1] associated with it.

        Note : The uncertainty in coherence function is strictly valid for
               Gaussian statistics only.

        Returns
        -------
        tuple : tuple of np.ndarray
            Tuple of coherence function and uncertainty.

        References
        ----------
        .. [1] http://iopscience.iop.org/article/10.1086/310430/pdf

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
        uncertainty = (2 ** 0.5 * coh * (1 - coh)) / (
        np.abs(coh) * self.m ** 0.5)

        return (coh, uncertainty)

    def time_lag(self):
        """Calculate time lag and uncertainty.

        Formula from Bendat & Piersol 1986

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
