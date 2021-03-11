import copy
import warnings

import numpy as np
import scipy
import scipy.optimize
import scipy.stats

import stingray.utils as utils
from stingray.crossspectrum import AveragedCrossspectrum, Crossspectrum
from stingray.gti import bin_intervals_from_gtis, check_gtis
from stingray.largememory import createChunkedSpectra, saveData
from stingray.stats import pds_probability
from stingray.utils import genDataPath
from stingray.utils import genDataPath, rebin_data, rebin_data_log, simon

from .events import EventList
from .gti import cross_two_gtis
from .lightcurve import Lightcurve

from nitime.lazy import matplotlib_mlab as mlab
from nitime.lazy import scipy_linalg as linalg
from nitime.lazy import scipy_signal as sig
from nitime.lazy import scipy_interpolate as interpolate
from nitime.lazy import scipy_fftpack as fftpack

import nitime.utils as utils
from nitime.utils import tapered_spectra, dpss_windows
import nitime.algorithms as tsa
import nitime.utils as utils
from nitime.viz import winspect
from nitime.viz import plot_spectral_estimate

# To support older versions of numpy that don't have tril_indices:
from nitime.index_utils import tril_indices, triu_indices

try:
    from tqdm import tqdm as show_progress
except ImportError:
    def show_progress(a, **kwargs):
        return a

__all__ = ["Multitaper", "plot_dB"]


def plot_dB(freq, data, labels=None, axis=None, title=None, marker='-', save=False,
            filename=None):
    """
        Plot the data in decibel (dB) vs. the frequency using ``matplotlib``.

        Parameters
        ----------
        freq : numpy.ndarray
            Frequency bin values, x - axis

        data : numpy.ndarray
            Data to be ploted in dB, generally power spectrum, y - axis

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

    data_dB = 10 * np.log10(data)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib required for plot()")

    plt.figure('multitaper')
    plt.plot(freq,
             np.abs(data_dB),
             marker,
             color='c',
             label='Amplitude - dB')

    if labels is not None:
        try:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
        except TypeError:
            simon("``labels`` must be either a list or tuple with "
                  "x and y labels.")
            raise
        except IndexError:
            simon("``labels`` must have two labels for x and y "
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
            plt.savefig('multitaper.png')
        else:
            plt.savefig(filename)
    else:
        plt.show(block=False)


class Multitaper(Crossspectrum):
    """
    Class to calculate the multitaper periodogram from a lightcurve data.
    Parameters
    ----------
    data: :class:`stingray.Lightcurve` object, optional, default ``None``
        The light curve data to be Fourier-transformed.

    norm: {``leahy`` | ``frac`` | ``abs`` | ``none`` }, optional, default ``frac``
        The normaliation of the power spectrum to be used. Options are
        ``leahy``, ``frac``, ``abs`` and ``none``, default is ``frac``.

    Other Parameters
    ----------------
    gti: 2-d float array
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care!

    Attributes
    ----------
    norm: {``leahy`` | ``frac`` | ``abs`` | ``none`` }
        the normalization of the power spectrun

    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples

    power: numpy.ndarray
        The array of normalized squared absolute values of Fourier
        amplitudes

    power_err: numpy.ndarray
        The uncertainties of ``power``.
        An approximation for each bin given by ``power_err= power/sqrt(m)``.
        Where ``m`` is the number of power averaged in each bin (by frequency
        binning, or averaging power spectrum). Note that for a single
        realization (``m=1``) the error is equal to the power.

    df: float
        The frequency resolution

    m: int
        The number of averaged powers in each bin

    n: int
        The number of data points in the light curve

    nphots: float
        The total number of photons in the light curve

    """

    def __init__(self, data=None, norm="frac", gti=None,
                 dt=None, lc=None, NW=None, bandwidth=None, adaptive=False,
                 jackknife=True, low_bias=True, Fs=1):
        if lc is not None:
            warnings.warn("The lc keyword is now deprecated. Use data "
                          "instead", DeprecationWarning)
        if data is None:
            data = lc

        Crossspectrum.__init__(self, data1=data, data2=data, norm=norm, gti=gti,
                               dt=dt)
        self.nphots = self.nphots1
        self.dt = dt

        self._multitaper_periodogram(data, NW=NW, adaptive=adaptive,
                                     bandwidth=bandwidth, jackknife=jackknife, low_bias=low_bias, Fs=Fs)
        self.power = self._normalize_crossspectrum(self.unnorm_power, data.tseg)

    def _multitaper_periodogram(self, lc, NW=None, bandwidth=None, adaptive=False,
                                jackknife=True, low_bias=True, Fs=1):
        if not isinstance(lc, Lightcurve):
            raise TypeError("lc must be a lightcurve.Lightcurve object")

        nitime_freq, unnorm_mtp, self.jk_var_deg_freedom = tsa.multi_taper_psd(
            lc.counts, NW=NW, adaptive=adaptive, BW=bandwidth, jackknife=jackknife,
            low_bias=low_bias, sides="onesided", Fs=Fs)

        self.unnorm_power = unnorm_mtp/2

        len_correct = abs(len(unnorm_mtp)-len(self.freq))
        self.unnorm_power = self.unnorm_power[len_correct:]  # Making length same as self.freq

    def rebin(self, df=None, f=None, method="mean"):
        """
        Rebin the power spectrum.

        Parameters
        ----------
        df: float
            The new frequency resolution

        Other Parameters
        ----------------
        f: float
            the rebin factor. If specified, it substitutes ``df`` with ``f*self.df``

        Returns
        -------
        bin_cs = :class:`Powerspectrum` object
            The newly binned power spectrum.
        """
        bin_ps = Crossspectrum.rebin(self, df=df, f=f, method=method)
        bin_ps.nphots = bin_ps.nphots1

        return bin_ps

    def compute_rms(self, min_freq, max_freq, white_noise_offset=0.):
        """
        Compute the fractional rms amplitude in the power spectrum
        between two frequencies.

        Parameters
        ----------
        min_freq: float
            The lower frequency bound for the calculation

        max_freq: float
            The upper frequency bound for the calculation

        Other parameters
        ----------------
        white_noise_offset : float, default 0
            This is the white noise level, in Leahy normalization. In the ideal
            case, this is 2. Dead time and other instrumental effects can alter
            it. The user can fit the white noise level outside this function
            and it will get subtracted from powers here.

        Returns
        -------
        rms: float
            The fractional rms amplitude contained between ``min_freq`` and
            ``max_freq``

        rms_err: float
            The error on the fractional rms amplitude

        """
        minind = self.freq.searchsorted(min_freq)
        maxind = self.freq.searchsorted(max_freq)
        powers = self.power[minind:maxind]
        nphots = self.nphots

        if self.norm.lower() == 'leahy':
            powers_leahy = powers.copy()
        elif self.norm.lower() == "frac":
            powers_leahy = \
                self.unnorm_power[minind:maxind].real * 2 / nphots
        else:
            raise TypeError("Normalization not recognized!")

        rms = np.sqrt(np.sum(powers_leahy - white_noise_offset) / nphots)
        rms_err = self._rms_error(powers_leahy)

        return rms, rms_err

    def _rms_error(self, powers):
        """
        Compute the error on the fractional rms amplitude using error
        propagation.
        Note: this uses the actual measured powers, which is not
        strictly correct. We should be using the underlying power spectrum,
        but in the absence of an estimate of that, this will have to do.

        .. math::

           r = \sqrt{P}

        .. math::

           \delta r = \\frac{1}{2 * \sqrt{P}} \delta P

        Parameters
        ----------
        powers: iterable
            The list of powers used to compute the fractional rms amplitude.

        Returns
        -------
        delta_rms: float
            The error on the fractional rms amplitude
        """
        nphots = self.nphots
        p_err = scipy.stats.chi2(2.0 * self.m).var() * powers / self.m / nphots

        rms = np.sum(powers) / nphots
        pow = np.sqrt(rms)

        drms_dp = 1 / (2 * pow)

        sq_sum_err = np.sqrt(np.sum(p_err**2))
        delta_rms = sq_sum_err * drms_dp
        return delta_rms

    def classical_significances(self, threshold=1, trial_correction=False):
        """
        Compute the classical significances for the powers in the power
        spectrum, assuming an underlying noise distribution that follows a
        chi-square distributions with 2M degrees of freedom, where M is the
        number of powers averaged in each bin.

        Note that this function will *only* produce correct results when the
        following underlying assumptions are fulfilled:

        1. The power spectrum is Leahy-normalized
        2. There is no source of variability in the data other than the
           periodic signal to be determined with this method. This is important!
           If there are other sources of (aperiodic) variability in the data, this
           method will *not* produce correct results, but instead produce a large
           number of spurious false positive detections!
        3. There are no significant instrumental effects changing the
           statistical distribution of the powers (e.g. pile-up or dead time)

        By default, the method produces ``(index,p-values)`` for all powers in
        the power spectrum, where index is the numerical index of the power in
        question. If a ``threshold`` is set, then only powers with p-values
        *below* that threshold with their respective indices. If
        ``trial_correction`` is set to ``True``, then the threshold will be corrected
        for the number of trials (frequencies) in the power spectrum before
        being used.

        Parameters
        ----------
        threshold : float, optional, default ``1``
            The threshold to be used when reporting p-values of potentially
            significant powers. Must be between 0 and 1.
            Default is ``1`` (all p-values will be reported).

        trial_correction : bool, optional, default ``False``
            A Boolean flag that sets whether the ``threshold`` will be corrected
            by the number of frequencies before being applied. This decreases
            the ``threshold`` (p-values need to be lower to count as significant).
            Default is ``False`` (report all powers) though for any application
            where `threshold`` is set to something meaningful, this should also
            be applied!

        Returns
        -------
        pvals : iterable
            A list of ``(index, p-value)`` tuples for all powers that have p-values
            lower than the threshold specified in ``threshold``.

        """
        if not self.norm == "leahy":
            raise ValueError("This method only works on "
                             "Leahy-normalized power spectra!")

        if trial_correction:
            ntrial = self.power.shape[0]
        else:
            ntrial = 1

        if np.size(self.m) == 1:
            # calculate p-values for all powers
            # leave out zeroth power since it just encodes the number of photons!
            pv = pds_probability(self.power, n_summed_spectra=self.m,
                                 ntrial=ntrial)
        else:
            pv = np.array([pds_probability(power, n_summed_spectra=m,
                                           ntrial=ntrial)
                           for power, m in zip(self.power, self.m)])

        # need to add 1 to the indices to make up for the fact that
        # we left out the first power above!
        indices = np.where(pv < threshold)[0]

        pvals = np.vstack([pv[indices], indices])

        return pvals
