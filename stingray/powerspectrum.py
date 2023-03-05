import copy
import warnings
from collections.abc import Generator, Iterable

import numpy as np
import scipy
import scipy.optimize
import scipy.stats

import stingray.utils as utils
from stingray.crossspectrum import AveragedCrossspectrum, Crossspectrum
from stingray.gti import bin_intervals_from_gtis, check_gtis
from stingray.largememory import createChunkedSpectra, saveData, HAS_ZARR
from stingray.stats import pds_probability, amplitude_upper_limit
from stingray.utils import genDataPath

from .events import EventList
from .gti import cross_two_gtis
from .lightcurve import Lightcurve
from .fourier import avg_pds_from_iterable
from .fourier import avg_pds_from_events
from .fourier import fftfreq, fft
from .fourier import get_flux_iterable_from_segments

try:
    from tqdm import tqdm as show_progress
except ImportError:

    def show_progress(a, **kwargs):
        return a


__all__ = ["Powerspectrum", "AveragedPowerspectrum", "DynamicalPowerspectrum"]


class Powerspectrum(Crossspectrum):
    type = "powerspectrum"
    """
    Make a :class:`Powerspectrum` (also called periodogram) from a (binned)
    light curve. Periodograms can be normalized by either Leahy normalization,
    fractional rms normalization, absolute rms normalization, or not at all.

    You can also make an empty :class:`Powerspectrum` object to populate with
    your own fourier-transformed data (this can sometimes be useful when making
    binned power spectra).

    Parameters
    ----------
    data: :class:`stingray.Lightcurve` object, optional, default ``None``
        The light curve data to be Fourier-transformed.

    norm: {"leahy" | "frac" | "abs" | "none" }, optional, default "frac"
        The normaliation of the power spectrum to be used. Options are
        "leahy", "frac", "abs" and "none", default is "frac".

    Other Parameters
    ----------------
    gti: 2-d float array
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care, especially if these GTIs have overlaps with the input
        object GTIs! If you're getting errors regarding your GTIs, don't
        use this and only give GTIs to the input object before making
        the power spectrum.

    skip_checks: bool
        Skip initial checks, for speed or other reasons (you need to trust your
        inputs!).

    Attributes
    ----------
    norm: {"leahy" | "frac" | "abs" | "none" }
        The normalization of the power spectrum.

    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples.

    power: numpy.ndarray
        The array of normalized squared absolute values of Fourier
        amplitudes.

    power_err: numpy.ndarray
        The uncertainties of ``power``.
        An approximation for each bin given by ``power_err= power/sqrt(m)``.
        Where ``m`` is the number of power averaged in each bin (by frequency
        binning, or averaging power spectra of segments of a light curve).
        Note that for a single realization (``m=1``) the error is equal to the
        power.

    df: float
        The frequency resolution.

    m: int
        The number of averaged powers in each bin.

    n: int
        The number of data points in the light curve.

    nphots: float
        The total number of photons in the light curve.

    legacy: bool
        Use the legacy machinery of ``AveragedPowerspectrum``. This might be
        useful to compare with old results, and is also needed to use light
        curve lists as an input, to conserve the spectra of each segment, or
        to use the large_data option.
    """

    def __init__(
        self, data=None, norm="frac", gti=None, dt=None, lc=None, skip_checks=False, legacy=False
    ):

        self._type = None
        if lc is not None:
            warnings.warn(
                "The lc keyword is now deprecated. Use data " "instead", DeprecationWarning
            )
        if data is None:
            data = lc

        good_input = True
        if not skip_checks:
            good_input = self.initial_checks(
                data1=data, data2=data, norm=norm, gti=gti, lc1=lc, lc2=lc, dt=dt
            )

        norm = norm.lower()
        self.norm = norm
        self.dt = dt

        if not good_input:
            return self._initialize_empty()

        if not legacy and data is not None:
            return self._initialize_from_any_input(data, dt=dt, norm=norm)

        Crossspectrum.__init__(
            self, data1=data, data2=data, norm=norm, gti=gti, dt=dt, skip_checks=True, legacy=legacy
        )
        self.nphots = self.nphots1
        self.dt = dt

    def rebin(self, df=None, f=None, method="mean"):
        """
        Rebin the power spectrum.

        Parameters
        ----------
        df: float
            The new frequency resolution.

        Other Parameters
        ----------------
        f: float
            The rebin factor. If specified, it substitutes ``df`` with
            ``f*self.df``, so ``f>1`` is recommended.

        Returns
        -------
        bin_cs = :class:`Powerspectrum` object
            The newly binned power spectrum.
        """
        bin_ps = Crossspectrum.rebin(self, df=df, f=f, method=method)
        bin_ps.nphots = bin_ps.nphots1

        return bin_ps

    def compute_rms(self, min_freq, max_freq, white_noise_offset=2.0):
        """
        Compute the fractional rms amplitude in the power spectrum
        between two frequencies.

        Parameters
        ----------
        min_freq: float
            The lower frequency bound for the calculation.

        max_freq: float
            The upper frequency bound for the calculation.

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
            ``max_freq``.

        rms_err: float
            The error on the fractional rms amplitude.

        """
        minind = self.freq.searchsorted(min_freq)
        maxind = self.freq.searchsorted(max_freq)
        powers = self.power[minind:maxind]
        nphots = self.nphots

        if self.norm.lower() == "leahy":
            powers_leahy = powers.copy()
        else:
            powers_leahy = self.unnorm_power[minind:maxind].real * 2 / nphots

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
            The error on the fractional rms amplitude.
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
           periodic signal to be determined with this method. This is
           important! If there are other sources of (aperiodic) variability in
           the data, this method will *not* produce correct results, but
           instead produce a large number of spurious false positive
           detections!
        3. There are no significant instrumental effects changing the
           statistical distribution of the powers (e.g. pile-up or dead time)

        By default, the method produces ``(index,p-values)`` for all powers in
        the power spectrum, where index is the numerical index of the power in
        question. If a ``threshold`` is set, then only powers with p-values
        *below* that threshold with their respective indices. If
        ``trial_correction`` is set to ``True``, then the threshold will be
        corrected for the number of trials (frequencies) in the power spectrum
        before being used.

        Parameters
        ----------
        threshold : float, optional, default ``1``
            The threshold to be used when reporting p-values of potentially
            significant powers. Must be between 0 and 1.
            Default is ``1`` (all p-values will be reported).

        trial_correction : bool, optional, default ``False``
            A Boolean flag that sets whether the ``threshold`` will be
            corrected by the number of frequencies before being applied. This
            decreases the ``threshold`` (p-values need to be lower to count as
            significant). Default is ``False`` (report all powers) though for
            any application where `threshold`` is set to something meaningful,
            this should also be applied!

        Returns
        -------
        pvals : iterable
            A list of ``(p-value, index)`` tuples for all powers that have
            p-values lower than the threshold specified in ``threshold``.

        """
        if not self.norm == "leahy":
            raise ValueError("This method only works on " "Leahy-normalized power spectra!")

        if trial_correction:
            ntrial = self.power.shape[0]
        else:
            ntrial = 1

        if np.size(self.m) == 1:
            # calculate p-values for all powers
            # leave out zeroth power since it just encodes the number of
            # photons!
            pv = pds_probability(self.power, n_summed_spectra=self.m, ntrial=ntrial)
        else:
            pv = np.array(
                [
                    pds_probability(power, n_summed_spectra=m, ntrial=ntrial)
                    for power, m in zip(self.power, self.m)
                ]
            )

        # need to add 1 to the indices to make up for the fact that
        # we left out the first power above!
        indices = np.where(pv < threshold)[0]

        pvals = np.vstack([pv[indices], indices])

        return pvals

    def modulation_upper_limit(self, fmin=None, fmax=None, c=0.95):
        """
        Upper limit on a sinusoidal modulation.

        To understand the meaning of this amplitude: if the modulation is
        described by:

        ..math:: p = \overline{p} (1 + a * \sin(x))

        this function returns a.

        If it is a sum of sinusoidal harmonics instead
        ..math:: p = \overline{p} (1 + \sum_l a_l * \sin(lx))
        a is equivalent to :math:`\sqrt(\sum_l a_l^2)`.

        See `stingray.stats.power_upper_limit`,
        `stingray.stats.amplitude_upper_limit`
        for more information.

        The formula used to calculate the upper limit assumes the Leahy
        normalization.
        If the periodogram is in another normalization, we will internally
        convert it to Leahy before calculating the upper limit.

        Parameters
        ----------
        fmin: float
            The minimum frequency to search (defaults to the first nonzero bin)

        fmax: float
            The maximum frequency to search (defaults to the Nyquist frequency)

        Other Parameters
        ----------------
        c: float
            The confidence value for the upper limit (e.g. 0.95 = 95%)

        Returns
        -------
        a: float
            The modulation amplitude that could produce P>pmeas with 1 - c
            probability.

        Examples
        --------
        >>> pds = Powerspectrum()
        >>> pds.norm = "leahy"
        >>> pds.freq = np.arange(0., 5.)
        >>> # Note: this pds has 40 as maximum value between 2 and 5 Hz
        >>> pds.power = np.array([100000, 1, 1, 40, 1])
        >>> pds.m = 1
        >>> pds.nphots = 30000
        >>> pds.modulation_upper_limit(fmin=2, fmax=5, c=0.99)
        0.1016...
        """

        pds = self
        if self.norm != "leahy":
            pds = self.to_norm("leahy")

        freq = pds.freq
        fnyq = np.max(freq)
        power = pds.power
        freq_mask = freq > 0
        if fmin is not None or fmax is not None:
            if fmin is not None:
                freq_mask[freq < fmin] = 0
            if fmax is not None:
                freq_mask[freq > fmax] = 0
        freq = freq[freq_mask]
        power = power[freq_mask]

        maximum_val = np.argmax(power)
        nyq_ratio = freq[maximum_val] / fnyq

        # I multiply by M because the formulas from Vaughan+94 treat summed
        # powers, while here we have averaged powers.
        return amplitude_upper_limit(
            power[maximum_val] * pds.m, pds.nphots, n=pds.m, c=c, nyq_ratio=nyq_ratio, fft_corr=True
        )

    @staticmethod
    def from_time_array(
        times, dt, segment_size=None, gti=None, norm="frac", silent=False, use_common_mean=True
    ):
        """
        Calculate an average power spectrum from an array of event times.

        Parameters
        ----------
        times : `np.array`
            Event arrival times.
        dt : float
            The time resolution of the intermediate light curves
            (sets the Nyquist frequency).

        Other parameters
        ----------------
        segment_size : float
            The length, in seconds, of the light curve segments that will be
            averaged. Only relevant (and required) for
            ``AveragedPowerspectrum``.
        gti: ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
            Additional, optional Good Time intervals that get intersected with
            the GTIs of the input object. Can cause errors if there are
            overlaps between these GTIs and the input object GTIs. If that
            happens, assign the desired GTIs to the input object.
        norm : str, default "frac"
            The normalization of the periodogram. `abs` is absolute rms, `frac`
            is fractional rms, `leahy` is Leahy+83 normalization, and `none` is
            the unnormalized periodogram.
        use_common_mean : bool, default True
            The mean of the light curve can be estimated in each interval, or
            on the full light curve. This gives different results
            (Alston+2013). By default, we assume the mean is calculated on the
            full light curve, but the user can set ``use_common_mean`` to False
            to calculate it on a per-segment basis.
        silent : bool, default False
            Silence the progress bars.
        """

        return powerspectrum_from_time_array(
            times,
            dt,
            segment_size=segment_size,
            gti=gti,
            norm=norm,
            silent=silent,
            use_common_mean=use_common_mean,
        )

    @staticmethod
    def from_events(
        events, dt, segment_size=None, gti=None, norm="frac", silent=False, use_common_mean=True
    ):
        """
        Calculate an average power spectrum from an event list.

        Parameters
        ----------
        events : `stingray.EventList`
            Event list to be analyzed.
        dt : float
            The time resolution of the intermediate light curves
            (sets the Nyquist frequency).

        Other parameters
        ----------------
        segment_size : float
            The length, in seconds, of the light curve segments that will be
            averaged. Only relevant (and required) for
            ``AveragedPowerspectrum``.
        gti: ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
            Additional, optional Good Time intervals that get intersected with
            the GTIs of the input object. Can cause errors if there are
            overlaps between these GTIs and the input object GTIs. If that
            happens, assign the desired GTIs to the input object.
        norm : str, default "frac"
            The normalization of the periodogram. `abs` is absolute rms, `frac`
            is fractional rms, `leahy` is Leahy+83 normalization, and `none` is
            the unnormalized periodogram.
        use_common_mean : bool, default True
            The mean of the light curve can be estimated in each interval, or
            on the full light curve. This gives different results
            (Alston+2013). By default, we assume the mean is calculated on the
            full light curve, but the user can set ``use_common_mean`` to False
            to calculate it on a per-segment basis.
        silent : bool, default False
            Silence the progress bars.
        """
        if gti is None:
            gti = events.gti
        return powerspectrum_from_events(
            events,
            dt,
            segment_size=segment_size,
            gti=gti,
            norm=norm,
            silent=silent,
            use_common_mean=use_common_mean,
        )

    @staticmethod
    def from_lightcurve(
        lc, segment_size=None, gti=None, norm="frac", silent=False, use_common_mean=True
    ):
        """
        Calculate a power spectrum from a light curve.

        Parameters
        ----------
        events : `stingray.Lightcurve`
            Light curve to be analyzed.
        dt : float
            The time resolution of the intermediate light curves
            (sets the Nyquist frequency).

        Other parameters
        ----------------
        segment_size : float
            The length, in seconds, of the light curve segments that will be
            averaged. Only relevant (and required) for
            ``AveragedPowerspectrum``.
        gti: ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
            Additional, optional Good Time intervals that get intersected with
            the GTIs of the input object. Can cause errors if there are
            overlaps between these GTIs and the input object GTIs. If that
            happens, assign the desired GTIs to the input object.
        norm : str, default "frac"
            The normalization of the periodogram. `abs` is absolute rms, `frac`
            is fractional rms, `leahy` is Leahy+83 normalization, and `none` is
            the unnormalized periodogram.
        use_common_mean : bool, default True
            The mean of the light curve can be estimated in each interval, or
            on the full light curve. This gives different results
            (Alston+2013). By default, we assume the mean is calculated on the
            full light curve, but the user can set ``use_common_mean`` to False
            to calculate it on a per-segment basis.
        silent : bool, default False
            Silence the progress bars.
        """
        if gti is None:
            gti = lc.gti
        return powerspectrum_from_lightcurve(
            lc,
            segment_size=segment_size,
            gti=gti,
            norm=norm,
            silent=silent,
            use_common_mean=use_common_mean,
        )

    @staticmethod
    def from_lc_iterable(
        iter_lc, dt, segment_size=None, gti=None, norm="frac", silent=False, use_common_mean=True
    ):
        """
        Calculate the average power spectrum of an iterable collection of
        light curves.

        Parameters
        ----------
        iter_lc : iterable of `stingray.Lightcurve` objects or `np.array`
            Light curves. If arrays, use them as counts.
        dt : float
            The time resolution of the light curves
            (sets the Nyquist frequency)

        Other parameters
        ----------------
        segment_size : float
            The length, in seconds, of the light curve segments that will be
            averaged. Only relevant (and required) for
            ``AveragedPowerspectrum``.
        gti: ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
            Additional, optional Good Time intervals that get intersected with
            the GTIs of the input object. Can cause errors if there are
            overlaps between these GTIs and the input object GTIs. If that
            happens, assign the desired GTIs to the input object.
        norm : str, default "frac"
            The normalization of the periodogram. `abs` is absolute rms, `frac`
            is fractional rms, `leahy` is Leahy+83 normalization, and `none` is
            the unnormalized periodogram.
        use_common_mean : bool, default True
            The mean of the light curve can be estimated in each interval, or
            on the full light curve. This gives different results
            (Alston+2013). By default, we assume the mean is calculated on the
            full light curve, but the user can set ``use_common_mean`` to False
            to calculate it on a per-segment basis.
        silent : bool, default False
            Silence the progress bars.
        """

        return powerspectrum_from_lc_iterable(
            iter_lc,
            dt,
            segment_size=segment_size,
            gti=gti,
            norm=norm,
            silent=silent,
            use_common_mean=use_common_mean,
        )

    def _initialize_from_any_input(
        self,
        data,
        dt=None,
        segment_size=None,
        gti=None,
        norm="frac",
        silent=False,
        use_common_mean=True,
    ):
        """
        Initialize the class, trying to understand the input types.

        The input arguments are the same as ``__init__()``. Based on the type
        of ``data``, this method will call the appropriate
        ``powerspectrum_from_XXXX`` function, and initialize ``self`` with
        the correct attributes.
        """
        if isinstance(data, EventList):
            spec = powerspectrum_from_events(
                data,
                dt,
                segment_size,
                norm=norm.lower(),
                silent=silent,
                use_common_mean=use_common_mean,
                gti=gti,
            )
        elif isinstance(data, Lightcurve):
            spec = powerspectrum_from_lightcurve(
                data,
                segment_size,
                norm=norm,
                silent=silent,
                use_common_mean=use_common_mean,
                gti=gti,
            )
            spec.lc1 = data
        elif isinstance(data, (tuple, list)):
            if not isinstance(data[0], Lightcurve):  # pragma: no cover
                raise TypeError(f"Bad inputs to Powerspectrum: {type(data[0])}")
            dt = data[0].dt
            # This is a list of light curves.
            spec = powerspectrum_from_lc_iterable(
                data,
                dt,
                segment_size,
                norm=norm,
                silent=silent,
                use_common_mean=use_common_mean,
                gti=gti,
            )
        else:  # pragma: no cover
            raise TypeError(f"Bad inputs to Powerspectrum: {type(data)}")

        for key, val in spec.__dict__.items():
            setattr(self, key, val)
        return

    def _initialize_empty(self):
        """Set all attributes to None."""
        self.freq = None
        self.power = None
        self.power_err = None
        self.unnorm_power = None
        self.unnorm_power_err = None
        self.df = None
        self.dt = None
        self.nphots1 = None
        self.m = 1
        self.n = None
        return


class AveragedPowerspectrum(AveragedCrossspectrum, Powerspectrum):
    type = "powerspectrum"
    """
    Make an averaged periodogram from a light curve by segmenting the light
    curve, Fourier-transforming each segment and then averaging the
    resulting periodograms.

    Parameters
    ----------
    data: :class:`stingray.Lightcurve`object OR iterable of :class:`stingray.Lightcurve` objects OR :class:`stingray.EventList` object
        The light curve data to be Fourier-transformed.

    segment_size: float
        The size of each segment to average. Note that if the total
        duration of each :class:`Lightcurve` object in lc is not an integer
        multiple of the ``segment_size``, then any fraction left-over at the
        end of the time series will be lost.

    norm: {"leahy" | "frac" | "abs" | "none" }, optional, default "frac"
        The normalization of the periodogram to be used.

    Other Parameters
    ----------------
    gti: 2-d float array
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care, especially if these GTIs have overlaps with the input
        object GTIs! If you're getting errors regarding your GTIs, don't
        use this and only give GTIs to the input object before making
        the power spectrum.

    silent : bool, default False
         Do not show a progress bar when generating an averaged cross spectrum.
         Useful for the batch execution of many spectra.

    dt: float
        The time resolution of the light curve. Only needed when constructing
        light curves in the case where data is of :class:EventList.

    large_data : bool, default False
        Use only for data larger than 10**7 data points!! Uses zarr and dask
        for computation.

    save_all : bool, default False
        Save all intermediate PDSs used for the final average. Use with care.
        This is likely to fill up your RAM on medium-sized datasets, and to
        slow down the computation when rebinning.

    skip_checks: bool
        Skip initial checks, for speed or other reasons (you need to trust your
        inputs!).

    Attributes
    ----------
    norm: {``leahy`` | ``frac`` | ``abs`` | ``none`` }
        The normalization of the periodogram.

    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples.

    power: numpy.ndarray
        The array of normalized squared absolute values of Fourier
        amplitudes.

    power_err: numpy.ndarray
        The uncertainties of ``power``.
        An approximation for each bin given by ``power_err= power/sqrt(m)``.
        Where ``m`` is the number of power averaged in each bin (by frequency
        binning, or averaging power spectra of segments of a light curve).
        Note that for a single realization (``m=1``) the error is equal to the
        power.

    df: float
        The frequency resolution.

    m: int
        The number of averaged periodograms.

    n: int
        The number of data points in the light curve.

    nphots: float
        The total number of photons in the light curve.

    legacy: bool
        Use the legacy machinery of ``AveragedPowerspectrum``. This might be
        useful to compare with old results, and is also needed to use light
        curve lists as an input, to conserve the spectra of each segment, or to
        use the large_data option.
    """

    def __init__(
        self,
        data=None,
        segment_size=None,
        norm="frac",
        gti=None,
        silent=False,
        dt=None,
        lc=None,
        large_data=False,
        save_all=False,
        skip_checks=False,
        use_common_mean=True,
        legacy=False,
    ):

        self._type = None
        if lc is not None:
            warnings.warn(
                "The lc keyword is now deprecated. Use data " "instead", DeprecationWarning
            )
        # Backwards compatibility: user might have supplied lc instead
        if data is None:
            data = lc

        good_input = True
        if not skip_checks:
            good_input = self.initial_checks(
                data1=data,
                data2=data,
                norm=norm,
                gti=gti,
                lc1=lc,
                lc2=lc,
                dt=dt,
                segment_size=segment_size,
            )

        norm = norm.lower()
        self.norm = norm
        self.dt = dt
        self.save_all = save_all
        self.segment_size = segment_size
        self.show_progress = not silent

        if not good_input:
            return self._initialize_empty()

        if isinstance(data, Generator):
            warnings.warn(
                "The averaged power spectrum from a generator of "
                "light curves pre-allocates the full list of light "
                "curves, losing all advantage of lazy loading. If it "
                "is important for you, use the "
                "AveragedPowerspectrum.from_lc_iterable static "
                "method, specifying the sampling time `dt`."
            )
            data = list(data)

        # The large_data option requires the legacy interface.
        if (large_data or save_all) and not legacy:
            warnings.warn(
                "The large_data option and the save_all options are"
                " only available with the legacy interface"
                " (legacy=True)."
            )
            legacy = True

        if not legacy and data is not None:
            return self._initialize_from_any_input(
                data,
                dt=dt,
                segment_size=segment_size,
                norm=norm,
                silent=silent,
                use_common_mean=use_common_mean,
            )

        if large_data and data is not None:
            if not HAS_ZARR:
                raise ImportError("The large_data option requires zarr.")
            chunks = None

            if isinstance(data, EventList):
                input_data = "EventList"
            elif isinstance(data, Lightcurve):
                input_data = "Lightcurve"
                chunks = int(np.rint(segment_size // data.dt))
                segment_size = chunks * data.dt
            else:
                raise ValueError(f"Invalid input data type: {type(data).__name__}")

            dir_path = saveData(data, persist=False, chunks=chunks)

            data_path = genDataPath(dir_path)
            spec = createChunkedSpectra(
                input_data,
                "AveragedPowerspectrum",
                data_path=data_path,
                segment_size=segment_size,
                norm=norm,
                gti=gti,
                power_type=None,
                silent=silent,
                dt=dt,
            )
            for key, val in spec.__dict__.items():
                setattr(self, key, val)

            return

        if isinstance(data, EventList):
            lengths = data.gti[:, 1] - data.gti[:, 0]
            good = lengths >= segment_size
            data.gti = data.gti[good]

        Powerspectrum.__init__(self, data, norm, gti=gti, dt=dt, skip_checks=True, legacy=legacy)

        return

    def initial_checks(self, *args, **kwargs):
        return AveragedCrossspectrum.initial_checks(self, *args, **kwargs)

    def _make_segment_spectrum(self, lc, segment_size, silent=False):
        """
        Split the light curves into segments of size ``segment_size``, and
        calculate a power spectrum for each.

        Parameters
        ----------
        lc  : :class:`stingray.Lightcurve` objects
            The input light curve.

        segment_size : ``numpy.float``
            Size of each light curve segment to use for averaging.

        Other parameters
        ----------------
        silent : bool, default False
            Suppress progress bars.

        Returns
        -------
        power_all : list of :class:`Powerspectrum` objects
            A list of power spectra calculated independently from each light
            curve segment.

        nphots_all : ``numpy.ndarray``
            List containing the number of photons for all segments calculated
            from ``lc``.
        """
        if not isinstance(lc, Lightcurve):
            raise TypeError("lc must be a Lightcurve object")

        current_gtis = lc.gti

        if self.gti is None:
            self.gti = lc.gti
        else:
            if not np.allclose(lc.gti, self.gti):
                self.gti = np.vstack([self.gti, lc.gti])

        check_gtis(self.gti)

        start_inds, end_inds = bin_intervals_from_gtis(
            current_gtis, segment_size, lc.time, dt=lc.dt
        )

        power_all = []
        nphots_all = []

        local_show_progress = show_progress
        if not self.show_progress or silent:

            def local_show_progress(a):
                return a

        for start_ind, end_ind in local_show_progress(zip(start_inds, end_inds)):
            time = lc.time[start_ind:end_ind]
            counts = lc.counts[start_ind:end_ind]
            counts_err = lc.counts_err[start_ind:end_ind]

            if np.sum(counts) == 0:
                warnings.warn("No counts in interval {}--{}s".format(time[0], time[-1]))
                continue

            lc_seg = Lightcurve(
                time,
                counts,
                err=counts_err,
                err_dist=lc.err_dist.lower(),
                skip_checks=True,
                dt=lc.dt,
            )

            power_seg = Powerspectrum(lc_seg, norm=self.norm)
            power_all.append(power_seg)
            nphots_all.append(np.sum(lc_seg.counts))

        return power_all, nphots_all


class DynamicalPowerspectrum(AveragedPowerspectrum):
    type = "powerspectrum"
    """
    Create a dynamical power spectrum, also often called a *spectrogram*.

    This class will divide a :class:`Lightcurve` object into segments of
    length ``segment_size``, create a power spectrum for each segment and store
    all powers in a matrix as a function of both time (using the mid-point of
    each segment) and frequency.

    This is often used to trace changes in period of a (quasi-)periodic signal
    over time.

    Parameters
    ----------
    lc : :class:`stingray.Lightcurve` or :class:`stingray.EventList` object
        The time series or event list of which the dynamical power spectrum is
        to be calculated.

    segment_size : float, default 1
         Length of the segment of light curve, default value is 1 (in whatever
         units the ``time`` array in the :class:`Lightcurve`` object uses).

    norm: {"leahy" | "frac" | "abs" | "none" }, optional, default "frac"
        The normaliation of the periodogram to be used.

    Other Parameters
    ----------------
    gti: 2-d float array
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care, especially if these GTIs have overlaps with the input
        object GTIs! If you're getting errors regarding your GTIs, don't
        use this and only give GTIs to the input object before making
        the power spectrum.

    Attributes
    ----------
    segment_size: float
        The size of each segment to average. Note that if the total
        duration of each input object in lc is not an integer multiple
        of the ``segment_size``, then any fraction left-over at the end of the
        time series will be lost.

    dyn_ps : np.ndarray
        The matrix of normalized squared absolute values of Fourier
        amplitudes. The axis are given by the ``freq``
        and ``time`` attributes.

    norm: {``leahy`` | ``frac`` | ``abs`` | ``none``}
        The normalization of the periodogram.

    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples.

    df: float
        The frequency resolution.

    dt: float
        The time resolution.
    """

    def __init__(self, lc, segment_size, norm="frac", gti=None, dt=None):
        if isinstance(lc, EventList) and dt is None:
            raise ValueError("To pass an input event lists, please specify dt")

        if isinstance(lc, EventList):
            lc = lc.to_lc(dt)

        if segment_size < 2 * lc.dt:
            raise ValueError("Length of the segment is too short to form a " "light curve!")
        elif segment_size > lc.tseg:
            raise ValueError(
                "Length of the segment is too long to create " "any segments of the light curve!"
            )
        AveragedPowerspectrum.__init__(
            self, data=lc, segment_size=segment_size, norm=norm, gti=gti, dt=dt
        )
        self._make_matrix(lc)

    def _make_matrix(self, lc):
        """
        Create a matrix of powers for each time step and each frequency step.

        Time increases with row index, frequency with column index.

        Parameters
        ----------
        lc : :class:`Lightcurve` object
            The :class:`Lightcurve` object from which to generate the dynamical
            power spectrum.
        """
        ps_all, _ = AveragedPowerspectrum._make_segment_spectrum(self, lc, self.segment_size)
        self.dyn_ps = np.array([ps.power for ps in ps_all]).T

        self.freq = ps_all[0].freq
        current_gti = lc.gti
        if self.gti is not None:
            current_gti = cross_two_gtis(self.gti, current_gti)

        start_inds, end_inds = bin_intervals_from_gtis(
            current_gti, self.segment_size, lc.time, dt=lc.dt
        )

        tstart = lc.time[start_inds]
        tend = lc.time[end_inds]

        self.time = tstart + 0.5 * (tend - tstart)

        # Assign length of lightcurve as time resolution if only one value
        if len(self.time) > 1:
            self.dt = self.time[1] - self.time[0]
        else:
            self.dt = lc.n

        # Assign biggest freq. resolution if only one value
        if len(self.freq) > 1:
            self.df = self.freq[1] - self.freq[0]
        else:
            self.df = 1 / lc.n

    def rebin_frequency(self, df_new, method="sum"):
        """
        Rebin the Dynamic Power Spectrum to a new frequency resolution.
        Rebinning is an in-place operation, i.e. will replace the existing
        ``dyn_ps`` attribute.

        While the new resolution need not be an integer multiple of the
        previous frequency resolution, be aware that if it is not, the last
        bin will be cut off by the fraction left over by the integer division.

        Parameters
        ----------
        df_new: float
            The new frequency resolution of the dynamical power spectrum.
            Must be larger than the frequency resolution of the old dynamical
            power spectrum!

        method: {"sum" | "mean" | "average"}, optional, default "sum"
            This keyword argument sets whether the counts in the new bins
            should be summed or averaged.
        """
        new_dynspec_object = copy.deepcopy(self)
        dynspec_new = []
        for data in self.dyn_ps.T:
            freq_new, bin_counts, bin_err, _ = utils.rebin_data(
                self.freq, data, dx_new=df_new, method=method
            )
            dynspec_new.append(bin_counts)

        new_dynspec_object.freq = freq_new
        new_dynspec_object.dyn_ps = np.array(dynspec_new).T
        new_dynspec_object.df = df_new
        return new_dynspec_object

    def trace_maximum(self, min_freq=None, max_freq=None):
        """
        Return the indices of the maximum powers in each segment
        :class:`Powerspectrum` between specified frequencies.

        Parameters
        ----------
        min_freq: float, default ``None``
            The lower frequency bound.

        max_freq: float, default ``None``
            The upper frequency bound.

        Returns
        -------
        max_positions : np.array
            The array of indices of the maximum power in each segment having
            frequency between ``min_freq`` and ``max_freq``.
        """
        if min_freq is None:
            min_freq = np.min(self.freq)
        if max_freq is None:
            max_freq = np.max(self.freq)

        max_positions = []
        for ps in self.dyn_ps.T:
            indices = np.logical_and(self.freq <= max_freq, min_freq <= self.freq)
            max_power = np.max(ps[indices])
            max_positions.append(np.where(ps == max_power)[0][0])

        return np.array(max_positions)

    def rebin_time(self, dt_new, method="sum"):
        """
        Rebin the Dynamic Power Spectrum to a new time resolution.
        While the new resolution need not be an integer multiple of the
        previous time resolution, be aware that if it is not, the last bin
        will be cut off by the fraction left over by the integer division.

        Parameters
        ----------
        dt_new: float
            The new time resolution of  the dynamical power spectrum.
            Must be larger than the time resolution of the old dynamical power
            spectrum!

        method: {"sum" | "mean" | "average"}, optional, default "sum"
            This keyword argument sets whether the counts in the new bins
            should be summed or averaged.

        Returns
        -------
        time_new: numpy.ndarray
            Time axis with new rebinned time resolution.

        dynspec_new: numpy.ndarray
            New rebinned Dynamical Power Spectrum.
        """
        if dt_new < self.dt:
            raise ValueError("New time resolution must be larger than " "old time resolution!")

        new_dynspec_object = copy.deepcopy(self)

        dynspec_new = []
        for data in self.dyn_ps:
            time_new, bin_counts, bin_err, _ = utils.rebin_data(
                self.time, data, dt_new, method=method
            )
            dynspec_new.append(bin_counts)

        new_dynspec_object.time = time_new
        new_dynspec_object.dyn_ps = np.array(dynspec_new)
        new_dynspec_object.dt = dt_new
        return new_dynspec_object


def powerspectrum_from_time_array(
    times, dt, segment_size=None, gti=None, norm="frac", silent=False, use_common_mean=True
):
    """
    Calculate a power spectrum from an array of event times.

    Parameters
    ----------
    times : `np.array`
        Event arrival times.
    dt : float
        The time resolution of the intermediate light curves
        (sets the Nyquist frequency).

    Other parameters
    ----------------
    segment_size : float
        The length, in seconds, of the light curve segments that will be
        averaged. Only required (and used) for ``AveragedPowerspectrum``.
    gti : ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
        Additional, optional Good Time intervals that get intersected with
        the GTIs of the input object. Can cause errors if there are
        overlaps between these GTIs and the input object GTIs. If that
        happens, assign the desired GTIs to the input object.
    norm : str, default "frac"
        The normalization of the periodogram. `abs` is absolute rms, `frac`
        is fractional rms, `leahy` is Leahy+83 normalization, and `none` is
        the unnormalized periodogram.
    use_common_mean : bool, default True
        The mean of the light curve can be estimated in each interval, or
        on the full light curve. This gives different results
        (Alston+2013). By default, we assume the mean is calculated on the
        full light curve, but the user can set ``use_common_mean`` to False
        to calculate it on a per-segment basis.
    silent : bool, default False
        Silence the progress bars.

    Returns
    -------
    spec : `AveragedPowerspectrum` or `Powerspectrum`
        The output periodogram.
    """
    force_averaged = segment_size is not None
    # Suppress progress bar for single periodogram
    silent = silent or (segment_size is None)
    table = avg_pds_from_events(
        times, gti, segment_size, dt, norm=norm, use_common_mean=use_common_mean, silent=silent
    )

    return _create_powerspectrum_from_result_table(table, force_averaged=force_averaged)


def powerspectrum_from_events(
    events, dt, segment_size=None, gti=None, norm="frac", silent=False, use_common_mean=True
):
    """
    Calculate a power spectrum from an event list.

    Parameters
    ----------
    events : `stingray.EventList`
        Event list to be analyzed.
    dt : float
        The time resolution of the intermediate light curves
        (sets the Nyquist frequency)

    Other parameters
    ----------------
    segment_size : float
        The length, in seconds, of the light curve segments that will be
        averaged. Only required (and used) for ``AveragedPowerspectrum``.
    gti : ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
        Additional, optional Good Time intervals that get intersected with
        the GTIs of the input object. Can cause errors if there are
        overlaps between these GTIs and the input object GTIs. If that
        happens, assign the desired GTIs to the input object.
    norm : str, default "frac"
        The normalization of the periodogram. `abs` is absolute rms, `frac`
        is fractional rms, `leahy` is Leahy+83 normalization, and `none` is
        the unnormalized periodogram.
    use_common_mean : bool, default True
        The mean of the light curve can be estimated in each interval, or
        on the full light curve. This gives different results
        (Alston+2013). By default, we assume the mean is calculated on the
        full light curve, but the user can set ``use_common_mean`` to False
        to calculate it on a per-segment basis.
    silent : bool, default False
        Silence the progress bars.

    Returns
    -------
    spec : `AveragedPowerspectrum` or `Powerspectrum`
        The output periodogram.
    """
    if gti is None:
        gti = events.gti
    return powerspectrum_from_time_array(
        events.time,
        dt,
        segment_size,
        gti,
        norm=norm,
        silent=silent,
        use_common_mean=use_common_mean,
    )


def powerspectrum_from_lightcurve(
    lc, segment_size=None, gti=None, norm="frac", silent=False, use_common_mean=True
):
    """
    Calculate a power spectrum from a light curve

    Parameters
    ----------
    events : `stingray.Lightcurve`
        Light curve to be analyzed.
    dt : float
        The time resolution of the intermediate light curves
        (sets the Nyquist frequency)

    Other parameters
    ----------------
    segment_size : float
        The length, in seconds, of the light curve segments that will be
        averaged. Only required (and used) for ``AveragedPowerspectrum``.
    gti : ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
        Additional, optional Good Time intervals that get intersected with
        the GTIs of the input object. Can cause errors if there are
        overlaps between these GTIs and the input object GTIs. If that
        happens, assign the desired GTIs to the input object.
    norm : str, default "frac"
        The normalization of the periodogram. `abs` is absolute rms, `frac`
        is fractional rms, `leahy` is Leahy+83 normalization, and `none` is
        the unnormalized periodogram.
    use_common_mean : bool, default True
        The mean of the light curve can be estimated in each interval, or
        on the full light curve. This gives different results
        (Alston+2013). By default, we assume the mean is calculated on the
        full light curve, but the user can set ``use_common_mean`` to False
        to calculate it on a per-segment basis.
    silent : bool, default False
        Silence the progress bars.

    Returns
    -------
    spec : `AveragedPowerspectrum` or `Powerspectrum`
        The output periodogram.
    """
    force_averaged = segment_size is not None
    # Suppress progress bar for single periodogram
    silent = silent or (segment_size is None)
    err = None
    if lc.err_dist == "gauss":
        err = lc.counts_err
    if gti is None:
        gti = lc.gti

    table = avg_pds_from_events(
        lc.time,
        gti,
        segment_size,
        lc.dt,
        norm=norm,
        use_common_mean=use_common_mean,
        silent=silent,
        fluxes=lc.counts,
        errors=err,
    )

    return _create_powerspectrum_from_result_table(table, force_averaged=force_averaged)


def powerspectrum_from_lc_iterable(
    iter_lc, dt, segment_size=None, gti=None, norm="frac", silent=False, use_common_mean=True
):
    """
    Calculate an average power spectrum from an iterable collection of light
    curves.

    Parameters
    ----------
    iter_lc : iterable of `stingray.Lightcurve` objects or `np.array`
        Light curves. If arrays, use them as counts.
    dt : float
        The time resolution of the light curves
        (sets the Nyquist frequency).

    Other parameters
    ----------------
    segment_size : float, default None
        The length, in seconds, of the light curve segments that will be
        averaged. If not ``None``, it will be used to check the segment size of
         the output.
    gti : ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
        Additional, optional Good Time intervals that get intersected with
        the GTIs of the input object. Can cause errors if there are
        overlaps between these GTIs and the input object GTIs. If that
        happens, assign the desired GTIs to the input object.
    norm : str, default "frac"
        The normalization of the periodogram. `abs` is absolute rms, `frac`
        is fractional rms, `leahy` is Leahy+83 normalization, and `none` is
        the unnormalized periodogram.
    use_common_mean : bool, default True
        The mean of the light curve can be estimated in each interval, or
        on the full light curve. This gives different results
        (Alston+2013). By default, we assume the mean is calculated on the
        full light curve, but the user can set ``use_common_mean`` to False
        to calculate it on a per-segment basis.
    silent : bool, default False
        Silence the progress bars.

    Returns
    -------
    spec : `AveragedPowerspectrum` or `Powerspectrum`
        The output periodogram.
    """
    force_averaged = segment_size is not None
    # Suppress progress bar for single periodogram
    silent = silent or (segment_size is None)

    common_gti = gti

    def iterate_lc_counts(iter_lc):
        for lc in iter_lc:
            if hasattr(lc, "counts"):
                n_bin = np.rint(segment_size / lc.dt).astype(int)

                gti = lc.gti
                if common_gti is not None:
                    gti = cross_two_gtis(common_gti, lc.gti)
                err = None
                if lc.err_dist == "gauss":
                    err = lc.counts_err

                flux_iterable = get_flux_iterable_from_segments(
                    lc.time, gti, segment_size, n_bin, fluxes=lc.counts, errors=err
                )
                for out in flux_iterable:
                    yield out
            elif isinstance(lc, Iterable):
                yield lc
            else:
                raise TypeError(
                    "The inputs to `powerspectrum_from_lc_iterable`"
                    " must be Lightcurve objects or arrays"
                )

    table = avg_pds_from_iterable(
        iterate_lc_counts(iter_lc), dt, norm=norm, use_common_mean=use_common_mean, silent=silent
    )
    return _create_powerspectrum_from_result_table(table, force_averaged=force_averaged)


def _create_powerspectrum_from_result_table(table, force_averaged=False):
    """
    Copy the columns and metadata from the results of
    ``stingray.fourier.avg_pds_from_XX`` functions into
    `AveragedPowerspectrum` or `Powerspectrum` objects.

    By default, allocates a Powerspectrum object if the number of
    averaged spectra is 1, otherwise an AveragedPowerspectrum.
    If the user specifies ``force_averaged=True``, it always allocates
    an AveragedPowerspectrum.

    Parameters
    ----------
    table : `astropy.table.Table`
        results of `avg_cs_from_iterables` or `avg_cs_from_iterables_quick`

    Other parameters
    ----------------
    force_averaged : bool, default False

    Returns
    -------
    spec : `AveragedPowerspectrum` or `Powerspectrum`
        The output periodogram.
    """
    if table.meta["m"] > 1 or force_averaged:
        cs = AveragedPowerspectrum()
    else:
        cs = Powerspectrum()

    cs.freq = np.array(table["freq"])
    cs.power = np.array(table["power"])
    cs.unnorm_power = np.array(table["unnorm_power"])

    for attr, val in table.meta.items():
        setattr(cs, attr, val)

    cs.err_dist = "poisson"
    if hasattr(cs, "variance") and cs.variance is not None:
        cs.err_dist = "gauss"

    cs.power_err = cs.power / np.sqrt(cs.m)
    cs.unnorm_power_err = cs.unnorm_power / np.sqrt(cs.m)
    cs.nphots1 = cs.nphots
    return cs
