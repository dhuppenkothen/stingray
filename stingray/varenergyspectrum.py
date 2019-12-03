
import numpy as np
from stingray.gti import check_separate, cross_two_gtis, create_gti_mask
from stingray.lightcurve import Lightcurve
from stingray.utils import assign_value_if_none, simon, excess_variance
from stingray.crossspectrum import AveragedCrossspectrum
from abc import ABCMeta, abstractmethod
import six


__all__ = ["VarEnergySpectrum", "RmsEnergySpectrum", "LagEnergySpectrum", "ExcessVarianceSpectrum"]


def _decode_energy_specification(energy_spec):
    """Decode the energy specification tuple.

    Parameters
    ----------
    energy_spec : iterable
        list containing the energy specification
        Must have the following structure:
            * energy_spec[0]: lower edge of (log) energy space
            * energy_spec[1]: upper edge of (log) energy space
            * energy_spec[2] +1 : energy bin edges (hence the +1)
            * {`lin` | `log`} flat deciding whether the energy space is linear
              or logarithmic

    Returns
    -------
    energies : numpy.ndarray
        An array of lower/upper bin edges for the energy array

    Examples
    --------
    >>> _decode_energy_specification([0, 2, 2, 'lin'])
    Traceback (most recent call last):
     ...
    ValueError: Energy specification must be a tuple
    >>> a = _decode_energy_specification((0, 2, 2, 'lin'))
    >>> np.all(a == [0, 1, 2])
    True
    >>> a = _decode_energy_specification((1, 4, 2, 'log'))
    >>> np.all(a == [1, 2, 4])
    True
    """
    if not isinstance(energy_spec, tuple):
        raise ValueError("Energy specification must be a tuple")

    if energy_spec[-1].lower() not in ["lin", "log"]:
        raise ValueError("Incorrect energy specification")

    log_distr = True if energy_spec[-1].lower() == "log" else False

    if log_distr:
        energies = np.logspace(np.log10(energy_spec[0]),
                               np.log10(energy_spec[1]),
                               energy_spec[2] + 1)
    else:
        energies = np.linspace(energy_spec[0], energy_spec[1],
                               energy_spec[2] + 1)

    return energies


@six.add_metaclass(ABCMeta)
class VarEnergySpectrum(object):
    """
    Base class for variability-energy spectrum.

    This class is only a base for the various variability spectra, and it's
    not to be instantiated by itself.

    Parameters
    ----------
    events : :class:`stingray.events.EventList` object
        event list

    freq_interval : ``[f0, f1]``, floats
        the frequency range over which calculating the variability quantity

    energy_spec : list or tuple ``(emin, emax, N, type)``
        if a ``list`` is specified, this is interpreted as a list of bin edges;
        if a ``tuple`` is provided, this will encode the minimum and maximum
        energies, the number of intervals, and ``lin`` or ``log``.

    Other Parameters
    ----------------
    ref_band : ``[emin, emax``], floats; default ``None``
        minimum and maximum energy of the reference band. If ``None``, the
        full band is used.

    use_pi : bool, default ``False``
        Use channel instead of energy

    events2 : :class:`stingray.events.EventList` object
        event list for the second channel, if not the same. Useful if the
        reference band has to be taken from another detector.

    Attributes
    ----------
    events1 : array-like
        list of events used to produce the spectrum

    events2 : array-like
        if the spectrum requires it, second list of events

    freq_interval : array-like
        interval of frequencies used to calculate the spectrum

    energy_intervals : ``[[e00, e01], [e10, e11], ...]``
        energy intervals used for the spectrum

    spectrum : array-like
        the spectral values, corresponding to each energy interval

    spectrum_error : array-like
        the error bars corresponding to spectrum

    """
    def __init__(self, events, freq_interval, energy_spec, ref_band=None,
                 bin_time=1, use_pi=False, segment_size=None, events2=None):

        self.events1 = events
        self.events2 = assign_value_if_none(events2, events)
        self.freq_interval = freq_interval
        self.use_pi = use_pi
        self.bin_time = bin_time
        if isinstance(energy_spec, tuple):
            energies = _decode_energy_specification(energy_spec)
        else:
            energies = np.asarray(energy_spec)

        self.energy_intervals = list(zip(energies[0: -1], energies[1:]))

        self.ref_band = np.asarray(assign_value_if_none(ref_band,
                                                        [0, np.inf]))

        if len(self.ref_band.shape) <= 1:
            self.ref_band = np.asarray([self.ref_band])

        self.segment_size = segment_size
        self.spectrum, self.spectrum_error = self._spectrum_function()

    def _decide_ref_intervals(self, channel_band, ref_band):
        """
        Ensures that the ``channel_band`` (i.e. the band of interest) is
        not contained within the ``ref_band`` (i.e. the reference band)

        Parameters
        ----------
        channel_band : iterable of type ``[elow, ehigh]``
            The lower/upper limits of the energies to be contained in the band
            of interest

        ref_band : iterable
            The lower/upper limits of the energies in the reference band

        Returns
        -------
        ref_intervals : iterable
            The channels that are both in the reference band in not in the
            bands of interest
        """
        channel_band = np.asarray(channel_band)
        ref_band = np.asarray(ref_band)
        if len(ref_band.shape) <= 1:
            ref_band = np.asarray([ref_band])
        if check_separate(ref_band, [channel_band]):
            return np.asarray(ref_band)
        not_channel_band = [[0, channel_band[0]],
                            [channel_band[1], np.max([np.max(ref_band),
                                                      channel_band[1] + 1])]]

        return cross_two_gtis(ref_band, not_channel_band)

    def _construct_lightcurves(self, channel_band, tstart=None, tstop=None,
                               exclude=True, only_base=False):
        """
        Construct light curves from event data, for each band of interest.

        Parameters
        ----------
        channel_band : iterable of type ``[elow, ehigh]``
            The lower/upper limits of the energies to be contained in the band
            of interest

        tstart : float, optional, default ``None``
            A common start time (if start of observation is different from
            the first recorded event)

        tstop : float, optional, default ``None``
            A common stop time (if start of observation is different from
            the first recorded event)

        exclude : bool, optional, default ``True``
            if ``True``, exclude the band of interest from the reference band

        only_base : bool, optional, default ``False``
            if ``True``, only return the light curve of the channel of interest, not
            that of the reference band

        Returns
        -------
        base_lc : :class:`Lightcurve` object
            The light curve of the channels of interest

        ref_lc : :class:`Lightcurve` object (only returned if ``only_base`` is ``False``)
            The reference light curve for comparison with ``base_lc``
        """
        if self.use_pi:
            energies1 = self.events1.pi
            energies2 = self.events2.pi
        else:
            energies2 = self.events2.energy
            energies1 = self.events1.energy

        gti = cross_two_gtis(self.events1.gti, self.events2.gti)

        tstart = assign_value_if_none(tstart, gti[0, 0])
        tstop = assign_value_if_none(tstop, gti[-1, -1])

        good = (energies1 >= channel_band[0]) & (energies1 < channel_band[1])
        base_lc = Lightcurve.make_lightcurve(self.events1.time[good],
                                             self.bin_time,
                                             tstart=tstart,
                                             tseg=tstop - tstart,
                                             gti=gti,
                                             mjdref=self.events1.mjdref)

        if only_base:
            return base_lc

        if exclude:
            ref_intervals = self._decide_ref_intervals(channel_band,
                                                       self.ref_band)
        else:
            ref_intervals = self.ref_band

        ref_lc = Lightcurve(base_lc.time, np.zeros_like(base_lc.counts),
                            gti=base_lc.gti, mjdref=base_lc.mjdref,
                            err_dist='gauss')

        for i in ref_intervals:
            good = (energies2 >= i[0]) & (energies2 < i[1])
            new_lc = Lightcurve.make_lightcurve(self.events2.time[good],
                                                self.bin_time,
                                                tstart=tstart,
                                                tseg=tstop - tstart,
                                                gti=base_lc.gti,
                                                mjdref=self.events2.mjdref)
            ref_lc = ref_lc + new_lc

        ref_lc.err_dist = base_lc.err_dist
        return base_lc, ref_lc

    @abstractmethod
    def _spectrum_function(self):
        pass


class RmsEnergySpectrum(VarEnergySpectrum):
    """Calculate the rms-Energy spectrum.

    For each energy interval, calculate the power density spectrum in
    fractional r.m.s. normalization. If ``events2`` is specified, the cospectrum
    is used instead of the PDS.

    Parameters
    ----------
    events : :class:`stingray.events.EventList` object
        event list

    freq_interval : ``[f0, f1]``, list of float
        the frequency range over which calculating the variability quantity

    energy_spec : list or tuple ``(emin, emax, N, type)``
        if a ``list`` is specified, this is interpreted as a list of bin edges;
        if a ``tuple`` is provided, this will encode the minimum and maximum
        energies, the number of intervals, and ``lin`` or ``log``.

    Other Parameters
    ----------------
    ref_band : ``[emin, emax]``, float; default ``None``
        minimum and maximum energy of the reference band. If ``None``, the
        full band is used.

    use_pi : bool, default ``False``
        Use channel instead of energy

    events2 : :class:`stingray.events.EventList` object
        event list for the second channel, if not the same. Useful if the
        reference band has to be taken from another detector.

    Attributes
    ----------
    events1 : array-like
        list of events used to produce the spectrum

    events2 : array-like
        if the spectrum requires it, second list of events

    freq_interval : array-like
        interval of frequencies used to calculate the spectrum

    energy_intervals : ``[[e00, e01], [e10, e11], ...]``
        energy intervals used for the spectrum

    spectrum : array-like
        the spectral values, corresponding to each energy interval

    spectrum_error : array-like
        the errorbars corresponding to spectrum
    """
    def _spectrum_function(self):

        rms_spec = np.zeros(len(self.energy_intervals))
        rms_spec_err = np.zeros_like(rms_spec)
        for i, eint in enumerate(self.energy_intervals):
            base_lc, ref_lc = self._construct_lightcurves(eint,
                                                          exclude=False)
            try:
                xspect = AveragedCrossspectrum(base_lc, ref_lc,
                                               segment_size=self.segment_size,
                                               norm='frac')
            except AssertionError as e:
                # Avoid "Mean count rate is <= 0. Something went wrong" assertion.
                simon("AssertionError: " + str(e))
            else:
                good = (xspect.freq >= self.freq_interval[0]) & \
                       (xspect.freq < self.freq_interval[1])
                rms_spec[i] = np.sqrt(np.sum(xspect.power[good] * xspect.df))

                # Root squared sum of errors of the spectrum
                root_sq_err_sum = \
                    np.sqrt(np.sum((xspect.power_err[good] * xspect.df) ** 2))
                # But the rms is the squared root. So,
                # Error propagation
                rms_spec_err[i] = 1 / (2 * rms_spec[i]) * root_sq_err_sum

        return rms_spec, rms_spec_err


class LagEnergySpectrum(VarEnergySpectrum):
    """Calculate the lag-energy spectrum.

    For each energy interval, calculate the mean lag in the specified frequency
    range. If ``events2`` is specified, the reference band is taken from the second
    event list.

    Parameters
    ----------
    events : :class:`stingray.events.EventList` object
        event list

    freq_interval : ``[f0, f1]``, list of float
        the frequency range over which calculating the variability quantity

    energy_spec : list or tuple ``(emin, emax, N, type)``
        if a list is specified, this is interpreted as a list of bin edges;
        if a tuple is provided, this will encode the minimum and maximum
        energies, the number of intervals, and ``lin`` or ``log``.

    Other Parameters
    ----------------
    ref_band : ``[emin, emax]``, float; default ``None``
        minimum and maximum energy of the reference band. If ``None``, the
        full band is used.

    use_pi : bool, default ``False``
        Use channel instead of energy

    events2 : :class:`stingray.events.EventList` object
        event list for the second channel, if not the same. Useful if the
        reference band has to be taken from another detector.

    Attributes
    ----------
    events1 : array-like
        list of events used to produce the spectrum

    events2 : array-like
        if the spectrum requires it, second list of events

    freq_interval : array-like
        interval of frequencies used to calculate the spectrum

    energy_intervals : ``[[e00, e01], [e10, e11], ...]``
        energy intervals used for the spectrum

    spectrum : array-like
        the spectral values, corresponding to each energy interval

    spectrum_error : array-like
        the errorbars corresponding to spectrum
    """
    def _spectrum_function(self):

        lag_spec = np.zeros(len(self.energy_intervals))
        lag_spec_err = np.zeros_like(lag_spec)
        for i, eint in enumerate(self.energy_intervals):
            base_lc, ref_lc = self._construct_lightcurves(eint)
            try:
                xspect = AveragedCrossspectrum(base_lc, ref_lc,
                                               segment_size=self.segment_size)
            except AssertionError as e:
                # Avoid assertions in AveragedCrossspectrum.
                simon("AssertionError: " + str(e))
            else:
                good = (xspect.freq >= self.freq_interval[0]) & \
                       (xspect.freq < self.freq_interval[1])
                lag, lag_err = xspect.time_lag()
                good_lag, good_lag_err = lag[good], lag_err[good]
                coh, coh_err = xspect.coherence()
                lag_spec[i] = np.mean(good_lag)
                coh_check = coh > 1.2 / (1 + 0.2 * xspect.m)
                if not np.all(coh_check[good]):
                    simon("Coherence is not ideal over the specified energy "
                          "range. Lag values and uncertainties might be "
                          "underestimated. See Epitropakis and Papadakis, "
                          "A\&A 591, 1113, 2016")

                # Root squared sum of errors of the spectrum
                # Verified!
                lag_spec_err[i] = \
                    np.sqrt(np.sum(good_lag_err**2) / len(good_lag))

        return lag_spec, lag_spec_err


class ExcessVarianceSpectrum(VarEnergySpectrum):
    """Calculate the Excess Variance spectrum.

    For each energy interval, calculate the excess variance in the specified
    frequency range.

    Parameters
    ----------
    events : :class:`stingray.events.EventList` object
        event list

    freq_interval : ``[f0, f1]``, list of float
        the frequency range over which calculating the variability quantity

    energy_spec : list or tuple ``(emin, emax, N, type)``
        if a list is specified, this is interpreted as a list of bin edges;
        if a tuple is provided, this will encode the minimum and maximum
        energies, the number of intervals, and ``lin`` or ``log``.

    Other Parameters
    ----------------
    ref_band : ``[emin, emax]``, floats; default ``None``
        minimum and maximum energy of the reference band. If ``None``, the
        full band is used.

    use_pi : bool, default ``False``
        Use channel instead of energy

    Attributes
    ----------
    events1 : array-like
        list of events used to produce the spectrum

    freq_interval : array-like
        interval of frequencies used to calculate the spectrum

    energy_intervals : ``[[e00, e01], [e10, e11], ...]``
        energy intervals used for the spectrum

    spectrum : array-like
        the spectral values, corresponding to each energy interval

    spectrum_error : array-like
        the errorbars corresponding to spectrum
    """
    def __init__(self, events, freq_interval, energy_spec,
                 bin_time=1, use_pi=False, segment_size=None,
                 normalization='fvar'):

        self.normalization = normalization
        accepted_normalizations = ['fvar', 'none']
        if normalization not in accepted_normalizations:
            raise ValueError('The normalization of excess variance must be '
                             'one of {}'.format(accepted_normalizations))

        VarEnergySpectrum.__init__(self, events, freq_interval, energy_spec,
                                   bin_time=bin_time, use_pi=use_pi,
                                   segment_size=segment_size)

    def _spectrum_function(self):
        spec = np.zeros(len(self.energy_intervals))
        spec_err = np.zeros_like(spec)
        for i, eint in enumerate(self.energy_intervals):
            lc = self._construct_lightcurves(eint, exclude=False,
                                             only_base=True)

            spec[i], spec_err[i] = excess_variance(lc, self.normalization)

        return spec, spec_err
