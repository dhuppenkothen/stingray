"""
Basic pulsar-related functions and statistics.
"""
from __future__ import division, print_function, absolute_import
import numpy as np
import collections
from ..utils import simon, jit
from scipy.optimize import minimize, basinhopping, curve_fit


def _default_value_if_no_key(dictionary, key, default):
    try:
        return dictionary[key]
    except:
        return default


def pulse_phase(times, *frequency_derivatives, **opts):
    """Calculate pulse phase from the frequency and its derivatives.

    Parameters
    ----------
    times : array of floats
        The times at which the phase is calculated
    *frequency_derivatives: floats
        List of derivatives in increasing order, starting from zero.

    Returns
    -------
    phases : array of floats
        The absolute pulse phase

    Other Parameters
    ----------------
    ph0 : float
        The starting phase
    to_1 : bool, default True
        Only return the fractional part of the phase, normalized from 0 to 1
    """
    ph0 = _default_value_if_no_key(opts, "ph0", 0)
    to_1 = _default_value_if_no_key(opts, "to_1", True)
    ph = ph0

    for i_f, f in enumerate(frequency_derivatives):
        ph += 1 / np.math.factorial(i_f + 1) * times**(i_f + 1) * f

    if to_1:
        ph -= np.floor(ph)
    return ph


def phase_exposure(start_time, stop_time, period, nbin=16, gtis=None):
    '''Calculate the exposure on each phase of a pulse profile.

    Parameters
    ----------
    start_time, stop_time : float
        Starting and stopping time (or phase if ``period``==1)
    period : float
        The pulse period (if 1, equivalent to phases)

    Returns
    -------
    expo : array of floats
        The normalized exposure of each bin in the pulse profile (1 is the
        highest exposure, 0 the lowest)

    Other parameters
    ----------------
    nbin : int, optional, default 16
        The number of bins in the profile
    gtis : [[gti0_0, gti0_1], [gti1_0, gti1_1], ...], optional, default None
        Good Time Intervals
    '''
    if gtis is None:
        gtis = np.array([[start_time, stop_time]])

    # Use precise floating points -------------
    start_time = np.longdouble(start_time)
    stop_time = np.longdouble(stop_time)
    period = np.longdouble(period)
    gtis = np.array(gtis, dtype=np.longdouble)
    # -----------------------------------------

    expo = np.zeros(nbin)
    phs = np.linspace(0, 1, nbin+1)
    phs = np.array(list(zip(phs[0:-1], phs[1:])))

    # Discard gtis outside [start, stop]
    good = np.logical_and(gtis[:, 0] < stop_time, gtis[:, 1] > start_time)
    gtis = gtis[good]

    for g in gtis:
        g0 = g[0]
        g1 = g[1]
        if g0 < start_time:
            # If the start of the fold is inside a gti, start from there
            g0 = start_time
        if g1 > stop_time:
            # If the end of the fold is inside a gti, end there
            g1 = stop_time
        length = g1 - g0
        # How many periods inside this length?
        nraw = length / period
        # How many integer periods?
        nper = nraw.astype(int)

        # First raw exposure: the number of periods
        expo += nper / nbin

        # FRACTIONAL PART =================
        # What remains is additional exposure for part of the profile.
        start_phase = np.fmod(g0 / period, 1)
        end_phase = nraw - nper + start_phase

        limits = [[start_phase, end_phase]]
        # start_phase is always < 1. end_phase not always. In this case...
        if end_phase > 1:
            limits = [[0, end_phase - 1], [start_phase, 1]]

        for l in limits:
            l0 = l[0]
            l1 = l[1]
            # Discards bins untouched by these limits
            goodbins = np.logical_and(phs[:, 0] < l1, phs[:, 1] > l0)
            idxs = np.arange(len(phs), dtype=int)[goodbins]
            for i in idxs:
                start = max([phs[i, 0], l0])
                stop = min([phs[i, 1], l1])
                w = stop - start
                expo[i] += w

    return expo / np.max(expo)


def fold_events(times, *frequency_derivatives, **opts):
    '''Epoch folding with exposure correction.

    Parameters
    ----------
    times : array of floats
    f, fdot, fddot... : float
        The frequency and any number of derivatives.

    Returns
    -------
    phase_bins : array of floats
    The phases corresponding to the pulse profile
    profile : array of floats
    The pulse profile
    profile_err : array of floats
    The uncertainties on the pulse profile

    Other Parameters
    ----------------
    nbin : int, optional, default 16
        The number of bins in the pulse profile
    weights : float or array of floats, optional
        The weights of the data. It can either be specified as a single value
        for all points, or an array with the same length as ``time``
    gtis : [[gti0_0, gti0_1], [gti1_0, gti1_1], ...], optional
        Good time intervals
    ref_time : float, optional, default 0
        Reference time for the timing solution

    '''
    nbin = _default_value_if_no_key(opts, "nbin", 16)
    weights = _default_value_if_no_key(opts, "weights", 1)
    gtis = _default_value_if_no_key(opts, "gtis",
                                    np.array([[times[0], times[-1]]]))
    ref_time = _default_value_if_no_key(opts, "ref_time", 0)
    expocorr = _default_value_if_no_key(opts, "expocorr", False)

    if not isinstance(weights, collections.Iterable):
        weights *= np.ones(len(times))

    gtis -= ref_time
    times -= ref_time
    dt = times[1] - times[0]
    start_time = times[0]
    stop_time = times[-1] + dt

    phases = pulse_phase(times, *frequency_derivatives, to_1=True)
    gti_phases = pulse_phase(gtis, *frequency_derivatives, to_1=False)
    start_phase, stop_phase = pulse_phase(np.array([start_time, stop_time]),
                                          *frequency_derivatives,
                                          to_1=False)

    raw_profile, bins = np.histogram(phases,
                                     bins=np.linspace(0, 1, nbin + 1),
                                     weights=weights)

    if expocorr:
        expo_norm = phase_exposure(start_phase, stop_phase, 1, nbin)
        simon("For exposure != 1, the uncertainty might be incorrect")
    else:
        expo_norm = 1

    # TODO: this is wrong. Need to extend this to non-1 weights
    raw_profile_err = np.sqrt(raw_profile)

    return bins[:-1] + np.diff(bins) / 2, raw_profile / expo_norm, \
        raw_profile_err / expo_norm


def stat(profile, err=None):
    """Calculate the epoch folding statistics \'a la Leahy et al. (1983).

    Parameters
    ----------
    profile : array
        The pulse profile

    Returns
    -------
    stat : float
        The epoch folding statistics

    Other Parameters
    ----------------
    err : float or array
        The uncertainties on the pulse profile
    """
    mean = np.mean(profile)
    if err is None:
        err = np.sqrt(mean)
    return np.sum((profile - mean) ** 2 / err ** 2)


def fold_profile_probability(stat, nbin, ntrial=1):
    """Calculate the probability of a certain folded profile, due to noise.

    Parameters
    ----------
    stat : float
        The epoch folding statistics
    nbin : int
        The number of bins in the profile

    Returns
    -------
    p : float
        The probability that the profile has been produced by noise

    Other Parameters
    ----------------
    ntrial : int
        The number of trials executed to find this profile
    """
    if ntrial > 1:
        simon("fold: The treatment of ntrial is very rough. Use with caution")
    from scipy import stats
    return stats.chi2.sf(stat, (nbin - 1)) * ntrial


def fold_detection_level(nbin, epsilon=0.01, ntrial=1):
    """Return the detection level for a folded profile.

    See Leahy et al. (1983).

    Parameters
    ----------
    nbin : int
        The number of bins in the profile
    epsilon : float, default 0.01
        The fractional probability that the signal has been produced by noise

    Returns
    -------
    detlev : float
        The epoch folding statistics corresponding to a probability
        epsilon * 100 % that the signal has been produced by noise

    Other Parameters
    ----------------
    ntrial : int
        The number of trials executed to find this profile
    """
    if ntrial > 1:
        simon("fold: The treatment of ntrial is very rough. Use with caution")
    from scipy import stats
    return stats.chi2.isf(epsilon, nbin - 1)


def z_n(phase, n=2, norm=1):
    '''Z^2_n statistics, a` la Buccheri+03, A&A, 128, 245, eq. 2.

    Parameters
    ----------
    phase : array of floats
        The phases of the events
    n : int, default 2
        The ``n`` in $Z^2_n$.

    Returns
    -------
    z2_n : float
        The Z^2_n statistics of the events.

    Other Parameters
    ----------------
    norm : float or array of floats
        A normalization factor that gets multiplied as a weight.
    '''
    nbin = len(phase)
    if nbin == 0:
        return 0
    phase = phase * 2 * np.pi
    return 2 / nbin * \
        np.sum([np.sum(np.cos(k * phase) * norm) ** 2 +
                np.sum(np.sin(k * phase) * norm) ** 2
                for k in range(1, n + 1)])


def z2_n_detection_level(n=2, epsilon=0.01, ntrial=1):
    """Return the detection level for the Z^2_n statistics.

    See Buccheri et al. (1983), Bendat and Piersol (1971).

    Parameters
    ----------
    n : int, default 2
        The ``n`` in $Z^2_n$
    epsilon : float, default 0.01
        The fractional probability that the signal has been produced by noise

    Returns
    -------
    detlev : float
        The epoch folding statistics corresponding to a probability
        epsilon * 100 % that the signal has been produced by noise

    Other Parameters
    ----------------
    ntrial : int
        The number of trials executed to find this profile
    """
    if ntrial > 1:
        simon("Z2_n: The treatment of ntrial is very rough. Use with caution")
    from scipy import stats
    return stats.chi2.isf(epsilon / ntrial, 2 * n)


def z2_2_probability(z2, n=2, ntrial=1):
    """Calculate the probability of a certain folded profile, due to noise.

    Parameters
    ----------
    z2 : float
        A Z^2_n statistics value
    n : int, default 2
        The ``n`` in $Z^2_n$

    Returns
    -------
    p : float
        The probability that the Z^2_n value has been produced by noise

    Other Parameters
    ----------------
    ntrial : int
        The number of trials executed to find this profile
    """
    if ntrial > 1:
        simon("Z2_n: The treatment of ntrial is very rough. Use with caution")
    from scipy import stats
    return stats.chi2.sf(stat, 2 * n) * ntrial


def fftfit_fun(profile, template, amplitude, phase):
    '''Function to be minimized for the FFTFIT method.

    From Taylor (1992).

    Parameters
    ----------
    profile : array
        The pulse profile
    template : array
        A pulse shape template, of the same length as profile.
    amplitude, phase : float
        The amplitude and phase of the template w.r.t the real profile.

    Returns
    -------
    fftfit_chisq : float
        The chi square-like statistics of FFTFIT
    '''

    prof_ft = np.fft.fft(profile)
    temp_ft = np.fft.fft(template)
    freq = np.fft.fftfreq(len(profile))
    good = freq > 0
    idx = np.arange(0, len(prof_ft), dtype=int)
    sigma = np.std(prof_ft[good])
    return np.sum(np.absolute(prof_ft -
                  temp_ft*amplitude*np.exp(-2*np.pi*1.0j*idx*phase))**2 /
                  sigma)


def _fft_fun_wrap(pars, data):
    '''Wrap parameters and input data up in order to be used with minimization
    algorithms.'''
    amplitude, phase = pars
    profile, template = data
    return fftfit_fun(profile, template, amplitude, phase)


def _triple_sinusoid_model(phase, a0, ph, a1, ph1, a2, ph2):
    twopi = np.pi * 2
    return a0 * np.cos(twopi * (phase - ph)) + \
        a1 * np.cos(twopi * (2 * (phase - ph - ph1))) + \
        a2 * np.cos(twopi * (3 * (phase - ph - ph2)))


def _pulse_template(phase, prof):
    pres, pcov = curve_fit(_triple_sinusoid_model, phase, prof)

    template = _triple_sinusoid_model(phase, *pres)

    fine_phase = np.arange(0, 1, 0.0001)
    fine_template = _triple_sinusoid_model(fine_phase, *pres)
    additional_phase = fine_phase[np.argmax(fine_template)]
    return template, additional_phase


def fftfit(prof, template=None, **fftfit_kwargs):
    """Align a template to a pulse profile.

    Parameters
    ----------
    phase : array
        The phases corresponding to each bin of the profile
    prof : array
        The pulse profile
    template : array, default None
        The template of the pulse used to perform the TOA calculation. If None,
        a simple sinusoid is used

    Returns
    -------
    mean_amp, std_amp : floats
        Mean and standard deviation of the amplitude
    mean_phase, std_phase : floats
        Mean and standard deviation of the phase

    Other Parameters
    ----------------
    fftfit_kwargs : arguments
        Additional arguments to be passed to error calculation
    """
    prof = prof - np.mean(prof)

    nbin = len(prof)

    ph = np.arange(0, 1, 1/nbin)
    if template is None:
        template = np.cos(2 * np.pi * ph)
    template = template - np.mean(template)

    p0 = [np.max(prof), np.float(np.argmax(prof) / nbin)]

    res = basinhopping(_fft_fun_wrap, p0,
                       minimizer_kwargs={'args': ([prof, template],),
                                         'bounds': [[0, None], [0, None]]},
                       niter=10000, niter_success=200)

    return fftfit_error(ph, prof, template, res.x, **fftfit_kwargs)


def fftfit_error(phase, prof, template, p0, **fftfit_kwargs):
    """Calculate the error on the fit parameters from FFTFIT.

    Parameters
    ----------
    phase : array
        The phases corresponding to each bin of the profile
    prof : array
        The pulse profile
    template : array
        The template of the pulse used to perform the TOA calculation
    p0 : list
        The initial parameters for the fit

    Returns
    -------
    mean_amp, std_amp : floats
        Mean and standard deviation of the amplitude
    mean_phase, std_phase : floats
        Mean and standard deviation of the phase

    Other parameters
    ----------------
    nstep : int, optional, default 100
        Number of steps for the bootstrap method
    """
    nstep = _default_value_if_no_key(fftfit_kwargs, "nstep", 100)

    approx, _ = _pulse_template(phase, prof)
    sigma = np.std(prof - approx)
    nbin = len(prof)

    ph_fit = np.zeros(nstep)
    amp_fit = np.zeros(nstep)
    # use bootstrap method to calculate errors
    for i in range(nstep):
        newprof = approx + np.random.normal(0, sigma, nbin)
        res = minimize(_fft_fun_wrap, p0, args=([newprof, template],),
                       method='L-BFGS-B',
                       bounds=[[0, None], [0, None]])
        amp_fit[i] = res.x[0]
        ph_fit[i] = res.x[1]

    p0 -= np.floor(p0)

    std_save = 1e32
    # avoid problems if phase around 0 or 1: shift, calculate std,
    # if less save new std
    for shift in np.arange(0, 0.8, 0.1):
        phs = ph_fit + shift
        phs -= np.floor(phs)
        std = np.std(phs)
        if std <= std_save:
            std_save = std
            mean_save = np.mean(phs) - shift

    # But still, never less than half a bin!
    std_save = np.max([std_save, np.diff(phase)[0] / 2])
    return np.mean(amp_fit), np.std(amp_fit), mean_save, std_save


def get_TOA(prof, period, tstart, template=None, additional_phase=0,
            **fftfit_kwargs):
    """Calculate the Time-Of-Arrival of a pulse.

    Parameters
    ----------
    prof : array
        The pulse profile
    template : array, default None
        The template of the pulse used to perform the TOA calculation, if any.
        Otherwise use the default of fftfit
    tstart : float
        The time at the start of the pulse profile

    Returns
    -------
    toa, toastd : floats
        Mean and standard deviation of the TOA

    Other parameters
    ----------------
    nstep : int, optional, default 100
        Number of steps for the bootstrap method
    """
    mean_amp, std_amp, phase_res, phase_res_err = \
        fftfit(prof, template=template, **fftfit_kwargs)
    phase_res = phase_res + additional_phase
    phase_res = phase_res - np.floor(phase_res)

    toa = tstart + phase_res * period
    toaerr = phase_res_err * period
    return toa, toaerr
