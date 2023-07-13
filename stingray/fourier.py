import copy
import warnings
from collections.abc import Iterable

import numpy as np
from astropy.table import Table

from .gti import (
    generate_indices_of_segment_boundaries_binned,
    generate_indices_of_segment_boundaries_unbinned,
)
from .utils import histogram, show_progress, sum_if_not_none_or_initialize, fft, fftfreq


def positive_fft_bins(n_bin, include_zero=False):
    """
    Give the range of positive frequencies of a complex FFT.

    This assumes we are using Numpy's FFT, or something compatible
    with it, like ``pyfftw.interfaces.numpy_fft``, where the positive
    frequencies come before the negative ones, the Nyquist frequency is
    included in the negative frequencies but only in even number of bins,
    and so on.
    This is mostly to avoid using the ``freq > 0`` mask, which is
    memory-hungry and inefficient with large arrays. We use instead a
    slice object, giving the range of bins of the positive frequencies.

    See https://numpy.org/doc/stable/reference/routines.fft.html#implementation-details

    Parameters
    ----------
    n_bin : int
        The number of bins in the FFT, including all frequencies

    Other Parameters
    ----------------
    include_zero : bool, default False
        Include the zero frequency in the output slice

    Returns
    -------
    positive_bins : `slice`
        Slice object encoding the positive frequency bins. See examples.

    Examples
    --------
    Let us calculate the positive frequencies using the usual mask
    >>> freq = np.fft.fftfreq(10)
    >>> good = freq > 0

    This works well, but it is highly inefficient in large arrays.
    This function will instead return a `slice object`, which will work
    as an equivalent mask for the positive bins. Below, a few tests that
    this works as expected.
    >>> goodbins = positive_fft_bins(10)
    >>> np.allclose(freq[good], freq[goodbins])
    True
    >>> freq = np.fft.fftfreq(11)
    >>> good = freq > 0
    >>> goodbins = positive_fft_bins(11)
    >>> np.allclose(freq[good], freq[goodbins])
    True
    >>> freq = np.fft.fftfreq(10)
    >>> good = freq >= 0
    >>> goodbins = positive_fft_bins(10, include_zero=True)
    >>> np.allclose(freq[good], freq[goodbins])
    True
    >>> freq = np.fft.fftfreq(11)
    >>> good = freq >= 0
    >>> goodbins = positive_fft_bins(11, include_zero=True)
    >>> np.allclose(freq[good], freq[goodbins])
    True
    """
    # The zeroth bin is 0 Hz. We usually don't include it, but
    # if the user wants it, we do.
    minbin = 1
    if include_zero:
        minbin = 0

    if n_bin % 2 == 0:
        return slice(minbin, n_bin // 2)

    return slice(minbin, (n_bin + 1) // 2)


def poisson_level(norm="frac", meanrate=None, n_ph=None, backrate=0):
    r"""
    Poisson (white)-noise level in a periodogram of pure counting noise.

    For Leahy normalization, this is:
    .. math::
        P = 2

    For the fractional r.m.s. normalization, this is
    .. math::
        P = \frac{2}{\mu}
    where :math:`\mu` is the average count rate

    For the absolute r.m.s. normalization, this is
    .. math::
        P = 2 \mu

    Finally, for the unnormalized periodogram, this is
    .. math::
        P = N_{ph}

    Parameters
    ----------
    norm : str, default "frac"
        Normalization of the periodogram. One of ["abs", "frac", "leahy",
        "none"].

    Other Parameters
    ----------------
    meanrate : float, default None
        Mean count rate in counts/s. Needed for r.m.s. norms ("abs" and
        "frac").
    n_ph : float, default None
        Total number of counts in the light curve. Needed if ``norm=="none"``.
    backrate : float, default 0
        Background count rate in counts/s. Optional for fractional r.m.s. norm.

    Raises
    ------
    ValueError
        If the inputs are incompatible with the required normalization.

    Returns
    -------
    power_noise : float
        The Poisson noise level in the wanted normalization.

    Examples
    --------
    >>> poisson_level(norm="leahy")
    2.0
    >>> poisson_level(norm="abs", meanrate=10.)
    20.0
    >>> poisson_level(norm="frac", meanrate=10.)
    0.2
    >>> poisson_level(norm="none", n_ph=10)
    10.0
    >>> poisson_level(norm="asdfwrqfasdh3r", meanrate=10.)
    Traceback (most recent call last):
    ...
    ValueError: Unknown value for norm: asdfwrqfasdh3r...
    >>> poisson_level(norm="none", meanrate=10)
    Traceback (most recent call last):
    ...
    ValueError: Bad input parameters for norm none...
    >>> poisson_level(norm="abs", n_ph=10)
    Traceback (most recent call last):
    ...
    ValueError: Bad input parameters for norm abs...
    """
    # Various ways the parameters are wrong.
    # We want the noise in rms norm, but don't specify the mean rate.
    bad_input = norm.lower() in ["abs", "frac"] and meanrate is None
    # We want the noise in unnormalized powers, without giving n_ph.
    bad_input = bad_input or (norm.lower() == "none" and n_ph is None)

    if bad_input:
        raise ValueError(
            f"Bad input parameters for norm {norm}: n_ph={n_ph}, " f"meanrate={meanrate}"
        )

    if norm == "abs":
        return 2.0 * meanrate
    if norm == "frac":
        return 2.0 / (meanrate - backrate) ** 2 * meanrate
    if norm == "leahy":
        return 2.0
    if norm == "none":
        return float(n_ph)

    raise ValueError(f"Unknown value for norm: {norm}")


def normalize_frac(unnorm_power, dt, n_bin, mean_flux, background_flux=0):
    r"""
    Fractional rms normalization.

    ..math::
        P = \frac{P_{Leahy}}{\mu} = \frac{2T}{N_{ph}^2}P_{unnorm}

    where :math:`\mu` is the mean count rate, :math:`T` is the length of
    the observation, and :math:`N_{ph}` the number of photons.
    Alternative formulas found in the literature substitute :math:`T=N\,dt`,
    :math:`\mu=N_{ph}/T`, which give equivalent results.

    If the background can be estimated, one can calculate the source rms
    normalized periodogram as
    ..math::
        P = P_{Leahy} * \frac{\mu}{(\mu - \beta)^2}

    or
    ..math::
        P = \frac{2T}{(N_{ph} - \beta T)^2}P_{unnorm}

    where :math:`\beta` is the background count rate.

    This is also called the Belloni or Miyamoto normalization.
    In this normalization, the periodogram is in units of
    :math:`(rms/mean)^2 Hz^{-1}`, and the squared root of the
    integrated periodogram will give the fractional rms in the
    required frequency range.

    Belloni & Hasinger (1990) A&A 230, 103

    Miyamoto et al. (1991), ApJ 383, 784

    Parameters
    ----------
    unnorm_power : `np.array` of `float` or `complex`
        The unnormalized (cross-)spectral powers
    dt : float
        The sampling time
    n_bin : int
        The number of bins in the light curve
    mean_flux : float
        The mean of the light curve used to calculate the periodogram.
        If the light curve is in counts, it gives the mean counts per
        bin.

    Other parameters
    ----------------
    background_flux : float, default 0
        The background flux, in the same units as `mean_flux`.

    Returns
    -------
    power : `np.array` of the same kind and shape as `unnorm_power`
        The normalized powers.

    Examples
    --------
    >>> mean = var = 1000000
    >>> back = 100000
    >>> n_bin = 1000000
    >>> dt = 0.2
    >>> meanrate = mean / dt
    >>> backrate = back / dt
    >>> lc = np.random.poisson(mean, n_bin)
    >>> pds = np.abs(fft(lc))**2
    >>> pdsnorm = normalize_frac(pds, dt, lc.size, mean)
    >>> np.isclose(pdsnorm[1:n_bin//2].mean(), poisson_level(meanrate=meanrate,norm="frac"), rtol=0.01)
    True
    >>> pdsnorm = normalize_frac(pds, dt, lc.size, mean, background_flux=back)
    >>> np.isclose(pdsnorm[1:n_bin//2].mean(),
    ...            poisson_level(meanrate=meanrate,norm="frac",backrate=backrate), rtol=0.01)
    True
    """
    #     (mean * n_bin) / (mean /dt) = n_bin * dt
    #     It's Leahy / meanrate;
    #     n_ph = mean * n_bin
    #     meanrate = mean / dt
    #     norm = 2 / (n_ph * meanrate) = 2 * dt / (mean**2 * n_bin)

    if background_flux > 0:
        power = unnorm_power * 2.0 * dt / ((mean_flux - background_flux) ** 2 * n_bin)
    else:
        # Note: this corresponds to eq. 3 in Uttley+14
        power = unnorm_power * 2.0 * dt / (mean_flux**2 * n_bin)
    return power


def normalize_abs(unnorm_power, dt, n_bin):
    r"""
    Absolute rms normalization.

    .. math::
        P = P_{frac} * \mu^2

    where :math:`\mu` is the mean count rate, or equivalently
    .. math::
        P = \frac{2}{T}P_{unnorm}

    In this normalization, the periodogram is in units of
    :math:`rms^2 Hz^{-1}`, and the squared root of the
    integrated periodogram will give the absolute rms in the
    required frequency range.

    e.g. Uttley & McHardy, MNRAS 323, L26

    Parameters
    ----------
    unnorm_power : `np.array` of `float` or `complex`
        The unnormalized (cross-)spectral powers
    dt : float
        The sampling time
    n_bin : int
        The number of bins in the light curve

    Returns
    -------
    power : `np.array` of the same kind and shape as `unnorm_power`
        The normalized powers.

    Examples
    --------
    >>> mean = var = 100000
    >>> n_bin = 1000000
    >>> dt = 0.2
    >>> meanrate = mean / dt
    >>> lc = np.random.poisson(mean, n_bin)
    >>> pds = np.abs(fft(lc))**2
    >>> pdsnorm = normalize_abs(pds, dt, lc.size)
    >>> np.isclose(pdsnorm[1:n_bin//2].mean(), poisson_level(norm="abs", meanrate=meanrate), rtol=0.01)
    True
    """
    #     It's frac * meanrate**2; Leahy / meanrate * meanrate**2
    #     n_ph = mean * n_bin
    #     meanrate = mean / dt
    #     norm = 2 / (n_ph * meanrate) * meanrate**2 = 2 * dt / (mean**2 * n_bin) * mean**2 / dt**2

    return unnorm_power * 2.0 / n_bin / dt


def normalize_leahy_from_variance(unnorm_power, variance, n_bin):
    r"""
    Leahy+83 normalization, from the variance of the lc.

    .. math::
        P = \frac{P_{unnorm}}{N <\delta{x}^2>}

    In this normalization, the periodogram of a single light curve
    is distributed according to a chi squared distribution with two
    degrees of freedom.

    In this version, the normalization is obtained by the variance
    of the light curve bins, instead of the more usual version with the
    number of photons. This allows to obtain this normalization also
    in the case of non-Poisson distributed data.

    Parameters
    ----------
    unnorm_power : `np.array` of `float` or `complex`
        The unnormalized (cross-)spectral powers
    variance : float
        The mean variance of the light curve bins
    n_bin : int
        The number of bins in the light curve

    Returns
    -------
    power : `np.array` of the same kind and shape as `unnorm_power`
        The normalized powers.

    Examples
    --------
    >>> mean = var = 100000.
    >>> n_bin = 1000000
    >>> lc = np.random.poisson(mean, n_bin).astype(float)
    >>> pds = np.abs(fft(lc))**2
    >>> pdsnorm = normalize_leahy_from_variance(pds, var, lc.size)
    >>> np.isclose(pdsnorm[0], 2 * np.sum(lc), rtol=0.01)
    True
    >>> np.isclose(pdsnorm[1:n_bin//2].mean(), poisson_level(norm="leahy"), rtol=0.01)
    True

    If the variance is zero, it will fail:
    >>> pdsnorm = normalize_leahy_from_variance(pds, 0., lc.size)
    Traceback (most recent call last):
    ...
    ValueError: The variance used to normalize the ...
    """
    if variance == 0.0:
        raise ValueError("The variance used to normalize the periodogram is 0.")
    return unnorm_power * 2.0 / (variance * n_bin)


def normalize_leahy_poisson(unnorm_power, n_ph):
    """
    Leahy+83 normalization.

    .. math::
        P = \frac{2}{N_{ph}} P_{unnorm}

    In this normalization, the periodogram of a single light curve
    is distributed according to a chi squared distribution with two
    degrees of freedom.

    Leahy et al. 1983, ApJ 266, 160

    Parameters
    ----------
    unnorm_power : `np.array` of `float` or `complex`
        The unnormalized (cross-)spectral powers
    variance : float
        The mean variance of the light curve bins
    n_bin : int
        The number of bins in the light curve

    Returns
    -------
    power : `np.array` of the same kind and shape as `unnorm_power`
        The normalized powers.

    Examples
    --------
    >>> mean = var = 100000.
    >>> n_bin = 1000000
    >>> lc = np.random.poisson(mean, n_bin).astype(float)
    >>> pds = np.abs(fft(lc))**2
    >>> pdsnorm = normalize_leahy_poisson(pds, np.sum(lc))
    >>> np.isclose(pdsnorm[0], 2 * np.sum(lc), rtol=0.01)
    True
    >>> np.isclose(pdsnorm[1:n_bin//2].mean(), poisson_level(norm="leahy"), rtol=0.01)
    True
    """
    return unnorm_power * 2.0 / n_ph


def normalize_periodograms(
    unnorm_power,
    dt,
    n_bin,
    mean_flux=None,
    n_ph=None,
    variance=None,
    background_flux=0.0,
    norm="frac",
    power_type="all",
):
    """
    Wrapper around all the normalize_NORM methods.

    Normalize the cross-spectrum or the power-spectrum to Leahy, absolute rms^2,
    fractional rms^2 normalization, or not at all.

    Parameters
    ----------
    unnorm_power: numpy.ndarray
        The unnormalized cross spectrum.

    dt: float
        The sampling time of the light curve

    n_bin: int
        The number of bins in the light curve

    Other parameters
    ----------------
    mean_flux: float
        The mean of the light curve used to calculate the powers
        (If a cross spectrum, the geometrical mean of the light
        curves in the two channels). Only relevant for "frac" normalization

    n_ph: int or float
        The number of counts in the light curve used to calculate
        the unnormalized periodogram. Only relevant for Leahy normalization.

    variance: float
        The average variance of the measurements in light curve (if a cross
        spectrum,  the geometrical mean of the variances in the two channels).
        **NOT** the variance of the light curve, but of each flux measurement
        (square of light curve error bar)! Only relevant for the Leahy
        normalization of non-Poissonian data.

    norm : str
        One of ``leahy`` (Leahy+83), ``frac`` (fractional rms), ``abs``
        (absolute rms),

    power_type : str
        One of ``real`` (real part), ``all`` (all complex powers), ``abs``
        (absolute value)

    background_flux : float, default 0
        The background flux, in the same units as `mean_flux`.

    Returns
    -------
    power: numpy.nd.array
        The normalized co-spectrum (real part of the cross spectrum). For
        'none' normalization, imaginary part is returned as well.
    """

    if norm == "leahy" and variance is not None:
        pds = normalize_leahy_from_variance(unnorm_power, variance, n_bin)
    elif norm == "leahy":
        pds = normalize_leahy_poisson(unnorm_power, n_ph)
    elif norm == "frac":
        pds = normalize_frac(unnorm_power, dt, n_bin, mean_flux, background_flux=background_flux)
    elif norm == "abs":
        pds = normalize_abs(unnorm_power, dt, n_bin)
    elif norm == "none":
        pds = unnorm_power
    else:
        raise ValueError("Unknown value for the norm")

    if power_type == "all":
        return pds
    if power_type == "real":
        return pds.real
    if power_type in ["abs", "absolute"]:
        return np.abs(pds)
    raise ValueError("Unrecognized power type")


def unnormalize_periodograms(
    norm_power, dt, n_bin, n_ph, variance=None, background_flux=0.0, norm=None, power_type="all"
):
    """
    Wrapper around all the normalize_NORM methods.

    Unnormalize the power of the cross-spectrum to Leahy, absolute rms^2,
    fractional rms^2 normalization, or not at all.

    Parameters
    ----------
    norm_power: numpy.ndarray
        The normalized cross-spectrum or poisson noise

    dt: float
        The sampling time of the light curve

    n_bin: int
        The number of bins in the light curve

    Other parameters
    ----------------
    mean_flux: float
        The mean of the light curve used to calculate the powers
        (If a cross spectrum, the geometrical mean of the light
        curves in the two channels). Only relevant for "frac" normalization

    n_ph: int or float
        The number of counts in the light curve used to calculate
        the unnormalized periodogram. Only relevant for Leahy normalization.

    variance: float
        The average variance of the measurements in light curve (if a cross
        spectrum,  the geometrical mean of the variances in the two channels).
        **NOT** the variance of the light curve, but of each flux measurement
        (square of light curve error bar)! Only relevant for the Leahy
        normalization of non-Poissonian data.

    norm : str
        One of ``leahy`` (Leahy+83), ``frac`` (fractional rms), ``abs``
        (absolute rms),

    power_type : str
        One of ``real`` (real part), ``all`` (all complex powers), ``abs``
        (absolute value)

    background_flux : float, default 0
        The background flux, in the same units as `mean_flux`.

    Returns
    -------
    power: numpy.nd.array
        The normalized co-spectrum (real part of the cross spectrum). For
        'none' normalization, imaginary part is returned as well.
    """

    if norm == "leahy" and variance is not None:
        unnorm_power = norm_power * (variance * n_ph) / 2.0
    elif norm == "leahy":
        unnorm_power = norm_power * n_ph / 2.0
    elif norm == "frac":
        if background_flux > 0:
            unnorm_power = norm_power * ((n_ph / n_bin - background_flux) ** 2 * n_bin) / (2.0 * dt)
        else:
            unnorm_power = norm_power * (n_ph**2 / n_bin) / (2.0 * dt)
    elif norm == "abs":
        unnorm_power = norm_power * dt * n_bin / 2.0
    elif norm == "none":
        unnorm_power = norm_power
    else:
        raise ValueError("Unknown value for the norm")

    if power_type == "all":
        return unnorm_power
    if power_type == "real":
        return unnorm_power.real
    if power_type in ["abs", "absolute"]:
        return np.abs(unnorm_power)
    raise ValueError("Unrecognized power type")


def bias_term(power1, power2, power1_noise, power2_noise, n_ave, intrinsic_coherence=1.0):
    """
    Bias term needed to calculate the coherence.

    Introduced by
    Vaughan & Nowak 1997, ApJ 474, L43

    but implemented here according to the formulation in
    Ingram 2019, MNRAS 489, 392

    As recommended in the latter paper, returns 0 if n_ave > 500

    Parameters
    ----------
    power1 : float `np.array`
        sub-band periodogram
    power2 : float `np.array`
        reference-band periodogram
    power1_noise : float
        Poisson noise level of the sub-band periodogram
    power2_noise : float
        Poisson noise level of the reference-band periodogram
    n_ave : int
        number of intervals that have been averaged to obtain the input spectra

    Other Parameters
    ----------------
    intrinsic_coherence : float, default 1
        If known, the intrinsic coherence.

    Returns
    -------
    bias : float `np.array`, same shape as ``power1`` and ``power2``
        The bias term
    """
    if (isinstance(n_ave, Iterable) and np.all(n_ave > 500)) or (
        not isinstance(n_ave, Iterable) and n_ave > 500
    ):
        return 0.0 * power1
    bsq = power1 * power2 - intrinsic_coherence * (power1 - power1_noise) * (power2 - power2_noise)
    return bsq / n_ave


def raw_coherence(
    cross_power, power1, power2, power1_noise, power2_noise, n_ave, intrinsic_coherence=1
):
    """
    Raw coherence estimations from cross and power spectra.

    Vaughan & Nowak 1997, ApJ 474, L43

    Parameters
    ----------
    cross_power : complex `np.array`
        cross spectrum
    power1 : float `np.array`
        sub-band periodogram
    power2 : float `np.array`
        reference-band periodogram
    power1_noise : float
        Poisson noise level of the sub-band periodogram
    power2_noise : float
        Poisson noise level of the reference-band periodogram
    n_ave : int
        number of intervals that have been averaged to obtain the input spectra

    Other Parameters
    ----------------
    intrinsic_coherence : float, default 1
        If known, the intrinsic coherence.

    Returns
    -------
    coherence : float `np.array`
        The raw coherence values at all frequencies.
    """
    bsq = bias_term(
        power1, power2, power1_noise, power2_noise, n_ave, intrinsic_coherence=intrinsic_coherence
    )
    num = (cross_power * np.conj(cross_power)).real - bsq
    if isinstance(num, Iterable):
        num[num < 0] = (cross_power * np.conj(cross_power)).real[num < 0]
    elif num < 0:
        warnings.warn("Negative numerator in raw_coherence calculation. Setting bias term to 0")
        num = (cross_power * np.conj(cross_power)).real
    den = power1 * power2
    return num / den


def _estimate_intrinsic_coherence_single(
    cross_power, power1, power2, power1_noise, power2_noise, n_ave
):
    """
    Estimate intrinsic coherence.

    Use the iterative procedure from sec. 5 of

    Ingram 2019, MNRAS 489, 392

    Parameters
    ----------
    cross_power : complex
        cross spectrum
    power1 : float
        sub-band power
    power2 : float
        reference-band power
    power1_noise : float
        Poisson noise level of the sub-band periodogram
    power2_noise : float
        Poisson noise level of the reference-band periodogram
    n_ave : int
        number of intervals that have been averaged to obtain the input spectra

    Returns
    -------
    coherence : float `np.array`
        The estimated intrinsic coherence, at all frequencies.
    """
    new_coherence = 1
    old_coherence = 0
    count = 0
    while not np.isclose(new_coherence, old_coherence, atol=0.01) and count < 40:
        old_coherence = new_coherence
        bsq = bias_term(
            power1, power2, power1_noise, power2_noise, n_ave, intrinsic_coherence=new_coherence
        )
        den = (power1 - power1_noise) * (power2 - power2_noise)
        num = (cross_power * np.conj(cross_power)).real - bsq
        if num < 0:
            num = (cross_power * np.conj(cross_power)).real
        new_coherence = num / den
        count += 1
    return new_coherence


# This is the vectorized version of the function above.
estimate_intrinsic_coherence_vec = np.vectorize(_estimate_intrinsic_coherence_single)


def estimate_intrinsic_coherence(cross_power, power1, power2, power1_noise, power2_noise, n_ave):
    """
    Estimate intrinsic coherence

    Use the iterative procedure from sec. 5 of

    Ingram 2019, MNRAS 489, 392

    Parameters
    ----------
    cross_power : complex `np.array`
        cross spectrum
    power1 : float `np.array`
        sub-band periodogram
    power2 : float `np.array`
        reference-band periodogram
    power1_noise : float
        Poisson noise level of the sub-band periodogram
    power2_noise : float
        Poisson noise level of the reference-band periodogram
    n_ave : int
        number of intervals that have been averaged to obtain the input spectra

    Returns
    -------
    coherence : float `np.array`
        The estimated intrinsic coherence, at all frequencies.
    """
    new_coherence = estimate_intrinsic_coherence_vec(
        cross_power, power1, power2, power1_noise, power2_noise, n_ave
    )
    return new_coherence


def rms_calculation(
    unnorm_powers,
    min_freq,
    max_freq,
    nphots,
    T,
    M_freqs,
    K_freqs,
    freq_bins,
    poisson_noise_unnrom,
    deadtime=0.0,
):
    """
    Compute the fractional rms amplitude in the given power or cross spectrum

    NOTE: all array quantities are already in the correct energy range

    Parameters
    ----------
    unnrom_powers: array of float
        unnormalised power or cross spectrum, the array has already been
        filtered for the given frequency range

    min_freq: float
        The lower frequency bound for the calculation (from the freq grid).

    max_freq: float
        The upper frequency bound for the calculation (from the freq grid).

    nphots: float
        Number of photons for the full power or cross spectrum

    T: float
        Time length of the light curve

    M_freq: scalar or array of float
        If scalar, it is the number of segments in the AveragedCrossspectrum
        If array, it is the number of segments times the rebinning sample
        in the given frequency range.

    K_freq: scalar or array of float
        If scalar, the power or cross spectrum is not rebinned (K_freq = 1)
        If array,  the power or cross spectrum is rebinned and it is the
        rebinned sample in the given frequency range.

    freq_bins: integer
        if the cross or power spectrum is rebinned freq_bins = 1,
        if it NOT rebinned freq_bins is the number of frequency bins
        in the given frequency range.

    poisson_noise_unnrom : float
        This is the Poisson noise level unnormalised.

    Other parameters
    ----------------
    deadtime: float
        Deadtime of the instrument

    Returns
    -------
    rms: float
        The fractional rms amplitude contained between ``min_freq`` and
        ``max_freq``.

    rms_err: float
        The error on the fractional rms amplitude.

    """
    rms_squared = (
        np.sum((unnorm_powers - poisson_noise_unnrom) * 1 / T * K_freqs) * 2 * T / nphots**2
    )

    if rms_squared < 0.0:
        rms_err = np.sqrt(
            np.var((unnorm_powers - poisson_noise_unnrom) * 1 / T * K_freqs) * 2 * T / nphots**2
        )
        return 0.0, rms_err
    rms = np.sqrt(rms_squared)

    rms_noise_squared = (
        poisson_noise_unnrom * (max_freq - min_freq) * 2 * T / nphots**2
    )  # rms of the noise
    rms_err_squared = (2 * rms_squared * rms_noise_squared + rms_noise_squared**2) / (
        2 * np.sum(M_freqs) * freq_bins * rms_squared
    )
    rms_err = np.sqrt(rms_err_squared)

    return rms, rms_err


def error_on_averaged_cross_spectrum(
    cross_power, seg_power, ref_power, n_ave, seg_power_noise, ref_power_noise, common_ref=False
):
    """
    Error on cross spectral quantities, From Ingram 2019.

    Note: this is only valid for a very large number of averaged powers.
    Beware if n_ave < 50 or so.

    Parameters
    ----------
    cross_power : complex `np.array`
        cross spectrum
    seg_power : float `np.array`
        sub-band periodogram
    ref_power : float `np.array`
        reference-band periodogram
    seg_power_noise : float
        Poisson noise level of the sub-band periodogram
    ref_power_noise : float
        Poisson noise level of the reference-band periodogram
    n_ave : int
        number of intervals that have been averaged to obtain the input spectra

    Other Parameters
    ----------------
    common_ref : bool, default False
        Are data in the sub-band also included in the reference band?

    Returns
    -------
    dRe : float `np.array`
        Error on the real part of the cross spectrum
    dIm : float `np.array`
        Error on the imaginary part of the cross spectrum
    dphi : float `np.array`
        Error on the angle (or phase lag)
    dG : float `np.array`
        Error on the modulus of the cross spectrum

    """
    if (not isinstance(n_ave, Iterable) and n_ave < 30) or (
        isinstance(n_ave, Iterable) and np.any(n_ave) < 30
    ):
        warnings.warn(
            "n_ave is below 30. Please note that the error bars "
            "on the quantities derived from the cross spectrum "
            "are only reliable for a large number of averaged "
            "powers."
        )
    two_n_ave = 2 * n_ave
    if common_ref:
        Gsq = (cross_power * np.conj(cross_power)).real
        bsq = bias_term(seg_power, ref_power, seg_power_noise, ref_power_noise, n_ave)
        frac = (Gsq - bsq) / (ref_power - ref_power_noise)
        power_over_2n = ref_power / two_n_ave

        # Eq. 18
        dRe = dIm = dG = np.sqrt(power_over_2n * (seg_power - frac))
        # Eq. 19
        dphi = np.sqrt(
            power_over_2n * (seg_power / (Gsq - bsq) - 1 / (ref_power - ref_power_noise))
        )

    else:
        PrPs = ref_power * seg_power
        dRe = np.sqrt((PrPs + cross_power.real**2 - cross_power.imag**2) / two_n_ave)
        dIm = np.sqrt((PrPs - cross_power.real**2 + cross_power.imag**2) / two_n_ave)
        gsq = raw_coherence(
            cross_power, seg_power, ref_power, seg_power_noise, ref_power_noise, n_ave
        )
        dphi = np.sqrt((1 - gsq) / (2 * gsq * n_ave))
        dG = np.sqrt(PrPs / n_ave)

    return dRe, dIm, dphi, dG


def cross_to_covariance(cross_power, ref_power, ref_power_noise, delta_nu):
    """
    Convert a cross spectrum into a covariance spectrum.

    Covariance:
    Wilkinson & Uttley 2009, MNRAS, 397, 666

    Complex covariance:
    Mastroserio et al. 2018, MNRAS, 475, 4027

    Parameters
    ----------
    cross_power : complex `np.array`
        cross spectrum
    ref_power : float `np.array`
        reference-band periodogram
    ref_power_noise : float
        Poisson noise level of the reference-band periodogram
    delta_nu : float or `np.array`
        spectral resolution. Can be a float, or an array if the spectral
        resolution is not constant throughout the periodograms

    Returns
    -------
    covariance: complex `np.array`
        The cross spectrum, normalized as a covariance.

    """
    return cross_power * np.sqrt(delta_nu / (ref_power - ref_power_noise))


def _which_segment_idx_fun(binned=False, dt=None):
    """
    Select which segment index function from ``gti.py`` to use.

    If ``binned`` is ``False``, call the unbinned function.

    If ``binned`` is not ``True``, call the binned function.

    Note that in the binned function ``dt`` is an optional parameter.
    We pass it if the user specifies it.
    """
    # Make function interface equal (fluxes gets ignored)
    if not binned:
        fun = generate_indices_of_segment_boundaries_unbinned
    else:
        # Define a new function, so that we can pass the correct dt as an
        # argument.
        def fun(*args, **kwargs):
            return generate_indices_of_segment_boundaries_binned(*args, dt=dt, **kwargs)

    return fun


def get_average_ctrate(times, gti, segment_size, counts=None):
    """
    Calculate the average count rate during the observation.

    This function finds the same segments that the averaged periodogram
    functions (``avg_cs_from_iterables``, ``avg_pds_from_iterables`` etc) will
    use, and returns the mean count rate.
    If ``counts`` is ``None``, the input times are interpreted as events.
    Otherwise, the number of events is taken from ``counts``

    Parameters
    ----------
    times : float `np.array`
        Array of times
    gti : [[gti00, gti01], [gti10, gti11], ...]
        good time intervals
    segment_size : float
        length of segments

    Other parameters
    ----------------
    counts : float `np.array`, default None
        Array of counts per bin

    Returns
    -------
    ctrate : float
        The average count rate in the segments that are used for the analysis.

    Examples
    --------
    >>> times = np.sort(np.random.uniform(0, 1000, 1000))
    >>> gti = np.asarray([[0, 1000]])
    >>> counts, _ = np.histogram(times, bins=np.linspace(0, 1000, 11))
    >>> bin_times = np.arange(50, 1000, 100)
    >>> get_average_ctrate(bin_times, gti, 1000, counts=counts)
    1.0
    >>> get_average_ctrate(times, gti, 1000)
    1.0
    """
    n_ph = 0
    n_intvs = 0
    binned = counts is not None
    func = _which_segment_idx_fun(binned)

    for _, _, idx0, idx1 in func(times, gti, segment_size):
        if not binned:
            n_ph += idx1 - idx0
        else:
            n_ph += np.sum(counts[idx0:idx1])
        n_intvs += 1

    return n_ph / (n_intvs * segment_size)


def get_flux_iterable_from_segments(times, gti, segment_size, n_bin=None, fluxes=None, errors=None):
    """
    Get fluxes from different segments of the observation.

    If ``fluxes`` is ``None``, the input times are interpreted as events, and
    they are split into many binned series of length ``segment_size`` with
    ``n_bin`` bins.

    If ``fluxes`` is an array, the number of events corresponding to each time
    bin is taken from ``fluxes``

    Therefore, at least one of either ``n_bin`` and ``fluxes`` needs to be
    specified.

    Parameters
    ----------
    times : float `np.array`
        Array of times
    gti : [[gti00, gti01], [gti10, gti11], ...]
        good time intervals
    segment_size : float
        length of segments

    Other parameters
    ----------------
    n_bin : int, default None
        Number of bins to divide the ``segment_size`` in
    fluxes : float `np.array`, default None
        Array of fluxes.
    errors : float `np.array`, default None
        Array of error bars corresponding to the flux values above.

    Yields
    ------
    flux : `np.array`
        Array of fluxes
    err : `np.array`
        (optional) if ``errors`` is None, an array of errors in the segment

    """
    if fluxes is None and n_bin is None:
        raise ValueError(
            "At least one between fluxes (if light curve) and " "n_bin (if events) has to be set"
        )

    dt = None
    binned = fluxes is not None
    if binned:
        dt = np.median(np.diff(times[:100]))

    fun = _which_segment_idx_fun(binned, dt)

    for s, e, idx0, idx1 in fun(times, gti, segment_size):
        if idx1 - idx0 < 2:
            yield None
            continue
        if not binned:
            event_times = times[idx0:idx1]
            # astype here serves to avoid integer rounding issues in Windows,
            # where long is a 32-bit integer.
            cts = histogram(
                (event_times - s).astype(float), bins=n_bin, range=[0, segment_size]
            ).astype(float)
            cts = np.array(cts)
        else:
            cts = fluxes[idx0:idx1].astype(float)
            if errors is not None:
                cts = cts, errors[idx0:idx1]

        yield cts


def bispectrum_and_bicoherence_from_iterable(flux_iterable, dt):
    """Calculate the bispectrum and the bicoherence.

    Parameters
    ----------
    flux_iterable : `iterable` of `np.array`s or of tuples (`np.array`, `np.array`)
        Iterable providing either equal-length series of count measurements, or
        of tuples (fluxes, errors). They must all be of the same length.
    dt : float
        Time resolution of the light curves used to produce periodograms

    Returns
    -------
    freqs : `np.array`
        The frequency in each row or column
    bispectrum: 2-d `np.array`
        The unnormalized bispectrum
    bicoherence: 2-d `np.array`
        The bicoherence, calculated with the normalization from
        Kim & Powers (1979), IEEE Trans. Plasma Sci., PS-7, 120
    """

    B = B1 = B2 = None
    for flux in show_progress(flux_iterable):
        if flux is None:
            continue

        if isinstance(flux, tuple):
            flux, _ = flux

        if B is None:
            n_bin = flux.size
            fgt0 = positive_fft_bins(n_bin)
            Nft = fgt0.stop - fgt0.start
            B = np.zeros((Nft, Nft), dtype=complex)
            B1 = np.zeros((Nft, Nft))
            B2 = np.zeros((Nft, Nft))
            freqs = np.fft.fftfreq(n_bin, dt)[fgt0]

        ft = fft(flux)[fgt0]

        M = 0
        for i in range(B.shape[0]):
            b1 = ft * ft[i]
            b2 = np.roll(ft, -i)
            B[i, :] += b1 * b2.conj()
            B1[i, :] += (b1 * b1.conj()).real
            B2[i, :] += (b2 * b2.conj()).real
            M += 1

    BC = (B * B.conj()).real / (B1 * B2)

    return freqs, B / M, BC


def avg_pds_from_iterable(flux_iterable, dt, norm="frac", use_common_mean=True, silent=False):
    """
    Calculate the average periodogram from an iterable of light curves

    Parameters
    ----------
    flux_iterable : `iterable` of `np.array`s or of tuples (`np.array`, `np.array`)
        Iterable providing either equal-length series of count measurements,
        or of tuples (fluxes, errors). They must all be of the same length.
    dt : float
        Time resolution of the light curves used to produce periodograms

    Other Parameters
    ----------------
    norm : str, default "frac"
        The normalization of the periodogram. "abs" is absolute rms, "frac" is
        fractional rms, "leahy" is Leahy+83 normalization, and "none" is the
        unnormalized periodogram
    use_common_mean : bool, default True
        The mean of the light curve can be estimated in each interval, or on
        the full light curve. This gives different results (Alston+2013).
        Here we assume the mean is calculated on the full light curve, but
        the user can set ``use_common_mean`` to False to calculate it on a
        per-segment basis.
    silent : bool, default False
        Silence the progress bars

    Returns
    -------
    results : :class:`astropy.table.Table`
        Table containing the following columns:
        freq : `np.array`
            The periodogram frequencies
        power : `np.array`
            The normalized periodogram powers
        unnorm_power : `np.array`
            The unnormalized periodogram powers

        And a number of other useful diagnostics in the metadata, including
        all attributes needed to allocate Powerspectrum objects, such as all
        the input arguments of this function (``dt``, ``segment_size``), and,
        e.g.
        n : int
            the number of bins in the light curves used in each segment
        m : int
            the number of averaged periodograms
        mean : float
            the mean flux
    """
    local_show_progress = show_progress
    if silent:

        def local_show_progress(a):
            return a

    # Initialize stuff
    cross = unnorm_cross = None
    n_ave = 0

    sum_of_photons = 0
    common_variance = None
    for flux in local_show_progress(flux_iterable):
        if flux is None or np.all(flux == 0):
            continue

        # If the iterable returns the uncertainty, use it to calculate the
        # variance.
        variance = None
        if isinstance(flux, tuple):
            flux, err = flux
            variance = np.mean(err) ** 2

        # Calculate the FFT
        n_bin = flux.size
        ft = fft(flux)

        # This will only be used by the Leahy normalization, so only if
        # the input light curve is in units of counts/bin
        n_ph = flux.sum()
        unnorm_power = (ft * ft.conj()).real

        # Accumulate the sum of means and variances, to get the final mean and
        # variance the end
        sum_of_photons += n_ph

        if variance is not None:
            common_variance = sum_if_not_none_or_initialize(common_variance, variance)

        # In the first loop, define the frequency and the freq. interval > 0
        if cross is None:
            fgt0 = positive_fft_bins(n_bin)
            freq = fftfreq(n_bin, dt)[fgt0]

        # No need for the negative frequencies
        unnorm_power = unnorm_power[fgt0]

        # If the user wants to normalize using the mean of the total
        # lightcurve, normalize it here
        cs_seg = unnorm_power
        if not use_common_mean:
            mean = n_ph / n_bin

            cs_seg = normalize_periodograms(
                unnorm_power,
                dt,
                n_bin,
                mean,
                n_ph=n_ph,
                norm=norm,
                variance=variance,
            )

        # Accumulate the total sum cross spectrum
        cross = sum_if_not_none_or_initialize(cross, cs_seg)
        unnorm_cross = sum_if_not_none_or_initialize(unnorm_cross, unnorm_power)

        n_ave += 1

    # If there were no good intervals, return None
    if cross is None:
        return None

    # Calculate the mean number of photons per chunk
    n_ph = sum_of_photons / n_ave
    # Calculate the mean number of photons per bin
    common_mean = n_ph / n_bin

    if common_variance is not None:
        # Note: the variances we summed were means, not sums.
        # Hence M, not M*n_bin
        common_variance /= n_ave

    # Transform a sum into the average
    unnorm_cross = unnorm_cross / n_ave
    cross = cross / n_ave

    # Final normalization (If not done already!)
    if use_common_mean:
        cross = normalize_periodograms(
            unnorm_cross, dt, n_bin, common_mean, n_ph=n_ph, norm=norm, variance=common_variance
        )

    results = Table()
    results["freq"] = freq
    results["power"] = cross
    results["unnorm_power"] = unnorm_cross
    results.meta.update(
        {
            "n": n_bin,
            "m": n_ave,
            "dt": dt,
            "norm": norm,
            "df": 1 / (dt * n_bin),
            "nphots": n_ph,
            "mean": common_mean,
            "variance": common_variance,
            "segment_size": dt * n_bin,
        }
    )

    return results


def avg_cs_from_iterables_quick(flux_iterable1, flux_iterable2, dt, norm="frac"):
    """Like `avg_cs_from_iterables`, with default options that make it quick.

    Assumes that:

    * the flux iterables return counts/bin, no other units
    * the mean is calculated over the whole light curve, and normalization
      is done at the end
    * no auxiliary PDSs are returned
    * only positive frequencies are returned
    * the spectrum is complex, no real parts or absolutes
    * no progress bars

    Parameters
    ----------
    flux_iterable1 : `iterable` of `np.array`s or of tuples (`np.array`, `np.array`)
        Iterable providing either equal-length series of count measurements, or
        of tuples (fluxes, errors). They must all be of the same length.
    flux_iterable2 : `iterable` of `np.array`s or of tuples (`np.array`, `np.array`)
        Same as ``flux_iterable1``, for the reference channel
    dt : float
        Time resolution of the light curves used to produce periodograms

    Other Parameters
    ----------------
    norm : str, default "frac"
        The normalization of the periodogram. "abs" is absolute rms, "frac" is
        fractional rms, "leahy" is Leahy+83 normalization, and "none" is the
        unnormalized periodogram

    Returns
    -------
    results : :class:`astropy.table.Table`
        Table containing the following columns:
        freq : `np.array`
            The periodogram frequencies
        power : `np.array`
            The normalized periodogram powers
        unnorm_power : `np.array`
            The unnormalized periodogram powers

        And a number of other useful diagnostics in the metadata, including
        all attributes needed to allocate Crossspectrum objects, such as all
        the input arguments of this function (``dt``, ``segment_size``), and,
        e.g.
        n : int
            the number of bins in the light curves used in each segment
        m : int
            the number of averaged periodograms
        mean : float
            the mean flux (geometrical average of the mean fluxes in the two
            channels)

    """
    # Initialize stuff
    unnorm_cross = unnorm_pds1 = unnorm_pds2 = None
    n_ave = 0

    sum_of_photons1 = sum_of_photons2 = 0

    for flux1, flux2 in zip(flux_iterable1, flux_iterable2):
        if flux1 is None or flux2 is None or np.all(flux1 == 0) or np.all(flux2 == 0):
            continue

        n_bin = flux1.size

        # Calculate the sum of each light curve, to calculate the mean
        n_ph1 = flux1.sum()
        n_ph2 = flux2.sum()

        # At the first loop, we define the frequency array and the range of
        # positive frequency bins (after the first loop, cross will not be
        # None anymore)
        if unnorm_cross is None:
            freq = fftfreq(n_bin, dt)
            fgt0 = positive_fft_bins(n_bin)

        # Calculate the FFTs
        ft1 = fft(flux1)
        ft2 = fft(flux2)

        # Calculate the unnormalized cross spectrum
        unnorm_power = ft1.conj() * ft2

        # Accumulate the sum to calculate the total mean of the lc
        sum_of_photons1 += n_ph1
        sum_of_photons2 += n_ph2

        # Take only positive frequencies
        unnorm_power = unnorm_power[fgt0]

        # Initialize or accumulate final averaged spectrum
        unnorm_cross = sum_if_not_none_or_initialize(unnorm_cross, unnorm_power)

        n_ave += 1

    # If no valid intervals were found, return only `None`s
    if unnorm_cross is None:
        return None

    # Calculate the mean number of photons per chunk
    n_ph1 = sum_of_photons1 / n_ave
    n_ph2 = sum_of_photons2 / n_ave
    n_ph = np.sqrt(n_ph1 * n_ph2)
    # Calculate the mean number of photons per bin
    common_mean1 = n_ph1 / n_bin
    common_mean2 = n_ph2 / n_bin
    common_mean = n_ph / n_bin

    # Transform the sums into averages
    unnorm_cross /= n_ave

    # Finally, normalize the cross spectrum (only if not already done on an
    # interval-to-interval basis)
    cross = normalize_periodograms(
        unnorm_cross,
        dt,
        n_bin,
        common_mean,
        n_ph=n_ph,
        norm=norm,
        variance=None,
        power_type="all",
    )

    # No negative frequencies
    freq = freq[fgt0]

    results = Table()
    results["freq"] = freq
    results["power"] = cross
    results["unnorm_power"] = unnorm_cross
    results.meta.update(
        {
            "n": n_bin,
            "m": n_ave,
            "dt": dt,
            "norm": norm,
            "df": 1 / (dt * n_bin),
            "nphots": n_ph,
            "nphots1": n_ph1,
            "nphots2": n_ph2,
            "variance": None,
            "mean": common_mean,
            "mean1": common_mean1,
            "mean2": common_mean2,
            "power_type": "all",
            "fullspec": False,
            "segment_size": dt * n_bin,
        }
    )

    return results


def avg_cs_from_iterables(
    flux_iterable1,
    flux_iterable2,
    dt,
    norm="frac",
    use_common_mean=True,
    silent=False,
    fullspec=False,
    power_type="all",
    return_auxil=False,
):
    """Calculate the average cross spectrum from an iterable of light curves

    Parameters
    ----------
    flux_iterable1 : `iterable` of `np.array`s or of tuples (`np.array`, `np.array`)
        Iterable providing either equal-length series of count measurements, or
        of tuples (fluxes, errors). They must all be of the same length.
    flux_iterable2 : `iterable` of `np.array`s or of tuples (`np.array`, `np.array`)
        Same as ``flux_iterable1``, for the reference channel
    dt : float
        Time resolution of the light curves used to produce periodograms

    Other Parameters
    ----------------
    norm : str, default "frac"
        The normalization of the periodogram. "abs" is absolute rms, "frac" is
        fractional rms, "leahy" is Leahy+83 normalization, and "none" is the
        unnormalized periodogram
    use_common_mean : bool, default True
        The mean of the light curve can be estimated in each interval, or on
        the full light curve. This gives different results (Alston+2013).
        Here we assume the mean is calculated on the full light curve, but
        the user can set ``use_common_mean`` to False to calculate it on a
        per-segment basis.
    fullspec : bool, default False
        Return the full periodogram, including negative frequencies
    silent : bool, default False
        Silence the progress bars
    power_type : str, default 'all'
        If 'all', give complex powers. If 'abs', the absolute value; if 'real',
        the real part
    return_auxil : bool, default False
        Return the auxiliary unnormalized PDSs from the two separate channels

    Returns
    -------
    results : :class:`astropy.table.Table`
        Table containing the following columns:
        freq : `np.array`
            The frequencies.
        power : `np.array`
            The normalized cross spectral powers.
        unnorm_power : `np.array`
            The unnormalized cross spectral power.
        unnorm_pds1 : `np.array`
            The unnormalized auxiliary PDS from channel 1. Only returned if
            ``return_auxil`` is ``True``.
        unnorm_pds2 : `np.array`
            The unnormalized auxiliary PDS from channel 2. Only returned if
            ``return_auxil`` is ``True``.

        And a number of other useful diagnostics in the metadata, including
        all attributes needed to allocate Crossspectrum objects, such as all
        the input arguments of this function (``dt``, ``segment_size``), and,
        e.g.
        n : int
            The number of bins in the light curves used in each segment.
        m : int
            The number of averaged periodograms.
        mean : float
            The mean flux (geometrical average of the mean fluxes in the two
            channels).
    """

    local_show_progress = show_progress
    if silent:

        def local_show_progress(a):
            return a

    # Initialize stuff
    cross = unnorm_cross = unnorm_pds1 = unnorm_pds2 = pds1 = pds2 = None
    n_ave = 0

    sum_of_photons1 = sum_of_photons2 = 0
    common_variance1 = common_variance2 = common_variance = None

    for flux1, flux2 in local_show_progress(zip(flux_iterable1, flux_iterable2)):
        if flux1 is None or flux2 is None or np.all(flux1 == 0) or np.all(flux2 == 0):
            continue

        # Does the flux iterable return the uncertainty?
        # If so, define the variances
        variance1 = variance2 = None
        if isinstance(flux1, tuple):
            flux1, err1 = flux1
            variance1 = np.mean(err1) ** 2
        if isinstance(flux2, tuple):
            flux2, err2 = flux2
            variance2 = np.mean(err2) ** 2

        # Only use the variance if both flux iterables define it.
        if variance1 is None or variance2 is None:
            variance1 = variance2 = None
        else:
            common_variance1 = sum_if_not_none_or_initialize(common_variance1, variance1)
            common_variance2 = sum_if_not_none_or_initialize(common_variance2, variance2)

        n_bin = flux1.size

        # At the first loop, we define the frequency array and the range of
        # positive frequency bins (after the first loop, cross will not be
        # None anymore)
        if cross is None:
            freq = fftfreq(n_bin, dt)
            fgt0 = positive_fft_bins(n_bin)

        # Calculate the FFTs
        ft1 = fft(flux1)
        ft2 = fft(flux2)

        # Calculate the sum of each light curve, to calculate the mean
        n_ph1 = flux1.sum()
        n_ph2 = flux2.sum()
        n_ph = np.sqrt(n_ph1 * n_ph2)

        # Calculate the unnormalized cross spectrum
        unnorm_power = ft1.conj() * ft2
        unnorm_pd1 = unnorm_pd2 = 0

        # If requested, calculate the auxiliary PDSs
        if return_auxil:
            unnorm_pd1 = (ft1 * ft1.conj()).real
            unnorm_pd2 = (ft2 * ft2.conj()).real

        # Accumulate the sum to calculate the total mean of the lc
        sum_of_photons1 += n_ph1
        sum_of_photons2 += n_ph2

        # Take only positive frequencies unless the user wants the full
        # spectrum
        if not fullspec:
            unnorm_power = unnorm_power[fgt0]
            if return_auxil:
                unnorm_pd1 = unnorm_pd1[fgt0]
                unnorm_pd2 = unnorm_pd2[fgt0]

        cs_seg = unnorm_power
        p1_seg = unnorm_pd1
        p2_seg = unnorm_pd2

        # If normalization has to be done interval by interval, do it here.
        if not use_common_mean:
            mean1 = n_ph1 / n_bin
            mean2 = n_ph2 / n_bin
            mean = n_ph / n_bin
            variance = None

            if variance1 is not None:
                variance = np.sqrt(variance1 * variance2)

            cs_seg = normalize_periodograms(
                unnorm_power,
                dt,
                n_bin,
                mean,
                n_ph=n_ph,
                norm=norm,
                power_type=power_type,
                variance=variance,
            )
            p1_seg = normalize_periodograms(
                unnorm_pd1,
                dt,
                n_bin,
                mean1,
                n_ph=n_ph1,
                norm=norm,
                power_type=power_type,
                variance=variance1,
            )
            p2_seg = normalize_periodograms(
                unnorm_pd2,
                dt,
                n_bin,
                mean2,
                n_ph=n_ph2,
                norm=norm,
                power_type=power_type,
                variance=variance2,
            )

        # Initialize or accumulate final averaged spectra
        cross = sum_if_not_none_or_initialize(cross, cs_seg)
        unnorm_cross = sum_if_not_none_or_initialize(unnorm_cross, unnorm_power)

        if return_auxil:
            unnorm_pds1 = sum_if_not_none_or_initialize(unnorm_pds1, unnorm_pd1)
            unnorm_pds2 = sum_if_not_none_or_initialize(unnorm_pds2, unnorm_pd2)
            pds1 = sum_if_not_none_or_initialize(pds1, p1_seg)
            pds2 = sum_if_not_none_or_initialize(pds2, p2_seg)

        n_ave += 1

    # If no valid intervals were found, return only `None`s
    if cross is None:
        return None

    # Calculate the mean number of photons per chunk
    n_ph1 = sum_of_photons1 / n_ave
    n_ph2 = sum_of_photons2 / n_ave
    n_ph = np.sqrt(n_ph1 * n_ph2)

    # Calculate the common mean number of photons per bin
    common_mean1 = n_ph1 / n_bin
    common_mean2 = n_ph2 / n_bin
    common_mean = n_ph / n_bin

    if common_variance1 is not None:
        # Note: the variances we summed were means, not sums. Hence M, not M*N
        common_variance1 /= n_ave
        common_variance2 /= n_ave
        common_variance = np.sqrt(common_variance1 * common_variance2)

    # Transform the sums into averages
    cross /= n_ave
    unnorm_cross /= n_ave
    if return_auxil:
        unnorm_pds1 /= n_ave
        unnorm_pds2 /= n_ave

    # Finally, normalize the cross spectrum (only if not already done on an
    # interval-to-interval basis)
    if use_common_mean:
        cross = normalize_periodograms(
            unnorm_cross,
            dt,
            n_bin,
            common_mean,
            n_ph=n_ph,
            norm=norm,
            variance=common_variance,
            power_type=power_type,
        )
        if return_auxil:
            pds1 = normalize_periodograms(
                unnorm_pds1,
                dt,
                n_bin,
                common_mean1,
                n_ph=n_ph1,
                norm=norm,
                variance=common_variance1,
                power_type=power_type,
            )
            pds2 = normalize_periodograms(
                unnorm_pds2,
                dt,
                n_bin,
                common_mean2,
                n_ph=n_ph2,
                norm=norm,
                variance=common_variance2,
                power_type=power_type,
            )
    # If the user does not want negative frequencies, don't give them
    if not fullspec:
        freq = freq[fgt0]

    results = Table()
    results["freq"] = freq
    results["power"] = cross
    results["unnorm_power"] = unnorm_cross
    results.meta.update(
        {
            "n": n_bin,
            "m": n_ave,
            "dt": dt,
            "norm": norm,
            "df": 1 / (dt * n_bin),
            "segment_size": dt * n_bin,
            "nphots": n_ph,
            "nphots1": n_ph1,
            "nphots2": n_ph2,
            "countrate1": common_mean1 / dt,
            "countrate2": common_mean2 / dt,
            "mean": common_mean,
            "mean1": common_mean1,
            "mean2": common_mean2,
            "power_type": power_type,
            "fullspec": fullspec,
            "variance": common_variance,
            "variance1": common_variance1,
            "variance2": common_variance2,
        }
    )

    if return_auxil:
        results["pds1"] = pds1
        results["pds2"] = pds2
        results["unnorm_pds1"] = unnorm_pds1
        results["unnorm_pds2"] = unnorm_pds2

    return results


def avg_pds_from_events(
    times,
    gti,
    segment_size,
    dt,
    norm="frac",
    use_common_mean=True,
    silent=False,
    fluxes=None,
    errors=None,
):
    """
    Calculate the average periodogram from a list of event times or a light
    curve.

    If the input is a light curve, the time array needs to be uniformly sampled
    inside GTIs (it can have gaps outside), and the fluxes need to be passed
    through the ``fluxes`` array.
    Otherwise, times are interpeted as photon arrival times.

    Parameters
    ----------
    times : float `np.array`
        Array of times.
    gti : [[gti00, gti01], [gti10, gti11], ...]
        Good time intervals.
    segment_size : float
        Length of segments.
    dt : float
        Time resolution of the light curves used to produce periodograms.

    Other Parameters
    ----------------
    norm : str, default "frac"
        The normalization of the periodogram. "abs" is absolute rms, "frac" is
        fractional rms, "leahy" is Leahy+83 normalization, and "none" is the
        unnormalized periodogram
    use_common_mean : bool, default True
        The mean of the light curve can be estimated in each interval, or on
        the full light curve. This gives different results (Alston+2013).
        Here we assume the mean is calculated on the full light curve, but
        the user can set ``use_common_mean`` to False to calculate it on a
        per-segment basis.
    silent : bool, default False
        Silence the progress bars
    fluxes : float `np.array`, default None
        Array of counts per bin or fluxes
    errors : float `np.array`, default None
        Array of errors on the fluxes above

    Returns
    -------
    freq : `np.array`
        The periodogram frequencies
    power : `np.array`
        The normalized periodogram powers
    n_bin : int
        the number of bins in the light curves used in each segment
    n_ave : int
        the number of averaged periodograms
    mean : float
        the mean flux
    """
    if segment_size is None:
        segment_size = gti.max() - gti.min()
    n_bin = np.rint(segment_size / dt).astype(int)
    dt = segment_size / n_bin

    flux_iterable = get_flux_iterable_from_segments(
        times, gti, segment_size, n_bin, fluxes=fluxes, errors=errors
    )
    cross = avg_pds_from_iterable(
        flux_iterable, dt, norm=norm, use_common_mean=use_common_mean, silent=silent
    )
    if cross is not None:
        cross.meta["gti"] = gti
    return cross


def avg_cs_from_events(
    times1,
    times2,
    gti,
    segment_size,
    dt,
    norm="frac",
    use_common_mean=True,
    fullspec=False,
    silent=False,
    power_type="all",
    fluxes1=None,
    fluxes2=None,
    errors1=None,
    errors2=None,
    return_auxil=False,
):
    """
    Calculate the average cross spectrum from a list of event times or a light
    curve.

    If the input is a light curve, the time arrays need to be uniformly sampled
    inside GTIs (they can have gaps outside), and the fluxes need to be passed
    through the ``fluxes1`` and ``fluxes2`` arrays.
    Otherwise, times are interpeted as photon arrival times

    Parameters
    ----------
    times1 : float `np.array`
        Array of times in the sub-band
    times2 : float `np.array`
        Array of times in the reference band
    gti : [[gti00, gti01], [gti10, gti11], ...]
        common good time intervals
    segment_size : float
        length of segments
    dt : float
        Time resolution of the light curves used to produce periodograms

    Other Parameters
    ----------------
    norm : str, default "frac"
        The normalization of the periodogram. "abs" is absolute rms, "frac" is
        fractional rms, "leahy" is Leahy+83 normalization, and "none" is the
        unnormalized periodogram
    use_common_mean : bool, default True
        The mean of the light curve can be estimated in each interval, or on
        the full light curve. This gives different results (Alston+2013).
        Here we assume the mean is calculated on the full light curve, but
        the user can set ``use_common_mean`` to False to calculate it on a
        per-segment basis.
    fullspec : bool, default False
        Return the full periodogram, including negative frequencies
    silent : bool, default False
        Silence the progress bars
    power_type : str, default 'all'
        If 'all', give complex powers. If 'abs', the absolute value; if 'real',
        the real part
    fluxes1 : float `np.array`, default None
        Array of fluxes or counts per bin for channel 1
    fluxes2 : float `np.array`, default None
        Array of fluxes or counts per bin for channel 2
    errors1 : float `np.array`, default None
        Array of errors on the fluxes on channel 1
    errors2 : float `np.array`, default None
        Array of errors on the fluxes on channel 2

    Returns
    -------
    freq : `np.array`
        The periodogram frequencies
    pds : `np.array`
        The normalized periodogram powers
    n_bin : int
        the number of bins in the light curves used in each segment
    n_ave : int
        the number of averaged periodograms
    """
    if segment_size is None:
        segment_size = gti.max() - gti.min()
    n_bin = np.rint(segment_size / dt).astype(int)
    # adjust dt
    dt = segment_size / n_bin

    flux_iterable1 = get_flux_iterable_from_segments(
        times1, gti, segment_size, n_bin, fluxes=fluxes1, errors=errors1
    )
    flux_iterable2 = get_flux_iterable_from_segments(
        times2, gti, segment_size, n_bin, fluxes=fluxes2, errors=errors2
    )

    is_events = np.all([val is None for val in (fluxes1, fluxes2, errors1, errors2)])

    if (
        is_events
        and silent
        and use_common_mean
        and power_type == "all"
        and not fullspec
        and not return_auxil
    ):
        results = avg_cs_from_iterables_quick(flux_iterable1, flux_iterable2, dt, norm=norm)

    else:
        results = avg_cs_from_iterables(
            flux_iterable1,
            flux_iterable2,
            dt,
            norm=norm,
            use_common_mean=use_common_mean,
            silent=silent,
            fullspec=fullspec,
            power_type=power_type,
            return_auxil=return_auxil,
        )
    if results is not None:
        results.meta["gti"] = gti
    return results
