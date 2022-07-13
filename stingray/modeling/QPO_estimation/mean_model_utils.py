import numpy as np


def polynomial(times: np.ndarray, a0: float, a1: float, a2: float, a3: float, a4: float) -> np.ndarray:
    """ A simple polynomial model.

    Parameters
    ----------
    times:
        The time coordinates
    a0, a1, a2, a3, a4:
        The parameters of the polynomial up to order 4.

    Returns
    -------
    The y values.
    """
    t = times.copy()
    t -= t[0]
    t -= t[-1] / 2
    return a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4


def skew_exponential(
        times: np.ndarray, log_amplitude: float, t_0: float, log_sigma_rise: float, log_sigma_fall: float)\
        -> np.ndarray:
    """ A skew exponential flare shape.

    Parameters
    ----------
    times:
        The time coordinates.
    log_amplitude:
        Natural log of the maximum of the flare.
    t_0:
        The location of the maximum.
    log_sigma_rise:
        The width parameter for the rising edge.
    log_sigma_fall:
        The width parameter for the falling edge.

    Returns
    -------
    The y values.
    """
    amplitude = np.exp(log_amplitude)
    sigma_rise = np.exp(log_sigma_rise)
    sigma_fall = np.exp(log_sigma_fall)

    before_burst_indices = np.where(times <= t_0)
    after_burst_indices = np.where(times > t_0)
    envelope = np.zeros(len(times))
    envelope[before_burst_indices] = amplitude * np.exp((times[before_burst_indices] - t_0) / sigma_rise)
    envelope[after_burst_indices] = amplitude * np.exp(-(times[after_burst_indices] - t_0) / sigma_fall)
    return envelope


def fred(times: np.ndarray, log_amplitude: float, log_psi: float, t_0: float, delta: float) -> np.ndarray:
    """ The fast-rise exponential-decay model.

    Parameters
    ----------
    times:
        The time coordinates.
    log_amplitude:
        Natural log of the maximum of the flare.
    log_psi:
        The psi shape parameter.
    t_0:
        The location of the maximum.
    delta:
        The delta shape parameter.

    Returns
    -------
    The y values.
    """
    amplitude = np.exp(log_amplitude)
    psi = np.exp(log_psi)

    frac = (times + delta) / t_0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return amplitude * np.exp(-psi * (frac + 1 / frac)) * np.exp(2 * psi)


def fred_extended(
        times: np.ndarray, log_amplitude: float, log_psi: float, t_0: float,
        delta: float, log_gamma: float, log_nu: float) -> np.ndarray:
    """ The extended FRED model.

    Parameters
    ----------
    times:
        The time coordinates.
    log_amplitude:
        Natural log of the maximum of the flare.
    log_psi:
        The log psi shape parameter.
    t_0:
        The location of the maximum.
    delta:
        The delta shape parameter.
    log_gamma:
        The log gamma shape parameter.
    log_nu:
        The log nu shape parameter.

    Returns
    -------
    The y values.
    """
    amplitude = np.exp(log_amplitude)
    nu = np.exp(log_nu)
    gamma = np.exp(log_gamma)
    psi = np.exp(log_psi)

    frac = (times + delta) / t_0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return amplitude * np.exp(-psi**gamma * frac**gamma - psi**nu / frac**nu) * np.exp(2 * psi)


def gaussian(times: np.ndarray, log_amplitude: float, t_0: float, log_sigma: float) -> np.ndarray:
    """ A simple Gaussian bell curve.

    Parameters
    ----------
    times:
        The time coordinates.
    log_amplitude:
        Natural log of the maximum of the flare.
    t_0:
        The location of the maximum.
    log_sigma:
        The width parameter.

    Returns
    -------
    The y values.
    """
    amplitude = np.exp(log_amplitude)
    sigma = np.exp(log_sigma)
    return amplitude * np.exp(-(times - t_0) ** 2 / (2 * sigma ** 2))


def skew_gaussian(
        times: np.ndarray, log_amplitude: float, t_0: float,
        log_sigma_rise: float, log_sigma_fall: float) -> np.ndarray:
    """ A skewed Gaussian model.

    Parameters
    ----------
    times:
        The time coordinates.
    log_amplitude:
        Natural log of the maximum of the flare.
    t_0:
        The location of the maximum.
    log_sigma_rise:
        The width parameter on the rising edge.
    log_sigma_fall:
        The width parameter on the falling edge.

    Returns
    -------
    The y values.
    """
    before_burst_indices = np.where(times <= t_0)
    after_burst_indices = np.where(times > t_0)
    envelope = np.zeros(len(times))
    envelope[before_burst_indices] = gaussian(times=times[before_burst_indices], log_amplitude=log_amplitude,
                                              t_0=t_0, log_sigma=log_sigma_rise)
    envelope[after_burst_indices] = gaussian(times=times[after_burst_indices], log_amplitude=log_amplitude,
                                             t_0=t_0, log_sigma=log_sigma_fall)
    return envelope


def log_normal(times: np.ndarray, log_amplitude: float, t_0: float, log_sigma: float) -> np.ndarray:
    """ A log normal distribution.

    Parameters
    ----------
    times:
        The time coordinates.
    log_amplitude:
        The log amplitude.
    t_0:
        The time offset.
    log_sigma:
        The width parameter.

    Returns
    -------
    The y values.
    """
    amplitude = np.exp(log_amplitude)
    sigma = np.exp(log_sigma)
    return amplitude / times * np.exp(-(np.log(times) - t_0) ** 2 / (2 * sigma ** 2))


def lorentzian(times: np.ndarray, log_amplitude: float, t_0: float, log_sigma: float) -> np.ndarray:
    amplitude = np.exp(log_amplitude)
    sigma = np.exp(log_sigma)
    return amplitude * (sigma / ((times - t_0) ** 2 + (sigma ** 2)))
