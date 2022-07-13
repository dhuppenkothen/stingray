import bilby.gw.detector
import numpy as np


def red_noise(frequencies: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """ A simple power-law model.

    Parameters
    ----------
    frequencies:
        The frequencies.
    alpha:
        The power-law index.
    beta:
        The PSD at 1 Hz.

    Returns
    -------
    The PSD.
    """
    return beta * frequencies ** (-alpha)


def white_noise(frequencies: np.ndarray, sigma: float) -> np.ndarray:
    """ Returns a constant array.

    Parameters
    ----------
    frequencies:
        The frequencies.
    sigma:
        The white noise level.

    Returns
    -------
    The PSD.
    """
    return sigma * np.ones(len(frequencies))


def broken_power_law_noise(
        frequencies: np.ndarray, alpha_1: float, alpha_2: float, beta: float, delta: float, rho: float) -> np.ndarray:
    """ A broken power-law model.

    Parameters
    ----------
    frequencies:
        The frequencies.
    alpha_1:
        The lower-frequency power law index.
    alpha_2:
        The higher-frequency power law index.
    beta:
        The PSD at 1 Hz.
    delta:
        The frequency at which the power law is broken.
    rho:
        A smoothing parameter.

    Returns
    -------
    The PSD.
    """
    return beta * frequencies ** (-alpha_1) * (1 + (frequencies / delta) ** ((alpha_2 - alpha_1) / rho)) ** (-rho)


def lorentzian(frequencies: np.ndarray, amplitude: float, central_frequency: float, width: float) -> np.ndarray:
    """ A simple Lorentzian shape that can model a QPO.

    Parameters
    ----------
    frequencies:
        The frequencies.
    amplitude:
        The amplitude of the Lorentzian.
    central_frequency:
        The peak frequency.
    width:
        The width of the parameter (FWHM).

    Returns
    -------
    The PSD.
    """
    return amplitude * (width ** 2 / ((frequencies - central_frequency) ** 2 + width ** 2)) / np.pi / width


def periodogram_chi_square_test(
        frequencies: np.ndarray, powers: np.ndarray, psd: bilby.gw.detector.PowerSpectralDensity,
        degrees_of_freedom: int) -> float:
    """ A chi-square test for periodogram fits. Based on Nita et al. 2014.

    Parameters
    ----------
    frequencies:
        The frequencies.
    powers:
        The periodogram powers.
    psd:
        The psd as a bilby PSD object.
    degrees_of_freedom:
        The number of frequency bins minus the number of parameters of the model.

    Returns
    -------
    The PSD.
    """
    return \
        np.sum(np.nan_to_num((1 - powers/psd.power_spectral_density_interpolated(frequencies)), nan=0)**2) \
        / degrees_of_freedom
