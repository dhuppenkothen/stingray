from __future__ import division, print_function, absolute_import

def lorenzian(x, p):
    """
    Generalized lorenzian function.

    Parameters
    ----------

    x: numpy.ndarray
        non-zero frequencies

    p: iterable
        p[0] = peak centeral frequency
        p[1] = FWHM of the peak (gamma)
        p[2] = peak value at x=x0
        p[3] = power coefficient [n]

    Returns
    -------
    model: numpy.ndarray
        generalized lorenzian psd model
    """

    assert p[3] > 0., "The power coefficient should be greater than zero."
    return p[2] * (p[1] / 2)**p[3] * 1./(abs(x - p[0])**p[3] + (p[1] / 2)**p[3]) 

def smoothbknpo(x, p):
    """
    Smooth broken power law function.

    Parameters
    ----------

    x: numpy.ndarray
        non-zero frequencies

    p: iterable
        p[0] = normalization frequency
        p[1] = power law index for f --> zero
        p[2] = power law index for f --> infinity
        p[3] = break frequency

    Returns
    -------
    model: numpy.ndarray
        generalized smooth broken power law psd model
    """

    return p[0] * x**(-p[1]) / (1. + (x / p[3])**2)**(-(p[1] - p[2]) / 2)
