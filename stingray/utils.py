from __future__ import (absolute_import, unicode_literals, division,
                        print_function)
import sys
import collections
import numbers

import warnings
import numpy as np

# If numba is installed, import jit. Otherwise, define an empty decorator with
# the same name.

HAS_NUMBA = False
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    warnings.warn("Numba not installed. Faking it")

    class jit(object):

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, func):
            def wrapped_f(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapped_f


def _root_squared_mean(array):
    return np.sqrt(np.sum(array**2)) / len(array)


def simon(message, **kwargs):
    """The Statistical Interpretation MONitor.

    A warning system designed to always remind the user that Simon
    is watching him/her.

    Parameters
    ----------
    message : string
        The message that is thrown

    kwargs : dict
        The rest of the arguments that are passed to warnings.warn
    """

    warnings.warn("SIMON says: {0}".format(message), **kwargs)


def rebin_data(x, y, dx_new, yerr=None, method='sum', dx=None):

    """Rebin some data to an arbitrary new data resolution. Either sum
    the data points in the new bins or average them.

    Parameters
    ----------
    x: iterable
        The dependent variable with some resolution dx_old = x[1]-x[0]

    y: iterable
        The independent variable to be binned

    dx_new: float
        The new resolution of the dependent variable x

    Other parameters
    ----------------
    yerr: iterable, optional
        The uncertainties of y, to be propagated during binning.

    method: {"sum" | "average" | "mean"}, optional, default "sum"
        The method to be used in binning. Either sum the samples y in
        each new bin of x, or take the arithmetic mean.

    dx: float
        The old resolution (otherwise, calculated from median diff)


    Returns
    -------
    xbin: numpy.ndarray
        The midpoints of the new bins in x

    ybin: numpy.ndarray
        The binned quantity y

    ybin_err: numpy.ndarray
        The uncertainties of the binned values of y.

    step_size: float
        The size of the binning step
    """

    y = np.asarray(y)
    yerr = np.asarray(assign_value_if_none(yerr, np.zeros_like(y)))

    dx_old = assign_value_if_none(dx, np.median(np.diff(x)))

    if dx_new < dx_old:
        raise ValueError("New frequency resolution must be larger than "
                         "old frequency resolution.")

    step_size = dx_new / dx_old

    output = []
    outputerr = []
    for i in np.arange(0, y.shape[0], step_size):
        total = 0
        totalerr = 0

        int_i = int(i)
        prev_frac = int_i + 1 - i
        prev_bin = int_i
        total += prev_frac * y[prev_bin]
        totalerr += prev_frac * (yerr[prev_bin]**2)

        if i + step_size < len(x):
            # Fractional part of next bin:
            next_frac = i + step_size - int(i + step_size)
            next_bin = int(i + step_size)
            total += next_frac * y[next_bin]
            totalerr += next_frac * (yerr[next_bin]**2)

        total += sum(y[int(i+1):int(i+step_size)])
        totalerr += sum(yerr[int(i+1):int(step_size)]**2)
        output.append(total)
        outputerr.append(np.sqrt(totalerr))

    output = np.asarray(output)
    outputerr = np.asarray(outputerr)

    if method in ['mean', 'avg', 'average', 'arithmetic mean']:
        ybin = output / np.float(step_size)
        ybinerr = outputerr / np.sqrt(np.float(step_size))

    elif method == "sum":
        ybin = output
        ybinerr = outputerr

    else:
        raise ValueError("Method for summing or averaging not recognized. "
                         "Please enter either 'sum' or 'mean'.")

    tseg = x[-1] - x[0] + dx_old

    if (tseg / dx_new % 1) > 0:
        ybin = ybin[:-1]
        ybinerr = ybinerr[:-1]

    new_x0 = (x[0] - (0.5*dx_old)) + (0.5*dx_new)
    xbin = np.arange(ybin.shape[0]) * dx_new + new_x0

    return xbin, ybin, ybinerr, step_size


def rebin_data_log(x, y, f, y_err=None, dx=None):
    """Logarithmic rebin of the periodogram.

    The new frequency depends on the previous frequency modified by a factor f:

    dnu_j = dnu_{j-1}*(1+f)

    Parameters
    ----------
    x: iterable
        The dependent variable with some resolution dx_old = x[1]-x[0]

    y: iterable
        The independent variable to be binned

    f: float
        The factor of increase of each bin wrt the previous one.

    Other Parameters
    ----------------
    yerr: iterable, optional
        The uncertainties of y, to be propagated during binning.

    method: {"sum" | "average" | "mean"}, optional, default "sum"
        The method to be used in binning. Either sum the samples y in
        each new bin of x, or take the arithmetic mean.

    dx: float, optional
        The binning step of the initial xs

    Returns
    -------
    xbin: numpy.ndarray
        The midpoints of the new bins in x

    ybin: numpy.ndarray
        The binned quantity y

    ybin_err: numpy.ndarray
        The uncertainties of the binned values of y.

    step_size: float
        The size of the binning step
    """
    import scipy
    dx_init = assign_value_if_none(dx, np.median(np.diff(x)))
    y = np.asarray(y)
    y_err = np.asarray(assign_value_if_none(y_err, np.zeros_like(y)))

    minx = x[1] * 0.5  # frequency to start from
    maxx = x[-1]  # maximum frequency to end
    binx = [minx, minx + dx_init]  # first
    dx = x[1]  # the frequency resolution of the first bin

    # until we reach the maximum frequency, increase the width of each
    # frequency bin by f
    while binx[-1] <= maxx:
        binx.append(binx[-1] + dx*(1.0+f))
        dx = binx[-1] - binx[-2]

    # compute the mean of the ys that fall into each new frequency bin.
    # we cast to np.double due to scipy's bad handling of longdoubles
    biny, bin_edges, binno = scipy.stats.binned_statistic(
        x.astype(np.double), y.astype(np.double),
        statistic="mean", bins=binx)

    biny_err, bin_edges, binno = scipy.stats.binned_statistic(
        x.astype(np.double), y_err.astype(np.double),
        statistic=_root_squared_mean, bins=binx)

    # compute the number of powers in each frequency bin
    nsamples = np.array([len(binno[np.where(binno == i)[0]])
                         for i in range(np.max(binno))])

    return binx, biny, biny_err, nsamples


def assign_value_if_none(value, default):
    return default if value is None else value


def look_for_array_in_array(array1, array2):
    return next((i for i in array1 if i in array2), None)


def is_string(s):  # pragma : no cover
    """Portable function to answer this question."""

    PY2 = sys.version_info[0] == 2
    if PY2:
        return isinstance(s, basestring)  # NOQA
    else:
        return isinstance(s, str)  # NOQA


def is_iterable(stuff):
    """Test if stuff is an iterable."""

    return isinstance(stuff, collections.Iterable)


def order_list_of_arrays(data, order):
    if hasattr(data, 'items'):
        data = dict([(key, value[order])
                     for key, value in data.items()])
    elif is_iterable(data):
        data = [i[order] for i in data]
    else:
        data = None
    return data


def optimal_bin_time(fftlen, tbin):
    """Vary slightly the bin time to have a power of two number of bins.

    Given an FFT length and a proposed bin time, return a bin time
    slightly shorter than the original, that will produce a power-of-two number
    of FFT bins.
    """

    return fftlen / (2 ** np.ceil(np.log2(fftlen / tbin)))


def contiguous_regions(condition):
    """Find contiguous True regions of the boolean array "condition".

    Return a 2D array where the first column is the start index of the region
    and the second column is the end index.

    Parameters
    ----------
    condition : boolean array

    Returns
    -------
    idx : [[i0_0, i0_1], [i1_0, i1_1], ...]
        A list of integer couples, with the start and end of each True blocks
        in the original array

    Notes
    -----
    From : http://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-
    fulfilling-condition-in-a-numpy-array
    """

    # NOQA
    # Find the indices of changes in "condition"
    diff = np.logical_xor(condition[1:], condition[:-1])
    idx, = diff.nonzero()
    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1
    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]
    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]
    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


def is_int(obj):
    return isinstance(obj, (numbers.Integral, np.integer))


def get_random_state(random_state = None):
    if not random_state:
        random_state = np.random.mtrand._rand
    else:
        if is_int(random_state):
            random_state = np.random.RandomState(random_state)
        elif not isinstance(random_state, np.random.RandomState):
            raise ValueError("{value} can't be used to generate a numpy.random.RandomState".format(
                value = random_state
            ))

    return random_state


def baseline_als(y, lam, p, niter=10):
    """Baseline Correction with Asymmetric Least Squares Smoothing.

    Modifications to the routine from Eilers & Boelens 2005
    https://www.researchgate.net/publication/228961729_Technical_Report_Baseline_Correction_with_Asymmetric_Least_Squares_Smoothing
    The Python translation is partly from
    http://stackoverflow.com/questions/29156532/python-baseline-correction-library
    
    Parameters
    ----------
    y : array of floats
        the "light curve". It assumes equal spacing.
    lam : float
        "smoothness" parameter. Larger values make the baseline stiffer
        Typically 1e2 < lam < 1e9
    p : float
        "asymmetry" parameter. Smaller values make the baseline more 
        "horizontal". Typically 0.001 < p < 0.1, but not necessary.
    """
    from scipy import sparse
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = sparse.linalg.spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


def excess_variance(lc, normalization='fvar'):
    """Calculate the excess variance.

    Vaughan+03

    Parameters
    ----------
    lc : a :class:`Lightcurve` object
    normalization : str
        if 'fvar', return normalized square-root excess variance. If 'none',
        return the unnormalized variance

    Returns
    -------
    var_xs : float
    var_xs_err : float
    """
    lc_mean_var = np.mean(lc.counts_err ** 2)
    lc_actual_var = np.var(lc.counts)
    var_xs = lc_actual_var - lc_mean_var
    mean_lc = np.mean(lc.counts)
    mean_ctvar = mean_lc ** 2

    fvar = np.sqrt(var_xs / mean_ctvar)

    N = len(lc.counts)
    var_xs_err_A = np.sqrt(2 / N) * lc_mean_var / mean_lc ** 2
    var_xs_err_B = np.sqrt(mean_lc ** 2 / N) * 2 * fvar / mean_lc
    var_xs_err = np.sqrt(var_xs_err_A ** 2 + var_xs_err_B ** 2)

    fvar_err = var_xs_err / (2 * fvar)

    if normalization == 'fvar':
        return fvar, fvar_err
    elif normalization == 'none' or normalization is None:
        return var_xs, var_xs_err

