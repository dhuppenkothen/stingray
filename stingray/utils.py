from __future__ import (absolute_import, unicode_literals, division,
                        print_function)
import numpy as np
import warnings
import sys
# If numba is installed, import jit. Otherwise, define an empty decorator with
# the same name.

try:
    from numba import jit
except:
    def jit(fun):
        return fun


def simon(message, **kwargs):
    """
    The Statistical Interpretation MONitor.

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


def rebin_data(x, y, dx_new, method='sum'):

    """
    Rebin some data to an arbitrary new data resolution. Either sum
    the data points in the new bins or average them.

    Parameters
    ----------
    x: iterable
        The dependent variable with some resolution dx_old = x[1]-x[0]

    y: iterable
        The independent variable to be binned

    dx_new: float
        The new resolution of the dependent variable x

    method: {"sum" | "average" | "mean"}, optional, default "sum"
        The method to be used in binning. Either sum the samples y in
        each new bin of x, or take the arithmetic mean.


    Returns
    -------
    xbin: numpy.ndarray
        The midpoints of the new bins in x

    ybin: numpy.ndarray
        The binned quantity y
    """

    y = np.asarray(y)

    dx_old = x[1] - x[0]

    assert dx_new >= dx_old, "New frequency resolution must be larger than " \
                             "old frequency resolution."

    step_size = dx_new / dx_old

    output = []
    for i in np.arange(0, y.shape[0], step_size):
        total = 0

        prev_frac = int(i+1) - i
        prev_bin = int(i)
        total += prev_frac * y[prev_bin]

        if i + step_size < len(x):
            # Fractional part of next bin:
            next_frac = i+step_size - int(i+step_size)
            next_bin = int(i+step_size)
            total += next_frac * y[next_bin]

        total += sum(y[int(i+1):int(i+step_size)])
        output.append(total)

    output = np.asarray(output)

    if method in ['mean', 'avg', 'average', 'arithmetic mean']:
        ybin = output / np.float(step_size)

    elif method == "sum":
        ybin = output
    else:
        raise Exception("Method for summing or averaging not recognized. "
                        "Please enter either 'sum' or 'mean'.")

    tseg = x[-1] - x[0] + dx_old

    if (tseg/dx_new % 1) > 0:
        ybin = ybin[:-1]

    xbin = np.arange(ybin.shape[0])*dx_new + x[0] - dx_old + dx_new

    return xbin, ybin, step_size


def _assign_value_if_none(value, default):
    if value is None:
        return default
    else:
        return value


def _look_for_array_in_array(array1, array2):
    for a1 in array1:
        if a1 in array2:
            return a1


def is_string(s):  # pragma : no cover
    """Portable function to answer this question."""
    PY2 = sys.version_info[0] == 2
    if PY2:
        return isinstance(s, basestring)  # NOQA
    else:
        return isinstance(s, str)  # NOQA


def is_iterable(stuff):
    """Test if stuff is an iterable."""
    import collections

    return isinstance(stuff, collections.Iterable)


def _order_list_of_arrays(data, order):
    if hasattr(data, 'items'):
        data = dict([(i[0], i[1][order])
                     for i in data.items()])
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
    import numpy as np
    return fftlen / (2 ** np.ceil(np.log2(fftlen / tbin)))
