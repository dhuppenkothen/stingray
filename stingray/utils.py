
import sys
from collections.abc import Iterable
import numbers
from six import string_types

import warnings
import numpy as np
import scipy

# If numba is installed, import jit. Otherwise, define an empty decorator with
# the same name.

HAS_NUMBA = False
try:
    from numba import jit

    HAS_NUMBA = True
    from numba import njit, prange, vectorize, float32, float64, int32, int64
except ImportError:
    warnings.warn("Numba not installed. Faking it")

    class jit(object):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, func):
            def wrapped_f(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapped_f

    class njit(object):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, func):
            def wrapped_f(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapped_f

    class vectorize(object):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, func):
            wrapped_f = np.vectorize(func)

            return wrapped_f

    def generic(x, y=None):
        return None

    float32 = float64 = int32 = int64 = generic

    def prange(x):
        return range(x)

try:
    from tqdm import tqdm as show_progress
except ImportError:
    def show_progress(a):
        return a

try:
    from statsmodels.robust import mad as mad  # pylint: disable=unused-import
except ImportError:
    def mad(data, c=0.6745, axis=None):
        """
        Mean Absolute Deviation (MAD) along an axis.

        Straight from statsmodels's source code, adapted

        Parameters
        ----------
        data : iterable
            The data along which to calculate the MAD

        c : float, optional
            The normalization constant. Defined as
            ``scipy.stats.norm.ppf(3/4.)``, which is approximately ``.6745``.

        axis : int, optional, default ``0``
            Axis along which to calculate ``mad``. Default is ``0``, can also
            be ``None``
        """
        data = np.asarray(data)
        if axis is not None:
            center = np.apply_over_axes(np.median, data, axis)
        else:
            center = np.median(data)
        return np.median((np.fabs(data - center)) / c, axis=axis)

__all__ = ['simon', 'rebin_data', 'rebin_data_log', 'look_for_array_in_array',
           'is_string', 'is_iterable', 'order_list_of_arrays',
           'optimal_bin_time', 'contiguous_regions', 'is_int',
           'get_random_state', 'baseline_als', 'excess_variance',
           'create_window', 'poisson_symmetrical_errors', 'standard_error',
           'nearest_power_of_two', 'find_nearest']


def _root_squared_mean(array):
    array = np.asarray(array)
    return np.sqrt(np.sum(array ** 2)) / array.size


def simon(message, **kwargs):
    """The Statistical Interpretation MONitor.

    A warning system designed to always remind the user that Simon
    is watching him/her.

    Parameters
    ----------
    message : string
        The message that is thrown

    kwargs : dict
        The rest of the arguments that are passed to ``warnings.warn``
    """

    warnings.warn("SIMON says: {0}".format(message), **kwargs)


def rebin_data(x, y, dx_new, yerr=None, method='sum'):
    """Rebin some data to an arbitrary new data resolution. Either sum
    the data points in the new bins or average them.

    Parameters
    ----------
    x: iterable
        The dependent variable with some resolution, which can vary throughout 
        the time series.

    y: iterable
        The independent variable to be binned

    dx_new: float
        The new resolution of the dependent variable ``x``

    Other parameters
    ----------------
    yerr: iterable, optional
        The uncertainties of ``y``, to be propagated during binning.

    method: {``sum`` | ``average`` | ``mean``}, optional, default ``sum``
        The method to be used in binning. Either sum the samples ``y`` in
        each new bin of ``x``, or take the arithmetic mean.

    dx: float
        The old resolution (otherwise, calculated from median diff)

    Returns
    -------
    xbin: numpy.ndarray
        The midpoints of the new bins in ``x``

    ybin: numpy.ndarray
        The binned quantity ``y``

    ybin_err: numpy.ndarray
        The uncertainties of the binned values of ``y``.

    step_size: float
        The size of the binning step
    """

    y = np.asarray(y)
    yerr = np.asarray(apply_function_if_none(yerr, y, np.zeros_like))

    dx_old = np.diff(x)

    if np.any(dx_new < dx_old):
        raise ValueError("New frequency resolution must be larger than "
                         "old frequency resolution.")

    # left and right bin edges
    # assumes that the points given in `x` correspond to 
    # the left bin edges
    xedges = np.hstack([x, x[-1]+np.diff(x)[-1]])

    # new regularly binned resolution
    xbin = np.arange(xedges[0], xedges[-1]+dx_new, dx_new)

    output = np.zeros(xbin.shape[0] - 1)
    outputerr = np.zeros(xbin.shape[0] - 1)
    step_size = np.zeros(xbin.shape[0] - 1)

    for i in range(len(xbin)-1):

        xmin = xbin[i]
        xmax = xbin[i+1]
        min_ind = xedges.searchsorted(xmin)
        max_ind = xedges.searchsorted(xmax)
        
        output[i] = np.sum(y[min_ind:max_ind-1])
        outputerr[i] = np.sum(yerr[min_ind:max_ind-1])
        step_size[i] = len(y[min_ind:max_ind-1])

        prev_dx = xedges[min_ind] - xedges[min_ind-1]
        prev_frac = (xedges[min_ind] - xmin)/prev_dx
        output[i] += y[min_ind-1]*prev_frac
        outputerr[i] += yerr[min_ind-1]*prev_frac
        step_size[i] += prev_frac

        if not max_ind == len(xedges):
            dx_post = xedges[max_ind] - xedges[max_ind-1]
            post_frac = (xmax-xedges[max_ind-1])/dx_post
            output[i] += y[max_ind-1]*post_frac
            outputerr[i] += yerr[max_ind-1]*post_frac
            step_size[i] += post_frac
        
    if method in ['mean', 'avg', 'average', 'arithmetic mean']:
        ybin = output / step_size
        ybinerr = outputerr / step_size

    elif method == "sum":
        ybin = output
        ybinerr = outputerr

    else:
        raise ValueError("Method for summing or averaging not recognized. "
                         "Please enter either 'sum' or 'mean'.")

    tseg = x[-1] - x[0] + dx_old[-1]

    if (tseg / dx_new % 1) > 0:
        ybin = ybin[:-1]
        ybinerr = ybinerr[:-1]
        step_size = step_size[:-1]
        
    new_x0 = (x[0] - (0.5 * dx_old[0])) + (0.5 * dx_new)
    xbin = np.arange(ybin.shape[0]) * dx_new + new_x0

    return xbin, ybin, ybinerr, step_size



def rebin_data_log(x, y, f, y_err=None, dx=None):
    """Logarithmic re-bin of some data. Particularly useful for the power
    spectrum.

    The new dependent variable depends on the previous dependent variable
    modified by a factor f:

    .. math::

        d\\nu_j = d\\nu_{j-1} (1+f)

    Parameters
    ----------
    x: iterable
        The dependent variable with some resolution ``dx_old = x[1]-x[0]``

    y: iterable
        The independent variable to be binned

    f: float
        The factor of increase of each bin wrt the previous one.

    Other Parameters
    ----------------
    yerr: iterable, optional
        The uncertainties of ``y`` to be propagated during binning.

    method: {``sum`` | ``average`` | ``mean``}, optional, default ``sum``
        The method to be used in binning. Either sum the samples ``y`` in
        each new bin of ``x`` or take the arithmetic mean.

    dx: float, optional
        The binning step of the initial ``x``

    Returns
    -------
    xbin: numpy.ndarray
        The midpoints of the new bins in ``x``

    ybin: numpy.ndarray
        The binned quantity ``y``

    ybin_err: numpy.ndarray
        The uncertainties of the binned values of ``y``

    step_size: float
        The size of the binning step
    """

    dx_init = apply_function_if_none(dx, np.diff(x), np.median)
    x = np.asarray(x)
    y = np.asarray(y)
    y_err = np.asarray(apply_function_if_none(y_err, y, np.zeros_like))

    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must be of the same length!")
    if y.shape[0] != y_err.shape[0]:
        raise ValueError("y and y_err must be of the same length!")

    minx = x[0] * 0.5  # frequency to start from
    maxx = x[-1]  # maximum frequency to end
    binx = [minx, minx + dx_init]  # first
    dx = dx_init  # the frequency resolution of the first bin

    # until we reach the maximum frequency, increase the width of each
    # frequency bin by f
    while binx[-1] <= maxx:
        binx.append(binx[-1] + dx * (1.0 + f))
        dx = binx[-1] - binx[-2]

    binx = np.asarray(binx)

    real = y.real
    real_err = y_err.real
    # compute the mean of the ys that fall into each new frequency bin.
    # we cast to np.double due to scipy's bad handling of longdoubles
    biny, bin_edges, binno = scipy.stats.binned_statistic(
        x.astype(np.double), real.astype(np.double),
        statistic="mean", bins=binx)

    biny_err, bin_edges, binno = scipy.stats.binned_statistic(
        x.astype(np.double), real_err.astype(np.double),
        statistic=_root_squared_mean, bins=binx)

    if np.iscomplexobj(y):
        imag = y.imag
        biny_imag, bin_edges, binno = scipy.stats.binned_statistic(
            x.astype(np.double), imag.astype(np.double),
            statistic="mean", bins=binx)

        biny = biny + 1j * biny_imag

    if np.iscomplexobj(y_err):
        imag_err = y_err.imag

        biny_err_imag, bin_edges, binno = scipy.stats.binned_statistic(
            x.astype(np.double), imag_err.astype(np.double),
            statistic=_root_squared_mean, bins=binx)
        biny_err = biny_err + 1j * biny_err_imag

    # compute the number of powers in each frequency bin
    nsamples = np.array([len(binno[np.where(binno == i)[0]])
                         for i in range(1, np.max(binno) + 1, 1)])

    return binx, biny, biny_err, nsamples


def apply_function_if_none(variable, value, func):
    """
    Assign a function value to a variable if that variable has value ``None`` on input.

    Parameters
    ----------
    variable : object
        A variable with either some assigned value, or ``None``

    value : object
        A variable to go into the function

    func : function
        Function to apply to ``value``. Result is assigned to ``variable``

    Returns
    -------
        new_value : object
            The new value of ``variable``

    Examples
    --------
    >>> var = 4
    >>> value = np.zeros(10)
    >>> apply_function_if_none(var, value, np.mean)
    4
    >>> var = None
    >>> apply_function_if_none(var, value, lambda y: np.mean(y))
    0.0
    """
    if variable is None:
        return func(value)
    else:
        return variable

def assign_value_if_none(value, default):
    """
    Assign a value to a variable if that variable has value ``None`` on input.

    Parameters
    ----------
    value : object
        A variable with either some assigned value, or ``None``

    default : object The value to assign to the variable ``value`` if
    ``value is None`` returns ``True``

    Returns
    -------
        new_value : object
            The new value of ``value``

    """
    return default if value is None else value


def look_for_array_in_array(array1, array2):
    """
    Find a subset of values in an array.

    Parameters
    ----------
    array1 : iterable
        An array with values to be searched

    array2 : iterable
        A second array which potentially contains a subset of values
        also contained in ``array1``

    Returns ------- array3 : iterable An array with the subset of values
    contained in both ``array1`` and ``array2``

    """
    return next((i for i in array1 if i in array2), None)


def is_string(s):  # pragma : no cover
    """
    Portable function to answer whether a variable is a string.

    Parameters
    ----------
    s : object
        An object that is potentially a string

    Returns
    -------
    isstring : bool
        A boolean decision on whether ``s`` is a string or not
    """

    PY2 = sys.version_info[0] == 2
    if PY2:
        return isinstance(s, basestring)  # NOQA
    else:
        return isinstance(s, str)  # NOQA


def is_iterable(var):
    """Test if a variable  is an iterable.

    Parameters
    ----------
    var : object
        The variable to be tested for iterably-ness

    Returns
    -------
    is_iter : bool
        Returns ``True`` if ``var`` is an ``Iterable``, ``False`` otherwise
    """
    return isinstance(var, Iterable)


def order_list_of_arrays(data, order):
    """Sort an array according to the specified order.

    Parameters
    ----------
    data : iterable

    Returns
    -------
    data : list or dict
    """
    if hasattr(data, 'items'):
        data = dict([(key, value[order]) for key, value in data.items()])
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

    Parameters
    ----------
    fftlen : int
        Number of positive frequencies in a proposed Fourier spectrum

    tbin : float
        The proposed time resolution of a light curve

    Returns
    -------
    res : float
        A time resolution that will produce a Fourier spectrum with ``fftlen`` frequencies and
        a number of FFT bins that are a power of two
    """

    return fftlen / (2 ** np.ceil(np.log2(fftlen / tbin)))


def contiguous_regions(condition):
    """Find contiguous ``True`` regions of the boolean array ``condition``.

    Return a 2D array where the first column is the start index of the region
    and the second column is the end index, found on [so-contiguous]_.

    Parameters
    ----------
    condition : bool array

    Returns
    -------
    idx : ``[[i0_0, i0_1], [i1_0, i1_1], ...]``
        A list of integer couples, with the start and end of each ``True`` blocks
        in the original array

    Notes
    -----
    .. [so-contiguous] http://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array
    """
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
    """Test if object is an integer."""
    return isinstance(obj, (numbers.Integral, np.integer))


def get_random_state(random_state=None):
    """Return a Mersenne Twister pseudo-random number generator.

    Parameters
    ----------
    seed : integer or ``numpy.random.RandomState``, optional, default ``None``

    Returns
    -------
    random_state : mtrand.RandomState object
    """
    if not random_state:
        random_state = np.random.mtrand._rand
    else:
        if is_int(random_state):
            random_state = np.random.RandomState(random_state)
        elif not isinstance(random_state, np.random.RandomState):
            raise ValueError(
                "{value} can't be used to generate a numpy.random.RandomState".format(
                    value=random_state
                ))

    return random_state


def _offset(x, off):
    """An offset."""
    return off


def offset_fit(x, y, offset_start=0):
    """Fit a constant offset to the data.

    Parameters
    ----------
    x : array-like
    y : array-like
    offset_start : float
        Constant offset, initial value

    Returns
    -------
    offset : float
        Fitted offset
    """
    from scipy.optimize import curve_fit
    par, _ = curve_fit(_offset, x, y, [offset_start],
                       maxfev=6000)
    return par[0]


def _als(y, lam, p, niter=10):
    """Baseline Correction with Asymmetric Least Squares Smoothing.

    Modifications to the routine from Eilers & Boelens 2005 [eilers-2005]_.
    The Python translation is partly from [so-als]_.

    Parameters
    ----------
    y : array-like
        the data series corresponding to ``x``
    lam : float
        the lambda parameter of the ALS method. This control how much the
        baseline can adapt to local changes. A higher value corresponds to a
        stiffer baseline
    p : float
        the asymmetry parameter of the ALS method. This controls the overall
        slope tollerated for the baseline. A higher value correspond to a
        higher possible slope

    Other parameters
    ----------------
    niter : int
        The number of iterations to perform

    Returns
    -------
    z : array-like, same size as ``y``
        Fitted baseline.

    References
    ----------
    .. [eilers-2005] https://www.researchgate.net/publication/228961729_Technical_Report_Baseline_Correction_with_Asymmetric_Least_Squares_Smoothing
    .. [so-als] http://stackoverflow.com/questions/29156532/python-baseline-correction-library

    """
    from scipy import sparse
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = sparse.linalg.spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def baseline_als(x, y, lam=None, p=None, niter=10, return_baseline=False,
                 offset_correction=False):
    """Baseline Correction with Asymmetric Least Squares Smoothing.

    Parameters
    ----------
    x : array-like
        the sample time/number/position
    y : array-like
        the data series corresponding to ``x``
    lam : float
        the lambda parameter of the ALS method. This control how much the
        baseline can adapt to local changes. A higher value corresponds to a
        stiffer baseline
    p : float
        the asymmetry parameter of the ALS method. This controls the overall
        slope tolerated for the baseline. A higher value correspond to a
        higher possible slope

    Other Parameters
    ----------------
    niter : int
        The number of iterations to perform
    return_baseline : bool
        return the baseline?
    offset_correction : bool
        also correct for an offset to align with the running mean of the scan

    Returns
    -------
    y_subtracted : array-like, same size as ``y``
        The initial time series, subtracted from the trend
    baseline : array-like, same size as ``y``
        Fitted baseline. Only returned if return_baseline is ``True``

    Examples
    --------
    >>> x = np.arange(0, 10, 0.01)
    >>> y = np.zeros_like(x) + 10
    >>> ysub = baseline_als(x, y)
    >>> np.all(ysub < 0.001)
    True
    """

    if lam is None:
        lam = 1e11
    if p is None:
        p = 0.001

    z = _als(y, lam, p, niter=niter)

    ysub = y - z
    offset = 0
    if offset_correction:
        std = mad(ysub)
        good = np.abs(ysub) < 10 * std
        if len(x[good]) < 10:
            good = np.ones(len(x), dtype=bool)
            warnings.warn('Too few bins to perform baseline offset correction'
                          ' precisely. Beware of results')
        offset = offset_fit(x[good], ysub[good], 0)

    if return_baseline:
        return ysub - offset, z + offset
    else:
        return ysub - offset


def excess_variance(lc, normalization='fvar'):
    """Calculate the excess variance.

    Vaughan et al. 2003, MNRAS 345, 1271 give three measurements of source
    intrinsic variance: if a light curve has a total variance of :math:`S^2`,
    and each point has an error bar :math:`\sigma_{err}`, the *excess variance*
    is defined as

    .. math:: \sigma_{XS} = S^2 - \overline{\sigma_{err}}^2;

    the *normalized excess variance* is the excess variance divided by the
    square of the mean intensity:

    .. math:: \sigma_{NXS} = \dfrac{\sigma_{XS}}{\overline{x}^2};

    the *fractional mean square variability amplitude*, or
    :math:`F_{var}`, is finally defined as

    .. math:: F_{var} = \sqrt{\dfrac{\sigma_{XS}}{\overline{x}^2}}

    Parameters
    ----------
    lc : a :class:`Lightcurve` object
    normalization : str
        if ``fvar``, return the fractional mean square variability :math:`F_{var}`.
        If ``none``, return the unnormalized excess variance variance
        :math:`\sigma_{XS}`. If ``norm_xs``, return the normalized excess variance
        :math:`\sigma_{XS}`
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
    var_nxs = var_xs / mean_lc ** 2

    fvar = np.sqrt(var_xs / mean_ctvar)

    N = len(lc.counts)
    var_nxs_err_A = np.sqrt(2 / N) * lc_mean_var / mean_lc ** 2
    var_nxs_err_B = np.sqrt(lc_mean_var / N) * 2 * fvar / mean_lc
    var_nxs_err = np.sqrt(var_nxs_err_A ** 2 + var_nxs_err_B ** 2)

    fvar_err = var_nxs_err / (2 * fvar)

    if normalization == 'fvar':
        return fvar, fvar_err
    elif normalization == 'norm_xs':
        return var_nxs, var_nxs_err
    elif normalization == 'none' or normalization is None:
        return var_xs, var_nxs_err * mean_lc ** 2


def create_window(N, window_type='uniform'):
    """A method to create window functions commonly used in signal processing.

    Windows supported are:
    Hamming, Hanning, uniform (rectangular window), triangular window,
    blackmann window among others.

    Parameters
    ----------
    N : int
        Total number of data points in window. If negative, abs is taken.
    window_type : {``uniform``, ``parzen``, ``hamming``, ``hanning``, ``triangular``,\
                 ``welch``, ``blackmann``, ``flat-top``}, optional, default ``uniform``
        Type of window to create.

    Returns
    -------
    window: numpy.ndarray
        Window function of length ``N``.
    """

    if not isinstance(N, int):
        raise TypeError('N (window length) must be an integer')

    windows = ['uniform', 'parzen', 'hamming', 'hanning', 'triangular',
               'welch', 'blackmann', 'flat-top']

    if not isinstance(window_type, string_types):
        raise TypeError('type of window must be specified as string!')

    window_type = window_type.lower()
    if window_type not in windows:
        raise ValueError(
            "Wrong window type specified or window function is not available")

    # Return empty array as window if N = 0
    if N == 0:
        return np.array([])

    window = None
    N = abs(N)

    # Window samples index
    n = np.arange(N)

    # Constants
    N_minus_1 = N - 1
    N_by_2 = np.int((np.floor((N_minus_1) / 2)))

    # Create Windows
    if window_type == 'uniform':
        window = np.ones(N)

    if window_type == 'parzen':
        N_parzen = np.int(np.ceil((N + 1) / 2))
        N2_plus_1 = np.int(np.floor((N_parzen / 2))) + 1

        window = np.zeros(N_parzen)
        windlag0 = np.arange(0, N2_plus_1) / (N_parzen - 1)
        windlag1 = 1 - np.arange(N2_plus_1, N_parzen) / (N_parzen - 1)
        window[:N2_plus_1] = 1 - (1 - windlag0) * windlag0 * windlag0 * 6
        window[N2_plus_1:] = windlag1 * windlag1 * windlag1 * 2
        lagindex = np.arange(N_parzen - 1, 0, -1)
        window = np.concatenate((window[lagindex], window))
        window = window[:N]

    if window_type == 'hamming':
        window = 0.54 - 0.46 * np.cos((2 * np.pi * n) / N_minus_1)

    if window_type == 'hanning':
        window = 0.5 * (1 - np.cos(2 * np.pi * n / N_minus_1))

    if window_type == 'triangular':
        window = 1 - np.abs((n - (N_by_2)) / N)

    if window_type == 'welch':
        N_minus_1_by_2 = N_minus_1 / 2
        window = 1 - np.square((n - N_minus_1_by_2) / N_minus_1_by_2)

    if window_type == 'blackmann':
        a0 = 0.42659
        a1 = 0.49656
        a2 = 0.076849
        window = a0 - a1 * np.cos((2 * np.pi * n) / N_minus_1) + a2 * np.cos(
            (4 * np.pi * n) / N_minus_1)

    if window_type == 'flat-top':
        a0 = 1
        a1 = 1.93
        a2 = 1.29
        a3 = 0.388
        a4 = 0.028
        window = a0 - a1 * np.cos((2 * np.pi * n) / N_minus_1) + \
                 a2 * np.cos((4 * np.pi * n) / N_minus_1) - \
                 a3 * np.cos((6 * np.pi * n) / N_minus_1) + \
                 a4 * np.cos((8 * np.pi * n) / N_minus_1)

    return window


def poisson_symmetrical_errors(counts):
    """Optimized version of frequentist symmetrical errors.

    Uses a lookup table in order to limit the calls to poisson_conf_interval

    Parameters
    ----------
    counts : iterable
        An array of Poisson-distributed numbers

    Returns
    -------
    err : numpy.ndarray
        An array of uncertainties associated with the Poisson counts in
        ``counts``

    Examples
    --------
    >>> from astropy.stats import poisson_conf_interval
    >>> counts = np.random.randint(0, 1000, 100)
    >>> # ---- Do it without the lookup table ----
    >>> err_low, err_high = poisson_conf_interval(np.asarray(counts),
    ...                 interval='frequentist-confidence', sigma=1)
    >>> err_low -= np.asarray(counts)
    >>> err_high -= np.asarray(counts)
    >>> err = (np.absolute(err_low) + np.absolute(err_high))/2.0
    >>> # Do it with this function
    >>> err_thisfun = poisson_symmetrical_errors(counts)
    >>> # Test that results are always the same
    >>> assert np.all(err_thisfun == err)
    """
    from astropy.stats import poisson_conf_interval
    counts_int = np.asarray(counts, dtype=np.int64)
    count_values = np.nonzero(np.bincount(counts_int))[0]
    err_low, err_high = \
        poisson_conf_interval(count_values,
                              interval='frequentist-confidence', sigma=1)
    # calculate approximately symmetric uncertainties
    err_low -= np.asarray(count_values)
    err_high -= np.asarray(count_values)
    err = (np.absolute(err_low) + np.absolute(err_high)) / 2.0

    idxs = np.searchsorted(count_values, counts_int)
    return err[idxs]


def standard_error(xs, mean):
    """
    Return the standard error of the mean (SEM) of an array of arrays.

    Parameters
    ----------
    xs : 2-d float array
        List of data point arrays.

    mean : 1-d float array
        Average of the data points.

    Returns
    -------
    standard_error : 1-d float array
        Standard error of the mean (SEM).

    """

    n_seg = len(xs)
    xs_diff_sq = np.subtract(xs, mean) ** 2
    standard_deviation = np.sum(xs_diff_sq, axis=0) / (n_seg - 1)
    error = np.sqrt(standard_deviation / n_seg)
    return error


def nearest_power_of_two(x):
    """
    Return a number which is nearest to `x` and is the integral power of two.

    Parameters
    ----------
    x : int, float

    Returns
    -------
    x_nearest : int
        Number closest to `x` and is the integral power of two.

    """
    x = int(x)
    x_lower = 1 if x == 0 else 2 ** (x - 2).bit_length()
    x_upper = 1 if x == 0 else 2 ** (x - 1).bit_length()
    x_nearest = x_lower if (x - x_lower) < (x_upper - x) else x_upper
    return x_nearest


def find_nearest(array, value):
    """
    Return the array value that is closest to the input value (Abigail Stevens:
    Thanks StackOverflow!)

    Parameters
    ----------
    array : np.array of ints or floats
        1-D array of numbers to search through. Should already be sorted
        from low values to high values.

    value : int or float
        The value you want to find the closest to in the array.

    Returns
    -------
    array[idx] : int or float
        The array value that is closest to the input value.

    idx : int
        The index of the array of the closest value.

    """
    idx = np.searchsorted(array, value, side="left")
    if idx == len(array) or np.fabs(value - array[idx - 1]) < \
            np.fabs(value - array[idx]):
        return array[idx - 1], idx - 1
    else:
        return array[idx], idx
