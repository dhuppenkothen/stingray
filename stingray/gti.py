from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import numpy as np
import logging

from astropy.io import fits
from .io import assign_value_if_none
from .utils import contiguous_regions
from stingray.exceptions import StingrayError

def load_gtis(fits_file, gtistring=None):
    """Load GTI from HDU EVENTS of file fits_file."""

    gtistring = assign_value_if_none(gtistring, 'GTI')
    logging.info("Loading GTIS from file %s" % fits_file)
    lchdulist = fits.open(fits_file, checksum=True)
    lchdulist.verify('warn')

    gtitable = lchdulist[gtistring].data
    gti_list = np.array([[a, b]
                         for a, b in zip(gtitable.field('START'),
                                         gtitable.field('STOP'))],
                        dtype=np.longdouble)
    lchdulist.close()
    return gti_list

def _get_gti_from_extension(lchdulist, accepted_gtistrings=['GTI']):
    hdunames = [h.name for h in lchdulist]
    gtiextn = [ix for ix, x in enumerate(hdunames)
               if x in accepted_gtistrings][0]
    gtiext = lchdulist[gtiextn]
    gtitable = gtiext.data

    colnames = [col.name for col in gtitable.columns.columns]
    # Default: NuSTAR: START, STOP. Otherwise, try RXTE: Start, Stop
    if 'START' in colnames:
        startstr, stopstr = 'START', 'STOP'
    else:
        startstr, stopstr = 'Start', 'Stop'

    gtistart = np.array(gtitable.field(startstr), dtype=np.longdouble)
    gtistop = np.array(gtitable.field(stopstr), dtype=np.longdouble)
    gti_list = np.array([[a, b]
                         for a, b in zip(gtistart,
                                         gtistop)],
                        dtype=np.longdouble)
    return gti_list

def check_gtis(gti):
    """Check if GTIs are well-behaved.

    Check that:
    1) the shape of the GTI array is correct;
    2) No start>end
    3) no overlaps.

    Raises
    ------
    TypeError
        If GTIs are of the wrong shape
    ValueError
        If GTIs have overlapping or displaced values
    """
    gti = np.asarray(gti)
    if len(gti) != gti.shape[0] or len(gti.shape) != 2 or \
                    len(gti) != gti.shape[0]:
        raise TypeError("Please check formatting of GTIs. They need to be"
                        " provided as [[gti00, gti01], [gti10, gti11], ...]")

    gti_start = gti[:, 0]
    gti_end = gti[:, 1]

    logging.debug('-- GTI: ' + repr(gti))
    # Check that GTIs are well-behaved
    if not np.all(gti_end >= gti_start):
        raise ValueError('This GTI end times must be larger than '
                         'GTI start times')

    # Check that there are no overlaps in GTIs
    if not np.all(gti_start[1:] >= gti_end[:-1]):
        raise ValueError('This GTI has overlaps')

    logging.debug('-- Correct')

    return

def create_gti_mask(time, gtis, safe_interval=0, min_length=0,
                    return_new_gtis=False, dt=None, epsilon=0.001):
    """Create GTI mask.

    Assumes that no overlaps are present between GTIs

    Parameters
    ----------
    time : float array
    gtis : [[g0_0, g0_1], [g1_0, g1_1], ...], float array-like

    Returns
    -------
    mask : boolean array
    new_gtis : Nx2 array

    Other parameters
    ----------------
    safe_interval : float or [float, float]
        A safe interval to exclude at both ends (if single float) or the start
        and the end (if pair of values) of GTIs.
    min_length : float
    return_new_gtis : bool
    dt : float
    epsilon : float
        fraction of dt that is tolerated at the borders of a GTI
    """
    import collections

    check_gtis(gtis)

    dt = assign_value_if_none(dt,
                              np.zeros_like(time) +
                              np.median(np.diff(time)) / 2)

    mask = np.zeros(len(time), dtype=bool)

    if not isinstance(safe_interval, collections.Iterable):
        safe_interval = [safe_interval, safe_interval]

    newgtis = np.zeros_like(gtis)
    # Whose GTIs, including safe intervals, are longer than min_length
    newgtimask = np.zeros(len(newgtis), dtype=np.bool)

    for ig, gti in enumerate(gtis):
        limmin, limmax = gti
        limmin += safe_interval[0]
        limmax -= safe_interval[1]
        if limmax - limmin >= min_length:
            newgtis[ig][:] = [limmin, limmax]
            cond1 = time - dt + epsilon*dt >= limmin
            cond2 = time + dt - epsilon*dt <= limmax
            good = np.logical_and(cond1, cond2)
            mask[good] = True
            newgtimask[ig] = True

    res = mask
    if return_new_gtis:
        res = [res, newgtis[newgtimask]]
    return res

def create_gti_from_condition(time, condition,
                              safe_interval=0, dt=None):
    """Create a GTI list from a time array and a boolean mask ("condition").

    Parameters
    ----------
    time : array-like
        Array containing times
    condition : array-like
        An array of bools, of the same length of time.
        A possible condition can be, e.g., the result of lc > 0.

    Returns
    -------
    gtis : [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        The newly created GTIs

    Other parameters
    ----------------
    safe_interval : float or [float, float]
        A safe interval to exclude at both ends (if single float) or the start
        and the end (if pair of values) of GTIs.
    dt : float
        The width (in sec) of each bin of the time array. Can be irregular.
    """
    import collections

    if len(time) != len(condition):
        raise StingrayError('The length of the condition and '
                            'time arrays must be the same.')

    idxs = contiguous_regions(condition)

    if not isinstance(safe_interval, collections.Iterable):
        safe_interval = [safe_interval, safe_interval]

    dt = assign_value_if_none(dt,
                               np.zeros_like(time) + (time[1] - time[0]) / 2)

    gtis = []
    for idx in idxs:
        logging.debug(idx)
        startidx = idx[0]
        stopidx = idx[1] - 1

        t0 = time[startidx] - dt[startidx] + safe_interval[0]
        t1 = time[stopidx] + dt[stopidx] - safe_interval[1]
        if t1 - t0 < 0:
            continue
        gtis.append([t0, t1])
    return np.array(gtis)


def cross_two_gtis(gti0, gti1):
    """Extract the common intervals from two GTI lists *EXACTLY*.

    Parameters
    ----------
    gti0 : [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
    gti1 : [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]

    Returns
    -------
    gtis : [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        The newly created GTIs

    See Also
    --------
    cross_gtis : From multiple GTI lists, extract common intervals *EXACTLY*

    """
    gti0 = np.asarray(gti0)
    gti1 = np.asarray(gti1)
    # Check GTIs
    check_gtis(gti0)
    check_gtis(gti1)

    gti0_start = gti0[:, 0]
    gti0_end = gti0[:, 1]
    gti1_start = gti1[:, 0]
    gti1_end = gti1[:, 1]

    # Create a list that references to the two start and end series
    gti_start = [gti0_start, gti1_start]
    gti_end = [gti0_end, gti1_end]

    # Concatenate the series, while keeping track of the correct origin of
    # each start and end time
    gti0_tag = np.array([0 for g in gti0_start], dtype=bool)
    gti1_tag = np.array([1 for g in gti1_start], dtype=bool)
    conc_start = np.concatenate((gti0_start, gti1_start))
    conc_end = np.concatenate((gti0_end, gti1_end))
    conc_tag = np.concatenate((gti0_tag, gti1_tag))

    # Put in time order
    order = np.argsort(conc_end)
    conc_start = conc_start[order]
    conc_end = conc_end[order]
    conc_tag = conc_tag[order]

    last_end = conc_start[0] - 1
    final_gti = []
    for ie, e in enumerate(conc_end):
        # Is this ending in series 0 or 1?
        this_series = conc_tag[ie]
        other_series = not this_series

        # Check that this closes intervals in both series.
        # 1. Check that there is an opening in both series 0 and 1 lower than e
        try:
            st_pos = \
                np.argmax(gti_start[this_series][gti_start[this_series] < e])
            so_pos = \
                np.argmax(gti_start[other_series][gti_start[other_series] < e])
            st = gti_start[this_series][st_pos]
            so = gti_start[other_series][so_pos]

            s = max([st, so])
        except:  # pragma: no cover
            continue

        # If this start is inside the last interval (It can happen for equal
        # GTI start times between the two series), then skip!
        if s <= last_end:
            continue
        # 2. Check that there is no closing before e in the "other series",
        # from intervals starting either after s, or starting and ending
        # between the last closed interval and this one
        cond1 = (gti_end[other_series] > s) * (gti_end[other_series] < e)
        cond2 = gti_end[other_series][so_pos] < s
        condition = np.any(np.logical_or(cond1, cond2))
        # Well, if none of the conditions at point 2 apply, then you can
        # create the new gti!
        if not condition:
            final_gti.append([s, e])
            last_end = e

    return np.array(final_gti)


def cross_gtis(gti_list):
    """From multiple GTI lists, extract the common intervals *EXACTLY*.

    Parameters
    ----------
    gti_list : array-like
        List of GTI arrays, each one in the usual format [[gti0_0, gti0_1],
        [gti1_0, gti1_1], ...]

    Returns
    -------
    gti0: 2-d float array
        [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        The newly created GTIs

    See Also
    --------
    cross_two_gtis : Extract the common intervals from two GTI lists *EXACTLY*
    """
    gti_list = np.asarray(gti_list)
    for g in gti_list:
        check_gtis(g)

    ninst = len(gti_list)
    if ninst == 1:
        return gti_list[0]

    gti0 = gti_list[0]

    for gti in gti_list[1:]:
        gti0 = cross_two_gtis(gti0, gti)

    return gti0


def get_btis(gtis, start_time=None, stop_time=None):
    """From GTIs, obtain bad time intervals.

    GTIs have to be well-behaved, in the sense that they have to pass
    `check_gtis`.
    """
    # Check GTIs
    if len(gtis) == 0:
        if not start_time or not stop_time:
            raise ValueError('Empty GTI and no valid start_time '
                             'and stop_time. BAD!')

        return np.asarray([[start_time, stop_time]])
    check_gtis(gtis)

    start_time = assign_value_if_none(start_time, gtis[0][0])
    stop_time = assign_value_if_none(stop_time, gtis[-1][1])

    if gtis[0][0] - start_time <= 0:
        btis = []
    else:
        btis = [[gtis[0][0] - start_time]]
    # Transform GTI list in
    flat_gtis = gtis.flatten()
    new_flat_btis = zip(flat_gtis[1:-2:2], flat_gtis[2:-1:2])
    btis.extend(new_flat_btis)

    if stop_time - gtis[-1][1] > 0:
        btis.extend([[gtis[0][0] - stop_time]])

    return np.asarray(btis)


def gti_len(gti):
    """Return the total good time from a list of GTIs."""
    return np.sum([g[1] - g[0] for g in gti])

def check_separate(gti0, gti1):
    """Check if two GTIs do not overlap.

    Parameters
    ----------
    gti0: 2-d float array
        [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]

    gti1: 2-d float array
        [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]

    Returns
    -------
    separate: boolean
        True if GTIs are mutually exclusive, False if not
    """

    gti0 = np.asarray(gti0)
    gti1 = np.asarray(gti1)
    if len(gti0) == 0 or len(gti1) == 0:
        return True

    # Check if independently GTIs are well behaved
    check_gtis(gti0)
    check_gtis(gti1)

    gti0_start = gti0[:, 0][0]
    gti0_end = gti0[:, 1][-1]
    gti1_start = gti1[:, 0][0]
    gti1_end = gti1[:, 1][-1]

    if (gti0_end <= gti1_start) or (gti1_end <= gti0_start):
        return True
    else:
        return False

def append_gtis(gti0, gti1):
    """Union of two non-overlapping GTIs.

    Parameters
    ----------
    gti0: 2-d float array
        [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]

    gti1: 2-d float array
        [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]

    Returns
    -------
    gti: 2-d float array
        The newly created GTI
    """

    gti0 = np.asarray(gti0)
    gti1 = np.asarray(gti1)

    # Check if independently GTIs are well behaved.
    check_gtis(gti0)
    check_gtis(gti1)

    # Check if GTIs are mutually exclusive.
    if not check_separate(gti0, gti1):
        raise ValueError('In order to append, GTIs must be mutually'
            'exclusive.')

    return np.concatenate([gti0, gti1])


def join_gtis(gti0, gti1):
    """Union of two GTIs.

    If GTIs are mutually exclusive, it calls `append_gtis`. Otherwise we put the
    extremes of partially overlapping GTIs on an ideal line and look at the
    number of opened and closed intervals. When the number of closed and opened
    intervals is the same, the full GTI is complete and we close it.

    In practice, we assign to each opening time of a GTI the value -1, and
    the value 1 to each closing time; when the cumulative sum is zero, the
    GTI has ended. The timestamp after each closed GTI is the start of a new
    one.

    (cumsum)   -1   -2         -1   0   -1 -2           -1  -2  -1        0
    GTI A      |-----:----------|   :    |--:------------|   |---:--------|
    FINAL GTI  |-----:--------------|    |--:--------------------:--------|
    GTI B            |--------------|       |--------------------|

    Parameters
    ----------
    gti0: 2-d float array
        [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]

    gti1: 2-d float array
        [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]

    Returns
    -------
    gti: 2-d float array
        The newly created GTI
    """

    gti0 = np.asarray(gti0)
    gti1 = np.asarray(gti1)

    # Check if independently GTIs are well behaved.
    check_gtis(gti0)
    check_gtis(gti1)

    if check_separate(gti0, gti1):
        return append_gtis(gti0, gti1)

    g0 = gti0.flatten()
    # Opening GTI: type = 1; Closing: type = -1
    g0_type = np.asarray(list(zip(-np.ones(int(len(g0) / 2)),
                                  np.ones(int(len(g0) / 2)))))
    g1 = gti1.flatten()
    g1_type = np.asarray(list(zip(-np.ones(int(len(g1) / 2)),
                                  np.ones(int(len(g1) / 2)))))

    g_all = np.append(g0, g1)
    g_type_all = np.append(g0_type, g1_type)
    order = np.argsort(g_all)
    g_all = g_all[order]
    g_type_all = g_type_all[order]

    sums = np.cumsum(g_type_all)

    # Where the cumulative sum is zero, we close the GTI
    closing_bins = sums == 0
    # The next element in the sequence is the start of the new GTI. In the case
    # of the last element, the next is the first. Numpy.roll gives this for
    # free.
    starting_bins = np.roll(closing_bins, 1)

    starting_times = g_all[starting_bins]
    closing_times = g_all[closing_bins]

    final_gti = []
    for start, stop in zip(starting_times, closing_times):
        final_gti.append([start, stop])

    return np.sort(final_gti, axis=0)


def time_intervals_from_gtis(gtis, chunk_length):
    """Returns equal time intervals compatible with GTIs.

    Used to start each FFT/PDS/cospectrum from the start of a GTI,
    and stop before the next gap in data (end of GTI).

    Parameters
    ----------
    gtis : 2-d float array
        [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
    chunk_length : float
        Length of the chunks

    Returns
    -------
    spectrum_start_times : array-like
        List of starting times to use in the spectral calculations.

    spectrum_stop_times : array-like
        List of end times to use in the spectral calculations.

    """
    spectrum_start_times = np.array([], dtype=np.longdouble)
    for g in gtis:
        if g[1] - g[0] < chunk_length:
            continue

        newtimes = np.arange(g[0], g[1] - chunk_length,
                             np.longdouble(chunk_length),
                             dtype=np.longdouble)
        spectrum_start_times = \
            np.append(spectrum_start_times,
                      newtimes)

    assert len(spectrum_start_times) > 0, \
        ("No GTIs are equal to or longer than chunk_length.")
    return spectrum_start_times, spectrum_start_times + chunk_length


def bin_intervals_from_gtis(gtis, chunk_length, time, dt=None, epsilon=0.001):
    """Similar to intervals_from_gtis, but given an input time array.

    Used to start each FFT/PDS/cospectrum from the start of a GTI,
    and stop before the next gap in data (end of GTI).
    In this case, it is necessary to specify the time array containing the
    times of the light curve bins.
    Returns start and stop bins of the intervals to use for the PDS

    Parameters
    ----------
    gtis : 2-d float array
        [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
    chunk_length : float
        Length of the chunks
    time : array-like
        Times of light curve bins

    Other Parameters
    ----------------
    dt : float, default median(diff(time))
        Time resolution of the light curve.
    epsilon : float, default 0.001
        The tolerance, in fraction of dt, for the comparisons at the borders

    Examples
    --------
    >>> time = np.arange(0.5, 13.5)

    >>> gtis = [[0, 5], [6, 8]]

    >>> chunk_length = 2

    >>> start_bins, stop_bins = bin_intervals_from_gtis(gtis,chunk_length,time)

    >>> np.all(start_bins == [0, 2, 6])
    True
    >>> np.all(stop_bins == [2, 4, 8])
    True
    >>> np.all(time[start_bins[0]:stop_bins[0]] == [0.5, 1.5])
    True
    >>> np.all(time[start_bins[1]:stop_bins[1]] == [2.5, 3.5])
    True

    Returns
    -------
    spectrum_start_bins : array-like
        List of starting bins in the original time array to use in spectral
        calculations.
    spectrum_stop_bins : array-like
        List of end bins to use in the spectral calculations.
    """
    if dt is None:
        dt = np.median(np.diff(time))
    nbin = np.long(chunk_length / dt)

    if time[-1] < np.min(gtis) or time[0] > np.max(gtis):
        raise ValueError("Invalid time interval for the given GTIs")

    spectrum_start_bins = np.array([], dtype=np.long)
    for g in gtis:
        if g[1] - g[0] + epsilon * dt < chunk_length:
            continue
        good_low = time - dt / 2 >= g[0] - epsilon * dt
        good_up = time + dt / 2 <= g[1] + epsilon * dt
        good = good_low & good_up
        t_good = time[good]
        if len(t_good) == 0:
            continue
        startbin = np.argmin(np.abs(time - dt / 2 - g[0]))
        stopbin = np.searchsorted(time + dt / 2, g[1], 'right') + 1
        if stopbin > len(time):
            stopbin = len(time)

        if time[startbin] < g[0] + dt/2 - epsilon*dt:
            startbin += 1
        # Would be g[1] - dt/2, but stopbin is the end of an interval
        # so one has to add one bin
        if time[stopbin - 1] > g[1] - dt/2 + epsilon*dt:
            stopbin -= 1

        newbins = np.arange(startbin, stopbin - nbin + 1, nbin,
                            dtype=np.long)
        spectrum_start_bins = \
            np.append(spectrum_start_bins,
                      newbins)
    assert len(spectrum_start_bins) > 0, \
        ("No GTIs are equal to or longer than chunk_length.")
    return spectrum_start_bins, spectrum_start_bins + nbin


def gti_border_bins(gtis, time, dt=None, epsilon=0.001):
    """Find the bins in a time array corresponding to the borders of GTIs.

    GTIs shorter than the bin time are not returned.

    Parameters
    ----------
    gtis : 2-d float array
        [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
    time : array-like
        Times of light curve bins

    Returns
    -------
    spectrum_start_bins : array-like
        List of starting bins of each GTI

    spectrum_stop_bins : array-like
        List of stop bins of each GTI. The elements corresponding to these bins
        should *not* be included.

    Examples
    --------
    >>> times = np.arange(0.5, 13.5)

    >>> start_bins, stop_bins = gti_border_bins(
    ...    [[0, 5], [6, 8]], times)

    >>> np.all(start_bins == [0, 6])
    True
    >>> np.all(stop_bins == [5, 8])
    True
    >>> np.all(times[start_bins[0]:stop_bins[0]] == [ 0.5, 1.5, 2.5, 3.5, 4.5])
    True
    >>> np.all(times[start_bins[1]:stop_bins[1]] == [6.5, 7.5])
    True
    """
    if dt is None:
        dt = np.median(np.diff(time))

    spectrum_start_bins = np.array([], dtype=np.long)
    spectrum_stop_bins = np.array([], dtype=np.long)
    for g in gtis:
        good = (time - dt / 2 >= g[0])&(time + dt / 2 <= g[1])
        t_good = time[good]
        if len(t_good) == 0:
            continue
        startbin = np.argmin(np.abs(time - dt / 2 - g[0]))
        stopbin = np.searchsorted(time + dt / 2, g[1], 'right') + 1
        if stopbin > len(time):
            stopbin = len(time)

        if time[startbin] < g[0] + dt/2 - epsilon*dt:
            startbin += 1
        # Would be g[1] - dt/2, but stopbin is the end of an interval
        # so one has to add one bin
        if time[stopbin - 1] > g[1] - dt/2 + epsilon*dt:
            stopbin -= 1
        spectrum_start_bins = \
            np.append(spectrum_start_bins,
                      [startbin])
        spectrum_stop_bins = \
            np.append(spectrum_stop_bins,
                      [stopbin])
    assert len(spectrum_start_bins) > 0, \
        ("No GTIs are equal to or longer than chunk_length.")
    return spectrum_start_bins, spectrum_stop_bins
