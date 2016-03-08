from __future__ import (absolute_import, unicode_literals, division,
                        print_function)
import numpy as np
import logging
import warnings
import os

from .utils import _order_list_of_arrays, is_string
from .utils import _assign_value_if_none


def get_file_extension(fname):
    """Get the extension from the file name."""
    return os.path.splitext(fname)[1]


def high_precision_keyword_read(hdr, keyword):
    """Read FITS header keywords, also if split in two.

    In the case where the keyword is split in two, like

        MJDREF = MJDREFI + MJDREFF

    in some missions, this function returns the summed value. Otherwise, the
    content of the single keyword

    Parameters
    ----------
    hdr : dict_like
        The FITS header structure, or a dictionary
    keyword : str
        The key to read in the header

    Returns
    -------
    value : long double
        The value of the key, or None if something went wrong
    """
    try:
        value = np.longdouble(hdr[keyword])
        return value
    except:
        pass
    try:
        if len(keyword) == 8:
            keyword = keyword[:7]
        value = np.longdouble(hdr[keyword + 'I'])
        value += np.longdouble(hdr[keyword + 'F'])
        return value
    except:
        return None


def load_gtis(fits_file, gtistring=None):
    """Load GTI from HDU EVENTS of file fits_file."""
    from astropy.io import fits as pf
    import numpy as np

    gtistring = _assign_value_if_none(gtistring, 'GTI')
    logging.info("Loading GTIS from file %s" % fits_file)
    lchdulist = pf.open(fits_file, checksum=True)
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


def _get_additional_data(lctable, additional_columns):
    additional_data = {}
    if additional_columns is not None:
        for a in additional_columns:
            try:
                additional_data[a] = np.array(lctable.field(a))
            except:  # pragma: no cover
                if a == 'PI':
                    logging.warning('Column PI not found. Trying with PHA')
                    additional_data[a] = np.array(lctable.field('PHA'))
                else:
                    raise Exception('Column' + a + 'not found')

    return additional_data


def load_events_and_gtis(fits_file, additional_columns=None,
                         gtistring='GTI,STDGTI',
                         gti_file=None, hduname='EVENTS', column='TIME'):
    """Load event lists and GTIs from one or more files.

    Loads event list from HDU EVENTS of file fits_file, with Good Time
    intervals. Optionally, returns additional columns of data from the same
    HDU of the events.

    Parameters
    ----------
    fits_file : str
    return_limits: bool, optional
        Return the TSTART and TSTOP keyword values
    additional_columns: list of str, optional
        A list of keys corresponding to the additional columns to extract from
        the event HDU (ex.: ['PI', 'X'])

    Returns
    -------
    ev_list : array-like
    gtis: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
    additional_data: dict
        A dictionary, where each key is the one specified in additional_colums.
        The data are an array with the values of the specified column in the
        fits file.
    t_start : float
    t_stop : float
    """
    from astropy.io import fits as pf

    gtistring = _assign_value_if_none(gtistring, 'GTI,STDGTI')
    lchdulist = pf.open(fits_file)

    # Load data table
    try:
        lctable = lchdulist[hduname].data
    except:  # pragma: no cover
        logging.warning('HDU %s not found. Trying first extension' % hduname)
        lctable = lchdulist[1].data

    # Read event list
    ev_list = np.array(lctable.field(column), dtype=np.longdouble)

    # Read TIMEZERO keyword and apply it to events
    try:
        timezero = np.longdouble(lchdulist[1].header['TIMEZERO'])
    except:  # pragma: no cover
        logging.warning("No TIMEZERO in file")
        timezero = np.longdouble(0.)

    ev_list += timezero

    # Read TSTART, TSTOP from header
    try:
        t_start = np.longdouble(lchdulist[1].header['TSTART'])
        t_stop = np.longdouble(lchdulist[1].header['TSTOP'])
    except:  # pragma: no cover
        logging.warning("Tstart and Tstop error. using defaults")
        t_start = ev_list[0]
        t_stop = ev_list[-1]

    # Read and handle GTI extension
    accepted_gtistrings = gtistring.split(',')

    if gti_file is None:
        # Select first GTI with accepted name
        try:
            gti_list = \
                _get_gti_from_extension(
                    lchdulist, accepted_gtistrings=accepted_gtistrings)
        except:  # pragma: no cover
            warnings.warn("No extensions found with a valid name. "
                          "Please check the `accepted_gtistrings` values.")
            gti_list = np.array([[t_start, t_stop]],
                                dtype=np.longdouble)
    else:
        gti_list = load_gtis(gti_file, gtistring)

    additional_data = _get_additional_data(lctable, additional_columns)

    lchdulist.close()

    # Sort event list
    order = np.argsort(ev_list)
    ev_list = ev_list[order]

    additional_data = _order_list_of_arrays(additional_data, order)

    returns = _empty()
    returns.ev_list = ev_list
    returns.gti_list = gti_list
    returns.additional_data = additional_data
    returns.t_start = t_start
    returns.t_stop = t_stop

    return returns


class _empty():
    def __init__(self):
        pass


def mkdir_p(path):  # pragma: no cover
    """Safe mkdir function.

    Parameters
    ----------
    path : str
        Name of the directory/ies to create

    Notes
    -----
    Found at
    http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    """
    import os
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def read_header_key(fits_file, key, hdu=1):
    """Read the header key key from HDU hdu of the file fits_file.

    Parameters
    ----------
    fits_file: str
    key: str
        The keyword to be read

    Other Parameters
    ----------------
    hdu : int
    """
    from astropy.io import fits as pf

    hdulist = pf.open(fits_file)
    try:
        value = hdulist[hdu].header[key]
    except:  # pragma: no cover
        value = ''
    hdulist.close()
    return value


def ref_mjd(fits_file, hdu=1):
    """Read MJDREFF+ MJDREFI or, if failed, MJDREF, from the FITS header.

    Parameters
    ----------
    fits_file : str

    Returns
    -------
    mjdref : numpy.longdouble
        the reference MJD

    Other Parameters
    ----------------
    hdu : int
    """
    import collections

    if isinstance(fits_file, collections.Iterable) and\
            not is_string(fits_file):  # pragma: no cover
        fits_file = fits_file[0]
        logging.info("opening %s" % fits_file)

    from astropy.io import fits as pf

    hdulist = pf.open(fits_file)

    ref_mjd_val = high_precision_keyword_read(hdulist[hdu].header, "MJDREF")

    hdulist.close()
    return ref_mjd_val


def common_name(str1, str2, default='common'):
    """Strip two strings of the letters not in common.

    Filenames must be of same length and only differ by a few letters.

    Parameters
    ----------
    str1 : str
    str2 : str

    Returns
    -------
    common_str : str
        A string containing the parts of the two names in common

    Other Parameters
    ----------------
    default : str
        The string to return if common_str is empty
    """
    if not len(str1) == len(str2):
        return default
    common_str = ''
    # Extract the MP root of the name (in case they're event files)

    for i, letter in enumerate(str1):
        if str2[i] == letter:
            common_str += letter
    # Remove leading and trailing underscores and dashes
    common_str = common_str.rstrip('_').rstrip('-')
    common_str = common_str.lstrip('_').lstrip('-')
    if common_str == '':
        common_str = default
    logging.debug('common_name: %s %s -> %s' % (str1, str2, common_str))
    return common_str


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
    From http://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array
    """  # NOQA
    # Find the indicies of changes in "condition"
    diff = np.diff(condition)
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


def check_gtis(gti):
    """Check if GTIs are well-behaved. No start>end, no overlaps.

    Raises
    ------
    AssertionError
        If GTIs are not well-behaved.
    """
    gti_start = gti[:, 0]
    gti_end = gti[:, 1]

    logging.debug('-- GTI: ' + repr(gti))
    # Check that GTIs are well-behaved
    assert np.all(gti_end >= gti_start), 'This GTI is incorrect'
    # Check that there are no overlaps in GTIs
    assert np.all(gti_start[1:] >= gti_end[:-1]), 'This GTI has overlaps'
    logging.debug('-- Correct')

    return


def create_gti_mask(time, gtis, safe_interval=0, min_length=0,
                    return_new_gtis=False, dt=None):
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
    """
    import collections

    check_gtis(gtis)

    dt = _assign_value_if_none(dt,
                               np.zeros_like(time) + (time[1] - time[0]) / 2)

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
            cond1 = time - dt >= limmin
            cond2 = time + dt <= limmax
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

    assert len(time) == len(condition), \
        'The length of the condition and time arrays must be the same.'
    idxs = contiguous_regions(condition)

    if not isinstance(safe_interval, collections.Iterable):
        safe_interval = [safe_interval, safe_interval]

    dt = _assign_value_if_none(dt,
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
    gti0 = np.array(gti0, dtype=np.longdouble)
    gti1 = np.array(gti1, dtype=np.longdouble)
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

    return np.array(final_gti, dtype=np.longdouble)


def cross_gtis(gti_list):
    """From multiple GTI lists, extract the common intervals *EXACTLY*.

    Parameters
    ----------
    gti_list : array-like
        List of GTI arrays, each one in the usual format [[gti0_0, gti0_1],
        [gti1_0, gti1_1], ...]

    Returns
    -------
    gtis : [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        The newly created GTIs

    See Also
    --------
    cross_two_gtis : Extract the common intervals from two GTI lists *EXACTLY*
    """
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
        assert start_time is not None and stop_time is not None, \
            'Empty GTI and no valid start_time and stop_time. BAD!'

        return np.array([[start_time, stop_time]], dtype=np.longdouble)
    check_gtis(gtis)

    start_time = _assign_value_if_none(start_time, gtis[0][0])
    stop_time = _assign_value_if_none(stop_time, gtis[-1][1])

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

    return np.array(btis, dtype=np.longdouble)


def gti_len(gti):
    """Return the total good time from a list of GTIs."""
    return np.sum([g[1] - g[0] for g in gti])
