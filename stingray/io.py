from __future__ import (absolute_import, division,
                        print_function)

import numpy as np
import logging
import warnings
import os
import six

from astropy.io import fits
from astropy.table import Table

import stingray.utils as utils
from .utils import order_list_of_arrays, is_string
from .utils import assign_value_if_none

from .gti import  _get_gti_from_extension, load_gtis

try:
    # Python 2
    import cPickle as pickle
except:
    # Python 3
    import pickle

_H5PY_INSTALLED = True

try:
    import h5py 
except:
    _H5PY_INSTALLED = False

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

    gtistring = assign_value_if_none(gtistring, 'GTI,STDGTI')
    lchdulist = fits.open(fits_file)

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

    additional_data = order_list_of_arrays(additional_data, order)

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

    hdulist = fits.open(fits_file)
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

    hdulist = fits.open(fits_file)

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

def _save_pickle_object(object, filename):
    """
    Save a class object in pickle format.

    Parameters
    ----------
    object: class instance
        A class object whose attributes would be saved in a dictionary format.

    filename: str
        The file name to save to
    """

    with open(filename, "wb" ) as f:
        pickle.dump(object, f)

def _retrieve_pickle_object(filename):
    """
    Retrieves a pickled class object.

    Parameters
    ----------
    filename: str
        The name of file with which object was saved

    Returns
    -------
    data: class object
    """
    with open(filename, "rb" ) as f:
        return pickle.load(f)

def _save_hdf5_object(object, filename):
    """
    Save a class object in hdf5 format.

    Parameters
    ----------
    object: class instance
        A class object whose attributes would be saved in a dictionary format.

    filename: str
        The file name to save to
    """
    items = vars(object)
    attrs = [name for name in items]

    with h5py.File(filename, 'w') as hf:   
        for attr in attrs:
            data = items[attr]
            
            # If data is a single number, store as an attribute.
            if _isattribute(data):

                if isinstance(data, np.longdouble):
                    data = np.double(data) 
                    utils.simon("Casting data as double instead of longdouble.")
                hf.attrs[attr] = data
            
            # If data is a numpy array, create a dataset.
            else:

                if isinstance(data[0], np.longdouble):
                    data = np.double(data) 
                    utils.simon("Casting data as double instead of longdouble.")
                hf.create_dataset(attr, data=data) 

def _retrieve_hdf5_object(filename):
    """
    Retrieves an hdf5 format class object.

    Parameters
    ----------
    filename: str
        The name of file with which object was saved

    Returns
    -------
    data: dictionary
        Loads the data from an hdf5 object file and returns
        in dictionary format.
    """
    with h5py.File(filename, 'r') as hf:
        dset_keys = hf.keys()
        attr_keys = hf.attrs.keys()
        data = {}

        for key in dset_keys:
            data[key] = hf[key].value

        for key in attr_keys:
            data[key] = hf.attrs[key]
    
    return data

def _save_ascii_object(object, filename, fmt="%.18e", **kwargs):
    """
    Save an array to a text file.

    Parameters
    ----------
    object : numpy.ndarray
        An array with the data to be saved

    filename : str
        The file name to save to

    fmt : str or sequence of strs, optional
        Use for formatting of columns. See `numpy.savetxt` documentation
        for details.

    Other Parameters
    ----------------
    kwargs : any keyword argument taken by `numpy.savetxt`

    """

    try:
        np.savetxt(filename, object, fmt=fmt, **kwargs)
    except TypeError:
        raise Exception("Formatting of columns not recognized! Use 'fmt' option to "
              "format columns including strings or mixed types!")

    pass

def _retrieve_ascii_object(filename, **kwargs):
    """
    Helper function to retrieve ascii objects from file.
    Uses astropy.Table for reading and storing the data.

    Parameters
    ----------
    filename : str
        The name of the file with the data to be retrieved.

    Additional Keyword Parameters
    -----------------------------
    usecols : {int | iterable}
        The indices of the columns in the file to be returned.
        By default, all columns will be returned

    skiprows : int
        The number of rows at the beginning to skip
        By default, no rows will be skipped.

    names : iterable
        A list of column names to be attached to the columns.
        By default, no column names are added, unless they are specified
        in the file header and can be read by astropy.Table.read
        automatically.

    Returns
    -------
    data : astropy.Table object
        An astropy.Table object with the data from the file


    Example
    -------
    """

    assert isinstance(filename, six.string_types), \
        "filename must be string!"

    if 'usecols' in list(kwargs.keys()):
        assert np.size(kwargs['usecols']) == 2, "Need to define two columns"
        usecols = kwargs["usecols"]
    else:
        usecols = None

    if 'skiprows' in list(kwargs.keys()):
        assert isinstance(kwargs["skiprows"], int)
        skiprows = kwargs["skiprows"]
    else:
        skiprows = 0

    if "names" in list(kwargs.keys()):
        names = kwargs["names"]
    else:
        names = None

    data = Table.read(filename, data_start=skiprows,
                      names=names, format="ascii")

    if usecols is None:
        return data
    else:
        colnames = np.array(data.colnames)
        cols = colnames[usecols]

        return data[cols]

def _isattribute(data):
    """
    Check if data is a single number or an array.
    """

    try:
        return len(data) < 0
    except:
        return True

def write(input_, filename, format_='pickle', **kwargs):
    """
    Pickle a class instance. For parameters depending on
    `format_`, see individual function definitions.

    Parameters
    ----------
    object: a class instance
    filename: str
        name of the file to be created.
    format_: str
        pickle, hdf5, ascii ...
    """

    if format_ == 'pickle':
        _save_pickle_object(input_, filename)

    elif format_ == 'hdf5':
        if _H5PY_INSTALLED:
            _save_hdf5_object(input_, filename)

        else:
            utils.simon('h5py not installed, using pickle instead to save object.')
            _save_pickle_object(input_, filename)

    elif format_ == 'ascii':
        _save_ascii_object(input_, filename, **kwargs)

    else:
        utils.simon('Format not understood.')

def read(filename, format_='pickle', **kwargs):
    """
    Return a pickled class instance.

    Parameters
    ----------
    filename: str
        name of the file to be retrieved.
    format_: str
        pickle, hdf5, ascii ...
    """

    if format_ == 'pickle':
        return _retrieve_pickle_object(filename)

    elif format_ == 'hdf5':
        if _H5PY_INSTALLED:
            return _retrieve_hdf5_object(filename)
        else:
            utils.simon('h5py not installed, cannot read an hdf5 object.')

    elif format_ == 'ascii':
        return _retrieve_ascii_object(filename, **kwargs)
    
    else:
        utils.simon('Format not understood.')
        
def savefig(filename, **kwargs):
    """
    Save a figure plotted by Matplotlib.

    Note : This function is supposed to be used after the ``plot``
    function. Otherwise it will save a blank image with no plot.

    Parameters
    ----------
    filename : str
        The name of the image file. Extension must be specified in the
        file name. For example filename with `.png` extension will give a
        rasterized image while `.pdf` extension will give a vectorized
        output.

    kwargs : keyword arguments
        Keyword arguments to be passed to ``savefig`` function of
        ``matplotlib.pyplot``. For example use `bbox_inches='tight'` to
        remove the undesirable whitepace around the image.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib required for savefig()")

    if not plt.fignum_exists(1):
        utils.simon("use ``plot`` function to plot the image first and "
                    "then use ``savefig`` to save the figure.")

    plt.savefig(filename, **kwargs)
