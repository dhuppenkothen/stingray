from __future__ import (absolute_import, division,
                        print_function)

import collections
import logging
import math
import numpy as np
import os
import six
import warnings

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

def split_numbers(number):
    """
    Split high precision number(s) into doubles.
    TODO: Consider the option of using a third number to specify shift.

    Parameters
    ----------
    number: long double
        The input high precision number which is to be split

    Returns
    -------
    number_I: double
        First part of high precision number

    number_F: double
        Second part of high precision number
    """

    if isinstance(number, collections.Iterable):
        mods = [math.modf(n) for n in number]
        number_F = [f for f,_ in mods]
        number_I = [i for _,i in mods] 
    else:
        number_F, number_I = math.modf(number)

    return np.double(number_I), np.double(number_F)

def _save_pickle_object(object, filename):
    """
    Save a class object in pickle format.

    Parameters
    ----------
    object: class instance
        A class object whose attributes are saved in a 
        dictionary format

    filename: str
        Name of the file in which object is saved
    """

    with open(filename, "wb" ) as f:
        pickle.dump(object, f)

def _retrieve_pickle_object(filename):
    """
    Retrieves a pickled class object.

    Parameters
    ----------
    filename: str
        Name of the file in which object is saved

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
        A class object whose attributes are saved in a 
        dictionary format

    filename: str
        Name of the file in which object is saved
    """

    items = vars(object)
    attrs = [name for name in items if items[name] is not None]

    with h5py.File(filename, 'w') as hf:   
        for attr in attrs:
            data = items[attr]
            
            # If data is a single number, store as an attribute.
            if _isattribute(data):
                if isinstance(data, np.longdouble):
                    data_I, data_F= split_numbers(data)
                    names = [attr+'_I', attr+'_F']
                    hf.attrs[names[0]] = data_I
                    hf.attrs[names[1]] = data_F
                else:
                    hf.attrs[attr] = data
            
            # If data is an array or list, create a dataset.
            else:
                try:
                    if isinstance(data[0], np.longdouble):
                        data_I, data_F= split_numbers(data)
                        names = [attr+'_I', attr+'_F']
                        hf.create_dataset(names[0], data=data_I)
                        hf.create_dataset(names[1], data=data_F)
                    else:
                        hf.create_dataset(attr, data=data) 
                except IndexError:
                    # To account for numpy arrays of type 'None' (0-d)
                    pass

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

        dset_copy = list(dset_keys)[:]
        for key in dset_keys:

            # Make sure key hasn't been removed
            if key in dset_copy:
                # Longdouble case
                if key[-2:] in ['_I', '_F']:
                    m_key = key[:-2]
                    # Add integer and float parts
                    data[m_key] = np.longdouble(hf[m_key+'_I'].value) 
                    data[m_key] += np.longdouble(hf[m_key+'_F'].value)
                    # Remove integer and float parts from attributes
                    dset_copy.remove(m_key+'_I')
                    dset_copy.remove(m_key+'_F')
                else:
                    data[key] = hf[key].value
        
        attr_copy = list(attr_keys)[:]
        for key in attr_keys:
            
            # Make sure key hasn't been removed
            if key in attr_copy:
                # Longdouble case
                if key[-2:] in ['_I', '_F']:
                    m_key = key[:-2]
                    # Add integer and float parts
                    data[m_key] = np.longdouble(hf.attrs[m_key+'_I'])
                    data[m_key] += np.longdouble(hf.attrs[m_key+'_F'])
                    # Remove integer and float parts from attributes
                    attr_copy.remove(m_key+'_I')
                    attr_copy.remove(m_key+'_F')
                else:
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
    """
    if not isinstance(filename, six.string_types):
        raise TypeError("filename must be string!")

    if 'usecols' in list(kwargs.keys()):
        if np.size(kwargs['usecols']) != 2:
            raise ValueError("Need to define two columns")
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

def _save_fits_object(object, filename, **kwargs):
    """
    Save a class object in fits format.

    Parameters
    ----------
    object: class instance
        A class object whose attributes would be saved in a dictionary format.

    filename: str
        The file name to save to

    Additional Keyword Parameters
    -----------------------------
    tnames: str iterable
        The names of HDU tables. For instance, in case of eventlist, 
        tnames could be ['EVENTS', 'GTI']

    colsassign: dictionary iterable
        This indicates the correct tables to which to assign columns
        to. If this is None or if a column is not provided, it/they will
        be assigned to the first table.

        For example, [{'gti':'GTI'}] indicates that gti values should be 
        stored in GTI table.
    """

    tables = []

    if 'colsassign' in list(kwargs.keys()):
        colsassign = kwargs['colsassign']
        iscolsassigned = True
    else:
        iscolsassigned = False

    if 'tnames' in list(kwargs.keys()): 
        tables = kwargs['tnames']
    else:
        tables = ['MAIN']
    
    items = vars(object)
    attrs = [name for name in items if items[name] is not None]
    
    cols = []
    hdrs = []

    for t in tables:
        cols.append([])
        hdrs.append(fits.Header())
    
    for attr in attrs:
        data = items[attr]

        # Get the index of table to which column belongs
        if iscolsassigned and attr in colsassign.keys():
            index = tables.index(colsassign[attr])
        else:
            index = 0
        
        # If data is a single number, store as metadata
        if _isattribute(data): 
            if isinstance(data, np.longdouble):
                # Longdouble case. Split and save integer and float parts
                data_I, data_F = split_numbers(data)
                names = [attr+'_I', attr+'_F'] 
                hdrs[index][names[0]] = data_I
                hdrs[index][names[1]] = data_F
            else:
                # Normal case. Save as it is
                hdrs[index][attr] = data
        
        # If data is an array or list, insert as table column
        else:
            try:
                if isinstance(data[0], np.longdouble):
                    # Longdouble case. Split and save integer and float parts
                    data_I, data_F= split_numbers(data)
                    names = [attr+'_I', attr+'_F']
                    cols[index].append(fits.Column(name=names[0],format='D', array=data_I))
                    cols[index].append(fits.Column(name=names[1],format='D', array=data_F))
                else:
                    # Normal case. Save as it is
                    cols[index].append(fits.Column(name=attr,format=_lookup_format(data[0]), 
                        array=data))
            except IndexError:
                # To account for numpy arrays of type 'None' (0-d)
                pass

    tbhdu = fits.HDUList()

    # Create binary tables
    for i in range(0, len(tables)):
        if cols[i] != []:
            tbhdu.append(fits.BinTableHDU.from_columns(cols[i], header=hdrs[i], name=tables[i]))
    
    tbhdu.writeto(filename)

def _retrieve_fits_object(filename, **kwargs):
    """
    Retrieves a fits format class object.

    Parameters
    ----------
    filename: str
        The name of file with which object was saved

    Additional Keyword Parameters
    -----------------------------
    cols: str iterable
        The names of columns to extract from fits tables.

    Returns
    -------
    data: dictionary
        Loads the data from a fits object file and returns
        in dictionary format.
    """

    data = {}

    if 'cols' in list(kwargs.keys()):
        cols = [col.upper() for col in kwargs['cols']]
    else:
        cols = []

    with fits.open(filename) as hdulist:
        fits_cols = []

        # Get columns from all tables
        for i in range(1,len(hdulist)):
            fits_cols.append([h.upper() for h in hdulist[i].data.names])

        for c in cols:
            for i in range(0, len(fits_cols)):
                # .upper() is used because `fits` stores values in upper case
                hdr_keys = [h.upper() for h in hdulist[i+1].header.keys()]

                # Longdouble case. Check for columns
                if c+'_I' in fits_cols[i] or c+'_F' in fits_cols[i]:
                    if c not in data.keys():
                        data[c] = np.longdouble(hdulist[i+1].data[c+'_I'])
                        data[c] += np.longdouble(hdulist[i+1].data[c+'_F'])

                # Longdouble case. Check for header keys
                if c+'_I' in hdr_keys or c+'_F' in hdr_keys:
                    if c not in data.keys():
                        data[c] = np.longdouble(hdulist[i+1].header[c+'_I'])
                        data[c] += np.longdouble(hdulist[i+1].header[c+'_F'])

                # Normal case. Check for columns
                elif c in fits_cols[i]:
                    data[c] = hdulist[i+1].data[c]

                # Normal case. Check for header keys
                elif c in hdr_keys:
                    data[c] = hdulist[i+1].header[c]

    return data

def _lookup_format(var):
    """
    Looks up relevant format in fits.
    """

    lookup = {"<type 'int'>":"J", "<type 'float'>":"E", 
        "<type 'numpy.int64'>": "K", "<type 'numpy.float64'>":"D", 
        "<type 'numpy.float128'>":"D", "<type 'str'>":"30A", 
        "<type 'bool'": "L"}

    form = type(var)

    try:
        return lookup[str(form)]
    except KeyError:
        # If an entry is not contained in lookup dictionary
        return "D"

def _isattribute(data):
    """
    Check if data is a single number or an array.
    """

    if isinstance(data, np.ndarray) or isinstance(data, list):
        return False
    else:
        return True

def write(input_, filename, format_='pickle', **kwargs):
    """
    Pickle a class instance. For parameters depending on
    `format_`, see individual function definitions.

    Parameters
    ----------
    object: a class instance
        The object to be stored.

    filename: str
        The name of the file to be created.

    format_: str
        The format in which to store file. Formats supported 
        are pickle, hdf5, ascii or fits.  
    """

    if format_ == 'pickle':
        _save_pickle_object(input_, filename)

    elif format_ == 'hdf5':
        if _H5PY_INSTALLED:
            _save_hdf5_object(input_, filename)
        else:
            utils.simon('h5py not installed, using pickle instead' \
                'to save object.')
            _save_pickle_object(input_, filename.split('.')[0]+
                '.pickle')

    elif format_ == 'ascii':
        _save_ascii_object(input_, filename, **kwargs)

    elif format_ == 'fits':
        _save_fits_object(input_, filename, **kwargs)

    else:
        utils.simon('Format not understood.')

def read(filename, format_='pickle', **kwargs):
    """
    Return a pickled class instance.

    Parameters
    ----------
    filename: str
        The name of the file to be retrieved.

    format_: str
        The format used to store file. Supported formats are
        pickle, hdf5, ascii or fits.
    
    Returns
    -------
    If format_ is 'pickle', a class object is returned.
    If format_ is 'ascii', astropy.table object is returned.
    If format_ is 'hdf5' or 'fits', a dictionary object is returned.
    """

    if format_ == 'pickle':
        return _retrieve_pickle_object(filename)

    elif format_ == 'hdf5':
        if _H5PY_INSTALLED:
            return _retrieve_hdf5_object(filename)
        else:
            utils.simon('h5py not installed, cannot read an' \
                'hdf5 object.')

    elif format_ == 'ascii':
        return _retrieve_ascii_object(filename, **kwargs)

    elif format_ == 'fits':
        return _retrieve_fits_object(filename, **kwargs)
    
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
