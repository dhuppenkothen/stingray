from __future__ import (absolute_import, unicode_literals, division,
                        print_function)
import numpy as np
import os

from astropy.tests.helper import pytest


from ..io import read, write
from ..io import ref_mjd
from ..io import high_precision_keyword_read
from ..io import load_events_and_gtis
from ..io import read_header_key

import warnings

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, 'data')


class TestIO(object):

    """Real unit tests."""


    def test_common_name(self):
        """Test the common_name function."""
        from ..io import common_name
        a = 'A_3-50_A'
        b = 'B_3-50_B'
        assert common_name(a, b) == '3-50'

    def test_high_precision_keyword(self):
        """Test high precision FITS keyword read."""
        hdr = {"MJDTESTI": 100, "MJDTESTF": np.longdouble(0.5),
               "CIAO": np.longdouble(0.)}
        assert \
            high_precision_keyword_read(hdr,
                                        "MJDTEST") == np.longdouble(100.5), \
            "Keyword MJDTEST read incorrectly"
        assert \
            high_precision_keyword_read(hdr,
                                        "MJDTESTA") == np.longdouble(100.5), \
            "Keyword MJDTESTA read incorrectly"
        assert \
            high_precision_keyword_read(hdr, "CIAO") == np.longdouble(0.), \
            "Keyword CIAO read incorrectly"
        assert high_precision_keyword_read(hdr, "BU") is None, \
            "Inexistent key read incorrectly"

    def test_event_file_read(self):
        """Test event file reading."""
        fname = os.path.join(datadir, 'monol_testA.evt')
        load_events_and_gtis(fname, additional_columns=["PI"])


    def test_read_header_key(self):
        """Test event file reading."""
        fname = os.path.join(datadir, 'monol_testA.evt')
        assert read_header_key(fname, "INSTRUME") == 'FPMA'
        assert read_header_key(fname, "BU") == ""

    def test_read_mjdref(self):
        """Test event file reading."""
        fname = os.path.join(datadir, 'monol_testA.evt')
        print(ref_mjd(fname))
        assert ref_mjd(fname) is not None


class TestIOReadWrite(object):
    """A class to test all the read and write functions."""
    def __init__(self):
        self.x = 10
        self.y = np.array([1,2,3])

    def test_operation(self):
        return self.x * 10


class TestFileFormats(object):

    def test_pickle_with_class_objects(self):
        """Test pickle object writing and reading."""
        test_object = TestIOReadWrite()
        write(test_object, filename='test.pickle', format_='pickle')
        assert read('test.pickle', 'pickle') is not None
        os.remove('test.pickle')

    def test_pickle_with_dict_objects(self):
        """Test pickle object writing and reading."""
        test_object = TestIOReadWrite()
        write(test_object, filename='test.pickle', format_='pickle', save_as_dict=True)
        assert read('test.pickle', 'pickle') is not None
        os.remove('test.pickle')

    def test_pickle_attributes_with_class_objects(self):
        """Test if pickle maintains class object attributes."""
        test_object = TestIOReadWrite()
        write(test_object, filename='test.pickle', format_='pickle')
        assert read('test.pickle', 'pickle').x == test_object.x
        assert (read('test.pickle', 'pickle').y == test_object.y).all()
        os.remove('test.pickle')

    def test_pickle_attributes_with_dict_objects(self):
        """Test if pickle maintains class object attributes."""
        test_object = TestIOReadWrite()
        write(test_object, 'test.pickle', 'pickle', save_as_dict=True)
        assert read('test.pickle', 'pickle').x == test_object.x
        assert (read('test.pickle', 'pickle').y == test_object.y).all()
        os.remove('test.pickle')

    def test_pickle_functions(self):
        """Test if pickle maintains class methods."""
        test_object = TestIOReadWrite()
        write(test_object,'test.pickle', 'pickle')
        assert read('test.pickle', 'pickle').test_operation() == test_object.x * 10
        os.remove('test.pickle')

    def test_hdf5_write(self):
        """Test write functionality of hdf5."""
        test_object = TestIOReadWrite()
        write(test_object, 'test.hdf5', 'hdf5')
        os.remove('test.hdf5')

    def test_hdf5_read(self):
        """Test read functionality of hdf5."""
        test_object = TestIOReadWrite()
        write(test_object, 'test.hdf5', 'hdf5')
        read('test.hdf5','hdf5')
        os.remove('test.hdf5')

    def test_hdf5_data_recovery(self):
        """Test if hdf5 stored data is properly recovered."""
        test_object = TestIOReadWrite()
        write(test_object, 'test.hdf5', 'hdf5')
        data = read('test.hdf5','hdf5')
        assert data['x'] == test_object.x
        assert (data['y'] == np.array(test_object.y)).all()
        os.remove('test.hdf5')

    def test_save_ascii(self):
        time = [1, 2, 3, 4]
        counts = [2, 3, 41, 4]

        write(np.array([time, counts]).T, "ascii_test.txt",
              "ascii")

        os.remove("ascii_test.txt")

    def test_save_ascii_with_mixed_types(self):
        time = ["bla", 1, 2, 3]
        counts = [2,3,41,4]
        with pytest.raises(Exception):
             write(np.array([time, counts]).T,
                   "ascii_test.txt", "ascii")

    def test_save_ascii_with_format(self):
        time = ["bla", 1, 2, 3]
        counts = [2,3,41,4]
        write(np.array([time, counts]).T,
              filename="ascii_test.txt", format_="ascii",
              fmt=["%s", "%s"])

    def test_read_ascii(self):
        time = [1,2,3,4,5]
        counts = [5,7,8,2,3]
        np.savetxt("ascii_test.txt", np.array([time, counts]).T)
        read("ascii_test.txt", "ascii")
        os.remove("ascii_test.txt")

    def test_savefig_matplotlib_not_installed(self):
        from ..io import savefig
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            lc = Lightcurve([1, 2, 3], [2, 2, 2])
            try:
                savefig("test.png")
            except Exception as e:
                assert type(e) is ImportError
                assert str(e) == "Matplotlib required for savefig()"

    def test_savefig_without_plot(self):
        import matplotlib.pyplot as plt
        from ..io import savefig
        plt.close()
        with warnings.catch_warnings(record=True) as w:
            savefig('test.png')
            assert "plot the image first" in str(w[0].message)
        os.unlink('test.png')

    def test_savefig(self):
        import matplotlib.pyplot as plt
        from ..io import savefig
        plt.plot([1, 2, 3])
        savefig("test.png")
        os.unlink("test.png")
