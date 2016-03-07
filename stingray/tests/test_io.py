from __future__ import (absolute_import, unicode_literals, division,
                        print_function)
import numpy as np
import os
curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, 'data')


class TestIO(object):

    """Real unit tests."""

    def test_crossgti1(self):
        """Test the basic working of the intersection of GTIs."""
        from ..io import cross_gtis
        gti1 = np.array([[1, 4]])
        gti2 = np.array([[2, 5]])
        newgti = cross_gtis([gti1, gti2])

        assert np.all(newgti == [[2, 4]]), 'GTIs do not coincide!'

    def test_crossgti2(self):
        """A more complicated example of intersection of GTIs."""
        from ..io import cross_gtis
        gti1 = np.array([[1, 2], [4, 5], [7, 10], [11, 11.2], [12.2, 13.2]])
        gti2 = np.array([[2, 5], [6, 9], [11.4, 14]])
        newgti = cross_gtis([gti1, gti2])

        assert np.all(newgti == [[4.0, 5.0], [7.0, 9.0], [12.2, 13.2]]), \
            'GTIs do not coincide!'

    def test_contiguous(self):
        """A more complicated example of intersection of GTIs."""
        from ..io import contiguous_regions
        array = np.array([0, 1, 1, 0, 1, 1, 1], dtype=bool)
        cont = contiguous_regions(array)
        assert np.all(cont == np.array([[1, 3], [4, 7]])), \
            'Contiguous region wrong'

    def test_bti(self):
        """Test the inversion of GTIs."""
        from ..io import get_btis
        gti = np.array([[1, 2], [4, 5], [7, 10], [11, 11.2], [12.2, 13.2]])
        bti = get_btis(gti)

        assert np.all(bti == [[2, 4], [5, 7], [10, 11], [11.2, 12.2]]), \
            'BTI is wrong!, %s' % repr(bti)

    def test_gti_mask(self):
        from ..io import create_gti_mask
        arr = np.array([0, 1, 2, 3, 4, 5, 6])
        gti = np.array([[0, 2.1], [3.9, 5]])
        mask = create_gti_mask(arr, gti)
        print(mask)
        # NOTE: the time bin has to be fully inside the GTI. That is why the bin at times \
        # 0, 2, 4 and 5 are not in.
        assert np.all(mask == np.array([0, 1, 0, 0, 0, 0, 0], dtype=bool))

    def test_gti_gti_from_condition(self):
        from ..io import create_gti_from_condition
        t = np.array([0, 1, 2, 3, 4, 5, 6])
        condition = np.array([1, 1, 0, 0, 1, 0, 0], dtype=bool)
        gti = create_gti_from_condition(t, condition)
        assert np.all(gti == np.array([[-0.5, 1.5], [3.5, 4.5]]))


    def test_common_name(self):
        """Test the common_name function."""
        from ..io import common_name
        a = 'A_3-50_A'
        b = 'B_3-50_B'
        assert common_name(a, b) == '3-50'

    def test_high_precision_keyword(self):
        """Test high precision FITS keyword read."""
        from ..io import high_precision_keyword_read
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
        assert high_precision_keyword_read(hdr, "BU") == None, "Inexistent key read incorrectly"

    def test_event_file_read(self):
        """Test event file reading."""
        from ..io import load_events_and_gtis
        fname = os.path.join(datadir, 'monol_testA.evt')
        load_events_and_gtis(fname, additional_columns=["PI"])

    def test_load_gtis(self):
        """Test event file reading."""
        from ..io import load_gtis
        fname = os.path.join(datadir, 'monol_testA.evt')
        load_gtis(fname, gtistring="GTI")

    def test_read_header_key(self):
        """Test event file reading."""
        from ..io import read_header_key
        fname = os.path.join(datadir, 'monol_testA.evt')
        assert read_header_key(fname, "INSTRUME") == 'FPMA'
        assert read_header_key(fname, "BU") == ""

    def test_read_mjdref(self):
        """Test event file reading."""
        from ..io import ref_mjd
        fname = os.path.join(datadir, 'monol_testA.evt')
        print(ref_mjd(fname))
        assert ref_mjd(fname) is not None
