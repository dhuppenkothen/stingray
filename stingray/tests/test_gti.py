from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import numpy as np
import pytest
import os

from ..utils import contiguous_regions
from ..gti import cross_gtis, append_gtis, load_gtis, get_btis, join_gtis
from ..gti import check_separate, create_gti_mask, check_gtis
from ..gti import create_gti_from_condition, gti_len, gti_border_bins
from ..gti import time_intervals_from_gtis, bin_intervals_from_gtis

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, 'data')

class TestGTI(object):

    """Real unit tests."""

    def test_crossgti1(self):
        """Test the basic working of the intersection of GTIs."""
        gti1 = np.array([[1, 4]])
        gti2 = np.array([[2, 5]])
        newgti = cross_gtis([gti1, gti2])

        assert np.all(newgti == [[2, 4]]), 'GTIs do not coincide!'

    def test_crossgti2(self):
        """A more complicated example of intersection of GTIs."""
        gti1 = np.array([[1, 2], [4, 5], [7, 10], [11, 11.2], [12.2, 13.2]])
        gti2 = np.array([[2, 5], [6, 9], [11.4, 14]])
        newgti = cross_gtis([gti1, gti2])

        assert np.all(newgti == [[4.0, 5.0], [7.0, 9.0], [12.2, 13.2]]), \
            'GTIs do not coincide!'

    def test_crossgti2(self):
        """A more complicated example of intersection of GTIs."""
        gti1 = np.array([[1, 2], [4, 5], [7, 10]])
        newgti = cross_gtis([gti1])

        assert np.all(newgti == gti1), \
            'GTIs do not coincide!'
    def test_bti(self):
        """Test the inversion of GTIs."""
        gti = np.array([[1, 2], [4, 5], [7, 10], [11, 11.2], [12.2, 13.2]])
        bti = get_btis(gti)

        assert np.all(bti == [[2, 4], [5, 7], [10, 11], [11.2, 12.2]]), \
            'BTI is wrong!, %s' % repr(bti)

    def test_gti_mask(self):
        arr = np.array([0, 1, 2, 3, 4, 5, 6])
        gti = np.array([[0, 2.1], [3.9, 5]])
        mask, new_gtis = create_gti_mask(arr, gti, return_new_gtis=True)
        # NOTE: the time bin has to be fully inside the GTI. That is why the
        # bin at times 0, 2, 4 and 5 are not in.
        assert np.all(mask == np.array([0, 1, 0, 0, 0, 0, 0], dtype=bool))

    def test_gti_from_condition1(self):
        t = np.array([0, 1, 2, 3, 4, 5, 6])
        condition = np.array([1, 1, 0, 0, 1, 0, 0], dtype=bool)
        gti = create_gti_from_condition(t, condition)
        assert np.all(gti == np.array([[-0.5, 1.5], [3.5, 4.5]]))

    def test_gti_from_condition2(self):
        t = np.array([0, 1, 2, 3, 4, 5, 6])
        condition = np.array([1, 1, 1, 1, 0, 1, 0], dtype=bool)
        gti = create_gti_from_condition(t, condition, safe_interval=1)
        assert np.all(gti == np.array([[0.5, 2.5]]))

    def test_load_gtis(self):
        """Test event file reading."""
        fname = os.path.join(datadir, 'monol_testA.evt')
        load_gtis(fname, gtistring="GTI")

    def test_check_separate_overlapping_case(self):
        """Test if intersection between two GTIs can be detected. """
        gti1 = np.array([[1, 2], [4, 5], [7, 10], [11, 11.2], [12.2, 13.2]])
        gti2 = np.array([[2, 5], [6, 9], [11.4, 14]])
        assert check_separate(gti1, gti2) == False

    def test_check_separate_nonoverlapping_case(self):
        """Test if two non-overlapping GTIs can be detected."""
        gti1 = np.array([[1, 2], [4, 5]])
        gti2 = np.array([[6, 7], [8, 9]])
        assert check_separate(gti1, gti2) == True

    def test_check_separate_empty_case(self):
        """Test if intersection between two GTIs can be detected. """
        gti1 = np.array([[1, 2], [4, 5], [7, 10], [11, 11.2], [12.2, 13.2]])
        gti2 = np.array([])
        assert check_separate(gti1, gti2) == True

    def test_append_gtis(self):
        """Test if two non-overlapping GTIs can be appended. """
        gti1 = np.array([[1, 2], [4, 5]])
        gti2 = np.array([[6, 7], [8, 9]])
        assert np.all(append_gtis(gti1, gti2) == [[1,2],[4,5],[6,7],[8,9]])

    def test_append_overlapping_gtis(self):
        """Test if exception is raised in event of overlapping gtis."""
        gti1 = np.array([[1, 2], [4, 5]])
        gti2 = np.array([[3, 4.5], [8, 9]])

        with pytest.raises(ValueError):
            append_gtis(gti1, gti2)

    def test_join_gtis_nonoverlapping(self):
        gti0 = [[0, 1], [2, 3]]
        gti1 = [[10, 11], [12, 13]]
        assert np.all(join_gtis(gti0, gti1) == np.array([[0, 1], [2, 3],
                                                         [10, 11], [12, 13]]))

    def test_join_gtis_overlapping(self):
        gti0 = [[0, 1], [2, 3], [4, 8]]
        gti1 = [[7, 8], [10, 11], [12, 13]]
        assert np.all(join_gtis(gti0, gti1) == np.array([[0, 1], [2, 3], [4, 8],
                                                         [10, 11], [12, 13]]))


    def test_time_intervals_from_gtis(self):
        """Test the division of start and end times to calculate spectra."""
        start_times, stop_times = \
            time_intervals_from_gtis([[0, 400], [1022, 1200],
                                      [1210, 1220]], 128)
        assert np.all(start_times == np.array([0, 128, 256, 1022]))
        assert np.all(stop_times == np.array([0, 128, 256, 1022]) + 128)

    def test_time_intervals_from_gtis_frac(self):
        """Test the division of start and end times to calculate spectra."""
        start_times, stop_times = \
            time_intervals_from_gtis([[0, 400], [1022, 1200],
                                      [1210, 1220]], 128, fraction_step=0.5)
        assert np.all(start_times == np.array([0, 64, 128, 192, 256, 1022]))
        assert np.all(stop_times == start_times + 128)

    def test_bin_intervals_from_gtis(self):
        """Test the division of start and end times to calculate spectra."""
        times = np.arange(0.5, 13.5)
        start_bins, stop_bins = \
            bin_intervals_from_gtis([[0, 5], [6, 8]], 2, times)

        assert np.all(start_bins == np.array([0, 2, 6]))
        assert np.all(stop_bins == np.array([2, 4, 8]))

    def test_bin_intervals_from_gtis_frac(self):
        """Test the division of start and end times to calculate spectra."""
        times = np.arange(0.5, 13.5)
        start_bins, stop_bins = \
            bin_intervals_from_gtis([[0, 5], [6, 8]], 2, times,
                                    fraction_step=0.5)

        assert np.all(start_bins == np.array([0, 1, 2, 3, 6]))
        assert np.all(stop_bins == np.array([2, 3, 4, 5, 8]))

    def test_gti_border_bins(self):
        times = np.arange(0.5, 2.5)

        start_bins, stop_bins = gti_border_bins([[0, 2]], times)
        assert start_bins == [0]
        assert stop_bins == [2]

    def test_gti_border_bins_many_bins(self):
        times = np.arange(0, 2, 0.0001) + 0.00005

        start_bins, stop_bins = gti_border_bins([[0, 2]], times)
        assert start_bins == [0]
        assert stop_bins == [len(times)]

    def test_decide_spectrum_lc_intervals_invalid(self):
        with pytest.raises(ValueError):
            a, b = bin_intervals_from_gtis([[0, 400]], 128, [500, 501])
        with pytest.raises(ValueError):
            a, b = bin_intervals_from_gtis([[1000, 1400]], 128, [500, 501])

    def test_gti_length(self):
        assert gti_len([[0, 5], [6, 7]]) == 6

    def test_check_gtis_shape(self):
        with pytest.raises(TypeError):
            check_gtis([0, 1])

        with pytest.raises(TypeError):
            check_gtis([[0, 1], [0]])

    def test_check_gtis_values(self):
        with pytest.raises(ValueError):
            check_gtis([[0, 2], [1, 3]])

        with pytest.raises(ValueError):
            check_gtis([[1, 0]])
