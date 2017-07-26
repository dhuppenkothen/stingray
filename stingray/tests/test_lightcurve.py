import numpy as np

from astropy.tests.helper import pytest
import warnings
import os
import matplotlib.pyplot as plt

from stingray import Lightcurve
from stingray.exceptions import StingrayError
from stingray.gti import create_gti_mask

np.random.seed(20150907)

_H5PY_INSTALLED = True

try:
    import h5py
except ImportError:
    _H5PY_INSTALLED = False


class TestLightcurve(object):

    @classmethod
    def setup_class(cls):
        cls.times = np.array([1, 2, 3, 4])
        cls.counts = np.array([2, 2, 2, 2])
        cls.dt = 1.0
        cls.gti = [[0.5, 4.5]]

    def test_create(self):
        """
        Demonstrate that we can create a trivial Lightcurve object.
        """
        lc = Lightcurve(self.times, self.counts)
    
    def test_irregular_time_warning(self):
        """
        Check if inputting an irregularly spaced time iterable throws out
        a warning.
        """
        times = [1, 2, 3, 5, 6]
        counts = [2, 2, 2, 2, 2]
        warn_str = ("SIMON says: Bin sizes in input time array aren't equal "
                    "throughout! This could cause problems with Fourier "
                    "transforms. Please make the input time evenly sampled.")

        with warnings.catch_warnings(record=True) as w:
            lc = Lightcurve(times, counts, err_dist="poisson")
            assert str(w[0].message) == warn_str

    def test_unrecognize_err_dist_warning(self):
        """
        Check if a non-poisson error_dist throws the correct warning.
        """
        times = [1, 2, 3, 4, 5]
        counts = [2, 2, 2, 2, 2]
        warn_str = ("SIMON says: Stingray only uses poisson err_dist at "
                    "the moment, We are setting your errors to zero. Sorry for "
                    "the inconvenience.")

        with warnings.catch_warnings(record=True) as w:
            lc = Lightcurve(times, counts, err_dist='gauss')
            assert str(w[0].message) == warn_str

    def test_dummy_err_dist_fail(self):
        """
        Check if inputting an irregularly spaced time iterable throws out
        a warning.
        """
        times = [1, 2, 3, 4, 5]
        counts = [2, 2, 2, 2, 2]

        with pytest.raises(StingrayError):
            lc = Lightcurve(times, counts, err_dist='joke')

    def test_invalid_data(self):
        times = [1, 2, 3, 4, 5]
        counts = [2, 2, np.nan, 2, 2]
        counts_err = [1, 2, 3, np.nan, 2]

        with pytest.raises(ValueError):
            lc = Lightcurve(times, counts)

        with pytest.raises(ValueError):
            lc = Lightcurve(times, [2]*5, err=counts_err)

    def test_n(self):
        lc = Lightcurve(self.times, self.counts)
        assert lc.n == 4

    def test_bin_edges(self):
        bin_lo = [0.5,  1.5,  2.5,  3.5]
        bin_hi = [1.5,  2.5,  3.5,  4.5]
        lc = Lightcurve(self.times, self.counts)
        assert np.allclose(lc.bin_lo, bin_lo)
        assert np.allclose(lc.bin_hi, bin_hi)

    def test_lightcurve_from_toa(self):
        lc = Lightcurve.make_lightcurve(self.times, self.dt)

    def test_tstart(self):
        tstart = 0.0
        lc = Lightcurve.make_lightcurve(self.times, self.dt, tstart=0.0)
        assert lc.tstart == tstart
        assert lc.time[0] == tstart + 0.5*self.dt

    def test_tseg(self):
        tstart = 0.0
        tseg = 5.0
        lc = Lightcurve.make_lightcurve(self.times, self.dt,
                                        tseg=tseg, tstart=tstart)

        assert lc.tseg == tseg
        assert lc.time[-1] - lc.time[0] == tseg - self.dt

    def test_nondivisble_tseg(self):
        """
        If the light curve length input is not divisible by the time
        resolution, the last (fractional) time bin will be dropped.
        """
        tstart = 0.0
        tseg = 5.5
        lc = Lightcurve.make_lightcurve(self.times, self.dt,
                                        tseg=tseg, tstart=tstart)
        assert lc.tseg == int(tseg/self.dt)

    def test_correct_timeresolution(self):
        lc = Lightcurve.make_lightcurve(self.times, self.dt)
        assert np.isclose(lc.dt, self.dt)

    def test_bin_correctly(self):
        ncounts = np.array([2, 1, 0, 3])
        tstart = 0.0
        tseg = 4.0

        toa = np.hstack([np.random.uniform(i, i+1, size=n)
                         for i, n in enumerate(ncounts)])

        dt = 1.0
        lc = Lightcurve.make_lightcurve(toa, dt, tseg=tseg, tstart=tstart)

        assert np.allclose(lc.counts, ncounts)

    def test_countrate(self):
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0 + dt/2, 5 - dt/2, dt)
        counts = np.zeros_like(times) + mean_counts
        lc = Lightcurve(times, counts)
        assert np.allclose(lc.countrate, np.zeros_like(counts) +
                           mean_counts/dt)

    def test_input_countrate(self):
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0 + dt/2, 5 - dt/2, dt)
        countrate = np.zeros_like(times) + mean_counts
        lc = Lightcurve(times, countrate, input_counts=False)
        assert np.allclose(lc.counts, np.zeros_like(countrate) +
                           mean_counts*dt)

    def test_meanrate(self):
        times = [0.5, 1.0, 1.5, 2.0]
        counts = [2, 3, 3, 4]
        lc = Lightcurve(times, counts)
        assert lc.meanrate == 6

    def test_meancounts(self):
        counts = [2, 3, 3, 4]
        lc = Lightcurve(self.times, counts)
        assert lc.meancounts == 3

    def test_lc_gtis(self):
        t = [0.5, 1.5, 2.5, 3.5, 4.5]
        lc = [5, 5, 0, 5, 5]
        gtis = [[0, 2], [3, 5]]
        lc = Lightcurve(t, lc, gti=gtis)

        assert lc.meanrate == 5
        assert lc.meancounts == 5

    def test_creating_lightcurve_raises_type_error_when_input_is_none(self):
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0 + dt/2, 5 - dt/2, dt)
        counts = np.array([None] * times.shape[0])
        with pytest.raises(TypeError):
            lc = Lightcurve(times, counts)

    def test_creating_lightcurve_raises_type_error_when_input_is_inf(self):
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0 + dt/2, 5 - dt/2, dt)
        counts = np.array([np.inf] * times.shape[0])
        with pytest.raises(ValueError):
            lc = Lightcurve(times, counts)

    def test_creating_lightcurve_raises_type_error_when_input_is_nan(self):
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0 + dt/2, 5 - dt/2, dt)
        counts = np.array([np.nan] * times.shape[0])
        with pytest.raises(ValueError):
            lc = Lightcurve(times, counts)

    def test_init_with_diff_array_lengths(self):
        time = [1, 2, 3]
        counts = [2, 2, 2, 2]

        with pytest.raises(StingrayError):
            lc = Lightcurve(time, counts)

    def test_add_with_different_time_arrays(self):
        _times = [1.1, 2.1, 3.1, 4.1, 5.1]
        _counts = [2, 2, 2, 2, 2]

        with pytest.raises(ValueError):

            lc1 = Lightcurve(self.times, self.counts)
            lc2 = Lightcurve(_times, _counts)

            lc = lc1 + lc2

    def test_add_with_different_err_dist(self):
        lc1 = Lightcurve(self.times, self.counts)
        lc2 = Lightcurve(self.times, self.counts, err=self.counts / 2,
                         err_dist="gauss")
        with warnings.catch_warnings(record=True) as w:
            lc = lc1 + lc2
            assert "ightcurves have different statistics" in str(w[0].message)

    def test_add_with_same_gtis(self):
        lc1 = Lightcurve(self.times, self.counts, gti=self.gti)
        lc2 = Lightcurve(self.times, self.counts, gti=self.gti)
        lc = lc1 + lc2
        np.testing.assert_almost_equal(lc.gti, self.gti)

    def test_add_with_different_mjdref(self):
        lc1 = Lightcurve(self.times, self.counts, gti=self.gti, mjdref=57000)
        lc2 = Lightcurve(self.times, self.counts, gti=self.gti, mjdref=57001)
        with pytest.raises(ValueError):
            lc1 + lc2

    def test_add_with_different_gtis(self):
        gti = [[0., 3.5]]
        lc1 = Lightcurve(self.times, self.counts, gti=self.gti)
        lc2 = Lightcurve(self.times, self.counts, gti=gti)
        lc = lc1 + lc2
        np.testing.assert_almost_equal(lc.gti, [[0.5, 3.5]])

    def test_add_with_unequal_time_arrays(self):
        _times = [1, 3, 5, 7]

        with pytest.raises(ValueError):
            lc1 = Lightcurve(self.times, self.counts)
            lc2 = Lightcurve(_times, self.counts)

            lc = lc1 + lc2

    def test_add_with_equal_time_arrays(self):
        _counts = [1, 1, 1, 1]

        lc1 = Lightcurve(self.times, self.counts)
        lc2 = Lightcurve(self.times, _counts)

        lc = lc1 + lc2

        assert np.all(lc.counts == lc1.counts + lc2.counts)
        assert np.all(lc.countrate == lc1.countrate + lc2.countrate)
        assert lc1.mjdref == lc.mjdref

    def test_sub_with_diff_time_arrays(self):
        _times = [1.1, 2.1, 3.1, 4.1, 5.1]
        _counts = [2, 2, 2, 2, 2]

        with pytest.raises(ValueError):
            lc1 = Lightcurve(self.times, self.counts)
            lc2 = Lightcurve(_times, _counts)

            lc = lc1 - lc2

    def test_sub_with_different_err_dist(self):
        lc1 = Lightcurve(self.times, self.counts)
        lc2 = Lightcurve(self.times, self.counts, err=self.counts / 2,
                         err_dist="gauss")
        with warnings.catch_warnings(record=True) as w:
            lc = lc1 - lc2
            assert "ightcurves have different statistics" in str(w[0].message)

    def test_sub_with_different_mjdref(self):
        lc1 = Lightcurve(self.times, self.counts, gti=self.gti, mjdref=57000)
        lc2 = Lightcurve(self.times, self.counts, gti=self.gti, mjdref=57001)
        with pytest.raises(ValueError):
            lc1 - lc2

    def test_subtraction(self):
        _counts = [3, 4, 5, 6]

        lc1 = Lightcurve(self.times, self.counts)
        lc2 = Lightcurve(self.times, _counts)

        lc = lc2 - lc1

        expected_counts = np.array([1, 2, 3, 4])
        assert np.all(lc.counts == expected_counts)
        assert lc1.mjdref == lc.mjdref

    def test_negation(self):
        lc = Lightcurve(self.times, self.counts)

        _lc = lc + (-lc)

        assert not np.all(_lc.counts)
        assert _lc.mjdref == lc.mjdref

    def test_len_function(self):
        lc = Lightcurve(self.times, self.counts)

        assert len(lc) == 4

    def test_indexing_with_unexpected_type(self):
        lc = Lightcurve(self.times, self.counts)

        with pytest.raises(IndexError):
            count = lc['first']

    def test_indexing(self):
        lc = Lightcurve(self.times, self.counts)

        assert lc[0] == lc[1] == lc[2] == lc[3] == 2

    def test_slicing(self):
        lc = Lightcurve(self.times, self.counts)

        assert np.all(lc[1:3].counts == np.array([2, 2]))
        assert np.all(lc[:2].counts == np.array([2, 2]))
        assert np.all(lc[2:].counts == np.array([2, 2]))
        assert np.all(lc[:].counts == np.array([2, 2, 2, 2]))
        assert lc[:].mjdref == lc.mjdref


    def test_slicing_index_error(self):
        lc = Lightcurve(self.times, self.counts)

        with pytest.raises(StingrayError):
            lc_new = lc[1:2]

    def test_join_with_different_dt(self):
        _times = [5, 5.5, 6]
        _counts = [2, 2, 2]

        lc1 = Lightcurve(self.times, self.counts)
        lc2 = Lightcurve(_times, _counts)

        with warnings.catch_warnings(record=True) as w:
            lc1.join(lc2)
            assert "different bin widths" in str(w[0].message)

    def test_join_with_different_mjdref(self):
        lc1 = Lightcurve(self.times, self.counts, gti=self.gti, mjdref=57000)
        lc2 = Lightcurve(self.times, self.counts, gti=self.gti, mjdref=57001)
        with pytest.raises(ValueError):
            lc1.join(lc2)

    def test_join_disjoint_time_arrays(self):
        _times = [5, 6, 7, 8]
        _counts = [2, 2, 2, 2]

        lc1 = Lightcurve(self.times, self.counts)
        lc2 = Lightcurve(_times, _counts)

        lc = lc1.join(lc2)

        assert len(lc.counts) == len(lc.time) == 8
        assert np.all(lc.counts == 2)
        assert lc.mjdref == lc1.mjdref

    def test_join_overlapping_time_arrays(self):
        _times = [3, 4, 5, 6]
        _counts = [4, 4, 4, 4]

        lc1 = Lightcurve(self.times, self.counts)
        lc2 = Lightcurve(_times, _counts)

        with warnings.catch_warnings(record=True) as w:
            lc = lc1.join(lc2)
            assert "overlapping time ranges" in str(w[0].message)

        assert len(lc.counts) == len(lc.time) == 6
        assert np.all(lc.counts == np.array([2, 2, 3, 3, 4, 4]))

    def test_truncate_by_index(self):
        lc = Lightcurve(self.times, self.counts, gti=self.gti)

        lc1 = lc.truncate(start=1)
        assert np.all(lc1.time == np.array([2, 3, 4]))
        assert np.all(lc1.counts == np.array([2, 2, 2]))
        np.testing.assert_almost_equal(lc1.gti[0][0], 1.5)
        assert lc1.mjdref == lc.mjdref

        lc2 = lc.truncate(stop=2)
        assert np.all(lc2.time == np.array([1, 2]))
        assert np.all(lc2.counts == np.array([2, 2]))
        np.testing.assert_almost_equal(lc2.gti[-1][-1], 2.5)
        assert lc2.mjdref == lc.mjdref

    def test_truncate_by_time_stop_less_than_start(self):
        lc = Lightcurve(self.times, self.counts)

        with pytest.raises(ValueError):
            lc1 = lc.truncate(start=2, stop=1, method='time')

    def test_truncate_fails_with_incorrect_method(self):
        lc = Lightcurve(self.times, self.counts)
        with pytest.raises(ValueError):
            lc1 = lc.truncate(start=1, method="wrong")

    def test_truncate_by_time(self):
        lc = Lightcurve(self.times, self.counts, gti=self.gti)

        lc1 = lc.truncate(start=1, method='time')
        assert np.all(lc1.time == np.array([1, 2, 3, 4]))
        assert np.all(lc1.counts == np.array([2, 2, 2, 2]))
        np.testing.assert_almost_equal(lc1.gti[0][0], 0.5)
        assert lc1.mjdref == lc.mjdref

        lc2 = lc.truncate(stop=3, method='time')
        assert np.all(lc2.time == np.array([1, 2]))
        assert np.all(lc2.counts == np.array([2, 2]))
        np.testing.assert_almost_equal(lc2.gti[-1][-1], 2.5)
        assert lc2.mjdref == lc.mjdref

    def test_sort(self):
        _times = [1, 2, 3, 4]
        _counts = [40, 10, 20, 5]
        lc = Lightcurve(_times, _counts, mjdref=57000)
        mjdref = lc.mjdref

        lc.sort()

        assert np.all(lc.counts == np.array([5, 10, 20, 40]))
        assert np.all(lc.time == np.array([4, 2, 3, 1]))
        assert lc.mjdref == mjdref

        lc.sort(reverse=True)

        assert np.all(lc.counts == np.array([40, 20, 10,  5]))
        assert np.all(lc.time == np.array([1, 3, 2, 4]))
        assert lc.mjdref == mjdref

    def test_plot_matplotlib_not_installed(self):
        try:
            import matplotlib.pyplot as plt
        except Exception as e:

            lc = Lightcurve(self.times, self.counts)
            try:
                lc.plot()
            except Exception as e:
                assert type(e) is ImportError
                assert str(e) == "Matplotlib required for plot()"

    def test_plot_simple(self):
        lc = Lightcurve(self.times, self.counts)
        lc.plot()
        assert plt.fignum_exists(1)

    def test_plot_wrong_label_type(self):
        lc = Lightcurve(self.times, self.counts)

        with pytest.raises(TypeError):
            with warnings.catch_warnings(record=True) as w:
                lc.plot(labels=123)
                assert "must be either a list or tuple" in str(w[0].message)

    def test_plot_labels_index_error(self):
        lc = Lightcurve(self.times, self.counts)
        with warnings.catch_warnings(record=True) as w:
            lc.plot(labels=('x'))
            assert "must have two labels" in str(w[0].message)

    def test_plot_default_filename(self):
        lc = Lightcurve(self.times, self.counts)
        lc.plot(save=True)
        assert os.path.isfile('out.png')
        os.unlink('out.png')

    def test_plot_custom_filename(self):
        lc = Lightcurve(self.times, self.counts)
        lc.plot(save=True, filename='lc.png')
        assert os.path.isfile('lc.png')
        os.unlink('lc.png')

    def test_plot_axis(self):
        lc = Lightcurve(self.times, self.counts)
        lc.plot(axis=[0, 1, 0, 100])
        assert plt.fignum_exists(1)

    def test_plot_title(self):
        lc = Lightcurve(self.times, self.counts)
        lc.plot(title="Test Lightcurve")
        assert plt.fignum_exists(1)

    def test_io_with_ascii(self):
        lc = Lightcurve(self.times, self.counts)
        lc.write('ascii_lc.txt', format_='ascii')
        lc.read('ascii_lc.txt', format_='ascii')
        os.remove('ascii_lc.txt')

    def test_io_with_pickle(self):
        lc = Lightcurve(self.times, self.counts)
        lc.write('lc.pickle', format_='pickle')
        lc.read('lc.pickle', format_='pickle')
        assert np.all(lc.time == self.times)
        assert np.all(lc.counts == self.counts)
        assert np.all(lc.gti == self.gti)
        os.remove('lc.pickle')

    def test_io_with_hdf5(self):
        lc = Lightcurve(self.times, self.counts)
        lc.write('lc.hdf5', format_='hdf5')

        if _H5PY_INSTALLED:
            data = lc.read('lc.hdf5', format_='hdf5')
            assert np.all(data['time'] == self.times)
            assert np.all(data['counts'] == self.counts)
            assert np.all(data['gti'] == self.gti)
            os.remove('lc.hdf5')

        else:
            lc.read('lc.pickle', format_='pickle')
            assert np.all(lc.time == self.times)
            assert np.all(lc.counts == self.counts)
            assert np.all(lc.gti == self.gti)
            os.remove('lc.pickle')

    def test_split_lc_by_gtis(self):
        times = [1, 2, 3, 4, 5, 6, 7, 8]
        counts = [1, 1, 1, 1, 2, 3, 3, 2]
        gti = [[0.5, 4.5], [5.5, 7.5]]

        lc = Lightcurve(times, counts, gti=gti)
        list_of_lcs = lc.split_by_gti()
        lc0 = list_of_lcs[0]
        lc1 = list_of_lcs[1]
        assert np.all(lc0.time == [1, 2, 3, 4])
        assert np.all(lc1.time == [6, 7])
        assert np.all(lc0.counts == [1, 1, 1, 1])
        assert np.all(lc1.counts == [3, 3])
        assert np.all(lc0.gti == [[0.5, 4.5]])
        assert np.all(lc1.gti == [[5.5, 7.5]])

    def test_shift(self):
        times = [1, 2, 3, 4, 5, 6, 7, 8]
        counts = [1, 1, 1, 1, 2, 3, 3, 2]
        lc = Lightcurve(times, counts, input_counts=True)
        lc2 = lc.shift(1)
        assert np.all(lc2.time - 1 == times)
        lc2 = lc.shift(-1)
        assert np.all(lc2.time + 1 == times)
        assert np.all(lc2.counts == lc.counts)
        assert np.all(lc2.countrate == lc.countrate)
        lc = Lightcurve(times, counts, input_counts=False)
        lc2 = lc.shift(1)
        assert np.all(lc2.counts == lc.counts)
        assert np.all(lc2.countrate == lc.countrate)


class TestLightcurveRebin(object):

    @classmethod
    def setup_class(cls):
        dt = 0.0001220703125
        n = 1384132
        mean_counts = 2.0
        times = np.arange(dt/2, dt/2 + n*dt, dt)
        counts = np.zeros_like(times) + mean_counts
        cls.lc = Lightcurve(times, counts)

    def test_rebin_even(self):
        dt_new = 2.0
        lc_binned = self.lc.rebin(dt_new)
        assert np.isclose(lc_binned.dt, dt_new)
        counts_test = np.zeros_like(lc_binned.time) + \
            self.lc.counts[0]*dt_new/self.lc.dt
        assert np.allclose(lc_binned.counts, counts_test)

    def test_rebin_odd(self):
        dt_new = 1.5
        lc_binned = self.lc.rebin(dt_new)
        assert np.isclose(lc_binned.dt, dt_new)

        counts_test = np.zeros_like(lc_binned.time) + \
            self.lc.counts[0]*dt_new/self.lc.dt
        assert np.allclose(lc_binned.counts, counts_test)


    @pytest.mark.parametrize(("dt"), [2, 3, np.pi, 5])
    def rebin_several(self, dt):
        lc_binned = self.lc.rebin(dt)
        assert len(lc_binned.time) == len(lc_binned.counts)

    def test_rebin_equal_numbers(self):
        dt_all = [2, 3, np.pi, 5]
        for dt in dt_all:
            self.rebin_several(dt)

    def test_lc_baseline(self):
        times = np.arange(0, 100, 0.01)
        counts = np.random.normal(100, 0.1, len(times)) + \
            0.001 * times
        gti = [[-0.005, 50.005], [59.005, 100.005]]
        good = create_gti_mask(times, gti)
        counts[np.logical_not(good)] = 0
        lc = Lightcurve(times, counts, gti=gti)
        baseline = lc.baseline(10000, 0.01)
        assert np.all(lc.counts - baseline < 1)

    def test_change_mjdref(self):
        lc_new = self.lc.change_mjdref(57000)
        assert lc_new.mjdref == 57000