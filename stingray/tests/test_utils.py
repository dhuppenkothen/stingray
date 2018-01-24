
from astropy.tests.helper import pytest
import numpy as np
import stingray.utils as utils

np.random.seed(20150907)


class TestRebinData(object):

    @classmethod
    def setup_class(cls):
        cls.dx = 1.0
        cls.n = 10
        cls.counts = 2.0
        cls.x = np.arange(cls.dx/2, cls.dx/2+cls.n*cls.dx, cls.dx)
        cls.y = np.zeros_like(cls.x)+cls.counts
        cls.yerr = np.sqrt(cls.y)

    def test_new_stepsize(self):
        dx_new = 2.0
        xbin, ybin, yerr, step_size = utils.rebin_data(self.x, self.y, dx_new,
                                                       self.yerr)
        assert step_size == dx_new/self.dx

    def test_arrays(self):
        dx_new = 2.0
        xbin, ybin, yerr, step_size = utils.rebin_data(self.x, self.y, dx_new,
                                                       self.yerr)
        assert isinstance(xbin, np.ndarray)
        assert isinstance(ybin, np.ndarray)

    def test_length_matches(self):
        dx_new = 2.0
        xbin, ybin, yerr, step_size = utils.rebin_data(self.x, self.y, dx_new,
                                                       self.yerr)
        assert xbin.shape[0] == ybin.shape[0]

    def test_binned_counts(self):
        dx_new = 2.0

        xbin, ybin, yerr, step_size = utils.rebin_data(self.x, self.y, dx_new,
                                                       self.yerr)

        ybin_test = np.zeros_like(xbin) + self.counts*dx_new/self.dx
        assert np.allclose(ybin, ybin_test)

    def test_uneven_bins(self):
        dx_new = 1.5
        xbin, ybin, yerr, step_size = utils.rebin_data(self.x, self.y, dx_new,
                                                       self.yerr)
        assert np.isclose(xbin[1]-xbin[0], dx_new)

    def test_uneven_binned_counts(self):
        dx_new = 1.5
        xbin, ybin, yerr, step_size = utils.rebin_data(self.x, self.y, dx_new,
                                                       self.yerr)
        ybin_test = np.zeros_like(xbin) + self.counts*dx_new/self.dx
        assert np.allclose(ybin_test, ybin)

    def test_rebin_data_should_raise_error_when_method_is_different_than_allowed(self):
        dx_new = 2.0
        with pytest.raises(ValueError):
            utils.rebin_data(self.x, self.y, dx_new, self.yerr,
                             method='not_allowed_method')


class TestUtils(object):

    def test_optimal_bin_time(self):
        assert utils.optimal_bin_time(512, 2.1) == 2
        assert utils.optimal_bin_time(512, 3.9) == 2
        assert utils.optimal_bin_time(512, 4.1) == 4

    def test_order_list_of_arrays(self):
        alist = [np.array([1, 0]), np.array([2, 3])]

        order = np.argsort(alist[0])
        assert np.all(np.array([np.array([0, 1]), np.array([3, 2])]) ==
                      np.array(utils.order_list_of_arrays(alist, order)))

        alist = {"a": np.array([1, 0]), "b": np.array([2, 3])}
        alist_new = utils.order_list_of_arrays(alist, order)
        assert np.all(np.array([0, 1]) == alist_new["a"])
        assert np.all(np.array([3, 2]) == alist_new["b"])

        alist = 0
        assert utils.order_list_of_arrays(alist, order) is None

    def test_look_for_array(self):
        assert utils.look_for_array_in_array(np.arange(2), np.arange(1, 3))
        assert not utils.look_for_array_in_array(np.arange(2),
                                                 np.arange(2, 4))

    def test_assign_value_if_none(self):
        assert utils.assign_value_if_none(None, 2) == 2
        assert utils.assign_value_if_none(1, 2) == 1

    def test_contiguous(self):
        """A more complicated example of intersection of GTIs."""
        array = np.array([0, 1, 1, 0, 1, 1, 1], dtype=bool)
        cont = utils.contiguous_regions(array)
        assert np.all(cont == np.array([[1, 3], [4, 7]])), \
            'Contiguous region wrong'

    def test_get_random_state(self):
        # Life, Universe and Everything
        lue = 42
        random_state = np.random.RandomState(lue)

        assert utils.get_random_state(None) is np.random.mtrand._rand
        assert np.all(utils.get_random_state(lue).randn(lue) == np.random.RandomState(lue).randn(lue))
        assert np.all(utils.get_random_state(np.random.RandomState(lue)).randn(lue) == np.random.RandomState(lue).randn(lue))

        with pytest.raises(ValueError):
            utils.get_random_state('foobar')

class TestCreateWindow(object):
    @classmethod
    def setup_class(cls):
        cls.N = 5
        cls.uniform_window = 'uniform'
        cls.parzen_window = 'parzen'
        cls.hamming_window = 'hamming'
        cls.hanning_window = 'hanning'
        cls.triangular_window = 'triangular'
        cls.welch_window = 'welch'
        cls.blackmann_window = 'blackmann'
        cls.flattop_window = 'flat-top'

    def test_bad_N(self):
        N_bad = 'abc'
        with pytest.raises(TypeError):
            window = utils.create_window(N_bad, self.uniform_window)

    def test_bad_window_type(self):
        window_bad = 123
        with pytest.raises(TypeError):
            window = utils.create_window(self.N, window_bad)

    def test_not_available_window(self):
        window_not = 'kaiser'
        with pytest.raises(ValueError):
            window = utils.create_window(self.N, window_not)

    def test_N_equals_zero(self):
        N = 0
        window = utils.create_window(N)
        assert len(window) == 0

    def test_uniform_window(self):
        result = np.ones(self.N)
        window = utils.create_window(self.N)
        assert np.allclose(window, result)

    def test_parzen_window(self):
        result = np.array([0, 0.25, 1, 0.25, 0])
        window = utils.create_window(self.N, self.parzen_window)
        assert np.allclose(window, result)

    def test_hamming_window(self):
        result = np.array([0.08, 0.54, 1, 0.54, 0.08])
        window = utils.create_window(self.N, self.hamming_window)
        assert np.allclose(window, result)

    def test_hanning_window(self):
        result = np.array([0, 0.5, 1, 0.5, 0])
        window = utils.create_window(self.N, self.hanning_window)
        assert np.allclose(window, result)

    def test_triangular_window(self):
        result = np.array([0.6, 0.8, 1, 0.8, 0.6])
        window = utils.create_window(self.N, self.triangular_window)
        assert np.allclose(window, result)

    def test_welch_window(self):
        result = np.array([0, 0.75, 1, 0.75, 0])
        window = utils.create_window(self.N, self.welch_window)
        assert np.allclose(window, result)

    def test_blackmann_window(self):
        result = np.array([0.006879, 0.349741, 0.999999, 0.349741, 0.006879])
        window = utils.create_window(self.N, self.blackmann_window)
        assert np.allclose(window, result)

    def test_flat_top_window(self):
        result = np.array([8.67361738e-17, -2.62000000e-01, 4.63600000e+00, -2.62000000e-01,
                           8.67361738e-17])
        window = utils.create_window(self.N, self.flattop_window)
        assert np.allclose(window, result)
