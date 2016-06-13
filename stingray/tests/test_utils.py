
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

    def test_new_stepsize(self):
        dx_new = 2.0
        xbin, ybin, step_size = utils.rebin_data(self.x, self.y, dx_new)
        assert step_size == dx_new/self.dx

    def test_arrays(self):
        dx_new = 2.0
        xbin, ybin, step_size = utils.rebin_data(self.x, self.y, dx_new)
        assert isinstance(xbin, np.ndarray)
        assert isinstance(ybin, np.ndarray)

    def test_length_matches(self):
        dx_new = 2.0
        xbin, ybin, step_size = utils.rebin_data(self.x, self.y, dx_new)
        assert xbin.shape[0] == ybin.shape[0]

    def test_binned_counts(self):
        dx_new = 2.0

        xbin, ybin, step_size = utils.rebin_data(self.x, self.y, dx_new)

        ybin_test = np.zeros_like(xbin) + self.counts*dx_new/self.dx
        assert np.allclose(ybin, ybin_test)

    def test_uneven_bins(self):
        dx_new = 1.5
        xbin, ybin, step_size = utils.rebin_data(self.x, self.y, dx_new)
        assert np.isclose(xbin[1]-xbin[0], dx_new)

    def test_uneven_binned_counts(self):
        dx_new = 1.5
        xbin, ybin, step_size = utils.rebin_data(self.x, self.y, dx_new)
        print(xbin)
        print(ybin)
        ybin_test = np.zeros_like(xbin) + self.counts*dx_new/self.dx
        assert np.allclose(ybin_test, ybin)

    def test_rebin_data_should_raise_error_when_method_is_different_than_allowed(self):
        dx_new = 2.0
        try:
            utils.rebin_data(self.x, self.y, dx_new, method='not_allowed_method')
        except utils.UnrecognizedMethod:
            pass
        else:
            raise AssertionError("Expected exception does not occurred")


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

