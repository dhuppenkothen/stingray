
import numpy as np
import stingray.utils as utils

np.random.seed(20150907)

class TestRebinData(object):

    def setUp(self):
        self.dx = 1.0
        self.n = 10
        self.counts = 2.0
        self.x = np.arange(self.dx/2, self.dx/2+self.n*self.dx, self.dx)
        self.y = np.zeros_like(self.x)+self.counts

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

    def test_new_bins(self):
        dx_new = 2.0
        xbin_test = np.arange(self.x[0]-self.dx/2., self.x[-1]+self.dx/2., \
                              dx_new) + dx_new/2.

        xbin, ybin, step_size = utils.rebin_data(self.x, self.y, dx_new)
        assert np.allclose(xbin, xbin_test)

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

        ybin_test = np.zeros_like(xbin) + self.counts*dx_new/self.dx
        assert np.allclose(ybin_test, ybin)

