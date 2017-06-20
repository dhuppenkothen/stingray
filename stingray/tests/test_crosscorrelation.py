import numpy as np

import pytest
import matplotlib.pyplot as plt
import warnings
import os

from stingray import Lightcurve
from stingray import CrossCorrelation
from stingray.exceptions import StingrayError


class TestCrossCorrelation(object):

    @classmethod
    def setup_class(cls):
        cls.lc1 = Lightcurve([1, 2, 3, 4, 5], [2, 3, 2, 4, 1])
        cls.lc2 = Lightcurve([1, 2, 3, 4, 5], [4, 8, 1, 9, 11])
        # Smaller Light curve
        cls.lc_s = Lightcurve([1, 2, 3], [5, 3, 2])
        # lc with different time resolution
        cls.lc_u = Lightcurve([1, 3, 5, 7, 9], [4, 8, 1, 9, 11])

    def test_empty_cross_correlation(self):
        cr = CrossCorrelation()
        assert cr.corr is None
        assert cr.time_shift is None
        assert cr.time_lags is None
        assert cr.dt is None

    def test_empty_cross_correlation_with_dt(self):
        cr = CrossCorrelation()
        with pytest.raises(StingrayError):
            cr.cal_timeshift(dt = 2.0)
        assert cr.dt == 2.0

    def test_cross_correlation_with_unequal_lc(self):
        with pytest.raises(StingrayError):
            cr = CrossCorrelation(self.lc1,self.lc_s)

    def test_init_with_invalid_lc1(self):
        data = np.array([[2,3,2,4,1]])
        with pytest.raises(TypeError):
            cr = CrossCorrelation(data,self.lc2)

    def test_init_with_invalid_lc2(self):
        data = np.array([[2, 3, 2, 4, 1]])
        with pytest.raises(TypeError):
            cr = CrossCorrelation(self.lc1,data)

    def test_init_with_diff_time_bin(self):
        with pytest.raises(StingrayError):
            cr = CrossCorrelation(self.lc_u, self.lc2)


    def test_corr_is_correct(self):
        result = np.array([22, 51, 51, 81, 81, 41, 41, 24, 4])
        lags_result = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        cr = CrossCorrelation(self.lc1, self.lc2)
        assert np.array_equal(cr.corr,result)
        assert cr.dt == self.lc1.dt
        assert cr.n == 9
        assert np.array_equal(cr.time_lags,lags_result)
        assert cr.time_shift == -1.0

    def test_plot_matplotlib_not_installed(self):
        try:
            import matplotlib.pyplot as plt
        except Exception as e:

            cr = CrossCorrelation(self.lc1, self.lc2)
            try:
                cr.plot()
            except Exception as e:
                assert type(e) is ImportError
                assert str(e) == "Matplotlib required for plot()"

    def test_simple_plot(self):
        cr = CrossCorrelation(self.lc1, self.lc2)
        cr.plot()
        assert plt.fignum_exists(1)

    def test_plot_wrong_label_type(self):
        cr = CrossCorrelation(self.lc1, self.lc2)

        with pytest.raises(TypeError):
            with warnings.catch_warnings(record=True) as w:
                cr.plot(labels=123)
                assert "must be either a list or tuple" in str(w[0].message)

    def test_plot_labels_index_error(self):
        cr = CrossCorrelation(self.lc1, self.lc2)
        with warnings.catch_warnings(record=True) as w:
            cr.plot(labels=('x'))
            assert "must have two labels" in str(w[0].message)

    def test_plot_axis(self):
        cr = CrossCorrelation(self.lc1, self.lc2)
        cr.plot(axis=[0, 1, 0, 100])
        assert plt.fignum_exists(1)

    def test_plot_title(self):
        cr = CrossCorrelation(self.lc1, self.lc2)
        cr.plot(title="Test for Cross Correlation")
        assert plt.fignum_exists(1)

    def test_plot_default_filename(self):
        cr = CrossCorrelation(self.lc1, self.lc2)
        cr.plot(save=True)
        assert os.path.isfile('corr.png')
        os.unlink('corr.png')

    def test_plot_custom_filename(self):
        cr = CrossCorrelation(self.lc1, self.lc2)
        cr.plot(save=True, filename='cr.png')
        assert os.path.isfile('cr.png')
        os.unlink('cr.png')
