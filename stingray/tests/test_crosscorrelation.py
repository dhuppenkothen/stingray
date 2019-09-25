import numpy as np

import pytest
import warnings
import os

from stingray import Lightcurve
from stingray.crosscorrelation import CrossCorrelation, AutoCorrelation
from stingray.exceptions import StingrayError

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


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
        assert cr.lc1 is None
        assert cr.lc2 is None
        assert cr.corr is None
        assert cr.time_shift is None
        assert cr.time_lags is None
        assert cr.dt is None
        assert cr.n is None
        assert cr.auto is False

    def test_empty_cross_correlation_with_dt(self):
        cr = CrossCorrelation()
        with pytest.raises(StingrayError):
            cr.cal_timeshift(dt=2.0)
        assert np.isclose(cr.dt, 2.0)

    def test_init_with_only_one_lc(self):
        with pytest.raises(TypeError):
            CrossCorrelation(self.lc1)

    def test_init_with_invalid_lc1(self):
        data = np.array([[2, 3, 2, 4, 1]])
        with pytest.raises(TypeError):
            CrossCorrelation(data, self.lc2)

    def test_init_with_invalid_lc2(self):
        data = np.array([[2, 3, 2, 4, 1]])
        with pytest.raises(TypeError):
            CrossCorrelation(self.lc1, data)

    def test_init_with_diff_time_bin(self):
        with pytest.raises(StingrayError):
            CrossCorrelation(self.lc_u, self.lc2)

    def test_corr_is_correct(self):
        result = np.array([1.92, 2.16, 1.8, -14.44, 11.12])
        lags_result = np.array([-2, -1, 0, 1, 2])
        cr = CrossCorrelation(self.lc1, self.lc2)
        assert np.allclose(cr.lc1, self.lc1)
        assert np.allclose(cr.lc2, self.lc2)
        assert np.allclose(cr.corr, result)
        assert np.isclose(cr.dt, self.lc1.dt)
        assert cr.n == 5
        assert np.allclose(cr.time_lags, lags_result)
        assert np.isclose(cr.time_shift, 2.0)
        assert cr.mode == 'same'
        assert cr.auto is False

    def test_crossparam_input(self):
        # need to create new results to check against
        spec = Crossspectrum(self.lc1, self.lc2)
        ifft = abs(scipy.fftpack.ifft(spec.power).real)
        time = scipy.fftpack.fftfreq(len(ifft), spec.df)
        time, resultifft = (list(t) for t in zip(*sorted(zip(time, ifft))))
        cr2 = CrossCorrelation(cross=spec)

        assert np.allclose(cr2.cross.power, spec.power)
        assert np.allclose(cr2.cross.freq, spec.freq)
        assert np.allclose(cr2.corr, resultifft)
        assert np.isclose(cr2.dt, self.lc1.dt)
        assert cr2.n == 5
        assert np.allclose(cr2.time_lags, lags_result)
        assert cr2.mode == 'same'
        assert cr2.auto is False

    def test_cross_correlation_with_unequal_lc(self):
        result = np.array([-0.66666667, -0.33333333, -1., 0.66666667, 3.13333333])
        lags_result = np.array([-2, -1, 0, 1, 2])
        cr = CrossCorrelation(self.lc1, self.lc_s)
        assert np.allclose(cr.lc1, self.lc1)
        assert np.allclose(cr.lc2, self.lc_s)
        assert np.allclose(cr.corr, result)
        assert np.isclose(cr.dt, self.lc1.dt)
        assert cr.n == 5
        assert np.allclose(cr.time_lags, lags_result)
        assert np.isclose(cr.time_shift, 2.0)
        assert cr.mode == 'same'
        assert cr.auto is False

    def test_mode_with_bad_input(self):
        with pytest.raises(TypeError):
            CrossCorrelation(self.lc1, self.lc2, mode=123)

    def test_mode_with_wrong_input(self):
        with pytest.raises(ValueError):
            CrossCorrelation(self.lc1, self.lc2, mode='default')

    def test_full_mode_is_correct(self):
        result = np.array([-1.76, 1.68, 1.92, 2.16, 1.8, -14.44, 11.12, -6.12, 3.64])
        lags_result = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        cr = CrossCorrelation(self.lc1, self.lc2, mode='full')
        assert np.allclose(cr.lc1, self.lc1)
        assert np.allclose(cr.lc2, self.lc2)
        assert cr.mode == 'full'
        assert cr.n == 9
        assert np.allclose(cr.corr, result)
        assert np.allclose(cr.time_lags, lags_result)
        assert np.isclose(cr.time_shift, 2.0)
        assert cr.auto is False

    def test_timeshift_with_no_corr_but_lc_assigned(self):
        result = np.array([1.92, 2.16, 1.8, -14.44, 11.12])
        lags_result = np.array([-2, -1, 0, 1, 2])
        cr = CrossCorrelation()
        cr.lc1 = self.lc1
        cr.lc2 = self.lc2
        cr.cal_timeshift(dt=1.0)
        assert np.allclose(cr.lc1, self.lc1)
        assert np.allclose(cr.lc2, self.lc2)
        assert np.allclose(cr.corr, result)
        assert np.isclose(cr.dt, self.lc1.dt)
        assert cr.n == 5
        assert np.allclose(cr.time_lags, lags_result)
        assert np.isclose(cr.time_shift, 2.0)
        assert cr.mode == 'same'
        assert cr.auto is False

    def test_timeshift_with_corr_and_lc_assigned(self):
        result = np.array([1.92, 2.16, 1.8, -14.44, 11.12])
        lags_result = np.array([-2, -1, 0, 1, 2])
        cr = CrossCorrelation()
        cr.lc1 = self.lc1
        cr.lc2 = self.lc2
        cr.corr = result
        cr.cal_timeshift(dt=1.0)
        assert np.allclose(cr.lc1, self.lc1)
        assert np.allclose(cr.lc2, self.lc2)
        assert np.allclose(cr.corr, result)
        assert np.isclose(cr.dt, self.lc1.dt)
        assert cr.n == 5
        assert np.allclose(cr.time_lags, lags_result)
        assert np.isclose(cr.time_shift, 2.0)
        assert cr.mode == 'same'
        assert cr.auto is False

    @pytest.mark.skipif(HAS_MPL, reason='Matplotlib is already installed if condition is met')
    def test_plot_matplotlib_not_installed(self):
        cr = CrossCorrelation(self.lc1, self.lc2)
        with pytest.raises(ImportError) as excinfo:
            cr.plot()
        message = str(excinfo.value)
        assert "Matplotlib required for plot()" in message

    @pytest.mark.skipif(not HAS_MPL, reason='Matplotlib is not installed')
    def test_simple_plot(self):
        cr = CrossCorrelation(self.lc1, self.lc2)
        cr.plot()
        assert plt.fignum_exists(1)

    @pytest.mark.skipif(not HAS_MPL, reason='Matplotlib is not installed')
    def test_plot_wrong_label_type(self):
        cr = CrossCorrelation(self.lc1, self.lc2)
        with pytest.raises(TypeError):
            with warnings.catch_warnings(record=True) as w:
                cr.plot(labels=123)
                assert np.any(["must be either a list or tuple" in str(wi.message) for wi in w])

    @pytest.mark.skipif(not HAS_MPL, reason='Matplotlib is not installed')
    def test_plot_labels_index_error(self):
        cr = CrossCorrelation(self.lc1, self.lc2)
        with warnings.catch_warnings(record=True) as w:
            cr.plot(labels='x')
            assert np.any(["must have two labels" in str(wi.message) for wi in w])

    @pytest.mark.skipif(not HAS_MPL, reason='Matplotlib is not installed')
    def test_plot_axis(self):
        cr = CrossCorrelation(self.lc1, self.lc2)
        cr.plot(axis=[0, 1, 0, 100])
        assert plt.fignum_exists(1)

    @pytest.mark.skipif(not HAS_MPL, reason='Matplotlib is not installed')
    def test_plot_title(self):
        cr = CrossCorrelation(self.lc1, self.lc2)
        cr.plot(title="Test for Cross Correlation")
        assert plt.fignum_exists(1)

    @pytest.mark.skipif(not HAS_MPL, reason='Matplotlib is not installed')
    def test_plot_default_filename(self):
        cr = CrossCorrelation(self.lc1, self.lc2)
        cr.plot(save=True, title="Correlation")
        assert os.path.isfile('corr.pdf')
        os.unlink('corr.pdf')

    @pytest.mark.skipif(not HAS_MPL, reason='Matplotlib is not installed')
    def test_plot_custom_filename(self):
        cr = CrossCorrelation(self.lc1, self.lc2)
        cr.plot(save=True, filename='cr.png')
        assert os.path.isfile('cr.png')
        os.unlink('cr.png')

    def test_auto_correlation(self):
        result = np.array([1.68, -3.36, 5.2, -3.36, 1.68])
        lags_result = np.array([-2, -1, 0, 1, 2])
        ac = AutoCorrelation(self.lc1)
        assert np.allclose(ac.lc1, self.lc1)
        assert np.allclose(ac.lc2, self.lc1)
        assert np.allclose(ac.corr, result)
        assert np.isclose(ac.dt, self.lc1.dt)
        assert ac.n == 5
        assert np.allclose(ac.time_lags, lags_result)
        assert np.isclose(ac.time_shift, 0.0)
        assert ac.mode == 'same'
        assert ac.auto is True

    def test_auto_correlation_with_full_mode(self):
        result = np.array([0.56, -1.48, 1.68, -3.36, 5.2, -3.36, 1.68, -1.48, 0.56])
        lags_result = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        ac = AutoCorrelation(self.lc1, mode='full')
        assert np.allclose(ac.lc1, self.lc1)
        assert np.allclose(ac.lc2, self.lc1)
        assert np.allclose(ac.corr, result)
        assert np.isclose(ac.dt, self.lc1.dt)
        assert ac.n == 9
        assert np.allclose(ac.time_lags, lags_result)
        assert np.isclose(ac.time_shift, 0.0)
        assert ac.mode == 'full'
        assert ac.auto is True
