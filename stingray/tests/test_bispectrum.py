import numpy as np

from astropy.tests.helper import pytest
import warnings
import os

from stingray import Lightcurve
from stingray.bispectrum import Bispectrum
from stingray.exceptions import StingrayError

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class TestBispectrum(object):
    @classmethod
    def setup_class(cls):
        cls.lc = Lightcurve([1, 2, 3, 4, 5], [2, 3, 2, 4, 1])
        cls.lc1 = Lightcurve([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [2, 1, 3, 1, 4, 2])

    def test_create_bispectrum(self):
        """
        Demonstrate that we can create a Bispectrum object.
        """
        Bispectrum(self.lc)

    def test_wrong_lc(self):
        lc = [1, 2, 3, 4]
        with pytest.raises(TypeError):
            Bispectrum(lc)

    def test_wrong_maxlag(self):
        with pytest.raises(ValueError):
            Bispectrum(self.lc, maxlag='123')

    def test_maxlag_none(self):
        bs = Bispectrum(self.lc)
        assert bs.maxlag == np.int(self.lc.n / 2)

    def test_neg_maxlag(self):
        bs = Bispectrum(self.lc, maxlag=-2)
        assert bs.maxlag == 2

    def test_bispectrum_with_none_maxlag(self):
        bs = Bispectrum(self.lc)
        lags = np.array([-2, -1, 0, 1, 2])
        freq = np.array([-0.5, -0.25, 0., 0.25, 0.5])
        cum3 = np.array([[0.0576, 0.1216, 0.1376, -0.2176, -0.0448],
                         [0.1216, -0.6752, 0.4128, 0.1216, -0.0096],
                         [0.1376, 0.4128, 0.288, -0.6752, 0.0576],
                         [-0.2176, 0.1216, -0.6752, 0.4128, 0.1216],
                         [-0.0448, -0.0096, 0.0576, 0.1216, 0.1376]])

        bispec = np.array([[1.26572936 - 1.96410531e+00j, 1.01680863 - 2.33088331e+00j,
                            0.53357150 - 3.20079088e-01j, -0.39280863 + 1.79793854e+00j,
                            0.37973002 + 0.00000000e+00j],
                           [1.01680863 - 2.33088331e+00j, -0.29772936 - 1.20447928e+00j,
                            -0.11757150 - 7.55604229e-02j, 0.03626998 + 1.11022302e-16j,
                            -0.39280863 - 1.79793854e+00j],
                           [0.53357150 - 3.20079088e-01j, -0.11757150 - 7.55604229e-02j,
                            0.27200000 + 0.00000000e+00j, -0.11757150 + 7.55604229e-02j,
                            0.53357150 + 3.20079088e-01j],
                           [-0.39280863 + 1.79793854e+00j, 0.03626998 - 1.11022302e-16j,
                            -0.11757150 + 7.55604229e-02j, -0.29772936 + 1.20447928e+00j,
                            1.01680863 + 2.33088331e+00j],
                           [0.37973002 + 0.00000000e+00j, -0.39280863 - 1.79793854e+00j,
                            0.53357150 + 3.20079088e-01j, 1.01680863 + 2.33088331e+00j,
                            1.26572936 + 1.96410531e+00j]])

        bispec_mag = np.array([[2.33661732, 2.54301333, 0.62221312, 1.84034823, 0.37973002],
                               [2.54301333, 1.24073087, 0.13975849, 0.03626998, 1.84034823],
                               [0.62221312, 0.13975849, 0.272, 0.13975849, 0.62221312],
                               [1.84034823, 0.03626998, 0.13975849, 1.24073087, 2.54301333],
                               [0.37973002, 1.84034823, 0.62221312, 2.54301333, 2.33661732]])

        bispec_phase = np.array([[-9.98346367e-01, -1.15944968e+00, -5.40331561e-01, 1.78589369e+00,
                                  0.00000000e+00],
                                 [-1.15944968e+00, -1.81312395e+00, -2.57038310e+00, 3.06099713e-15,
                                  -1.78589369e+00],
                                 [-5.40331561e-01, -2.57038310e+00, 0.00000000e+00, 2.57038310e+00, 5.40331561e-01],
                                 [1.78589369e+00, -3.06099713e-15, 2.57038310e+00, 1.81312395e+00,
                                  1.15944968e+00],
                                 [0.00000000e+00, -1.78589369e+00, 5.40331561e-01, 1.15944968e+00,
                                  9.98346367e-01]])
        assert bs.lc == self.lc
        assert np.isclose(bs.fs, 1)
        assert bs.maxlag == 2
        assert bs.n == 5
        assert bs.scale == 'biased'
        assert np.allclose(bs.lags, lags)
        assert np.allclose(bs.freq, freq)
        assert np.allclose(bs.cum3, cum3)
        assert np.allclose(bs.bispec, bispec)
        assert np.allclose(bs.bispec_mag, bispec_mag)
        assert np.allclose(bs.bispec_phase, bispec_phase)

    def test_wrong_scale_type(self):
        with pytest.raises(TypeError):
            Bispectrum(self.lc, scale=1)

    def test_wrong_scale_value(self):
        with pytest.raises(ValueError):
            Bispectrum(self.lc, scale='non-biased')

    def test_bispectrum_unbiased_scale(self):
        bs = Bispectrum(self.lc, scale='unbiased')

        lags = np.array([-2, -1, 0, 1, 2])
        freq = np.array([-0.5, -0.25, 0., 0.25, 0.5])

        cum3 = np.array([[0.096, 0.20266667, 0.22933333, - 0.544, - 0.224],
                         [0.20266667, - 0.844, 0.516, 0.20266667, - 0.024],
                         [0.22933333, 0.516, 0.288, - 0.844, 0.096],
                         [-0.544, 0.20266667, - 0.844, 0.516, 0.20266667],
                         [-0.224, - 0.024, 0.096, 0.20266667, 0.22933333]])

        bispec = np.array([[1.78211085 - 2.10567238e+00j, 1.56502881 - 3.02261738e+00j,
                            0.63208770 - 8.00197720e-01j, - 1.20769548 + 2.56558544e+00j,
                            0.49792363 + 2.22044605e-16j],
                           [1.56502881 - 3.02261738e+00j, - 1.12477752 - 1.08193727e+00j,
                            0.12524563 - 1.88901057e-01j, 0.25940971 + 2.22044605e-16j,
                            - 1.20769548 - 2.56558544e+00j],
                           [0.63208770 - 8.00197720e-01j,
                            0.12524563 - 1.88901057e-01j,
                            - 0.08800000 + 0.00000000e+00j,
                            0.12524563 + 1.88901057e-01j,
                            0.63208770 + 8.00197720e-01j],
                           [-1.20769548 + 2.56558544e+00j, 0.25940971 - 2.22044605e-16j,
                            0.12524563 + 1.88901057e-01j, - 1.12477752 + 1.08193727e+00j,
                            1.56502881 + 3.02261738e+00j],
                           [0.49792363 - 2.22044605e-16j, - 1.20769548 - 2.56558544e+00j,
                            0.63208770 + 8.00197720e-01j,
                            1.56502881 + 3.02261738e+00j,
                            1.78211085 + 2.10567238e+00j]])

        bispec_mag = np.array([[2.75858211, 3.40375249, 1.01973097, 2.83562286, 0.49792363],
                               [3.40375249, 1.56067701, 0.22664968, 0.25940971, 2.83562286],
                               [1.01973097, 0.22664968, 0.088, 0.22664968, 1.01973097],
                               [2.83562286, 0.25940971, 0.22664968, 1.56067701, 3.40375249],
                               [0.49792363, 2.83562286, 1.01973097, 3.40375249, 2.75858211]])

        bispec_phase = np.array([[-8.68432005e-01, -1.09303185e+00, -9.02235465e-01, 2.01075415e+00,
                                  4.45941091e-16],
                                 [-1.09303185e+00, -2.37560564e+00, -9.85320934e-01, 8.55961046e-16,
                                  -2.01075415e+00],
                                 [-9.02235465e-01, -9.85320934e-01, 3.14159265e+00, 9.85320934e-01,
                                  9.02235465e-01],
                                 [2.01075415e+00, -8.55961046e-16, 9.85320934e-01, 2.37560564e+00,
                                  1.09303185e+00],
                                 [-4.45941091e-16, -2.01075415e+00, 9.02235465e-01, 1.09303185e+00,
                                  8.68432005e-01]])

        assert bs.lc == self.lc
        assert np.isclose(bs.fs, 1)
        assert bs.maxlag == 2
        assert bs.n == 5
        assert bs.scale == 'unbiased'
        assert np.allclose(bs.lags, lags)
        assert np.allclose(bs.freq, freq)
        assert np.allclose(bs.cum3, cum3)
        assert np.allclose(bs.bispec, bispec)
        assert np.allclose(bs.bispec_mag, bispec_mag)
        assert np.allclose(bs.bispec_phase, bispec_phase)

    def test_lc1_with_diff_lag(self):
        bs = Bispectrum(self.lc1, maxlag=1)
        lags = np.array([-0.5, 0, 0.5])
        freq = np.array([-1, 0., 1])
        cum3 = np.array([[0.37114198, -0.62885802, -0.02160494],
                         [-0.62885802, 0.59259259, 0.37114198],
                         [-0.02160494, 0.37114198, -0.62885802]])

        bispec = np.array([[0.93595679 + 2.59807621e+00j, 0.61419753 + 0j,
                            0.61419753 + 2.22044605e-16j],
                           [0.61419753 + 1.11022302e-16j, -0.22376543 + 0j,
                            0.61419753 - 1.11022302e-16j],
                           [0.61419753 - 2.22044605e-16j, 0.61419753 + 0j,
                            0.93595679 - 2.59807621e+00j]])

        bispec_mag = np.array([[2.76152406, 0.61419753, 0.61419753],
                               [0.61419753, 0.22376543, 0.61419753],
                               [0.61419753, 0.61419753, 2.76152406]])

        bispec_phase = np.array([[1.22501950e+00, 0, 3.61519859e-16],
                                 [1.80759930e-16, 3.14159265e+00, -1.80759930e-16],
                                 [-3.61519859e-16, 0, -1.22501950e+00]])

        assert bs.lc == self.lc1
        assert np.isclose(bs.fs, 2)
        assert bs.maxlag == 1
        assert bs.n == 6
        assert bs.scale == 'biased'
        assert np.allclose(bs.lags, lags)
        assert np.allclose(bs.freq, freq)
        assert np.allclose(bs.cum3, cum3)
        assert np.allclose(bs.bispec, bispec)
        assert np.allclose(bs.bispec_mag, bispec_mag)
        assert np.allclose(bs.bispec_phase, bispec_phase)

    def test_lc1_unbiased_scale(self):
        bs = Bispectrum(self.lc1, maxlag=1, scale='unbiased')
        lags = np.array([-0.5, 0, 0.5])
        freq = np.array([-1, 0., 1])

        cum3 = np.array([[0.44537037, -0.75462963, -0.03240741],
                         [-0.75462963, 0.59259259, 0.44537037],
                         [-0.03240741, 0.44537037, -0.75462963]])

        bispec = np.array([[0.99166667 + 3.11769145e+00j, 0.62500000 + 0j,
                            0.62500000 + 0j],
                           [0.62500000 + 2.22044605e-16j, -0.40000000 + 0j,
                            0.62500000 - 2.22044605e-16j],
                           [0.62500000 + 0j, 0.62500000 + 0j,
                            0.99166667 - 3.11769145e+00j]])

        bispec_mag = np.array([[3.27160554, 0.625, 0.625],
                               [0.625, 0.4, 0.625],
                               [0.625, 0.625, 3.27160554]])

        bispec_phase = np.array([[1.26283852, 0, 0],
                                 [0, 3.14159265, 0],
                                 [0, 0, -1.26283852]])

        assert bs.lc == self.lc1
        assert np.isclose(bs.fs, 2)
        assert bs.maxlag == 1
        assert bs.n == 6
        assert bs.scale == 'unbiased'
        assert np.allclose(bs.lags, lags)
        assert np.allclose(bs.freq, freq)
        assert np.allclose(bs.cum3, cum3)
        assert np.allclose(bs.bispec, bispec)
        assert np.allclose(bs.bispec_mag, bispec_mag)
        assert np.allclose(bs.bispec_phase, bispec_phase)

    @pytest.mark.skipif(HAS_MPL, reason='Matplotlib is already installed if condition is met')
    def test_plot_matplotlib_not_installed(self):
        bs = Bispectrum(self.lc)
        with pytest.raises(ImportError) as excinfo:
            bs.plot_cum3()
            bs.plot_mag()
            bs.plot_phase()
        message = str(excinfo.value)
        assert "Matplotlib required for plot()" in message

    @pytest.mark.skipif(not HAS_MPL, reason='Matplotlib is not installed')
    def test_plot_cum3(self):
        bs = Bispectrum(self.lc)
        bs.plot_cum3()
        assert plt.fignum_exists(1)

    @pytest.mark.skipif(not HAS_MPL, reason='Matplotlib is not installed')
    def test_plot_mag(self):
        bs = Bispectrum(self.lc)
        bs.plot_mag()
        assert plt.fignum_exists(1)

    @pytest.mark.skipif(not HAS_MPL, reason='Matplotlib is not installed')
    def test_plot_phase(self):
        bs = Bispectrum(self.lc)
        bs.plot_phase()
        assert plt.fignum_exists(1)

    @pytest.mark.skipif(not HAS_MPL, reason='Matplotlib is not installed')
    def test_plot_cum3_axis(self):
        bs = Bispectrum(self.lc)
        bs.plot_cum3(axis=[0, 1, 0, 100])
        assert plt.fignum_exists(1)

    @pytest.mark.skipif(not HAS_MPL, reason='Matplotlib is not installed')
    def test_plot_mag_axis(self):
        bs = Bispectrum(self.lc)
        bs.plot_mag(axis=[0, 1, 0, 100])
        assert plt.fignum_exists(1)

    @pytest.mark.skipif(not HAS_MPL, reason='Matplotlib is not installed')
    def test_plot_phase_axis(self):
        bs = Bispectrum(self.lc)
        bs.plot_phase(axis=[0, 1, 0, 100])
        assert plt.fignum_exists(1)

    @pytest.mark.skipif(not HAS_MPL, reason='Matplotlib is not installed')
    def test_plot_cum3_default_filename(self):
        bs = Bispectrum(self.lc)
        bs.plot_cum3(save=True)
        assert os.path.isfile('bispec_cum3.png')
        os.unlink('bispec_cum3.png')

    @pytest.mark.skipif(not HAS_MPL, reason='Matplotlib is not installed')
    def test_plot_mag_default_filename(self):
        bs = Bispectrum(self.lc)
        bs.plot_mag(save=True)
        assert os.path.isfile('bispec_mag.png')
        os.unlink('bispec_mag.png')

    @pytest.mark.skipif(not HAS_MPL, reason='Matplotlib is not installed')
    def test_plot_phase_default_filename(self):
        bs = Bispectrum(self.lc)
        bs.plot_phase(save=True)
        assert os.path.isfile('bispec_phase.png')
        os.unlink('bispec_phase.png')

    @pytest.mark.skipif(not HAS_MPL, reason='Matplotlib is not installed')
    def test_plot_cum3_custom_filename(self):
        bs = Bispectrum(self.lc)
        bs.plot_cum3(save=True, filename='cum3.png')
        assert os.path.isfile('cum3.png')
        os.unlink('cum3.png')

    @pytest.mark.skipif(not HAS_MPL, reason='Matplotlib is not installed')
    def test_plot_mag_custom_filename(self):
        bs = Bispectrum(self.lc)
        bs.plot_phase(save=True, filename='mag.png')
        assert os.path.isfile('mag.png')
        os.unlink('mag.png')

    @pytest.mark.skipif(not HAS_MPL, reason='Matplotlib is not installed')
    def test_plot_phase_custom_filename(self):
        bs = Bispectrum(self.lc)
        bs.plot_phase(save=True, filename='phase.png')
        assert os.path.isfile('phase.png')
        os.unlink('phase.png')