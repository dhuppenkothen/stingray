import tinygp
import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
import numpyro

from stingray.modeling.gpmodelling import QPO_kernel
from stingray.modeling.gpmodelling import get_kernel
from stingray.modeling.gpmodelling import GP
from stingray.modeling.gpmodelling import GPResult
from stingray.modeling.gpmodelling import qpo_numpyro_model

from stingray import Lightcurve, Crossspectrum, AveragedCrossspectrum


class TestGP(object):
    def setup_class(self):
        dt = 0.03125  # seconds
        exposure = 8.0  # seconds
        times = np.arange(0, exposure, dt)  # seconds

        signal_1 = 300 * np.sin(2.0 * np.pi * times / 0.5) + 1000  # 1000  # counts/s
        noisy_1 = np.random.poisson(signal_1 * dt)  # counts

        self.lc = Lightcurve(times, noisy_1)
        self.Model_type = ("qpo", "zero")
        self.params = {
            "amp": jnp.exp(np.log([2.0])),
            "decay": jnp.exp(np.log([1.0 / 5.0])),
            "freq": jnp.exp(
                np.log(
                    [
                        1.0 / 3.0,
                    ]
                )
            ),
            "diag": 1e-1,
            "mean": 0.0,
        }
        self.GP1 = GP(self.lc, self.Model_type, self.params)

    def test_get_model(self):
        assert (self.Model_type, self.params) == self.GP1.get_model()

    def test_get_logprob(self):
        cond = self.GP1.maingp.condition(self.lc.counts)
        assert cond.log_probability == self.GP1.get_logprob()

    def test_plot_kernel(self):
        self.GP1.plot_kernel()
        assert plt.fignum_exists(1)

    def test_plot_originalgp(self):
        X_test = np.linspace(0, 8, 256)
        self.GP1.plot_originalgp(X_test, sample_no=1)
        assert plt.fignum_exists(1)

    def test_plot_gp(self):
        X_test = np.linspace(0, 8, 256)
        self.GP1.plot_gp(X_test, sample_no=1)
        assert plt.fignum_exists(1)
