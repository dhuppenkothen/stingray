import tinygp
import jax
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

from tinygp import GaussianProcess
from stingray import Lightcurve

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

jax.config.update("jax_enable_x64", True)

__all__ = ["QPO_kernel", "GP", "GPResult", "get_kernel"]


class QPO_kernel(tinygp.kernels.Kernel):
    """
    An inheritance class for tinygp.kernels.kernels which makes a
    kernel for a QPO structure.
    This QPO covariance function is based on the Lorentzian function with
    qpo representation.

    Parameters
    ----------
    amp: float
        The amplitude of the lorentzian

    decay : float
        The decay value for the qpo frequency

    freq : float
        The modelled frequency for the QPO
    """

    def __init__(self, amp, decay, freq):
        self.amp = jnp.atleast_1d(amp)
        self.decay = jnp.atleast_1d(decay)
        self.freq = jnp.atleast_1d(freq)

    def evaluate(self, X1, X2):
        tau = jnp.atleast_1d(jnp.abs(X1 - X2))[..., None]
        return jnp.sum(
            self.amp
            * jnp.prod(jnp.exp(-self.decay * tau) * jnp.cos(2 * jnp.pi * self.freq * tau), axis=-1)
        )


def get_kernel(kernel_type: str, kernel_param=None):
    """
    A wrapper class which returns a GP kernel for the given
    kernel type and kernel parameters

    Parameters
    ----------
    kernel_type : str {qpo, rn-qpo, wn}
        It is a string which directs the type of kernel to be made

    kernel_param : dict,  default : None
        The dictionary contains the values for the kernel parameters.
        The dictionary must have all the keys for the kernel for which it
        is to be made, kindly check that you have inserted all.
        Optionally, if put to None, then, some pre-set parameters are
        used for making the parameters
    """
    if kernel_type == "qpo":
        if not kernel_param:
            amp = jnp.exp(np.log([2.0]))
            decay = jnp.exp(np.log([1.0 / 5.0]))
            freq = jnp.exp(
                np.log(
                    [
                        1.0 / 3.0,
                    ]
                )
            )
        else:
            amp = kernel_param["amp"]
            decay = kernel_param["decay"]
            freq = kernel_param["freq"]

        return QPO_kernel(amp, decay, freq)


class GP:
    """
    Makes a GP object which takes in a Stingray.Lightcurve and fits a Gaussian
    Process on the lightcurve data, for the given kernel.

    Parameters
    ----------
    lc: Stingray.Lightcurve object
        The lightcurve on which the gaussian process, is to be fitted

    Model_type: string tuple
        Has two strings with the first being the name of the kernel type
        and the secound being the mean type

    Model_parameter: dict, default = None
        Dictionary conatining the parameters for the mean and kernel
        The keys should be accourding to the selected kernel and mean
        coressponding to the Model_type
        By default, it takes a value None, and the kernel and mean are
        then bulit using the pre-set parameters.

    Other Parameters
    ----------------
    kernel: class: `TinyGp.kernel` object
        The tinygp kernel for the GP

    maingp: class: `TinyGp.GaussianProcess` object
        The tinygp gaussian process made on the lightcurve

    """

    def __init__(self, Lc: Lightcurve, Model_type: tuple, Model_params: dict = None) -> None:
        self.lc = Lc
        self.Model_type = Model_type
        self.Model_param = Model_params
        self.kernel = get_kernel(self.Model_type[0], self.Model_param)

        self.maingp = GaussianProcess(self.kernel, Lc.time, diag=Model_params["diag"])

    def get_logprob(self):
        """
        Returns the logprobability of the lightcurves counts for the
        given kernel for the Gaussian Process
        """
        cond = self.maingp.condition(self.lc.counts)
        return cond.log_probability

    def get_model(self):
        """
        Returns the model of the Gaussian Process
        """
        return (self.Model_type, self.Model_param)

    def plot_kernel(self):
        """
        Plots the kernel of the Gaussian Process
        """
        X = self.lc.time
        Y = self.kernel(X, np.array([0.0]))
        plt.plot(X, Y)
        plt.xlabel("distance")
        plt.ylabel("Value")
        plt.title("Kernel Function")

    def plot_originalgp(self, X_test, sample_no=1):
        """
        Plots samples obtained from the gaussian process for the kernel

        Parameters
        ----------
        X_test: jnp.array
            Array over which the sampled points are to be obtained
            Can be made default with lc.times as default

        sample_no: int , default = 1
            Number of GP samples to be taken

        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        y_samp = self.maingp.sample(jax.random.PRNGKey(3), shape=(2,))
        ax.plot(X_test, y_samp[0], "C0", lw=0.5, alpha=0.5, label="samples")
        ax.plot(X_test, y_samp[1:].T, "C0", lw=0.5, alpha=0.5)
        ax.set_xlabel("time")
        ax.set_ylabel("counts")
        ax.legend(loc="best")

    def plot_gp(self, X_test, sample_no=1):
        """
        Plots gaussian process, conditioned on the lightcurve
        Also, plots the lightcurve along with it

        Parameters
        ----------
        X_test: jnp.array
            Array over which the sampled points are to be obtained
            Can be made default with lc.times as default

        sample_no: int , default = 1
            Number of GP samples to be taken

        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        _, cond_gp = self.maingp.condition(self.lc.counts, X_test)
        mu = cond_gp.mean
        # std = np.sqrt(cond_gp.variance)

        ax.plot(self.lc.time, self.lc.counts, lw=2, color="blue", label="Lightcurve")
        ax.plot(X_test, mu, "C1", label="Gaussian Process")
        y_samp = cond_gp.sample(jax.random.PRNGKey(3), shape=(sample_no,))
        ax.plot(X_test, y_samp[0], "C0", lw=0.5, alpha=0.5)
        ax.set_xlabel("time")
        ax.set_ylabel("counts")
        ax.legend(loc="best")


def qpo_numpyro_model(t, y=None):
    """
    A numpyro sampling function for a lorenzian QPO kernel
    The parameters are set by default
    """
    amp = numpyro.sample("amp", dist.Uniform(0.0, 5.0))
    decay = numpyro.sample("decay", dist.Uniform(0.0, 5.0))
    freq = numpyro.sample("freq", dist.Uniform(0.0, 5.0))
    kernel = QPO_kernel(amp, decay, freq)
    gp = GaussianProcess(kernel, t, diag=1e-1)
    numpyro.sample("gp", gp.numpyro_dist(), obs=y)


class GPResult:
    """
    Makes a GP regressor for a given GP class and a prior over it.
    Provides the sampled hyperparameters and tabulates their charachtersistics
    Using numpyro for bayesian inferencing and Hyperparameter sampling.

    Parameters
    ----------
    GP: class: GP
        The initial GP class, on which we will apply our regressor.

    prior_type: string tuple
        Has two strings with the first being the name of the kernel type
        and the secound being the mean type for the prior

    prior_parameters: dict, default = None
        Dictionary containing the parameters for the mean and kernel priors
        The keys should be accourding to the selected kernel and mean
        prior coressponding to the prior_type
        By default, it takes a value None, and the kernel and mean priors are
        then bulit using the pre-set parameters.

    Other Parameters
    ----------------
    lc: Stingray.Lightcurve object
        The lightcurve on which the gaussian process regression, is to be done

    self.mcmc: `numpyro.mcmc`
        Numpyro MCMC sampler for the evaluated Gaussian Process

    self.samples: dict
        Hyperparamter samples obtained from the bayesian sampling

    self.parameters: dict
        Dictionary containing the optimal hyperparameters obtained from the
        mean of the calculated samples

    """

    def __init__(self, GP: GP, prior_type: tuple, prior_parameters=None) -> None:
        self.gpclass = GP
        self.prior_type = prior_type
        self.prior_parameters = prior_parameters
        self.lc = GP.lc

    def run_sampling(self):
        """
        Runs a sampling process for the hyperparameters for the GP model.
        Based on No U turn Sampling from the numpyro module
        """
        self.nuts_kernel = NUTS(qpo_numpyro_model, dense_mass=True, target_accept_prob=0.9)
        self.mcmc = MCMC(
            self.nuts_kernel,
            num_warmup=1000,
            num_samples=1000,
            num_chains=2,
            progress_bar=False,
        )
        rng_key = jax.random.PRNGKey(34923)
        t = self.lc.time
        y = self.lc.counts
        self.mcmc.run(rng_key, t, y=y)
        self.samples = self.mcmc.get_samples()

    def print_summary(self):
        """
        Prints a summary table for the model parameters
        """
        self.mcmc.print_summary()

    def get_parameters(self):
        """
        Returns the optimal parameters for the model based on the NUTS sampling
        """
        self.parameters = {}
        for key in self.samples:
            self.parameters[key] = (self.samples[key]).mean()
        return self.parameters

    def plot_posterior(self, X_test):
        """
        Plots posterior gaussian process, conditioned on the lightcurve
        Also, plots the lightcurve along with it

        Parameters
        ----------
        X_test: jnp.array
            Array over which the Gaussian process values are to be obtained
            Can be made default with lc.times as default

        """
        kernelf = get_kernel(self.prior_type[0], self.parameters)
        gp = GaussianProcess(kernelf, self.lc.time, diag=1e-1)
        _, cond_gp = gp.condition(self.lc.counts, X_test)

        mu = cond_gp.mean
        # std = np.sqrt(cond_gp.variance)
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.plot(self.lc.time, self.lc.counts, lw=2, color="blue", label="lightcurve")
        ax.plot(X_test, mu, "C1", label="Gaussian Process")
        ax.set_xlabel("time")
        ax.set_ylabel("counts")
        ax.legend(loc="best")
