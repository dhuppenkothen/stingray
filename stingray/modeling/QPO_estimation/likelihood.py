import numpy as np
from bilby.core.likelihood import Likelihood, CeleriteLikelihood, GeorgeLikelihood
import celerite
import george
from typing import Union
from kernel_utils import lorentzian, red_noise, white_noise, broken_power_law_noise


class ParameterAccessor(object):
    def __init__(self, parameter_name: str) -> None:
        """ Handy accessor for the likelihood parameter dict so we can call them as if they are attributes.

        Parameters
        ----------
        parameter_name:
            The name of the parameter.
        """
        self.parameter_name = parameter_name

    def __get__(self, instance, owner):
        return instance.parameters[self.parameter_name]

    def __set__(self, instance, value):
        instance.parameters[self.parameter_name] = value


class WhittleLikelihood(Likelihood):
    VALID_NOISE_MODELS = ["red_noise", "broken_power_law", "pure_qpo", "white_noise"]
    alpha = ParameterAccessor("alpha")
    alpha_1 = ParameterAccessor("alpha_1")
    alpha_2 = ParameterAccessor("alpha_2")
    log_beta = ParameterAccessor("log_beta")
    log_sigma = ParameterAccessor("log_sigma")
    rho = ParameterAccessor("rho")
    log_delta = ParameterAccessor("log_delta")
    log_amplitude = ParameterAccessor("log_amplitude")
    log_width = ParameterAccessor("log_width")
    log_frequency = ParameterAccessor("log_frequency")

    def __init__(
            self, frequencies: np.ndarray, periodogram: np.ndarray, frequency_mask: np.ndarray,
            noise_model: str = "red_noise") -> None:
        """ A Whittle likelihood class for use with `bilby`.

        Parameters
        ----------
        frequencies:
            The periodogram frequencies.
        periodogram:
            The periodogram powers.
        frequency_mask:
            A mask we can apply if we want to mask out certain frequencies/powers.
            Provide as indices which to retain.
        noise_model:
            The noise model. Should be 'red_noise' or 'broken_power_law'.
        """
        super(WhittleLikelihood, self).__init__(
            parameters=dict(alpha=0, alpha_1=0, alpha_2=0, log_beta=0, log_sigma=0, log_delta=0, rho=0,
                            log_amplitude=0, log_width=1, log_frequency=127))
        self.frequencies = frequencies
        self._periodogram = periodogram
        self.frequency_mask = frequency_mask
        self.noise_model = noise_model

    @property
    def beta(self) -> Union[float, np.ndarray]:
        return np.exp(self.log_beta)

    @property
    def sigma(self) -> Union[float, np.ndarray]:
        return np.exp(self.log_sigma)

    @property
    def delta(self) -> Union[float, np.ndarray]:
        return np.exp(self.log_delta)

    @property
    def amplitude(self) -> Union[float, np.ndarray]:
        return np.exp(self.log_amplitude)

    @property
    def width(self) -> Union[float, np.ndarray]:
        return np.exp(self.log_width)

    @property
    def frequency(self) -> Union[float, np.ndarray]:
        return np.exp(self.log_frequency)

    @property
    def frequencies(self) -> np.ndarray:
        return self._frequencies[self.frequency_mask]

    @frequencies.setter
    def frequencies(self, frequencies: np.ndarray) -> None:
        self._frequencies = frequencies

    @property
    def model(self) -> np.ndarray:
        return self.lorentzian

    @property
    def periodogram(self) -> np.ndarray:
        return self._periodogram[self.frequency_mask]

    def log_likelihood(self) -> float:
        """ Calculates the log-likelihood.

        Returns
        -------
        The log-likelihood.
        """
        psd = self.psd + self.model
        return -np.sum(np.log(psd) + self.periodogram / psd)

    @property
    def lorentzian(self) -> np.ndarray:
        return lorentzian(self.frequencies, self.amplitude, self.frequency, self.width)

    @property
    def noise_model(self) -> str:
        return self._noise_model

    @noise_model.setter
    def noise_model(self, noise_model: str) -> None:
        if noise_model in self.VALID_NOISE_MODELS:
            self._noise_model = noise_model
        elif noise_model == "qpo_plus_red_noise":
            self._noise_model = "red_noise"
        else:
            raise ValueError(f"Unknown noise model {noise_model}")

    @property
    def psd(self) -> np.ndarray:
        if self.noise_model == "red_noise":
            return red_noise(frequencies=self.frequencies, alpha=self.alpha, beta=self.beta) \
                   + white_noise(frequencies=self.frequencies, sigma=self.sigma)
        elif self.noise_model in ["pure_qpo", "white_noise"]:
            return white_noise(frequencies=self.frequencies, sigma=self.sigma)
        elif self.noise_model == "broken_power_law":
            return broken_power_law_noise(frequencies=self.frequencies, alpha_1=self.alpha_1,
                                          alpha_2=self.alpha_2, beta=self.beta, delta=self.delta, rho=self.rho) \
                   + white_noise(frequencies=self.frequencies, sigma=self.sigma)


class WindowedCeleriteLikelihood(CeleriteLikelihood):

    def __init__(
            self, mean_model: celerite.modeling.Model, kernel: celerite.terms.Term, t: np.ndarray,
            y: np.ndarray, yerr: np.ndarray) -> None:
        """
        `celerite` to `bilby` likelihood interface for GP that has defined start and end time within series.
        The likelihood adds two parameters 'window_minimum' and 'window_maximum'. Inside this window we apply the GP.
        Outside we only assume white noise.

        Parameters
        ----------
        mean_model:
            The celerite mean model.
        kernel:
            The celerite kernel.
        t:
            The time coordinates.
        y:
            The y-values.
        yerr:
            The y-error-values.
        """
        super(WindowedCeleriteLikelihood, self).__init__(
            kernel=kernel, mean_model=mean_model, t=t, y=y, yerr=yerr)
        self.parameters["window_minimum"] = t[0]
        self.parameters["window_maximum"] = t[-1]

        self._white_noise_kernel = celerite.terms.JitterTerm(log_sigma=-20)
        self.white_noise_gp = celerite.GP(kernel=self._white_noise_kernel, mean=self.mean_model)
        self.white_noise_gp.compute(self.gp._t, self.yerr) # noqa
        self.white_noise_log_likelihood = self.white_noise_gp.log_likelihood(y=y)

    def log_likelihood(self) -> Union[float, np.ndarray]:
        """

        Returns
        -------
        The log-likelihood.
        """
        if self._check_valid_indices_distribution():
            return -np.inf

        self._setup_gps()

        log_l = self.gp.log_likelihood(self.y[self.windowed_indices]) + \
            self.white_noise_gp.log_likelihood(self.y[self.edge_indices])
        return np.nan_to_num(log_l, nan=-np.inf)

    def _check_valid_indices_distribution(self) -> bool:
        return len(self.windowed_indices) == 0 or len(self.edge_indices) == 0

    def _setup_gps(self) -> None:
        self.gp.compute(self.t[self.windowed_indices], self.yerr[self.windowed_indices])
        self.white_noise_gp.compute(self.t[self.edge_indices], self.yerr[self.edge_indices] + self.jitter)
        self._set_parameters_to_gps()

    def _set_parameters_to_gps(self) -> None:
        for name, value in self.parameters.items():
            if "window" in name:
                continue
            if "mean" in name:
                self.white_noise_gp.set_parameter(name=name, value=value)
            self.gp.set_parameter(name=name, value=value)

    @property
    def jitter(self) -> float:
        for k in self.parameters.keys():
            if k.endswith("log_sigma"):
                return np.exp(self.parameters[k])
        return 0

    @property
    def edge_indices(self) -> np.ndarray:
        return np.where(np.logical_or(self.window_minimum > self.t, self.t > self.window_maximum))[0]

    @property
    def windowed_indices(self) -> np.ndarray:
        return np.where(np.logical_and(self.window_minimum < self.t, self.t < self.window_maximum))[0]

    @property
    def window_minimum(self) -> float:
        return self.parameters["window_minimum"]

    @property
    def window_maximum(self) -> float:
        return self.parameters["window_maximum"]

    def noise_log_likelihood(self) -> float:
        """ log-likelihood assuming everything is white noise.

        Returns
        -------
        The noise log-likelihood.
        """
        return self.white_noise_log_likelihood



LIKELIHOOD_MODEL_DICT = dict(
    celerite=CeleriteLikelihood, celerite_windowed=WindowedCeleriteLikelihood, george=GeorgeLikelihood)


def get_gp_likelihood(
        mean_model: Union[celerite.modeling.Model, george.modeling.Model],  # noqa
        kernel: Union[celerite.terms.Term, george.kernels.Kernel], times: np.ndarray, y: np.ndarray, yerr:
        np.ndarray, likelihood_model: str = "celerite") \
        -> Union[CeleriteLikelihood, GeorgeLikelihood, WindowedCeleriteLikelihood]:
    """ Creates the correct likelihood instance for the inference process.

    Parameters
    ----------
    mean_model:
        The mean model we use.
    kernel:
        The kernel function.
    times:
        The time coordinates.
    y:
        The y-values.
    yerr:
        The y-error values.
    likelihood_model:
        The likelihood model. Must be from `QPOEstimation.LIKELIHOOD_MODELS`.

    Returns
    -------
    The instance of the likelihood class.
    """
    return LIKELIHOOD_MODEL_DICT[likelihood_model](mean_model=mean_model, kernel=kernel, t=times, y=y, yerr=yerr)

if __name__ == '__main__':
    from mean_model import mean_model
    from kernel import kernel

    model_type = 'gaussian'
    n_components = 1
    offset = True
    likelihood_model = 'celerite'
    mm = mean_model(model_type, n_components, offset, likelihood_model)
    k = kernel("qpo")

    times = np.arange(10)
    y = np.arange(10)
    yerr = np.arange(10)

    gp_likelihood = get_gp_likelihood(mm, k, times, y, yerr)

    print(gp_likelihood)