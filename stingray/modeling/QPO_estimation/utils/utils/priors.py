import os
from pathlib import Path
from scipy.signal import periodogram
import numpy as np
from typing import Union
import bilby
from bilby.core.prior import ConditionalPriorDict, Uniform, Beta, DeltaFunction, ConditionalBeta, Prior
import math
import json
import matplotlib
import matplotlib.pyplot as plt


class MinimumPrior(ConditionalBeta):
    def __init__(self, order: int, minimum: float = 0, maximum: float = 1, name: str = None,
                 minimum_spacing: float = 0, latex_label: str = None, unit: str = None, boundary: str = None,
                 reference_name: str = None) -> None:
        """ A Conditional Beta prior that implements the conditional probabilities of Uniform order statistics

        Parameters
        ----------
        order:
            The order number of the parameter.
        minimum:
            The minimum of the prior range.
        maximum:
            The maximum of the prior range.
        name:
            The name of the prior.
        minimum_spacing:
            The minimal time-difference between two flares.
        latex_label:
            The latex label for the corner plot.
        unit:
            The unit for the corner plot.
        boundary:
            The boundary behaviour for the sampler. Must be from ['reflective', 'periodic', None].
        reference_name:
            The reference parameter name, which would be the prior with order of one less than the current.
        """

        super().__init__(
            alpha=1, beta=order, minimum=minimum, maximum=maximum,
            name=name, latex_label=latex_label, unit=unit,
            boundary=boundary, condition_func=self.minimum_condition
        )
        self.reference_params["order"] = order
        self.reference_params["minimum_spacing"] = minimum_spacing
        self.order = order

        if reference_name is None:
            self.reference_name = self.name[:-1] + str(int(self.name[-1]) - 1)
        else:
            self.reference_name = reference_name
        self._required_variables = [self.reference_name]
        self.minimum_spacing = minimum_spacing
        self.__class__.__name__ = "MinimumPrior"
        self.__class__.__qualname__ = "MinimumPrior"

    def minimum_condition(self, reference_params, **kwargs):  # noqa
        return dict(minimum=kwargs[self.reference_name] + self.minimum_spacing)

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        return Prior.get_instantiation_dict(self)

    def to_json(self):
        self.reset_to_reference_parameters()
        return json.dumps(self, cls=bilby.utils.BilbyJsonEncoder)


def get_gaussian_priors(n_components: int, minimum_spacing: int, t_0_min: float, t_0_max: float, amplitude_min: float,
                        amplitude_max: float, sigma_min: float, sigma_max: float) -> ConditionalPriorDict:
    priors = ConditionalPriorDict()
    for ii in range(n_components):
        if math.isclose(t_0_min,t_0_max):
            priors[f"mean:t_0_{ii}"] = DeltaFunction(t_0_min, name=f"mean:t_0_{ii}")
        elif n_components == 1:
            priors[f"mean:t_0_{ii}"] = Uniform(t_0_min, t_0_max, name=f"mean:t_0_{ii}")
        elif ii == 0:
            priors[f"mean:t_0_{ii}"] = Beta(minimum=t_0_min, maximum=t_0_max, alpha=1,
                                            beta=n_components, name=f"mean:t_0_{ii}")
        else:
            priors[f"mean:t_0_{ii}"] = MinimumPrior(
                order=n_components - ii, minimum_spacing=minimum_spacing, minimum=t_0_min,
                maximum=t_0_max, name=f"mean:t_0_{ii}")

        if math.isclose(np.log(amplitude_min), np.log(amplitude_max)):
            priors[f"mean:log_amplitude_{ii}"] = \
                bilby.prior.DeltaFunction(peak=np.log(amplitude_max), name=f"ln A_{ii}")
        else:
            priors[f"mean:log_amplitude_{ii}"] = bilby.core.prior.Uniform(
                minimum=np.log(amplitude_min),
                maximum=np.log(amplitude_max), name=f"ln A_{ii}")

        if math.isclose(np.log(sigma_min), np.log(sigma_max)):
            priors[f"mean:log_sigma_{ii}"] = \
                bilby.prior.DeltaFunction(peak=np.log(sigma_max), name=f"ln sigma_{ii}")
        else:
            priors[f"mean:log_sigma_{ii}"] = bilby.core.prior.Uniform(
                minimum=np.log(sigma_min),
                maximum=np.log(sigma_max), name=f"ln sigma_{ii}")
    return priors


def get_skew_exponential_priors(n_components: int, minimum_spacing: int,
                                t_0_min: float, t_0_max: float, amplitude_min: float,
                                amplitude_max: float, sigma_min: float, sigma_max: float):
    priors = get_gaussian_priors(n_components=n_components, minimum_spacing=minimum_spacing,
                                 t_0_max=t_0_max, t_0_min=t_0_min, amplitude_max=amplitude_max,
                                 amplitude_min=amplitude_min, sigma_max=sigma_max, sigma_min=sigma_min)
    for p in list(priors.keys()):
        if "sigma" in p:
            del priors[p]
    for ii in range(n_components):
        if math.isclose(sigma_min, sigma_max):
            priors[f"mean:log_sigma_rise_{ii}"] = bilby.core.prior.DeltaFunction(
                peak=np.log(sigma_max), name=f"ln sigma_rise_{ii}")
            priors[f"mean:log_sigma_fall_{ii}"] = bilby.core.prior.DeltaFunction(
                peak=np.log(sigma_max), name=f"ln sigma_fall_{ii}")
        else:
            priors[f"mean:log_sigma_rise_{ii}"] = bilby.core.prior.Uniform(
                minimum=np.log(sigma_min), maximum=np.log(sigma_max), name=f"ln sigma_rise_{ii}")
            priors[f"mean:log_sigma_fall_{ii}"] = bilby.core.prior.Uniform(
                minimum=np.log(sigma_min), maximum=np.log(sigma_max), name=f"ln sigma_fall_{ii}")
    return priors



def get_skew_gaussian_priors(n_components: int, minimum_spacing: int,
                             t_0_min: float, t_0_max: float, amplitude_min: float,
                             amplitude_max: float, sigma_min: float, sigma_max: float):
    priors = get_gaussian_priors(n_components=n_components, minimum_spacing=minimum_spacing,
                                 t_0_max=t_0_max, t_0_min=t_0_min, amplitude_max=amplitude_max,
                                 amplitude_min=amplitude_min, sigma_max=sigma_max, sigma_min=sigma_min)

    for ii in range(n_components):
        del priors[f"mean:log_sigma_{ii}"]
        if math.isclose(np.log(sigma_min), np.log(sigma_max)):
            priors[f"mean:log_sigma_rise_{ii}"] = bilby.core.prior.DeltaFunction(
                peak=np.log(sigma_max), name=f"ln sigma_rise_{ii}")
            priors[f"mean:log_sigma_fall_{ii}"] = bilby.core.prior.DeltaFunction(
                peak=np.log(sigma_max), name=f"ln sigma_fall_{ii}")
        else:
            priors[f"mean:log_sigma_rise_{ii}"] = bilby.core.prior.Uniform(
                minimum=np.log(sigma_min), maximum=np.log(sigma_max), name=f"ln sigma_rise_{ii}")
            priors[f"mean:log_sigma_fall_{ii}"] = bilby.core.prior.Uniform(
                minimum=np.log(sigma_min), maximum=np.log(sigma_max), name=f"ln sigma_fall_{ii}")

    return priors


def get_fred_priors(times: np.ndarray, n_components: int, minimum_spacing: int,
                    t_0_min: float, t_0_max: float, amplitude_min: float,
                    amplitude_max: float, sigma_min: float, sigma_max: float):

    priors = get_gaussian_priors(n_components=n_components, minimum_spacing=minimum_spacing,
                                 t_0_max=t_0_max, t_0_min=t_0_min, amplitude_max=amplitude_max,
                                 amplitude_min=amplitude_min, sigma_max=sigma_max, sigma_min=sigma_min)

    for ii in range(n_components):
        del priors[f"mean:log_sigma_{ii}"]
        priors[f"mean:log_psi_{ii}"] = bilby.core.prior.Uniform(minimum=np.log(2e-2),
                                                                maximum=np.log(2e4), name=f"psi_{ii}")
        priors[f"mean:delta_{ii}"] = bilby.core.prior.Uniform(minimum=0, maximum=times[-1],
                                                              name=f"delta_{ii}")
    return priors


def get_fred_extended_priors(times: np.ndarray, n_components: int, minimum_spacing: int,
                             t_0_min: float, t_0_max: float, amplitude_min: float,
                             amplitude_max: float, sigma_min: float, sigma_max: float):

    priors = get_fred_priors(times=times, n_components=n_components, minimum_spacing=minimum_spacing,
                             t_0_max=t_0_max, t_0_min=t_0_min, amplitude_max=amplitude_max,
                             amplitude_min=amplitude_min, sigma_max=sigma_max, sigma_min=sigma_min)

    for ii in range(n_components):
        priors[f"mean:log_gamma_{ii}"] = bilby.core.prior.Uniform(minimum=np.log(1e-3), maximum=np.log(1e3),
                                                                  name=f"log_gamma_{ii}")
        priors[f"mean:log_nu_{ii}"] = bilby.core.prior.Uniform(minimum=np.log(1e-3), maximum=np.log(1e3),
                                                               name=f"log_nu_{ii}")
    return priors


def get_polynomial_priors(n_components: int, polynomial_max: int):
    priors = bilby.core.prior.PriorDict()
    for i in range(n_components):
        if polynomial_max == 0:
            priors[f"mean:a{i}"] = 0
        else:
            priors[f"mean:a{i}"] = bilby.core.prior.Uniform(
                minimum=-polynomial_max, maximum=polynomial_max, name=f"mean:a{i}")
    return priors

