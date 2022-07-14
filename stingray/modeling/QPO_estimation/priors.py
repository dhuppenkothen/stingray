import numpy as np
from typing import Union
import bilby
import math
from stingray.modeling.QPO_estimation.utils.utils.priors import (get_fred_priors, get_fred_extended_priors,
                                                                 get_gaussian_priors, get_skew_exponential_priors,
                                                                 get_skew_gaussian_priors, get_polynomial_priors)

from stingray.modeling.QPO_estimation.utils.utils.kernel_priors import (get_white_noise_prior,
                                                                        get_red_noise_prior,
                                                                        get_pure_qpo_prior,
                                                                        get_qpo_prior,
                                                                        get_double_red_noise_prior,
                                                                        get_qpo_plus_red_noise_prior,
                                                                        get_double_qpo_prior,
                                                                        get_sho_prior,
                                                                        get_double_sho_prior,
                                                                        get_matern_32_prior,
                                                                        get_matern_52_prior,
                                                                        get_exp_sine2_prior,
                                                                        get_exp_sine2_rn_prior,
                                                                        get_rational_quadratic_prior,
                                                                        get_square_exponential_prior)

from stingray.modeling.QPO_estimation.utils.utils.priors import MinimumPrior


def mean_prior(times: np.ndarray, counts: np.ndarray,
               model_type: str = 'skew_gaussian', polynomial_max: int = 2,
               minimum_window_spacing: int = 0, n_components: int = 1,
               offset: bool = False, amplitude_min: Union[float, None] = None,
               amplitude_max: Union[float, None] = None, offset_min: Union[float, None] = None,
               offset_max: Union[float, None] = None, sigma_min: Union[float, None] = None,
               sigma_max: Union[float, None] = None, t_0_min: Union[float, None] = None,
               t_0_max: Union[float, None] = None):

    minimum = np.min(counts) if offset else 0
    maximum = np.max(counts)
    span = maximum - minimum
    if amplitude_min is None:
        amplitude_min = 0.1 * span
    if amplitude_max is None:
        amplitude_max = 2 * span

    if model_type == 'gaussian':
        priors = get_gaussian_priors(n_components=n_components, minimum_spacing=minimum_window_spacing,
                                     t_0_min=t_0_min, t_0_max=t_0_max, amplitude_min=amplitude_min,
                                     amplitude_max=amplitude_max, sigma_max=sigma_max, sigma_min=sigma_min)

    elif model_type == 'skew_gaussian':
        priors = get_skew_gaussian_priors(n_components=n_components, minimum_spacing=minimum_window_spacing,
                                          t_0_min=t_0_min, t_0_max=t_0_max, amplitude_min=amplitude_min,
                                          amplitude_max=amplitude_max, sigma_max=sigma_max, sigma_min=sigma_min)

    elif model_type == 'skew_exponential':
        priors = get_skew_exponential_priors(n_components=n_components, minimum_spacing=minimum_window_spacing,
                                             t_0_min=t_0_min, t_0_max=t_0_max, amplitude_min=amplitude_min,
                                             amplitude_max=amplitude_max, sigma_max=sigma_max, sigma_min=sigma_min)

    elif model_type == 'fred':
        priors = get_fred_priors(times=times, n_components=n_components, minimum_spacing=minimum_window_spacing,
                                 t_0_min=t_0_min, t_0_max=t_0_max, amplitude_min=amplitude_min,
                                 amplitude_max=amplitude_max, sigma_max=sigma_max, sigma_min=sigma_min)

    elif model_type == 'fred_extended':
        priors = get_fred_extended_priors(times=times, n_components=n_components,
                                          minimum_spacing=minimum_window_spacing, t_0_min=t_0_min, t_0_max=t_0_max,
                                          amplitude_min=amplitude_min, amplitude_max=amplitude_max,
                                          sigma_max=sigma_max, sigma_min=sigma_min)

    elif model_type == 'polynomial':
        priors = get_polynomial_priors(n_components=n_components, polynomial_max=polynomial_max)

    else:
        priors = dict()

    offset_min = offset_min
    offset_max = offset_max
    if offset_min is None:
        offset_min = minimum
    if offset_max is None:
        offset_max = maximum

    if offset:
        if math.isclose(offset_min, offset_max):
            priors["mean:offset"] = bilby.prior.DeltaFunction(peak=offset_max, name="log offset")
        else:
            priors["mean:offset"] = bilby.prior.Uniform(minimum=offset_min, maximum=offset_max,
                                                        name="offset")

    return priors


def kernel_prior(kernel_type: str,
                 min_log_a: float,
                 max_log_a: float,
                 min_log_c_red_noise: float,
                 max_log_c_red_noise: float,
                 min_log_c_qpo: float,
                 max_log_c_qpo: float,
                 band_minimum: float,
                 band_maximum: float,
                 jitter_term: bool):

    if max_log_c_qpo is None or np.isnan(max_log_c_qpo):
        max_log_c_qpo = np.log(band_maximum)

    if kernel_type == 'white_noise':
        priors = get_white_noise_prior(jitter_term=jitter_term)

    elif kernel_type == 'red_noise':
        priors = get_red_noise_prior(max_log_a=max_log_a, max_log_c_red_noise=max_log_c_red_noise,
                                     min_log_a=min_log_a, min_log_c_red_noise=min_log_c_red_noise)

    elif kernel_type == 'pure_qpo':
        priors = get_pure_qpo_prior(band_maximum=band_maximum,
                                    band_minimum=band_minimum,
                                    max_log_a=max_log_a,
                                    max_log_c_qpo=max_log_c_qpo,
                                    min_log_a=min_log_a,
                                    min_log_c_qpo=min_log_c_qpo)

    elif kernel_type == 'qpo':
        priors = get_qpo_prior(band_maximum=band_maximum,
                               band_minimum=band_minimum,
                               max_log_a=max_log_a,
                               max_log_c=max_log_c_qpo,
                               min_log_a=min_log_a,
                               min_log_c_qpo=min_log_c_qpo)

    elif kernel_type == 'double_red_noise':
        priors = get_double_red_noise_prior(max_log_a=max_log_a, max_log_c_red_noise=max_log_c_red_noise,
                                            min_log_a=min_log_a, min_log_c_red_noise=min_log_c_red_noise)

    elif kernel_type == 'qpo_plus_red_noise':
        priors = get_qpo_plus_red_noise_prior(band_maximum=band_maximum,
                                              band_minimum=band_minimum,
                                              max_log_a=max_log_a,
                                              max_log_c_red_noise=max_log_c_red_noise,
                                              max_log_c_qpo=max_log_c_qpo,
                                              min_log_a=min_log_a,
                                              min_log_c_red_noise=min_log_c_red_noise,
                                              min_log_c_qpo=min_log_c_qpo)

    elif kernel_type == 'double_qpo':
        priors = get_double_qpo_prior(band_maximum=band_maximum,
                                      band_minimum=band_minimum,
                                      max_log_a=max_log_a,
                                      max_log_c_qpo=max_log_c_qpo,
                                      min_log_a=min_log_a,
                                      min_log_c_qpo=min_log_c_qpo)

    elif kernel_type == 'sho':
        priors = get_sho_prior(band_maximum=band_maximum,
                               band_minimum=band_minimum,
                               max_log_a=max_log_a,
                               min_log_a=min_log_a)

    elif kernel_type == 'double_sho':
        priors = get_double_sho_prior(band_maximum=band_maximum,
                                      band_minimum=band_minimum,
                                      max_log_a=max_log_a,
                                      min_log_a=min_log_a)

    elif kernel_type == 'matern32':
        priors = get_matern_32_prior()

    elif kernel_type == 'matern52':
        priors = get_matern_52_prior()

    elif kernel_type == 'exp_sine2':
        priors = get_exp_sine2_prior(band_maximum=band_maximum,
                                     band_minimum=band_minimum)

    elif kernel_type == 'exp_sine2_rn':
        priors = get_exp_sine2_rn_prior(band_maximum=band_maximum,
                                        band_minimum=band_minimum)

    elif kernel_type == 'rational_quadratic':
        priors = get_rational_quadratic_prior()

    elif kernel_type == 'exp_squared':
        priors = get_square_exponential_prior()

    else:
        priors = bilby.prior.PriorDict()

    return priors


def window_prior(times):

    priors = bilby.core.prior.ConditionalPriorDict()
    priors["window_minimum"] = bilby.core.prior.Beta(minimum=times[0], maximum=times[-1], alpha=1, beta=2,
                                                     name="window_minimum", boundary="reflective")
    priors["window_maximum"] = MinimumPrior(minimum=times[0], maximum=times[-1], order=1,
                                            reference_name="window_minimum", name="window_maximum",
                                            minimum_spacing=0.1, boundary="reflective")
    return priors