import os
from pathlib import Path
from scipy.signal import periodogram
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
                                                                        get_qpo_prior)
import matplotlib
import matplotlib.pyplot as plt

'''_model_to_func_map = {'gaussian': get_gaussian_priors,
                      'skew_exponential': get_skew_exponential_priors,
                      'skew_gaussian': get_skew_gaussian_priors,
                      'fred': get_fred_priors,
                      'fred_extended': get_fred_extended_priors,
                      'polynomial': get_polynomial_priors}
                      
'''


def _get_mean_prior(times: np.ndarray, counts: np.ndarray,
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


def _get_kernel_prior(kernel_type: str,
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


    else:
        priors = bilby.prior.PriorDict()

    return priors




def all_priors(times: np.ndarray, counts: np.ndarray, yerr: np.ndarray,
               likelihood_model: str = 'celerite', kernel_type: str = 'qpo_plus_red_noise',
               model_type: str = 'skew_gaussian', polynomial_max: int = 2,
               minimum_window_spacing: int = 0, n_components: int = 1,
               offset: bool = False, jitter_term: bool = False,
               amplitude_min: Union[float, None] = None, amplitude_max: Union[float, None] = None,
               offset_min: Union[float, None] = None, offset_max: Union[float, None] = None,
               sigma_min: Union[float, None] = None, sigma_max: Union[float, None] = None,
               t_0_min: Union[float, None] = None, t_0_max: Union[float, None] = None,
               min_log_a: Union[float, None] = None, max_log_a: Union[float, None] = None,
               min_log_c_red_noise: Union[float, None] = None, max_log_c_red_noise: Union[float, None] = None,
               min_log_c_qpo: Union[float, None] = None, max_log_c_qpo: Union[float, None] = None,
               band_minimum: Union[float, None] = None, band_maximum: Union[float, None] = None):

    segment_length = times[-1] - times[0]
    sampling_frequency = 1 / (times[1] - times[0])

    if band_minimum is None:
        band_minimum = 2 / segment_length

    if band_maximum is None:
        band_maximum = band_maximum / 2

    if min_log_c_red_noise is None:
        min_log_c_red_noise = np.log(1 / segment_length)

    if min_log_c_qpo is None:
        min_log_c_qpo = np.log(1 / 10 / segment_length)

    if max_log_c_red_noise is None:
        max_log_c_red_noise = np.log(sampling_frequency)

    if max_log_c_qpo is None:
        if band_maximum is None:
            max_log_c_qpo = np.log(1/2/sampling_frequency)
        else:
            max_log_c_qpo = np.log(band_maximum)

    minimum = np.min(counts) if offset else 0
    maximum = np.max(counts)
    span = maximum - minimum

    if min_log_a is None:
        if yerr is not None:
            min_log_a = np.log(min(yerr))
        else:
            min_log_a = np.log(0.1 * span)

        if np.isinf(min_log_a):
            min_log_a = np.log(0.1 * span)

    if max_log_a is None:
        max_log_a = np.log(2*span)

    if t_0_min is None:
        t_0_min = times[0] - 0.1 * segment_length

    if t_0_max is None:
        t_0_max = times[-1] + 0.1 * segment_length

    if sigma_min is None:
        sigma_min = 0.5 * 1 / sampling_frequency

    if sigma_max is None:
        sigma_max = 2 * segment_length

    priors = bilby.core.prior.ConditionalPriorDict()
    mean_priors = _get_mean_prior(times, counts, model_type, polynomial_max,
                                  minimum_window_spacing, n_components,
                                  offset, amplitude_min, amplitude_max,
                                  offset_min, offset_max, sigma_min, sigma_max,
                                  t_0_min, t_0_max)
    kernel_priors = _get_kernel_prior(**kwargs)
    window_priors = gp._get_window_priors(**kwargs)
