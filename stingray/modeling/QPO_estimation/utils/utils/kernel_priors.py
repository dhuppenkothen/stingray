import math
import numpy as np
from copy import deepcopy
import bilby
from priors import MinimumPrior


def add_jitter_term(priors):
    new_priors = deepcopy(priors)
    for k, v in priors.items():
        if ":" in k:
            if "[" not in k:
                new_key = k.replace(":", ":terms[0]:")
                new_priors[new_key] = new_priors[k]
                new_priors[new_key].name = f"terms[0]:{new_priors[new_key].name}"
                del new_priors[k]
    n_terms = 0
    for k in new_priors.keys():
        for i in range(10):
            if f"[{i}]" in k and n_terms < i + 1:
                n_terms = i + 1
    if n_terms == 0:
        new_priors[f"kernel:log_sigma"] = bilby.core.prior.Uniform(
            minimum=-25, maximum=5, name=f"log_sigma")
    else:
        new_priors[f"kernel:terms[{n_terms}]:log_sigma"] = bilby.core.prior.Uniform(
            minimum=-25, maximum=5, name=f"terms[{n_terms}]:log_sigma")
    return new_priors


def get_white_noise_prior(jitter_term=False):
    priors = bilby.prior.PriorDict()
    if jitter_term:
        return add_jitter_term(priors)
    else:
        priors["kernel:log_sigma"] = bilby.core.prior.DeltaFunction(peak=-20, name="log_sigma")
        return priors


def get_qpo_plus_red_noise_prior(
        band_maximum, band_minimum, max_log_a, max_log_c_red_noise, max_log_c_qpo, min_log_a, min_log_c_red_noise,
        min_log_c_qpo):
    min_log_f = np.log(band_minimum)
    max_log_f = np.log(band_maximum)

    priors = bilby.prior.PriorDict()
    add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label="terms[0]:log_a")
    add_individual_kernel_prior(priors=priors, minimum=min_log_c_qpo, maximum=max_log_c_qpo, label="terms[0]:log_c")
    add_individual_kernel_prior(priors=priors, minimum=min_log_f, maximum=max_log_f, label="terms[0]:log_f")
    add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label="terms[1]:log_a")
    add_individual_kernel_prior(priors=priors, minimum=min_log_c_red_noise,
                                maximum=max_log_c_red_noise, label="terms[1]:log_c")
    priors["decay_constraint"] = bilby.core.prior.Constraint(minimum=-1000, maximum=0.0, name="decay_constraint")
    priors.conversion_function = decay_constraint_conversion_function
    return priors


def get_red_noise_prior(max_log_a, max_log_c_red_noise, min_log_a, min_log_c_red_noise):
    priors = bilby.prior.PriorDict()
    add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label="log_a")
    add_individual_kernel_prior(priors=priors, minimum=min_log_c_red_noise, maximum=max_log_c_red_noise, label="log_c")
    return priors


def get_sho_prior(band_maximum, band_minimum, max_log_a, min_log_a):
    priors = bilby.prior.PriorDict()
    priors["kernel:log_S0"] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a, name="log_S0")
    priors["kernel:log_Q"] = bilby.core.prior.Uniform(minimum=-10, maximum=10, name="log_Q")
    priors["kernel:log_omega0"] = bilby.core.prior.Uniform(
        minimum=np.log(2*np.pi*band_minimum), maximum=np.log(2*np.pi*band_maximum), name="log_omega0")
    return priors


def get_double_sho_prior(band_maximum, band_minimum, max_log_a, min_log_a):
    priors = bilby.prior.PriorDict()
    priors["kernel:terms[0]:log_S0"] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a, name="log_S0")
    priors["kernel:terms[0]:log_Q"] = bilby.core.prior.Uniform(minimum=-10, maximum=np.log(0.5), name="log_Q")
    priors["kernel:terms[0]:log_omega0"] = bilby.core.prior.Uniform(
        minimum=np.log(2*np.pi*band_minimum), maximum=np.log(2*np.pi*band_maximum), name="log_omega0")
    priors["kernel:terms[1]:log_S0"] = bilby.core.prior.Uniform(minimum=min_log_a, maximum=max_log_a, name="log_S0")
    priors["kernel:terms[1]:log_Q"] = bilby.core.prior.Uniform(minimum=np.log(0.5), maximum=10, name="log_Q")
    priors["kernel:terms[1]:log_omega0"] = bilby.core.prior.Uniform(
        minimum=np.log(2*np.pi*band_minimum), maximum=np.log(2*np.pi*band_maximum), name="log_omega0")
    return priors


def get_double_red_noise_prior(max_log_a, max_log_c_red_noise, min_log_a, min_log_c_red_noise):
    priors = bilby.prior.ConditionalPriorDict()
    add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label="terms[0]:log_a")
    add_individual_kernel_prior(
        priors=priors, minimum=min_log_c_red_noise, maximum=max_log_c_red_noise, label="terms[0]:log_c")
    add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label="terms[1]:log_a")
    add_individual_kernel_prior(
        priors=priors, minimum=min_log_c_red_noise, maximum=max_log_c_red_noise, label="terms[1]:log_c")

    priors["kernel:terms[0]:log_a"] = \
        bilby.core.prior.Beta(
            minimum=min_log_a, maximum=max_log_a, alpha=1, beta=2,
            name="kernel:terms[0]:log_a", boundary="reflective")
    priors["kernel:terms[1]:log_a"] = MinimumPrior(
        minimum=min_log_a, maximum=max_log_a, order=1, reference_name="kernel:terms[0]:log_a",
        name="kernel:terms[1]:log_a", minimum_spacing=0.0, boundary="reflective")
    return priors


def get_double_qpo_prior(band_maximum, band_minimum, max_log_a, max_log_c_qpo, min_log_a, min_log_c_qpo):
    min_log_f = np.log(band_minimum)
    max_log_f = np.log(band_maximum)

    priors = bilby.prior.ConditionalPriorDict()
    add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label="terms[0]:log_a")
    add_individual_kernel_prior(priors=priors, minimum=min_log_c_qpo, maximum=max_log_c_qpo, label="terms[0]:log_c")
    add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label="terms[1]:log_a")
    add_individual_kernel_prior(priors=priors, minimum=min_log_c_qpo, maximum=max_log_c_qpo, label="terms[1]:log_c")
    priors["kernel:terms[0]:log_f"] = \
        bilby.core.prior.Beta(
            minimum=min_log_f, maximum=max_log_f, alpha=1, beta=2,
            name="kernel:terms[0]:log_f", boundary="reflective")
    priors["kernel:terms[1]:log_f"] = MinimumPrior(
        minimum=min_log_f, maximum=max_log_f, order=1, reference_name="kernel:terms[0]:log_f",
        name="kernel:terms[1]:log_f", minimum_spacing=0.0, boundary="reflective")

    priors["decay_constraint"] = bilby.core.prior.Constraint(minimum=-1000, maximum=0.0, name="decay_constraint")
    priors["decay_constraint_2"] = bilby.core.prior.Constraint(minimum=-1000, maximum=0.0, name="decay_constraint_2")
    priors.conversion_function = decay_constraint_conversion_function
    return priors


def get_pure_qpo_prior(band_maximum, band_minimum, max_log_a, max_log_c_qpo, min_log_a, min_log_c_qpo):
    min_log_f = np.log(band_minimum)
    max_log_f = np.log(band_maximum)

    priors = bilby.prior.PriorDict()
    add_individual_kernel_prior(priors=priors, minimum=min_log_a, maximum=max_log_a, label="log_a")
    add_individual_kernel_prior(priors=priors, minimum=min_log_c_qpo, maximum=max_log_c_qpo, label="log_c")
    add_individual_kernel_prior(priors=priors, minimum=min_log_f, maximum=max_log_f, label="log_f")
    priors["decay_constraint"] = bilby.core.prior.Constraint(minimum=-1000, maximum=0.0, name="decay_constraint")
    priors.conversion_function = decay_constraint_conversion_function
    return priors


def get_qpo_prior(band_maximum, band_minimum, max_log_a, max_log_c, min_log_a, min_log_c_qpo):
    priors = bilby.prior.PriorDict()
    priors["kernel:log_b"] = bilby.core.prior.DeltaFunction(peak=-10, name="log_b")
    priors.update(get_pure_qpo_prior(band_maximum=band_maximum, band_minimum=band_minimum, max_log_a=max_log_a,
                                      max_log_c_qpo=max_log_c, min_log_a=min_log_a, min_log_c_qpo=min_log_c_qpo))
    return priors


def get_matern_32_prior():
    priors = bilby.prior.PriorDict()
    priors["kernel:k1:metric:log_M_0_0"] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name="log_M_0_0")
    priors["kernel:k2:log_constant"] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name="log_alpha")
    return priors


def get_matern_52_prior():
    priors = bilby.prior.PriorDict()
    priors["kernel:k1:metric:log_M_0_0"] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name="log_M_0_0")
    priors["kernel:k2:log_constant"] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name="log_alpha")
    return priors


def get_exp_sine2_prior(band_minimum, band_maximum):
    priors = bilby.prior.PriorDict()
    priors["kernel:k1:metric:log_M_0_0"] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name="log_M_0_0")
    priors["kernel:k2:gamma"] = bilby.core.prior.Uniform(minimum=-10, maximum=1000, name="gamma")
    priors["kernel:k2:log_period"] = bilby.core.prior.Uniform(
        minimum=-np.log(band_maximum), maximum=-np.log(band_minimum), name="log_period")
    return priors


def get_exp_sine2_rn_prior(band_minimum, band_maximum):
    priors = bilby.prior.PriorDict()
    priors["kernel:k1:k1:gamma"] = bilby.core.prior.Uniform(minimum=-10, maximum=1000, name="gamma")
    priors["kernel:k1:k1:log_period"] = bilby.core.prior.Uniform(
        minimum=-np.log(band_maximum), maximum=-np.log(band_minimum), name="log_period")
    priors["kernel:k1:k2:log_constant"] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name="log_alpha_1")
    priors["kernel:k2:k1:metric:log_M_0_0"] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name="log_M_0_0")
    priors["kernel:k2:k2:log_constant"] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name="log_alpha_2")
    return priors


def get_rational_quadratic_prior():
    priors = bilby.prior.PriorDict()
    priors["kernel:metric:log_M_0_0"] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name="log_M_0_0")
    priors["kernel:log_alpha"] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name="log_alpha")
    return priors


def get_square_exponential_prior():
    priors = bilby.prior.PriorDict()
    priors["kernel:k1:metric:log_M_0_0"] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name="log_M_0_0")
    priors["kernel:k2:log_constant"] = bilby.core.prior.Uniform(minimum=-15, maximum=15, name="log_alpha")
    return priors


def add_individual_kernel_prior(priors, minimum, maximum, label):
    if math.isclose(minimum, maximum):
        priors[f"kernel:{label}"] = bilby.core.prior.DeltaFunction(peak=maximum, name=label)
    else:
        priors[f"kernel:{label}"] = bilby.core.prior.Uniform(minimum=minimum, maximum=maximum, name=label)


def get_window_priors(times, likelihood_model="celerite_windowed"):
    if likelihood_model == "celerite_windowed":
        priors = bilby.core.prior.ConditionalPriorDict()
        priors["window_minimum"] = bilby.core.prior.Beta(minimum=times[0], maximum=times[-1], alpha=1, beta=2,
                                                         name="window_minimum", boundary="reflective")
        priors["window_maximum"] = MinimumPrior(minimum=times[0], maximum=times[-1], order=1,
                                                reference_name="window_minimum", name="window_maximum",
                                                minimum_spacing=0.1, boundary="reflective")
        return priors
    else:
        return bilby.prior.PriorDict()


def decay_constraint_conversion_function(sample):
    out_sample = sample.copy()
    if "kernel:log_f" in sample.keys():
        out_sample["decay_constraint"] = out_sample["kernel:log_c"] - out_sample["kernel:log_f"]
    elif "kernel:terms[0]:log_f" in sample.keys():
        out_sample["decay_constraint"] = out_sample["kernel:terms[0]:log_c"] - out_sample["kernel:terms[0]:log_f"]
    return out_sample


def decay_constrain_conversion_function_2(sample):
    out_sample = sample.copy()
    out_sample["decay_constraint"] = out_sample["kernel:terms[0]:log_c"] - out_sample["kernel:terms[0]:log_f"]
    out_sample["decay_constraint_2"] = out_sample["kernel:terms[1]:log_c"] - out_sample["kernel:terms[1]:log_f"]
    return out_sample