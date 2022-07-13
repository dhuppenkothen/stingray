import numpy as np
from typing import Union
import celerite
import george
from celerite import terms
from kernel_utils import red_noise, white_noise, broken_power_law_noise, lorentzian


class QPOTerm(terms.Term):
    """ Kernel with equal amplitude and damping time exponential and cosine component.
    Proposed in the `celerite` paper, but we don't use it. """
    parameter_names = ("log_a", "log_b", "log_c", "log_f")

    def get_real_coefficients(self, params: tuple) -> tuple:
        log_a, log_b, log_c, log_f = params
        b = np.exp(log_b)
        return (
            np.exp(log_a) * (1.0 + b) / (2.0 + b), np.exp(log_c),
        )

    def get_complex_coefficients(self, params: tuple) -> tuple:
        log_a, log_b, log_c, log_f = params
        b = np.exp(log_b)
        return (
            np.exp(log_a) / (2.0 + b), 0.0,
            np.exp(log_c), 2 * np.pi * np.exp(log_f),
        )

    def compute_gradient(self, *args, **kwargs):
        pass


class ExponentialTerm(terms.Term):
    """ Exponential kernel that we use as our red noise model. """
    parameter_names = ("log_a", "log_c")

    def get_real_coefficients(self, params: tuple) -> tuple:
        log_a, log_c = params
        b = np.exp(10)
        return (
            np.exp(log_a) * (1.0 + b) / (2.0 + b), np.exp(log_c),
        )

    def get_complex_coefficients(self, params: tuple) -> tuple:
        log_a, log_c = params
        b = np.exp(10)
        return (
            np.exp(log_a) / (2.0 + b), 0.0,
            np.exp(log_c), 50,
        )

    def compute_gradient(self, *args, **kwargs):
        pass


class PureQPOTerm(terms.Term):
    """ Exponential kernel that we use as our red noise model. """
    parameter_names = ("log_a", "log_c", "log_f")

    def get_real_coefficients(self, params: tuple) -> tuple:
        log_a, log_c, log_f = params
        return 0, np.exp(log_c),

    def get_complex_coefficients(self, params: tuple) -> tuple:
        log_a, log_c, log_f = params
        a = np.exp(log_a)
        c = np.exp(log_c)
        f = np.exp(log_f)
        return a, 0.0, c, 2 * np.pi * f,

    def compute_gradient(self, *args, **kwargs):
        pass


def kernel(kernel_type: str, jitter_term: bool = False) -> Union[celerite.terms.Term, george.kernels.Kernel]:
    """ Catch all kernel getter.

    Parameters
    ----------
    kernel_type: The name of the kernel. Must be from `QPOEstimation.MODES`.
    jitter_term: Whether to add a `JitterTerm`, i.e. an additional white noise term.

    Returns
    -------
    The kernel.
    """

    if kernel_type == "white_noise":
        return celerite.terms.JitterTerm(log_sigma=-20)
    elif kernel_type == "qpo":
        res = QPOTerm(log_a=0.1, log_b=-10, log_c=-0.01, log_f=3)
    elif kernel_type == "pure_qpo":
        res = PureQPOTerm(log_a=0.1, log_c=-0.01, log_f=3)
    elif kernel_type == "red_noise":
        res = ExponentialTerm(log_a=0.1, log_c=-0.01)
    elif kernel_type == "qpo_plus_red_noise":
        res = PureQPOTerm(log_a=0.1, log_c=-0.01, log_f=3) + ExponentialTerm(log_a=0.1, log_c=-0.01)
    elif kernel_type == "double_red_noise":
        res = ExponentialTerm(log_a=0.1, log_c=-0.01) + ExponentialTerm(log_a=0.1, log_c=-0.01)
    elif kernel_type == "double_qpo":
        res = PureQPOTerm(log_a=0.1, log_c=-0.01, log_f=3) + PureQPOTerm(log_a=0.1, log_c=-0.01, log_f=3)
    elif kernel_type == "fourier_series":
        res = PureQPOTerm(log_a=0.1, log_c=-0.01, log_f=3) + \
              PureQPOTerm(log_a=0.1, log_c=-0.01, log_f=3) + \
              PureQPOTerm(log_a=0.1, log_c=-0.01, log_f=3) + \
              ExponentialTerm(log_a=0.1, log_c=-0.01)
    elif kernel_type == "sho":
        res = celerite.terms.SHOTerm(log_S0=1, log_Q=0, log_omega0=0)
    elif kernel_type == "double_sho":
        res = celerite.terms.SHOTerm(log_S0=1, log_Q=0, log_omega0=0) + \
              celerite.terms.SHOTerm(log_S0=1, log_Q=0, log_omega0=0)
    elif kernel_type == "matern32":
        res = george.kernels.Matern32Kernel(metric=1.0) * george.kernels.ConstantKernel(log_constant=0)
    elif kernel_type == "matern52":
        res = george.kernels.Matern52Kernel(metric=1.0) * george.kernels.ConstantKernel(log_constant=0)
    elif kernel_type == "exp_sine2":
        res = george.kernels.ExpSine2Kernel(gamma=1.0, log_period=10.0) * george.kernels.ConstantKernel(
            log_constant=0)
    elif kernel_type == "rational_quadratic":
        res = george.kernels.RationalQuadraticKernel(log_alpha=0.0, metric=1.0)
    elif kernel_type == "exp_squared":
        res = george.kernels.ExpSquaredKernel(metric=1.0) * george.kernels.ConstantKernel(log_constant=0)
    elif kernel_type == "exp_sine2_rn":
        res = george.kernels.ExpSine2Kernel(gamma=1.0, log_period=10.0) * george.kernels.ConstantKernel(
            log_constant=0) \
              + george.kernels.ExpKernel(metric=1.0) * george.kernels.ConstantKernel(log_constant=0)
    else:
        raise ValueError("Recovery mode not defined")

    if jitter_term:
        res += celerite.terms.JitterTerm(log_sigma=-20)

    return res

if __name__ == '__main__':
    kernel = kernel("exp_sine2_rn")
    print(kernel)