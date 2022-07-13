import numpy as np
from typing import Union
import celerite
import george
from bilby.core.likelihood import Likelihood, function_to_celerite_mean_model, function_to_george_mean_model
from mean_model_utils import polynomial
import mean_model_utils as mmu
import bilby
from george.modeling import Model as GeorgeModel
from celerite.modeling import Model as CeleriteModel


model_type_to_func = {'skew_exponential': mmu.skew_exponential,
                      'fred': mmu.fred,
                      'fred_extended': mmu.fred_extended,
                      'gaussian': mmu.gaussian,
                      'skew_gaussian': mmu.skew_gaussian,
                      'log_normal': mmu.log_normal,
                      'lorentzian': mmu.lorentzian}

LIKELIHOOD_MODEL_DICT = dict(george=GeorgeModel, celerite=CeleriteModel, celerite_windowed=CeleriteModel)


def _get_parameter_names(base_names: list, n_models: int, offset: bool) -> tuple:
    """ Takes a list of parameter names and modifies them to account for a multiple component model.

    Parameters
    ----------
    base_names:
        The parameter names of the basis function.
    n_models:
        Number of flare shapes we want to compound.
    offset:
        Whether we want to add an offset parameter

    Returns
    -------
    The parameter names of the `celerite` or `george` model.
    """
    names = []
    for i in range(n_models):
        for base in base_names:
            names.extend([f"{base}_{i}"])
    if offset:
        names.extend(["offset"])
    return tuple(names)


def get_n_component_mean_model(
        model: callable, n_models: int = 1, offset: bool = False,
        likelihood_model: str = "celerite") -> Union[celerite.modeling.Model, george.modeling.Model]:

    """ Takes a function and turns it into an n component `celerite` or `george` mean model.

        Parameters
        ----------
        model:
            The model with the x-coordinate as the first function argument.
        n_models:
            Number of flare shapes we want to compound.
        offset:
            Whether we want to include a constant offset in the model.
        likelihood_model:
            'celerite' or 'george'

        Returns
        -------
        The `celerite` or `george` mean model
        """

    base_names = bilby.core.utils.infer_parameters_from_function(func=model)
    names = _get_parameter_names(base_names, n_models, offset)
    defaults = {name: 0.1 for name in names}
    m = LIKELIHOOD_MODEL_DICT[likelihood_model]

    class MultipleMeanModel(m):
        parameter_names = names

        def get_value(self, t):
            res = np.zeros(len(t))
            for j in range(n_models):
                res += model(t, **{f"{b}": getattr(self, f"{b}_{j}") for b in base_names})
            if offset:
                res += getattr(self, "offset")
            return res

        def compute_gradient(self, *args, **kwargs):
            pass

    return MultipleMeanModel(**defaults)


def mean_model(model_type: str,
               n_components: int,
               offset: bool = 'False',
               likelihood_model: str = 'celeste') -> Union[celerite.modeling.Model, george.modeling.Model]:
    """ Creates a mean model instance for use in the likelihood.

        Parameters
        ----------
        model_type:
            The model type as a string.
        n_components:
            The number of flare shapes to use.
        y:
            The y-coordinates of the data. Only relevant if we use a constant mean as a mean model.
        offset:
            If we are using a constant offset component.
        likelihood_model:
            The likelihood model we use. Must be from ['celerite', 'celerite_windowed', 'george'].

        Returns
        -------
        The mean model.
        """

    if model_type == 'polynomial':
        if likelihood_model in ["celerite", "celerite_windowed"]:
            return function_to_celerite_mean_model(polynomial)(a0=0, a1=0, a2=0, a3=0, a4=0)
        elif likelihood_model == "george":
            return function_to_george_mean_model(polynomial)(a0=0, a1=0, a2=0, a3=0, a4=0)

    elif model_type in model_type_to_func.keys():
        return get_n_component_mean_model(model_type_to_func[model_type], n_models=n_components, offset=offset,
                                          likelihood_model=likelihood_model)


    else:
        return ValueError(f"Model type {model_type} not supported.")

if __name__ == '__main__':

    model_type = 'gaussian'
    n_components = 4
    offset = True
    likelihood_model = 'george'
    mean_model = mean_model(model_type, n_components, offset, likelihood_model)
    print(type(mean_model))
