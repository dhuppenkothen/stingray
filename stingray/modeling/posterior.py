from __future__ import division

import abc
import warnings

import numpy as np
import six

np.seterr('warn')

from scipy.special import gamma as scipy_gamma
from scipy.special import gammaln as scipy_gammaln
from astropy.modeling.fitting import _fitter_to_model_params
from astropy.modeling import models

from stingray import Lightcurve, Powerspectrum


# TODO: Add checks and balances to code

#from stingray.modeling.parametricmodels import logmin

__all__ = ["set_logprior", "Posterior", "PSDPosterior", "LogLikelihood",
           "PSDLogLikelihood", "GaussianLogLikelihood",
           "PoissonPosterior", "GaussianPosterior",
           "PriorUndefinedError", "LikelihoodUndefinedError"]

logmin = -10000000000000000.0

class PriorUndefinedError(Exception):
    pass

class LikelihoodUndefinedError(Exception):
    pass

class IncorrectParameterError(Exception):
    pass

def set_logprior(lpost, priors):
    """
    This function constructs the `logprior` method required to successfully
    use a `Posterior` object.

    All instances of lass `Posterior` and its subclasses require to implement a
    `logprior` methods. However, priors are strongly problem-dependent and
    therefore usually user-defined.

    This function allows for setting the `logprior` method on any instance
    of class `Posterior` efficiently by allowing the user to pass a
    dictionary of priors and an instance of class `Posterior`.

    Parameters
    ----------
    lpost : Posterior object
        An instance of class Posterior or any of its subclasses

    priors : dictionary
        A dictionary containing the prior definitions. Keys are parameter
        names as defined by the model used in `lpost`. Items are functions
        that take a parameter as input and return the log-prior probability
         of that parameter.

    Returns
    -------
        logprior : function
            The function definition for the prior

    Example
    -------
    Make a light curve and power spectrum

    >>> photon_arrivals = np.sort(np.random.uniform(0,1000, size=10000))
    >>> lc = Lightcurve.make_lightcurve(photon_arrivals, dt=1.0)
    >>> ps = Powerspectrum(lc, norm="frac")

    Define the model

    >>> pl = models.PowerLaw1D()
    >>> pl.x_0.fixed = True

    Instantiate the posterior:

    >>> lpost = PSDPosterior(ps.freq, ps.power, pl, m=ps.m)

    Define the priors:

    >>> p_alpha = lambda alpha: ((-1. <= alpha) & (alpha <= 5.))
    >>> p_amplitude = lambda amplitude: ((-10 <= np.log(amplitude)) &
    ...                                 ((np.log(amplitude) <= 10.0)))
    >>> priors = {"alpha":p_alpha, "amplitude":p_amplitude}

    Set the logprior method in the lpost object:

    >>> lpost.logprior = set_logprior(lpost, priors)
    """

    # get the number of free parameters in the model
    #free_params = [p for p in lpost.model.param_names if not
    #                getattr(lpost.model, p).fixed]
    free_params = [key for key, l in lpost.model.fixed.items() if not l]

    # define the logprior
    def logprior(t0, neg=False):
        """
        The logarithm of the prior distribution for the
        model defined in self.model.

        Parameters:
        ------------
        t0: {list | numpy.ndarray}
            The list with parameters for the model

        Returns:
        --------
        logp: float
            The logarithm of the prior distribution for the model and
            parameters given.
        """

        if len(t0) != len(free_params):
            raise IncorrectParameterError("The number of parameters passed into "
                                          "the prior does not match the number "
                                          "of parameters in the model.")

        logp = 0.0 # initialize log-prior
        ii = 0 # counter for the variable parameter

        # loop through all parameter names, but only compute
        # prior for those that are not fixed
        # Note: need to do it this way to preserve order of parameters
        # correctly!
        for pname in lpost.model.param_names:
            if not lpost.model.fixed[pname]:
                with warnings.catch_warnings(record=True) as out:
                    logp += np.log(priors[pname](t0[ii]))
                    if len(out) > 0:
                        if isinstance(out[0].message, RuntimeWarning):
                            logp = np.nan

                ii += 1

        if not np.isfinite(logp):
            logp = logmin

        if neg:
            return -logp
        else:
            return logp

    return logprior


@six.add_metaclass(abc.ABCMeta)
class LogLikelihood(object):

    def __init__(self, x, y, model, **kwargs):
        """
        x : iterable
            x-coordinate of the data. Could be multi-dimensional.

        y : iterable
            y-coordinate of the data. Could be multi-dimensional.

        model : probably astropy.modeling.FittableModel instance
            Your model

        kwargs :
            keyword arguments specific to the individual sub-classes. For
            details, see the respective docstrings for each subclass

        """
        self.x = x
        self.y = y

        self.model = model

    @abc.abstractmethod
    def evaluate(self, parameters):
        """
        This is where you define your log-likelihood. Do this!

        """
        pass

    def __call__(self, parameters, neg=False):
        return self.evaluate(parameters, neg)


class GaussianLogLikelihood(LogLikelihood):

    def __init__(self, x, y, yerr, model):
        """
        A Gaussian likelihood.

        Parameters
        ----------
        x : iterable
            x-coordinate of the data

        y : iterable
            y-coordinte of the data

        yerr: iterable
            the uncertainty on the data, as standard deviation

        model: an Astropy Model instance
            The model to use in the likelihood.

        """

        self.x = x
        self.y = y
        self.yerr = yerr
        self.model = model

        self.npar = 0
        for pname in self.model.param_names:
            if not self.model.fixed[pname]:
                self.npar += 1


    def evaluate(self, pars, neg=False):
        if np.size(pars) != self.npar:
            raise IncorrectParameterError("Input parameters must" +
                                          " match model parameters!")

        _fitter_to_model_params(self.model, pars)

        mean_model = self.model(self.x)

        loglike = np.sum(-0.5*np.log(2.*np.pi) - np.log(self.yerr) -
                         (self.y-mean_model)**2/(2.*self.yerr**2))

        if not np.isfinite(loglike):
            loglike = logmin

        if neg:
            return -loglike
        else:
            return loglike


class PoissonLogLikelihood(LogLikelihood):

    def __init__(self, x, y, model):
        """
        A Gaussian likelihood.

        Parameters
        ----------
        x : iterable
            x-coordinate of the data

        y : iterable
            y-coordinte of the data

        model: an Astropy Model instance
            The model to use in the likelihood.

        """

        self.x = x
        self.y = y
        self.model = model
        self.npar = 0
        for pname in self.model.param_names:
            if not self.model.fixed[pname]:
                self.npar += 1

    def evaluate(self, pars, neg=False):

        if np.size(pars) != self.npar:
            raise IncorrectParameterError("Input parameters must" +
                                          " match model parameters!")

        _fitter_to_model_params(self.model, pars)

        mean_model = self.model(self.x)

        loglike = np.sum(-mean_model + self.y*np.log(mean_model) \
               - scipy_gammaln(self.y + 1.))

        if not np.isfinite(loglike):
            loglike = logmin

        if neg:
            return -loglike
        else:
            return loglike


class PSDLogLikelihood(LogLikelihood):

    def __init__(self, freq, power, model, m=1):
        """
        A Gaussian likelihood.

        Parameters
        ----------
        freq: iterable
            Array with frequencies

        power: iterable
            Array with (averaged/singular) powers corresponding to the
            frequencies in `freq`

        model: an Astropy Model instance
            The model to use in the likelihood.

        m : int
            1/2 of the degrees of freedom

        """

        LogLikelihood.__init__(self, freq, power, model)

        self.m = m
        self.npar = 0
        for pname in self.model.param_names:
            if not self.model.fixed[pname] and not self.model.tied[pname]:
                self.npar += 1

    def evaluate(self, pars, neg=False):

        if np.size(pars) != self.npar:
            raise IncorrectParameterError("Input parameters must" +
                                          " match model parameters!")

        _fitter_to_model_params(self.model, pars)

        mean_model = self.model(self.x)

        with warnings.catch_warnings(record=True) as out:

            if self.m == 1:
                loglike = -np.sum(np.log(mean_model)) - \
                          np.sum(self.y/mean_model)

            else:

                    loglike = -2.0*self.m*(np.sum(np.log(mean_model)) +
                                       np.sum(self.y/mean_model) +
                                       np.sum((2.0 / (2. * self.m) - 1.0) *
                                              np.log(self.y)))

        if not np.isfinite(loglike):
            loglike = logmin

        if neg:
            return -loglike
        else:
            return loglike

class Posterior(object):

    def __init__(self, x, y, model, **kwargs):
        """
        Define a posterior object.

        The posterior describes the Bayesian probability distribution of
        a set of parameters $\theta$ given some observed data $D$ and
        some prior assumptions $I$.

        It is defined as

            $p(\theta | D, I) = p(D | \theta, I) p(\theta | I)/p(D| I)

        where $p(D | \theta, I)$ describes the likelihood, i.e. the
        sampling distribution of the data and the (parametric) model, and
        $p(\theta | I)$ describes the prior distribution, i.e. our information
        about the parameters $\theta$ before we gathered the data.
        The marginal likelihood $p(D| I)$ describes the probability of
        observing the data given the model assumptions, integrated over the
        space of all parameters.

        Parameters
        ----------
        x : iterable
            The abscissa or independent variable of the data. This could
            in principle be a multi-dimensional array.

        y : iterable
            The ordinate or dependent variable of the data.

        model: astropy.modeling.models class instance
            The parametric model supposed to represent the data. For details
            see the astropy.modeling documentation

        kwargs :
            keyword arguments related to the subclases of `Posterior`. For
            details, see the documentation of the individual subclasses

        References
        ----------

        * Sivia, D. S., and J. Skilling. "Data Analysis:
            A Bayesian Tutorial. 2006."
        * Gelman, Andrew, et al. Bayesian data analysis. Vol. 2. Boca Raton,
            FL, USA: Chapman & Hall/CRC, 2014.
        * von Toussaint, Udo. "Bayesian inference in physics."
            Reviews of Modern Physics 83.3 (2011): 943.
        * Hogg, David W. "Probability Calculus for inference".
            arxiv: 1205.4446

        """
        self.x = x
        self.y = y

        self.model = model

        self.npar = 0
        for pname in self.model.param_names:
            if not self.model.fixed[pname]:
                self.npar += 1

    def logposterior(self, t0, neg=False):

        if not hasattr(self, "logprior"):
            raise PriorUndefinedError("There is no prior implemented. " +
                                      "Cannot calculate posterior!")

        if not hasattr(self, "loglikelihood"):
            raise LikelihoodUndefinedError("There is no likelihood implemented. " +
                                           "Cannot calculate posterior!")

        lpost = self.loglikelihood(t0) + self.logprior(t0)

        if neg is True:
            return -lpost
        else:
            return lpost

    def __call__(self, t0, neg=False):
        return self.logposterior(t0, neg=neg)


class PSDPosterior(Posterior):

    def __init__(self, freq, power, model, priors=None, m=1):
        """
        Posterior distribution for power spectra.
        Uses an exponential distribution for the errors in the likelihood,
        or a $\chi^2$ distribution with $2M$ degrees of freedom, where $M$ is
        the number of frequency bins or power spectra averaged in each bin.


        Parameters
        ----------
        ps: {Powerspectrum | AveragedPowerspectrum} instance
            the Powerspectrum object containing the data

        model: instance of any subclass of parameterclass.ParametricModel
            The model for the power spectrum. Note that in order to define
            the posterior properly, the ParametricModel subclass must be
            instantiated with the hyperpars parameter set, or there won't
            be a prior to be calculated! If all this object is used
            for a maximum likelihood-style analysis, no prior is required.

        priors : dict of form {"parameter name": function}, optional
            A dictionary with the definitions for the prior probabilities.
            For each parameter in `model`, there must be a prior defined with
            a key of the exact same name as stored in `model.param_names`.
            The item for each key is a function definition defining the prior
            (e.g. a lambda function or a `scipy.stats.distribution.pdf`.
            If `priors = None`, then no prior is set. This means priors need
            to be added by hand using the `set_logprior` function defined in
            this module. Note that it is impossible to call the posterior object
            itself or the `self.logposterior` method without defining a prior.

        m: int, default 1
            The number of averaged periodograms or frequency bins in ps.
            Useful for binned/averaged periodograms, since the value of
            m will change the likelihood function!

        Attributes
        ----------
        ps: {Powerspectrum | AveragedPowerspectrum} instance
            the Powerspectrum object containing the data

        x: numpy.ndarray
            The independent variable (list of frequencies) stored in ps.freq

        y: numpy.ndarray
            The dependent variable (list of powers) stored in ps.power

        model: instance of any subclass of parameterclass.ParametricModel
               The model for the power spectrum. Note that in order to define
               the posterior properly, the ParametricModel subclass must be
               instantiated with the hyperpars parameter set, or there won't
               be a prior to be calculated! If all this object is used
               for a maximum likelihood-style analysis, no prior is required.

        """
        self.loglikelihood = PSDLogLikelihood(freq, power,
                                              model, m=m)

        self.m = m
        Posterior.__init__(self, freq, power, model)

        if not priors is None:
            self.logprior = set_logprior(self, priors)


class PoissonPosterior(Posterior):

    def __init__(self, x, y, model, priors=None):
        """
        Posterior for Poisson lightcurve data. Primary intended use is for
        modelling X-ray light curves, but alternative uses are conceivable.

        TODO: Include astropy.modeling models

        Parameters
        ----------
        x : numpy.ndarray
            The independent variable (e.g. time stamps of a light curve)

        y : numpy.ndarray
            The dependent variable (e.g. counts per bin of a light curve)

        model: instance of any subclass of parameterclass.ParametricModel
            The model for the power spectrum. Note that in order to define
            the posterior properly, the ParametricModel subclass must be
            instantiated with the hyperpars parameter set, or there won't
            be a prior to be calculated! If all this object is used
            for a maximum likelihood-style analysis, no prior is required.

        priors : dict of form {"parameter name": function}, optional
            A dictionary with the definitions for the prior probabilities.
            For each parameter in `model`, there must be a prior defined with
            a key of the exact same name as stored in `model.param_names`.
            The item for each key is a function definition defining the prior
            (e.g. a lambda function or a `scipy.stats.distribution.pdf`.
            If `priors = None`, then no prior is set. This means priors need
            to be added by hand using the `set_logprior` function defined in
            this module. Note that it is impossible to call the posterior object
            itself or the `self.logposterior` method without defining a prior.

        Attributes
        ----------
        x: numpy.ndarray
            The independent variable (list of frequencies) stored in ps.freq

        y: numpy.ndarray
            The dependent variable (list of powers) stored in ps.power

        model: instance of any subclass of parameterclass.ParametricModel
               The model for the power spectrum. Note that in order to define
               the posterior properly, the ParametricModel subclass must be
               instantiated with the hyperpars parameter set, or there won't
               be a prior to be calculated! If all this object is used
               for a maximum likelihood-style analysis, no prior is required.

        """
        self.x = x
        self.y = y

        self.loglikelihood = PoissonLogLikelihood(self.x, self.y, model)

        Posterior.__init__(self, self.x, self.y, model)

        if not priors is None:
            self.logprior = set_logprior(self, priors)


class GaussianPosterior(Posterior):

    def __init__(self, x, y, yerr, model, priors=None):
        """
        A general class for two-dimensional data following a Gaussian
        sampling distribution.

        Parameters
        ----------
        x: numpy.ndarray
            independent variable

        y: numpy.ndarray
            dependent variable

        yerr: numpy.ndarray
            measurement uncertainties for y

        model: instance of any subclass of parameterclass.ParametricModel
            The model for the power spectrum. Note that in order to define
            the posterior properly, the ParametricModel subclass must be
            instantiated with the hyperpars parameter set, or there won't
            be a prior to be calculated! If all this object is used
            for a maximum likelihood-style analysis, no prior is required.

        """
        self.loglikelihood = GaussianLogLikelihood(x, y, yerr, model)

        Posterior.__init__(self, x, y, model)

        self.yerr = yerr

        if not priors is None:
            self.logprior = set_logprior(self, priors)
