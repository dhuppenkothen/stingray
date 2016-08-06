from __future__ import division

import abc

import numpy as np
from scipy.special import gamma as scipy_gamma

# TODO: Find out whether there is a gamma function in numpy!

from stingray.modeling.parametricmodels import logmin

__all__ = ["Posterior", "PSDPosterior",
           "PoissonPosterior", "GaussianPosterior"]


class Posterior(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, x, y, model):
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

        model: ParametricModel subclass instance
            The parametric model supposed to represent the data. Has to be
             an instance of a subclass of ParametricModel.


        References
        ----------

        * Sivia, D. S., and J. Skilling. "Data Analysis:
            A Bayesian Tutorial. 2006."
        * Gelman, Andrew, et al. Bayesian data analysis. Vol. 2. Boca Raton,
            FL, USA: Chapman & Hall/CRC, 2014.
        * von Toussaint, Udo. "Bayesian inference in physics."
            Reviews of Modern Physics 83.3 (2011): 943.
        *Hogg, David W. "Probability Calculus for inference".
            arxiv: 1205.4446

        """
        self.x = x
        self.y = y

        self.model = model

    def logprior(self, t0):
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
        assert hasattr(self.model, "logprior")
        assert np.size(t0) == self.model.npar, "Input parameters must " \
                                               "match model parameters!"

        return self.model.logprior(*t0)

    @abc.abstractmethod
    def loglikelihood(self, t0, neg=False):
        pass

    def logposterior(self, t0, neg=False):
        lpost = self.loglikelihood(t0) + self.logprior(t0)
        if neg is True:
            return -lpost
        else:
            return lpost

    def __call__(self, t0, neg=False):
        return self.logposterior(t0, neg=neg)


class PSDPosterior(Posterior):

    def __init__(self, ps, model):
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


        Attributes
        ----------
        ps: {Powerspectrum | AveragedPowerspectrum} instance
            the Powerspectrum object containing the data

        m: int, optional, default is 1
            The number of averaged periodograms or frequency bins in ps.
            Useful for binned/averaged periodograms, since the value of
            m will change the likelihood function!

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
        self.m = ps.m
        Posterior.__init__(self, ps.freq, ps.power, model)

    def loglikelihood(self, t0, neg=False):
        """
        The log-likelihood for the model defined in self.model
        and the parameters in t0. Uses a $\Chi^2$ model for
        the uncertainty.

        Parameters:
        ------------
        t0: {list | numpy.ndarray}
            The list with parameters for the model

        Returns:
        --------
        logl: float
            The logarithm of the likelihood function for the model and
            parameters given.

        """
        assert np.size(t0) == self.model.npar, "Input parameters must" \
                                               " match model parameters!"

        funcval = self.model(self.x[1:], *t0)

        if self.m == 1:
            res = -np.sum(np.log(funcval)) - np.sum(self.y[1:]/funcval)
        else:
            res = -2.0*self.m*(np.sum(np.log(funcval)) +
                               np.sum(self.y[1:]/funcval) +
                               np.sum((2.0 / (2. * self.m) - 1.0) *
                                      np.log(self.y[1:])))

        if np.isfinite(res) is False:
            res = logmin

        if neg:
            return -res
        else:
            return res


class PoissonPosterior(Posterior):

    def __init__(self, lc, model):
        """
        Posterior for Poisson lightcurve data. Primary intended use is for
        modelling X-ray light curves, but alternative uses are conceivable.

        Parameters
        ----------
        lc: lightcurve.Lightcurve object
            Object containing the light curve to be modelled

        model: instance of any subclass of parameterclass.ParametricModel
            The model for the power spectrum. Note that in order to define
            the posterior properly, the ParametricModel subclass must be
            instantiated with the hyperpars parameter set, or there won't
            be a prior to be calculated! If all this object is used
            for a maximum likelihood-style analysis, no prior is required.
        Attributes
        ----------
        lc: lightcurve.Lightcurve instance
            the Lightcurve object containing the data

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
        self.lc = lc
        Posterior.__init__(self, lc.time, lc.counts, model)

    def loglikelihood(self, t0, neg=False):

        assert np.size(t0) == self.model.npar, "Input parameters must" \
                                               " match model parameters!"

        funcval = self.model(self.x[1:], *t0)

        # THIS IS WRONG! FIX THIS!
        res = -funcval + self.y*np.log(funcval) - scipy_gamma(self.y + 1.)

        if np.isfinite(res) is False:
            res = logmin

        if neg:
            return -res
        else:
            return res


class GaussianPosterior(Posterior):

    def __init__(self, x, y, model):
        """
        A general class for two-dimensional data following a Gaussian
        sampling distribution.

        Parameters
        ----------
        x: numpy.ndarray
            independent variable

        y: numpy.ndarray
            dependent variable

        model: instance of any subclass of parameterclass.ParametricModel
            The model for the power spectrum. Note that in order to define
            the posterior properly, the ParametricModel subclass must be
            instantiated with the hyperpars parameter set, or there won't
            be a prior to be calculated! If all this object is used
            for a maximum likelihood-style analysis, no prior is required.

        """
        Posterior.__init__(self, x, y, model)

    def loglikelihood(self, t0, neg=False):

        assert np.size(t0) == self.model.npar, "Input parameters must" \
                                               " match model parameters!"

        funcval = self.model(self.x[1:], *t0)

        # TODO: Add Gaussian likelihood
        res = 1.0

        if np.isfinite(res) is False:
            res = logmin

        if neg:
            return -res
        else:
            return res
