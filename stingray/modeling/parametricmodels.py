
__all__ = ["ParametricModel", "Const",
           "PowerLaw", "PowerLawConst",
           "BrokenPowerLaw", "BrokenPowerLawConst",
           "Lorentzian", "FixedCentroidLorentzian",
           "CombinedModel"]

from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.stats

logmin = -10000000000000000.0


class ParametricModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, npar, name, parnames=None):
        """
        The superclass container for all parametric models
        to be used in Maximum-A-Posteriori estimates, Maximum Likelihood
        Estimates and Markov Chain Monte Carlo.

        Do not call this class in itself; it does not define a model!
        Instead, use one of the subclass below or add your own!

        Parameters
        ----------
        npar: int
            The number of parameters of the model

        name: string
            A string describing the name of the model

        parnames: {iterable of strings | None}, optional, default None
            Optional keyword allowing the user to set names for the
            parameters of the model, to be used for example in plotting
            MCMC chains.

        """
        if not isinstance(npar, int):
            raise ValueError("npar must be an integer!")

        self.npar = npar
        self.name = name

        if parnames is not None:
            self.parnames = parnames

    @abstractmethod
    def func(self, *pars):
        pass

    def __call__(self, freq, *pars):
        return self.func(freq, *pars)


class Const(ParametricModel):

    def __init__(self, hyperpars=None):

        """
        A constant model of the form

            $y(x) = a$

        The model has a single parameter $a$ describing the constant
        level.
        By default, this model defines a Gaussian prior in set_prior with
        parameters `a_mean` and `a_var` to be set in `hyperpars`.
        If the user wishes to define their own prior that differs,
        they can subclass this class and replace `set_prior` with their own
        prior definition.

        Parameters
        ----------
        hyperpars: {dict | None}, optional, default None
            If None (default), no prior will be set for the model. If not
            None, it must be a dictionary with two entries:

                `a_mean`: mean of the Gaussian prior distribution and
                `a_var`: variance of the Gaussian prior distribution

            defined in `set_prior`.


        Attributes
        ----------
        npar: int
            The number of parameters of the model

        name: string
            A string describing the name of the model

        parnames: iterable of strings
            Definition of the names for the parameters of the model,
            to be used for example in plotting MCMC chains.


        """

        npar = 1
        name = "const"
        parnames = ["amplitude"]

        ParametricModel.__init__(self, npar, name, parnames)

        if hyperpars is not None:
            self.set_prior(hyperpars)

    def set_prior(self, hyperpars):
        """
        Set a Gaussian prior for the constant model.

        Parameters:
        -----------
        a_mean: float
            Mean of the Gaussian distribution
        a_var: float
            Variance of the Gaussian distribution
        """

        a_mean = hyperpars["a_mean"]
        a_var = hyperpars["a_var"]

        def logprior(a):

            if not np.isfinite(a):
                raise ValueError("A must be finite.")

            pp = scipy.stats.norm.pdf(a, a_mean, a_var)
            if pp == 0.0:
                return logmin
            else:
                return np.log(pp)

        self.logprior = logprior

    def func(self, x, a):
        """
        A constant model of the form

        $y(x) = a$

        Parameters:
        ------------
        x: numpy.ndarray
            The independent variable
        a: float
            The amplitude of the constant model

        Returns:
        --------
        model: numpy.ndarray
            The power law model for all values in x.
        """
        if not np.isfinite(a):
            raise ValueError("A must be finite.")
        return np.ones_like(x)*a


class PowerLaw(ParametricModel):

    def __init__(self, hyperpars=None):
        """
        A power law model of the form

            $y(x) = Ax^\alpha$

        The model has a two parameters $\alpha$  and $A$ describing the
        power law index and its amplitude, respectively.

        By default, this model defines a flat prior between
        $alpha_{\mathrm{min}}$ and $alpha_{\mathrm{max}}$ for the power law
        index and a Jeffrey's prior (flat prior on $\log(A)$) between
        $A_\mathrm{min}$ and $A_\mathrm{max}$ for the amplitude.
        The priors are to be set in `hyperpars`.

        If the user wishes to define their own prior that differs,
        they can subclass this class and replace `set_prior` with their own
        prior definition.

        Parameters
        ----------
        hyperpars: {dict | None}, optional, default None
            If None (default), no prior will be set for the model. If not
            None, it must be a dictionary with four entries

                `alpha_min`: lower edge of the uniform prior distribution for
                             the power law index
                `alpha_max`: upper edge of the uniform prior distribution for
                             the power law index
                `amplitude_min`: lower edge of the uniform prior distribution
                                 for the log(amplitude)
                `amplitude_max`: upper edge of the uniform prior distribution
                                 for the log(amplitude)

            describing the hyperparameters for the distributions used
            in `set_prior` to define the prior.

        Attributes
        ----------
        npar: int
            The number of parameters of the model

        name: string
            A string describing the name of the model

        parnames: iterable of strings
            Definition of the names for the parameters of the model,
            to be used for example in plotting MCMC chains.


        """
        npar = 2  # number of parameters in the model
        name = "powerlaw"  # model name
        parnames = ["alpha", "amplitude"]
        ParametricModel.__init__(self, npar, name, parnames)

        if hyperpars is not None:
            self.set_prior(hyperpars)

    def set_prior(self, hyperpars):
        """
        Set the hyper parameters for the power law parameters.
        The power law index alpha has a flat prior over the specified range,
        the amplitude is defined such that the log-amplitude has a flat
        prior over a given range, too, which translates to an exponential prior
        beween the specified ranges.

        Parameters:
        -----------
        alpha_min, alpha_max: float, float
            The minimum and maximum values for the power-law index.
        amplitude_min, amplitude_max: float, float
            The minimum and maximum values for the log-amplitude
        """

        alpha_min = hyperpars["alpha_min"]
        alpha_max = hyperpars["alpha_max"]
        amplitude_min = hyperpars["amplitude_min"]
        amplitude_max = hyperpars["amplitude_max"]

        def logprior(alpha, amplitude):

            if not np.isfinite(alpha):
                raise ValueError("alpha must be finite.")

            if not np.isfinite(amplitude):
                raise ValueError("amplitude must be finite.")

            p_alpha = (alpha_min <= alpha <= alpha_max) / \
                      (alpha_max-alpha_min)
            p_amplitude = (amplitude_min <= amplitude <= amplitude_max) / \
                          (amplitude_max-amplitude_min)

            pp = p_alpha*p_amplitude

            if pp == 0:
                return logmin
            else:
                return np.log(pp)

        self.logprior = logprior

    def func(self, x, alpha, amplitude):
        """
        Power law model of the form

            $y(x) = A x^\alpha$

        where $A$ is the amplitude and $\alpha$ the power law index.

        Parameters:
        -----------
        x: numpy.ndarray
            The independent variable
        alpha: float
            The  power law index
        amplitude: float
            The *logarithm* of the normalization or amplitude of the power law

        Returns:
        --------
        model: numpy.ndarray
            The power law model for all values in x.
        """
        if not np.isfinite(alpha):
            raise ValueError("alpha must be finite.")

        if not np.isfinite(amplitude):
            raise ValueError("amplitude must be finite.")


        res = -alpha*np.log(x) + amplitude
        return np.exp(res)


class BrokenPowerLaw(ParametricModel):

    def __init__(self, hyperpars=None):
        """
        A broken power law model of the form

            $y(x) = Ax^{-1}(1 + (x/x_\mathrm{break})^{\alpha_1-alpha_2}^
            {-alpha_2}$

        The model has a variable bending parameter (setting the sharpness
        of the bend) a four parameters:
            $\alpha_1$: the power law index for values below the break
            $\alpha_2$: the power law index for values above the break
            $x_\mathrm{break}$: the location of the break between the power law
            indices
            $A$: the amplitude (the log-amplitude in the function definition)

        By default, this model defines uniform priors for $\alpha_1$, $\alpha_2$
        and $x_\mathrm{break}$ and a Jeffrey's prior (flat prior on $\log(A)$)
        for the amplitude.
        The hyperparameters for the priors are to be set in `hyperpars`.

        If the user wishes to define their own prior that differs,
        they can subclass this class and replace `set_prior` with their own
        prior definition.

        Parameters
        ----------
        hyperpars: {dict | None}, optional, default None
            If None (default), no prior will be set for the model. If not
            None, it must be a dictionary with eight entries

                `alpha1_min`: lower edge of the uniform prior distribution for
                             the power law index below the break
                `alpha1_max`: upper edge of the uniform prior distribution for
                             the power law index below the break
                `alpha2_min`: lower edge of the uniform prior distribution for
                             the power law index above the break
                `alpha2_max`: upper edge of the uniform prior distribution for
                             the power law index above the break
                `x_break_min`: lower edge of teh uniform prior distribution for
                               the location of the break between power laws
                `x_break_max`: upper edge of teh uniform prior distribution for
                               the location of the break between power laws
                `amplitude_min`: lower edge of the uniform prior distribution
                                 for the log(amplitude)
                `amplitude_max`: upper edge of the uniform prior distribution
                                 for the log(amplitude)

            describing the hyperparameters for the distributions used
            in `set_prior` to define the prior.

        Attributes
        ----------
        npar: int
            The number of parameters of the model

        name: string
            A string describing the name of the model

        parnames: iterable of strings
            Definition of the names for the parameters of the model,
            to be used for example in plotting MCMC chains.

        """

        npar = 4
        name = "bentpowerlaw"
        parnames = ["alpha1", "amplitude", "alpha2", "x_break"]
        ParametricModel.__init__(self, npar, name, parnames)

        if hyperpars is not None:
            self.set_prior(hyperpars)

    def set_prior(self, hyperpars):

        alpha1_min = hyperpars["alpha1_min"]
        alpha1_max = hyperpars["alpha1_max"]
        amplitude_min = hyperpars["amplitude_min"]
        amplitude_max = hyperpars["amplitude_max"]
        alpha2_min = hyperpars["alpha2_min"]
        alpha2_max = hyperpars["alpha2_max"]
        x_break_min = hyperpars["x_break_min"]
        x_break_max = hyperpars["x_break_max"]

        def logprior(alpha1, alpha2, x_break, amplitude):

            for p, n in zip([alpha1, alpha2, x_break, amplitude],
                            self.parnames):

                if not np.isfinite(p):
                    raise ValueError("%s must be finite!"%n)

            p_alpha1 = (alpha1_min <= alpha1 <= alpha1_max) / \
                       (alpha1_max-alpha1_min)
            p_amplitude = (amplitude_min <= amplitude <= amplitude_max) / \
                          (amplitude_max-amplitude_min)
            p_alpha2 = (alpha2_min <= alpha2 <= alpha2_max) / \
                       (alpha2_max-alpha2_min)
            p_x_break = (x_break_min <= x_break <= x_break_max) / \
                        (x_break_max-x_break_min)

            pp = p_alpha1 * p_amplitude * p_alpha2 * p_x_break

            if pp == 0.0:
                return logmin
            else:
                return np.log(pp)

        self.logprior = logprior

    def func(self, x, alpha1, alpha2, x_break, amplitude):
        """
        A broken power law of the form

            $y(x) = Ax^{-1}(1 + (x/x_\mathrm{break})^{\alpha_1-alpha_2}^
            {-alpha_2}$


        Parameters:
        -----------
        x: numpy.ndarray
            The independent variable
        alpha1: float
            The  power law index at small x
        alpha2: float
            The power law index at large x
        x_break: float
            The log-position in x where the break between alpha1 and alpha2
            occurs
        amplitude: float
            The log-normalization or log-amplitude of the bent power law

        """
        # compute bending factor

        for p, n in zip([alpha1, alpha2, x_break, amplitude], self.parnames):

            if not np.isfinite(p):
                raise ValueError("%s must be finite!"%n)

        logz = (alpha2 - alpha1)*(np.log(x) - x_break)

        # be careful with very large or very small values
        logqsum = sum(np.where(logz < -100, 1.0, 0.0))
        if logqsum > 0.0:
            logq = np.where(logz < -100, 1.0, logz)
        else:
            logq = logz
        logqsum = np.sum(np.where((-100 <= logz) & (logz <= 100.0),
                                  np.log(1.0 + np.exp(logz)), 0.0))
        if logqsum > 0.0:
            logqnew = np.where((-100 <= logz) & (logz <= 100.0),
                               np.log(1.0 + np.exp(logz)), logq)
        else:
            logqnew = logq

        logy = -alpha1*np.log(x) - logqnew + amplitude
        return np.exp(logy)


class Lorentzian(ParametricModel):

    def __init__(self, hyperpars=None):
        """
        A Lorentzian function of the form

            $y(x) = \frac{A\gamma/}{2\pi((x-x_0)^2 + (\gamma/2)^2)}$

        The model has three parameters:
            $x_0$: centroid location of the Lorentzian, i.e. the location of
                    the peak
            $\gamma$: the width of the Lorentzian (log-width in the function
                      definition)
            $A$: the amplitude (the log-amplitude in the function definition)

        By default, this model defines a uniform prior for $x_0$ and Jeffrey's
        priors (flat prior on log(parameter)) for the the width $\gamma$ and
        the amplitude $A$.
        The hyperparameters for the priors are to be set in `hyperpars`.

        If the user wishes to define their own prior that differs,
        they can subclass this class and replace `set_prior` with their own
        prior definition.

        Parameters
        ----------
        hyperpars: {dict | None}, optional, default None
            If None (default), no prior will be set for the model. If not
            None, it must be a dictionary with eight entries

                `x0_min`: lower edge of the uniform prior distribution for
                             the centroid location of the Lorentzian
                `x0_max`: upper edge of the uniform prior distribution for
                             the centroid location of the Lorentzian
                `gamma_min`: lower edge of the uniform prior distribution for
                             the log(width) of the Lorentzian
                `gamma_max`: upper edge of the uniform prior distribution for
                             the log(width) of the Lorentzian
                `amplitude_min`: lower edge of the uniform prior distribution
                                 for the log(amplitude)
                `amplitude_max`: upper edge of the uniform prior distribution
                                 for the log(amplitude)

            describing the hyperparameters for the distributions used
            in `set_prior` to define the prior.

        Attributes
        ----------
        npar: int
            The number of parameters of the model

        name: string
            A string describing the name of the model

        parnames: iterable of strings
            Definition of the names for the parameters of the model,
            to be used for example in plotting MCMC chains.

        """
        npar = 3
        name = "lorentzian"
        parnames = ["x0", "gamma", "A"]
        ParametricModel.__init__(self, npar, name, parnames)
        if hyperpars is not None:
            self.set_prior(hyperpars)

    def set_prior(self, hyperpars):
        x0_min = hyperpars["x0_min"]
        x0_max = hyperpars["x0_max"]
        gamma_min = hyperpars["gamma_min"]
        gamma_max = hyperpars["gamma_max"]
        amplitude_min = hyperpars["amplitude_min"]
        amplitude_max = hyperpars["amplitude_max"]

        def logprior(x0, gamma, amplitude):

            for p, n in zip([x0, gamma, amplitude], self.parnames):
                if not np.isfinite(p):
                    raise ValueError("%s must be finite!"%n)

            p_gamma = (gamma_min <= gamma <= gamma_max) / \
                      (gamma_max - gamma_min)
            p_amplitude = (amplitude_min <= amplitude <= amplitude_max) / \
                          (amplitude_max - amplitude_min)

            p_x0 = (x0_min <= x0 <= x0_max) / (x0_max - x0_min)

            pp = p_gamma * p_amplitude * p_x0

            if pp == 0.0:
                return logmin
            else:
                return np.log(pp)

        self.logprior = logprior

    def func(self, x, x0, gamma, amplitude):
        """
        Lorentzian profile commonly used for fitting QPOs.

        Parameters:
        -----------
        x: numpy.ndarray
            The independent variable
        x0: float
            The position of the centroid of the Lorentzian profile
        gamma: float
            The width of the Lorentzian profile
        amplitude: float
            The height or amplitude of the Lorentzian profile
        """

        for p, n in zip([x0, gamma, amplitude], self.parnames):
            if not np.isfinite(p):
                raise ValueError("%s must be finite!"%n)

        gamma = np.exp(gamma)
        amplitude = np.exp(amplitude)

        alpha = 0.5 * amplitude / (gamma * np.pi)
        y = alpha / ((x - x0)**2.0 + (0.5 * gamma)**2.0)
        return y


class FixedCentroidLorentzian(ParametricModel):

    def __init__(self, x0, hyperpars=None):

        assert np.isfinite(x0), "x0 must be finite!"

        self.x0 = x0
        npar = 2
        name = "fixedcentroidlorentzian"
        parnames = ["gamma", "A"]

        ParametricModel.__init__(self, npar, name, parnames)
        if hyperpars is not None:
            self.set_prior(hyperpars)

    def set_prior(self, hyperpars):

        gamma_min = hyperpars["gamma_min"]
        gamma_max = hyperpars["gamma_max"]
        amplitude_min = hyperpars["amplitude_min"]
        amplitude_max = hyperpars["amplitude_max"]

        def logprior(gamma, amplitude):

            if not np.isfinite(gamma):
                raise ValueError( "gamma must be finite")

            if not np.isfinite(amplitude):
                raise ValueError("amplitude must be finite")

            p_gamma = (gamma_min <= gamma <= gamma_max) / \
                      (gamma_max-gamma_min)
            p_amplitude = (amplitude_min <= amplitude <= amplitude_max) / \
                          (amplitude_max-amplitude_min)

            pp = p_gamma * p_amplitude
            if pp == 0.0:
                return logmin
            else:
                return np.log(pp)

        self.logprior = logprior

    def func(self, x, gamma, amplitude):
        """
        Lorentzian profile for fitting QPOs.

        Parameters:
        -----------
        x: numpy.ndarray
            The independent variable
        gamma: float
            The width of the Lorentzian profile
        amplitude: float
            The height or amplitude of the Lorentzian profile
        """

        if not np.isfinite(gamma):
            raise ValueError( "gamma must be finite")

        if not np.isfinite(amplitude):
            raise ValueError("amplitude must be finite")

        y = Lorentzian().func(x, self.x0, gamma, amplitude)
        return y


class Gaussian(ParametricModel):

    def __init__(self):
        # TODO: make a Gaussian parametric model
        pass

    def func(self):
        pass


class PowerLawConst(ParametricModel):

    def __init__(self, hyperpars=None):

        self.models = [PowerLaw(hyperpars), Const(hyperpars)]

        npar = 3  # number of parameters in the model
        name = "powerlawconst"  # model name
        parnames = ["alpha", "amplitude", "const"]
        ParametricModel.__init__(self, npar, name, parnames)

        if hyperpars is not None:
            self.set_prior()

    def set_prior(self):
        """
        Set the prior for the combined power law + constant model.
        The hyperparameters for each individual model are set in the logprior
        function of each model itself. All that's left to do here is add them
        together and make sure the parameters get distributed in the right way.

        """
        def logprior(alpha, amplitude, const):
            for p, n in zip([alpha, amplitude, const], self.parnames):
                if not np.isfinite(p):
                    raise ValueError("%s must be finite!"%n)

            pp = self.models[0].logprior(alpha, amplitude) + \
                 self.models[1].logprior(const)

            return pp

        self.logprior = logprior

    def func(self, x, alpha, amplitude, const):
        """
        Power law model + constant

        Parameters:
        -----------
        x: numpy.ndarray
            The independent variable
        alpha: float
            The  power law index
        amplitude: float
            The *logarithm* of the normalization or amplitude of the power law
        const: float
            The *logarithm* of the constant background level

        Returns:
        --------
        model: numpy.ndarray
            The power law model for all values in x.
        """

        for p, n in zip([alpha, amplitude, const], self.parnames):
           if not np.isfinite(p):
                raise ValueError("%s must be finite!"%n)

        res = self.models[0].func(x, alpha, amplitude) + \
              self.models[1].func(x, const)

        return res


class BrokenPowerLawConst(ParametricModel):

    def __init__(self, hyperpars=None):

        self.models = [BrokenPowerLaw(hyperpars), Const(hyperpars)]

        npar = 5  # number of parameters in the model
        name = "bentpowerlawconst"  # model name
        parnames = ["alpha1", "amplitude", "alpha2", "x_break", "const"]

        ParametricModel.__init__(self, npar, name, parnames)

        if hyperpars is not None:
            self.set_prior()

    def set_prior(self):
        """
        Set the prior for the combined bent power law + constant model.
        The hyperparameters for each individual model are set in the logprior
        function of each model itself. All that's left to do here is add them
        together and make sure the parameters get distributed in the right way.

        """
        def logprior(alpha1, amplitude, alpha2, x_break, const):
                pp = self.models[0].logprior(alpha1, amplitude, alpha2,
                                             x_break) + \
                     self.models[1].logprior(const)
                return pp

        self.logprior = logprior

    def func(self, x, alpha1, amplitude, alpha2, x_break, const):
        """
        Bent Power law model + constant

        Parameters:
        -----------
        x: numpy.ndarray
            The independent variable
        alpha: float
            The  power law index
        amplitude: float
            The *logarithm* of the normalization or amplitude of the power law
        const: float
            The *logarithm* of the constant background level

        Returns:
        --------
        model: numpy.ndarray
            The power law model for all values in x.
        """
        res = self.models[0].func(x, alpha1, amplitude, alpha2, x_break) + \
              self.models[1].func(x, const)

        return res


class CombinedModel(object):

    def __init__(self, models, hyperpars=None):
        """
        Combine two or more parametric models into an additive
        mixture model.

        Parameters
        ----------

        models: iterable of ParametricModel definitions
            A list of models to be added to a final mixture model

        hyperpars: {iterable | None}, optional, default None
            If hyperpars is not None, __init__ will call the method
            set_priors to set the priors for this model. The length of
            hyperpars must match the sum of all parameters in all individual
            components in `models`.

        Example
        -------
        TODO: Add an example

        """

        # initialize all models
        self.models = [m() for m in models]

        # set the total number of parameters
        self.npar = np.sum([m.npar for m in self.models])

        # set the name for the combined model
        self.name = [m.name for m in self.models]

        # set hyperparameters
        if hyperpars is not None:
            for m in self.models:
                self.set_prior(hyperpars)

    def set_prior(self, hyperpars):

        def logprior(*pars):
            counter = 0
            pp = 0.
            for m in self.models:
                m.set_prior(hyperpars)
                pp += m.logprior(*pars[counter:counter+m.npar])
                counter += m.npar

            return pp

        self.logprior = logprior

    def func(self, x, *pars):
        model = np.zeros(x.shape[0])
        counter = 0
        for m in self.models:
            model += m.func(x, *pars[counter:counter+m.npar])
            counter += m.npar
        return model

    def __call__(self, x, *pars):
        return self.func(x, *pars)
