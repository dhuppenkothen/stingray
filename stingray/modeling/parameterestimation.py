from __future__ import print_function, division

__all__ = ["OptimizationResults", "ParameterEstimation", "PSDParEst",
           "SamplingResults"]


# check whether matplotlib is installed for easy plotting
try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    can_plot = True
except ImportError:
    can_plot = False


# check whether emcee is installed for sampling
try:
    import emcee
    can_sample = True
except ImportError:
    can_sample = False

try:
    import corner
    use_corner = True
except ImportError:
    use_corner = False

import logging

import numpy as np
import scipy
import scipy.optimize
import scipy.stats
import scipy.signal
import copy

try:
    from statsmodels.tools.numdiff import approx_hess
    comp_hessian = True
except ImportError:
    comp_hessian = False

from astropy.modeling.fitting import _fitter_to_model_params, \
    _model_to_fit_params, _validate_model, _convert_input

from stingray.modeling.posterior import Posterior, PSDPosterior, \
    LogLikelihood, PSDLogLikelihood


class OptimizationResults(object):

    def __init__(self, lpost, res, neg=True):
        """
        Helper class that will contain the results of the regression.
        Less fiddly than a dictionary.

        Parameters
        ----------
        lpost: instance of Posterior or one of its subclasses
            The object containing the function that is being optimized
            in the regression


        res: instance of scipy's OptimizeResult class
            The object containing the results from a optimization run

        """
        self.neg = neg
        self.result = res.fun
        self.p_opt = res.x
        self.model = lpost.model

        self._compute_covariance(lpost, res)
        self._compute_model(lpost)
        self._compute_criteria(lpost)
        self._compute_statistics(lpost)

    def _compute_covariance(self, lpost, res):

        if hasattr(res, "hess_inv"):
            if not isinstance(res.hess_inv, np.ndarray):
                self.cov = np.asarray(res.hess_inv.todense())
            else:
                self.cov = res.hess_inv

            self.err = np.sqrt(np.diag(self.cov))
        else:
            if comp_hessian:
                # calculate Hessian approximating with finite differences
                logging.info("Approximating Hessian with finite differences ...")

                phess = approx_hess(self.p_opt, lpost)

                self.cov = np.linalg.inv(phess)
                self.err = np.sqrt(np.diag(np.abs(self.cov)))

            else:
                self.cov = None
                self.err = None

    def _compute_model(self, lpost):

        _fitter_to_model_params(lpost.model, self.p_opt)

        self.mfit = lpost.model(lpost.x)

    def _compute_criteria(self, lpost):

        if isinstance(lpost, Posterior):
            self.deviance = -2.0*lpost.loglikelihood(self.p_opt, neg=False)
        elif isinstance(lpost, LogLikelihood):
            self.deviance = 2.0*self.result

        # Akaike Information Criterion
        self.aic = self.result+2.0*self.p_opt.shape[0]

        # Bayesian Information Criterion
        self.bic = self.result + self.p_opt.shape[0]*np.log(lpost.x.shape[0])

        # Deviance Information Criterion
        # TODO: Add Deviance Information Criterion

    def _compute_statistics(self, lpost):
        try:
            self.mfit
        except AttributeError:
            self._compute_model(lpost)

        self.merit = np.sum(((lpost.y-self.mfit)/self.mfit)**2.0)
        self.dof = lpost.y.shape[0] - float(self.p_opt.shape[0])
        self.sexp = 2.0*len(lpost.x)*len(self.p_opt)
        self.ssd = np.sqrt(2.0*self.sexp)
        self.sobs = np.sum(lpost.y-self.mfit)

    def print_summary(self, lpost):

        logging.info("The best-fit model parameters plus errors are:")

        fixed = [lpost.model.fixed[n] for n in lpost.model.param_names]
        tied = [lpost.model.tied[n] for n in lpost.model.param_names]
        bounds = [lpost.model.bounds[n] for n in lpost.model.param_names]

        parnames = [n for n, f in zip(lpost.model.param_names,
                                      np.logical_or(fixed, tied)) \
                    if f is False]

        all_parnames = [n for n in lpost.model.param_names]
        for i, par in enumerate(all_parnames):
            print("{:3}) Parameter {:<20}: ".format(i, par), end="")

            if par in parnames:
                idx = parnames.index(par)
                print("{:<20.5f} +/- {:<20.5f} ".format(self.p_opt[idx],
                                                  self.err[idx]), end="")
                print("[{:>10} {:>10}]".format(str(bounds[i][0]),
                                               str(bounds[i][1])))
            elif fixed[i]:
                print("{:<20.5f} (Fixed) ".format(lpost.model.parameters[i]))
            elif tied[i]:
                print("{:<20.5f} (Tied) ".format(lpost.model.parameters[i]))

        logging.info("\n")

        logging.info("Fitting statistics: ")
        logging.info(" -- number of data points: %i"%(len(lpost.x)))

        try:
            self.deviance
        except AttributeError:
            self._compute_criteria(lpost)

        logging.info(" -- Deviance [-2 log L] D = %f.3"%self.deviance)
        logging.info(" -- The Akaike Information Criterion of the model is: " +
              str(self.aic) + ".")

        logging.info(" -- The Bayesian Information Criterion of the model is: " +
              str(self.bic) + ".")

        try:
            self.merit
        except AttributeError:
            self._compute_statistics(lpost)

        logging.info(" -- The figure-of-merit function for this model " +
              " is: %f.5f"%self.merit +
              " and the fit for %i dof is %f.3f"%(self.dof,
                                                  self.merit/self.dof))

        logging.info(" -- Summed Residuals S = %f.5f"%self.sobs)
        logging.info(" -- Expected S ~ %f.5 +/- %f.5"%(self.sexp, self.ssd))
        logging.info(" -- merit function (SSE) M = %f.5f \n\n"%self.merit)

        return


class ParameterEstimation(object):

    def __init__(self, fitmethod='BFGS', max_post=True):
        """
        Parameter estimation of two-dimensional data, either via
        optimization or MCMC.
        Note: optimization with bounds is not supported. If something like
        this is required, define (uniform) priors in the ParametricModel
        instances to be used below.

        Parameters:
        -----------
        fitmethod: string, optional, default "L-BFGS-B"
            Any of the strings allowed in scipy.optimize.minimize in
            the method keyword. Sets the fit method to be used.

        max_post: bool, optional, default True
            If True, then compute the Maximum-A-Posteriori estimate. If False,
            compute a Maximum Likelihood estimate.
        """

        self.fitmethod = fitmethod

        self.max_post = max_post

    def fit(self, lpost, t0, neg=True, scipy_optimize_options=None):
        """
        Do either a Maximum A Posteriori or Maximum Likelihood
        fit to the data.

        Parameters:
        -----------
        lpost: Posterior (or subclass) instance
            and instance of class Posterior or one of its subclasses
            that defines the function to be minized (either in loglikelihood
            or logposterior)

        t0 : {list | numpy.ndarray}
            List/array with set of initial parameters

        neg : bool, optional, default True
            Boolean to be passed to `lpost`, setting whether to use the
            *negative* posterior or the *negative* log-likelihood. Since
            `Posterior` and `LogLikelihood` objects are generally defined in

        scipy_optimize_options : dict, optional, default None
            A dictionary with options for `scipy.optimize.minimize`,
            directly passed on as keyword arguments.

        Returns:
        --------
        fitparams: dict
            A dictionary with the fit results
            TODO: Add description of keywords in the class!
        """

        if not isinstance(lpost, Posterior) and not isinstance(lpost,
                                                               LogLikelihood):
            raise TypeError("lpost must be a subclass of "
                            "Posterior or LogLikelihoood.")

        newmod = lpost.model.copy()
        newmod.parameters = t0
        p0, _ = _model_to_fit_params(newmod)
        # p0 will be shorter than t0, if there are any frozen/tied parameters
        # this has to match with the npar attribute.
        if not len(p0) == lpost.npar:
            raise ValueError("Parameter set t0 must be of right "
                             "length for model in lpost.")

        if scipy.__version__ < "0.10.0":
            args = [neg]
        else:
            args = (neg,)

        if not scipy_optimize_options:
            scipy_optimize_options = {}

        # different commands for different fitting methods,
        # at least until scipy 0.11 is out
        funcval = 100.0
        i = 0

        while funcval == 100 or funcval == 200 or \
                funcval == 0.0 or not np.isfinite(funcval):

            if i > 20:
                raise Exception("Fitting unsuccessful!")
            # perturb parameters slightly
            t0_p = np.random.multivariate_normal(p0, np.diag(np.abs(p0)/100.))

            # print(lpost.model, dir(lpost.model), lpost.model.parameter_constraints, lpost.model.param_names)
            params = [getattr(newmod,name) for name in newmod.param_names]
            bounds = [p.bounds for p in params if not np.any([p.tied, p.fixed])]
            # print(params, bounds)
            # if max_post is True, do the Maximum-A-Posteriori Fit
            if self.max_post:
                opt = scipy.optimize.minimize(lpost, t0_p,
                                              method=self.fitmethod,
                                              args=args, tol=1.e-10,
                                              bounds=bounds,
                                              **scipy_optimize_options)

            # if max_post is False, then do a Maximum Likelihood Fit
            else:
                if isinstance(lpost, Posterior):
                    # This could be a `Posterior` object
                    opt = scipy.optimize.minimize(lpost.loglikelihood, t0_p,
                                                  method=self.fitmethod,
                                                  args=args, tol=1.e-10,
                                                  bounds=bounds,
                                                  **scipy_optimize_options)

                elif isinstance(lpost, LogLikelihood):
                    # Except this could be a `LogLikelihood object
                    # In which case, use the evaluate function
                    # if it's not either, give up and break!
                    opt = scipy.optimize.minimize(lpost.evaluate, t0_p,
                                                  method=self.fitmethod,
                                                  args=args, tol=1.e-10,
                                                  bounds=bounds,
                                                  **scipy_optimize_options)

            funcval = opt.fun
            i += 1

        res = OptimizationResults(lpost, opt, neg=neg)

        return res

    def compute_lrt(self, lpost1, t1, lpost2, t2, neg=True, max_post=False):
        """
        This function computes the Likelihood Ratio Test between two
        nested models.

        Parameters
        ----------
        lpost1 : object of a subclass of Posterior
            The posterior object for model 1

        t1 : iterable
            The starting parameters for model 1

        lpost2 : object of a subclass of Posterior
            The posterior object for model 2

        t2 : iterable
            The starting parameters for model 2

        neg : bool, optional, default True
            Boolean flag to decide whether to use the negative log-likelihood
            or log-posterior

        max_post: bool, optional, default False
            If True, set the internal state to do the optimization with the
            log-likelihood rather than the log-posterior.

        """

        self.max_post = max_post

        # fit data with both models
        res1 = self.fit(lpost1, t1, neg=neg)
        res2 = self.fit(lpost2, t2, neg=neg)

        # compute log likelihood ratio as difference between the deviances
        lrt = res1.deviance - res2.deviance

        return lrt, res1, res2

    def sample(self, lpost, t0, cov=None,
               nwalkers=500, niter=100, burnin=100, threads=1,
               print_results=True, plot=False, namestr="test"):
        """
        Sample the posterior distribution defined in `lpost` using MCMC.
        Here we use the `emcee` package, but other implementations could
        in principle be used.

        Parameters
        ----------
        lpost : instance of a Posterior subclass
            and instance of class Posterior or one of its subclasses
            that defines the function to be minized (either in loglikelihood
            or logposterior)

        t0 : iterable
            list or array containing the starting parameters. Its length
            must match `lpost.model.npar`.

        nwalkers : int
            The number of walkers (chains) to use during the MCMC procedure.
            The more walkers are used, the slower the estimation will be, but
            the better the final distribution is likely to be.

        niter : int
            The number of iterations to run the MCMC chains for. The larger this
            number, the longer the estimation will take, but the higher the
            chance that the walkers have actually converged on the true
            posterior distribution.

        burnin : int
            The number of iterations to run the walkers before convergence is
            assumed to have occurred. This part of the chain will be discarded
            before sampling from what is then assumed to be the posterior
            distribution desired.

        threads : int
            The number of threads for parallelization.
            Default is 1, i.e. no parallelization

        print_results : bool
            Boolean flag setting whether the results of the MCMC run should
            be printed to standard output. Default: True

        plot : bool
            Boolean flag setting whether summary plots of the MCMC chains
            should be produced. Default: False

        namestr : str
            Optional string for output file names for the plotting.

        Returns
        -------

        res : SamplingResults object

        """
        if not can_sample:
            raise ImportError("emcee not installed! Can't sample!")

        ndim = len(t0)

        if cov is None:
            # do a MAP fitting step to find good starting positions for
            # the sampler
            res = self.fit(lpost, t0, neg=True)
            cov = res.cov
        # sample random starting positions for each walker from
        # a multivariate Gaussian
        p0 = np.array([np.random.multivariate_normal(t0, cov) for
                       i in range(nwalkers)])

        # initialize the sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lpost, args=[False],
                                        threads=threads)

        # run the burn-in
        pos, prob, state = sampler.run_mcmc(p0, burnin)

        sampler.reset()

        # do the actual MCMC run
        _, _, _ = sampler.run_mcmc(pos, niter, rstate0=state)

        res = SamplingResults(sampler)

        if print_results:
            res.print_results()

        if plot:
            fig = res.plot_results(fig=None, save_plot=True,
                                   filename=namestr + "_corner.pdf")

        return res

    def _generate_model(self, lpost, pars):
        """
        Helper function that generates a fake PSD similar to the
        one in the data, but with different parameters.

        Parameters
        ----------
        lpost : instance of a Posterior or LogLikelihood subclass
            The object containing the relevant information about the
            data and the model

        pars : iterable
            A list of parameters to be passed to lpost.model in oder
            to generate a model data set.

        Returns:
        --------
        model_data : numpy.ndarray
            An array of model values for each bin in lpost.x

        """

        assert isinstance(lpost, LogLikelihood) or isinstance(lpost, Posterior), \
            "lpost must be of type LogLikelihood or Posterior or one of its " \
            "subclasses!"

        # assert pars is of correct length
        assert len(pars) == lpost.npar, "pars must be a list " \
                                        "of %i parameters"%lpost.npar
        # get the model
        m = lpost.model

        # reset the parameters
        _fitter_to_model_params(m, pars)

        # make a model spectrum
        model_data = lpost.model(lpost.x)

        return model_data

    @staticmethod
    def _compute_pvalue(obs_val, sim):
        """
        Compute the p-value given an observed value of a test statistic
        and some simulations of that same test statistic.

        Parameters
        ----------
        obs_value : float
            The observed value of the test statistic in question

        sim: iterable
            A list or array of simulated values for the test statistic

        Returns
        -------
        pval : float [0, 1]
            The p-value for the test statistic given the simulations.

        """

        # cast the simulations as a numpy array
        sim = np.array(sim)

        # find all simulations that are larger than
        # the observed value
        ntail = sim[sim > obs_val].shape[0]

        # divide by the total number of simulations
        pval = np.float(ntail) / np.float(sim.shape[0])

        return pval

    def calibrate_lrt(self, lpost1, t1, lpost2, t2, sample=None, neg=True,
                      max_post=False,
                      nsim=1000, niter=200, nwalkers=500, burnin=200,
                      namestr="test"):

        """
        Calibrate the outcome of a Likelihood Ratio Test via MCMC.

        In order to compare models via likelihood ratio test, one generally
        aims to compute a p-value for the null hypothesis (generally the
        simpler model). There are two special cases where the theoretical
        distribution used to compute that p-value analytically given the
        observed likelihood ratio (a chi-square distribution) is not
        applicable:
        * the models are not nested (i.e. Model 1 is not a special, simpler
        case of Model 2),
        * the parameter values fixed in Model 2 to retrieve Model 1 are at the
        edges of parameter space (e.g. if one must set, say, an amplitude to
        zero in order to remove a component in the more complex model, and
        negative amplitudes are excluded a priori)

        In these cases, the observed likelihood ratio must be calibrated via
         simulations of the simpler model (Model 1), using MCMC to take into
        account the uncertainty in the parameters. This function does
        exactly that: it computes the likelihood ratio for the observed data,
        and produces simulations to calibrate the likelihood ratio and
        compute a p-value for observing the data under the assumption that
        Model 1 istrue.

        If `max_post=True`, the code will use MCMC to sample the posterior
        of the parameters and simulate fake data from there.

        If `max_post=False`, the code will use the covariance matrix derived
        from the fit to simulate data sets for comparison.

        Parameters
        ----------
        lpost1 : object of a subclass of Posterior
            The posterior object for model 1

        t1 : iterable
            The starting parameters for model 1

        lpost2 : object of a subclass of Posterior
            The posterior object for model 2

        t2 : iterable
            The starting parameters for model 2

        neg : bool, optional, default True
            Boolean flag to decide whether to use the negative
            log-likelihood or log-posterior

        max_post: bool, optional, default False
            If True, set the internal state to do the optimization with the
            log-likelihood rather than the log-posterior.


        Returns
        -------
        pvalue : float [0,1]
            p-value 'n stuff

        """

        # compute the observed likelihood ratio
        lrt_obs, res1, res2 = self.compute_lrt(lpost1, t1,
                                               lpost2, t2,
                                               neg=neg,
                                               max_post=max_post)

        # simulate parameter sets from the simpler model
        if not max_post:
            # using Maximum Likelihood, so I'm going to simulate parameters
            # from a multivariate Gaussian

            # set up the distribution
            mvn = scipy.stats.multivariate_normal(mean=res1.p_opt,
                                                  cov=res1.cov)

            # sample parameters
            s_all = mvn.rvs(size=nsim)

        else:
            if sample is None:
                # sample the posterior using MCMC
                sample = self.sample(lpost1, res1.p_opt, cov=res1.cov,
                                       nwalkers=nwalkers, niter=niter,
                                       burnin=burnin, namestr=namestr)

            # pick nsim samples out of the posterior sample
            s_all = sample[
                np.random.choice(sample.shape[0], nsim, replace=False)]

        # simulate LRTs
        # this method is defined in the subclasses!
        lrt_sim = self.simulate_lrts(s_all, lpost1, t1, lpost2, t2,
                                      max_post=max_post, neg=neg)

        # now I can compute the p-value:
        pval = ParameterEstimation._compute_pvalue(lrt_obs, lrt_sim)

        return pval


class SamplingResults(object):

    def __init__(self, sampler, ci_min=0.05, ci_max=0.95):
        """
        Helper class that will contain the results of the sampling
        in a handly format.
        Less fiddly than a dictionary.

        Parameters
        ----------
        sampler: emcee.EnsembleSampler object
            The object containing the sampler that's done all the work.

        ci_min: float out of [0,1]
            The lower bound percentile for printing confidence intervals
            on the parameters

        ci_max: float out of [0,1]
            The upper bound percentile for printing confidence intervals
            on the parameters

        """

        # store all the samples
        self.samples = sampler.flatchain

        self.nwalkers = np.float(sampler.chain.shape[0])
        self.niter = np.float(sampler.iterations)

        # store number of dimensions
        self.ndim = sampler.dim

        # compute and store acceptance fraction
        self.acceptance = np.nanmean(sampler.acceptance_fraction)
        self.L = self.acceptance*self.samples.shape[0]

        self._check_convergence(sampler)
        self._infer(ci_min, ci_max)

    def _check_convergence(self, sampler):

        # compute and store autocorrelation time
        try:
            self.acor = sampler.acor
        except emcee.autocorr.AutocorrError:
            logging.info("Chains too short to compute autocorrelation lengths.")

        self.rhat = self._compute_rhat(sampler)

    def _compute_rhat(self, sampler):

        # between-sequence variance
        mean_samples_iter = np.nanmean(sampler.chain, axis=1)

        # mean over the means over iterations: (self.ndim)
        mean_samples = np.nanmean(sampler.chain, axis=(0,1))

        # now compute between-sequence variance
        bb = (self.niter / (self.nwalkers - 1)) * np.sum((mean_samples_iter -
                                                          mean_samples)**2.,
                                                         axis=0)

        # compute variance of each chain
        var_samples = np.nanvar(sampler.chain, axis=1)

        # compute mean of variance
        ww = np.nanmean(var_samples, axis=0)

        # compute weighted average of ww and bb:
        rhat = ((self.niter - 1) / self.niter) * ww + (1 / self.niter) * bb

        return rhat

    def _infer(self, ci_min=0.05, ci_max=0.95):
        self.mean = np.mean(self.samples, axis=0)
        self.std = np.std(self.samples, axis=0)
        self.ci = np.percentile(self.samples, [ci_min, ci_max], axis=0)

    def print_results(self):
        """
        Print results of the MCMC run.

        """

        logging.info("-- The acceptance fraction is: %f.5"%self.acceptance)
        try:
            logging.info("-- The autocorrelation time is: %f.5"%self.acor)
        except AttributeError:
            pass
        logging.info("R_hat for the parameters is: " + str(self.rhat))

        logging.info("-- Posterior Summary of Parameters: \n")
        logging.info("parameter \t mean \t\t sd \t\t 5% \t\t 95% \n")
        logging.info("---------------------------------------------\n")
        for i in range(self.ndim):
            logging.info("theta[" + str(i) + "] \t " +
                  str(self.mean[i]) + "\t" + str(self.std[i]) + "\t" +
                  str(self.ci[0, i]) + "\t" + str(self.ci[1, i]) + "\n")

    def plot_results(self, nsamples=1000, fig=None, save_plot=False,
                     filename="test.pdf"):

        """
        Plot some results in a triangle plot.
        If installed, will use `corner` for the plotting
        (available here https://github.com/dfm/corner.py or
        through pip), if not, uses its own code to make a triangle
        plot.

        By default, this method returns a matplotlib.Figure object, but
        if `save_plot=True`, the plot can be saved to file automatically,

        Parameters
        ----------

        nsamples: int, default 1000
            The maximum number of samples used for plotting.

        fig: matplotlib.Figure instance, default None
            If created externally, you can pass a Figure instance to this method.
            If none is passed, the method will create one internally.

        save_plot: bool, default False
            If True, save the plot to file with a file name specified by the
            keyword `filename`. If False, just return the `Figure` object

        filename: str
            Name of the output file with the figure

        """
        assert can_plot, "Need to have matplotlib installed for plotting"
        if use_corner:
            corner.corner(self.samples, labels=None, fig=fig, bins=int(20),
                          quantiles=[0.16, 0.5, 0.84],
                          show_titles=True, title_args={"fontsize": 12})

        else:
            if fig is None:
                fig = plt.figure(figsize=(15, 15))

            plt.subplots_adjust(top=0.925, bottom=0.025,
                                left=0.025, right=0.975,
                                wspace=0.2, hspace=0.2)

            ind_all = np.random.choice(np.arange(self.samples.shape[0]),
                                       size=nsamples)
            samples = self.samples[ind_all]
            for i in range(self.ndim):
                for j in range(self.ndim):
                    xmin, xmax = samples[:, j].min(), samples[:, j].max()
                    ymin, ymax = samples[:, i].min(), samples[:, i].max()
                    ax = fig.add_subplot(self.ndim, self.ndim, i*self.ndim+j+1)

                    ax.xaxis.set_major_locator(MaxNLocator(5))
                    ax.ticklabel_format(style="sci", scilimits=(-2, 2))

                    if i == j:
                        ntemp, binstemp, patchestemp = \
                            ax.hist(samples[:, i], 30, normed=True,
                                    histtype='stepfilled')
                        ax.axis([ymin, ymax, 0, max(ntemp)*1.2])

                    else:

                        ax.axis([xmin, xmax, ymin, ymax])

                        # make a scatter plot first
                        ax.scatter(samples[:, j], samples[:, i], s=7)
                        # then add contours
                        xmin, xmax = samples[:, j].min(), samples[:, j].max()
                        ymin, ymax = samples[:, i].min(), samples[:, i].max()

                        # Perform Kernel density estimate on data
                        try:
                            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                            positions = np.vstack([xx.ravel(), yy.ravel()])
                            values = np.vstack([samples[:, j], samples[:, i]])
                            kernel = scipy.stats.gaussian_kde(values)
                            zz = np.reshape(kernel(positions).T, xx.shape)

                            ax.contour(xx, yy, zz, 7)
                        except ValueError:
                            logging.info("Not making contours.")

        if save_plot:
            plt.savefig(filename, format='pdf')

        return fig


class PSDParEst(ParameterEstimation):

    def __init__(self, ps, fitmethod='BFGS', max_post=True):

        self.ps = ps
        ParameterEstimation.__init__(self, fitmethod=fitmethod,
                                     max_post=max_post)

    def fit(self, lpost, t0, neg=True):

        self.lpost = lpost

        res = ParameterEstimation.fit(self, self.lpost, t0, neg=neg)

        res.maxpow, res.maxfreq, res.maxind = \
            self._compute_highest_outlier(self.lpost, res)

        return res

    def sample(self, lpost, t0, cov=None,
               nwalkers=500, niter=100, burnin=100, threads=1,
               print_results=True, plot=False, namestr="test"):

        self.lpost = lpost

        fit_res = ParameterEstimation.fit(self, self.lpost, t0, neg=True)

        res = ParameterEstimation.sample(self, self.lpost, fit_res.p_opt,
                                         cov=fit_res.cov,
                                         nwalkers=nwalkers,
                                         niter=niter, burnin=burnin,
                                         threads=threads,
                                         print_results=print_results, plot=plot,
                                         namestr=namestr)

        return res

    def _generate_data(self, lpost, pars):
        """
        Generate a fake power spectrum from a model.

        Parameters:
        ----------
        lpost : instance of a Posterior or LogLikelihood subclass
            The object containing the relevant information about the
            data and the model

        pars : iterable
            A list of parameters to be passed to lpost.model in oder
            to generate a model data set.

        Returns:
        --------
        sim_ps : stingray.Powerspectrum object
            The simulated Powerspectrum object

        """

        model_spectrum = self._generate_model(lpost, pars)

        # use chi-square distribution to get fake data
        model_powers = model_spectrum * \
                       np.random.chisquare(2 * self.ps.m,
                                           size=model_spectrum.shape[0]) \
                                                / (2. * self.ps.m)

        sim_ps = copy.copy(self.ps)

        sim_ps.power = model_powers

        return sim_ps

    def simulate_lrts(self, s_all, lpost1, t1, lpost2, t2, max_post=True,
                       neg=False):

        nsim = s_all.shape[0]
        lrt_sim = np.zeros(nsim)

        # now I can loop over all simulated parameter sets to generate a PSD
        for i, s in enumerate(s_all):

            # generate fake PSD
            sim_ps = self._generate_data(lpost1, s)

            # make LogLikelihood objects for both:
            if not max_post:
                sim_lpost1 = PSDLogLikelihood(sim_ps.freq, sim_ps.power,
                                              model=lpost1.model)
                sim_lpost2 = PSDLogLikelihood(sim_ps.freq, sim_ps.power,
                                              model=lpost2.model, m=sim_ps.m)
            else:
                # make a Posterior object
                sim_lpost1 = PSDPosterior(sim_ps.freq, sim_ps.power,
                                          lpost1.model, m=sim_ps.m)
                sim_lpost1.logprior = lpost1.logprior

                sim_lpost2 = PSDPosterior(sim_ps.freq, sim_ps.power,
                                          lpost2.model, m=sim_ps.m)

                sim_lpost2.logprior = lpost2.logprior

            parest_sim = PSDParEst(sim_ps, max_post=max_post)

            lrt_sim[i], _, _ = parest_sim.compute_lrt(sim_lpost1, t1,
                                                      sim_lpost2, t2,
                                                      neg=neg,
                                                      max_post=max_post)
        return lrt_sim


    def calibrate_highest_outlier(self, lpost, t0, sample=None,
                                  max_post=False,
                                  nsim=1000, niter=200, nwalkers=500,
                                  burnin=200, namestr="test"):

        """

        """
        # fit the model to the data
        res = self.fit(lpost, t0, neg=True)

        # find the highest data/model outlier:
        out_high, _, _ = self._compute_highest_outlier(lpost, res)
        # simulate parameter sets from the simpler model
        if not max_post:
            # using Maximum Likelihood, so I'm going to simulate parameters
            # from a multivariate Gaussian

            # set up the distribution
            mvn = scipy.stats.multivariate_normal(mean=res.p_opt,
                                                  cov=res.cov)

            # sample parameters
            s_all = mvn.rvs(size=nsim)

        else:
            if sample is None:
                # sample the posterior using MCMC
                sample = self.sample(lpost, res.p_opt, cov=res.cov,
                                       nwalkers=nwalkers, niter=niter,
                                       burnin=burnin, namestr=namestr)

            # pick nsim samples out of the posterior sample
            s_all = sample[
                np.random.choice(sample.shape[0], nsim, replace=False)]

        # simulate LRTs
        # this method is defined in the subclasses!
        out_high_sim = self.simulate_highest_outlier(s_all, lpost, t0,
                                                max_post=max_post)
        # now I can compute the p-value:
        pval = ParameterEstimation._compute_pvalue(out_high, out_high_sim)

        return pval

    def simulate_highest_outlier(self, s_all, lpost, t0, max_post=True):

        # the number of simulations
        nsim = s_all.shape[0]

        # empty array for the simulation results
        max_y_all = np.zeros(nsim)

        # now I can loop over all simulated parameter sets to generate a PSD
        for i, s in enumerate(s_all):

            # generate fake PSD
            sim_ps = self._generate_data(lpost, s)

            # make LogLikelihood objects for both:
            if not max_post:
                sim_lpost = PSDLogLikelihood(sim_ps.freq, sim_ps.power,
                                              model=lpost.model, m=sim_ps.m)
            else:
                # make a Posterior object
                sim_lpost = PSDPosterior(sim_ps.freq, sim_ps.power,
                                         lpost.model, m=sim_ps.m)
                sim_lpost.logprior = lpost.logprior

            parest_sim = PSDParEst(sim_ps, max_post=max_post)

            res = parest_sim.fit(sim_lpost, t0, neg=True)
            max_y_all[i], maxfreq, maxind = self._compute_highest_outlier(sim_lpost,
                                                               res,
                                                               nmax=1)
        return np.hstack(max_y_all)

    def _compute_highest_outlier(self, lpost, res, nmax=1):

        residuals = 2.0 * lpost.y/ res.mfit

        ratio_sort = copy.copy(residuals)
        ratio_sort.sort()
        max_y = ratio_sort[-nmax:]

        max_x = np.zeros(max_y.shape[0])
        max_ind = np.zeros(max_y.shape[0])

        for i, my in enumerate(max_y):
            max_x[i], max_ind[i] = self._find_outlier(lpost.x, residuals, my)

        return max_y, max_x, max_ind

    @staticmethod
    def _find_outlier(xdata, ratio, max_y):
        max_ind = np.where(ratio == max_y)[0][0]
        max_x = xdata[max_ind]

        return max_x, max_ind

    def plotfits(self, res1, res2=None, save_plot=False,
                 namestr='test', log=False):

        if not can_plot:
            logging.info("No matplotlib imported. Can't plot!")
        else:
            # make a figure
            f = plt.figure(figsize=(12, 10))
            # adjust subplots such that the space between the top and bottom
            # of each are zero
            plt.subplots_adjust(hspace=0.0, wspace=0.4)

            # first subplot of the grid, twice as high as the other two
            # This is the periodogram with the two fitted models overplotted
            s1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)

            if log:
                logx = np.log10(self.ps.freq)
                logy = np.log10(self.ps.power)
                logpar1 = np.log10(res1.mfit)

                p1, = s1.plot(logx, logy, color='black', linestyle='steps-mid')
                p2, = s1.plot(logx, logpar1, color='blue', lw=2)
                s1.set_xlim([min(logx), max(logx)])
                s1.set_ylim([min(logy)-1.0, max(logy)+1])
                if self.ps.norm == "leahy":
                    s1.set_ylabel('log(Leahy-Normalized Power)', fontsize=18)
                elif self.ps.norm == "rms":
                    s1.set_ylabel('log(RMS-Normalized Power)', fontsize=18)
                else:
                    s1.set_ylabel("log(Power)", fontsize=18)

            else:
                p1, = s1.plot(self.ps.freq, self.ps.power,
                              color='black', linestyle='steps-mid')
                p2, = s1.plot(self.ps.freq, res1.mfit,
                              color='blue', lw=2)

                s1.set_xscale("log")
                s1.set_yscale("log")

                s1.set_xlim([min(self.ps.freq), max(self.ps.freq)])
                s1.set_ylim([min(self.ps.freq)/10.0,
                             max(self.ps.power)*10.0])

                if self.ps.norm == "leahy":
                    s1.set_ylabel('Leahy-Normalized Power', fontsize=18)
                elif self.ps.norm == "rms":
                    s1.set_ylabel('RMS-Normalized Power', fontsize=18)
                else:
                    s1.set_ylabel("Power", fontsize=18)

            if res2 is not None:
                if log:
                    logpar2 = np.log10(res2.mfit)
                    p3, = s1.plot(logx, logpar2, color='red', lw=2)
                else:
                    p3, = s1.plot(self.ps.freq, res2.mfit,
                                  color='red', lw=2)
                s1.legend([p1, p2, p3], ["data", "model 1 fit", "model 2 fit"])
            else:
                s1.legend([p1, p2], ["data", "model fit"])

            s1.set_title("Periodogram and fits for data set " + namestr,
                         fontsize=18)

            # second subplot: power/model for Power law and straight line
            s2 = plt.subplot2grid((4, 1), (2, 0), rowspan=1)
            pldif = self.ps.power / res1.mfit

            s2.set_ylabel("Residuals, \n first model",
                          fontsize=18)

            if log:
                s2.plot(logx, pldif, color='black', linestyle='steps-mid')
                s2.plot(logx, np.ones(self.ps.freq.shape[0]),
                        color='blue', lw=2)
                s2.set_xlim([min(logx), max(logx)])
                s2.set_ylim([min(pldif), max(pldif)])

            else:
                s2.plot(self.ps.freq, pldif, color='black',
                        linestyle='steps-mid')
                s2.plot(self.ps.freq, np.ones_like(self.ps.power),
                        color='blue', lw=2)

                s2.set_xscale("log")
                s2.set_yscale("log")
                s2.set_xlim([min(self.ps.freq), max(self.ps.freq)])
                s2.set_ylim([min(pldif), max(pldif)])

            if res2 is not None:
                bpldif = self.ps.power/res2.mfit

            # third subplot: power/model for bent power law and straight line
                s3 = plt.subplot2grid((4, 1), (3, 0), rowspan=1)

                if log:
                    s3.plot(logx, bpldif, color='black', linestyle='steps-mid')
                    s3.plot(logx, np.ones(len(self.ps.freq)),
                            color='red', lw=2)
                    s3.axis([min(logx), max(logx), min(bpldif), max(bpldif)])
                    s3.set_xlabel("log(Frequency) [Hz]", fontsize=18)

                else:
                    s3.plot(self.ps.freq, bpldif,
                            color='black', linestyle='steps-mid')
                    s3.plot(self.ps.freq, np.ones(len(self.ps.freq)),
                            color='red', lw=2)
                    s3.set_xscale("log")
                    s3.set_yscale("log")
                    s3.set_xlim([min(self.ps.freq), max(self.ps.freq)])
                    s3.set_ylim([min(bpldif), max(bpldif)])
                    s3.set_xlabel("Frequency [Hz]", fontsize=18)

                s3.set_ylabel("Residuals, \n second model",
                              fontsize=18)

            else:
                if log:
                    s2.set_xlabel("log(Frequency) [Hz]", fontsize=18)
                else:
                    s2.set_xlabel("Frequency [Hz]", fontsize=18)

            ax = plt.gca()

            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(14)

            # make sure xticks are taken from first plots, but don't
            # appear there
            plt.setp(s1.get_xticklabels(), visible=False)

            if save_plot:
                # save figure in png file and close plot device
                plt.savefig(namestr + '_ps_fit.png', format='png')

        return
