from __future__ import division, print_function
import numpy as np
import scipy.stats
import os
import re

from astropy.tests.helper import pytest
from astropy.modeling import models
from astropy.modeling.fitting import _fitter_to_model_params

from stingray import Powerspectrum

from stingray.modeling import ParameterEstimation, PSDParEst, \
    OptimizationResults, SamplingResults
from stingray.modeling import PSDPosterior, set_logprior, PSDLogLikelihood

try:
    from statsmodels.tools.numdiff import approx_hess
    comp_hessian = True
except ImportError:
    comp_hessian = False

try:
    import emcee
    can_sample = True
except ImportError:
    can_sample = False


class TestParameterEstimation(object):

    @classmethod
    def setup_class(cls):
        m = 1
        nfreq = 100000
        freq = np.arange(nfreq)
        noise = np.random.exponential(size=nfreq)
        power = noise * 2.0

        ps = Powerspectrum()
        ps.freq = freq
        ps.power = power
        ps.m = m
        ps.df = freq[1] - freq[0]
        ps.norm = "leahy"

        cls.ps = ps
        cls.a_mean, cls.a_var = 2.0, 1.0

        cls.model = models.Const1D()

        p_amplitude = lambda amplitude: \
            scipy.stats.norm(loc=cls.a_mean, scale=cls.a_var).pdf(amplitude)

        cls.priors = {"amplitude": p_amplitude}
        cls.lpost = PSDPosterior(cls.ps.freq, cls.ps.power, cls.model,
                                 m=cls.ps.m)
        cls.lpost.logprior = set_logprior(cls.lpost, cls.priors)

    def test_par_est_initializes(self):
        pe = ParameterEstimation()

    def test_parest_stores_max_post_correctly(self):
        """
        Make sure the keyword for Maximum A Posteriori fits is stored correctly
        as a default.
        """
        pe = ParameterEstimation()

        assert pe.max_post is True, "max_post should be set to True as a default."

    def test_object_works_with_loglikelihood_object(self):
        llike = PSDLogLikelihood(self.ps.freq, self.ps.power,
                                 self.model, m=self.ps.m)
        pe = ParameterEstimation()
        res = pe.fit(llike, [2.0])

        pass

    def test_fit_fails_when_object_is_not_posterior_or_likelihood(self):
        x = np.ones(10)
        y = np.ones(10)
        pe = ParameterEstimation()
        with pytest.raises(TypeError):
            res = pe.fit(x, y)

    def test_fit_fails_without_lpost_or_t0(self):
        pe = ParameterEstimation()
        with pytest.raises(TypeError):
            res = pe.fit()

    def test_fit_fails_without_t0(self):
        pe = ParameterEstimation()
        with pytest.raises(TypeError):
            res = pe.fit(np.ones(10))

    def test_fit_fails_with_incorrect_number_of_parameters(self):
        pe = ParameterEstimation()
        t0 = [1, 2]
        with pytest.raises(ValueError):
            res = pe.fit(self.lpost, t0)

    def test_fit_method_works_with_correct_parameter(self):
        pe = ParameterEstimation()
        t0 = [2.0]
        res = pe.fit(self.lpost, t0)

    def test_fit_method_returns_optimization_results_object(self):
        pe = ParameterEstimation()
        t0 = [2.0]
        res = pe.fit(self.lpost, t0)
        assert isinstance(res,
                          OptimizationResults), "res must be of type OptimizationResults"

    def test_compute_lrt_fails_when_garbage_goes_in(self):
        pe = ParameterEstimation()
        t0 = [2.0]

        with pytest.raises(TypeError):
            pe.compute_lrt(self.lpost, t0, None, t0)

        with pytest.raises(ValueError):
            pe.compute_lrt(self.lpost, t0[:-1], self.lpost, t0)

    def test_compute_lrt_sets_max_post_to_false(self):
        t0 = [2.0]
        pe = ParameterEstimation(max_post=True)

        assert pe.max_post is True
        delta_deviance = pe.compute_lrt(self.lpost, t0, self.lpost, t0)

        assert pe.max_post is False

    def test_compute_lrt_computes_deviance_correctly(self):
        t0 = [2.0]
        pe = ParameterEstimation()

        delta_deviance, opt1, opt2 = pe.compute_lrt(self.lpost, t0,
                                                    self.lpost, t0)

        assert delta_deviance < 1e-7

    def test_sampler_runs(self):
        pe = ParameterEstimation()
        if os.path.exists("test_corner.pdf"):
            os.unlink("test_corner.pdf")
        sample_res = pe.sample(self.lpost, [2.0], nwalkers=100, niter=10,
                               burnin=50, print_results=True, plot=True)

        assert os.path.exists("test_corner.pdf")
        assert sample_res.acceptance > 0.25
        assert isinstance(sample_res, SamplingResults)


class TestOptimizationResults(object):

    @classmethod
    def setup_class(cls):
        m = 1
        nfreq = 100000
        freq = np.arange(nfreq)
        noise = np.random.exponential(size=nfreq)
        power = noise * 2.0

        ps = Powerspectrum()
        ps.freq = freq
        ps.power = power
        ps.m = m
        ps.df = freq[1] - freq[0]
        ps.norm = "leahy"

        cls.ps = ps
        cls.a_mean, cls.a_var = 2.0, 1.0

        cls.model = models.Const1D()

        p_amplitude = lambda amplitude: \
            scipy.stats.norm(loc=cls.a_mean, scale=cls.a_var).pdf(amplitude)

        cls.priors = {"amplitude": p_amplitude}
        cls.lpost = PSDPosterior(cls.ps.freq, cls.ps.power,
                                 cls.model, m=cls.ps.m)
        cls.lpost.logprior = set_logprior(cls.lpost, cls.priors)

        cls.fitmethod = "BFGS"
        cls.max_post = True
        cls.t0 = [2.0]
        cls.neg = True
        cls.opt = scipy.optimize.minimize(cls.lpost, cls.t0,
                                          method=cls.fitmethod,
                                          args=cls.neg, tol=1.e-10)

    def test_object_initializes(self):
        res = OptimizationResults(self.lpost, self.opt, neg=self.neg)

    def test_object_has_right_attributes(self):
        res = OptimizationResults(self.lpost, self.opt, neg=self.neg)

        assert hasattr(res, "p_opt")
        assert hasattr(res, "result")
        assert hasattr(res, "deviance")
        assert hasattr(res, "aic")
        assert hasattr(res, "bic")
        assert hasattr(res, "model")
        assert isinstance(res.model, models.Const1D)

    def test_p_opt_is_correct(self):
        res = OptimizationResults(self.lpost, self.opt, neg=self.neg)
        assert res.p_opt == self.opt.x, "res.p_opt must be the same as opt.x!"
        assert np.isclose(res.p_opt[0], 2.0, atol=0.1, rtol=0.1)

    def test_model_is_same_as_in_lpost(self):
        res = OptimizationResults(self.lpost, self.opt, neg=self.neg)
        assert res.model == self.lpost.model

    def test_result_is_same_as_in_opt(self):
        res = OptimizationResults(self.lpost, self.opt, neg=self.neg)
        assert res.result == self.opt.fun

    def test_compute_model_works_correctly(self):
        res = OptimizationResults(self.lpost, self.opt, neg=self.neg)
        mean_model = np.ones_like(self.lpost.x) * self.opt.x[0]
        assert np.all(
            res.mfit == mean_model), "res.model should be exactly " \
                                     "the model for the data."

    def test_compute_criteria_computes_criteria_correctly(self):
        res = OptimizationResults(self.lpost, self.opt, neg=self.neg)

        test_aic = 169440.83719024697
        test_bic = 169245.62163088709
        test_deviance = 337950.6823459795

        assert np.isclose(res.aic, test_aic, atol=0.1, rtol=0.1)
        assert np.isclose(res.bic, test_bic, atol=0.1, rtol=0.1)
        assert np.isclose(res.deviance, test_deviance, atol=0.1, rtol=0.1)

    def test_merit_calculated_correctly(self):
        res = OptimizationResults(self.lpost, self.opt, neg=self.neg)

        test_merit = 98770.654981073574
        assert np.isclose(res.merit, test_merit, atol=0.1, rtol=0.1)

    def test_res_is_of_correct_type(self):
        pe = ParameterEstimation()
        t0 = [2.0]
        res = pe.fit(self.lpost, t0)

        assert isinstance(res, OptimizationResults)


class OptimizationResultsSubclassDummy(OptimizationResults):

    def __init__(self, lpost, res, neg):
        self.neg = neg
        self.result = res.fun
        self.p_opt = res.x
        self.model = lpost.model


class TestOptimizationResultInternalFunctions(object):

    @classmethod
    def setup_class(cls):
        m = 1
        nfreq = 100000
        freq = np.linspace(1, 1000, nfreq)

        np.random.seed(100)  # set the seed for the random number generator
        noise = np.random.exponential(size=nfreq)

        cls.model = models.PowerLaw1D() + models.Const1D()
        cls.model.x_0_0.fixed = True

        cls.alpha_0 = 2.0
        cls.amplitude_0 = 100.0
        cls.amplitude_1 = 2.0

        cls.model.alpha_0 = cls.alpha_0
        cls.model.amplitude_0 = cls.amplitude_0
        cls.model.amplitude_1 = cls.amplitude_1

        p = cls.model(freq)
        power = noise * p

        ps = Powerspectrum()
        ps.freq = freq
        ps.power = power
        ps.m = m
        ps.df = freq[1] - freq[0]
        ps.norm = "leahy"

        cls.ps = ps
        cls.a_mean, cls.a_var = 2.0, 1.0
        cls.a2_mean, cls.a2_var = 100.0, 10.0

        p_amplitude_1 = lambda amplitude: \
            scipy.stats.norm(loc=cls.a_mean, scale=cls.a_var).pdf(amplitude)

        p_alpha_0 = lambda alpha: \
            scipy.stats.uniform(0.0, 5.0).pdf(alpha)

        p_amplitude_0 = lambda amplitude: \
            scipy.stats.norm(loc=cls.a2_mean, scale=cls.a2_var).pdf(
                amplitude)

        cls.priors = {"amplitude_1": p_amplitude_1,
                      "amplitude_0": p_amplitude_0,
                      "alpha_0": p_alpha_0}

        cls.lpost = PSDPosterior(cls.ps.freq, cls.ps.power,
                                 cls.model, m=cls.ps.m)
        cls.lpost.logprior = set_logprior(cls.lpost, cls.priors)

        cls.fitmethod = "BFGS"
        cls.max_post = True
        cls.t0 = [cls.amplitude_0, cls.alpha_0, cls.amplitude_1]
        cls.neg = True
        cls.opt = scipy.optimize.minimize(cls.lpost, cls.t0,
                                          method=cls.fitmethod,
                                          args=cls.neg, tol=1.e-5)

        cls.optres = OptimizationResultsSubclassDummy(cls.lpost, cls.opt,
                                                      neg=True)

    def test_compute_model(self):
        optres = OptimizationResultsSubclassDummy(self.lpost, self.opt,
                                                  neg=True)

        optres._compute_model(self.lpost)

        assert hasattr(optres,
                       "mfit"), "OptimizationResult object should have mfit " \
                                "attribute at this point!"

        _fitter_to_model_params(self.model, self.opt.x)
        mfit_test = self.model(self.lpost.x)

        assert np.all(optres.mfit == mfit_test)

    def test_compute_statistics_computes_mfit(self):
        optres = OptimizationResultsSubclassDummy(self.lpost, self.opt,
                                                  neg=True)

        assert hasattr(optres, "mfit") is False
        optres._compute_statistics(self.lpost)

        assert hasattr(optres, "mfit")

    def test_compute_statistics_computes_all_statistics(self):
        optres = OptimizationResultsSubclassDummy(self.lpost, self.opt,
                                                  neg=True)

        optres._compute_statistics(self.lpost)

        assert hasattr(optres, "merit")
        assert hasattr(optres, "dof")
        assert hasattr(optres, "sexp")
        assert hasattr(optres, "ssd")
        assert hasattr(optres, "sobs")

    def test_compute_statistics_returns_correct_values(self):
        test_merit = 99765.718448514497
        test_dof = 99997.0
        test_sexp = 600000.0
        test_ssd = 1095.4451150103323
        test_sobs = -154.901207861497

        optres = OptimizationResultsSubclassDummy(self.lpost, self.opt,
                                                  neg=True)

        optres._compute_statistics(self.lpost)

        assert np.isclose(test_merit, optres.merit, atol=0.01, rtol=0.01)
        assert test_dof == optres.dof
        assert test_sexp == optres.sexp
        assert test_ssd == optres.ssd
        assert np.isclose(test_sobs, optres.sobs, atol=0.01, rtol=0.01)

    def test_compute_criteria_returns_correct_attributes(self):
        optres = OptimizationResultsSubclassDummy(self.lpost, self.opt,
                                                  neg=True)

        optres._compute_criteria(self.lpost)

        assert hasattr(optres, "aic")
        assert hasattr(optres, "bic")
        assert hasattr(optres, "deviance")

    def test_compute_criteria_returns_correct_values(self):
        optres = OptimizationResultsSubclassDummy(self.lpost, self.opt,
                                                  neg=True)

        optres._compute_criteria(self.lpost)

        test_aic = 170988.9174964963
        test_bic = 171017.4562728912
        test_deviance = 341954.16168676887

        assert np.isclose(test_aic, optres.aic)
        assert np.isclose(test_bic, optres.bic)
        assert np.isclose(test_deviance, optres.deviance)

    def test_compute_covariance_with_hess_inverse(self):
        optres = OptimizationResultsSubclassDummy(self.lpost, self.opt,
                                                  neg=True)

        optres._compute_covariance(self.lpost, self.opt)

        assert np.all(optres.cov == np.asarray(self.opt.hess_inv))
        assert np.all(optres.err == np.sqrt(np.diag(self.opt.hess_inv)))

    def test_compute_covariance_without_hess_inverse(self):
        fitmethod = "powell"
        opt = scipy.optimize.minimize(self.lpost, self.t0,
                                      method=fitmethod,
                                      args=self.neg, tol=1.e-10)

        optres = OptimizationResultsSubclassDummy(self.lpost, opt,
                                                  neg=True)

        optres._compute_covariance(self.lpost, opt)

        if comp_hessian:
            phess = approx_hess(opt.x, self.lpost)
            hess_inv = np.linalg.inv(phess)

            assert np.all(optres.cov == hess_inv)
            assert np.all(optres.err == np.sqrt(np.diag(np.abs(hess_inv))))


if can_sample:
    class SamplingResultsDummy(SamplingResults):
        def __init__(self, sampler, ci_min=0.05, ci_max=0.95):
            # store all the samples
            self.samples = sampler.flatchain

            self.nwalkers = np.float(sampler.chain.shape[0])
            self.niter = np.float(sampler.iterations)

            # store number of dimensions
            self.ndim = sampler.dim

            # compute and store acceptance fraction
            self.acceptance = np.nanmean(sampler.acceptance_fraction)
            self.L = self.acceptance * self.samples.shape[0]


    class TestSamplingResults(object):
        @classmethod
        def setup_class(cls):
            m = 1
            nfreq = 100000
            freq = np.arange(nfreq)
            noise = np.random.exponential(size=nfreq)
            power = noise * 2.0

            ps = Powerspectrum()
            ps.freq = freq
            ps.power = power
            ps.m = m
            ps.df = freq[1] - freq[0]
            ps.norm = "leahy"

            cls.ps = ps
            cls.a_mean, cls.a_var = 2.0, 1.0

            cls.model = models.Const1D()

            p_amplitude = lambda amplitude: \
                scipy.stats.norm(loc=cls.a_mean, scale=cls.a_var).pdf(
                    amplitude)

            cls.priors = {"amplitude": p_amplitude}
            cls.lpost = PSDPosterior(cls.ps.freq, cls.ps.power,
                                     cls.model, m=cls.ps.m)
            cls.lpost.logprior = set_logprior(cls.lpost, cls.priors)

            cls.fitmethod = "BFGS"
            cls.max_post = True
            cls.t0 = [2.0]
            cls.neg = True

            pe = ParameterEstimation()
            res = pe.fit(cls.lpost, cls.t0)

            cls.nwalkers = 100
            cls.niter = 200

            np.random.seed(200)
            p0 = np.array(
                [np.random.multivariate_normal(res.p_opt, res.cov) for
                 i in range(cls.nwalkers)])

            cls.sampler = emcee.EnsembleSampler(cls.nwalkers,
                                                len(res.p_opt), cls.lpost,
                                                args=[False], threads=1)

            _, _, _ = cls.sampler.run_mcmc(p0, cls.niter)

        def test_sample_results_object_initializes(self):
            SamplingResults(self.sampler)

        def test_sample_results_produces_attributes(self):
            s = SamplingResults(self.sampler)

            assert s.samples.shape[0] == self.nwalkers * self.niter

        def test_sampling_results_acceptance_ratio(self):
            s = SamplingResults(self.sampler)

            assert s.acceptance > 0.25
            assert np.isclose(s.L,
                              s.acceptance * self.nwalkers * self.niter)

        def test_check_convergence_works(self):
            s = SamplingResultsDummy(self.sampler)
            s._check_convergence(self.sampler)

            assert hasattr(s, "rhat")

        def test_rhat_computes_correct_answer(self):
            s = SamplingResults(self.sampler)

            rhat_test = 3.81886815e-06

            assert np.isclose(rhat_test, s.rhat[0], atol=0.001, rtol=0.001)

        def test_infer_works(self):
            s = SamplingResultsDummy(self.sampler)
            s._infer()

            assert hasattr(s, "mean")
            assert hasattr(s, "std")
            assert hasattr(s, "ci")

        def test_infer_computes_correct_values(self):
            s = SamplingResults(self.sampler)

            test_mean = 2.00190793
            test_std = 0.00195719
            test_ci = [[1.99435539], [1.9971502]]

            assert np.isclose(test_mean, s.mean[0], atol=0.01, rtol=0.01)
            assert np.isclose(test_std, s.std[0], atol=0.01, rtol=0.01)
            assert np.all(np.isclose(test_ci, s.ci, atol=0.01, rtol=0.01))


class TestPSDParEst(object):

    @classmethod
    def setup_class(cls):

        m = 1
        nfreq = 100000
        freq = np.linspace(1, 10.0, nfreq)

        np.random.seed(100) # set the seed for the random number generator
        noise = np.random.exponential(size=nfreq)

        cls.model = models.Lorentz1D() + models.Const1D()

        cls.x_0_0 = 2.0
        cls.fwhm_0 = 0.1
        cls.amplitude_0 = 100.0

        cls.amplitude_1 = 2.0

        cls.model.x_0_0 = cls.x_0_0
        cls.model.fwhm_0 = cls.fwhm_0
        cls.model.amplitude_0 = cls.amplitude_0
        cls.model.amplitude_1 = cls.amplitude_1

        p = cls.model(freq)

        np.random.seed(400)
        power = noise*p

        ps = Powerspectrum()
        ps.freq = freq
        ps.power = power
        ps.m = m
        ps.df = freq[1]-freq[0]
        ps.norm = "leahy"

        cls.ps = ps
        cls.a_mean, cls.a_var = 2.0, 1.0
        cls.a2_mean, cls.a2_var = 100.0, 10.0

        p_amplitude_1 = lambda amplitude: \
            scipy.stats.norm(loc=cls.a_mean, scale=cls.a_var).pdf(amplitude)

        p_x_0_0 = lambda alpha: \
            scipy.stats.uniform(0.0, 5.0).pdf(alpha)

        p_fwhm_0 = lambda alpha: \
            scipy.stats.uniform(0.0, 0.5).pdf(alpha)

        p_amplitude_0 = lambda amplitude: \
            scipy.stats.norm(loc=cls.a2_mean, scale=cls.a2_var).pdf(amplitude)

        cls.priors = {"amplitude_1": p_amplitude_1,
                      "amplitude_0": p_amplitude_0,
                      "x_0_0": p_x_0_0,
                      "fwhm_0": p_fwhm_0}

        cls.lpost = PSDPosterior(cls.ps.freq, cls.ps.power,
                                 cls.model, m=cls.ps.m)
        cls.lpost.logprior = set_logprior(cls.lpost, cls.priors)

        cls.fitmethod = "BFGS"
        cls.max_post = True
        cls.t0 = [cls.x_0_0, cls.fwhm_0, cls.amplitude_0, cls.amplitude_1]
        cls.neg = True

    def test_fitting_with_ties_and_bounds(self, capsys):
        double_f = lambda model : model.x_0_0 * 2
        model = self.model.copy()
        model =  self.model + models.Lorentz1D(amplitude=model.amplitude_0,
                                   x_0 = model.x_0_0 * 2,
                                   fwhm = model.fwhm_0)
        model.x_0_0 = self.model.x_0_0
        model.amplitude_0 = self.model.amplitude_0
        model.amplitude_1 = self.model.amplitude_1
        model.fwhm_0 = self.model.fwhm_0
        model.x_0_2.tied = double_f
        model.fwhm_0.bounds = [0, 10]
        model.amplitude_0.fixed = True

        p = model(self.ps.freq)

        noise = np.random.exponential(size=len(p))
        power = noise*p

        ps = Powerspectrum()
        ps.freq = self.ps.freq
        ps.power = power
        ps.m = self.ps.m
        ps.df = self.ps.df
        ps.norm = "leahy"

        pe = PSDParEst(ps)
        llike = PSDLogLikelihood(ps.freq, ps.power, model)

        true_pars = [self.amplitude_0, self.x_0_0, self.fwhm_0,
                     self.amplitude_1,
                     model.amplitude_2.value, model.x_0_2.value,
                     model.fwhm_2.value]
        res = pe.fit(llike, true_pars)

        res.print_summary(llike)
        out, err = capsys.readouterr()
        assert "100.00000            (Fixed)" in out
        pattern = \
            re.compile(r"5\) Parameter x_0_2\s+: [0-9]\.[0-9]{5}\s+\(Tied\)")
        assert pattern.search(out)

        compare_pars = [self.x_0_0, self.fwhm_0,
                        self.amplitude_1,
                        model.amplitude_2.value,
                        model.fwhm_2.value]

        assert np.all(np.isclose(compare_pars, res.p_opt, rtol=0.5))

    def test_par_est_initializes(self):
        pe = PSDParEst(self.ps)

    def test_parest_stores_max_post_correctly(self):
        """
        Make sure the keyword for Maximum A Posteriori fits is stored correctly
        as a default.
        """
        pe = PSDParEst(self.ps)

        assert pe.max_post is True, "max_post should be set to True as a default."

    def test_fit_fails_when_object_is_not_posterior_or_likelihood(self):
        x = np.ones(10)
        y = np.ones(10)
        pe = PSDParEst(self.ps)
        with pytest.raises(TypeError):
            res = pe.fit(x, y)

    def test_fit_fails_without_lpost_or_t0(self):
        pe = PSDParEst(self.ps)
        with pytest.raises(TypeError):
            res = pe.fit()

    def test_fit_fails_without_t0(self):
        pe = PSDParEst(self.ps)
        with pytest.raises(TypeError):
            res = pe.fit(np.ones(10))

    def test_fit_fails_with_incorrect_number_of_parameters(self):
        pe = PSDParEst(self.ps)
        t0 = [1,2]
        with pytest.raises(ValueError):
            res = pe.fit(self.lpost, t0)

    def test_fit_method_works_with_correct_parameter(self):
        pe = PSDParEst(self.ps)
        lpost = PSDPosterior(self.ps.freq, self.ps.power,
                             self.model, self.priors, m=self.ps.m)
        t0 = [2.0, 1, 1, 1]
        res = pe.fit(lpost, t0)

    def test_fit_method_returns_optimization_results_object(self):
        pe = PSDParEst(self.ps)

        t0 = [2.0, 1, 1, 1]
        res = pe.fit(self.lpost, t0)
        assert isinstance(res, OptimizationResults), "res must be of type " \
                                                     "OptimizationResults"

    def test_plotfits_leahy(self):
        pe = PSDParEst(self.ps)
        t0 = [2.0, 1, 1, 1]
        lpost = PSDPosterior(self.ps.freq, self.ps.power,
                             self.model, self.priors, m=self.ps.m)

        res = pe.fit(lpost, t0)

        pe.plotfits(res, save_plot=True)

        assert os.path.exists("test_ps_fit.png")
        os.unlink("test_ps_fit.png")

    def test_plotfits_log_leahy(self):
        pe = PSDParEst(self.ps)
        t0 = [2.0, 1, 1, 1]

        res = pe.fit(self.lpost, t0)

        pe.plotfits(res, save_plot=True, log=True)

        assert os.path.exists("test_ps_fit.png")
        os.unlink("test_ps_fit.png")

    def test_plotfits_rms(self):
        t0 = [2.0, 1, 1, 1]
        ps = Powerspectrum()
        ps.freq = self.ps.freq
        ps.power = self.ps.power
        ps.m = self.ps.m
        ps.df = self.ps.df
        ps.norm = "rms"
        pe = PSDParEst(ps)

        res = pe.fit(self.lpost, t0)

        pe.plotfits(res, res2=res, save_plot=True)

        assert os.path.exists("test_ps_fit.png")
        os.unlink("test_ps_fit.png")

    def test_plotfits_log_rms(self):
        ps = Powerspectrum()
        ps.freq = self.ps.freq
        ps.power = self.ps.power
        ps.m = self.ps.m
        ps.df = self.ps.df
        ps.norm = "rms"
        pe = PSDParEst(ps)

        t0 = [2.0, 1, 1, 1]
        res = pe.fit(self.lpost, t0)

        pe.plotfits(res, res2=res, save_plot=True, log=True)

        assert os.path.exists("test_ps_fit.png")
        os.unlink("test_ps_fit.png")

    def test_plotfits_pow(self):
        t0 = [2.0, 1, 1, 1]
        ps = Powerspectrum()
        ps.freq = self.ps.freq
        ps.power = self.ps.power
        ps.m = self.ps.m
        ps.df = self.ps.df
        ps.norm = "none"
        pe = PSDParEst(ps)

        res = pe.fit(self.lpost, t0)

        pe.plotfits(res, res2=res, save_plot=True)

        assert os.path.exists("test_ps_fit.png")
        os.unlink("test_ps_fit.png")

    def test_plotfits_log_pow(self):
        ps = Powerspectrum()
        ps.freq = self.ps.freq
        ps.power = self.ps.power
        ps.m = self.ps.m
        ps.df = self.ps.df
        ps.norm = "none"
        pe = PSDParEst(ps)

        t0 = [2.0, 1, 1, 1]
        res = pe.fit(self.lpost, t0)

        pe.plotfits(res, res2=res, save_plot=True, log=True)

        assert os.path.exists("test_ps_fit.png")
        os.unlink("test_ps_fit.png")

    def test_compute_lrt_fails_when_garbage_goes_in(self):
        pe = PSDParEst(self.ps)
        t0 = [2.0, 1, 1, 1]

        with pytest.raises(TypeError):
            pe.compute_lrt(self.lpost, t0, None, t0)

        with pytest.raises(ValueError):
            pe.compute_lrt(self.lpost, t0[:-1], self.lpost, t0)

    def test_compute_lrt_sets_max_post_to_false(self):
        t0 = [2.0, 1, 1, 1]
        pe = PSDParEst(self.ps, max_post=True)

        assert pe.max_post is True
        delta_deviance, _, _ = pe.compute_lrt(self.lpost, t0, self.lpost, t0)

        assert pe.max_post is False

    def test_compute_lrt_computes_deviance_correctly(self):

        t0 = [2.0, 1, 1, 1]
        pe = PSDParEst(self.ps, max_post=True)

        # MB: This is a little too random
        delta_deviance, _, _ = pe.compute_lrt(self.lpost, t0,
                                        self.lpost, t0)

        assert np.absolute(delta_deviance) < 1.5e-4

    def test_sampler_runs(self):

        pe = PSDParEst(self.ps)

        sample_res = pe.sample(self.lpost, [2.0, 0.1, 100, 2.0], nwalkers=50,
                               niter=10, burnin=15, print_results=True,
                               plot=True)
        assert os.path.exists("test_corner.pdf")
        os.unlink("test_corner.pdf")
        assert sample_res.acceptance > 0.25
        assert isinstance(sample_res, SamplingResults)

    def test_generate_model_data(self):
        pe = PSDParEst(self.ps)

        print("lpost.model: " + str(self.lpost.model.parameters))
        print("lpost.model type: " + str(self.lpost.model))

        m = self.model
        _fitter_to_model_params(m, self.t0)

        model = m(self.ps.freq)

        print("len(t0): " + str(len(self.t0)))
        print("lpost.npar: " + str(self.lpost.npar))

        pe_model = pe._generate_model(self.lpost, [2.0, 0.1, 100, 2.0])

        assert np.allclose(model, pe_model)

    def test_generate_model_breaks_with_wrong_input(self):

        pe = PSDParEst(self.ps)

        with pytest.raises(AssertionError):
            pe_model = pe._generate_model([1, 2, 3, 4], [1, 2, 3, 4])

    def  test_generate_model_breaks_for_wrong_number_of_parameters(self):
        pe = PSDParEst(self.ps)

        with pytest.raises(AssertionError):
            pe_model = pe._generate_model(self.lpost, [1, 2, 3])

    def test_pvalue_calculated_correctly(self):
        a = [1, 1, 1, 2]
        obs_val = 1.5

        pe = PSDParEst(self.ps)
        pval = pe._compute_pvalue(obs_val, a)

        assert np.isclose(pval, 1./len(a))

    def test_calibrate_lrt_fails_without_lpost_objects(self):
        pe = PSDParEst(self.ps)

        with pytest.raises(TypeError):
            pval = pe.calibrate_lrt(self.lpost, [1, 2, 3, 4],
                                    np.arange(10), np.arange(4))

    def test_calibrate_lrt_fails_with_wrong_parameters(self):
        pe = PSDParEst(self.ps)

        with pytest.raises(ValueError):
            pval = pe.calibrate_lrt(self.lpost, [1, 2, 3, 4],
                                    self.lpost, [1, 2, 3])

    def test_find_highest_outlier_works_as_expected(self):

        mp_ind = 5
        max_power = 1000.0

        ps = Powerspectrum()
        ps.freq = np.arange(10)
        ps.power = np.ones_like(ps.freq)
        ps.power[mp_ind] = max_power
        ps.m = 1
        ps.df = ps.freq[1]-ps.freq[0]
        ps.norm = "leahy"

        pe = PSDParEst(ps)

        max_x, max_ind = pe._find_outlier(ps.freq, ps.power, max_power)

        assert np.isclose(max_x, ps.freq[mp_ind])
        assert max_ind == mp_ind

    def test_compute_highest_outlier_works(self):

        mp_ind = 5
        max_power = 1000.0

        ps = Powerspectrum()
        ps.freq = np.arange(10)
        ps.power = np.ones_like(ps.freq)
        ps.power[mp_ind] = max_power
        ps.m = 1
        ps.df = ps.freq[1]-ps.freq[0]
        ps.norm = "leahy"

        model = models.Const1D()
        p_amplitude = lambda amplitude: \
            scipy.stats.norm(loc=1.0, scale=1.0).pdf(
                amplitude)

        priors = {"amplitude": p_amplitude}

        lpost = PSDPosterior(ps.freq, ps.power, model, 1)
        lpost.logprior = set_logprior(lpost, priors)

        pe = PSDParEst(ps)

        res = pe.fit(lpost, [1.0])

        res.mfit = np.ones_like(ps.freq)

        max_y, max_x, max_ind = pe._compute_highest_outlier(lpost, res)

        assert np.isclose(max_y[0], 2*max_power)
        assert np.isclose(max_x[0], ps.freq[mp_ind])
        assert max_ind == mp_ind