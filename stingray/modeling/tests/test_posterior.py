import numpy as np
import scipy.stats

from astropy.tests.helper import pytest

from stingray import Powerspectrum
from stingray.modeling import Posterior, PSDPosterior
from stingray.modeling import Const

np.random.seed(20150907)



class PosteriorClassDummy(Posterior):
    """
    This is a test class that tests the basic functionality of the
    Posterior superclass.
    """
    def __init__(self, x, y, model):
        Posterior.__init__(self, x, y, model)

    def  loglikelihood(self, t0, neg=False):
        loglike = 1.0
        return loglike

    def logprior(self, t0):
        lp = 2.0
        return lp


class TestPosteriorABC(object):

    def test_instantiation_of_abcclass_fails(self):
        with pytest.raises(TypeError):
            p = Posterior()

    def test_failure_without_loglikelihood_method(self):
        """
        The abstract base class Posterior requires a method
        :loglikelihood: to be defined in any of its subclasses.
        Having a subclass without this method should cause failure.

        """
        class PartialPosterior(Posterior):
            def __init__(self, x, y, model):
                Posterior.__init__(self, x, y, model)

        with pytest.raises(TypeError):
            p = PartialPosterior()


class TestPosterior(object):

    @classmethod
    def setup_class(cls):
        cls.x = np.arange(100)
        cls.y = np.ones(cls.x.shape[0])
        cls.model = Const(hyperpars={"a_mean":2.0, "a_var":1.0})
        cls.p = PosteriorClassDummy(cls.x, cls.y, cls.model)

    def test_inputs(self):
        assert np.allclose(self.p.x, self.x)
        assert np.allclose(self.p.y, self.y)
        assert isinstance(self.p.model, Const)

    def test_call_method_positive(self):
        t0 = [1,2,3]
        post = self.p(t0, neg=False)
        assert post == 3.0

    def test_call_method_negative(self):
        t0 = [1,2,3]
        post = self.p(t0, neg=True)
        assert post == -3.0


class TestPSDPosterior(object):

    @classmethod
    def setup_class(cls):
        m = 1
        nfreq = 1000000
        freq = np.arange(nfreq)
        noise = np.random.exponential(size=nfreq)
        power = noise*2.0

        ps = Powerspectrum()
        ps.freq = freq
        ps.power = power
        ps.m = m
        ps.df = freq[1]-freq[0]
        ps.norm = "leahy"

        cls.ps = ps
        cls.a_mean, cls.a_var = 2.0, 1.0

        cls.model = Const(hyperpars={"a_mean":cls.a_mean, "a_var":cls.a_var})

    def test_logprior_fails_without_prior(self):
        model = Const()
        lpost = PSDPosterior(self.ps, model)
        with pytest.raises(AssertionError):
            lpost.logprior([1])

    def test_making_posterior(self):
        lpost = PSDPosterior(self.ps, self.model)
        #print(lpost.x)
        #print(self.ps.freq)
        assert lpost.x.all() == self.ps.freq.all()
        assert lpost.y.all() == self.ps.power.all()

    def test_correct_number_of_parameters(self):
        lpost = PSDPosterior(self.ps, self.model)
        with pytest.raises(AssertionError):
            lpost([2,3])

    def test_logprior(self):
        t0 = [2.0]

        lpost = PSDPosterior(self.ps, self.model)
        lp_test = lpost.logprior(t0)
        lp = np.log(scipy.stats.norm(2.0, 1.0).pdf(t0))
        assert lp == lp_test

    def test_loglikelihood(self):
        t0 = [2.0]
        m = self.model(self.ps.freq[1:], t0)
        loglike = -np.sum(self.ps.power[1:]/m + np.log(m))

        lpost = PSDPosterior(self.ps, self.model)
        loglike_test = lpost.loglikelihood(t0, neg=False)

        assert np.isclose(loglike, loglike_test)

    def test_loglikelihood_correctly_leaves_out_zeroth_freq(self):
        t0 = [2.0]
        m = self.model(self.ps.freq, t0)
        loglike = -np.sum(self.ps.power/m + np.log(m))

        lpost = PSDPosterior(self.ps, self.model)
        loglike_test = lpost.loglikelihood(t0, neg=False)

        with pytest.raises(AssertionError):
            assert np.isclose(loglike_test, loglike, atol=1.e-10, rtol=1.e-10)

    def test_negative_loglikelihood(self):
        t0 = [2.0]
        m = self.model(self.ps.freq[1:], t0)
        loglike = np.sum(self.ps.power[1:]/m + np.log(m))

        lpost = PSDPosterior(self.ps, self.model)
        loglike_test = lpost.loglikelihood(t0, neg=True)

        assert np.isclose(loglike, loglike_test)

    def test_posterior(self):
        t0 = [2.0]
        m = self.model(self.ps.freq[1:], t0)
        lpost = PSDPosterior(self.ps, self.model)
        post_test = lpost(t0, neg=False)

        loglike = -np.sum(self.ps.power[1:]/m + np.log(m))
        logprior = np.log(scipy.stats.norm(2.0, 1.0).pdf(t0))
        post = loglike + logprior

        assert np.isclose(post_test, post, atol=1.e-10)

    def test_negative_posterior(self):
        t0 = [2.0]
        m = self.model(self.ps.freq[1:], t0)
        lpost = PSDPosterior(self.ps, self.model)
        post_test = lpost(t0, neg=True)

        loglike = -np.sum(self.ps.power[1:]/m + np.log(m))
        logprior = np.log(scipy.stats.norm(2.0, 1.0).pdf(t0))
        post = -loglike - logprior

        assert np.isclose(post_test, post, atol=1.e-10)

class TestPerPosteriorAveragedPeriodogram(object):

    @classmethod
    def setup_class(cls):

        m = 10
        nfreq = 1000000
        freq = np.arange(nfreq)
        noise = scipy.stats.chi2(2.*m).rvs(size=nfreq)/np.float(m)
        power = noise

        ps = Powerspectrum()
        ps.freq = freq
        ps.power = power
        ps.m = m
        ps.df = freq[1]-freq[0]
        ps.norm = "leahy"

        cls.ps = ps
        cls.a_mean, cls.a_var = 2.0, 1.0

        cls.model = Const(hyperpars={"a_mean":cls.a_mean, "a_var":cls.a_var})

    def test_likelihood(self):
        t0 = [2.0]
        m = self.model(self.ps.freq[1:], t0)
        ## TODO: Finish this test!

## TODO: Write tests for Lightcurve posterior
## TODO: Write tests for Gaussian posterior
