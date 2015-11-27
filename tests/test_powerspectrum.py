

import numpy as np

from nose.tools import raises


from stingray import Lightcurve
from stingray import Powerspectrum, AveragedPowerspectrum

np.random.seed(20150907)

class TestPowerspectrum(object):

    def setUp(self):
        tstart = 0.0
        tend = 1.0
        dt = 0.0001

        time = np.linspace(tstart, tend, int((tend-tstart)/dt))

        mean_count_rate = 100.0
        mean_counts = mean_count_rate*dt

        poisson_counts = np.random.poisson(mean_counts,
                                           size=time.shape[0])

        self.lc = Lightcurve(time, counts=poisson_counts)


    def test_make_empty_periodogram(self):
        ps = Powerspectrum()
        assert ps.norm == "frac"
        assert ps.freq is None
        assert ps.ps is None
        assert ps.df is None
        assert ps.m == 1
        assert ps.n is None


    def test_make_periodogram_from_lightcurve(self):
        ps = Powerspectrum(lc=self.lc)
        assert ps.freq is not None
        assert ps.ps is not None
        assert ps.df == 1.0/self.lc.tseg
        assert ps.norm == "frac"
        assert ps.m == 1
        assert ps.n == self.lc.time.shape[0]
        assert ps.nphots == np.sum(self.lc.counts)

    def test_periodogram_types(self):
        ps = Powerspectrum(lc=self.lc)
        assert isinstance(ps.freq, np.ndarray)
        assert isinstance(ps.ps, np.ndarray)


    def test_init_with_lightcurve(self):
        assert Powerspectrum(self.lc)

    @raises(AssertionError)
    def test_init_without_lightcurve(self):
        assert Powerspectrum(self.lc.counts)

    @raises(AssertionError)
    def test_init_with_nonsense_data(self):
        nonsense_data = [None for i in xrange(100)]
        assert Powerspectrum(nonsense_data)

    @raises(AssertionError)
    def test_init_with_nonsense_norm(self):
        nonsense_norm = "bla"
        assert Powerspectrum(self.lc, norm=nonsense_norm)

    @raises(AssertionError)
    def test_init_with_wrong_norm_type(self):
        nonsense_norm = 1.0
        assert Powerspectrum(self.lc, norm=nonsense_norm)

    def test_fourier_amplitudes(self):
        """
        the integral of powers (or Riemann sum) should be close
        to the variance divided by twice the length of the light curve.

        Note: make sure the factors of ncounts match!
        """
        ps = Powerspectrum(lc=self.lc)
        nn = self.lc.ncounts
        pp = ps.unnorm_powers
        assert np.isclose(np.sqrt(pp[0]), np.sum(self.lc.counts))


    def test_dc_component(self):
        ps = Powerspectrum(lc=self.lc)
        pp = np.sqrt(ps.unnorm_powers)/ps.n
        print(pp[0])
        print(np.mean(self.lc.counts))
        assert pp[0]==np.mean(self.lc.counts)

    def test_total_variance(self):
        """
        the integral of powers (or Riemann sum) should be close
        to the variance divided by twice the length of the light curve.

        Note: make sure the factors of ncounts match!
        Also, make sure to *exclude* the zeroth power!
        """
        ps = Powerspectrum(lc=self.lc)
        nn = ps.n
        pp = ps.unnorm_powers/np.float(nn)**2.
        p_int = np.sum(pp[1:]*ps.df)-(pp[-1]*ps.df)/2.
        var_lc = np.var(self.lc.counts)/(2.*self.lc.tseg)
        assert np.isclose(p_int, var_lc)


    def test_rms_normalization_is_standard(self):
        """
        Make sure the standard normalization of a periodogram is
        rms and it stays that way!
        """
        ps = Powerspectrum(lc=self.lc)
        assert ps.norm == "frac"

    def test_rms_normalization_correct(self):
        """
        In rms normalization, the integral of the powers should be
        equal to the variance of the light curve divided by the mean
        of the light curve squared.
        """
        ps = Powerspectrum(lc=self.lc, norm="frac")
        ps_int = np.sum(ps.ps[1:-1]*ps.df) + ps.ps[-1]*ps.df/2.
        std_lc = np.var(self.lc.counts)/np.mean(self.lc.counts)**2.
        assert np.isclose(ps_int, std_lc)

    def test_fractional_rms_in_rms_norm(self):
        ps = Powerspectrum(lc=self.lc, norm="frac")
        rms_ps, rms_err = ps.compute_rms(min_freq=ps.freq[1],
                                         max_freq=ps.freq[-1])
        rms_lc = np.std(self.lc.counts)/np.mean(self.lc.counts)
        assert np.isclose(rms_ps, rms_lc, atol=0.01)

    def test_leahy_normalization_correct(self):
        ps = Powerspectrum(lc=self.lc, norm="leahy")
        assert ps.ps[0]/2. == np.sum(self.lc.counts)

    def test_leahy_norm_total_variance(self):
        """
        In Leahy normalization, the total variance should be the sum of
        powers multiplied by the number of counts and divided by the
        square of the number of data points in the light curve
        """
        ps = Powerspectrum(lc=self.lc, norm="Leahy")
        ps_var =  (np.sum(self.lc.counts)/ps.n**2.)*\
                  (np.sum(ps.ps[1:-1])+ps.ps[-1]/2.)
        assert np.isclose(ps_var, np.var(self.lc.counts))

    def test_fractional_rms_in_leahy_norm(self):
        """
        fractional rms should only be *approximately* equal the standard
        deviation divided by the mean of the light curve. Therefore, we allow
        for a larger tolerance in np.isclose()
        """
        ps = Powerspectrum(lc=self.lc, norm="Leahy")
        rms_ps, rms_err = ps.compute_rms(min_freq=ps.freq[1],
                                         max_freq=ps.freq[-1])

        rms_lc = np.std(self.lc.counts)/np.mean(self.lc.counts)
        assert np.isclose(rms_ps, rms_lc,atol=0.01)

    def test_fractional_rms_error(self):
        """
        TODO: Need to write a test for the fractional rms error.
        But I don't know how!
        """
        pass

    def test_rebin_makes_right_attributes(self):
        ps = Powerspectrum(lc = self.lc, norm="Leahy")
        ## replace powers
        ps.ps[1:] = np.ones_like(ps.ps[1:])*2.0
        rebin_factor = 2.0
        bin_ps = ps.rebin(rebin_factor*ps.df)

        assert bin_ps.freq is not None
        assert bin_ps.ps is not None
        assert bin_ps.df == rebin_factor*1.0/self.lc.tseg
        assert bin_ps.norm.lower() == "leahy"
        assert bin_ps.m == 2
        assert bin_ps.n == self.lc.time.shape[0]
        assert bin_ps.nphots == np.sum(self.lc.counts)

    def test_rebin_uses_mean(self):
        """
        Make sure the rebin-method uses "mean" to average instead of summing
        powers by default, and that this is not changed in the future!
        Note: function defaults come as a tuple, so the first keyword argument
        had better be 'method'
        """
        ps = Powerspectrum(lc = self.lc, norm="Leahy")
        assert ps.rebin.func_defaults[0] == "mean"

    def rebin_several(self, df):
        """
        TODO: Not sure how to write tests for the rebin method!
        """
        ps = Powerspectrum(lc = self.lc, norm="Leahy")
        bin_ps = ps.rebin(df)
        assert np.isclose(bin_ps.freq[0], bin_ps.df/2.)

    def test_rebin(self):
        df_all = [2, 3, 5, 1.5, 1,85]
        for df in df_all:
            yield self.rebin_several, df


class TestAveragedPowerspectrum(object):
    def setUp(self):
        tstart = 0.0
        tend = 10.0
        dt = 0.0001

        time = np.linspace(tstart, tend, int((tend-tstart)/dt))

        mean_count_rate = 1000.0
        mean_counts = mean_count_rate*dt

        poisson_counts = np.random.poisson(mean_counts,
                                           size=time.shape[0])

        self.lc = Lightcurve(time, counts=poisson_counts)

    def test_one_segment(self):
        segment_size = self.lc.tseg
        ps = AveragedPowerspectrum(self.lc, segment_size)
        assert np.isclose(ps.segment_size, segment_size)

    def test_n_segments(self):
        nseg_all = [1,2,3,5,10,20,100]
        for nseg in nseg_all:
            yield self.check_segment_size, nseg

    def check_segment_size(self, nseg):
        segment_size = self.lc.tseg/nseg
        ps = AveragedPowerspectrum(self.lc, segment_size)
        assert ps.m == nseg

    def test_segments_with_leftover(self):
        segment_size = self.lc.tseg/2. - 1.
        ps = AveragedPowerspectrum(self.lc, segment_size)
        assert np.isclose(ps.segment_size, segment_size)
        assert ps.m == 2


    @raises(TypeError)
    def test_init_without_segment(self):
        assert AveragedPowerspectrum(self.lc)

    @raises(TypeError)
    def test_init_with_nonsense_segment(self):
        segment_size = "foo"
        assert AveragedPowerspectrum(self.lc, segment_size)

    @raises(TypeError)
    def test_init_with_none_segment(self):
        segment_size = None
        assert AveragedPowerspectrum(self.lc, segment_size)

    @raises(AssertionError)
    def test_init_with_inf_segment(self):
        segment_size = np.inf
        assert AveragedPowerspectrum(self.lc, segment_size)

    @raises(AssertionError)
    def test_init_with_nan_segment(self):
        segment_size = np.nan
        assert AveragedPowerspectrum(self.lc, segment_size)


    def test_list_of_light_curves(self):
        n_lcs = 10

        tstart = 0.0
        tend = 1.0
        dt = 0.0001

        time = np.linspace(tstart, tend, int((tend-tstart)/dt))

        mean_count_rate = 1000.0
        mean_counts = mean_count_rate*dt

        lc_all = []
        for n in xrange(n_lcs):
            poisson_counts = np.random.poisson(mean_counts,
                                           size=len(time))

            lc = Lightcurve(time, counts=poisson_counts)
            lc_all.append(lc)

        segment_size = 0.5
        assert AveragedPowerspectrum(lc_all, segment_size)

    @raises(AssertionError)
    def test_list_with_nonsense_component(self):
        n_lcs = 10

        tstart = 0.0
        tend = 1.0
        dt = 0.0001

        time = np.linspace(tstart, tend, int((tend-tstart)/dt))

        mean_count_rate = 1000.0
        mean_counts = mean_count_rate*dt

        lc_all = []
        for n in xrange(n_lcs):
            poisson_counts = np.random.poisson(mean_counts,
                                           size=len(time))

            lc = Lightcurve(time, counts=poisson_counts)
            lc_all.append(lc)

        lc_all.append(1.0)
        segment_size = 0.5

        assert AveragedPowerspectrum(lc_all, segment_size)
