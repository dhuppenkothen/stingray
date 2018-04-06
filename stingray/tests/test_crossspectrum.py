from __future__ import division
import numpy as np
import pytest
import warnings
from stingray import Lightcurve, AveragedPowerspectrum
from stingray import Crossspectrum, AveragedCrossspectrum, coherence, time_lag
from stingray import StingrayError
from stingray.crossspectrum import cospectrum_pvalue
import copy

np.random.seed(20160528)


class TestCoherenceFunction(object):

    def setup_class(self):
        self.lc1 = Lightcurve([1, 2, 3, 4, 5], [2, 3, 2, 4, 1])
        self.lc2 = Lightcurve([1, 2, 3, 4, 5], [4, 8, 1, 9, 11])

    def test_coherence_runs(self):
        coh = coherence(self.lc1, self.lc2)

    def test_coherence_fails_if_data1_not_lc(self):
        data = np.array([[1,2,3,4,5],[2,3,4,5,1]])

        with pytest.raises(TypeError):
            coh = coherence(self.lc1, data)

    def test_coherence_fails_if_data2_not_lc(self):
        data = np.array([[1,2,3,4,5],[2,3,4,5,1]])

        with pytest.raises(TypeError):
            coh = coherence(data, self.lc2)

    def test_coherence_computes_correctly(self):

        coh = coherence(self.lc1, self.lc2)

        assert len(coh) == 2
        assert np.abs(np.mean(coh)) < 1

class TestTimelagFunction(object):

    def setup_class(self):
        self.lc1 = Lightcurve([1, 2, 3, 4, 5], [2, 3, 2, 4, 1])
        self.lc2 = Lightcurve([1, 2, 3, 4, 5], [4, 8, 1, 9, 11])

    def test_time_lag_runs(self):
        lag = time_lag(self.lc1, self.lc2)

    def test_time_lag_fails_if_data1_not_lc(self):
        data = np.array([[1,2,3,4,5],[2,3,4,5,1]])

        with pytest.raises(TypeError):
            lag = time_lag(self.lc1, data)

    def test_time_lag_fails_if_data2_not_lc(self):
        data = np.array([[1,2,3,4,5],[2,3,4,5,1]])

        with pytest.raises(TypeError):
            lag = time_lag(data, self.lc2)

    def test_time_lag_computes_correctly(self):

        lag = time_lag(self.lc1, self.lc2)

        assert np.max(lag) <= np.pi
        assert np.min(lag) >= -np.pi

class TestCoherence(object):

    def test_coherence(self):
        lc1 = Lightcurve([1, 2, 3, 4, 5], [2, 3, 2, 4, 1])
        lc2 = Lightcurve([1, 2, 3, 4, 5], [4, 8, 1, 9, 11])

        cs = Crossspectrum(lc1, lc2)
        coh = cs.coherence()

        assert len(coh) == 2
        assert np.abs(np.mean(coh)) < 1

    def test_high_coherence(self):
        import copy
        t = np.arange(1280)
        a = np.random.poisson(100, len(t))
        lc = Lightcurve(t, a)
        lc2 = Lightcurve(t, copy.copy(a))
        c = AveragedCrossspectrum(lc, lc2, 128)

        coh, _ = c.coherence()
        np.testing.assert_almost_equal(np.mean(coh).real, 1.0)


class TestCrossspectrum(object):

    def setup_class(self):
        tstart = 0.0
        tend = 1.0
        dt = 0.0001

        time = np.arange(tstart + 0.5*dt, tend + 0.5*dt, dt)

        counts1 = np.random.poisson(0.01, size=time.shape[0])
        counts2 = np.random.negative_binomial(1, 0.09, size=time.shape[0])

        self.lc1 = Lightcurve(time, counts1, gti=[[tstart, tend]], dt=dt)
        self.lc2 = Lightcurve(time, counts2, gti=[[tstart, tend]], dt=dt)

        self.cs = Crossspectrum(self.lc1, self.lc2)

    def test_make_empty_crossspectrum(self):
        cs = Crossspectrum()
        assert cs.freq is None
        assert cs.power is None
        assert cs.df is None
        assert cs.nphots1 is None
        assert cs.nphots2 is None
        assert cs.m == 1
        assert cs.n is None
        assert cs.power_err is None

    def test_init_with_one_lc_none(self):
        with pytest.raises(TypeError):
            cs = Crossspectrum(self.lc1)

    def test_init_with_multiple_gti(self):
        gti = np.array([[0.0, 0.2], [0.6, 1.0]])
        with pytest.raises(TypeError):
            cs = Crossspectrum(self.lc1, self.lc2, gti=gti)

    def test_init_with_norm_not_str(self):
        with pytest.raises(TypeError):
            cs = Crossspectrum(norm=1)

    def test_init_with_invalid_norm(self):
        with pytest.raises(ValueError):
            cs = Crossspectrum(norm='frabs')

    def test_init_with_wrong_lc1_instance(self):
        lc_ = Crossspectrum()
        with pytest.raises(TypeError):
            cs = Crossspectrum(lc_, self.lc2)

    def test_init_with_wrong_lc2_instance(self):
        lc_ = Crossspectrum()
        with pytest.raises(TypeError):
            cs = Crossspectrum(self.lc1, lc_)

    def test_make_crossspectrum_diff_lc_counts_shape(self):
        counts = np.array([1]*10001)
        time = np.linspace(0.0, 1.0001, 10001)
        lc_ = Lightcurve(time, counts)
        with pytest.raises(StingrayError):
            cs = Crossspectrum(self.lc1, lc_)

    def test_make_crossspectrum_diff_lc_stat(self):
        lc_ = copy.copy(self.lc1)
        lc_.err_dist = 'gauss'

        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(self.lc1, lc_)
        assert np.any(["different statistics" in r.message.args[0]
                       for r in record])

    def test_make_crossspectrum_bad_lc_stat(self):
        lc1 = copy.copy(self.lc1)
        lc1.err_dist = 'gauss'
        lc2 = copy.copy(self.lc1)
        lc2.err_dist = 'gauss'

        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(lc1, lc2)
        assert np.any(["is not poisson" in r.message.args[0]
                       for r in record])

    def test_make_crossspectrum_diff_dt(self):
        counts = np.array([1]*10000)
        time = np.linspace(0.0, 2.0, 10000)
        lc_ = Lightcurve(time, counts)
        with pytest.raises(StingrayError):
            cs = Crossspectrum(self.lc1, lc_)

    def test_rebin_smaller_resolution(self):
        # Original df is between 0.9 and 1.0
        with pytest.raises(ValueError):
            new_cs = self.cs.rebin(df=0.1)

    def test_rebin(self):
        new_cs = self.cs.rebin(df=1.5)
        assert new_cs.df == 1.5
        new_cs.time_lag()

    def test_rebin_factor(self):
        new_cs = self.cs.rebin(f=1.5)
        assert new_cs.df == self.cs.df * 1.5
        new_cs.time_lag()

    def test_rebin_log(self):
        # For now, just verify that it doesn't crash
        new_cs = self.cs.rebin_log(f=0.1)
        assert type(new_cs) == type(self.cs)
        new_cs.time_lag()

    def test_norm_leahy(self):
        cs = Crossspectrum(self.lc1, self.lc2, norm='leahy')
        assert len(cs.power) == 4999
        assert cs.norm == 'leahy'

    def test_norm_frac(self):
        cs = Crossspectrum(self.lc1, self.lc2, norm='frac')
        assert len(cs.power) == 4999
        assert cs.norm == 'frac'

    def test_norm_abs(self):
        cs = Crossspectrum(self.lc1, self.lc2, norm='abs')
        assert len(cs.power) == 4999
        assert cs.norm == 'abs'

    def test_failure_when_normalization_not_recognized(self):
        with pytest.raises(ValueError):
            cs = Crossspectrum(self.lc1, self.lc2, norm='wrong')

    def test_coherence(self):
        coh = self.cs.coherence()
        assert len(coh) == 4999
        assert np.abs(coh[0]) < 1

    def test_timelag(self):
        time_lag = self.cs.time_lag()
        assert np.max(time_lag) <= np.pi
        assert np.min(time_lag) >= -np.pi

    def test_nonzero_err(self):
        assert np.all(self.cs.power_err > 0)

    def test_timelag_error(self):
        class Child(Crossspectrum):
            def __init__(self):
                pass

        obj = Child()
        with pytest.raises(AttributeError):
            lag = obj.time_lag()

    def test_cospectrum_significances_run(self):
        cs = Crossspectrum(self.lc1, self.lc2, norm='Leahy')
        cs.cospectrum_significances()

    def test_cospectrum_significances_fails_in_rms(self):
        cs = Crossspectrum(self.lc1, self.lc2, norm='frac')
        with pytest.raises(ValueError):
            cs.cospectrum_significances()

    def test_cospectrum_significances_threshold(self):
        cs = Crossspectrum(self.lc1, self.lc2, norm='Leahy')

        # change the powers so that just one exceeds the threshold
        cs.power = np.zeros_like(cs.power) + 2.0

        index = 1
        cs.power[index] = 10.0

        threshold = 0.01

        pval = cs.cospectrum_significances(threshold=threshold,
                                          trial_correction=False)
        assert pval[0, 0] < threshold
        assert pval[1, 0] == index

    def test_cospectrum_significances_trial_correction(self):
        cs = Crossspectrum(self.lc1, self.lc2, norm='Leahy')
        # change the powers so that just one exceeds the threshold
        cs.power = np.zeros_like(cs.power) + 2.0
        index = 1
        cs.power[index] = 10.0
        threshold = 0.01
        pval = cs.cospectrum_significances(threshold=threshold,
                                          trial_correction=True)
        assert np.size(pval) == 0

    def test_cospectrum_significances_with_logbinned_psd(self):
        cs = Crossspectrum(self.lc1, self.lc2, norm='Leahy')
        cs_log = cs.rebin_log()
        pval = cs_log.cospectrum_significances(threshold=1.1, trial_correction=False)

        assert len(pval[0]) == len(cs_log.power)

    def test_cospectrum_significances_is_numpy_array(self):
        cs = Crossspectrum(self.lc1, self.lc2, norm='Leahy')
        # change the powers so that just one exceeds the threshold
        cs.power = np.zeros_like(cs.power) + 2.0

        index = 1
        cs.power[index] = 10.0

        threshold = 1.0

        pval = cs.cospectrum_significances(threshold=threshold,
                                          trial_correction=True)

        assert isinstance(pval, np.ndarray)
        assert pval.shape[0] == 2

class TestAveragedCrossspectrum(object):

    def setup_class(self):
        tstart = 0.0
        tend = 1.0
        dt = np.longdouble(0.0001)

        time = np.arange(tstart + 0.5*dt, tend + 0.5*dt, dt)

        counts1 = np.random.poisson(0.01, size=time.shape[0])
        counts2 = np.random.negative_binomial(1, 0.09, size=time.shape[0])

        self.lc1 = Lightcurve(time, counts1, gti=[[tstart, tend]], dt=dt)
        self.lc2 = Lightcurve(time, counts2, gti=[[tstart, tend]], dt=dt)

        self.cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1)

    def test_make_empty_crossspectrum(self):
        cs = AveragedCrossspectrum()
        assert cs.freq is None
        assert cs.power is None
        assert cs.df is None
        assert cs.nphots1 is None
        assert cs.nphots2 is None
        assert cs.m == 1
        assert cs.n is None
        assert cs.power_err is None

    def test_no_segment_size(self):
        with pytest.raises(ValueError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2)

    def test_invalid_type_attribute(self):
        with pytest.raises(ValueError):
            cs_test = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1)
            cs_test.type = 'invalid_type'
            assert AveragedCrossspectrum._make_crossspectrum(cs_test,
                                                             self.lc1,
                                                             self.lc2)

    def test_invalid_type_attribute_with_multiple_lcs(self):
        acs_test = AveragedCrossspectrum([self.lc1, self.lc2],
                                         [self.lc2, self.lc1],
                                         segment_size=1)
        acs_test.type = 'invalid_type'
        with pytest.raises(ValueError):
            assert AveragedCrossspectrum._make_crossspectrum(acs_test,
                                                             lc1=[self.lc1,
                                                                  self.lc2],
                                                             lc2=[self.lc2,
                                                                  self.lc1])

    def test_different_dt(self):
        time1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        counts1_test = np.random.poisson(0.01, size=len(time1))
        test_lc1 = Lightcurve(time1, counts1_test)

        time2 = [2, 4, 6, 8, 10]
        counts2_test = np.random.negative_binomial(1, 0.09, size=len(time2))
        test_lc2 = Lightcurve(time2, counts2_test)

        assert test_lc1.tseg == test_lc2.tseg

        assert test_lc1.dt != test_lc2.dt

        with pytest.raises(ValueError):
            assert AveragedCrossspectrum(test_lc1, test_lc2, segment_size=1)

    def test_different_tseg(self):
        time2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        counts2_test = np.random.poisson(0.01, size=len(time2))
        test_lc2 = Lightcurve(time2, counts2_test)

        time1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        counts1_test = np.random.negative_binomial(1, 0.09, size=len(time1))
        test_lc1 = Lightcurve(time1, counts1_test)

        assert test_lc2.dt == test_lc1.dt

        assert test_lc2.tseg != test_lc1.tseg

        with pytest.raises(ValueError):
            assert AveragedCrossspectrum(test_lc1, test_lc2, segment_size=1)

    def test_rebin_with_invalid_type_attribute(self):
        new_df = 2
        aps = AveragedPowerspectrum(lc=self.lc1, segment_size=1,
                                    norm='leahy')
        aps.type = 'invalid_type'
        with pytest.raises(AttributeError):
            assert aps.rebin(df=new_df)

    def test_rebin_with_valid_type_attribute(self):
        new_df = 2
        aps = AveragedPowerspectrum(lc=self.lc1, segment_size=1,
                                    norm='leahy')
        assert aps.rebin(df=new_df)

    def test_init_with_norm_not_str(self):
        with pytest.raises(TypeError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1,
                                       norm=1)

    def test_init_with_invalid_norm(self):
        with pytest.raises(ValueError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1,
                                       norm='frabs')

    def test_init_with_inifite_segment_size(self):
        with pytest.raises(ValueError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=np.inf)

    def test_with_iterable_of_lightcurves(self):
        def iter_lc(lc, n):
            "Generator of n parts of lc."
            t0 = int(len(lc) / n)
            t = t0
            i = 0
            while(True):
                lc_seg = lc[i:t]
                yield lc_seg
                if t + t0 > len(lc):
                    break
                else:
                    i, t = t, t + t0

        cs = AveragedCrossspectrum(iter_lc(self.lc1, 1), iter_lc(self.lc2, 1),
                                   segment_size=1)

    def test_coherence(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            coh = self.cs.coherence()

            assert len(coh[0]) == 4999
            assert len(coh[1]) == 4999
            assert issubclass(w[-1].category, UserWarning)

    def test_failure_when_normalization_not_recognized(self):
        with pytest.raises(ValueError):
            self.cs = AveragedCrossspectrum(self.lc1, self.lc2,
                                            segment_size=1,
                                            norm="wrong")

    def test_rebin(self):
        new_cs = self.cs.rebin(df=1.5)
        assert new_cs.df == 1.5
        new_cs.time_lag()

    def test_rebin_factor(self):
        new_cs = self.cs.rebin(f=1.5)
        assert new_cs.df == self.cs.df * 1.5
        new_cs.time_lag()

    def test_rebin_log(self):
        # For now, just verify that it doesn't crash
        new_cs = self.cs.rebin_log(f=0.1)
        assert type(new_cs) == type(self.cs)
        new_cs.time_lag()

    def test_rebin_log_returns_complex_values(self):
        # For now, just verify that it doesn't crash
        new_cs = self.cs.rebin_log(f=0.1)
        assert isinstance(new_cs.power[0], np.complex)

    def test_rebin_log_returns_complex_errors(self):
        # For now, just verify that it doesn't crash
        new_cs = self.cs.rebin_log(f=0.1)
        assert isinstance(new_cs.power_err[0], np.complex)

    def test_timelag(self):
        from ..simulator.simulator import Simulator
        dt = 0.1
        simulator = Simulator(dt, 10000, rms=0.8, mean=1000)
        test_lc1 = simulator.simulate(2)
        test_lc2 = Lightcurve(test_lc1.time,
                              np.array(np.roll(test_lc1.counts, 2)),
                              err_dist=test_lc1.err_dist,
                              dt=dt)

        cs = AveragedCrossspectrum(test_lc1, test_lc2, segment_size=5,
                                   norm="none")

        time_lag, time_lag_err = cs.time_lag()

        assert np.all(np.abs(time_lag[:6] - 0.1) < 3 * time_lag_err[:6])

    def test_errorbars(self):
        time = np.arange(10000) * 0.1
        test_lc1 = Lightcurve(time, np.random.poisson(200, 10000))
        test_lc2 = Lightcurve(time, np.random.poisson(200, 10000))

        cs = AveragedCrossspectrum(test_lc1, test_lc2, segment_size=10,
                                   norm="leahy")

        assert np.allclose(cs.power_err, np.sqrt(2/cs.m))

class TestCospectrumSignificances(object):
    def test_function_runs(self):
        power = 2.0
        m = 1.0
        cospectrum_pvalue(power, m)

    def test_power_is_not_infinite(self):
        power = np.inf
        m = 1
        with pytest.raises(ValueError):
            cospectrum_pvalue(power, m)

    def test_power_is_not_infinite2(self):
        power = -np.inf
        m = 1
        with pytest.raises(ValueError):
            cospectrum_pvalue(power, m)

    def test_power_is_non_nan(self):
        power = np.nan
        m = 1
        with pytest.raises(ValueError):
            cospectrum_pvalue(power, m)

    def test_m_is_not_infinite(self):
        power = 2.0
        m = np.inf
        with pytest.raises(ValueError):
            cospectrum_pvalue(power, m)

    def test_m_is_not_infinite2(self):
        power = 2.0
        m = -np.inf
        with pytest.raises(ValueError):
            cospectrum_pvalue(power, m)

    def test_m_is_not_nan(self):
        power = 2.0
        m = np.nan
        with pytest.raises(ValueError):
            cospectrum_pvalue(power, m)

    def test_m_is_positive(self):
        power = 2.0
        m = -1.0
        with pytest.raises(ValueError):
            cospectrum_pvalue(power, m)

    def test_m_is_nonzero(self):
        power = 2.0
        m = 0.0
        with pytest.raises(ValueError):
            cospectrum_pvalue(power, m)

    def test_m_is_an_integer_number(self):
        power = 2.0
        m = 2.5
        with pytest.raises(ValueError):
            cospectrum_pvalue(power, m)

    def test_m_float_type_okay(self):
        power = 2.0
        m = 2.0
        cospectrum_pvalue(power, m)

    def test_pvalue_decreases_with_increasing_power(self):
        power1 = 2.0
        power2 = 20.0
        m = 1.0
        pval1 = cospectrum_pvalue(power1, m)
        pval2 = cospectrum_pvalue(power2, m)

        assert pval1 - pval2 > 0.0

    def test_pvalue_must_decrease_with_increasing_m(self):
        power = 3.0
        m1 = 1.0
        m2 = 10.0

        pval1 = cospectrum_pvalue(power, m1)
        pval2 = cospectrum_pvalue(power, m2)

        assert pval1 - pval2 > 0.0

    def test_probability_distribution_is_symmetrical(self):
        power1 = 2.0
        power2 = -2.0
        m = 1.0
        pval1 = cospectrum_pvalue(power1, m)
        pval2 = cospectrum_pvalue(power2, m)
        assert np.isclose(pval1+pval2, 1.0)

    def test_very_large_powers_produce_zero_prob(self):
        power = 31000.0
        m = 1
        pval = cospectrum_pvalue(power, m)
        assert np.isclose(pval, 0.0)
