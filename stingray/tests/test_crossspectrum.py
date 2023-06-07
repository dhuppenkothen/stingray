import os
import numpy as np
import pytest
import warnings
import matplotlib.pyplot as plt
import scipy.special
from astropy.io import fits
from stingray import Lightcurve
from stingray import Crossspectrum, AveragedCrossspectrum
from stingray.crossspectrum import cospectra_pvalue
from stingray.crossspectrum import normalize_crossspectrum, normalize_crossspectrum_gauss
from stingray.crossspectrum import coherence, time_lag
from stingray import StingrayError
from stingray.simulator import Simulator
from stingray.fourier import poisson_level, raw_coherence

from stingray.events import EventList
import copy

_HAS_XARRAY = _HAS_PANDAS = _HAS_H5PY = True

try:
    import xarray
    from xarray import Dataset
except ImportError:
    _HAS_XARRAY = False

try:
    import pandas
    from pandas import DataFrame
except ImportError:
    _HAS_PANDAS = False

try:
    import h5py
except ImportError:
    _HAS_H5PY = False

np.random.seed(20160528)
curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")


def avg_cdf_two_spectra(x):
    prefac = 0.25

    if x >= 0:
        fac1 = 2 * scipy.special.gamma(2) - scipy.special.gammaincc(2, 2 * x)
        fac2 = 2.0 * scipy.special.gamma(1) - scipy.special.gammaincc(1, 2 * x)
    else:
        fac1 = scipy.special.gammaincc(2, -2 * x)
        fac2 = scipy.special.gammaincc(1, -2 * x)

    return prefac * (fac1 + fac2)


class TestClassicalPvalue(object):
    def test_pval_returns_float_when_float_input(self):
        power = 1.0
        nspec = 1.0
        pval = cospectra_pvalue(power, nspec)
        assert isinstance(pval, float)

    def test_pval_returns_iterable_when_iterable_input(self):
        power = [0, 1, 2]
        nspec = 1.0
        pval = cospectra_pvalue(power, nspec)
        assert isinstance(pval, np.ndarray)
        assert len(pval) == len(power)

    def test_pval_returns_iterable_when_iterable_input_nspec2(self):
        power = [0, 1, 2]
        nspec = 2
        # It will use the formulation by Huppenkothen & Bachetti
        pval = cospectra_pvalue(power, nspec)
        assert isinstance(pval, np.ndarray)
        assert len(pval) == len(power)

    def test_pval_fails_if_single_power_infinite(self):
        power = np.inf
        nspec = 1
        with pytest.raises(ValueError):
            pval = cospectra_pvalue(power, nspec)

    def test_pval_fails_if_single_power_nan(self):
        power = np.nan
        nspec = 1
        with pytest.raises(ValueError):
            pval = cospectra_pvalue(power, nspec)

    def test_pval_fails_if_multiple_powers_nan(self):
        power = [1, np.nan, 2.0]
        nspec = 1
        with pytest.raises(ValueError):
            pval = cospectra_pvalue(power, nspec)

    def test_pval_fails_if_multiple_powers_inf(self):
        power = [1, 2.0, np.inf]
        nspec = 1
        with pytest.raises(ValueError):
            pval = cospectra_pvalue(power, nspec)

    def test_pval_fails_if_nspec_zero(self):
        power = 1.0
        nspec = 0
        with pytest.raises(ValueError):
            pval = cospectra_pvalue(power, nspec)

    def test_pval_fails_if_nspec_negative(self):
        power = 1.0
        nspec = -10
        with pytest.raises(ValueError):
            pval = cospectra_pvalue(power, nspec)

    def test_pval_fails_if_nspec_not_integer(self):
        power = 1.0
        nspec = 1.5
        with pytest.raises(ValueError):
            pval = cospectra_pvalue(power, nspec)

    def test_single_spectrum(self):
        # the Laplace distribution is symmetric around
        # 0, so a power of 0 should return p=0.5
        power = 0.0
        nspec = 1
        assert cospectra_pvalue(power, nspec) == 0.5

    def test_single_spectrum_with_positive_power(self):
        """
        Because the Laplace distribution is always symmetric
        around zero, let's do a second version where I look
        for a different number.
        """
        power = 0.69314718055
        nspec = 1
        assert np.isclose(cospectra_pvalue(power, nspec), 0.25)

    def test_two_averaged_spectra(self):
        """
        For nspec=2, I can derive this by hand:
        """
        power = 1.0
        nspec = 2
        manual_pval = 1.0 - avg_cdf_two_spectra(power)
        assert np.isclose(cospectra_pvalue(power, nspec), manual_pval)

    def test_sixty_spectra(self):
        power = 1.0
        nspec = 60
        gauss = scipy.stats.norm(0, np.sqrt(2 / (nspec + 1)))
        pval_theory = gauss.sf(power)
        assert np.isclose(cospectra_pvalue(power, nspec), pval_theory)


class TestAveragedCrossspectrumEvents(object):
    def setup_class(self):
        tstart = 0.0
        tend = 1.0
        self.dt = np.longdouble(0.0001)
        segment_size = 1
        self.segment_size = segment_size
        N = np.rint(segment_size / self.dt).astype(int)
        # adjust dt
        self.dt = segment_size / N

        times1 = np.sort(np.random.uniform(tstart, tend, 1000000))
        times2 = np.sort(np.random.uniform(tstart, tend, 1000000))
        gti = np.array([[tstart, tend]])

        self.events1 = EventList(times1, gti=gti)
        self.events2 = EventList(times2, gti=gti)

        self.cs = Crossspectrum(self.events1, self.events2, dt=self.dt, norm="none")

        self.acs = AveragedCrossspectrum(
            self.events1.to_lc(self.dt),
            self.events2.to_lc(self.dt),
            silent=True,
            segment_size=segment_size,
            dt=self.dt,
            norm="none",
            power_type="all",
        )
        self.lc1, self.lc2 = self.events1, self.events2

    @pytest.mark.parametrize("norm", ["leahy", "frac", "abs", "none"])
    def test_common_mean_gives_comparable_scatter(self, norm):
        acs = AveragedCrossspectrum(
            self.events1,
            self.events2,
            dt=self.dt,
            silent=True,
            segment_size=self.segment_size,
            norm=norm,
            power_type="real",
            use_common_mean=False,
        )
        acs_comm = AveragedCrossspectrum(
            self.events1,
            self.events2,
            dt=self.dt,
            silent=True,
            segment_size=self.segment_size,
            norm=norm,
            power_type="real",
            use_common_mean=True,
        )

        assert np.isclose(acs_comm.power.std(), acs.power.std(), rtol=0.1)

    @pytest.mark.parametrize("use_common_mean", [True, False])
    @pytest.mark.parametrize("legacy", [True, False])
    def test_leahy_correct_for_multiple(self, legacy, use_common_mean):
        n = 30
        lc_all = []
        for i in range(n):
            time = np.arange(0.0, 10.0, 10.0 / 10000)
            counts = np.random.poisson(1000, size=time.size)
            lc = Lightcurve(time, counts)
            lc_all.append(lc)

        ps = AveragedCrossspectrum(
            lc_all, lc_all, 1.0, norm="leahy", legacy=legacy, use_common_mean=use_common_mean
        )

        assert ps.m == 300
        assert np.isclose(np.mean(ps.power), 2.0, atol=1e-2, rtol=1e-2)
        assert np.isclose(np.std(ps.power), 2.0 / np.sqrt(ps.m), atol=0.1, rtol=0.1)

    def test_from_events_works_cs(self):
        lccs = Crossspectrum.from_events(
            self.events1, self.events2, dt=self.dt, norm="none", silent=True
        )
        power1 = lccs.power.real
        power2 = self.cs.power.real
        assert np.allclose(power1, power2, rtol=0.01)
        lag1 = lccs.time_lag()
        lag2 = self.cs.time_lag()
        assert np.allclose(lag1, lag2)
        assert lccs.power_err is not None

    @pytest.mark.parametrize("legacy", [True, False])
    def test_internal_from_events_works_acs(self, legacy):
        lccs = AveragedCrossspectrum(
            self.events1,
            self.events2,
            segment_size=1,
            dt=self.dt,
            norm="none",
            silent=True,
            legacy=legacy,
        )
        power1 = lccs.power.real
        power2 = self.acs.power.real
        assert np.allclose(power1, power2, rtol=0.01)
        lag1, lag1_e = lccs.time_lag()
        lag2, lag2_e = self.acs.time_lag()
        assert np.allclose(lag1, lag2)
        good = ~np.isnan(lag2_e)
        assert np.allclose(lag1_e[good], lag2_e[good])
        assert lccs.power_err is not None

    def test_from_events_works_acs(self):
        lccs = AveragedCrossspectrum.from_events(
            self.events1, self.events2, segment_size=1, dt=self.dt, norm="none", silent=True
        )
        power1 = lccs.power.real
        power2 = self.acs.power.real
        assert np.allclose(power1, power2, rtol=0.01)
        lag1, lag1_e = lccs.time_lag()
        lag2, lag2_e = self.acs.time_lag()
        assert np.allclose(lag1, lag2)
        assert np.allclose(lag1_e, lag2_e, equal_nan=True)
        assert lccs.power_err is not None

    def test_from_lc_iter_works(self):
        lccs = AveragedCrossspectrum.from_lc_iterable(
            self.events1.to_lc_iter(self.dt, self.segment_size),
            self.events2.to_lc_iter(self.dt, self.segment_size),
            segment_size=self.segment_size,
            dt=self.dt,
            norm="none",
            silent=True,
        )
        power1 = lccs.power.real
        power2 = self.acs.power.real
        assert np.allclose(power1, power2, rtol=0.01)

    def test_from_lc_iter_with_err_works(self):
        def iter_lc_with_errs(iter_lc):
            for lc in iter_lc:
                lc._counts_err = np.zeros_like(lc.counts) + lc.counts.mean() ** 0.5
                yield lc

        lccs = AveragedCrossspectrum.from_lc_iterable(
            iter_lc_with_errs(self.events1.to_lc_iter(self.dt, self.segment_size)),
            iter_lc_with_errs(self.events2.to_lc_iter(self.dt, self.segment_size)),
            segment_size=self.segment_size,
            dt=self.dt,
            norm="none",
            silent=True,
        )
        power1 = lccs.power.real
        power2 = self.acs.power.real
        assert np.allclose(power1, power2, rtol=0.01)

    def test_from_lc_iter_counts_only_works(self):
        def iter_lc_counts_only(iter_lc):
            for lc in iter_lc:
                yield lc.counts

        lccs = AveragedCrossspectrum.from_lc_iterable(
            iter_lc_counts_only(self.events1.to_lc_iter(self.dt, self.segment_size)),
            iter_lc_counts_only(self.events2.to_lc_iter(self.dt, self.segment_size)),
            segment_size=self.segment_size,
            dt=self.dt,
            norm="none",
            silent=True,
        )
        power1 = lccs.power.real
        power2 = self.acs.power.real
        assert np.allclose(power1, power2, rtol=0.01)

    def test_from_time_array_works_with_memmap(self):
        with fits.open(os.path.join(datadir, "monol_testA.evt"), memmap=True) as hdul:
            times1 = hdul[1].data["TIME"]

            gti = np.array([[hdul[2].data["START"][0], hdul[2].data["STOP"][0]]])

            times2 = np.sort(np.random.uniform(gti[0, 0], gti[0, 1], 1000))

            _ = AveragedCrossspectrum.from_time_array(
                times1,
                times2,
                segment_size=128,
                dt=self.dt,
                gti=gti,
                norm="none",
                use_common_mean=False,
            )

    @pytest.mark.parametrize("norm", ["frac", "abs", "none", "leahy"])
    def test_from_lc_with_err_works(self, norm):
        lc1 = self.events1.to_lc(self.dt)
        lc2 = self.events2.to_lc(self.dt)
        lc1._counts_err = np.sqrt(lc1.counts.mean()) + np.zeros_like(lc1.counts)
        lc2._counts_err = np.sqrt(lc2.counts.mean()) + np.zeros_like(lc2.counts)
        pds = AveragedCrossspectrum.from_lightcurve(
            lc1, lc2, segment_size=self.segment_size, norm=norm
        )
        pds_ev = AveragedCrossspectrum.from_events(
            self.events1, self.events2, segment_size=self.segment_size, dt=self.dt, norm=norm
        )
        for attr in ["power", "freq", "m", "n", "nphots1", "nphots2", "segment_size"]:
            assert np.allclose(getattr(pds, attr), getattr(pds_ev, attr))

    def test_it_works_with_events(self):
        lc1 = self.events1.to_lc(self.dt)
        lc2 = self.events2.to_lc(self.dt)
        lccs = Crossspectrum(lc1, lc2, norm="none")
        assert np.allclose(lccs.power, self.cs.power)

    def test_no_segment_size(self):
        with pytest.raises(ValueError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2, dt=self.dt)

    def test_init_with_norm_not_str(self):
        with pytest.raises(TypeError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1, norm=1, dt=self.dt)

    def test_init_with_invalid_norm(self):
        with pytest.raises(ValueError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1, norm="frabs", dt=self.dt)

    def test_init_with_inifite_segment_size(self):
        with pytest.raises(ValueError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=np.inf, dt=self.dt)

    def test_coherence(self):
        with pytest.warns(UserWarning) as w:
            coh = self.acs.coherence()
        assert len(coh[0]) == 4999
        assert len(coh[1]) == 4999

    def test_failure_when_normalization_not_recognized(self):
        with pytest.raises(ValueError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1, norm="wrong", dt=self.dt)

    def test_failure_when_power_type_not_recognized(self):
        with pytest.raises(ValueError):
            cs = AveragedCrossspectrum(
                self.lc1, self.lc2, segment_size=1, power_type="wrong", dt=self.dt
            )

    def test_rebin(self):
        new_cs = self.acs.rebin(df=1.5)
        assert new_cs.df == 1.5
        new_cs.time_lag()

    def test_rebin_factor(self):
        new_cs = self.acs.rebin(f=1.5)
        assert new_cs.df == self.acs.df * 1.5
        new_cs.time_lag()

    def test_rebin_log(self):
        # For now, just verify that it doesn't crash
        new_cs = self.acs.rebin_log(f=0.1)
        assert type(new_cs) == type(self.acs)
        new_cs.time_lag()

    def test_rebin_log_returns_complex_values(self):
        # For now, just verify that it doesn't crash
        new_cs = self.acs.rebin_log(f=0.1)
        assert np.iscomplexobj(new_cs.power[0])

    def test_rebin_log_returns_complex_errors(self):
        # For now, just verify that it doesn't crash

        new_cs = self.acs.rebin_log(f=0.1)
        assert np.iscomplexobj(new_cs.power_err[0])


class TestCoherence(object):
    def test_coherence_is_one_on_single_interval(self):
        lc1 = Lightcurve([1, 2, 3, 4, 5], [2, 3, 2, 4, 1])
        lc2 = Lightcurve([1, 2, 3, 4, 5], [4, 8, 1, 9, 11])

        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(lc1, lc2)
            coh = cs.coherence()

        assert np.isclose(len(coh), 2, rtol=0.001)
        # The raw coherence of a single interval is 1 by definition
        assert np.isclose(np.abs(np.mean(coh)), 1, rtol=0.001)

    def test_high_coherence(self):
        t = np.arange(1280)
        a = np.random.poisson(100, len(t))
        lc = Lightcurve(t, a)
        lc2 = Lightcurve(t, copy.deepcopy(a))

        with pytest.warns(UserWarning) as record:
            c = AveragedCrossspectrum(lc, lc2, 128, use_common_mean=True)
            coh, _ = c.coherence()

        assert np.isclose(np.mean(coh).real, 1.0, atol=0.01)


class TestNormalization(object):
    def setup_class(self):
        tstart = 0.0
        self.tseg = 100000.0
        dt = 1

        time = np.arange(tstart + 0.5 * dt, self.tseg + 0.5 * dt, dt)

        np.random.seed(100)
        counts1 = np.random.poisson(10000, size=time.shape[0])
        counts1_norm = counts1 / 13.4
        counts1_norm_err = np.std(counts1) / 13.4
        self.lc1_norm = Lightcurve(
            time,
            counts1_norm,
            gti=[[tstart, self.tseg]],
            dt=dt,
            err_dist="gauss",
            err=np.zeros_like(counts1_norm) + counts1_norm_err,
        )
        self.lc1 = Lightcurve(time, counts1, gti=[[tstart, self.tseg]], dt=dt)
        self.rate1 = np.mean(counts1) / dt  # mean count rate (counts/sec) of light curve 1

        with pytest.warns(UserWarning) as record:
            self.cs = Crossspectrum(self.lc1, self.lc1, norm="none")

        with pytest.warns(UserWarning) as record:
            self.cs_norm = Crossspectrum(self.lc1_norm, self.lc1_norm, norm="none")

    @pytest.mark.parametrize("norm", ["leahy", "abs", "frac", "none"])
    def test_method_norm(self, norm):
        # Testing for a power spectrum of lc1
        cs1 = copy.deepcopy(self.cs)

        unnorm = copy.deepcopy(cs1.unnorm_power)
        new_cs = cs1.to_norm(norm)
        assert norm == new_cs.norm
        assert np.allclose(cs1.unnorm_power[1:], unnorm[1:], atol=0.5)
        power_norm = new_cs.power
        n_ph = np.sqrt(cs1.nphots1 * cs1.nphots2)
        mean1 = cs1.nphots1 / cs1.n
        mean2 = cs1.nphots2 / cs1.n
        mean = np.sqrt(mean1 * mean2)
        noise = poisson_level(norm=norm, meanrate=mean / cs1.dt, n_ph=n_ph)
        assert np.isclose(np.mean(power_norm[1:]), noise, rtol=0.01)

    def test_method_norm_gauss(self):
        norm = "leahy"
        # Testing for a power spectrum of lc1
        cs1 = copy.deepcopy(self.cs_norm)

        unnorm = copy.deepcopy(cs1.unnorm_power)
        new_cs = cs1.to_norm(norm)
        assert norm == new_cs.norm
        assert np.allclose(cs1.unnorm_power[1:], unnorm[1:], atol=0.5)
        power_norm = new_cs.power
        n_ph = np.sqrt(cs1.nphots1 * cs1.nphots2)
        mean1 = cs1.nphots1 / cs1.n
        mean2 = cs1.nphots2 / cs1.n
        mean = np.sqrt(mean1 * mean2)
        noise = poisson_level(norm=norm, meanrate=mean / cs1.dt, n_ph=n_ph)
        assert np.isclose(np.mean(power_norm[1:]), noise, rtol=0.01)

    @pytest.mark.parametrize("power_type", ["abs", "real", "all"])
    @pytest.mark.parametrize("norm", ["leahy", "abs", "frac", "none"])
    def test_method_norm_equivalent_old_method(self, norm, power_type):
        # Testing for a power spectrum of lc1
        cs1 = copy.deepcopy(self.cs)
        cs2 = copy.deepcopy(self.cs)
        cs1.norm = norm
        cs1.power = cs1._normalize_crossspectrum(cs1.unnorm_power)
        cs2 = cs2.to_norm(norm)
        assert np.allclose(cs1.power, cs2.power)

    @pytest.mark.parametrize("power_type", ["all", "real", "absolute"])
    def test_norm_abs(self, power_type):
        # Testing for a power spectrum of lc1
        self.cs.norm = "abs"
        # New lc with the same absolute variance, but mean-subtracted
        norm_lc_sub = copy.deepcopy(self.lc1)
        norm_lc_sub.counts = norm_lc_sub.counts - np.mean(norm_lc_sub.counts)
        norm_lc_sub.err_dist = "gauss"
        cs = Crossspectrum(norm_lc_sub, norm_lc_sub, norm="none")
        cs.norm = "abs"
        cs.power_type = power_type
        self.cs.power_type = power_type

        power = self.cs._normalize_crossspectrum(self.cs.unnorm_power)
        power_norm = cs._normalize_crossspectrum(cs.unnorm_power)
        abs_noise = 2.0 * self.rate1  # expected Poisson noise level
        assert np.isclose(np.mean(power[1:]), abs_noise, rtol=0.01)
        assert np.allclose(power[1:], power_norm[1:], atol=0.5)

    @pytest.mark.parametrize("power_type", ["all", "real", "absolute"])
    def test_norm_leahy(self, power_type):
        self.cs.norm = "leahy"
        self.cs_norm.norm = "leahy"
        self.cs.power_type = power_type
        self.cs_norm.power_type = power_type

        power = self.cs._normalize_crossspectrum(self.cs.unnorm_power)
        power_norm = self.cs_norm._normalize_crossspectrum(self.cs_norm.unnorm_power)

        assert np.allclose(power[1:], power_norm[1:], atol=0.5)
        leahy_noise = 2.0  # expected Poisson noise level
        assert np.isclose(np.mean(power[1:]), leahy_noise, rtol=0.02)

    @pytest.mark.parametrize("power_type", ["all", "real", "absolute"])
    def test_norm_frac(self, power_type):
        self.cs.norm = "frac"
        self.cs_norm.norm = "frac"
        self.cs.power_type = power_type
        self.cs_norm.power_type = power_type
        power = self.cs._normalize_crossspectrum(self.cs.unnorm_power)
        power_norm = self.cs_norm._normalize_crossspectrum(self.cs_norm.unnorm_power)

        assert np.allclose(power[1:], power_norm[1:])
        norm = 2.0 / self.rate1
        assert np.isclose(np.mean(power[1:]), norm, rtol=0.1)

    def test_failure_when_normalization_not_recognized(self):
        with pytest.raises(ValueError):
            power = normalize_crossspectrum(
                self.cs.power,
                self.lc1.tseg,
                self.lc1.n,
                self.cs.nphots1,
                self.cs.nphots2,
                norm="wrong",
            )
        self.cs.norm = "asdgfasdfa"
        self.cs_norm.norm = "adfafaf"
        with pytest.raises(ValueError):
            power = self.cs._normalize_crossspectrum(self.cs.unnorm_power)
        with pytest.raises(ValueError):
            power = self.cs_norm._normalize_crossspectrum(self.cs.unnorm_power)

    def test_failure_wrong_power_type(self):
        self.cs.power_type = "asdgfasdfa"
        self.cs_norm.power_type = "adfafaf"
        self.cs.norm = "leahy"
        self.cs_norm.norm = "leahy"

        with pytest.raises(ValueError):
            power = self.cs._normalize_crossspectrum(self.cs.unnorm_power)
        with pytest.raises(ValueError):
            power = self.cs_norm._normalize_crossspectrum(self.cs.unnorm_power)


class TestCrossspectrum(object):
    def setup_class(self):
        tstart = 0.0
        tend = 1.0
        dt = 0.0001

        time = np.arange(tstart + 0.5 * dt, tend + 0.5 * dt, dt)

        counts1 = np.random.poisson(0.01, size=time.shape[0])
        counts2 = np.random.negative_binomial(1, 0.09, size=time.shape[0])
        self.lc1 = Lightcurve(time, counts1, gti=[[tstart, tend]], dt=dt)
        self.lc2 = Lightcurve(time, counts2, gti=[[tstart, tend]], dt=dt)
        self.rate1 = 100.0  # mean count rate (counts/sec) of light curve 1

        with pytest.warns(UserWarning) as record:
            self.cs = Crossspectrum(self.lc1, self.lc2)

    @pytest.mark.parametrize("skip_checks", [True, False])
    def test_initialize_empty(self, skip_checks):
        cs = Crossspectrum(skip_checks=skip_checks)
        assert cs.freq is None

    def test_lc_keyword_deprecation(self):
        cs1 = Crossspectrum(self.lc1, self.lc2)
        with pytest.warns(DeprecationWarning) as record:
            cs2 = Crossspectrum(lc1=self.lc1, lc2=self.lc2)
        assert np.any(["lcN keywords" in r.message.args[0] for r in record])
        assert np.allclose(cs1.power, cs2.power)
        assert np.allclose(cs1.freq, cs2.freq)

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
        with pytest.raises(ValueError):
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
            cs = Crossspectrum(norm="frabs")

    def test_init_with_wrong_lc_instance(self):
        lc1_ = {"a": 1, "b": 2}
        lc2_ = {"a": 1, "b": 2}
        with pytest.raises(TypeError):
            cs = Crossspectrum(lc1_, lc2_)

    def test_init_with_wrong_lc1_instance(self):
        lc_ = {"a": 1, "b": 2}
        with pytest.raises(TypeError):
            cs = Crossspectrum(self.lc1, lc_)

    def test_init_with_wrong_lc2_instance(self):
        lc_ = {"a": 1, "b": 2}
        with pytest.raises(TypeError):
            cs = Crossspectrum(self.lc1, lc_)

    def test_make_crossspectrum_diff_lc_counts_shape(self):
        counts = np.array([1] * 10001)
        time = np.linspace(0.0, 1.0001, 10001)
        lc_ = Lightcurve(time, counts)
        with pytest.raises(StingrayError):
            cs = Crossspectrum(self.lc1, lc_, legacy=True)

    def test_make_crossspectrum_diff_lc_stat(self):
        lc_ = copy.deepcopy(self.lc1)
        lc_.err_dist = "gauss"

        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(self.lc1, lc_, legacy=True)
        assert np.any(["different statistics" in r.message.args[0] for r in record])

    def test_make_crossspectrum_diff_lc_iter_stat(self):
        lc_ = copy.deepcopy(self.lc1)
        lc_.err_dist = "gauss"

        with pytest.warns(UserWarning) as record:
            cs = AveragedCrossspectrum([self.lc1], [lc_], segment_size=1, legacy=True)
        assert np.any(["different statistics" in r.message.args[0] for r in record])

    def test_make_crossspectrum_bad_lc_stat(self):
        lc1 = copy.deepcopy(self.lc1)
        lc1.err_dist = "gauss"
        lc2 = copy.deepcopy(self.lc1)
        lc2.err_dist = "gauss"

        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(lc1, lc2, legacy=True)
        assert np.any(["is not poisson" in r.message.args[0] for r in record])

    def test_make_crossspectrum_diff_dt(self):
        counts = np.array([1] * 10000)
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

    @pytest.mark.parametrize("legacy", [True, False])
    def test_norm_abs(self, legacy):
        # Testing for a power spectrum of lc1
        cs = Crossspectrum(self.lc1, self.lc1, norm="abs", legacy=legacy)
        assert len(cs.power) == 4999
        assert cs.norm == "abs"
        abs_noise = 2.0 * self.rate1  # expected Poisson noise level
        assert np.isclose(np.mean(cs.power[1:]), abs_noise)

    @pytest.mark.parametrize("legacy", [True, False])
    def test_norm_leahy(self, legacy):
        # with pytest.warns(UserWarning) as record:
        cs = Crossspectrum(self.lc1, self.lc1, norm="leahy", legacy=legacy)
        assert len(cs.power) == 4999
        assert cs.norm == "leahy"
        leahy_noise = 2.0  # expected Poisson noise level
        assert np.isclose(np.mean(cs.power[1:]), leahy_noise, rtol=0.02)

    def test_norm_frac(self):
        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(self.lc1, self.lc1, norm="frac")
        assert len(cs.power) == 4999
        assert cs.norm == "frac"
        norm = 2.0 / self.rate1
        assert np.isclose(np.mean(cs.power[1:]), norm, rtol=0.2)

    def test_norm_abs(self):
        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(self.lc1, self.lc2, norm="abs")
        assert len(cs.power) == 4999
        assert cs.norm == "abs"

    def test_failure_when_normalization_not_recognized(self):
        with pytest.raises(ValueError):
            cs = Crossspectrum(self.lc1, self.lc2, norm="wrong")

    def test_coherence_one_on_single_interval(self):
        coh = self.cs.coherence()
        assert len(coh) == 4999
        assert np.isclose(coh[0], 1)

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

    def test_plot_simple(self):
        self.cs.plot()
        assert plt.fignum_exists("crossspectrum")
        plt.close("crossspectrum")

    def test_plot_labels_and_fname(self):
        outfname = "blabla.png"
        if os.path.exists(outfname):
            os.unlink(outfname)

        self.cs.plot(labels=["x", "y"], axis=[0, 10, 0, 10], filename=outfname, save=True)
        assert os.path.exists(outfname)
        os.unlink(outfname)

    def test_plot_labels_and_fname_default(self):
        outfname = "spec.png"
        if os.path.exists(outfname):
            os.unlink(outfname)
        self.cs.plot(labels=["x", "y"], save=True)
        assert os.path.exists(outfname)
        os.unlink(outfname)

    def test_plot_single_label(self):
        with pytest.warns(UserWarning) as record:
            self.cs.plot(labels=["x"])
        assert np.any(["must have two labels" in r.message.args[0] for r in record])

    def test_plot_axes(self):
        plt.subplot(211)
        plot2 = self.cs.plot(
            ax=plt.subplot(212), labels=("frequency", "amplitude"), title="Crossspectrum_leahy"
        )
        assert plt.fignum_exists(1)
        plt.close(1)

    def test_plot_labels_and_fname_for_axes(self):
        outfname = "blabla.png"
        if os.path.exists(outfname):
            os.unlink(outfname)

        plt.subplot(211)
        plot2 = self.cs.plot(
            ax=plt.subplot(212),
            labels=("frequency", "amplitude"),
            title="Crossspectrum_leahy",
            filename=outfname,
            save=True,
        )
        assert os.path.exists(outfname)
        os.unlink(outfname)

    def test_plot_labels_and_fname_for_axes_default(self):
        outfname = "spec.png"
        if os.path.exists(outfname):
            os.unlink(outfname)

        plt.subplot(211)
        plot2 = self.cs.plot(
            ax=plt.subplot(212),
            labels=("frequency", "amplitude"),
            title="Crossspectrum_leahy",
            save=True,
        )
        assert os.path.exists(outfname)
        os.unlink(outfname)

    def test_rebin_error(self):
        cs = Crossspectrum()
        with pytest.raises(ValueError):
            cs.rebin()

    def test_classical_significances_runs(self):
        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(self.lc1, self.lc2, norm="leahy")
        cs.classical_significances()

    def test_classical_significances_fails_in_rms(self):
        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(self.lc1, self.lc2, norm="frac")
        with pytest.raises(ValueError):
            cs.classical_significances()

    def test_classical_significances_threshold(self):
        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(self.lc1, self.lc2, norm="leahy")

        # change the powers so that just one exceeds the threshold
        cs.power = np.zeros_like(cs.power) + 2.0

        index = 1
        cs.power[index] = 10.0

        threshold = 0.01

        pval = cs.classical_significances(threshold=threshold, trial_correction=False)
        assert pval[0, 0] < threshold
        assert pval[1, 0] == index

    def test_classical_significances_trial_correction(self):
        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(self.lc1, self.lc2, norm="leahy")
        # change the powers so that just one exceeds the threshold
        cs.power = np.zeros_like(cs.power) + 2.0
        index = 1
        cs.power[index] = 10.0
        threshold = 0.01
        pval = cs.classical_significances(threshold=threshold, trial_correction=True)
        assert np.size(pval) == 0

    def test_classical_significances_with_logbinned_psd(self):
        with pytest.warns(UserWarning) as record:
            cs = Crossspectrum(self.lc1, self.lc2, norm="leahy")
        cs_log = cs.rebin_log()
        pval = cs_log.classical_significances(threshold=1.1, trial_correction=False)

        assert len(pval[0]) == len(cs_log.power)

    def test_pvals_is_numpy_array(self):
        cs = Crossspectrum(self.lc1, self.lc2, norm="leahy")
        # change the powers so that just one exceeds the threshold
        cs.power = np.zeros_like(cs.power) + 2.0

        index = 1
        cs.power[index] = 10.0

        threshold = 1.0

        pval = cs.classical_significances(threshold=threshold, trial_correction=True)

        assert isinstance(pval, np.ndarray)
        assert pval.shape[0] == 2

    @pytest.mark.parametrize("legacy", [True, False])
    def test_fullspec(self, legacy):
        csT = Crossspectrum(self.lc1, self.lc2, fullspec=True, legacy=legacy)
        assert csT.fullspec == True
        assert self.cs.fullspec == False
        assert csT.n == self.cs.n
        assert csT.n == len(csT.power)
        assert self.cs.n != len(self.cs.power)
        assert len(csT.power) >= len(self.cs.power)
        assert len(csT.power) == len(self.lc1)
        assert csT.freq[csT.n // 2] <= 0.0


class TestAveragedCrossspectrum(object):
    def setup_class(self):
        tstart = 0.0
        tend = 1.0
        dt = np.longdouble(0.0001)

        time = np.arange(tstart + 0.5 * dt, tend + 0.5 * dt, dt)

        counts1 = np.random.poisson(0.01, size=time.shape[0])
        counts2 = np.random.negative_binomial(1, 0.09, size=time.shape[0])

        self.lc1 = Lightcurve(time, counts1, gti=[[tstart, tend]], dt=dt)
        self.lc2 = Lightcurve(time, counts2, gti=[[tstart, tend]], dt=dt)

        with pytest.warns(UserWarning) as record:
            self.cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1, save_all=True)
        assert np.any(
            ["The large_data option and the save_all" in r.message.args[0] for r in record]
        )

    @pytest.mark.parametrize("skip_checks", [True, False])
    def test_initialize_empty(self, skip_checks):
        cs = AveragedCrossspectrum(skip_checks=skip_checks)
        assert cs.freq is None

    def test_save_all(self):
        with pytest.warns(UserWarning) as record:
            cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1, save_all=True)
        assert np.any(
            ["The large_data option and the save_all" in r.message.args[0] for r in record]
        )
        assert hasattr(self.cs, "cs_all")

    def test_lc_keyword_deprecation(self):
        cs1 = AveragedCrossspectrum(data1=self.lc1, data2=self.lc2, segment_size=1)
        with pytest.warns(DeprecationWarning) as record:
            cs2 = AveragedCrossspectrum(lc1=self.lc1, lc2=self.lc2, segment_size=1)
        assert np.any(["lcN keywords" in r.message.args[0] for r in record])
        assert np.allclose(cs1.power, cs2.power)
        assert np.allclose(cs1.freq, cs2.freq)

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

    def test_no_counts_warns(self):
        newlc = copy.deepcopy(self.lc1)
        newlc.counts[: newlc.counts.size // 2] = 0 * newlc.counts[: newlc.counts.size // 2]
        with pytest.warns(UserWarning) as record:
            ps = AveragedCrossspectrum(newlc, self.lc2, segment_size=0.2, legacy=True)

        assert np.any(["No counts in " in r.message.args[0] for r in record])

    def test_no_segment_size(self):
        with pytest.raises(ValueError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2)

    def test_invalid_type_attribute(self):
        with pytest.raises(ValueError):
            cs_test = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1, legacy=True)
            cs_test.type = "invalid_type"
            assert AveragedCrossspectrum._make_crossspectrum(cs_test, self.lc1, self.lc2)

    def test_invalid_type_attribute_with_multiple_lcs(self):
        with pytest.warns(UserWarning) as record:
            acs_test = AveragedCrossspectrum(
                [self.lc1, self.lc2], [self.lc2, self.lc1], segment_size=1
            )
        acs_test.type = "invalid_type"
        with pytest.raises(ValueError) as excinfo:
            assert AveragedCrossspectrum._make_crossspectrum(
                acs_test, [self.lc1, self.lc2], [self.lc2, self.lc1]
            )
        assert "Type of spectrum not recognized" in str(excinfo.value)

    def test_different_dt(self):
        time1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        counts1_test = np.random.poisson(0.01, size=len(time1))
        test_lc1 = Lightcurve(time1, counts1_test)

        time2 = [2, 4, 6, 8, 10]
        counts2_test = np.random.negative_binomial(1, 0.09, size=len(time2))
        test_lc2 = Lightcurve(time2, counts2_test)

        assert test_lc1.tseg == test_lc2.tseg

        assert test_lc1.dt != test_lc2.dt

        with pytest.raises(StingrayError):
            assert AveragedCrossspectrum(test_lc1, test_lc2, segment_size=1)

    def test_different_tseg(self):
        time2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        counts2_test = np.random.poisson(1000, size=len(time2))
        test_lc2 = Lightcurve(time2, counts2_test)

        time1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        counts1_test = np.random.poisson(1000, size=len(time1))
        test_lc1 = Lightcurve(time1, counts1_test)

        assert test_lc2.dt == test_lc1.dt

        assert test_lc2.tseg != test_lc1.tseg

        with pytest.warns(UserWarning) as record:
            AveragedCrossspectrum(test_lc1, test_lc2, segment_size=5, legacy=True)
            assert np.any(["same tseg" in r.message.args[0] for r in record])

    def test_with_zero_counts(self):
        nbins = 100
        x = np.linspace(0, 10, nbins)
        ycounts1 = np.random.normal(loc=10, scale=0.5, size=int(0.4 * nbins))
        ycounts2 = np.random.normal(loc=10, scale=0.5, size=int(0.4 * nbins))

        yzero = np.zeros(int(0.6 * nbins))
        y1 = np.hstack([ycounts1, yzero])
        y2 = np.hstack([ycounts2, yzero])

        lc1 = Lightcurve(x, y1)
        lc2 = Lightcurve(x, y2)

        with pytest.warns(UserWarning) as record:
            acs = AveragedCrossspectrum(lc1, lc2, segment_size=5.0, norm="leahy", legacy=True)
        assert acs.m == 1
        assert np.any(["No counts in interval" in r.message.args[0] for r in record])

    def test_rebin_with_invalid_type_attribute(self):
        new_df = 2

        with pytest.warns(UserWarning) as record:
            aps = AveragedCrossspectrum(
                lc1=self.lc1, lc2=self.lc2, segment_size=1, norm="leahy", legacy=True
            )
        aps.type = "invalid_type"
        with pytest.raises(ValueError) as excinfo:
            assert aps.rebin(df=new_df, method=aps.type)
        assert "Method for summing or averaging not recognized. " in str(excinfo.value)

    @pytest.mark.parametrize("legacy", [True, False])
    def test_rebin_with_valid_type_attribute(self, legacy):
        new_df = 2
        save_all = legacy
        aps = AveragedCrossspectrum(
            self.lc1, self.lc2, segment_size=1, norm="leahy", legacy=legacy, save_all=save_all
        )

        assert aps.rebin(df=new_df)

    def test_init_with_norm_not_str(self):
        with pytest.raises(TypeError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1, norm=1)

    def test_init_with_invalid_norm(self):
        with pytest.raises(ValueError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1, norm="frabs")

    def test_init_with_inifite_segment_size(self):
        with pytest.raises(ValueError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=np.inf)

    @pytest.mark.parametrize("legacy", [False, True])
    @pytest.mark.parametrize("err_dist", ["poisson", "gauss"])
    def test_with_iterable_of_lightcurves(self, legacy, err_dist):
        def iter_lc(lc, n):
            "Generator of n parts of lc."
            t0 = int(len(lc) / n)
            t = t0
            i = 0
            while True:
                lc_seg = lc[i:t]
                yield lc_seg
                if t + t0 > len(lc):
                    break
                else:
                    i, t = t, t + t0

        lc1 = copy.deepcopy(self.lc1)
        lc2 = copy.deepcopy(self.lc2)
        lc1.err_dist = lc2.err_dist = err_dist
        with pytest.warns(UserWarning) as record:
            cs = AveragedCrossspectrum(
                iter_lc(self.lc1, 1), iter_lc(self.lc2, 1), segment_size=1, legacy=legacy
            )
        message = "The averaged Cross spectrum from a generator "

        assert np.any([message in r.message.args[0] for r in record])

    def test_with_multiple_lightcurves_variable_length(self):
        gti = [[0, 0.05], [0.05, 0.5], [0.555, 1.0]]
        lc1 = copy.deepcopy(self.lc1)
        lc1.gti = gti
        lc2 = copy.deepcopy(self.lc2)
        lc2.gti = gti

        lc1_split = lc1.split_by_gti()
        lc2_split = lc2.split_by_gti()

        cs = AveragedCrossspectrum(
            lc1_split, lc2_split, segment_size=0.05, norm="leahy", silent=True
        )

    def test_coherence(self):
        with warnings.catch_warnings(record=True) as w:
            coh = self.cs.coherence()

            assert len(coh[0]) == 4999
            assert len(coh[1]) == 4999
            assert issubclass(w[-1].category, UserWarning)

    def test_failure_when_normalization_not_recognized(self):
        with pytest.raises(ValueError):
            self.cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1, norm="wrong")

    def test_failure_when_power_type_not_recognized(self):
        with pytest.raises(ValueError):
            self.cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1, power_type="wrong")

    def test_old_normalize_crossspectrum_warns(self):
        with pytest.warns(DeprecationWarning):
            normalize_crossspectrum(1.0, 2.0, 3.0, 4.0, 5.0, norm="abs")

    def test_old_normalize_crossspectrum_gauss_warns(self):
        with pytest.warns(DeprecationWarning):
            normalize_crossspectrum_gauss(1.0, 2.0, 3.0, 4.0, 5.0, norm="abs")

    def test_normalize_crossspectrum(self):
        cs1 = Crossspectrum(self.lc1, self.lc2, norm="leahy")
        cs2 = Crossspectrum(self.lc1, self.lc2, norm="leahy", power_type="all")
        cs3 = Crossspectrum(self.lc1, self.lc2, norm="leahy", power_type="real")
        cs4 = Crossspectrum(self.lc1, self.lc2, norm="leahy", power_type="absolute")
        assert np.allclose(cs1.power.real, cs3.power)
        assert np.all(np.isclose(np.abs(cs2.power), cs4.power, atol=0.0001))

    def test_normalize_crossspectrum_with_method_inplace(self):
        cs1 = AveragedCrossspectrum.from_lightcurve(self.lc1, self.lc2, segment_size=1, norm="abs")
        cs2 = cs1.to_norm("leahy", inplace=True)
        cs3 = cs1.to_norm("leahy", inplace=False)
        assert cs3 is not cs1
        assert cs2 is cs1

    @pytest.mark.parametrize("norm1", ["leahy", "abs", "frac", "none"])
    @pytest.mark.parametrize("norm2", ["leahy", "abs", "frac", "none"])
    def test_normalize_crossspectrum_with_method(self, norm1, norm2):
        cs1 = AveragedCrossspectrum.from_lightcurve(self.lc1, self.lc2, segment_size=1, norm=norm1)
        cs2 = AveragedCrossspectrum.from_lightcurve(self.lc1, self.lc2, segment_size=1, norm=norm2)
        cs3 = cs2.to_norm(norm1)
        for attr in ["power", "power_err", "unnorm_power", "unnorm_power_err"]:
            assert np.allclose(getattr(cs1, attr), getattr(cs3, attr))
            assert np.allclose(getattr(cs1.pds1, attr), getattr(cs3.pds1, attr))
            assert np.allclose(getattr(cs1.pds2, attr), getattr(cs3.pds2, attr))

    @pytest.mark.parametrize("f", [None, 1.5])
    @pytest.mark.parametrize("norm", ["leahy", "abs", "frac", "none"])
    def test_rebin_factor_rebins_all_attrs(self, f, norm):
        cs1 = AveragedCrossspectrum.from_lightcurve(self.lc1, self.lc2, segment_size=1, norm=norm)
        # N.B.: if f is not None, df gets ignored.
        new_cs = cs1.rebin(df=1.5, f=f)
        N = new_cs.freq.size
        for attr in ["power", "power_err", "unnorm_power", "unnorm_power_err"]:
            assert hasattr(new_cs, attr) and getattr(new_cs, attr).size == N
            assert hasattr(new_cs.pds1, attr) and getattr(new_cs.pds1, attr).size == N
            assert hasattr(new_cs.pds2, attr) and getattr(new_cs.pds2, attr).size == N

        for attr in cs1.meta_attrs():
            if attr not in ["df", "gti", "m"]:
                assert getattr(cs1, attr) == getattr(new_cs, attr)

    @pytest.mark.parametrize("norm", ["leahy", "abs", "frac", "none"])
    def test_rebin_factor_log_rebins_all_attrs(self, norm):
        cs1 = AveragedCrossspectrum.from_lightcurve(self.lc1, self.lc2, segment_size=1, norm=norm)
        new_cs = cs1.rebin_log(0.03)
        N = new_cs.freq.size
        for attr in ["power", "power_err", "unnorm_power", "unnorm_power_err"]:
            assert hasattr(new_cs, attr) and getattr(new_cs, attr).size == N
            assert hasattr(new_cs.pds1, attr) and getattr(new_cs.pds1, attr).size == N
            assert hasattr(new_cs.pds2, attr) and getattr(new_cs.pds2, attr).size == N

        for attr in cs1.meta_attrs():
            if attr not in ["df", "gti", "m", "k"]:
                assert np.all(getattr(cs1, attr) == getattr(new_cs, attr))

    def test_rebin(self):
        with warnings.catch_warnings(record=True) as w:
            new_cs = self.cs.rebin(df=1.5)
        assert hasattr(new_cs, "dt") and new_cs.dt is not None
        assert new_cs.df == 1.5
        new_cs.time_lag()

    def test_rebin_factor(self):
        with warnings.catch_warnings(record=True) as w:
            new_cs = self.cs.rebin(f=1.5)
        assert hasattr(new_cs, "dt") and new_cs.dt is not None
        assert new_cs.df == self.cs.df * 1.5
        new_cs.time_lag()

    def test_rebin_log(self):
        # For now, just verify that it doesn't crash
        with warnings.catch_warnings(record=True) as w:
            new_cs = self.cs.rebin_log(f=0.1)
        assert hasattr(new_cs, "dt") and new_cs.dt is not None
        assert type(new_cs) == type(self.cs)
        new_cs.time_lag()

    def test_rebin_log_returns_complex_values_and_errors(self):
        # For now, just verify that it doesn't crash
        with warnings.catch_warnings(record=True) as w:
            new_cs = self.cs.rebin_log(f=0.1)
        assert np.iscomplexobj(new_cs.power[0])
        assert np.iscomplexobj(new_cs.power_err[0])

    def test_timelag(self):
        dt = 0.1
        simulator = Simulator(dt, 10000, rms=0.2, mean=1000)
        test_lc1 = simulator.simulate(2)
        test_lc1.counts -= np.min(test_lc1.counts)

        with pytest.warns(UserWarning):
            test_lc1 = Lightcurve(test_lc1.time, test_lc1.counts, err_dist=test_lc1.err_dist, dt=dt)
            # The second light curve is delayed by two bins.
            # The time lag should be -2 * dt, because this will
            # become the reference band in AveragedCrossspectrum
            test_lc2 = Lightcurve(
                test_lc1.time,
                np.array(np.roll(test_lc1.counts, 2)),
                err_dist=test_lc1.err_dist,
                dt=dt,
            )

        with warnings.catch_warnings(record=True) as w:
            cs = AveragedCrossspectrum(test_lc1, test_lc2, segment_size=5, norm="none")

            time_lag, time_lag_err = cs.time_lag()

        # The actual measured time lag will be half that for AveragedCrosspectrum
        measured_lag = -dt
        assert np.all(np.abs(time_lag[:6] - measured_lag) < 3 * time_lag_err[:6])

    def test_errorbars_legacy(self):
        time = np.arange(10000) * 0.1
        test_lc1 = Lightcurve(time, np.random.poisson(200, 10000))
        test_lc2 = Lightcurve(time, np.random.poisson(200, 10000))

        with warnings.catch_warnings(record=True) as w:
            cs = AveragedCrossspectrum(
                test_lc1, test_lc2, segment_size=10, norm="leahy", legacy=True
            )

        assert np.allclose(cs.power_err, np.sqrt(2 / cs.m))

    def test_classical_significances(self):
        time = np.arange(10000) * 0.1
        np.random.seed(62)
        test_lc1 = Lightcurve(time, np.random.poisson(200, 10000))
        test_lc2 = Lightcurve(time, np.random.poisson(200, 10000))
        with warnings.catch_warnings(record=True) as w:
            cs = AveragedCrossspectrum(test_lc1, test_lc2, segment_size=10, norm="leahy")
        maxpower = np.max(cs.power)
        assert np.all(np.isfinite(cs.classical_significances(threshold=maxpower / 2.0)))


class TestCoherenceFunction(object):
    def setup_class(self):
        self.lc1 = Lightcurve([1, 2, 3, 4, 5], [2, 3, 2, 4, 1])
        self.lc2 = Lightcurve([1, 2, 3, 4, 5], [4, 8, 1, 9, 11])

    def test_coherence_runs(self):
        with pytest.warns(DeprecationWarning):
            coherence(self.lc1, self.lc2)

    def test_coherence_fails_if_data1_not_lc(self):
        data = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 1]])

        with pytest.warns(DeprecationWarning):
            with pytest.raises(TypeError):
                coherence(self.lc1, data)

    def test_coherence_fails_if_data2_not_lc(self):
        data = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 1]])

        with pytest.warns(DeprecationWarning):
            with pytest.raises(TypeError):
                coherence(data, self.lc2)

    def test_coherence_computes_correctly(self):
        with pytest.warns(DeprecationWarning):
            coh = coherence(self.lc1, self.lc2)

        assert np.isclose(len(coh), 2, rtol=0.001)
        assert np.isclose(np.abs(np.mean(coh)), 1, rtol=0.001)


class TestTimelagFunction(object):
    def setup_class(self):
        self.lc1 = Lightcurve([1, 2, 3, 4, 5], [2, 3, 2, 4, 1])
        self.lc2 = Lightcurve([1, 2, 3, 4, 5], [4, 8, 1, 9, 11])

    def test_time_lag_runs(self):
        with pytest.warns(DeprecationWarning):
            time_lag(self.lc1, self.lc2)

    def test_time_lag_fails_if_data1_not_lc(self):
        data = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 1]])

        with pytest.warns(DeprecationWarning):
            with pytest.raises(TypeError):
                time_lag(self.lc1, data)

    def test_time_lag_fails_if_data2_not_lc(self):
        data = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 1]])

        with pytest.warns(DeprecationWarning):
            with pytest.raises(TypeError):
                time_lag(data, self.lc2)

    def test_time_lag_computes_correctly(self):
        with pytest.warns(DeprecationWarning):
            lag = time_lag(self.lc1, self.lc2)

        assert np.max(lag) <= np.pi
        assert np.min(lag) >= -np.pi


class TestRoundTrip:
    @classmethod
    def setup_class(cls):
        cls.cs = Crossspectrum()
        cls.cs.freq = np.arange(10)
        cls.cs.power = np.random.uniform(0, 10, 10) + 3j
        cls.cs.m = 1
        cls.cs.nphots1 = 34
        cls.cs.nphots2 = 25

    def _check_equal(self, so, new_so):
        for attr in ["freq", "power"]:
            assert np.allclose(getattr(so, attr), getattr(new_so, attr))

        for attr in ["m", "nphots1", "nphots2"]:
            assert getattr(so, attr) == getattr(new_so, attr)

    def test_astropy_roundtrip(self):
        so = self.cs
        ts = so.to_astropy_table()
        new_so = so.from_astropy_table(ts)
        self._check_equal(so, new_so)

    @pytest.mark.skipif("not _HAS_XARRAY")
    def test_xarray_roundtrip(self):
        so = self.cs
        ts = so.to_xarray()
        new_so = so.from_xarray(ts)

        self._check_equal(so, new_so)

    @pytest.mark.skipif("not _HAS_PANDAS")
    def test_pandas_roundtrip(self):
        so = self.cs
        ts = so.to_pandas()
        new_so = so.from_pandas(ts)

        self._check_equal(so, new_so)

    @pytest.mark.parametrize("fmt", ["pickle", "ascii", "ascii.ecsv", "fits", "hdf5"])
    def test_file_roundtrip(self, fmt):
        so = self.cs
        fname = f"dummy.{fmt}"
        if not _HAS_H5PY and fmt == "hdf5":
            with pytest.raises(Exception) as excinfo:
                so.write(fname, fmt=fmt)
                assert h5py in str(excinfo.value)
            return True
        so.write(fname, fmt=fmt)
        new_so = so.read(fname, fmt=fmt)
        os.unlink(fname)

        self._check_equal(so, new_so)
