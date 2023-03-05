import os
from pickle import FALSE
import pytest
from stingray.fourier import *

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")


def compare_tables(table1, table2, rtol=0.001, discard=[]):
    for key in table1.meta.keys():
        if key in discard:
            continue
        oe, oc = table1.meta[key], table2.meta[key]

        if isinstance(oe, (int, str)):
            assert oe == oc
        elif oe is None:
            assert oc is None
        elif isinstance(oe, Iterable):
            assert np.allclose(oe, oc, rtol=rtol)
        else:
            assert np.isclose(oe, oc, rtol=rtol)
    for col in table1.colnames:
        if col in discard:
            continue
        oe, oc = table1[col], table2[col]
        assert np.allclose(oe, oc, rtol=rtol)


def test_norm():
    mean = var = 100000
    N = 1000000
    dt = 0.2
    meanrate = mean / dt
    lc = np.random.poisson(mean, N)
    pds = np.abs(fft(lc)) ** 2
    freq = fftfreq(N, dt)
    good = slice(1, N // 2)

    pdsabs = normalize_abs(pds, dt, lc.size)
    pdsfrac = normalize_frac(pds, dt, lc.size, mean)
    pois_abs = poisson_level(meanrate=meanrate, norm="abs")
    pois_frac = poisson_level(meanrate=meanrate, norm="frac")

    assert np.isclose(pdsabs[good].mean(), pois_abs, rtol=0.01)
    assert np.isclose(pdsfrac[good].mean(), pois_frac, rtol=0.01)


class TestCoherence(object):
    @classmethod
    def setup_class(cls):
        data = np.load(os.path.join(datadir, "sample_variable_lc.npy"))[:10000] * 1000
        cls.data1 = np.random.poisson(data)
        cls.data2 = np.random.poisson(data)
        ft1 = np.fft.fft(cls.data1)
        ft2 = np.fft.fft(cls.data2)
        dt = 0.01
        cls.N = data.size
        mean = np.mean(data)
        meanrate = mean / dt
        freq = np.fft.fftfreq(data.size, dt)
        good = (freq > 0) & (freq < 0.1)
        ft1, ft2 = ft1[good], ft2[good]
        cls.cross = normalize_periodograms(
            ft1.conj() * ft2, dt, cls.N, mean, norm="abs", power_type="all"
        )
        cls.pds1 = normalize_periodograms(
            ft1 * ft1.conj(), dt, cls.N, mean, norm="abs", power_type="real"
        )
        cls.pds2 = normalize_periodograms(
            ft2 * ft2.conj(), dt, cls.N, mean, norm="abs", power_type="real"
        )

        cls.p1noise = poisson_level(meanrate=meanrate, norm="abs")
        cls.p2noise = poisson_level(meanrate=meanrate, norm="abs")

    def test_intrinsic_coherence(self):
        coh = estimate_intrinsic_coherence(
            self.cross, self.pds1, self.pds2, self.p1noise, self.p2noise, self.N
        )
        assert np.allclose(coh, 1, atol=0.001)

    def test_raw_high_coherence(self):
        coh = raw_coherence(self.cross, self.pds1, self.pds2, self.p1noise, self.p2noise, self.N)
        assert np.allclose(coh, 1, atol=0.001)

    def test_raw_low_coherence(self):
        nbins = 2
        C, P1, P2 = self.cross[:nbins], self.pds1[:nbins], self.pds2[:nbins]
        bsq = bias_term(P1, P2, self.p1noise, self.p2noise, self.N)
        # must be lower than bsq!
        low_coh_cross = np.random.normal(bsq**0.5 / 10, bsq**0.5 / 100) + 0.0j
        coh = raw_coherence(low_coh_cross, P1, P2, self.p1noise, self.p2noise, self.N)
        assert np.allclose(coh, 0)
        # Do it with a single number
        coh = raw_coherence(low_coh_cross[0], P1[0], P2[0], self.p1noise, self.p2noise, self.N)
        # Do it with a single complex object
        coh = raw_coherence(
            complex(low_coh_cross[0]), P1[0], P2[0], self.p1noise, self.p2noise, self.N
        )

    def test_raw_high_bias(self):
        """Test when squared bias higher than squared norm of cross spec"""
        # Values chosen to have a high bias term, larger than |C|^2
        C = np.array([12986.0 + 8694.0j])
        P1 = np.array([476156.0])
        P2 = np.array([482751.0])
        P1noise = 495955
        P2noise = 494967

        coh = raw_coherence(C, P1, P2, P1noise, P2noise, 499, 1)
        coh_sngl = raw_coherence(C[0], P1[0], P2[0], P1noise, P2noise, 499, 1)
        assert np.allclose(coh, (C * np.conj(C)).real / (P1 * P2))
        assert np.isclose(coh_sngl, (C * np.conj(C)).real[0] / (P1[0] * P2[0]))


class TestFourier(object):
    @classmethod
    def setup_class(cls):
        cls.dt = 1
        cls.length = 100
        cls.ctrate = 10000
        cls.N = np.rint(cls.length / cls.dt).astype(int)
        cls.dt = cls.length / cls.N
        cls.times = np.sort(np.random.uniform(0, cls.length, int(cls.length * cls.ctrate)))
        cls.gti = np.asarray([[0, cls.length]])
        cls.counts, bins = np.histogram(cls.times, bins=np.linspace(0, cls.length, cls.N + 1))
        cls.errs = np.ones_like(cls.counts) * np.sqrt(cls.ctrate)
        cls.bin_times = (bins[:-1] + bins[1:]) / 2
        cls.segment_size = 5.0
        cls.times2 = np.sort(np.random.uniform(0, cls.length, int(cls.length * cls.ctrate)))
        cls.counts2, _ = np.histogram(cls.times2, bins=np.linspace(0, cls.length, cls.N + 1))
        cls.errs2 = np.ones_like(cls.counts2) * np.sqrt(cls.ctrate)

    def test_error_on_averaged_cross_spectrum_low_nave(self):
        with pytest.warns(UserWarning) as record:
            error_on_averaged_cross_spectrum(4 + 1.0j, 2, 4, 29, 2, 2)
        assert np.any(["n_ave is below 30." in r.message.args[0] for r in record])

    def test_ctrate_events(self):
        assert get_average_ctrate(self.times, self.gti, self.segment_size) == self.ctrate

    def test_ctrate_counts(self):
        assert (
            get_average_ctrate(self.bin_times, self.gti, self.segment_size, self.counts)
            == self.ctrate
        )

    def test_fts_from_segments_invalid(self):
        with pytest.raises(ValueError) as excinfo:
            # N and counts are both None. This should make the function fail immediately
            for _ in get_flux_iterable_from_segments(1, 2, 3, n_bin=None, fluxes=None):
                pass
        assert "At least one between fluxes" in str(excinfo.value)

    def test_fts_from_segments_cts_and_events_are_equal(self):
        N = np.rint(self.segment_size / self.dt).astype(int)
        fts_evts = [
            f
            for f in get_flux_iterable_from_segments(
                self.times, self.gti, self.segment_size, n_bin=N
            )
        ]
        fts_cts = [
            f
            for f in get_flux_iterable_from_segments(
                self.bin_times, self.gti, self.segment_size, fluxes=self.counts
            )
        ]
        for fe, fc in zip(fts_evts, fts_cts):
            assert np.allclose(fe, fc)

    def test_avg_pds_bad_input(self):
        times = np.sort(np.random.uniform(0, 1000, 1))
        out_ev = avg_pds_from_events(times, self.gti, self.segment_size, self.dt)
        assert out_ev is None

    @pytest.mark.parametrize("return_auxil", [True, False])
    def test_avg_cs_bad_input(self, return_auxil):
        times1 = np.sort(np.random.uniform(0, 1000, 1))
        times2 = np.sort(np.random.uniform(0, 1000, 1))
        out_ev = avg_cs_from_events(
            times1, times2, self.gti, self.segment_size, self.dt, return_auxil=return_auxil
        )
        assert out_ev is None

    @pytest.mark.parametrize("norm", ["frac", "abs", "none", "leahy"])
    def test_avg_pds_use_common_mean_similar_stats(self, norm):
        out_comm = avg_pds_from_events(
            self.times,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=True,
            silent=True,
            fluxes=None,
        )["power"]
        out = avg_pds_from_events(
            self.times,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=False,
            silent=True,
            fluxes=None,
        )["power"]
        assert np.isclose(out_comm.std(), out.std(), rtol=0.1)

    @pytest.mark.parametrize("norm", ["frac", "abs", "none", "leahy"])
    def test_avg_cs_use_common_mean_similar_stats(self, norm):
        out_comm = avg_cs_from_events(
            self.times,
            self.times2,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=True,
            silent=True,
        )["power"]
        out = avg_cs_from_events(
            self.times,
            self.times2,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=False,
            silent=True,
        )["power"]
        assert np.isclose(out_comm.std(), out.std(), rtol=0.1)

    @pytest.mark.parametrize("use_common_mean", [True, False])
    @pytest.mark.parametrize("norm", ["frac", "abs", "none", "leahy"])
    def test_avg_pds_cts_and_events_are_equal(self, norm, use_common_mean):
        out_ev = avg_pds_from_events(
            self.times,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=use_common_mean,
            silent=True,
            fluxes=None,
        )
        out_ct = avg_pds_from_events(
            self.bin_times,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=use_common_mean,
            silent=True,
            fluxes=self.counts,
        )
        compare_tables(out_ev, out_ct)

    @pytest.mark.parametrize("use_common_mean", [True, False])
    @pytest.mark.parametrize("norm", ["frac", "abs", "none", "leahy"])
    def test_avg_pds_cts_and_err_and_events_are_equal(self, norm, use_common_mean):
        out_ev = avg_pds_from_events(
            self.times,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=use_common_mean,
            silent=True,
            fluxes=None,
        )
        out_ct = avg_pds_from_events(
            self.bin_times,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=use_common_mean,
            silent=True,
            fluxes=self.counts,
            errors=self.errs,
        )
        # The variance is not _supposed_ to be equal, when we specify errors
        if use_common_mean:
            compare_tables(out_ev, out_ct, rtol=0.01, discard=["variance"])
        else:
            compare_tables(out_ev, out_ct, rtol=0.1, discard=["variance"])

    @pytest.mark.parametrize("use_common_mean", [True, False])
    @pytest.mark.parametrize("norm", ["frac", "abs", "none", "leahy"])
    def test_avg_cs_cts_and_events_are_equal(self, norm, use_common_mean):
        out_ev = avg_cs_from_events(
            self.times,
            self.times2,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=use_common_mean,
            silent=False,
        )
        out_ct = avg_cs_from_events(
            self.bin_times,
            self.bin_times,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=use_common_mean,
            silent=False,
            fluxes1=self.counts,
            fluxes2=self.counts2,
        )
        if use_common_mean:
            compare_tables(out_ev, out_ct, rtol=0.01)
        else:
            compare_tables(out_ev, out_ct, rtol=0.1)

    @pytest.mark.parametrize("use_common_mean", [True, False])
    @pytest.mark.parametrize("norm", ["frac", "abs", "none", "leahy"])
    def test_avg_cs_cts_and_err_and_events_are_equal(self, norm, use_common_mean):
        out_ev = avg_cs_from_events(
            self.times,
            self.times2,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=use_common_mean,
            silent=False,
        )
        out_ct = avg_cs_from_events(
            self.bin_times,
            self.bin_times,
            self.gti,
            self.segment_size,
            self.dt,
            norm=norm,
            use_common_mean=use_common_mean,
            silent=False,
            fluxes1=self.counts,
            fluxes2=self.counts2,
            errors1=self.errs,
            errors2=self.errs2,
        )
        discard = [m for m in out_ev.meta.keys() if "variance" in m]
        if use_common_mean:
            compare_tables(out_ev, out_ct, rtol=0.01, discard=discard)
        else:
            compare_tables(out_ev, out_ct, rtol=0.1, discard=discard)


class TestNorms(object):
    @classmethod
    def setup_class(cls):
        cls.mean = cls.var = 100000.0
        cls.N = 800000
        cls.dt = 0.2
        cls.df = 1 / (cls.N * cls.dt)
        freq = fftfreq(cls.N, cls.dt)
        good = freq > 0
        cls.good = good
        cls.meanrate = cls.mean / cls.dt
        cls.lc = np.random.poisson(cls.mean, cls.N).astype(float)
        cls.nph = np.sum(cls.lc)
        cls.pds = (np.abs(np.fft.fft(cls.lc)) ** 2)[good]
        cls.lc_bksub = cls.lc - cls.mean
        cls.pds_bksub = (np.abs(np.fft.fft(cls.lc_bksub)) ** 2)[good]
        cls.lc_renorm = cls.lc / cls.mean
        cls.pds_renorm = (np.abs(np.fft.fft(cls.lc_renorm)) ** 2)[good]
        cls.lc_renorm_bksub = cls.lc_renorm - 1
        cls.pds_renorm_bksub = (np.abs(np.fft.fft(cls.lc_renorm_bksub)) ** 2)[good]

    def test_leahy_bksub_var_vs_standard(self):
        """Test that the Leahy norm. does not change with background-subtracted lcs"""
        leahyvar = normalize_leahy_from_variance(self.pds_bksub, np.var(self.lc_bksub), self.N)
        leahy = 2 * self.pds / np.sum(self.lc)
        ratio = np.mean(leahyvar / leahy)
        assert np.isclose(ratio, 1, rtol=0.01)

    def test_abs_bksub(self):
        """Test that the abs rms normalization does not change with background-subtracted lcs"""
        ratio = normalize_abs(self.pds_bksub, self.dt, self.N) / normalize_abs(
            self.pds, self.dt, self.N
        )
        assert np.isclose(ratio.mean(), 1, rtol=0.01)

    def test_frac_renorm_constant(self):
        """Test that the fractional rms normalization is equivalent when renormalized"""
        ratio = normalize_frac(self.pds_renorm, self.dt, self.N, 1) / normalize_frac(
            self.pds, self.dt, self.N, self.mean
        )
        assert np.isclose(ratio.mean(), 1, rtol=0.01)

    def test_frac_to_abs_ctratesq(self):
        """Test that fractional rms normalization x ctrate**2 is equivalent to abs renormalized"""
        ratio = (
            normalize_frac(self.pds, self.dt, self.N, self.mean)
            / normalize_abs(self.pds, self.dt, self.N)
            * self.meanrate**2
        )
        assert np.isclose(ratio.mean(), 1, rtol=0.01)

    def test_total_variance(self):
        """Test that the total variance of the unnormalized pds is the same as
        the variance from the light curve
        Attention: VdK defines the variance as sum (x - x0)**2.
        The usual definition is divided by 'N'
        """
        vdk_total_variance = np.sum((self.lc - self.mean) ** 2)
        ratio = np.mean(self.pds) / vdk_total_variance
        assert np.isclose(ratio.mean(), 1, rtol=0.01)

    @pytest.mark.parametrize("norm", ["abs", "frac", "leahy"])
    def test_poisson_level(self, norm):
        pdsnorm = normalize_periodograms(
            self.pds, self.dt, self.N, self.mean, n_ph=self.nph, norm=norm
        )

        assert np.isclose(
            pdsnorm.mean(), poisson_level(meanrate=self.meanrate, norm=norm), rtol=0.01
        )

    @pytest.mark.parametrize("norm", ["abs", "frac", "leahy"])
    def test_poisson_level_real(self, norm):
        pdsnorm = normalize_periodograms(
            self.pds, self.dt, self.N, self.mean, n_ph=self.nph, norm=norm, power_type="real"
        )

        assert np.isclose(
            pdsnorm.mean(), poisson_level(meanrate=self.meanrate, norm=norm), rtol=0.01
        )

    @pytest.mark.parametrize("norm", ["abs", "frac", "leahy"])
    def test_poisson_level_absolute(self, norm):
        pdsnorm = normalize_periodograms(
            self.pds, self.dt, self.N, self.mean, n_ph=self.nph, norm=norm, power_type="abs"
        )

        assert np.isclose(
            pdsnorm.mean(), poisson_level(meanrate=self.meanrate, norm=norm), rtol=0.01
        )

    def test_normalize_with_variance(self):
        pdsnorm = normalize_periodograms(
            self.pds, self.dt, self.N, self.mean, variance=self.var, norm="leahy"
        )
        assert np.isclose(pdsnorm.mean(), 2, rtol=0.01)

    def test_normalize_with_variance_fails_if_variance_zero(self):
        # If the variance is zero, it will fail:
        with pytest.raises(ValueError) as excinfo:
            pdsnorm = normalize_leahy_from_variance(self.pds, 0.0, self.N)
        assert "The variance used to normalize the" in str(excinfo.value)

    def test_normalize_none(self):
        pdsnorm = normalize_periodograms(
            self.pds, self.dt, self.N, self.mean, n_ph=self.nph, norm="none"
        )
        assert np.isclose(pdsnorm.mean(), self.pds.mean(), rtol=0.01)

    def test_normalize_badnorm(self):
        with pytest.raises(ValueError):
            pdsnorm = normalize_periodograms(
                self.pds, self.var, self.N, self.mean, n_ph=self.nph, norm="asdfjlasdjf"
            )
