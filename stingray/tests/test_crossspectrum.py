from __future__ import division
import numpy as np
import pytest
import warnings
from stingray import Lightcurve
from stingray import Crossspectrum, AveragedCrossspectrum

np.random.seed(20160528)

class TestCoherence(object):

    def test_coherence(self):
        lc1 = Lightcurve([1, 2, 3, 4, 5], [2, 3, 2, 4, 1])
        lc2 = Lightcurve([1, 2, 3, 4, 5], [4, 8, 1, 9, 11])

        cs = Crossspectrum(lc1, lc2)
        coh = cs.coherence()

        assert len(coh) == 2
        assert np.abs(np.mean(coh)) < 1


class TestCrossspectrum(object):

    def setup_class(self):
        tstart = 0.0
        tend = 1.0
        dt = 0.0001

        time = np.linspace(tstart, tend, int((tend - tstart)/dt))

        counts1 = np.random.poisson(0.01, size=time.shape[0])
        counts2 = np.random.negative_binomial(1, 0.09, size=time.shape[0])

        self.lc1 = Lightcurve(time, counts1)
        self.lc2 = Lightcurve(time, counts2)

        self.cs = Crossspectrum(self.lc1, self.lc2)

    def test_make_empty_crossspectrum(self):
        cs = Crossspectrum()
        assert cs.freq is None
        assert cs.cs is None
        assert cs.df is None
        assert cs.nphots1 is None
        assert cs.nphots2 is None
        assert cs.m == 1
        assert cs.n is None

    def test_init_with_one_lc_none(self):
        with pytest.raises(TypeError):
            cs = Crossspectrum(self.lc1)

    def test_init_with_norm_not_str(self):
        with pytest.raises(AssertionError):
            cs = Crossspectrum(norm=1)

    def test_init_with_invalid_norm(self):
        with pytest.raises(AssertionError):
            cs = Crossspectrum(norm='frabs')

    def test_init_with_wrong_lc1_instance(self):
        lc_ = Crossspectrum()
        with pytest.raises(AssertionError):
            cs = Crossspectrum(lc_, self.lc2)

    def test_init_with_wrong_lc2_instance(self):
        lc_ = Crossspectrum()
        with pytest.raises(AssertionError):
            cs = Crossspectrum(self.lc1, lc_)

    def test_make_crossspectrum_diff_lc_counts_shape(self):
        counts = np.array([1]*10001)
        time = np.linspace(0.0, 1.0001, 10001)
        lc_ = Lightcurve(time, counts)
        with pytest.raises(AssertionError):
            cs = Crossspectrum(self.lc1, lc_)

    def test_make_crossspectrum_diff_dt(self):
        counts = np.array([1]*10000)
        time = np.linspace(0.0, 2.0, 10000)
        lc_ = Lightcurve(time, counts)
        with pytest.raises(AssertionError):
            cs = Crossspectrum(self.lc1, lc_)

    def test_rebin_smaller_resolution(self):
        # Original df is between 0.9 and 1.0
        with pytest.raises(AssertionError):
            new_cs = self.cs.rebin(df=0.1)

    def test_rebin(self):
        new_cs = self.cs.rebin(df=1.5)
        assert new_cs.df == 1.5

    def test_norm_leahy(self):
        cs = Crossspectrum(self.lc1, self.lc2, norm='leahy')
        assert len(cs.cs) == 4999
        assert cs.norm == 'leahy'

    def test_norm_frac(self):
        cs = Crossspectrum(self.lc1, self.lc2, norm='frac')
        assert len(cs.cs) == 4999
        assert cs.norm == 'frac'

    def test_norm_abs(self):
        cs = Crossspectrum(self.lc1, self.lc2, norm='abs')
        assert len(cs.cs) == 4999
        assert cs.norm == 'abs'

    def test_coherence(self):
        coh = self.cs.coherence()
        assert len(coh) == 4999
        assert np.abs(coh[0]) < 1

class TestAveragedCrossspectrum(object):

    def setup_class(self):
        tstart = 0.0
        tend = 1.0
        dt = 0.0001

        time = np.linspace(tstart, tend, int((tend - tstart)/dt))

        counts1 = np.random.poisson(0.01, size=time.shape[0])
        counts2 = np.random.negative_binomial(1, 0.09, size=time.shape[0])

        self.lc1 = Lightcurve(time, counts1)
        self.lc2 = Lightcurve(time, counts2)

        self.cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1)

    def test_init_with_norm_not_str(self):
        with pytest.raises(AssertionError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1,
                                       norm=1)

    def test_init_with_invalid_norm(self):
        with pytest.raises(AssertionError):
            cs = AveragedCrossspectrum(self.lc1, self.lc2, segment_size=1,
                                       norm='frabs')

    def test_init_with_inifite_segment_size(self):
        with pytest.raises(AssertionError):
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

            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
