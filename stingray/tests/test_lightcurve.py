import os
import copy
import numpy as np
from astropy.tests.helper import pytest
import warnings
import os
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.time import Time
import astropy.timeseries
from astropy.timeseries import TimeSeries
from astropy.table import Table
import scipy.stats

from stingray import Lightcurve
from stingray.exceptions import StingrayError
from stingray.gti import create_gti_mask

np.random.seed(20150907)

_H5PY_INSTALLED = True
_HAS_LIGHTKURVE = True
_HAS_YAML = True
_IS_WINDOWS = os.name == "nt"
_HAS_ULTRANEST = True

try:
    import h5py
except ImportError:
    _H5PY_INSTALLED = False

try:
    import lightkurve
except ImportError:
    _HAS_LIGHTKURVE = False

try:
    import yaml
except ImportError:
    _HAS_YAML = False

try:
    import ultranest
except ImportError:
    _HAS_ULTRANEST = False

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")


def fvar_fun(lc):
    from stingray.utils import excess_variance

    return excess_variance(lc, normalization="fvar")


def nvar_fun(lc):
    from stingray.utils import excess_variance

    return excess_variance(lc, normalization="norm_xs")


def evar_fun(lc):
    from stingray.utils import excess_variance

    return excess_variance(lc, normalization="none")


class TestProperties(object):
    @classmethod
    def setup_class(cls):
        dt = 0.1
        tstart = 0
        tstop = 1
        times = np.arange(tstart, tstop, dt)
        cls.gti = np.array([[tstart - dt / 2, tstop - dt / 2]])
        # Simulate something *clearly* non-constant
        counts = np.zeros_like(times) + 100

        cls.lc = Lightcurve(times, counts, gti=cls.gti)
        cls.lc_lowmem = Lightcurve(times, counts, gti=cls.gti, low_memory=True)

    @pytest.mark.skipif("not _IS_WINDOWS")
    def test_warn_on_windows(self):
        with pytest.warns(UserWarning) as record:
            _ = Lightcurve(self.lc.time, self.lc.counts, gti=self.lc.gti)
        assert np.any(["On Windows, the size of an integer" in r.message.args[0] for r in record])

    @pytest.mark.skipif("_IS_WINDOWS")
    def test_warn_on_windows(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")
        with pytest.warns(UserWarning) as record:
            _ = Lightcurve(self.lc.time, self.lc.counts, gti=self.lc.gti)

        assert np.any(["On Windows, the size of an integer" in r.message.args[0] for r in record])

    def test_warn_wrong_keywords(self):
        lc = copy.deepcopy(self.lc)
        with pytest.warns(UserWarning) as record:
            _ = Lightcurve(lc.time, lc.counts, gti=lc.gti, bubu="settete")
        assert np.any(["Unrecognized keywords:" in r.message.args[0] for r in record])

    def test_warn_estimate_chunk_length(self):
        times = np.arange(0, 1000, 1)
        lc = Lightcurve(times, counts=np.ones(times.size) + 200, dt=1, skip_checks=True)
        with pytest.warns(DeprecationWarning) as record:
            lc.estimate_chunk_length()

        assert np.any(
            [
                "This function was renamed to estimate_segment_size" in r.message.args[0]
                for r in record
            ]
        )

    def test_time(self):
        lc = copy.deepcopy(self.lc)
        assert lc._bin_lo is None
        # When I call bin_lo, _bin_lo gets set
        _ = lc.bin_lo
        assert lc._bin_lo is not None

        # When I set time, _bin_lo gets deleted.
        lc.time = lc.time / 10
        assert lc._bin_lo is None
        _ = lc.bin_lo
        assert lc._bin_lo is not None

    def test_lightcurve_from_astropy_time(self):
        time = Time([57483, 57484], format="mjd")
        counts = np.array([2, 2])
        lc = Lightcurve(time, counts)
        assert lc.dt == 86400
        assert np.allclose(lc.counts, counts)

    def test_time_is_quantity_or_astropy_time(self):
        counts = [34, 21.425]
        times = np.array([57000, 58000])

        times_q = (times - times[0]) * u.d
        times_t = Time(times, format="mjd")

        lc_q = Lightcurve(time=times_q, counts=counts, mjdref=times[0])
        lc_t = Lightcurve(time=times_t, counts=counts)
        assert_allclose(lc_q.time, lc_t.time)

    def test_gti(self):
        lc = copy.deepcopy(self.lc)
        assert lc._mask is None
        _ = lc.mask
        assert lc._mask is not None
        lc.gti = [[0, 1]]
        assert lc._mask is None

    def test_counts_and_countrate(self):
        lc = copy.deepcopy(self.lc)
        # At initialization, _countrate is None and _counts is not.
        assert lc._countrate is None
        assert lc._counts is not None
        assert lc._meancounts is None
        # Now we retrieve meancounts; it gets calculated.
        _ = lc.meancounts
        assert lc._meancounts is not None
        # Now we retrieve countrate, and it gets calculated
        _ = lc.countrate
        assert lc._countrate is not None
        # Now I set counts; countrate gets deleted together with the other
        # statistics.
        lc.counts = np.zeros_like(lc.counts) + 3
        assert lc._countrate is None
        assert lc._meancounts is None
        assert lc._meanrate is None
        # Now I retrieve meanrate. It gets calculated
        _ = lc.meanrate
        assert lc._meanrate is not None
        # Finally, we set count rate and test that the rest has been deleted.
        lc.countrate = np.zeros_like(lc.countrate) + 3
        lc.countrate_err = np.zeros_like(lc.countrate) + 3
        assert lc._counts is None
        assert lc._counts_err is None
        assert lc._meancounts is None
        _ = lc.counts_err
        assert lc._counts_err is not None

    def test_counts_and_countrate_lowmem(self):
        lc = copy.deepcopy(self.lc_lowmem)
        # At initialization, _countrate is None and _counts is not.
        assert lc._countrate is None
        assert lc._counts is not None
        assert lc._meancounts is None
        # Now we retrieve meancounts; it gets calculated.
        _ = lc.meancounts
        assert lc._meancounts is not None
        # Now we retrieve countrate, and it gets calculated but not saved
        # (because low_memory)
        _ = lc.countrate
        assert lc._countrate is None
        _ = lc.countrate_err
        assert lc._countrate_err is None
        # Now I set counts; countrate gets deleted together with the other
        # statistics.
        lc.counts = np.zeros_like(lc.counts) + 3
        assert lc.input_counts
        assert lc._countrate is None
        assert lc._meancounts is None
        assert lc._meanrate is None
        # Now I retrieve meanrate. It gets calculated
        _ = lc.meanrate
        assert lc._meanrate is not None
        # Finally, we set count rate and test that the rest has been deleted,
        # AND input_counts is changed to False.
        lc.countrate = np.zeros_like(lc.countrate) + 3
        assert lc._counts is None
        assert lc._meancounts is None
        assert not lc.input_counts
        _ = lc.counts
        # Now we retrieve counts, and it gets calculated but not saved
        # (because low_memory, and input_counts is now False)
        assert lc._counts is None
        _ = lc.counts_err
        # Now we retrieve counts, and it gets calculated but not saved
        # (because low_memory, and input_counts is now False)
        assert lc._counts_err is None

    @pytest.mark.parametrize(
        "property", "time,counts,counts_err," "countrate,countrate_err".split(",")
    )
    def test_assign_bad_shape_fails(self, property):
        lc = copy.deepcopy(self.lc)
        # Same shape passes
        setattr(lc, property, np.zeros_like(lc.time))
        # Different shape doesn't
        with pytest.raises(ValueError):
            setattr(lc, property, 3)
        with pytest.raises(ValueError):
            setattr(lc, property, np.arange(2))


class TestChunks(object):
    @classmethod
    def setup_class(cls):
        dt = 0.1
        tstart = 0
        tstop = 100
        times = np.arange(tstart, tstop, dt)
        cls.gti = np.array([[tstart - dt / 2, tstop - dt / 2]])
        # Simulate something *clearly* non-constant
        counts = np.random.poisson(10000 + 2000 * np.sin(2 * np.pi * times))

        cls.lc = Lightcurve(times, counts, gti=cls.gti)

    def test_analyze_lc_chunks_fvar_fracstep(self):

        start, stop, res = self.lc.analyze_lc_chunks(20, fvar_fun, fraction_step=0.5)
        # excess_variance returns fvar and fvar_err
        fvar, fvar_err = res

        assert np.allclose(start[0], self.gti[0, 0])
        assert np.all(fvar > 0)
        # This must be a clear measurement of fvar
        assert np.all(fvar > fvar_err)

    def test_analyze_lc_chunks_nvar_fracstep(self):
        start, stop, res = self.lc.analyze_lc_chunks(20, fvar_fun, fraction_step=0.5)
        # excess_variance returns fvar and fvar_err
        fvar, fvar_err = res
        start, stop, res = self.lc.analyze_lc_chunks(20, nvar_fun, fraction_step=0.5)
        # excess_variance returns fvar and fvar_err
        nevar, nevar_err = res
        assert np.allclose(nevar, fvar**2, rtol=0.01)

    def test_analyze_lc_chunks_nvar_fracstep_mean(self):
        start, stop, mean = self.lc.analyze_lc_chunks(20, np.mean, fraction_step=0.5)
        start, stop, res = self.lc.analyze_lc_chunks(20, evar_fun, fraction_step=0.5)
        # excess_variance returns fvar and fvar_err
        evar, evar_err = res
        start, stop, res = self.lc.analyze_lc_chunks(20, nvar_fun, fraction_step=0.5)
        # excess_variance returns fvar and fvar_err
        nevar, nevar_err = res
        assert np.allclose(nevar * mean**2, evar, rtol=0.01)
        assert np.allclose(nevar_err * mean**2, evar_err, rtol=0.01)


class TestLightcurve(object):
    @classmethod
    def setup_class(cls):
        cls.times = np.array([1, 2, 3, 4])
        cls.counts = np.array([2, 2, 2, 2])
        cls.counts_err = np.array([0.2, 0.2, 0.2, 0.2])
        cls.bg_counts = np.array([1, 0, 0, 1])
        cls.bg_ratio = np.array([1, 1, 0.5, 1])
        cls.frac_exp = np.array([1, 1, 1, 1])
        cls.dt = 1.0
        cls.gti = np.array([[0.5, 4.5]])

    def test_create(self):
        """
        Demonstrate that we can create a trivial Lightcurve object.
        """
        lc = Lightcurve(self.times, self.counts)

    def test_irregular_time_warning(self):
        """
        Check if inputting an irregularly spaced time iterable throws out
        a warning.
        """
        times = [1, 2, 3, 5, 6]
        counts = [2, 2, 2, 2, 2]
        warn_str = (
            "SIMON says: Bin sizes in input time array aren't equal "
            "throughout! This could cause problems with Fourier "
            "transforms. Please make the input time evenly sampled."
        )

        with warnings.catch_warnings(record=True) as w:
            _ = Lightcurve(times, counts, err_dist="poisson")
            assert np.any([str(wi.message) == warn_str for wi in w])

    def test_unrecognize_err_dist_warning(self):
        """
        Check if a non-poisson error_dist throws the correct warning.
        """
        times = [1, 2, 3, 4, 5]
        counts = [2, 2, 2, 2, 2]
        warn_str = "SIMON says: Stingray only uses poisson err_dist at " "the moment"

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            lc = Lightcurve(times, counts, err_dist="gauss")
            assert np.any([warn_str in str(wi.message) for wi in w])

    def test_dummy_err_dist_fail(self):
        """
        Check if inputting an irregularly spaced time iterable throws out
        a warning.
        """
        times = [1, 2, 3, 4, 5]
        counts = [2, 2, 2, 2, 2]

        with pytest.raises(StingrayError):
            lc = Lightcurve(times, counts, err_dist="joke")

    def test_invalid_data(self):
        times = [1, 2, 3, 4, 5]
        counts = [2, 2, np.nan, 2, 2]
        counts_err = [1, 2, 3, np.nan, 2]

        with pytest.raises(ValueError):
            lc = Lightcurve(times, counts)

        with pytest.raises(ValueError):
            lc = Lightcurve(times, [2] * 5, err=counts_err)

        times[2] = np.inf

        with pytest.raises(ValueError):
            lc = Lightcurve(times, [2] * 5)

    def test_n(self):
        lc = Lightcurve(self.times, self.counts)
        assert lc.n == 4

    def test_analyze_lc_chunks(self):
        lc = Lightcurve(self.times, self.counts, gti=self.gti)

        def func(lc):
            return lc.time[0]

        start, stop, res = lc.analyze_lc_chunks(2, func)
        assert start[0] == 0.5
        assert np.allclose(start + lc.dt / 2, res)

    def test_bin_edges(self):
        bin_lo = [0.5, 1.5, 2.5, 3.5]
        bin_hi = [1.5, 2.5, 3.5, 4.5]
        lc = Lightcurve(self.times, self.counts)
        assert np.allclose(lc.bin_lo, bin_lo)
        assert np.allclose(lc.bin_hi, bin_hi)

    def test_lightcurve_from_toa(self):
        lc = Lightcurve.make_lightcurve(self.times, self.dt, use_hist=True, tstart=0.5)
        lc2 = Lightcurve.make_lightcurve(self.times, self.dt, use_hist=False, tstart=0.5)
        assert np.allclose(lc.time, lc2.time)
        assert np.allclose(lc.counts, lc2.counts)

    def test_lightcurve_from_toa_gti(self):
        lc = Lightcurve.make_lightcurve(self.times, self.dt, gti=self.gti)
        lc2 = Lightcurve.make_lightcurve(self.times, self.dt, tstart=0.5, tseg=4.0)
        assert np.allclose(lc.time, lc2.time)
        assert np.allclose(lc.counts, lc2.counts)

    def test_lightcurve_from_toa_quantity(self):
        lc = Lightcurve.make_lightcurve(self.times * u.s, self.dt, use_hist=True, tstart=0.5)
        lc2 = Lightcurve.make_lightcurve(self.times, self.dt, use_hist=False, tstart=0.5)
        assert np.allclose(lc.time, lc2.time)
        assert np.allclose(lc.counts, lc2.counts)

    def test_lightcurve_from_toa_Time(self):
        mjdref = 56789
        mjds = Time(self.times / 86400 + mjdref, format="mjd")

        lc = Lightcurve.make_lightcurve(mjds, self.dt, mjdref=mjdref, use_hist=True, tstart=0.5)
        lc2 = Lightcurve.make_lightcurve(
            self.times, self.dt, use_hist=False, tstart=0.5, mjdref=mjdref
        )
        assert np.allclose(lc.time, lc2.time)
        assert np.allclose(lc.counts, lc2.counts)

    def test_lightcurve_from_toa_halfbin(self):
        lc = Lightcurve.make_lightcurve(self.times + 0.5, self.dt, use_hist=True, tstart=0.5)
        lc2 = Lightcurve.make_lightcurve(self.times + 0.5, self.dt, use_hist=False, tstart=0.5)
        assert np.allclose(lc.time, lc2.time)
        assert np.allclose(lc.counts, lc2.counts)

    def test_lightcurve_from_toa_random_nums(self):
        times = np.random.uniform(0, 10, 1000)
        lc = Lightcurve.make_lightcurve(times, self.dt, use_hist=True, tstart=0.5)
        lc2 = Lightcurve.make_lightcurve(times, self.dt, use_hist=False, tstart=0.5)
        assert np.allclose(lc.time, lc2.time)
        assert np.allclose(lc.counts, lc2.counts)

    def test_tstart(self):
        tstart = 0.0
        lc = Lightcurve.make_lightcurve(self.times, self.dt, tstart=0.0)
        assert lc.tstart == tstart
        assert lc.time[0] == tstart + 0.5 * self.dt

    def test_tseg(self):
        tstart = 0.0
        tseg = 5.0
        lc = Lightcurve.make_lightcurve(self.times, self.dt, tseg=tseg, tstart=tstart)

        assert lc.tseg == tseg
        assert lc.time[-1] - lc.time[0] == tseg - self.dt

    def test_nondivisble_tseg(self):
        """
        If the light curve length input is not divisible by the time
        resolution, the last (fractional) time bin will be dropped.
        """
        tstart = 0.0
        tseg = 5.5
        lc = Lightcurve.make_lightcurve(self.times, self.dt, tseg=tseg, tstart=tstart)
        assert lc.tseg == int(tseg / self.dt)

    def test_correct_timeresolution(self):
        lc = Lightcurve.make_lightcurve(self.times, self.dt)
        assert np.isclose(lc.dt, self.dt)

    def test_bin_correctly(self):
        ncounts = np.array([2, 1, 0, 3])
        tstart = 0.0
        tseg = 4.0

        toa = np.hstack([np.random.uniform(i, i + 1, size=n) for i, n in enumerate(ncounts)])

        dt = 1.0
        lc = Lightcurve.make_lightcurve(toa, dt, tseg=tseg, tstart=tstart)

        assert np.allclose(lc.counts, ncounts)

    def test_countrate(self):
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0 + dt / 2, 5 - dt / 2, dt)
        counts = np.zeros_like(times) + mean_counts
        lc = Lightcurve(times, counts)
        assert np.allclose(lc.countrate, np.zeros_like(counts) + mean_counts / dt)

    def test_input_countrate(self):
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0 + dt / 2, 5 - dt / 2, dt)
        countrate = np.zeros_like(times) + mean_counts
        lc = Lightcurve(times, countrate, input_counts=False)
        assert np.allclose(lc.counts, np.zeros_like(countrate) + mean_counts * dt)

    def test_meanrate(self):
        times = [0.5, 1.0, 1.5, 2.0]
        counts = [2, 3, 3, 4]
        lc = Lightcurve(times, counts)
        assert lc.meanrate == 6

    def test_meancounts(self):
        counts = [2, 3, 3, 4]
        lc = Lightcurve(self.times, counts)
        assert lc.meancounts == 3

    def test_lc_gtis(self):
        t = [0.5, 1.5, 2.5, 3.5, 4.5]
        lc = [5, 5, 0, 5, 5]
        gtis = [[0, 2], [3, 5]]
        lc = Lightcurve(t, lc, gti=gtis)

        assert lc.meanrate == 5
        assert lc.meancounts == 5

    def test_creating_lightcurve_raises_type_error_when_input_is_none(self):
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0 + dt / 2, 5 - dt / 2, dt)
        counts = np.array([None] * times.shape[0])
        with pytest.raises(TypeError):
            lc = Lightcurve(times, counts)

    def test_creating_lightcurve_raises_type_error_when_input_is_inf(self):
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0 + dt / 2, 5 - dt / 2, dt)
        counts = np.array([np.inf] * times.shape[0])
        with pytest.raises(ValueError):
            lc = Lightcurve(times, counts)

    def test_creating_lightcurve_raises_type_error_when_input_is_nan(self):
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0 + dt / 2, 5 - dt / 2, dt)
        counts = np.array([np.nan] * times.shape[0])
        with pytest.raises(ValueError):
            lc = Lightcurve(times, counts)

    def test_init_with_diff_array_lengths(self):
        time = [1, 2, 3]
        counts = [2, 2, 2, 2]

        with pytest.raises(StingrayError):
            lc = Lightcurve(time, counts)

    def test_add_with_different_time_arrays(self):
        _times = [1.1, 2.1, 3.1, 4.1, 5.1]
        _counts = [2, 2, 2, 2, 2]

        with pytest.raises(ValueError):

            lc1 = Lightcurve(self.times, self.counts)
            lc2 = Lightcurve(_times, _counts)

            lc = lc1 + lc2

    def test_add_with_different_err_dist(self):
        lc1 = Lightcurve(self.times, self.counts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            lc2 = Lightcurve(self.times, self.counts, err=self.counts / 2, err_dist="gauss")
        with warnings.catch_warnings(record=True) as w:
            lc = lc1 + lc2
            assert np.any(["ightcurves have different statistics" in str(wi.message) for wi in w])

    def test_add_with_same_gtis(self):
        lc1 = Lightcurve(self.times, self.counts, gti=self.gti)
        lc2 = Lightcurve(self.times, self.counts, gti=self.gti)
        lc = lc1 + lc2
        np.testing.assert_almost_equal(lc.gti, self.gti)

    def test_add_with_different_gtis(self):
        gti = [[0.0, 3.5]]
        lc1 = Lightcurve(self.times, self.counts, gti=self.gti)
        lc2 = Lightcurve(self.times, self.counts, gti=gti)
        lc = lc1 + lc2
        np.testing.assert_almost_equal(lc.gti, [[0.5, 3.5]])

    def test_add_with_unequal_time_arrays(self):
        _times = [1, 3, 5, 7]

        with pytest.raises(ValueError):
            lc1 = Lightcurve(self.times, self.counts)
            lc2 = Lightcurve(_times, self.counts)

            lc = lc1 + lc2

    def test_add_with_equal_time_arrays(self):
        _counts = [1, 1, 1, 1]

        lc1 = Lightcurve(self.times, self.counts)
        lc2 = Lightcurve(self.times, _counts)

        lc = lc1 + lc2

        assert np.allclose(lc.counts, lc1.counts + lc2.counts)
        assert np.allclose(lc.countrate, lc1.countrate + lc2.countrate)
        assert lc1.mjdref == lc.mjdref

    def test_sub_with_diff_time_arrays(self):
        _times = [1.1, 2.1, 3.1, 4.1, 5.1]
        _counts = [2, 2, 2, 2, 2]

        with pytest.raises(ValueError):
            lc1 = Lightcurve(self.times, self.counts)
            lc2 = Lightcurve(_times, _counts)

            lc = lc1 - lc2

    def test_sub_with_different_err_dist(self):
        lc1 = Lightcurve(self.times, self.counts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            lc2 = Lightcurve(self.times, self.counts, err=self.counts / 2, err_dist="gauss")
        with warnings.catch_warnings(record=True) as w:
            lc = lc1 - lc2
            assert np.any(["ightcurves have different statistics" in str(wi.message) for wi in w])

    def test_subtraction(self):
        _counts = [3, 4, 5, 6]

        lc1 = Lightcurve(self.times, self.counts)
        lc2 = Lightcurve(self.times, _counts)

        lc = lc2 - lc1

        expected_counts = np.array([1, 2, 3, 4])
        assert np.allclose(lc.counts, expected_counts)
        assert lc1.mjdref == lc.mjdref

    def test_negation(self):
        lc = Lightcurve(self.times, self.counts)

        _lc = lc + (-lc)

        assert not np.all(_lc.counts)
        assert _lc.mjdref == lc.mjdref

    def test_len_function(self):
        lc = Lightcurve(self.times, self.counts)

        assert len(lc) == 4

    def test_indexing_with_unexpected_type(self):
        lc = Lightcurve(self.times, self.counts)

        with pytest.raises(IndexError):
            count = lc["first"]

    def test_indexing(self):
        lc = Lightcurve(self.times, self.counts)

        assert lc[0] == lc[1] == lc[2] == lc[3] == 2

    def test_slicing(self):
        lc = Lightcurve(
            self.times,
            self.counts,
            dt=self.dt,
            gti=self.gti,
            err=self.counts / 10,
            err_dist="gauss",
        )
        assert np.allclose(lc[1:3].counts, np.array([2, 2]))
        assert np.allclose(lc[:2].counts, np.array([2, 2]))
        assert np.allclose(lc[:2].gti, [[0.5, 2.5]])
        assert np.allclose(lc[2:].counts, np.array([2, 2]))
        assert np.allclose(lc[2:].gti, [[2.5, 4.5]])
        assert np.allclose(lc[:].counts, np.array([2, 2, 2, 2]))
        assert np.allclose(lc[::2].gti, [[0.5, 1.5], [2.5, 3.5]])
        assert np.allclose(lc[:].gti, lc.gti)
        assert lc[:].mjdref == lc.mjdref
        assert lc[::2].n == 2
        assert np.allclose(lc[1:3].counts_err, np.array([2, 2]) / 10)
        assert np.allclose(lc[:2].counts_err, np.array([2, 2]) / 10)
        assert np.allclose(lc[2:].counts_err, np.array([2, 2]) / 10)
        assert np.allclose(lc[:].counts_err, np.array([2, 2, 2, 2]) / 10)
        assert lc[:].err_dist == lc.err_dist

    def test_slicing_index_error(self):
        lc = Lightcurve(self.times, self.counts)

        with pytest.raises(StingrayError):
            lc_new = lc[1:2]

    def test_index(self):
        lc = Lightcurve(self.times, self.counts)

        index = 1
        index_np32, index_np64 = np.int32(index), np.int64(index)
        assert lc[index] == lc[index_np32] == lc[index_np64]

    def test_join_with_different_dt(self):
        _times = [5, 5.5, 6]
        _counts = [2, 2, 2]

        lc1 = Lightcurve(self.times, self.counts)
        lc2 = Lightcurve(_times, _counts)

        with warnings.catch_warnings(record=True) as w:
            lc1.join(lc2)
            assert np.any(["different bin widths" in str(wi.message) for wi in w])

    def test_join_with_different_mjdref(self):
        shift = 86400.0  # day
        lc1 = Lightcurve(self.times + shift, self.counts, gti=self.gti + shift, mjdref=57000)
        lc2 = Lightcurve(self.times, self.counts, gti=self.gti, mjdref=57001)

        with warnings.catch_warnings(record=True) as w:
            newlc = lc1.join(lc2)
            # The join operation *averages* the overlapping arrays
            assert np.allclose(newlc.counts, lc1.counts)
            assert np.any(
                ["MJDref is different in the two light curves" in str(wi.message) for wi in w]
            )
            assert np.any(
                ["The two light curves have overlapping time ranges" in str(wi.message) for wi in w]
            )

    def test_sum_with_different_mjdref(self):
        shift = 86400.0  # day
        lc1 = Lightcurve(self.times + shift, self.counts, gti=self.gti + shift, mjdref=57000)
        lc2 = Lightcurve(self.times, self.counts, gti=self.gti, mjdref=57001)
        with pytest.warns(UserWarning) as record:
            newlc = lc1 + lc2
        assert np.any(["MJDref" in r.message.args[0] for r in record])

        assert np.allclose(newlc.counts, lc1.counts * 2)

    def test_subtract_with_different_mjdref(self):
        shift = 86400.0  # day
        lc1 = Lightcurve(self.times + shift, self.counts, gti=self.gti + shift, mjdref=57000)
        lc2 = Lightcurve(self.times, self.counts, gti=self.gti, mjdref=57001)
        with pytest.warns(UserWarning) as record:
            newlc = lc1 - lc2
        assert np.any(["MJDref" in r.message.args[0] for r in record])

        assert np.allclose(newlc.counts, 0)

    def test_join_disjoint_time_arrays(self):
        _times = [5, 6, 7, 8]
        _counts = [2, 2, 2, 2]

        lc1 = Lightcurve(self.times, self.counts)
        lc2 = Lightcurve(_times, _counts)

        lc = lc1.join(lc2)

        assert len(lc.counts) == len(lc.time) == 8
        assert np.allclose(lc.counts, 2)
        assert lc.mjdref == lc1.mjdref

    def test_join_overlapping_time_arrays(self):
        _times = [3, 4, 5, 6]
        _counts = [4, 4, 4, 4]

        lc1 = Lightcurve(self.times, self.counts)
        lc2 = Lightcurve(_times, _counts)

        with warnings.catch_warnings(record=True) as w:
            lc = lc1.join(lc2)
            assert np.any(["overlapping time ranges" in str(wi.message) for wi in w])

        assert len(lc.counts) == len(lc.time) == 6
        assert np.allclose(lc.counts, np.array([2, 2, 3, 3, 4, 4]))

    def test_join_different_err_dist_disjoint_times(self):
        _times = [5, 6, 7, 8]
        _counts = [2, 2, 2, 2]

        lc1 = Lightcurve(self.times, self.counts, err_dist="poisson")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            lc2 = Lightcurve(_times, _counts, err_dist="gauss")

        lc3 = lc1.join(lc2)

        assert np.allclose(lc3.counts_err[: len(self.times)], lc1.counts_err)
        assert np.allclose(lc3.counts_err[len(self.times) :], np.zeros_like(lc2.counts))

    def test_join_different_err_dist_overlapping_times(self):
        _times = [3, 4, 5, 6]
        _counts = [4, 4, 4, 4]

        lc1 = Lightcurve(self.times, self.counts, err_dist="poisson")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            lc2 = Lightcurve(_times, _counts, err_dist="gauss")

        with warnings.catch_warnings(record=True) as w:
            lc3 = lc1.join(lc2)
            assert "We are setting the errors to zero." in str(w[1].message)
            assert np.allclose(lc3.counts_err, np.zeros_like(lc3.time))

    def test_truncate_by_index(self):
        lc = Lightcurve(
            self.times,
            self.counts,
            err=self.counts_err,
            gti=self.gti,
            dt=self.dt,
            bg_counts=self.bg_counts,
            frac_exp=self.frac_exp,
            bg_ratio=self.bg_ratio,
        )

        lc1 = lc.truncate(start=1)
        assert np.allclose(lc1.time, np.array([2, 3, 4]))
        assert np.allclose(lc1.counts, np.array([2, 2, 2]))
        assert np.allclose(lc1.bg_counts, np.array([0, 0, 1]))
        assert np.allclose(lc1.bg_ratio, np.array([1, 0.5, 1]))
        assert np.allclose(lc1.frac_exp, np.array([1, 1, 1]))
        np.testing.assert_almost_equal(lc1.gti[0][0], 1.5)
        assert lc1.mjdref == lc.mjdref

        lc2 = lc.truncate(stop=2)
        assert np.allclose(lc2.time, np.array([1, 2]))
        assert np.allclose(lc2.counts, np.array([2, 2]))
        assert np.allclose(lc2.bg_counts, np.array([1, 0]))
        assert np.allclose(lc2.bg_ratio, np.array([1, 1]))
        assert np.allclose(lc2.frac_exp, np.array([1, 1]))
        np.testing.assert_almost_equal(lc2.gti[-1][-1], 2.5)
        assert lc2.mjdref == lc.mjdref

    def test_truncate_by_time_stop_less_than_start(self):
        lc = Lightcurve(self.times, self.counts)

        with pytest.raises(ValueError):
            lc1 = lc.truncate(start=2, stop=1, method="time")

    def test_truncate_fails_with_incorrect_method(self):
        lc = Lightcurve(self.times, self.counts)
        with pytest.raises(ValueError):
            lc1 = lc.truncate(start=1, method="wrong")

    def test_truncate_by_time(self):
        lc = Lightcurve(self.times, self.counts, gti=self.gti)

        lc1 = lc.truncate(start=1, method="time")
        assert np.allclose(lc1.time, np.array([1, 2, 3, 4]))
        assert np.allclose(lc1.counts, np.array([2, 2, 2, 2]))
        np.testing.assert_almost_equal(lc1.gti[0][0], 0.5)
        assert lc1.mjdref == lc.mjdref

        lc2 = lc.truncate(stop=3, method="time")
        assert np.allclose(lc2.time, np.array([1, 2]))
        assert np.allclose(lc2.counts, np.array([2, 2]))
        np.testing.assert_almost_equal(lc2.gti[-1][-1], 2.5)
        assert lc2.mjdref == lc.mjdref

    def test_split_with_two_segments(self):
        test_time = np.array([1, 2, 3, 6, 7, 8])
        test_counts = np.random.rand(len(test_time))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            lc_test = Lightcurve(test_time, test_counts)
        slc = lc_test.split(1.5)

        assert len(slc) == 2

    def test_split_has_correct_data_points(self):
        test_time = np.array([1, 2, 3, 6, 7, 8])
        test_counts = np.random.rand(len(test_time))
        test_bg_counts = np.random.rand(len(test_time))
        test_bg_ratio = np.random.rand(len(test_time))
        test_frac_exp = np.random.rand(len(test_time))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            lc_test = Lightcurve(
                test_time,
                test_counts,
                bg_counts=test_bg_counts,
                bg_ratio=test_bg_ratio,
                frac_exp=test_frac_exp,
            )
        slc = lc_test.split(1.5)

        assert np.allclose(slc[0].time, [1, 2, 3])
        assert np.allclose(slc[1].time, [6, 7, 8])
        assert np.allclose(slc[0].counts, test_counts[:3])
        assert np.allclose(slc[1].counts, test_counts[3:])
        assert np.allclose(slc[0].bg_counts, test_bg_counts[:3])
        assert np.allclose(slc[1].bg_counts, test_bg_counts[3:])
        assert np.allclose(slc[0].bg_ratio, test_bg_ratio[:3])
        assert np.allclose(slc[1].bg_ratio, test_bg_ratio[3:])
        assert np.allclose(slc[0].frac_exp, test_frac_exp[:3])
        assert np.allclose(slc[1].frac_exp, test_frac_exp[3:])

    def test_split_with_three_segments(self):
        test_time = np.array([1, 2, 3, 6, 7, 8, 10, 11, 12])
        test_counts = np.random.rand(len(test_time))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            lc_test = Lightcurve(test_time, test_counts)
        slc = lc_test.split(1.5)

        assert len(slc) == 3

    def test_threeway_split_has_correct_data_points(self):
        test_time = np.array([1, 2, 3, 6, 7, 8, 10, 11, 12])
        test_counts = np.random.rand(len(test_time))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            lc_test = Lightcurve(test_time, test_counts)
        slc = lc_test.split(1.5)

        assert np.allclose(slc[0].time, [1, 2, 3])
        assert np.allclose(slc[1].time, [6, 7, 8])
        assert np.allclose(slc[2].time, [10, 11, 12])
        assert np.allclose(slc[0].counts, test_counts[:3])
        assert np.allclose(slc[1].counts, test_counts[3:6])
        assert np.allclose(slc[2].counts, test_counts[6:])

    def test_split_with_gtis(self):
        test_time = np.array([1, 2, 3, 6, 7, 8, 10, 11, 12])
        test_counts = np.random.rand(len(test_time))
        gti = [[0, 4], [9, 13]]
        lc_test = Lightcurve(test_time, test_counts, gti=gti)
        slc = lc_test.split(1.5)

        assert np.allclose(slc[0].time, [1, 2, 3])
        assert np.allclose(slc[1].time, [10, 11, 12])
        assert np.allclose(slc[0].counts, test_counts[:3])
        assert np.allclose(slc[1].counts, test_counts[6:])

    def test_consecutive_gaps(self):
        test_time = np.array([1, 2, 3, 6, 9, 10, 11])
        test_counts = np.random.rand(len(test_time))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            lc_test = Lightcurve(test_time, test_counts)
        slc = lc_test.split(1.5)

        assert np.allclose(slc[0].time, [1, 2, 3])
        assert np.allclose(slc[1].time, [9, 10, 11])
        assert np.allclose(slc[0].counts, test_counts[:3])
        assert np.allclose(slc[1].counts, test_counts[4:])

    def test_sort(self):

        _times = [2, 1, 3, 4]
        _counts = [40, 10, 20, 5]
        _counts_err = [4, 1, 2, 0.5]
        _frac_exp = [1, 0.8, 0.6, 0.9]
        _bg_counts = [1, 3, 2, 1]
        _bg_ratio = [0.5, 1, 1, 0.4]

        lc = Lightcurve(
            _times,
            _counts,
            err=_counts_err,
            frac_exp=_frac_exp,
            mjdref=57000,
            bg_counts=_bg_counts,
            bg_ratio=_bg_ratio,
        )
        mjdref = lc.mjdref

        lc_new = lc.sort()

        assert np.allclose(lc_new.counts_err, np.array([1, 4, 2, 0.5]))
        assert np.allclose(lc_new.counts, np.array([10, 40, 20, 5]))
        assert np.allclose(lc_new.time, np.array([1, 2, 3, 4]))
        assert np.allclose(lc_new.frac_exp, np.array([0.8, 1, 0.6, 0.9]))
        assert np.allclose(lc_new.bg_counts, np.array([3, 1, 2, 1]))
        assert np.allclose(lc_new.bg_ratio, np.array([1, 0.5, 1, 0.4]))
        assert lc_new.mjdref == mjdref

        lc_new = lc.sort(reverse=True)

        assert np.allclose(lc_new.counts, np.array([5, 20, 40, 10]))
        assert np.allclose(lc_new.time, np.array([4, 3, 2, 1]))
        assert np.allclose(lc_new.frac_exp, np.array([0.9, 0.6, 1, 0.8]))
        assert np.allclose(lc_new.bg_counts, np.array([1, 2, 1, 3]))
        assert np.allclose(lc_new.bg_ratio, np.array([0.4, 1, 0.5, 1]))
        assert lc_new.mjdref == mjdref

    def test_sort_counts(self):
        _times = [1, 2, 3, 4]
        _counts = [40, 10, 20, 5]
        _frac_exp = [1, 0.8, 0.6, 0.9]
        _bg_counts = [1, 3, 2, 1]
        _bg_ratio = [0.5, 1, 1, 0.4]

        lc = Lightcurve(
            _times,
            _counts,
            mjdref=57000,
            frac_exp=_frac_exp,
            bg_counts=_bg_counts,
            bg_ratio=_bg_ratio,
        )
        mjdref = lc.mjdref

        lc_new = lc.sort_counts()

        assert np.allclose(lc_new.counts, np.array([5, 10, 20, 40]))
        assert np.allclose(lc_new.time, np.array([4, 2, 3, 1]))
        assert np.allclose(lc_new.frac_exp, np.array([0.9, 0.8, 0.6, 1]))
        assert np.allclose(lc_new.bg_counts, np.array([1, 3, 2, 1]))
        assert np.allclose(lc_new.bg_ratio, np.array([0.4, 1, 1, 0.5]))
        assert lc_new.mjdref == mjdref

        lc_new = lc.sort_counts(reverse=True)

        assert np.allclose(lc_new.counts, np.array([40, 20, 10, 5]))
        assert np.allclose(lc_new.time, np.array([1, 3, 2, 4]))
        assert np.allclose(lc_new.frac_exp, np.array([1, 0.6, 0.8, 0.9]))
        assert np.allclose(lc_new.bg_counts, np.array([1, 2, 3, 1]))
        assert np.allclose(lc_new.bg_ratio, np.array([0.5, 1, 1, 0.4]))
        assert lc_new.mjdref == mjdref

    def test_sort_reverse(self):
        times = np.arange(1000)
        counts = np.random.rand(1000) * 100
        lc = Lightcurve(times, counts)
        lc_1 = lc
        lc_2 = Lightcurve(np.arange(1000, 2000), np.random.rand(1000) * 1000)
        lc_long = lc_1.join(lc_2)  # Or vice-versa
        new_lc_long = lc_long[:]  # Copying into a new object
        assert new_lc_long.n == lc_long.n

    @pytest.mark.skipif("not _HAS_LIGHTKURVE")
    def test_to_lightkurve(self):
        time, counts, counts_err = np.arange(3), np.ones(3), np.zeros(3)
        lc = Lightcurve(time, counts, counts_err)
        lk = lc.to_lightkurve()
        out_time = Time(lc.time / 86400 + lc.mjdref, format="mjd", scale="utc")
        assert_allclose(lk.time.value, out_time.value)
        assert_allclose(lk.flux, counts)
        assert_allclose(lk.flux_err, counts_err)

    @pytest.mark.skipif("not _HAS_LIGHTKURVE", reason="Lightkurve not installed")
    def test_from_lightkurve(self):
        from lightkurve import LightCurve

        time, flux, flux_err = np.arange(3), np.ones(3), np.zeros(3)
        mjdref = 56000
        time = Time(time / 86400 + mjdref, format="mjd", scale="utc")

        # LightCurve wants a time object
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        sr = Lightcurve.from_lightkurve(lc)

        out_time = Time(sr.time / 86400 + sr.mjdref, format="mjd", scale="utc")

        assert_allclose(out_time.value, lc.time.value)
        assert_allclose(sr.counts, lc.flux)
        assert_allclose(sr.counts_err, lc.flux_err)

    def test_plot_simple(self):
        lc = Lightcurve(self.times, self.counts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            lc.plot()
        assert plt.fignum_exists(1)

    def test_plot_wrong_label_type(self):
        lc = Lightcurve(self.times, self.counts)

        with pytest.raises(TypeError):
            with warnings.catch_warnings(record=True) as w:
                lc.plot(labels=123)
                assert np.any(["must be either a list or tuple" in str(wi.message) for wi in w])

    def test_plot_labels_index_error(self):
        lc = Lightcurve(self.times, self.counts)
        with warnings.catch_warnings(record=True) as w:
            lc.plot(labels=("x"))

            assert np.any(["must have two labels" in str(wi.message) for wi in w])

    def test_plot_default_filename(self):
        lc = Lightcurve(self.times, self.counts)
        lc.plot(save=True)
        assert os.path.isfile("out.png")
        os.unlink("out.png")

    def test_plot_custom_filename(self):
        lc = Lightcurve(self.times, self.counts)
        lc.plot(save=True, filename="lc.png")
        assert os.path.isfile("lc.png")
        os.unlink("lc.png")

    def test_plot_axis(self):
        lc = Lightcurve(self.times, self.counts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            lc.plot(axis=[0, 1, 0, 100])
        assert plt.fignum_exists(1)

    def test_plot_title(self):
        lc = Lightcurve(self.times, self.counts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            lc.plot(title="Test Lightcurve")
        assert plt.fignum_exists(1)

    def test_read_from_lcurve_1(self):
        fname = "lcurveA.fits"
        with pytest.warns(UserWarning):
            lc = Lightcurve.read(os.path.join(datadir, fname), fmt="hea", skip_checks=True)
        ctrate = 1
        assert np.isclose(lc.countrate[0], ctrate)

    def test_read_from_lcurve_2(self):
        fname = "lcurve_new.fits"
        with pytest.warns(UserWarning):
            lc = Lightcurve.read(os.path.join(datadir, fname), fmt="hea", skip_checks=True)
        ctrate = 0.91

        assert np.isclose(lc.countrate[0], ctrate)
        assert np.isclose(lc.mjdref, 55197.00076601852)

    def test_io_warns(self):
        lc = Lightcurve(self.times, self.counts)
        with pytest.warns(DeprecationWarning):
            lc.write("lc.pickle", format_="pickle")
        with pytest.warns(DeprecationWarning):
            lc.read("lc.pickle", format_="pickle")

    @pytest.mark.skipif("not _HAS_YAML")
    def test_io_with_ascii(self):
        lc = Lightcurve(self.times, self.counts)
        lc.write("ascii_lc.ecsv", fmt="ascii")
        lc = lc.read("ascii_lc.ecsv", fmt="ascii")
        assert np.allclose(lc.time, self.times)
        assert np.allclose(lc.counts, self.counts)
        os.remove("ascii_lc.ecsv")

    def test_io_with_pickle(self):
        lc = Lightcurve(self.times, self.counts)
        lc.write("lc.pickle", fmt="pickle")
        lc.read("lc.pickle", fmt="pickle")
        assert np.allclose(lc.time, self.times)
        assert np.allclose(lc.counts, self.counts)
        assert np.allclose(lc.gti, self.gti)
        os.remove("lc.pickle")

    @pytest.mark.skipif("not _H5PY_INSTALLED")
    def test_io_with_hdf5(self):
        lc = Lightcurve(self.times, self.counts)
        lc.write("lc.hdf5", fmt="hdf5")

        data = lc.read("lc.hdf5", fmt="hdf5")
        assert np.allclose(data.time, self.times)
        assert np.allclose(data.counts, self.counts)
        assert np.allclose(data.gti, self.gti)
        os.remove("lc.hdf5")

    def test_split_lc_by_gtis(self):
        times = [1, 2, 3, 4, 5, 6, 7, 8]
        counts = [1, 1, 1, 1, 2, 3, 3, 2]
        bg_counts = [0, 0, 0, 1, 0, 1, 2, 0]
        bg_ratio = [0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1]
        frac_exp = [1, 0.5, 1, 1, 1, 0.5, 0.5, 1]
        gti = [[0.5, 4.5], [5.5, 7.5]]

        lc = Lightcurve(
            times, counts, gti=gti, bg_counts=bg_counts, bg_ratio=bg_ratio, frac_exp=frac_exp
        )
        list_of_lcs = lc.split_by_gti()
        lc0 = list_of_lcs[0]
        lc1 = list_of_lcs[1]
        assert np.allclose(lc0.time, [1, 2, 3, 4])
        assert np.allclose(lc1.time, [6, 7])
        assert np.allclose(lc0.counts, [1, 1, 1, 1])
        assert np.allclose(lc1.counts, [3, 3])
        assert np.allclose(lc0.gti, [[0.5, 4.5]])
        assert np.allclose(lc1.gti, [[5.5, 7.5]])
        # Check if new attributes are also splited accordingly
        assert np.allclose(lc0.bg_counts, [0, 0, 0, 1])
        assert np.allclose(lc1.bg_counts, [1, 2])
        assert np.allclose(lc0.bg_ratio, [0.1, 0.1, 0.1, 0.2])
        assert np.allclose(lc1.bg_ratio, [0.2, 0.2])
        assert np.allclose(lc0.frac_exp, [1, 0.5, 1, 1])
        assert np.allclose(lc1.frac_exp, [0.5, 0.5])

    def test_split_lc_by_gtis_minpoints(self):
        times = [1, 2, 3, 4, 5, 6, 7, 8]
        counts = [1, 1, 1, 1, 2, 3, 3, 2]
        gti = [[0.5, 3.5], [3.5, 5.5], [5.5, 8.5]]
        min_points = 3

        lc = Lightcurve(times, counts, gti=gti)
        list_of_lcs = lc.split_by_gti(min_points=min_points)
        lc0 = list_of_lcs[0]
        lc1 = list_of_lcs[1]
        assert np.allclose(lc0.time, [1, 2, 3])
        assert np.allclose(lc1.time, [6, 7, 8])
        assert np.allclose(lc0.counts, [1, 1, 1])
        assert np.allclose(lc1.counts, [3, 3, 2])

    def test_shift(self):
        times = [1, 2, 3, 4, 5, 6, 7, 8]
        counts = [1, 1, 1, 1, 2, 3, 3, 2]
        lc = Lightcurve(times, counts, input_counts=True)
        lc2 = lc.shift(1)
        assert np.allclose(lc2.time - 1, times)
        lc2 = lc.shift(-1)
        assert np.allclose(lc2.time + 1, times)
        assert np.allclose(lc2.counts, lc.counts)
        assert np.allclose(lc2.countrate, lc.countrate)
        lc = Lightcurve(times, counts, input_counts=False)
        lc2 = lc.shift(1)
        assert np.allclose(lc2.counts, lc.counts)
        assert np.allclose(lc2.countrate, lc.countrate)

    def test_table_roundtrip(self):
        """Test that io methods raise Key Error when
        wrong format is provided.
        """
        N = len(self.times)
        lc = Lightcurve(
            self.times,
            self.counts,
            err=self.counts_err,
            mission="BUBU",
            instr="BABA",
            mjdref=53467.0,
        )

        ts = lc.to_astropy_table()
        new_lc = lc.from_astropy_table(ts)
        for attr in ["time", "gti", "counts"]:
            assert np.allclose(getattr(lc, attr), getattr(new_lc, attr))
        for attr in ["mission", "instr", "mjdref"]:
            assert getattr(lc, attr) == getattr(new_lc, attr)

    def test_table_roundtrip_ctrate(self):
        """Test that io methods raise Key Error when
        wrong format is provided.
        """
        N = len(self.times)
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0 + dt / 2, 5 - dt / 2, dt)
        countrate = np.zeros_like(times) + mean_counts
        err = np.zeros_like(times) + mean_counts / 2

        lc = Lightcurve(
            times,
            countrate,
            err=err,
            mission="BUBU",
            instr="BABA",
            mjdref=53467.0,
            input_counts=False,
        )

        ts = lc.to_astropy_table()
        new_lc = Lightcurve.from_astropy_table(ts)
        for attr in ["time", "gti", "countrate"]:
            assert np.allclose(getattr(lc, attr), getattr(new_lc, attr))
        assert np.allclose(new_lc.counts, lc.countrate * lc.dt)
        for attr in ["mission", "instr", "mjdref"]:
            assert getattr(lc, attr) == getattr(new_lc, attr)

    def test_timeseries_roundtrip(self):
        """Test that io methods raise Key Error when
        wrong format is provided.
        """
        N = len(self.times)
        lc = Lightcurve(self.times, self.counts, mission="BUBU", instr="BABA", mjdref=53467.0)

        ts = lc.to_astropy_timeseries()
        new_lc = lc.from_astropy_timeseries(ts)
        for attr in ["time", "gti", "counts"]:
            assert np.allclose(getattr(lc, attr), getattr(new_lc, attr))
        for attr in ["mission", "instr", "mjdref"]:
            assert getattr(lc, attr) == getattr(new_lc, attr)

    def test_timeseries_roundtrip_ctrate(self):
        """Test that io methods raise Key Error when
        wrong format is provided.
        """
        N = len(self.times)
        dt = 0.5
        mean_counts = 2.0
        times = np.arange(0 + dt / 2, 5 - dt / 2, dt)
        countrate = np.zeros_like(times) + mean_counts

        lc = Lightcurve(
            times, countrate, mission="BUBU", instr="BABA", mjdref=53467.0, input_counts=False
        )

        ts = lc.to_astropy_timeseries()
        new_lc = lc.from_astropy_timeseries(ts)
        for attr in ["time", "gti", "countrate"]:
            assert np.allclose(getattr(lc, attr), getattr(new_lc, attr))
        assert np.allclose(new_lc.counts, lc.countrate * lc.dt)
        for attr in ["mission", "instr", "mjdref"]:
            assert getattr(lc, attr) == getattr(new_lc, attr)

    def test_from_timeseries_bad(self):
        from astropy.time import TimeDelta

        times = TimeDelta(np.arange(10) * u.s)
        ts = TimeSeries(time=times)
        with pytest.raises(ValueError) as excinfo:
            Lightcurve.from_astropy_timeseries(ts)
        assert "Input timeseries must contain at least" in str(excinfo.value)


class TestLightcurveRebin(object):
    @classmethod
    def setup_class(cls):
        dt = 0.0001220703125
        n = 1384132
        mean_counts = 2.0
        times = np.arange(dt / 2, dt / 2 + n * dt, dt)
        counts = np.zeros_like(times) + mean_counts
        cls.lc = Lightcurve(times, counts)

    def test_rebin_even(self):
        dt_new = 2.0
        lc_binned = self.lc.rebin(dt_new)
        assert np.isclose(lc_binned.dt, dt_new)
        counts_test = np.zeros_like(lc_binned.time) + self.lc.counts[0] * dt_new / self.lc.dt
        assert np.allclose(lc_binned.counts, counts_test)

    def test_rebin_even_factor(self):
        f = 200
        dt_new = f * self.lc.dt
        lc_binned = self.lc.rebin(f=f)
        assert np.isclose(dt_new, f * self.lc.dt)
        counts_test = np.zeros_like(lc_binned.time) + self.lc.counts[0] * dt_new / self.lc.dt
        assert np.allclose(lc_binned.counts, counts_test)

    def test_rebin_odd(self):
        dt_new = 1.5
        lc_binned = self.lc.rebin(dt_new)
        assert np.isclose(lc_binned.dt, dt_new)

        counts_test = np.zeros_like(lc_binned.time) + self.lc.counts[0] * dt_new / self.lc.dt
        assert np.allclose(lc_binned.counts, counts_test)

    def test_rebin_odd_factor(self):
        f = 100.5
        dt_new = f * self.lc.dt
        lc_binned = self.lc.rebin(f=f)
        assert np.isclose(dt_new, f * self.lc.dt)
        counts_test = np.zeros_like(lc_binned.time) + self.lc.counts[0] * dt_new / self.lc.dt
        assert np.allclose(lc_binned.counts, counts_test)

    def rebin_several(self, dt):
        lc_binned = self.lc.rebin(dt)
        assert len(lc_binned.time) == len(lc_binned.counts)

    def test_rebin_equal_numbers(self):
        dt_all = [2, 3, np.pi, 5]
        for dt in dt_all:
            self.rebin_several(dt)

    def test_rebin_with_gtis(self):
        times = np.arange(0, 100, 0.1)

        counts = np.random.normal(100, 0.1, size=times.shape[0])
        gti = [[0, 40], [60, 100]]

        good = create_gti_mask(times, gti)

        counts[np.logical_not(good)] = 0
        lc = Lightcurve(times, counts, gti=gti, skip_checks=True, dt=0.1)
        lc.apply_gtis()

        lc_rebin = lc.rebin(1.0)

        assert (lc_rebin.time[39] - lc_rebin.time[38]) > 1.0

    def test_lc_baseline(self):
        times = np.arange(0, 100, 0.01)
        counts = np.random.normal(100, 0.1, len(times)) + 0.001 * times
        gti = [[-0.005, 50.005], [59.005, 100.005]]
        good = create_gti_mask(times, gti)
        counts[np.logical_not(good)] = 0
        lc = Lightcurve(times, counts, gti=gti)
        baseline = lc.baseline(10000, 0.01)
        assert np.all(lc.counts - baseline < 1)

    def test_lc_baseline_offset(self):
        times = np.arange(0, 100, 0.01)
        input_stdev = 0.1
        counts = np.random.normal(100, input_stdev, len(times)) + 0.001 * times
        gti = [[-0.005, 50.005], [59.005, 100.005]]
        good = create_gti_mask(times, gti)
        counts[np.logical_not(good)] = 0
        lc = Lightcurve(times, counts, gti=gti)
        baseline = lc.baseline(10000, 0.01, offset_correction=True)
        assert np.isclose(np.std(lc.counts - baseline), input_stdev, rtol=0.1)

    def test_lc_baseline_offset_fewbins(self):
        times = np.arange(0, 4, 1)
        input_stdev = 0.1
        counts = np.random.normal(100, input_stdev, len(times)) + 0.001 * times
        gti = [[-0.005, 4.005]]
        lc = Lightcurve(times, counts, gti=gti)
        with pytest.warns(UserWarning) as record:
            lc.baseline(10000, 0.01, offset_correction=True)

        assert np.any(
            [
                "Too few bins to perform baseline offset correction" in r.message.args[0]
                for r in record
            ]
        )

    def test_change_mjdref(self):
        lc_new = self.lc.change_mjdref(57000)
        assert lc_new.mjdref == 57000

    def testapply_gtis(self):
        time = np.arange(150)
        count = np.zeros_like(time) + 3
        lc = Lightcurve(time, count, gti=[[-0.5, 150.5]])
        lc.gti = [[-0.5, 2.5], [12.5, 14.5]]
        lc.apply_gtis()
        assert lc.n == 5
        assert np.allclose(lc.time, np.array([0, 1, 2, 13, 14]))
        lc.gti = [[-0.5, 10.5]]
        lc.apply_gtis()
        assert np.allclose(lc.time, np.array([0, 1, 2]))

    def test_eq_operator(self):
        time = [1, 2, 3]
        count1 = [100, 200, 300]
        count2 = [100, 200, 300]
        lc1 = Lightcurve(time, count1)
        lc2 = Lightcurve(time, count2)
        assert lc1 == lc2

    def test_eq_bad_lc(self):
        time = [1, 2, 3]
        count1 = [100, 200, 300]
        count2 = [100, 200, 300]
        lc1 = Lightcurve(time, count1)
        with pytest.raises(ValueError):
            lc1 == count2

    def test_eq_different_times(self):
        time1 = [1, 2, 3]
        time2 = [2, 3, 4]
        count1 = [100, 200, 300]
        count2 = [100, 200, 300]
        lc1 = Lightcurve(time1, count1)
        lc2 = Lightcurve(time2, count2)
        assert not lc1 == lc2

    def test_eq_different_counts(self):
        time = [1, 2, 3, 4]
        count1 = [5, 10, 15, 20]
        count2 = [2, 4, 5, 8]
        lc1 = Lightcurve(time, count1)
        lc2 = Lightcurve(time, count2)
        assert not lc1 == lc2


class TestBexvar(object):
    @classmethod
    def setup_class(cls):

        fname_data = os.path.join(datadir, "LightCurve_bexvar.fits")
        lightcurve = Table.read(fname_data, hdu="RATE", format="fits")
        band = 0

        cls.time = lightcurve["TIME"] - lightcurve["TIME"][0]
        cls.time_delta = lightcurve["TIMEDEL"]
        cls.bg_counts = lightcurve["BACK_COUNTS"][:, band]
        cls.src_counts = lightcurve["COUNTS"][:, band]
        cls.bg_ratio = lightcurve["BACKRATIO"]
        cls.frac_exp = lightcurve["FRACEXP"][:, band]

        cls.fname_result = os.path.join(datadir, "bexvar_results_band_0.npy")
        cls.quantile = scipy.stats.norm().cdf([-1])

    @pytest.mark.skipif("not _HAS_ULTRANEST")
    def test_bexvar(self):

        # create lightcurve
        lc = Lightcurve(
            time=self.time,
            counts=self.src_counts,
            bg_counts=self.bg_counts,
            bg_ratio=self.bg_ratio,
            frac_exp=self.frac_exp,
        )

        log_cr_sigma_from_method = lc.bexvar()
        log_cr_sigma_result = np.load(self.fname_result, allow_pickle=True)[1]

        scatt_lo_function = scipy.stats.mstats.mquantiles(log_cr_sigma_from_method, self.quantile)
        scatt_lo_result = scipy.stats.mstats.mquantiles(log_cr_sigma_result, self.quantile)

        # Compares lower 1 sigma quantile of the estimated scatter of the log(count rate) in dex
        assert np.isclose(scatt_lo_function, scatt_lo_result, rtol=0.1)

    @pytest.mark.skipif("not _HAS_ULTRANEST")
    def test_bexvar_with_dt_as_array(self):

        # create lightcurve with ``dt`` as an array
        lc = Lightcurve(
            time=self.time,
            counts=self.src_counts,
            dt=self.time_delta,
            gti=[[self.time[0], self.time[-1]]],
            bg_counts=self.bg_counts,
            bg_ratio=self.bg_ratio,
            frac_exp=self.frac_exp,
        )

        # provide time intervals externally to find bexvar
        log_cr_sigma_from_method = lc.bexvar()
        log_cr_sigma_result = np.load(self.fname_result, allow_pickle=True)[1]

        scatt_lo_function = scipy.stats.mstats.mquantiles(log_cr_sigma_from_method, self.quantile)
        scatt_lo_result = scipy.stats.mstats.mquantiles(log_cr_sigma_result, self.quantile)

        # Compares lower 1 sigma quantile of the estimated scatter of the log(count rate) in dex
        assert np.isclose(scatt_lo_function, scatt_lo_result, rtol=0.1)


class TestArraydt(object):
    @classmethod
    def setup_class(cls):

        cls.times = np.array([1, 3, 4, 7])
        # setup dt as an array
        cls.dt = np.array([1, 2, 1, 3])
        cls.counts = np.array([2, 2, 2, 2])
        cls.counts_err = np.array([0.2, 0.2, 0.2, 0.2])
        cls.bg_counts = np.array([1, 0, 0, 1])
        cls.bg_ratio = np.array([1, 1, 0.5, 1])
        cls.frac_exp = np.array([1, 1, 1, 1])
        cls.gti = np.array([[0.5, 7.5]])

    def test_create_with_dt_as_array(self):
        """
        Demonstrate that we can create a Lightcurve object with dt being an array of floats.
        """
        times = np.array([1, 2, 3, 4])
        counts = np.array([2, 2, 2, 2])
        counts_err = np.array([0.2, 0.2, 0.2, 0.2])
        dt = np.array([1.0, 1.0, 1.0, 1.0])
        bg_counts = np.array([1, 0, 0, 1])
        bg_ratio = np.array([1, 1, 0.5, 1])
        frac_exp = np.array([1, 1, 1, 1])
        gti = np.array([[0.5, 4.5]])
        lc = Lightcurve(
            time=times,
            counts=counts,
            dt=dt,
            err=counts_err,
            gti=gti,
            bg_counts=bg_counts,
            bg_ratio=bg_ratio,
            frac_exp=frac_exp,
        )

        # demonstrate that we can create a Lightcurve object with dt being an array of floats
        # and without explicitly providing gtis.

        lc = Lightcurve(
            time=times,
            counts=counts,
            dt=dt,
            err=counts_err,
            bg_counts=bg_counts,
            bg_ratio=bg_ratio,
            frac_exp=frac_exp,
        )

    def test_warning_when_dt_is_array(self):

        with pytest.warns(UserWarning) as record:

            _ = Lightcurve(time=self.times, counts=self.counts, dt=self.dt)

        assert any(
            [
                "Some functionalities of Stingray Lightcurve will not work when `dt` is Iterable"
                in r.message.args[0]
                for r in record
            ]
        )

    def test_truncate_by_index_when_dt_is_array(self):
        """
        Checks if `truncate_by_index()` works if `dt` is an array.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            lc = Lightcurve(
                self.times,
                self.counts,
                err=self.counts_err,
                gti=self.gti,
                dt=self.dt,
                bg_counts=self.bg_counts,
                frac_exp=self.frac_exp,
                bg_ratio=self.bg_ratio,
            )

        lc1 = lc.truncate(start=1)
        assert np.allclose(lc1.time, np.array([3, 4, 7]))
        assert np.allclose(lc1.counts, np.array([2, 2, 2]))
        assert np.allclose(lc1.dt, np.array([2, 1, 3]))
        assert np.allclose(lc1.bg_counts, np.array([0, 0, 1]))
        assert np.allclose(lc1.bg_ratio, np.array([1, 0.5, 1]))
        assert np.allclose(lc1.frac_exp, np.array([1, 1, 1]))
        np.testing.assert_almost_equal(lc1.gti[0][0], 2.5)
        assert lc1.mjdref == lc.mjdref

        lc2 = lc.truncate(stop=2)
        assert np.allclose(lc2.time, np.array([1, 3]))
        assert np.allclose(lc2.counts, np.array([2, 2]))
        assert np.allclose(lc2.dt, np.array([1, 2]))
        assert np.allclose(lc2.bg_counts, np.array([1, 0]))
        assert np.allclose(lc2.bg_ratio, np.array([1, 1]))
        assert np.allclose(lc2.frac_exp, np.array([1, 1]))
        np.testing.assert_almost_equal(lc2.gti[-1][-1], 4.5)
        assert lc2.mjdref == lc.mjdref

    def test_split_has_correct_data_points_when_dt_is_array(self):
        """
        Checks if `split()` works when `dt` is an array.
        """

        test_time = np.array([1, 2, 3, 6, 7, 8])
        test_dt = np.array([1, 1, 1, 3, 1, 1])
        test_counts = np.random.rand(len(test_time))
        test_bg_counts = np.random.rand(len(test_time))
        test_bg_ratio = np.random.rand(len(test_time))
        test_frac_exp = np.random.rand(len(test_time))
        test_gti = np.array([[0.5, 8.5]])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            lc_test = Lightcurve(
                test_time,
                test_counts,
                bg_counts=test_bg_counts,
                dt=test_dt,
                bg_ratio=test_bg_ratio,
                frac_exp=test_frac_exp,
                gti=test_gti,
            )
        slc = lc_test.split(1.5)

        assert np.allclose(slc[0].time, [1, 2, 3])
        assert np.allclose(slc[1].time, [6, 7, 8])
        assert np.allclose(slc[0].dt, [1, 1, 1])
        assert np.allclose(slc[1].dt, [3, 1, 1])
        assert np.allclose(slc[0].counts, test_counts[:3])
        assert np.allclose(slc[1].counts, test_counts[3:])
        assert np.allclose(slc[0].bg_counts, test_bg_counts[:3])
        assert np.allclose(slc[1].bg_counts, test_bg_counts[3:])
        assert np.allclose(slc[0].bg_ratio, test_bg_ratio[:3])
        assert np.allclose(slc[1].bg_ratio, test_bg_ratio[3:])
        assert np.allclose(slc[0].frac_exp, test_frac_exp[:3])
        assert np.allclose(slc[1].frac_exp, test_frac_exp[3:])

    def test_split_lc_by_gtis_when_dt_is_array(self):
        """
        Checks if `split_by_gti()` works when `dt` is an array.
        """

        times = np.array([1, 2, 3, 5, 7, 8, 10, 11])
        dt = np.array([1, 1, 1, 2, 2, 1, 2, 1])
        counts = np.array([1, 1, 1, 1, 2, 3, 3, 2])
        bg_counts = np.array([0, 0, 0, 1, 0, 1, 2, 0])
        bg_ratio = np.array([0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1])
        frac_exp = np.array([1, 0.5, 1, 1, 1, 0.5, 0.5, 1])
        gti = np.array([[0.5, 5.5], [7.5, 10.5]])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            lc = Lightcurve(
                times,
                counts,
                gti=gti,
                dt=dt,
                bg_counts=bg_counts,
                bg_ratio=bg_ratio,
                frac_exp=frac_exp,
            )

        list_of_lcs = lc.split_by_gti()
        lc0 = list_of_lcs[0]
        lc1 = list_of_lcs[1]
        assert np.allclose(lc0.time, [1, 2, 3, 5])
        assert np.allclose(lc1.time, [8, 10])
        assert np.allclose(lc0.counts, [1, 1, 1, 1])
        assert np.allclose(lc1.counts, [3, 3])
        assert np.allclose(lc0.gti, [[0.5, 5.5]])
        assert np.allclose(lc1.gti, [[7.5, 10.5]])
        assert np.allclose(lc0.bg_counts, [0, 0, 0, 1])
        assert np.allclose(lc1.bg_counts, [1, 2])
        assert np.allclose(lc0.bg_ratio, [0.1, 0.1, 0.1, 0.2])
        assert np.allclose(lc1.bg_ratio, [0.2, 0.2])
        assert np.allclose(lc0.frac_exp, [1, 0.5, 1, 1])
        assert np.allclose(lc1.frac_exp, [0.5, 0.5])
        # Check if `dt` is also splited accordingly
        assert np.allclose(lc0.dt, [1, 1, 1, 2])
        assert np.allclose(lc1.dt, [1, 2])

    def test_sort_when_dt_is_array(self):
        """
        Checks if `sort()` works when `dt` is an array.
        """

        _times = np.array([4, 1, 3, 7])
        _dt = np.array([1, 1, 2, 3])
        _counts = np.array([40, 10, 20, 5])
        _counts_err = np.array([4, 1, 2, 0.5])
        _frac_exp = np.array([1, 0.8, 0.6, 0.9])
        _bg_counts = np.array([1, 3, 2, 1])
        _bg_ratio = np.array([0.5, 1, 1, 0.4])
        _gti = np.array([[0.5, 7.5]])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            lc = Lightcurve(
                _times,
                _counts,
                dt=_dt,
                err=_counts_err,
                frac_exp=_frac_exp,
                mjdref=57000,
                bg_counts=_bg_counts,
                bg_ratio=_bg_ratio,
                gti=_gti,
            )
        mjdref = lc.mjdref

        lc_new = lc.sort()

        assert np.allclose(lc_new.counts_err, np.array([1, 2, 4, 0.5]))
        assert np.allclose(lc_new.counts, np.array([10, 20, 40, 5]))
        assert np.allclose(lc_new.time, np.array([1, 3, 4, 7]))
        assert np.allclose(lc_new.dt, np.array([1, 2, 1, 3]))
        assert np.allclose(lc_new.frac_exp, np.array([0.8, 0.6, 1, 0.9]))
        assert np.allclose(lc_new.bg_counts, np.array([3, 2, 1, 1]))
        assert np.allclose(lc_new.bg_ratio, np.array([1, 1, 0.5, 0.4]))
        assert lc_new.mjdref == mjdref

        lc_new = lc.sort(reverse=True)

        assert np.allclose(lc_new.counts, np.array([5, 40, 20, 10]))
        assert np.allclose(lc_new.time, np.array([7, 4, 3, 1]))
        assert np.allclose(lc_new.dt, np.array([3, 1, 2, 1]))
        assert np.allclose(lc_new.frac_exp, np.array([0.9, 1, 0.6, 0.8]))
        assert np.allclose(lc_new.bg_counts, np.array([1, 1, 2, 3]))
        assert np.allclose(lc_new.bg_ratio, np.array([0.4, 0.5, 1, 1]))
        assert lc_new.mjdref == mjdref

    def test_sort_count_when_dt_is_array(self):
        """
        Checks if `sort_counts()` works when `dt` is an array.
        """
        _times = np.array([1, 2, 4, 7])
        _dt = np.array([1, 1, 2, 3])
        _counts = np.array([40, 10, 20, 5])
        _counts_err = np.array([4, 1, 2, 0.5])
        _frac_exp = np.array([1, 0.8, 0.6, 0.9])
        _bg_counts = np.array([1, 3, 2, 1])
        _bg_ratio = np.array([0.5, 1, 1, 0.4])
        _gti = np.array([[0.5, 7.5]])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            lc = Lightcurve(
                _times,
                _counts,
                dt=_dt,
                err=_counts_err,
                frac_exp=_frac_exp,
                mjdref=57000,
                bg_counts=_bg_counts,
                bg_ratio=_bg_ratio,
                gti=_gti,
            )
        mjdref = lc.mjdref

        lc_new = lc.sort_counts()

        assert np.allclose(lc_new.counts_err, np.array([0.5, 1, 2, 4]))
        assert np.allclose(lc_new.counts, np.array([5, 10, 20, 40]))
        assert np.allclose(lc_new.time, np.array([7, 2, 4, 1]))
        assert np.allclose(lc_new.dt, np.array([3, 1, 2, 1]))
        assert np.allclose(lc_new.frac_exp, np.array([0.9, 0.8, 0.6, 1]))
        assert np.allclose(lc_new.bg_counts, np.array([1, 3, 2, 1]))
        assert np.allclose(lc_new.bg_ratio, np.array([0.4, 1, 1, 0.5]))
        assert lc_new.mjdref == mjdref

        lc_new = lc.sort_counts(reverse=True)

        assert np.allclose(lc_new.counts, np.array([40, 20, 10, 5]))
        assert np.allclose(lc_new.time, np.array([1, 4, 2, 7]))
        assert np.allclose(lc_new.dt, np.array([1, 2, 1, 3]))
        assert np.allclose(lc_new.frac_exp, np.array([1, 0.6, 0.8, 0.9]))
        assert np.allclose(lc_new.bg_counts, np.array([1, 2, 3, 1]))
        assert np.allclose(lc_new.bg_ratio, np.array([0.5, 1, 1, 0.4]))
        assert lc_new.mjdref == mjdref
