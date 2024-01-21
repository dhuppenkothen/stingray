import os
import pytest
import numpy as np
from scipy.interpolate import interp1d

from stingray.lightcurve import Lightcurve
from stingray.powerspectrum import AveragedPowerspectrum
from stingray.deadtime.model import r_det, r_in, pds_model_zhang
from stingray.deadtime.model import check_A, check_B, heaviside
from stingray.filters import filter_for_deadtime


pytestmark = pytest.mark.slow


def test_heaviside():
    assert heaviside(2) == 1
    assert heaviside(0) == 1
    assert heaviside(-1) == 0


def simulate_events(rate, length, deadtime=2.5e-3, **filter_kwargs):
    events = np.random.uniform(0, length, int(rate * length))
    events = np.sort(events)
    events_dt = filter_for_deadtime(events, deadtime, **filter_kwargs)
    return events, events_dt


def test_deadtime_conversion():
    """Test the functions for count rate conversion."""
    original_rate = np.arange(1, 1000, 10)
    deadtime = 2.5e-3
    rdet = r_det(deadtime, original_rate)
    rin = r_in(deadtime, rdet)
    np.testing.assert_almost_equal(rin, original_rate)


def test_zhang_model_accurate():
    bintime = 1 / 4096
    deadtime = 2.5e-3
    length = 2000
    fftlen = 5
    r = 300

    events, events_dt = simulate_events(r, length, deadtime=deadtime)
    lc_dt = Lightcurve.make_lightcurve(events_dt, bintime, tstart=0, tseg=length)
    pds = AveragedPowerspectrum(lc_dt, fftlen, norm="leahy")

    zh_f, zh_p = pds_model_zhang(1000, r, deadtime, bintime, limit_k=100)

    deadtime_fun = interp1d(zh_f, zh_p, bounds_error=False, fill_value="extrapolate")
    ratio = pds.power / deadtime_fun(pds.freq)
    assert np.isclose(np.mean(ratio), 1, atol=0.001)
    assert np.isclose(np.std(ratio), 1 / np.sqrt(pds.m), atol=0.001)


def test_checkA():
    check_A(300, 2.5e-3, 0.001, max_k=100, save_to="check_A.png")
    assert os.path.exists("check_A.png")
    os.unlink("check_A.png")


def test_checkB():
    check_B(300, 2.5e-3, 0.001, max_k=100, save_to="check_B.png")
    assert os.path.exists("check_B.png")
    os.unlink("check_B.png")
