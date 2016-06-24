from __future__ import division, print_function, absolute_import
from ..pulsar import *

def _template_fun(phase, ph0, amplitude, baseline=0):
    return baseline + amplitude * np.cos((phase - ph0) * 2 * np.pi)


class TestAll(object):

    """Unit tests for the stingray.pulsar module."""

    def test_pulse_phase1(self):
        """Test pulse phase calculation, frequency only."""
        times = np.arange(0, 4, 0.5)
        ph = pulse_phase(times, 1, ph0=0, to_1=False)
        np.testing.assert_array_almost_equal(ph, times)

    def test_pulse_phase2(self):
        """Test pulse phase calculation, fdot only."""
        times = np.arange(0, 4, 0.5)
        ph = pulse_phase(times, 0, 1, ph0=0, to_1=False)
        np.testing.assert_array_almost_equal(ph, 0.5 * times ** 2)

    def test_pulse_phase3(self):
        """Test pulse phase calculation, fddot only."""
        times = np.arange(0, 4, 0.5)
        ph = pulse_phase(times, 0, 0, 1, ph0=0, to_1=False)
        np.testing.assert_array_almost_equal(ph, 1/6 * times ** 3)

    def test_phase_exposure1(self):
        start_time = 0
        stop_time = 1
        period = 1
        nbin = 16
        expo = phase_exposure(start_time, stop_time, period, nbin)
        np.testing.assert_array_almost_equal(expo, np.ones(nbin))

    def test_phase_exposure2(self):
        start_time = 0
        stop_time = 0.5
        period = 1
        nbin = 16
        expo = phase_exposure(start_time, stop_time, period, nbin)
        expected = np.ones(nbin)
        expected[nbin/2:] = 0
        np.testing.assert_array_almost_equal(expo, expected)

    def test_phase_exposure3(self):
        start_time = 0
        stop_time = 1
        gtis = np.array([[0, 0.5]])
        period = 1
        nbin = 16
        expo = phase_exposure(start_time, stop_time, period, nbin, gtis=gtis)
        expected = np.ones(nbin)
        expected[nbin/2:] = 0
        np.testing.assert_array_almost_equal(expo, expected)

    def test_pulse_profile1(self):
        nbin = 16
        times = np.arange(0, 1, 1/nbin)

        period = 1
        ph, p, pe = fold_events(times, 1, nbin=nbin)

        np.testing.assert_array_almost_equal(p, np.ones(nbin))
        np.testing.assert_array_almost_equal(ph, np.arange(nbin)/nbin +
                                             0.5/nbin)
        np.testing.assert_array_almost_equal(pe, np.ones(nbin))

    def test_pulse_profile2(self):
        nbin = 16
        times = np.arange(0, 1, 1/nbin)

        period = 1
        ph, p, pe = fold_events(times, 1, nbin=nbin, expocorr=True)

        np.testing.assert_array_almost_equal(ph, np.arange(nbin)/nbin +
                                             0.5/nbin)
        np.testing.assert_array_almost_equal(p, np.ones(nbin))
        np.testing.assert_array_almost_equal(pe, np.ones(nbin))

    def test_zn(self):
        np.testing.assert_almost_equal(z_n(np.arange(1), n=1, norm=1), 2)
        np.testing.assert_almost_equal(z_n(np.arange(1), n=2, norm=1), 4)
        np.testing.assert_almost_equal(z_n(np.arange(2), n=2, norm=1), 8)
        np.testing.assert_almost_equal(z_n(np.arange(2)+0.5, n=2, norm=1), 8)

    def test_get_TOA1(self):
        period = 1.2
        tstart = 122
        start_phase = 0.2123
        phases = np.arange(0, 1, 1 / 32)
        template = _template_fun(phases, start_phase, 10, 20)
        prof = np.random.poisson(template)

        toa, toaerr = \
            get_TOA(prof, period, tstart,
                    template=_template_fun(phases, 0, 1, 0))

        real_toa = tstart + start_phase * period
        assert (real_toa >= toa - toaerr * 3) & (real_toa <= toa + toaerr * 3)

    def test_get_TOA2(self):
        period = 1.2
        tstart = 122
        start_phase = 0.2123
        phases = np.arange(0, 1, 1 / 32)
        template = _template_fun(phases, start_phase, 10, 20)
        prof = np.random.poisson(template)

        toa, toaerr = \
            get_TOA(prof, period, tstart,
                    template=_template_fun(phases, 0, 1, 0), nstep=200)

        real_toa = tstart + start_phase * period
        assert (real_toa >= toa - toaerr * 3) & (real_toa <= toa + toaerr * 3)
