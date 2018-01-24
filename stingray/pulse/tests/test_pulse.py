from __future__ import division, print_function, absolute_import

import numpy as np
from stingray.pulse.pulsar import fold_events, get_TOA
from stingray.pulse.pulsar import stat, z_n, pulse_phase, phase_exposure
from stingray.pulse.pulsar import fold_detection_level, z2_n_detection_level
from stingray.pulse.pulsar import fold_profile_probability, z2_n_probability
from stingray.pulse.pulsar import get_orbital_correction_from_ephemeris_file
from ..pulsar import HAS_PINT
from astropy.tests.helper import remote_data
import pytest
import os

def _template_fun(phase, ph0, amplitude, baseline=0):
    return baseline + amplitude * np.cos((phase - ph0) * 2 * np.pi)


class TestAll(object):
    """Unit tests for the stingray.pulsar module."""
    @classmethod
    def setup_class(cls):
        cls.curdir = os.path.abspath(os.path.dirname(__file__))
        cls.datadir = os.path.join(cls.curdir, 'data')

    @remote_data
    @pytest.mark.skipif('not HAS_PINT')
    def test_pint_installed_correctly(self):
        import pint.toa as toa
        from pint.residuals import resids
        import pint.models.model_builder as mb
        import astropy.units as u
        parfile = os.path.join(self.datadir, 'example_pint.par')
        timfile = os.path.join(self.datadir, 'example_pint.tim')

        toas = toa.get_TOAs(timfile, ephem="DE405",
                            planets=False, include_bipm=False)
        model = mb.get_model(parfile)

        pint_resids_us = resids(toas, model, False).time_resids.to(u.s)

        # Due to the gps2utc clock correction. We are at 3e-8 seconds level.
        assert np.all(np.abs(pint_resids_us.value) < 3e-6)

    @remote_data
    @pytest.mark.skipif('not HAS_PINT')
    def test_orbit_from_parfile(self):
        import pint.toa as toa
        parfile = os.path.join(self.datadir, 'example_pint.par')
        timfile = os.path.join(self.datadir, 'example_pint.tim')

        toas = toa.get_TOAs(timfile, ephem="DE405",
                            planets=False, include_bipm=False)
        mjds = np.array([m.value for m in toas.get_mjds(high_precision=True)])

        mjdstart, mjdstop = mjds[0] - 1, mjds[-1] + 1

        correction_sec, correction_mjd, model = \
            get_orbital_correction_from_ephemeris_file(mjdstart, mjdstop,
                                                       parfile, ntimes=1000,
                                                       return_pint_model=True)

        mjdref = 50000
        toa_sec = (mjds - mjdref) * 86400
        corr = correction_mjd(mjds)
        corr_s = correction_sec (toa_sec, mjdref)
        assert np.allclose(corr, corr_s / 86400 + mjdref)

    @pytest.mark.skipif('HAS_PINT')
    def test_orbit_from_parfile_raises(self):
        print("Doesn't have pint")
        with pytest.raises(ImportError):
            get_orbital_correction_from_ephemeris_file(0, 0,
                                                       parfile='ciaociao')

    def test_stat(self):
        """Test pulse phase calculation, frequency only."""
        prof = np.array([2, 2, 2, 2])
        np.testing.assert_array_almost_equal(stat(prof), 0)

    def test_zn(self):
        """Test pulse phase calculation, frequency only."""
        ph = np.array([0, 1])
        np.testing.assert_array_almost_equal(z_n(ph), 8)
        ph = np.array([])
        np.testing.assert_array_almost_equal(z_n(ph), 0)
        ph = np.array([0.2, 0.7])
        ph2 = np.array([0, 0.5])
        np.testing.assert_array_almost_equal(z_n(ph), z_n(ph2))

    def test_fold_detection_level(self):
        """Test pulse phase calculation, frequency only."""
        np.testing.assert_almost_equal(fold_detection_level(16, 0.01),
                                       30.577914166892498)
        np.testing.assert_almost_equal(fold_detection_level(16, 0.01, ntrial=2),
                                       fold_detection_level(16, 0.01 / 2))

    def test_zn_detection_level(self):
        np.testing.assert_almost_equal(z2_n_detection_level(2),
                                       13.276704135987625)
        np.testing.assert_almost_equal(z2_n_detection_level(4, 0.01, ntrial=2),
                                       z2_n_detection_level(4, 0.01/2))

    def test_fold_probability(self):
        detlev = fold_detection_level(16, 0.1, ntrial=3)
        np.testing.assert_almost_equal(fold_profile_probability(detlev, 16,
                                                                ntrial=3),
                                       0.1)

    def test_zn_probability(self):
        detlev = z2_n_detection_level(2, 0.1, ntrial=3)
        np.testing.assert_almost_equal(z2_n_probability(detlev, 2, ntrial=3),
                                       0.1)

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
        expected[nbin//2:] = 0
        np.testing.assert_array_almost_equal(expo, expected)

    def test_phase_exposure3(self):
        start_time = 0
        stop_time = 1
        gtis = np.array([[0, 0.5]])
        period = 1
        nbin = 16
        expo = phase_exposure(start_time, stop_time, period, nbin, gtis=gtis)
        expected = np.ones(nbin)
        expected[nbin//2:] = 0
        np.testing.assert_array_almost_equal(expo, expected)

    def test_phase_exposure4(self):
        start_time = 0
        stop_time = 1
        gtis = np.array([[-0.2, 1.2]])
        period = 1
        nbin = 16
        expo = phase_exposure(start_time, stop_time, period, nbin, gtis=gtis)
        expected = np.ones(nbin)
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
        dt = 1/nbin
        times = np.arange(0, 2, dt)
        gtis = np.array([[-0.5*dt, 2 + 0.5*dt]])

        period = 1
        ph, p, pe = fold_events(times, 1, nbin=nbin, expocorr=True, gtis=gtis)

        np.testing.assert_array_almost_equal(ph, np.arange(nbin)/nbin +
                                             0.5/nbin)
        np.testing.assert_array_almost_equal(p, 2 * np.ones(nbin))
        np.testing.assert_array_almost_equal(pe, 2**0.5 * np.ones(nbin))

    def test_pulse_profile3(self):
        nbin = 16
        dt = 1/nbin
        times = np.arange(0, 2 - dt, dt)
        gtis = np.array([[-0.5*dt, 2 - dt]])

        ph, p, pe = fold_events(times, 1, nbin=nbin, expocorr=True,
                                gtis=gtis)

        np.testing.assert_array_almost_equal(ph, np.arange(nbin)/nbin +
                                             0.5/nbin)
        np.testing.assert_array_almost_equal(p, 2 * np.ones(nbin))
        expected_err = 2**0.5 * np.ones(nbin)
        expected_err[-1] = 2  # Because of the change of exposure
        np.testing.assert_array_almost_equal(pe, expected_err)

    def test_zn_2(self):
        np.testing.assert_almost_equal(z_n(np.arange(1), n=1, norm=1), 2)
        np.testing.assert_almost_equal(z_n(np.arange(1), n=2, norm=1), 4)
        np.testing.assert_almost_equal(z_n(np.arange(2), n=2, norm=1), 8)
        np.testing.assert_almost_equal(z_n(np.arange(2)+0.5, n=2, norm=1), 8)

    def test_get_TOA1(self):
        np.random.seed(1234)
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
        np.random.seed(1234)
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
