import numpy as np
from stingray.events import EventList
from stingray.varenergyspectrum import VarEnergySpectrum, RmsEnergySpectrum

from astropy.tests.helper import pytest
np.random.seed(20150907)


class TestPowerspectrum(object):

    @classmethod
    def setup_class(cls):
        tstart = 0.0
        tend = 100.0
        nphot = 1000
        alltimes = np.random.uniform(tstart, tend, nphot)
        alltimes.sort()
        cls.events = EventList(alltimes,
                               pha=np.random.uniform(0.3, 12, nphot),
                               gti = [[tstart, tend]])
        cls.vespec = VarEnergySpectrum(cls.events, [0., 10000],
                                       [0.5, 5, 10], [0.3, 10],
                                       bin_time=0.1)
        cls.vespeclog = \
            VarEnergySpectrum(cls.events, [0., 10000],
                              [0.5, 5, 10], [0.3, 10], log_distr=True)

    def test_intervals_overlapping(self):
        ref_int = self.vespec._decide_ref_intervals([0.5, 6], [0.3, 10])
        np.testing.assert_allclose(ref_int, [[0.3, 0.5], [6, 10]])
        ref_int = self.vespec._decide_ref_intervals([0.5, 11], [0.3, 10])
        np.testing.assert_allclose(ref_int, [[0.3, 0.5]])

    def test_intervals_non_overlapping(self):
        ref_int = self.vespec._decide_ref_intervals([6, 11], [0.3, 5])
        np.testing.assert_allclose(ref_int, [[0.3, 5]])

    def test_construct_lightcurves(self):
        events = EventList([0.09, 0.21, 0.23, 0.32, 0.4, 0.54],
                           pha=[0,0,0,0,1,1],
                           gti=[[0, 0.65]])
        vespec = VarEnergySpectrum(events, [0., 10000],
                                   [0, 1, 2], [0.5, 1.1],
                                   bin_time=0.1)
        base_lc, ref_lc = \
            vespec._construct_lightcurves([0, 0.5],
                                          tstart=0, tstop=0.65)
        np.testing.assert_allclose(base_lc.counts, [1, 0, 2, 1, 0, 0])
        np.testing.assert_allclose(ref_lc.counts, [0, 0, 0, 1, 0, 1])

    def test_construct_lightcurves_no_exclude(self):
        events = EventList([0.09, 0.21, 0.23, 0.32, 0.4, 0.54],
                           pha=[0,0,0,0,1,1],
                           gti=[[0, 0.65]])

        vespec = VarEnergySpectrum(events, [0., 10000],
                                   [0, 1, 2], [0, 0.5],
                                   bin_time=0.1)
        base_lc, ref_lc = \
            vespec._construct_lightcurves([0, 0.5],
                                          tstart=0, tstop=0.65,
                                          exclude=False)
        np.testing.assert_equal(base_lc.counts, ref_lc.counts)

    def test_construct_lightcurves_pi(self):
        events = EventList([0.09, 0.21, 0.23, 0.32, 0.4, 0.54],
                           pi=np.asarray([0, 0, 0, 0, 1, 1]),
                           gti=[[0, 0.65]])
        vespec = VarEnergySpectrum(events, [0., 10000],
                                   [0, 1, 2], [0.5, 1.1], use_pi=True,
                                   bin_time=0.1)
        base_lc, ref_lc = \
            vespec._construct_lightcurves([0, 0.5],
                                          tstart=0, tstop=0.65)
        np.testing.assert_allclose(base_lc.counts, [1, 0, 2, 1, 0, 0])
        np.testing.assert_allclose(ref_lc.counts, [0, 0, 0, 1, 0, 1])

    def test_rmsspectrum(self):
        from ..simulator.simulator import Simulator
        simulator = Simulator(0.1, 10000, rms=0.4, mean=200)
        test_lc = simulator.simulate(1)
        test_ev1, test_ev2 = EventList(), EventList()
        test_ev1.simulate_times(test_lc)
        test_ev2.simulate_times(test_lc)
        test_ev1.pha = np.random.uniform(0.3, 12, len(test_ev1.time))
        test_ev2.pha = np.random.uniform(0.3, 12, len(test_ev2.time))

        rms = RmsEnergySpectrum(test_ev1, [0., 100],
                                [0.3, 12, 5], [0.3, 12],
                                bin_time=0.01,
                                segment_size=100,
                                events2=test_ev2)

        # Assert that the rms measured at all energies is the same
        assert np.all(
            np.abs(rms.spectrum - rms.spectrum[0]) < rms.spectrum_error)

        # Assert that it is close to 0.4 (since we don't have infinite spectral
        # coverage, it will be a little less!)
        assert np.allclose(rms.spectrum, 0.37, 0.05)
