from __future__ import division
from stingray.simulator.base import simulate_times
from stingray.lightcurve import Lightcurve
import numpy as np


class TestSimulator(object):

    @classmethod
    def setup_class(self):
        self.time = [0.5, 1.5, 2.5, 3.5]
        self.counts_flat = [3000, 3000, 3000, 3000]
        self.gti = [[0, 4]]

    def test_simulate_times(self):
        """Simulate photon arrival times for an event list
        from light curve.
        """
        lc = Lightcurve(self.time, self.counts_flat, gti=self.gti)
        times = simulate_times(lc)
        lc_sim = Lightcurve.make_lightcurve(times, gti=lc.gti, dt=lc.dt,
                                            tstart=lc.tstart, tseg=lc.tseg)
        assert np.all((lc - lc_sim).counts < 3 * np.sqrt(lc.counts))

    def test_simulate_times_with_spline(self):
        """Simulate photon arrival times, with use_spline option
        enabled.
        """
        lc = Lightcurve(self.time, self.counts_flat, gti=self.gti)
        times = simulate_times(lc, use_spline=True)
        lc_sim = Lightcurve.make_lightcurve(times, gti=lc.gti, dt=lc.dt,
                                            tstart=lc.tstart, tseg=lc.tseg)
        assert np.all((lc - lc_sim).counts < 3 * np.sqrt(lc.counts))
