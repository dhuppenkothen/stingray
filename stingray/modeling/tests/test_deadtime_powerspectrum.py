from __future__ import division, print_function
import numpy as np

from stingray.modeling import deadtime_powerspectrum


class Testdeadtime_powerspectrum(object):

    @classmethod
    def setup_class(cls):
        cls.td = 1.
        cls.tb = 0.5
        cls.N = 10
        cls.r0 = 2.0

        cls.f = np.arange(10)
        cls.fnyq = 2.0

    def test_P1(self):
        P1 = deadtime_powerspectrum.P1(td=self.td, tb=self.tb, r0=self.r0)
        assert P1 is not None

    def test_P2(self):
        P2 = deadtime_powerspectrum.P2(td = self.td, tb = self.tb, r0 = self.r0, N = self.N)
        assert P2 is not None

    def test_deadtime_powerspectrum(self):
        P1 = deadtime_powerspectrum.P1(td=self.td, tb=self.tb, r0=self.r0)
        P2 = deadtime_powerspectrum.P2(td = self.td, tb = self.tb, r0 = self.r0, N = self.N)
        model = deadtime_powerspectrum.deadtime_powerspectrum(P1, P2)
        power = model(self.f, self.fnyq)
        assert power is not None
        assert len(power) == len(self.f)