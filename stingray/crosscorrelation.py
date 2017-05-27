from __future__ import division
import numpy as np
from scipy import signal

from stingray import lightcurve
from stingray.exceptions import StingrayError


class CrossCorrelation(object):
    def __init__(self, lc1=None, lc2=None):

        # Create an empty object to populate data later on
        if lc1 is None and lc2 is None:
            self.corr = None
            self.time_shift = None
            self.time_lags = None
            self.dt = None
            return
        else:
            if lc1.size != lc2.size:
                raise StingrayError('Both lightcurves should be of same length')
            self.dt = lc1.dt
            self.cal_corr(lc1, lc2)
            self.cal_timeshift()


def cal_corr(self, lc1, lc2):
    if not isinstance(lc1, lightcurve.Lightcurve):
        raise TypeError("lc1 must be a lightcurve.Lightcurve object")
    if not isinstance(lc2, lightcurve.Lightcurve):
        raise TypeError("lc2 must be a lightcurve.Lightcurve object")
    self.corr = signal.correlate(lc1.counts, lc2.counts)


def cal_timeshift(self, dt=1):
    if self.dt is None:
        self.dt = dt

    n = len(self.corr)
    dur = int(n / 2)
    x_lags = np.linspace(-dur, dur, n)
    self.time_lags = x_lags * self.dt
    # time_shift is the time of max. correlation
    self.time_shift = self.time_lags[np.argmax(self.corr)]