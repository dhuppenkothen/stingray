from __future__ import division
import numpy as np
from scipy import signal

from stingray import lightcurve
from stingray.exceptions import StingrayError


class CrossCorrelation(object):
    def __init__(self, lc1=None, lc2=None):

        """
        Make a cross-correlation from a light curves.
        You can also make an empty Crosscorrelation object to populate with your
        own cross-correlation data.

        Parameters
        ----------
        lc1: lightcurve.Lightcurve object, optional, default None
            The first light curve data for correlation calculations.

        lc2: lightcurve.Lightcurve object, optional, default None
            The light curve data for the correlation calculations.
        
        Attributes
        ----------
         corr: numpy.ndarray
             An array of correlation data calculated from two lighcurves
 
         time_lags: numpy.ndarray
             An array of all possible time lags against which each point in corr is calculated 
 
         dt: float
             The time resolution of each lightcurve (used in time_lag calculations)
 
         time_shift: float
             Time lag that gives maximum value of correlation between two lightcurves.
             There will be maximum correlation between lightcurves if one of the lightcurve is shifted by time_shift.    
        """

        ## Populate all attributes by None if user passes no lightcurve data
        if lc1 is None and lc2 is None:
            self.corr = None
            self.time_shift = None
            self.time_lags = None
            self.dt = None
            return
        else:
            self._make_corr(lc1, lc2)

    def _make_corr(self, lc1, lc2):

        """
        Creates Crosscorrelation Object.
        
        Parameters
        ----------
        lc1: lightcurve.Lightcurve object
            The first light curve data.
            
        lc2: lightcurve.Lightcurve object
            The second light curve data.
            
        Returns
        ----------
        None
        """

        # Sizes of both light curves are assumed to be equal for now
        if lc1.size != lc2.size:
            raise StingrayError('Both lightcurves should be of same length')

        if not isinstance(lc1, lightcurve.Lightcurve):
            raise TypeError("lc1 must be a lightcurve.Lightcurve object")
        if not isinstance(lc2, lightcurve.Lightcurve):
            raise TypeError("lc2 must be a lightcurve.Lightcurve object")

        if lc1.dt != lc2.dt:
            raise StingrayError("Light curves do not have "
                                "same time binning dt.")
        else:
            self.dt = lc1.dt

        # Calculates cross-correlation of two lightcurves
        self.corr = signal.correlate(lc1.counts, lc2.counts)

        self.time_shift, self.time_lags = self.cal_timeshift(dt=self.dt)

    def cal_timeshift(self, dt=1.0):
        """
        Creates Crosscorrelation Object.

        Parameters
        ----------
        dt: float , optional, default 1.0
            Time resolution of lightcurve, should be passed when object is populated with correlation data
            and no information about light curve can be extracted. Used to calculate time_lags.

        Returns
        ----------
        self.time_shift: float
             Value of time lag that gives maximum value of correlation between two lightcurves. 
             
        self.time_lags: numpy.ndarray
             An array of time_lags calculated from correlation data 
        """

        if self.dt is None:
            self.dt = dt
        n = len(self.corr)
        dur = int(n / 2)
        # Correlation against all possible lags, positive as well as negative lags are stored
        x_lags = np.linspace(-dur, dur, n)
        self.time_lags = x_lags * self.dt
        # time_shift is the time lag for max. correlation
        self.time_shift = self.time_lags[np.argmax(self.corr)]
        
        return self.time_shift, self.time_lags