from __future__ import division
import numpy as np
from scipy import signal

from stingray import lightcurve
from stingray.exceptions import StingrayError
import stingray.utils as utils


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
                 
         n: int
             Number of points in self.corr(Length of cross-correlation data) 
        """

        ## Populate all attributes by None if user passes no lightcurve data
        if lc1 is None and lc2 is None:
            self.corr = None
            self.time_shift = None
            self.time_lags = None
            self.dt = None
            self.n = None
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
        self.n = len(self.corr)

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
        self.n = len(self.corr)
        dur = int(self.n / 2)
        # Correlation against all possible lags, positive as well as negative lags are stored
        x_lags = np.linspace(-dur, dur, self.n)
        self.time_lags = x_lags * self.dt
        # time_shift is the time lag for max. correlation
        self.time_shift = self.time_lags[np.argmax(self.corr)]

        return self.time_shift, self.time_lags, self.n

    def plot(self, labels=None, axis=None, title=None, marker='-', save=False, filename=None):
        """
        Plot the :class:`Crosscorrelation` as function using Matplotlib.

        Plot the Crosscorrelation object on a graph ``self.time_lags`` on x-axis and
        ``self.corr`` on y-axis

        Parameters
        ----------
        labels : iterable, default None
            A list of tuple with xlabel and ylabel as strings.
        axis : list, tuple, string, default None
            Parameter to set axis properties of Matplotlib figure. For example
            it can be a list like ``[xmin, xmax, ymin, ymax]`` or any other
            acceptable argument for `matplotlib.pyplot.axis()` function.
        title : str, default None
            The title of the plot.
        marker : str, default '-'
            Line style and color of the plot. Line styles and colors are
            combined in a single format string, as in ``'bo'`` for blue
            circles. See `matplotlib.pyplot.plot` for more options.
        save : boolean, optional (default=False)
            If True, save the figure with specified filename.
        filename : str
            File name of the image to save. Depends on the boolean ``save``.
        """

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib required for plot()")

        fig = plt.figure()

        fig = plt.plot(self.time_lags, self.corr, marker)

        if labels is not None:
            try:
                plt.xlabel(labels[0])
                plt.ylabel(labels[1])
            except TypeError:
                utils.simon("``labels`` must be either a list or tuple with "
                            "x and y labels.")
                raise
            except IndexError:
                utils.simon("``labels`` must have two labels for x and y "
                            "axes.")
                # Not raising here because in case of len(labels)==1, only
                # x-axis will be labelled.

        if axis is not None:
            plt.axis(axis)

        if title is not None:
            plt.title(title)

        if save:
            if filename is None:
                plt.savefig('corr.png')
            else:
                plt.savefig(filename)
