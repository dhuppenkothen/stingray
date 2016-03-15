"""
Definition of :class:`Lightcurve`.

:class:`Lightcurve` is used to create light curves out of photon counting data
or to save existing light curves in a class that's easy to use.
"""
import numpy as np
import stingray.utils as utils
import logging

__all__ = ["Lightcurve"]


class Lightcurve(object):
    def __init__(self, time, counts, err=False, input_counts=True):
        """
        Make a light curve object from an array of time stamps and an
        array of counts.

        Parameters
        ----------
        time: iterable
            A list or array of time stamps for a light curve

        counts: iterable, optional, default None
            A list or array of the counts in each bin corresponding to the
            bins defined in `time` (note: use input_counts=False to input the
            count rate, i.e. counts/second, otherwise use counts/bin).

        err: iterable, optional, default False
            A list or array of the uncertainties (or standard deviation) in
            each bin corresponding to the bins defined in `time`. In units of
            counts/bin or counts/second (see `input_counts`).
            If False (default), it assumes the data is Poisson distributed.

        input_counts: bool, optional, default True
            If True, the code assumes that the input data in 'counts'
            is in units of counts/bin. If False, it assumes the data
            in 'counts' is in counts/second.

        Attributes
        ----------
        time: numpy.ndarray
            The array of midpoints of time bins

        counts: numpy.ndarray
            The counts per bin corresponding to the bins in `time`.

        counts_err: numpy.ndarray
            The uncertainties of the `counts` array

        countrate: numpy.ndarray
            The counts per second in each of the bins defined in `time`.

        countrate_err: numpy.ndarray
            The uncertainties of the `countrate` array

        ncounts: int
            The number of data points in the light curve.

        dt: float
            The time resolution of the light curve.

        tseg: float
            The total duration of the light curve.

        tstart: float
            The start time of the light curve.

        """
        assert np.all(np.isfinite(time)), "There are inf or NaN values in " \
                                          "your time array!"
        assert np.all(np.isfinite(counts)), "There are inf or NaN values in " \
                                            "your counts array!"
        if err is not False:
            assert np.all(np.isfinite(err)), "There are inf or NaN " \
                                             "values in your err array!"
        else:
            err = np.sqrt(np.asarray(counts))

        self.time = np.asarray(time)
        self.dt = time[1] - time[0]

        if input_counts:
            self.counts = np.asarray(counts)
            self.countrate = self.counts / self.dt
            self.counts_err = np.asarray(err)
            self.countrate_err = np.asarray(err) / self.dt
        else:
            self.countrate = np.asarray(counts)
            self.counts = self.countrate * self.dt
            self.counts_err = np.asarray(err) * self.dt
            self.countrate_err = np.asarray(err)

        self.ncounts = self.counts.shape[0]
        self.tseg = self.time[-1] - self.time[0] + self.dt
        self.tstart = self.time[0] - 0.5*self.dt

    @staticmethod
    def make_lightcurve(toa, dt, tseg=None, tstart=None):
        """
        Make a light curve out of photon arrival times.

        Parameters
        ----------
        toa: iterable
            list of photon arrival times

        dt: float
            time resolution of the light curve (the bin width)

        tseg: float, optional, default None
            The total duration of the light curve.
            If this is `None`, then the total duration of the light curve will
            be the interval between the arrival between the first and the last
            photon in `toa`.

                **Note**: If tseg is not divisible by dt (i.e. if tseg/dt is
                not an integer number), then the last fractional bin will be
                dropped!

        tstart: float, optional, default None
            The start time of the light curve.
            If this is None, the arrival time of the first photon will be used
            as the start time of the light curve.

        Returns
        -------
        lc: :class:`Lightcurve` object
            A light curve object with the binned light curve
        """
        # tstart is an optional parameter to set a starting time for
        # the light curve in case this does not coincide with the first photon
        if tstart is None:
            # if tstart is not set, assume light curve starts with first photon
            tstart = toa[0]

        # compute the number of bins in the light curve
        # for cases where tseg/dt are not integer, computer one
        # last time bin more that we have to subtract in the end
        if tseg is None:
            tseg = toa[-1] - toa[0]

        logging.info("make_lightcurve: tseg: " + str(tseg))
        timebin = np.int(tseg/dt)

        logging.info("make_lightcurve: timebin:  " + str(timebin))
        tend = tstart + timebin*dt

        counts, histbins = np.histogram(toa, bins=timebin,
                                        range=[tstart, tend])
        dt = histbins[1] - histbins[0]
        time = histbins[:-1] + 0.5*dt
        counts = np.asarray(counts)

        return Lightcurve(time, counts)

    def rebin_lightcurve(self, dt_new, method='sum'):
        """
        Rebin the light curve to a new time resolution. While the new
        resolution need not be an integer multiple of the previous time
        resolution, be aware that if it is not, the last bin will be cut
        off by the fraction left over by the integer division.

        Parameters
        ----------
        dt_new: float
            The new time resolution of the light curve. Must be larger than
            the time resolution of the old light curve!

        method: {"sum" | "mean" | "average"}, optional, default "sum"
            This keyword argument sets whether the counts in the new bins
            should be summed or averaged.


        Returns
        -------
        lc_new: :class:`Lightcurve` object
            The :class:`Lightcurve` object with the new, binned light curve.
        """
        assert dt_new >= self.dt, "New time resolution must be larger than " \
                                  "old time resolution!"

        bin_time, bin_counts, bin_err, _ = utils.rebin_data(self.time,
                                                            self.counts,
                                                            self.counts_err,
                                                            dt_new, method)

        lc_new = Lightcurve(bin_time, bin_counts, err=bin_err)
        return lc_new
