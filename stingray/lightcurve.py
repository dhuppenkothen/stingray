"""
Definition of :class:`Lightcurve`.

:class:`Lightcurve` is used to create light curves out of photon counting data
or to save existing light curves in a class that's easy to use.
"""
import logging
import numpy as np
import stingray.io as io
import stingray.utils as utils
from stingray.exceptions import StingrayError
from stingray.utils import simon, assign_value_if_none, baseline_als
from stingray.gti import cross_two_gtis, join_gtis, gti_border_bins
from stingray.gti import check_gtis, create_gti_mask
from astropy.stats import poisson_conf_interval

__all__ = ["Lightcurve"]

valid_statistics = ["poisson", "gauss", None]


class Lightcurve(object):
    def __init__(self, time, counts, err=None, input_counts=True,
                 gti=None, err_dist='poisson', mjdref=0, dt=None):
        """
        Make a light curve object from an array of time stamps and an
        array of counts.

        Parameters
        ----------
        time: iterable
            A list or array of time stamps for a light curve

        counts: iterable, optional, default None
            A list or array of the counts in each bin corresponding to the
            bins defined in `time` (note: use `input_counts=False` to
            input the count range, i.e. counts/second, otherwise use
            counts/bin).

        err: iterable, optional, default None:
            A list or array of the uncertainties in each bin corresponding to
            the bins defined in `time` (note: use `input_counts=False` to
            input the count rage, i.e. counts/second, otherwise use
            counts/bin). If None, we assume the data is poisson distributed
            and calculate the error from the average of the lower and upper 
            1-sigma confidence intervals for the Poissonian distribution with 
            mean equal to `counts`.

        input_counts: bool, optional, default True
            If True, the code assumes that the input data in 'counts'
            is in units of counts/bin. If False, it assumes the data
            in 'counts' is in counts/second.

        gti: 2-d float array, default None
            [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            Good Time Intervals. They are *not* applied to the data by default.
            They will be used by other methods to have an indication of the
            "safe" time intervals to use during analysis.

        err_dist: str, optional, default=None
            Statistic of the Lightcurve, it is used to calculate the
            uncertainties and other statistical values apropriately.
            Default makes no assumptions and keep errors equal to zero.

        mjdref: float
            MJD reference (useful in most high-energy mission data)


        Attributes
        ----------
        time: numpy.ndarray
            The array of midpoints of time bins.

        bin_lo:
            The array of lower time stamp of time bins.

        bin_hi:
            The array of higher time stamp of time bins.

        counts: numpy.ndarray
            The counts per bin corresponding to the bins in `time`.

        counts_err: numpy.ndarray
            The uncertainties corresponding to `counts`

        countrate: numpy.ndarray
            The counts per second in each of the bins defined in `time`.

        countrate_err: numpy.ndarray
            The uncertainties corresponding to `countrate`

        meanrate: float
            The mean count rate of the light curve.

        meancounts: float
            The mean counts of the light curve.

        n: int
            The number of data points in the light curve.

        dt: float
            The time resolution of the light curve.

        mjdref: float
            MJD reference date (tstart / 86400 gives the date in MJD at the
            start of the observation)

        tseg: float
            The total duration of the light curve.

        tstart: float
            The start time of the light curve.

        gti: 2-d float array
            [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            Good Time Intervals. They indicate the "safe" time intervals
            to be used during the analysis of the light curve.

        err_dist: string
            Statistic of the Lightcurve, it is used to calculate the
            uncertainties and other statistical values apropriately.
            It propagates to Spectrum classes.

        """

        if not np.all(np.isfinite(time)):
            raise ValueError("There are inf or NaN values in "
                             "your time array!")

        if not np.all(np.isfinite(counts)):
            raise ValueError("There are inf or NaN values in "
                             "your counts array!")

        if len(time) != len(counts):

            raise StingrayError("time and counts array are not "
                                "of the same length!")

        if len(time) <= 1:
            raise StingrayError("A single or no data points can not create "
                                "a lightcurve!")

        if err is not None:
            if not np.all(np.isfinite(err)):
                raise ValueError("There are inf or NaN values in "
                                 "your err array")
        else:
            if err_dist.lower() not in valid_statistics:
                # err_dist set can be increased with other statistics
                raise StingrayError("Statistic not recognized."
                                    "Please select one of these: ",
                                    "{}".format(valid_statistics))
            if err_dist.lower() == 'poisson':
                # Instead of the simple square root, we use confidence
                # intervals (should be valid for low fluxes too)
                err_low, err_high = poisson_conf_interval(np.asarray(counts),
                    interval='frequentist-confidence', sigma=1)
                # calculate approximately symmetric uncertainties
                err = (np.absolute(err_low) + np.absolute(err_high) -
                       2 * np.asarray(counts))/2.0
                # other estimators can be implemented for other statistics
            else:
                simon("Stingray only uses poisson err_dist at the moment, "
                      "We are setting your errors to zero. "
                      "Sorry for the inconvenience.")
                err = np.zeros_like(counts)

        self.mjdref = mjdref
        self.time = np.asarray(time)
        if dt is None:
            self.dt = np.median(self.time[1:] - self.time[:-1])
        else:
            self.dt = dt

        self.bin_lo = self.time - 0.5 * self.dt
        self.bin_hi = self.time + 0.5 * self.dt

        self.err_dist = err_dist

        self.tstart = self.time[0] - 0.5*self.dt
        self.tseg = self.time[-1] - self.time[0] + self.dt

        self.gti = \
            np.asarray(assign_value_if_none(gti,
                                            [[self.tstart,
                                              self.tstart + self.tseg]]))
        check_gtis(self.gti)

        good = create_gti_mask(self.time, self.gti)

        self.time = self.time[good]
        if input_counts:
            self.counts = np.asarray(counts)[good]
            self.countrate = self.counts / self.dt
            self.counts_err = np.asarray(err)[good]
            self.countrate_err = np.asarray(err)[good] / self.dt
        else:
            self.countrate = np.asarray(counts)[good]
            self.counts = self.countrate * self.dt
            self.counts_err = np.asarray(err)[good] * self.dt
            self.countrate_err = np.asarray(err)[good]

        self.meanrate = np.mean(self.countrate)
        self.meancounts = np.mean(self.counts)
        self.n = self.counts.shape[0]

        # Issue a warning if the input time iterable isn't regularly spaced,
        # i.e. the bin sizes aren't equal throughout.
        dt_array = np.diff(self.time)
        if not (np.allclose(dt_array, np.repeat(self.dt, dt_array.shape[0]))):
            simon("Bin sizes in input time array aren't equal throughout! "
                  "This could cause problems with Fourier transforms. "
                  "Please make the input time evenly sampled.")


    def change_mjdref(self, new_mjdref):
        """Change the MJDREF of the light curve.

        Times will be now referred to this new MJDREF

        Parameters
        ----------
        new_mjdref : float
            New MJDREF
        """
        time_shift = (new_mjdref - self.mjdref) * 86400

        new_lc = self.shift(time_shift)
        new_lc.mjdref = new_mjdref
        return new_lc

    def shift(self, time_shift):
        """
        Shift the light curve and the GTIs in time.

        Parameters
        ----------
        time_shift: float
            The amount of time that the light curve will be shifted
        """
        new_lc = Lightcurve(self.time + time_shift, self.counts,
                            gti=self.gti + time_shift, mjdref=self.mjdref)
        new_lc.countrate = self.countrate
        new_lc.counts = self.counts
        new_lc.counts_err = self.counts_err
        new_lc.meanrate = np.mean(new_lc.countrate)
        new_lc.meancounts = np.mean(new_lc.counts)
        return new_lc

    def _operation_with_other_lc(self, other, operation):
        if self.mjdref != other.mjdref:
            raise ValueError("MJDref is different in the two light curves")

        common_gti = cross_two_gtis(self.gti, other.gti)
        mask_self = create_gti_mask(self.time, common_gti)
        mask_other = create_gti_mask(other.time, common_gti)

        # ValueError is raised by Numpy while asserting np.equal over arrays
        # with different dimensions.
        try:
            assert np.all(np.equal(self.time[mask_self],
                                   other.time[mask_other]))
        except (ValueError, AssertionError):
            raise ValueError("GTI-filtered time arrays of both light curves "
                             "must be of same dimension and equal.")

        new_time = self.time[mask_self]
        new_counts = operation(self.counts[mask_self],
                               other.counts[mask_other])

        if self.err_dist.lower() != other.err_dist.lower():
            simon("Lightcurves have different statistics!"
                  "We are setting the errors to zero to avoid complications.")
            new_counts_err = np.zeros_like(new_counts)
        elif self.err_dist.lower() in valid_statistics:
                new_counts_err = np.sqrt(np.add(self.counts_err[mask_self]**2,
                                                other.counts_err[mask_other]**2))
            # More conditions can be implemented for other statistics
        else:
            raise StingrayError("Statistics not recognized."
                                " Please use one of these: "
                                "{}".format(valid_statistics))

        lc_new = Lightcurve(new_time, new_counts,
                            err=new_counts_err, gti=common_gti,
                            mjdref=self.mjdref)

        return lc_new

    def __add__(self, other):
        """
        Add two light curves element by element having the same time array.

        This magic method adds two Lightcurve objects having the same time
        array such that the corresponding counts arrays get summed up.

        GTIs are crossed, so that only common intervals are saved.

        Example
        -------
        >>> time = [5, 10, 15]
        >>> count1 = [300, 100, 400]
        >>> count2 = [600, 1200, 800]
        >>> gti1 = [[0, 20]]
        >>> gti2 = [[0, 25]]
        >>> lc1 = Lightcurve(time, count1, gti=gti1)
        >>> lc2 = Lightcurve(time, count2, gti=gti2)
        >>> lc = lc1 + lc2
        >>> lc.counts
        array([ 900, 1300, 1200])
        """

        return self._operation_with_other_lc(other, np.add)

    def __sub__(self, other):
        """
        Subtract two light curves element by element having the same time array.

        This magic method subtracts two Lightcurve objects having the same
        time array such that the corresponding counts arrays interferes with
        each other.

        GTIs are crossed, so that only common intervals are saved.

        Example
        -------
        >>> time = [10, 20, 30]
        >>> count1 = [600, 1200, 800]
        >>> count2 = [300, 100, 400]
        >>> gti1 = [[0, 35]]
        >>> gti2 = [[0, 40]]
        >>> lc1 = Lightcurve(time, count1, gti=gti1)
        >>> lc2 = Lightcurve(time, count2, gti=gti2)
        >>> lc = lc1 - lc2
        >>> lc.counts
        array([ 300, 1100,  400])
        """

        return self._operation_with_other_lc(other, np.subtract)

    def __neg__(self):
        """
        Implement the behavior of negation of the light curve objects.

        The negation operator ``-`` is supposed to invert the sign of the count
        values of a light curve object.

        Example
        -------
        >>> time = [1, 2, 3]
        >>> count1 = [100, 200, 300]
        >>> count2 = [200, 300, 400]
        >>> lc1 = Lightcurve(time, count1)
        >>> lc2 = Lightcurve(time, count2)
        >>> lc_new = -lc1 + lc2
        >>> lc_new.counts
        array([100, 100, 100])
        """
        lc_new = Lightcurve(self.time, -1*self.counts,
                            err=self.counts_err, gti=self.gti,
                            mjdref=self.mjdref)

        return lc_new

    def __len__(self):
        """
        Return the length of the data array of a light curve.

        This method implements overrides the len function over a Lightcurve
        object and returns the length of the time and count arrays.

        Example
        -------
        >>> time = [1, 2, 3]
        >>> count = [100, 200, 300]
        >>> lc = Lightcurve(time, count)
        >>> len(lc)
        3
        """
        return self.n

    def __getitem__(self, index):
        """
        Return the corresponding count value at the index or a new Lightcurve
        object upon slicing.

        This method adds functionality to retrieve the count value at
        a particular index. This also can be used for slicing and generating
        a new Lightcurve object.

        Parameters
        ----------
        index : int or slice instance
            Index value of the time array or a slice object.

        Example
        -------
        >>> time = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> count = [11, 22, 33, 44, 55, 66, 77, 88, 99]
        >>> lc = Lightcurve(time, count)
        >>> lc[2]
        33
        >>> lc[:2].counts
        array([11, 22])
        """
        if isinstance(index, int):
            return self.counts[index]
        elif isinstance(index, slice):
            new_counts = self.counts[index.start:index.stop:index.step]
            new_time = self.time[index.start:index.stop:index.step]
            return Lightcurve(new_time, new_counts, mjdref=self.mjdref)
        else:
            raise IndexError("The index must be either an integer or a slice "
                             "object !")

    def baseline(self, lam, p, niter=10):
        """Calculate the baseline of the light curve, accounting for GTIs.

        Parameters
        ----------
        lam : float
            "smoothness" parameter. Larger values make the baseline stiffer
            Typically 1e2 < lam < 1e9
        p : float
            "asymmetry" parameter. Smaller values make the baseline more 
            "horizontal". Typically 0.001 < p < 0.1, but not necessary.
        """
        baseline = np.zeros_like(self.time)
        for g in self.gti:
            good = create_gti_mask(self.time, [g])
            baseline[good] = baseline_als(self.counts[good], lam, p, niter)

        return baseline

    @staticmethod
    def make_lightcurve(toa, dt, tseg=None, tstart=None, gti=None, mjdref=0):

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

        gti: 2-d float array
            [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            Good Time Intervals

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
        # for cases where tseg/dt is not integer.
        # TODO: check that this is always consistent and that we
        # are not throwing away good events.

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

        return Lightcurve(time, counts, gti=gti, mjdref=mjdref, dt=dt)

    def rebin(self, dt_new, method='sum'):
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

        if dt_new < self.dt:
            raise ValueError("New time resolution must be larger than "
                             "old time resolution!")

        bin_time, bin_counts, bin_err, _ = \
            utils.rebin_data(self.time, self.counts, dt_new,
                             yerr=self.counts_err, method=method)

        lc_new = Lightcurve(bin_time, bin_counts, err=bin_err,
                            mjdref=self.mjdref, dt=dt_new)
        return lc_new

    def join(self, other):
        """
        Join two lightcurves into a single object.

        The new Lightcurve object will contain time stamps from both the
        objects. The count per bin in the resulting object will be the
        individual count per bin, or the average in case of overlapping
        time arrays of both lightcurve objects.

        Good Time intervals are also joined.

        Note : Time array of both lightcurves should not overlap each other.

        Parameters
        ----------
        other : Lightcurve object
            The other Lightcurve object which is supposed to be joined with.

        Returns
        -------
        lc_new : Lightcurve object
            The resulting lightcurve object.

        Example
        -------
        >>> time1 = [5, 10, 15]
        >>> count1 = [300, 100, 400]
        >>> time2 = [20, 25, 30]
        >>> count2 = [600, 1200, 800]
        >>> lc1 = Lightcurve(time1, count1)
        >>> lc2 = Lightcurve(time2, count2)
        >>> lc = lc1.join(lc2)
        >>> lc.time
        array([ 5, 10, 15, 20, 25, 30])
        >>> lc.counts
        array([ 300,  100,  400,  600, 1200,  800])
        """
        if self.mjdref != other.mjdref:
            raise ValueError("MJDref is different in the two light curves")

        if self.dt != other.dt:
            utils.simon("The two light curves have different bin widths.")

        if self.tstart <= other.tstart:
            new_time = np.unique(np.concatenate([self.time, other.time]))
        else:
            new_time = np.unique(np.concatenate([other.time, self.time]))

        if len(new_time) != len(self.time) + len(other.time):
            utils.simon("The two light curves have overlapping time ranges. "
                        "In the common time range, the resulting count will "
                        "be the average of the counts in the two light "
                        "curves. If you wish to sum, use `lc_sum = lc1 + "
                        "lc2`.")

        new_counts = []
        new_counts_err = []
        # For every time stamp, get the individual time counts and add them.
        for time in new_time:
            try:
                count1 = self.counts[np.where(self.time == time)[0][0]]
                count1_err = self.counts_err[np.where(self.time == time)[0][0]]
            except IndexError:
                count1 = None
                count1_err = None

            try:
                count2 = other.counts[np.where(other.time == time)[0][0]]
                count2_err = other.counts_err[np.where(other.time == time)[0][0]]
            except IndexError:
                count2 = None
                count2_err = None

            if count1 is not None:
                if count2 is not None:
                    # Average the overlapping counts
                    new_counts.append((count1 + count2) / 2)

                    if self.err_dist.lower() != other.err_dist.lower():
                        simon("Lightcurves have different statistics!"
                              "We are setting the errors to zero.")
                        new_counts_err = np.zeros_like(new_counts)
                    elif self.err_dist.lower() in valid_statistics:
                        new_counts_err.append(np.sqrt(((count1_err**2) +
                                                      (count2_err**2)) / 2))
                    # More conditions can be implemented for other statistics
                    else:
                        raise StingrayError("Statistics not recognized."
                                            " Please use one of these: "
                                            "{}".format(valid_statistics))
                else:
                    new_counts.append(count1)
                    new_counts_err.append(count1_err)
            else:
                new_counts.append(count2)
                new_counts_err.append(count2_err)

        new_counts = np.asarray(new_counts)
        new_counts_err = np.asarray(new_counts_err)

        gti = join_gtis(self.gti, other.gti)

        lc_new = Lightcurve(new_time, new_counts, err=new_counts_err, gti=gti,
                            mjdref=self.mjdref, dt=self.dt)

        return lc_new

    def truncate(self, start=0, stop=None, method="index"):
        """
        Truncate a Lightcurve object from points on the time array.

        This method allows the truncation of a Lightcurve object and returns
        a new light curve.

        Parameters
        ----------
        start : int, default 0
            Index of the starting point of the truncation.

        stop : int, default None
            Index of the ending point (exclusive) of the truncation. If no
            value of stop is set, then points including the last point in
            the counts array are taken in count.

        method : {"index" | "time"}, optional, default "index"
            Type of the start and stop values. If set to "index" then
            the values are treated as indices of the counts array, or
            if set to "time", the values are treated as actual time values.

        Example
        -------
        >>> time = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> count = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        >>> lc = Lightcurve(time, count)
        >>> lc_new = lc.truncate(start=2, stop=8)
        >>> lc_new.counts
        array([30, 40, 50, 60, 70, 80])
        >>> lc_new.time
        array([3, 4, 5, 6, 7, 8])

        # Truncation can also be done by time values
        >>> lc_new = lc.truncate(start=6, method='time')
        >>> lc_new.time
        array([6, 7, 8, 9])
        >>> lc_new.counts
        array([60, 70, 80, 90])

        Returns
        -------
        lc_new: :class:`Lightcurve` object
            The :class:`Lightcurve` object with truncated time and counts
            arrays.
        """

        if not isinstance(method, str):
            raise TypeError("method key word argument is not "
                            "a string !")

        if method.lower() not in ['index', 'time']:
            raise ValueError("Unknown method type " + method + ".")

        if method.lower() == 'index':
            return self._truncate_by_index(start, stop)
        else:
            return self._truncate_by_time(start, stop)

    def _truncate_by_index(self, start, stop):
        """Private method for truncation using index values."""
        time_new = self.time[start:stop]
        counts_new = self.counts[start:stop]
        counts_err_new = self.counts_err[start:stop]
        gti = \
            cross_two_gtis(self.gti,
                           np.asarray([[self.time[start] - 0.5 * self.dt,
                                        time_new[-1] + 0.5 * self.dt]]))

        return Lightcurve(time_new, counts_new, err=counts_err_new, gti=gti,
                          dt=self.dt)

    def _truncate_by_time(self, start, stop):
        """Private method for truncation using time values."""

        if stop is not None:
            if start > stop:
                raise ValueError("start time must be less than stop time!")

        if not start == 0:
            start = np.where(self.time == start)[0][0]

        if stop is not None:
            stop = np.where(self.time == stop)[0][0]

        return self._truncate_by_index(start, stop)

    def sort(self, reverse=False):
        """
        Sort a Lightcurve object in accordance with its counts array.

        A Lightcurve can be sorted in either increasing or decreasing order
        using this method. The counts array gets sorted and the time array is
        changed accordingly.

        Parameters
        ----------
        reverse : boolean, default False
            If True then the object is sorted in reverse order.

        Example
        -------
        >>> time = [1, 2, 3]
        >>> count = [200, 100, 300]
        >>> lc = Lightcurve(time, count)
        >>> lc.sort()
        >>> lc.counts
        array([100, 200, 300])
        >>> lc.time
        array([2, 1, 3])

        Returns
        -------
        lc_new: :class:`Lightcurve` object
            The :class:`Lightcurve` object with truncated time and counts
            arrays.
        """
        new_counts = sorted(self.counts, reverse=reverse)
        new_time = []
        new_counts_err = []
        for count in np.unique(new_counts):
            for index in np.where(self.counts == count)[0]:
                new_time.append(self.time[index])
                new_counts_err.append(self.counts_err[index])

        if reverse:
            new_time.reverse()
            new_counts_err.reverse()

        self.time = np.asarray(new_time)
        self.counts = np.asarray(new_counts)
        self.counts_err = np.asarray(new_counts_err)

    def plot(self, witherrors=False, labels=None, axis=None, title=None,
             marker='-', save=False, filename=None):
        """
        Plot the Lightcurve using Matplotlib.

        Plot the Lightcurve object on a graph ``self.time`` on x-axis and
        ``self.counts`` on y-axis with ``self.counts_err`` optionaly
        as error bars.

        Parameters
        ----------
        witherrors: boolean, default False
            Whether to plot the Lightcurve with errorbars or not

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
        if witherrors:
            fig = plt.errorbar(self.time, self.counts, yerr=self.counts_err,
                               fmt=marker)
        else:
            fig = plt.plot(self.time, self.counts, marker)

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
                plt.savefig('out.png')
            else:
                plt.savefig(filename)

    def write(self, filename, format_='pickle', **kwargs):
        """
        Exports LightCurve object.

        Parameters
        ----------
        filename: str
            Name of the LightCurve object to be created.

        format\_: str
            Available options are 'pickle', 'hdf5', 'ascii'
        """

        if format_ == 'ascii':
            io.write(np.array([self.time, self.counts]).T,
                     filename, format_, fmt=["%s", "%s"])

        elif format_ == 'pickle':
            io.write(self, filename, format_)

        elif format_ == 'hdf5':
            io.write(self, filename, format_)

        else:
            utils.simon("Format not understood.")

    def read(self, filename, format_='pickle'):
        """
        Imports LightCurve object.

        Parameters
        ----------
        filename: str
            Name of the LightCurve object to be read.

        format\_: str
            Available options are 'pickle', 'hdf5', 'ascii'

        Returns
        --------
        If format\_ is 'ascii': astropy.table is returned.
        If format\_ is 'hdf5': dictionary with key-value pairs is returned.
        If format\_ is 'pickle': class object is set.
        """

        if format_ == 'ascii' or format_ == 'hdf5':
            return io.read(filename, format_)

        elif format_ == 'pickle':
            self = io.read(filename, format_)

        else:
            utils.simon("Format not understood.")

    def split_by_gti(self):
        """
        Splits the `LightCurve` into a list of `LightCurve`s , using GTIs.
        """
        list_of_lcs = []

        start_bins, stop_bins = gti_border_bins(self.gti, self.time, self.dt)
        for i in range(len(start_bins)):
            start = start_bins[i]
            stop = stop_bins[i]
            # Note: GTIs are consistent with default in this case!
            new_lc = Lightcurve(self.time[start:stop], self.counts[start:stop],
                                err=self.counts_err[start:stop],
                                mjdref=self.mjdref, gti=[self.gti[i]],
                                dt=self.dt)
            list_of_lcs.append(new_lc)

        return list_of_lcs

    def apply_gtis(self):
        good = create_gti_mask(self.time, lc.gti)

