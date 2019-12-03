"""
Definition of :class::class:`Lightcurve`.

:class::class:`Lightcurve` is used to create light curves out of photon counting data
or to save existing light curves in a class that's easy to use.
"""

import logging
import numpy as np
import stingray.io as io
import stingray.utils as utils
from stingray.exceptions import StingrayError
from stingray.utils import simon, assign_value_if_none, baseline_als
from stingray.utils import poisson_symmetrical_errors
from stingray.gti import cross_two_gtis, join_gtis, gti_border_bins
from stingray.gti import check_gtis, create_gti_mask_complete, create_gti_mask, bin_intervals_from_gtis

__all__ = ["Lightcurve"]

valid_statistics = ["poisson", "gauss", None]


class Lightcurve(object):
    """
    Make a light curve object from an array of time stamps and an
    array of counts.

    Parameters
    ----------
    time: iterable
        A list or array of time stamps for a light curve

    counts: iterable, optional, default ``None``
        A list or array of the counts in each bin corresponding to the
        bins defined in `time` (note: use ``input_counts=False`` to
        input the count range, i.e. counts/second, otherwise use
        counts/bin).

    err: iterable, optional, default ``None``
        A list or array of the uncertainties in each bin corresponding to
        the bins defined in ``time`` (note: use ``input_counts=False`` to
        input the count rage, i.e. counts/second, otherwise use
        counts/bin). If ``None``, we assume the data is poisson distributed
        and calculate the error from the average of the lower and upper
        1-sigma confidence intervals for the Poissonian distribution with
        mean equal to ``counts``.

    input_counts: bool, optional, default True
        If True, the code assumes that the input data in ``counts``
        is in units of counts/bin. If False, it assumes the data
        in ``counts`` is in counts/second.

    gti: 2-d float array, default ``None``
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
        Good Time Intervals. They are *not* applied to the data by default.
        They will be used by other methods to have an indication of the
        "safe" time intervals to use during analysis.

    err_dist: str, optional, default ``None``
        Statistical distribution used to calculate the
        uncertainties and other statistical values appropriately.
        Default makes no assumptions and keep errors equal to zero.

    mjdref: float
        MJD reference (useful in most high-energy mission data)

    skip_checks: bool
        If True, the user specifies that data are already sorted and contain no
        infinite or nan points. Use at your own risk

    low_memory: bool
        If True, all the lazily evaluated attribute (e.g., countrate and
        countrate_err if input_counts is True) will _not_ be stored in memory,
        but calculated every time they are requested.

    Attributes
    ----------
    time: numpy.ndarray
        The array of midpoints of time bins.

    bin_lo: numpy.ndarray
        The array of lower time stamp of time bins.

    bin_hi: numpy.ndarray
        The array of higher time stamp of time bins.

    counts: numpy.ndarray
        The counts per bin corresponding to the bins in ``time``.

    counts_err: numpy.ndarray
        The uncertainties corresponding to ``counts``

    countrate: numpy.ndarray
        The counts per second in each of the bins defined in ``time``.

    countrate_err: numpy.ndarray
        The uncertainties corresponding to ``countrate``

    meanrate: float
        The mean count rate of the light curve.

    meancounts: float
        The mean counts of the light curve.

    n: int
        The number of data points in the light curve.

    dt: float
        The time resolution of the light curve.

    mjdref: float
        MJD reference date (``tstart`` / 86400 gives the date in MJD at the
        start of the observation)

    tseg: float
        The total duration of the light curve.

    tstart: float
        The start time of the light curve.

    gti: 2-d float array
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
        Good Time Intervals. They indicate the "safe" time intervals
        to be used during the analysis of the light curve.

    err_dist: string
        Statistic of the Lightcurve, it is used to calculate the
        uncertainties and other statistical values appropriately.
        It propagates to Spectrum classes.

    """
    def __init__(self, time, counts, err=None, input_counts=True,
                 gti=None, err_dist='poisson', mjdref=0, dt=None,
                 skip_checks=False, low_memory=False):

        time = np.asarray(time)
        counts = np.asarray(counts)
        if err is not None:
            err = np.asarray(err)

        if not skip_checks:
            time, counts, err = self.initial_optional_checks(time, counts, err)

        if time.size != counts.size:
            raise StingrayError("time and counts array are not "
                                "of the same length!")

        if time.size <= 1:
            raise StingrayError("A single or no data points can not create "
                                "a lightcurve!")

        if err_dist.lower() not in valid_statistics:
            # err_dist set can be increased with other statistics
            raise StingrayError("Statistic not recognized."
                                "Please select one of these: ",
                                "{}".format(valid_statistics))
        elif not err_dist.lower() == 'poisson':
            simon("Stingray only uses poisson err_dist at the moment. "
                  "All analysis in the light curve will assume Poisson "
                  "errors. "
                  "Sorry for the inconvenience.")

        if err is not None:
            err = np.asarray(err)
            if not skip_checks and not np.all(np.isfinite(err)):
                raise ValueError("There are inf or NaN values in "
                                 "your err array")

        self.mjdref = mjdref
        self._time = time

        if dt is None:
            logging.warning("Computing the bin time ``dt``. This can take "
                            "time. If you know the bin time, please specify it"
                            " at light curve creation")
            dt = np.median(np.diff(self._time))

        self.dt = dt
        self.err_dist = err_dist

        self.tstart = self._time[0] - 0.5 * self.dt
        self.tseg = self._time[-1] - self._time[0] + self.dt

        self._gti = None
        if gti is not None:
            self._gti = np.asarray(gti)

        self._mask = None
        self._counts = None
        self._counts_err = None
        self._countrate = None
        self._countrate_err = None
        self._meanrate = None
        self._meancounts = None
        self._bin_lo = None
        self._bin_hi = None
        self._n = None

        self.input_counts = input_counts
        self.low_memory = low_memory
        if input_counts:
            self._counts = np.asarray(counts)
            self._counts_err = err
        else:
            self._countrate = np.asarray(counts)
            self._countrate_err = err

        if not skip_checks:
            self.check_lightcurve()

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        value = np.asarray(value)
        if not value.shape == self.time.shape:
            raise ValueError('Can only assign new times of the same shape as '
                             'the original array')
        self._time = value
        self._bin_lo = None
        self._bin_hi = None

    @property
    def gti(self):
        if self._gti is None:
            self._gti = \
                np.asarray([[self.tstart, self.tstart + self.tseg]])
        return self._gti

    @gti.setter
    def gti(self, value):
        value = np.asarray(value)
        self._gti = value
        self._mask = None

    @property
    def mask(self):
        if self._mask is None:
            self._mask = create_gti_mask(self.time, self.gti, dt=self.dt)
        return self._mask

    @property
    def n(self):
        if self._n is None:
            self._n = self.counts.shape[0]
        return self._n

    @property
    def meanrate(self):
        if self._meanrate is None:
            self._meanrate = np.mean(self.countrate[self.mask])
        return self._meanrate

    @property
    def meancounts(self):
        if self._meancounts is None:
            self._meancounts = np.mean(self.counts[self.mask])
        return self._meancounts

    @property
    def counts(self):
        counts = self._counts
        if self._counts is None:
            counts = self._countrate * self.dt
            # If not in low-memory regime, cache the values
            if not self.low_memory or self.input_counts:
                self._counts = counts

        return counts

    @counts.setter
    def counts(self, value):
        value = np.asarray(value)
        if not value.shape == self.counts.shape:
            raise ValueError('Can only assign new counts array of the same '
                             'shape as the original array')
        self._counts = value
        self._countrate = None
        self._meancounts = None
        self._meancountrate = None
        self.input_counts = True

    @property
    def counts_err(self):
        counts_err = self._counts_err
        if counts_err is None and self._countrate_err is not None:
            counts_err = self._countrate_err * self.dt
        elif counts_err is None:
            if self.err_dist.lower() == 'poisson':
                counts_err = poisson_symmetrical_errors(self.counts)
            else:
                counts_err = np.zeros_like(self.counts)

        # If not in low-memory regime, cache the values ONLY if they have
        # been changed!
        if self._counts_err is not counts_err:
            if not self.low_memory or self.input_counts:
                self._counts_err = counts_err

        return counts_err

    @counts_err.setter
    def counts_err(self, value):
        value = np.asarray(value)
        if not value.shape == self.counts.shape:
            raise ValueError('Can only assign new error array of the same '
                             'shape as the original array')
        self._counts_err = value
        self._countrate_err = None

    @property
    def countrate(self):
        countrate = self._countrate
        if countrate is None:
            countrate = self._counts / self.dt
            # If not in low-memory regime, cache the values
            if not self.low_memory or not self.input_counts:
                self._countrate = countrate

        return countrate

    @countrate.setter
    def countrate(self, value):
        value = np.asarray(value)
        if not value.shape == self.countrate.shape:
            raise ValueError('Can only assign new countrate array of the same '
                             'shape as the original array')
        self._countrate = value
        self._counts = None
        self._meancounts = None
        self._meancountrate = None
        self.input_counts = False

    @property
    def countrate_err(self):
        countrate_err = self._countrate_err
        if countrate_err is None and self._counts_err is not None:
            countrate_err = self._counts_err / self.dt
        elif countrate_err is None:
            countrate_err = 0

        # If not in low-memory regime, cache the values ONLY if they have
        # been changed!
        if countrate_err is not self._countrate_err:
            if not self.low_memory or not self.input_counts:
                self._countrate_err = countrate_err

        return countrate_err

    @countrate_err.setter
    def countrate_err(self, value):
        value = np.asarray(value)
        if not value.shape == self.countrate.shape:
            raise ValueError('Can only assign new error array of the same '
                             'shape as the original array')
        self._countrate_err = value
        self._counts_err = None

    @property
    def bin_lo(self):
        if self._bin_lo is None:
            self._bin_lo = self.time - 0.5 * self.dt
        return self._bin_lo

    @property
    def bin_hi(self):
        if self._bin_hi is None:
            self._bin_hi = self.time + 0.5 * self.dt
        return self._bin_hi

    def initial_optional_checks(self, time, counts, err):
        logging.warning("Checking if light curve is well behaved. This "
                        "can take time, so if you are sure it is already "
                        "sorted, specify skip_checks=True at light curve "
                        "creation.")

        if not np.all(np.isfinite(time)):
            raise ValueError("There are inf or NaN values in "
                             "your time array!")

        if not np.all(np.isfinite(counts)):
            raise ValueError("There are inf or NaN values in "
                             "your counts array!")

        logging.warning("Checking if light curve is sorted.")
        dt_array = np.diff(time)
        unsorted = np.any(dt_array < 0)

        if unsorted:
            logging.warning("The light curve is unsorted. Now, sorting...")
            order = np.argsort(time)
            time = time[order]
            counts = counts[order]
            if err is not None:
                err = err[order]
        return time, counts, err

    def check_lightcurve(self):
        """Make various checks on the lightcurve.

        It can be slow, use it if you are not sure about your
        input data.
        """
        # Issue a warning if the input time iterable isn't regularly spaced,
        # i.e. the bin sizes aren't equal throughout.

        check_gtis(self.gti)

        dt_array = []
        for g in self.gti:
            mask = create_gti_mask(self.time, [g], dt=self.dt)
            t = self.time[mask]
            dt_array.extend(np.diff(t))
        dt_array = np.asarray(dt_array)

        if not (np.allclose(dt_array, np.repeat(self.dt, dt_array.shape[0]))):
            simon("Bin sizes in input time array aren't equal throughout! "
                  "This could cause problems with Fourier transforms. "
                  "Please make the input time evenly sampled.")

    def change_mjdref(self, new_mjdref):
        """Change the MJD reference time (MJDREF) of the light curve.

        Times will be now referred to this new MJDREF

        Parameters
        ----------
        new_mjdref : float
            New MJDREF

        Returns
        -------
        new_lc : lightcurve.Lightcurve object
            The new LC shifted by MJDREF
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
            The time interval by which the light curve will be shifted (in
            the same units as the time array in :class:`Lightcurve`

        Returns
        -------
        new_lc : lightcurve.Lightcurve object
            The new LC shifted by ``time_shift``

        """
        new_lc = Lightcurve(self.time + time_shift, self.counts,
                            err=self.counts_err,
                            gti=self.gti + time_shift, mjdref=self.mjdref,
                            dt=self.dt, err_dist=self.err_dist,
                            skip_checks=True)

        return new_lc

    def _operation_with_other_lc(self, other, operation):
        """
        Helper method to codify an operation of one light curve with another (e.g. add, subtract, ...).
        Takes into account the GTIs correctly, and returns a new :class:`Lightcurve` object.

        Parameters
        ----------
        other : :class:`Lightcurve` object
            A second light curve object

        operation : function
            An operation between the :class:`Lightcurve` object calling this method, and ``other``,
            operating on the ``counts`` attribute in each :class:`Lightcurve` object

        Returns
        -------
        lc_new : Lightcurve object
            The new light curve calculated in ``operation``
        """
        if self.mjdref != other.mjdref:
            raise ValueError("MJDref is different in the two light curves")

        common_gti = cross_two_gtis(self.gti, other.gti)
        mask_self = create_gti_mask(self.time, common_gti, dt=self.dt)
        mask_other = create_gti_mask(other.time, common_gti, dt=other.dt)

        # ValueError is raised by Numpy while asserting np.equal over arrays
        # with different dimensions.
        try:
            diff = np.abs((self.time[mask_self] - other.time[mask_other]))
            assert np.all(diff < self.dt / 100)
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
                new_counts_err = \
                    np.sqrt(np.add(self.counts_err[mask_self]**2,
                                   other.counts_err[mask_other]**2))
            # More conditions can be implemented for other statistics
        else:
            raise StingrayError("Statistics not recognized."
                                " Please use one of these: "
                                "{}".format(valid_statistics))

        lc_new = Lightcurve(new_time, new_counts,
                            err=new_counts_err, gti=common_gti,
                            mjdref=self.mjdref, skip_checks=True,
                            dt=self.dt)

        return lc_new

    def __add__(self, other):
        """
        Add the counts of two light curves element by element, assuming the light curves
        have the same time array.

        This magic method adds two :class:`Lightcurve` objects having the same time
        array such that the corresponding counts arrays get summed up.

        GTIs are crossed, so that only common intervals are saved.

        Examples
        --------
        >>> time = [5, 10, 15]
        >>> count1 = [300, 100, 400]
        >>> count2 = [600, 1200, 800]
        >>> gti1 = [[0, 20]]
        >>> gti2 = [[0, 25]]
        >>> lc1 = Lightcurve(time, count1, gti=gti1, dt=5)
        >>> lc2 = Lightcurve(time, count2, gti=gti2, dt=5)
        >>> lc = lc1 + lc2
        >>> lc.counts
        array([ 900, 1300, 1200])
        """

        return self._operation_with_other_lc(other, np.add)

    def __sub__(self, other):
        """
        Subtract the counts/flux of one light curve from the counts/flux of another
        light curve element by element, assuming the ``time`` arrays of the light curves
        match exactly.

        This magic method takes two :class:`Lightcurve` objects having the same
        ``time`` array and subtracts the ``counts`` of one :class:`Lightcurve` with
        that of another, while also updating ``countrate``, ``counts_err`` and ``countrate_err``
        correctly.

        GTIs are crossed, so that only common intervals are saved.

        Examples
        --------
        >>> time = [10, 20, 30]
        >>> count1 = [600, 1200, 800]
        >>> count2 = [300, 100, 400]
        >>> gti1 = [[0, 35]]
        >>> gti2 = [[0, 40]]
        >>> lc1 = Lightcurve(time, count1, gti=gti1, dt=5)
        >>> lc2 = Lightcurve(time, count2, gti=gti2, dt=5)
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

        Examples
        --------
        >>> time = [1, 2, 3]
        >>> count1 = [100, 200, 300]
        >>> count2 = [200, 300, 400]
        >>> lc1 = Lightcurve(time, count1)
        >>> lc2 = Lightcurve(time, count2)
        >>> lc_new = -lc1 + lc2
        >>> lc_new.counts
        array([100, 100, 100])
        """
        lc_new = Lightcurve(self.time, -1 * self.counts,
                            err=self.counts_err, gti=self.gti,
                            mjdref=self.mjdref, skip_checks=True,
                            dt=self.dt)

        return lc_new

    def __len__(self):
        """
        Return the number of time bins of a light curve.

        This method implements overrides the ``len`` function for a :class:`Lightcurve`
        object and returns the length of the ``time`` array (which should be equal to the
        length of the ``counts`` and ``countrate`` arrays).

        Examples
        --------
        >>> time = [1, 2, 3]
        >>> count = [100, 200, 300]
        >>> lc = Lightcurve(time, count, dt=1)
        >>> len(lc)
        3
        """
        return self.n

    def __getitem__(self, index):
        """
        Return the corresponding count value at the index or a new :class:`Lightcurve`
        object upon slicing.

        This method adds functionality to retrieve the count value at
        a particular index. This also can be used for slicing and generating
        a new :class:`Lightcurve` object. GTIs are recalculated based on the new light
        curve segment

        If the slice object is of kind ``start:stop:step``, GTIs are also sliced,
        and rewritten as ``zip(time - self.dt /2, time + self.dt / 2)``

        Parameters
        ----------
        index : int or slice instance
            Index value of the time array or a slice object.

        Examples
        --------
        >>> time = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> count = [11, 22, 33, 44, 55, 66, 77, 88, 99]
        >>> lc = Lightcurve(time, count, dt=1)
        >>> lc[2]
        33
        >>> lc[:2].counts
        array([11, 22])
        """
        if isinstance(index, (int, np.integer)):
            return self.counts[index]
        elif isinstance(index, slice):
            start = assign_value_if_none(index.start, 0)
            stop = assign_value_if_none(index.stop, len(self.counts))
            step = assign_value_if_none(index.step, 1)

            new_counts = self.counts[start:stop:step]
            new_time = self.time[start:stop:step]

            new_gti = [[self.time[start] - 0.5 * self.dt,
                        self.time[stop - 1] + 0.5 * self.dt]]
            new_gti = np.asarray(new_gti)
            if step > 1:
                new_gt1 = np.array(list(zip(new_time - self.dt / 2,
                                            new_time + self.dt / 2)))
                new_gti = cross_two_gtis(new_gti, new_gt1)
            new_gti = cross_two_gtis(self.gti, new_gti)

            return Lightcurve(new_time, new_counts, mjdref=self.mjdref,
                              gti=new_gti, dt=self.dt, skip_checks=True)
        else:
            raise IndexError("The index must be either an integer or a slice "
                             "object !")

    def __eq__(self, other_lc):
        """
        Compares two :class:`Lightcurve` objects.

        Light curves are equal only if their counts as well as times at which those counts occur equal.

        Examples
        --------
        >>> time = [1, 2, 3]
        >>> count1 = [100, 200, 300]
        >>> count2 = [100, 200, 300]
        >>> lc1 = Lightcurve(time, count1, dt=1)
        >>> lc2 = Lightcurve(time, count2, dt=1)
        >>> lc1 == lc2
        True
        """
        if not isinstance(other_lc, Lightcurve):
            raise ValueError('Lightcurve can only be compared with a Lightcurve Object')
        if (np.allclose(self.time, other_lc.time) and
                np.allclose(self.counts, other_lc.counts)):
            return True
        return False

    def baseline(self, lam, p, niter=10, offset_correction=False):
        """Calculate the baseline of the light curve, accounting for GTIs.

        Parameters
        ----------
        lam : float
            "smoothness" parameter. Larger values make the baseline stiffer
            Typically ``1e2 < lam < 1e9``
        p : float
            "asymmetry" parameter. Smaller values make the baseline more
            "horizontal". Typically ``0.001 < p < 0.1``, but not necessary.

        Other parameters
        ----------------
        offset_correction : bool, default False
            by default, this method does not align to the running mean of the
            light curve, but it goes below the light curve. Setting align to
            True, an additional step is done to shift the baseline so that it
            is shifted to the middle of the light curve noise distribution.


        Returns
        -------
        baseline : numpy.ndarray
            An array with the baseline of the light curve
        """
        baseline = np.zeros_like(self.time)
        for g in self.gti:
            good = create_gti_mask(self.time, [g], dt=self.dt)
            _, baseline[good] = \
                baseline_als(self.time[good], self.counts[good], lam, p,
                             niter, offset_correction=offset_correction,
                             return_baseline=True)

        return baseline

    @staticmethod
    def make_lightcurve(toa, dt, tseg=None, tstart=None, gti=None, mjdref=0,
                        use_hist=False):

        """
        Make a light curve out of photon arrival times, with a given time resolution ``dt``.
        Note that ``dt`` should be larger than the native time resolution of the instrument
        that has taken the data.

        Parameters
        ----------
        toa: iterable
            list of photon arrival times

        dt: float
            time resolution of the light curve (the bin width)

        tseg: float, optional, default ``None``
            The total duration of the light curve.
            If this is ``None``, then the total duration of the light curve will
            be the interval between the arrival between the first and the last
            photon in ``toa``.

                **Note**: If ``tseg`` is not divisible by ``dt`` (i.e. if ``tseg``/``dt`` is
                not an integer number), then the last fractional bin will be
                dropped!

        tstart: float, optional, default ``None``
            The start time of the light curve.
            If this is ``None``, the arrival time of the first photon will be used
            as the start time of the light curve.

        gti: 2-d float array
            ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
            Good Time Intervals

        use_hist : bool
            Use ``np.histogram`` instead of ``np.bincounts``. Might be advantageous
            for very short datasets.

        Returns
        -------
        lc: :class:`Lightcurve` object
            A :class:`Lightcurve` object with the binned light curve
        """

        toa = np.sort(np.asarray(toa))
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
            tseg = toa[-1] - tstart

        logging.info("make_lightcurve: tseg: " + str(tseg))

        timebin = np.int64(tseg / dt)
        logging.info("make_lightcurve: timebin:  " + str(timebin))

        tend = tstart + timebin * dt
        good = (tstart <= toa) & (toa < tend)
        if not use_hist:
            binned_toas = ((toa[good] - tstart) // dt).astype(np.int64)
            counts = \
                np.bincount(binned_toas, minlength=timebin)
            time = tstart + np.arange(0.5, 0.5 + len(counts)) * dt
        else:
            histbins = np.arange(tstart, tend + dt, dt)
            counts, histbins = np.histogram(toa[good], bins=histbins)
            time = histbins[:-1] + 0.5 * dt

        return Lightcurve(time, counts, gti=gti, mjdref=mjdref, dt=dt,
                          skip_checks=True, err_dist='poisson')

    def rebin(self, dt_new=None, f=None, method='sum'):
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

        method: {``sum`` | ``mean`` | ``average``}, optional, default ``sum``
            This keyword argument sets whether the counts in the new bins
            should be summed or averaged.

        Other Parameters
        ----------------
        f: float
            the rebin factor. If specified, it substitutes ``dt_new`` with
            ``f*self.dt``

        Returns
        -------
        lc_new: :class:`Lightcurve` object
            The :class:`Lightcurve` object with the new, binned light curve.
        """

        if f is None and dt_new is None:
            raise ValueError('You need to specify at least one between f and '
                             'dt_new')
        elif f is not None:
            dt_new = f * self.dt

        if dt_new < self.dt:
            raise ValueError("New time resolution must be larger than "
                             "old time resolution!")

        bin_time, bin_counts, bin_err = [], [], []
        gti_new = []
        for g in self.gti:
            if g[1] - g[0] < dt_new:
                continue
            else:
                # find start and end of GTI segment in data
                start_ind = self.time.searchsorted(g[0])
                end_ind = self.time.searchsorted(g[1])

                t_temp = self.time[start_ind:end_ind]
                c_temp = self.counts[start_ind:end_ind]
                e_temp = self.counts_err[start_ind:end_ind]

                bin_t, bin_c, bin_e, _ = \
                    utils.rebin_data(t_temp, c_temp, dt_new,
                                     yerr=e_temp, method=method)

                bin_time.extend(bin_t)
                bin_counts.extend(bin_c)
                bin_err.extend(bin_e)
                gti_new.append(g)

        if len(gti_new) == 0:
            raise ValueError("No valid GTIs after rebin.")

        lc_new = Lightcurve(bin_time, bin_counts, err=bin_err,
                            mjdref=self.mjdref, dt=dt_new, gti=gti_new,
                            skip_checks=True)
        return lc_new

    def join(self, other):
        """
        Join two lightcurves into a single object.

        The new :class:`Lightcurve` object will contain time stamps from both the
        objects. The ``counts`` and ``countrate`` attributes in the resulting object
        will contain the union of the non-overlapping parts of the two individual objects,
        or the average in case of overlapping ``time`` arrays of both :class:`Lightcurve` objects.

        Good Time Intervals are also joined.

        Note : Ideally, the ``time`` array of both lightcurves should not overlap.

        Parameters
        ----------
        other : :class:`Lightcurve` object
            The other :class:`Lightcurve` object which is supposed to be joined with.

        Returns
        -------
        lc_new : :class:`Lightcurve` object
            The resulting :class:`Lightcurve` object.

        Examples
        --------
        >>> time1 = [5, 10, 15]
        >>> count1 = [300, 100, 400]
        >>> time2 = [20, 25, 30]
        >>> count2 = [600, 1200, 800]
        >>> lc1 = Lightcurve(time1, count1, dt=5)
        >>> lc2 = Lightcurve(time2, count2, dt=5)
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

        if(self.tstart < other.tstart):
            first_lc = self
            second_lc = other
        else:
            first_lc = other
            second_lc = self

        if len(np.intersect1d(self.time, other.time) > 0):

            utils.simon("The two light curves have overlapping time ranges. "
                        "In the common time range, the resulting count will "
                        "be the average of the counts in the two light "
                        "curves. If you wish to sum, use `lc_sum = lc1 + "
                        "lc2`.")
            valid_err = False

            if self.err_dist.lower() != other.err_dist.lower():
                simon("Lightcurves have different statistics!"
                      "We are setting the errors to zero.")

            elif self.err_dist.lower() in valid_statistics:
                valid_err = True
            # More conditions can be implemented for other statistics
            else:
                raise StingrayError("Statistics not recognized."
                                    " Please use one of these: "
                                    "{}".format(valid_statistics))

            from collections import Counter
            counts = Counter()
            counts_err = Counter()

            for i, time in enumerate(first_lc.time):
                counts[time] = first_lc.counts[i]
                counts_err[time] = first_lc.counts_err[i]

            for i, time in enumerate(second_lc.time):

                if counts.get(time) is not None:  # Common time
                    counts[time] = (counts[time] + second_lc.counts[i]) / 2
                    counts_err[time] = \
                        np.sqrt(((counts_err[time] ** 2) +
                                 (second_lc.counts_err[i] ** 2)) / 2)

                else:
                    counts[time] = second_lc.counts[i]
                    counts_err[time] = second_lc.counts_err[i]

            new_time = list(counts.keys())
            new_counts = list(counts.values())
            if(valid_err):
                new_counts_err = list(counts_err.values())
            else:
                new_counts_err = np.zeros_like(new_counts)

            del[counts, counts_err]

        else:

            new_time = np.concatenate([first_lc.time, second_lc.time])
            new_counts = np.concatenate([first_lc.counts, second_lc.counts])
            new_counts_err = \
                np.concatenate([first_lc.counts_err, second_lc.counts_err])

        new_time = np.asarray(new_time)
        new_counts = np.asarray(new_counts)
        new_counts_err = np.asarray(new_counts_err)
        gti = join_gtis(self.gti, other.gti)

        lc_new = Lightcurve(new_time, new_counts, err=new_counts_err, gti=gti,
                            mjdref=self.mjdref, dt=self.dt)

        return lc_new

    def truncate(self, start=0, stop=None, method="index"):
        """
        Truncate a :class:`Lightcurve` object.

        This method takes a ``start`` and a ``stop`` point (either as indices,
        or as times in the same unit as those in the ``time`` attribute, and truncates
        all bins before ``start`` and after ``stop``, then returns a new :class:`Lightcurve`
        object with the truncated light curve.

        Parameters
        ----------
        start : int, default 0
            Index (or time stamp) of the starting point of the truncation. If no value is set
            for the start point, then all points from the first element in the ``time`` array
            are taken into account.

        stop : int, default ``None``
            Index (or time stamp) of the ending point (exclusive) of the truncation. If no
            value of stop is set, then points including the last point in
            the counts array are taken in count.

        method : {``index`` | ``time``}, optional, default ``index``
            Type of the start and stop values. If set to ``index`` then
            the values are treated as indices of the counts array, or
            if set to ``time``, the values are treated as actual time values.

        Returns
        -------
        lc_new: :class:`Lightcurve` object
            The :class:`Lightcurve` object with truncated time and counts
            arrays.

        Examples
        --------
        >>> time = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> count = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        >>> lc = Lightcurve(time, count, dt=1)
        >>> lc_new = lc.truncate(start=2, stop=8)
        >>> lc_new.counts
        array([30, 40, 50, 60, 70, 80])
        >>> lc_new.time
        array([3, 4, 5, 6, 7, 8])
        >>> # Truncation can also be done by time values
        >>> lc_new = lc.truncate(start=6, method='time')
        >>> lc_new.time
        array([6, 7, 8, 9])
        >>> lc_new.counts
        array([60, 70, 80, 90])

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
        """Helper method for truncation using time values.

        Parameters
        ----------
        start : float
            start time for new light curve; all time bins before this time will be discarded

        stop : float
            stop time for new light curve; all time bins after this point will be discarded

        Returns
        -------
            new_lc : Lightcurve
                A new :class:`Lightcurve` object with the truncated time bins

        """

        if stop is not None:
            if start > stop:
                raise ValueError("start time must be less than stop time!")

        if not start == 0:
            start = self.time.searchsorted(start)

        if stop is not None:
            stop = self.time.searchsorted(stop)

        return self._truncate_by_index(start, stop)

    def split(self, min_gap, min_points=1):
        """
        For data with gaps, it can sometimes be useful to be able to split
        the light curve into separate, evenly sampled objects along those
        data gaps. This method allows to do this: it finds data gaps of a
        specified minimum size, and produces a list of new `Lightcurve`
        objects for each contiguous segment.

        Parameters
        ----------
        min_gap : float
            The length of a data gap, in the same units as the `time` attribute
            of the `Lightcurve` object. Any smaller gaps will be ignored, any
            larger gaps will be identified and used to split the light curve.

        min_points : int, default 1
            The minimum number of data points in each light curve. Light
            curves with fewer data points will be ignored.

        Returns
        -------
        lc_split : iterable of `Lightcurve` objects
            The list of all contiguous light curves

        Example
        -------
        >>> time = np.array([1, 2, 3, 6, 7, 8, 11, 12, 13])
        >>> counts = np.random.rand(time.shape[0])
        >>> lc = Lightcurve(time, counts, dt=1)
        >>> split_lc = lc.split(1.5)

        """
        # calculate the difference between time bins
        tdiff = np.diff(self.time)
        # find all distances between time bins that are larger than `min_gap`
        gap_idx = np.where(tdiff >= min_gap)[0]

        # tolerance for the newly created GTIs: Note that this seems to work
        # with a tolerance of 2, but not if I substitute 10. I don't know why
        epsilon = np.min(tdiff)/2.0

        # calculate new GTIs
        gti_start = np.hstack([self.time[0]-epsilon, self.time[gap_idx+1]-epsilon])
        gti_stop = np.hstack([self.time[gap_idx]+epsilon, self.time[-1]+epsilon])

        gti = np.vstack([gti_start, gti_stop]).T
        if hasattr(self, 'gti') and self.gti is not None:
            gti = cross_two_gtis(self.gti, gti)
        self.gti = gti

        lc_split = self.split_by_gti(min_points=min_points)
        return lc_split

    def sort(self, reverse=False):
        """
        Sort a Lightcurve object by time.

        A Lightcurve can be sorted in either increasing or decreasing order
        using this method. The time array gets sorted and the counts array is
        changed accordingly.

        Parameters
        ----------
        reverse : boolean, default False
            If True then the object is sorted in reverse order.

        Examples
        --------
        >>> time = [2, 1, 3]
        >>> count = [200, 100, 300]
        >>> lc = Lightcurve(time, count, dt=1)
        >>> lc_new = lc.sort()
        >>> lc_new.time
        array([1, 2, 3])
        >>> lc_new.counts
        array([100, 200, 300])

        Returns
        -------
        lc_new: :class:`Lightcurve` object
            The :class:`Lightcurve` object with sorted time and counts
            arrays.
        """
        new_time, new_counts, new_counts_err = \
            zip(*sorted(zip(self.time, self.counts, self.counts_err),
                        reverse=reverse))
        new_time = np.asarray(new_time)
        new_counts = np.asarray(new_counts)
        new_counts_err = np.asarray(new_counts_err)

        new_lc = Lightcurve(new_time, new_counts, err=new_counts_err,
                            gti=self.gti, dt=self.dt, mjdref=self.mjdref,
                            skip_checks=True)

        return new_lc

    def sort_counts(self, reverse=False):
        """
        Sort a :class:`Lightcurve` object in accordance with its counts array.

        A :class:`Lightcurve` can be sorted in either increasing or decreasing order
        using this method. The counts array gets sorted and the time array is
        changed accordingly.

        Parameters
        ----------
        reverse : boolean, default ``False``
            If ``True`` then the object is sorted in reverse order.

        Returns
        -------
        lc_new: :class:`Lightcurve` object
            The :class:`Lightcurve` object with sorted ``time`` and ``counts``
            arrays.

        Examples
        --------
        >>> time = [1, 2, 3]
        >>> count = [200, 100, 300]
        >>> lc = Lightcurve(time, count, dt=1)
        >>> lc_new = lc.sort_counts()
        >>> lc_new.time
        array([2, 1, 3])
        >>> lc_new.counts
        array([100, 200, 300])
        """

        new_counts, new_time, new_counts_err = \
            zip(*sorted(zip(self.counts, self.time, self.counts_err),
                        reverse=reverse))

        new_lc = Lightcurve(new_time, new_counts, err=new_counts_err,
                            gti=self.gti, dt=self.dt, mjdref=self.mjdref,
                            skip_checks=True)

        return new_lc

    def estimate_chunk_length(self, min_total_counts=100, min_time_bins=100):
        """Estimate a reasonable segment length for chunk-by-chunk analysis.

        Choose a reasonable length for time segments, given a minimum number of total
        counts in the segment, and a minimum number of time bins in the segment.

        The user specifies a condition on the total counts in each segment and
        the minimum number of time bins.

        Other Parameters
        ----------------
        min_total_counts : int
            Minimum number of counts for each chunk
        min_time_bins : int
            Minimum number of time bins

        Returns
        -------
        chunk_length : float
            The length of the light curve chunks that satisfies the conditions

        Examples
        --------
        >>> import numpy as np
        >>> time = np.arange(150)
        >>> count = np.zeros_like(time) + 3
        >>> lc = Lightcurve(time, count, dt=1)
        >>> lc.estimate_chunk_length(min_total_counts=10, min_time_bins=3)
        4.0
        >>> lc.estimate_chunk_length(min_total_counts=10, min_time_bins=5)
        5.0
        >>> count[2:4] = 1
        >>> lc = Lightcurve(time, count, dt=1)
        >>> lc.estimate_chunk_length(min_total_counts=3, min_time_bins=1)
        4.0
        """

        rough_estimate = np.ceil(min_total_counts / self.meancounts) * self.dt

        chunk_length = np.max([rough_estimate, min_time_bins * self.dt])

        keep_searching = True
        while keep_searching:
            start_times, stop_times, results = \
                self.analyze_lc_chunks(chunk_length, np.sum)
            mincounts = np.min(results)
            if mincounts >= min_total_counts:
                keep_searching = False
            else:
                chunk_length *= np.ceil(min_total_counts / mincounts) * self.dt

        return chunk_length

    def analyze_lc_chunks(self, chunk_length, func, fraction_step=1, **kwargs):
        """Analyze segments of the light curve with any function.

        Parameters
        ----------
        chunk_length : float
            Length in seconds of the light curve segments
        func : function
            Function accepting a :class:`Lightcurve` object as single argument, plus
            possible additional keyword arguments, and returning a number or a
            tuple - e.g., ``(result, error)`` where both ``result`` and ``error`` are
            numbers.

        Other parameters
        ----------------
        fraction_step : float
            If the step is not a full ``chunk_length`` but less (e.g. a moving window),
            this indicates the ratio between step step and ``chunk_length`` (e.g.
            0.5 means that the window shifts of half ``chunk_length``)
        kwargs : keyword arguments
            These additional keyword arguments, if present, they will be passed
            to ``func``

        Returns
        -------
        start_times : array
            Lower time boundaries of all time segments.
        stop_times : array
            upper time boundaries of all segments.
        result : array of N elements
            The result of ``func`` for each segment of the light curve

        Examples
        --------
        >>> import numpy as np
        >>> time = np.arange(0, 10, 0.1)
        >>> counts = np.zeros_like(time) + 10
        >>> lc = Lightcurve(time, counts, dt=0.1)
        >>> # Define a function that calculates the mean
        >>> mean_func = lambda x: np.mean(x)
        >>> # Calculate the mean in segments of 5 seconds
        >>> start, stop, res = lc.analyze_lc_chunks(5, mean_func)
        >>> len(res) == 2
        True
        >>> np.all(res == 10)
        True
        """
        start, stop = bin_intervals_from_gtis(self.gti, chunk_length,
                                              self.time,
                                              fraction_step=fraction_step,
                                              dt=self.dt)
        start_times = self.time[start] - self.dt * 0.5

        # Remember that stop is one element above the last element, because
        # it's defined to be used in intervals start:stop
        stop_times = self.time[stop - 1] + self.dt * 1.5

        results = []
        for i, (st, sp) in enumerate(zip(start, stop)):
            lc_filt = self[st:sp]
            res = func(lc_filt, **kwargs)
            results.append(res)

        results = np.array(results)

        if len(results.shape) == 2:
            results = [results[:, i] for i in range(results.shape[1])]
        return start_times, stop_times, results

    def plot(self, witherrors=False, labels=None, axis=None, title=None,
             marker='-', save=False, filename=None):
        """
        Plot the light curve using ``matplotlib``.

        Plot the light curve object on a graph ``self.time`` on x-axis and
        ``self.counts`` on y-axis with ``self.counts_err`` optionally
        as error bars.

        Parameters
        ----------
        witherrors: boolean, default False
            Whether to plot the Lightcurve with errorbars or not

        labels : iterable, default ``None``
            A list of tuple with ``xlabel`` and ``ylabel`` as strings.

        axis : list, tuple, string, default ``None``
            Parameter to set axis properties of the ``matplotlib`` figure. For example
            it can be a list like ``[xmin, xmax, ymin, ymax]`` or any other
            acceptable argument for the``matplotlib.pyplot.axis()`` method.

        title : str, default ``None``
            The title of the plot.

        marker : str, default '-'
            Line style and color of the plot. Line styles and colors are
            combined in a single format string, as in ``'bo'`` for blue
            circles. See ``matplotlib.pyplot.plot`` for more options.

        save : boolean, optional, default ``False``
            If ``True``, save the figure with specified filename.

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
        else:
            plt.show(block=False)

    def write(self, filename, format_='pickle', **kwargs):
        """
        Write a :class:`Lightcurve` object to file. Currently supported formats are

        * pickle (not recommended for long-term storage)
        * HDF5
        * ASCII

        Parameters
        ----------
        filename: str
            Path and file name for the output file.

        format\_: str
            Available options are 'pickle', 'hdf5', 'ascii'
        """
        _ = self.counts, self.counts_err, \
            self.countrate, self.countrate_err, \
            self.gti
        if format_ == 'ascii':
            io.write(np.array([self.time, self.counts, self.counts_err]).T,
                     filename, format_, fmt=["%s", "%s", "%s"])
        elif format_ == 'pickle':
            io.write(self, filename, format_)
        elif format_ == 'hdf5':
            io.write(self, filename, format_)
        else:
            utils.simon("Format not understood.")

    def read(self, filename, format_='pickle', default_err_dist='gauss'):
        """
        Read a :class:`Lightcurve` object from file. Currently supported formats are

        * pickle (not recommended for long-term storage)
        * HDF5
        * ASCII

        Parameters
        ----------
        filename: str
            Path and file name for the file to be read.

        format\_: str
            Available options are 'pickle', 'hdf5', 'ascii'

        Other parameters
        ----------------

        default_err_dist: str, default='gauss'
            Default error distribution if not specified in the file (e.g. for
            ASCII files). The default is 'gauss' just because it is likely
            that people using ASCII light curves will want to specify Gaussian
            error bars, if any.

        Returns
        --------
        lc : ``astropy.table`` or ``dict`` or :class:`Lightcurve` object
            * If ``format\_`` is ``ascii``: ``astropy.table`` is returned.
            * If ``format\_`` is ``hdf5``: dictionary with key-value pairs is returned.
            * If ``format\_`` is ``pickle``: :class:`Lightcurve` object is returned.
        """

        if format_ == 'ascii':
            data_raw = io.read(filename, format_,
                               names=['time', 'counts', 'counts_err'])

            data = {'time': np.array(data_raw['time']),
                    'counts': np.array(data_raw['counts']),
                    'counts_err': np.array(data_raw['counts_err'])}
            data['dt'] = np.median(np.diff(data['time']))
            data['gti'] = np.array([[data['time'][0] - data['dt'] / 0,
                                     data['time'][-1] + data['dt'] / 0]])
            # We use default_err_dist == 'gauss' just because people using
            # ASCII files will generally use Gaussian errors. This can be
            # changed from the command line.
            data['err_dist'] = default_err_dist
            data['mjdref'] = 0
        elif format_ == 'hdf5':
            data_raw = io.read(filename, format_)

            data = {'time': np.array(data_raw['_time']),
                    'counts': np.array(data_raw['_counts']),
                    'counts_err': np.array(data_raw['_counts_err'])}
            data['dt'] = data_raw['dt']
            data['gti'] = None
            if 'gti' in data_raw:
                data['gti'] = data_raw['gti']
            data['err_dist'] = data_raw['err_dist']
            data['mjdref'] = data_raw['mjdref']
        elif format_ == 'pickle':
            return io.read(filename, format_)
        else:
            utils.simon("Format not understood.")
            return None
        return Lightcurve(data['time'], data['counts'], err=data['counts_err'],
                          dt=data['dt'], skip_checks=True, gti=data['gti'],
                          err_dist=data['err_dist'],
                          mjdref=data['mjdref'])

    def split_by_gti(self, min_points=2):
        """
        Split the current :class:`Lightcurve` object into a list of :class:`Lightcurve` objects, one
        for each continuous GTI segment as defined in the ``gti`` attribute.

        Parameters
        ----------
        min_points : int, default 1
            The minimum number of data points in each light curve. Light
            curves with fewer data points will be ignored.

        Returns
        -------
        list_of_lcs : list
            A list of :class:`Lightcurve` objects, one for each GTI segment
        """
        list_of_lcs = []

        start_bins, stop_bins = gti_border_bins(self.gti, self.time, self.dt)
        for i in range(len(start_bins)):
            start = start_bins[i]
            stop = stop_bins[i]

            if np.isclose(stop-start, 1):
                logging.warning("Segment with a single time bin! Ignoring this segment!")
                continue
            if (stop - start) < min_points:
                continue

            # Note: GTIs are consistent with default in this case!
            new_lc = Lightcurve(self.time[start:stop], self.counts[start:stop],
                                err=self.counts_err[start:stop],
                                mjdref=self.mjdref, gti=[self.gti[i]],
                                dt=self.dt, err_dist=self.err_dist,
                                skip_checks=True)
            list_of_lcs.append(new_lc)

        return list_of_lcs

    def apply_gtis(self):
        """
        Apply GTIs to a light curve. Filters the ``time``, ``counts``,
        ``countrate``, ``counts_err`` and ``countrate_err`` arrays for all bins
        that fall into Good Time Intervals and recalculates mean countrate
        and the number of bins.
        """
        check_gtis(self.gti)

        good = self.mask

        # nota bene: We set the private properties, otherwise we'll get a
        # ValueError from changing the shape of the arrays.
        self._time = self.time[good]
        self._counts = self.counts[good]
        if self._counts_err is not None:
            self._counts_err = self._counts_err[good]
        self._countrate = None
        self._countrate_err = None
        self._mask = None

        self._meanrate = None
        self._meancounts = None
        self._n = None
        self.tseg = np.max(self.gti) - np.min(self.gti)
        self.tstart = self.time - 0.5 * self.dt
