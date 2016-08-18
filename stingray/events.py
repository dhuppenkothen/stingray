"""
Definition of :class:`EventList`.

:class:`EventList` is used to handle photon arrival times.
"""
from __future__ import absolute_import, division, print_function

from .io import read, write
from .utils import simon, assign_value_if_none
from .gti import cross_gtis, append_gtis, check_separate

from .lightcurve import Lightcurve

import numpy as np
import numpy.random as ra


class EventList(object):
    def __init__(self, time=None, pha=None, ncounts=None, mjdref=0, dt=0, notes="", 
            gti=None, pi=None):
        """
        Make an event list object from an array of time stamps

        Parameters
        ----------
        time: iterable
            A list or array of time stamps

        Other Parameters
        ----------------
        dt: float
            The time resolution of the events. Only relevant when using events
            to produce light curves with similar bin time.

        pha: iterable
            A list of array of photon energy values

        mjdref : float
            The MJD used as a reference for the time array.

        ncounts: int
            Number of desired data points in event list.

        gtis: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            Good Time Intervals

        pi : integer, numpy.ndarray
            PI channels

        Attributes
        ----------
        time: numpy.ndarray
            The array of event arrival times, in seconds from the reference
            MJD (self.mjdref)

        pha: numpy.ndarray
            The array of photon energy values

        ncounts: int
            The number of data points in the event list

        dt: float
            The time resolution of the events. Only relevant when using events
            to produce light curves with similar bin time.

        mjdref : float
            The MJD used as a reference for the time array.

        gtis: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            Good Time Intervals

        pi : integer, numpy.ndarray
            PI channels

        """
        
        self.pha = None if pha is None else np.array(pha)
        self.notes = notes
        self.dt = dt
        self.mjdref = mjdref
        self.gti = gti
        self.pi = pi
        self.ncounts = ncounts

        if time is not None:
            self.time = np.array(time, dtype=np.longdouble)
            self.ncounts = len(time)
        else:
            self.time = None

        if (time is not None) and (pha is not None):
            if len(time) != len(pha):
                raise ValueError('Lengths of time and pha must be equal.')

    def to_lc(self, dt, tstart=None, tseg=None):
        """
        Convert event list to a light curve object.

        Parameters
        ----------
        dt: float
            Binning time of the light curve

        Other Parameters
        ----------------
        tstart : float
            Initial time of the light curve

        tseg: float
            Total duration of light curve
        
        Returns
        -------
        lc: `Lightcurve` object
        """

        return Lightcurve.make_lightcurve(self.time, dt, tstart=tstart, tseg=tseg)

    def set_times(self, lc, use_spline=False, bin_time=None):
        """
        Assign photon arrival times to event list, using acceptance-rejection
        method.

        Parameters
        ----------
        lc: `Lightcurve` object
        """ 
        
        try:
            import scipy.interpolate as sci
        
        except:
            if use_spline:
                simon("Scipy not available. Cannot use spline.")
                use_spline = False

        times = lc.time
        counts = lc.counts

        bin_time = assign_value_if_none(bin_time, times[1] - times[0])
        n_bin = len(counts)
        bin_start = 0
        maxlc = np.max(counts)
        intlc = maxlc * n_bin
        n_events_predict = int(intlc + 10 * np.sqrt(intlc))

        # Max number of events per chunk must be < 100000
        events_per_bin_predict = n_events_predict / n_bin
        
        if use_spline:
            max_bin = np.max([4, 1000000 / events_per_bin_predict])
            
        else:
            max_bin = np.max([4, 5000000 / events_per_bin_predict])

        ev_list = np.zeros(n_events_predict)
        nev = 0

        while bin_start < n_bin:
            
            t0 = times[bin_start]
            bin_stop = min([bin_start + max_bin, n_bin + 1])
            
            lc_filt = counts[bin_start:bin_stop]
            t_filt = times[bin_start:bin_stop]
            length = t_filt[-1] - t_filt[0]
            
            n_bin_filt = len(lc_filt)
            n_to_simulate = n_bin_filt * max(lc_filt)
            safety_factor = 10

            if n_to_simulate > 10000:
                safety_factor = 4.

            n_to_simulate += safety_factor * np.sqrt(n_to_simulate)
            n_to_simulate = int(np.ceil(n_to_simulate))
            n_predict = ra.poisson(np.sum(lc_filt))

            random_ts = ra.uniform(t_filt[0] - bin_time / 2,
                                   t_filt[-1] + bin_time / 2, n_to_simulate)

            random_amps = ra.uniform(0, max(lc_filt), n_to_simulate)
            
            if use_spline:
                lc_spl = sci.splrep(t_filt, lc_filt, s=np.longdouble(0), k=1)
                pts = sci.splev(random_ts, lc_spl)
            
            else:
                rough_bins = np.rint((random_ts - t0) / bin_time)
                rough_bins = rough_bins.astype(int)
                pts = lc_filt[rough_bins]
            
            good = random_amps < pts
            len1 = len(random_ts)
            random_ts = random_ts[good]
            
            len2 = len(random_ts)
            random_ts = random_ts[:n_predict]
            random_ts.sort()
            
            new_nev = len(random_ts)
            ev_list[nev:nev + new_nev] = random_ts[:]
            nev += new_nev
            bin_start += max_bin

        # Discard all zero entries at the end
        time = ev_list[:nev]
        time.sort()
        
        self.time = EventList(time).time
        self.ncounts = len(self.time)

    def set_pha(self, spectrum):
        """
        Assign energies to event list.

        Parameters
        ----------
        spectrum: 2-d array or list
            Energies versus corresponding fluxes. The 2-d array or list must
            have energies across the first dimension and fluxes across the
            second one.
        """

        if self.ncounts is None:
            simon("Either set time values or explicity provide counts.")
            return

        if isinstance(spectrum, list) or isinstance(spectrum, np.ndarray):
            
            pha = np.array(spectrum)[0]
            fluxes = np.array(spectrum)[1]

            if not isinstance(pha, np.ndarray):
                raise IndexError("Spectrum must be a 2-d array or list")
        
        else:
            raise TypeError("Spectrum must be a 2-d array or list")
        
        # Create a set of probability values
        prob = fluxes / float(sum(fluxes))

        # Calculate cumulative probability
        cum_prob = np.cumsum(prob)

        # Draw N random numbers between 0 and 1, where N is the size of event list
        R = ra.uniform(0, 1, self.ncounts)

        # Assign energies to events corresponding to the random numbers drawn
        self.pha = np.array([pha[np.argwhere(cum_prob == 
            min(cum_prob[(cum_prob - r) > 0]))] for r in R])

    def join(self, other):
        """
        Join two eventlist objects into one.

        Parameters
        ----------
        other : `EventList` object
            The other `EventList` object which is supposed to be joined with.

        Returns
        -------
        ev_new : EventList object
            The resulting EventList object.
        """

        ev_new = EventList()

        if (self.time is None) and (other.time is None):
            raise ValueError('Times of at least one event list must be set before joining.')

        elif (self.time is None):
            simon("One of the event lists you are concatenating is empty.")
            self.time = []

        elif (other.time is None):
            simon("One of the event lists you are concatenating is empty.")
            other.time = []

        ev_new.time = np.concatenate([self.time, other.time])
        order = np.argsort(ev_new.time)
        ev_new.time = ev_new.time[order] 

        if (self.pha is not None) and (other.pha is not None):
            ev_new.pha = np.concatenate([self.pha, other.pha])
            ev_new.pha = ev_new.pha[order]

        if (self.gti is not None) and (other.gti is not None):
            if check_separate(self.gti, other.gti):
                ev_new.gti = append_gtis(self.gti, other.gti)
                simon('GTIs in these two event lists do not overlap at all.'
                    'Merging instead of returning an overlap.')
            else:
                ev_new.gti = cross_gtis([self.gti, other.gti])

        return ev_new

    def read(self, filename, format_='pickle'):
        """
        Imports EventList object.

        Parameters
        ----------
        filename: str
            Name of the EventList object to be read.

        format_: str
            Available options are 'pickle', 'hdf5', 'ascii' and 'fits'.

        Returns
        -------
        ev: `EventList` object
        """
        attributes = ['time', 'pha', 'ncounts', 'mjdref', 'dt', 
                'notes', 'gti', 'pi']
        data = read(filename, format_, cols=attributes)

        if format_ == 'ascii':
            time = np.array(data.columns[0])
            return EventList(time=time)
        
        elif format_ == 'hdf5' or format_ == 'fits':
            keys = data.keys()
            values = []
            
            if format_ == 'fits':
                attributes = [a.upper() for a in attributes]

            for attribute in attributes:
                if attribute in keys:
                    values.append(data[attribute])

                else:
                    values.append(None)
                    
            return EventList(time=values[0], pha=values[1], ncounts=values[2], 
                mjdref=values[3], dt=values[4], notes=values[5], gti=values[6], pi=values[7])

        elif format_ == 'pickle':
            return data

        else:
            raise KeyError("Format not understood.")

    def write(self, filename, format_='pickle'):
        """
        Exports EventList object.

        Parameters
        ----------
        filename: str
            Name of the LightCurve object to be created.

        format_: str
            Available options are 'pickle', 'hdf5', 'ascii'
        """

        if format_ == 'ascii':
            write(np.array([self.time]).T, filename, format_, fmt=["%s"])

        elif format_ == 'pickle':
            write(self, filename, format_)

        elif format_ == 'hdf5':
            write(self, filename, format_)

        elif format_ == 'fits':
            write(self, filename, format_, tnames=['EVENTS','GTI'], 
                colsassign={'gti':'GTI'})

        else:
            raise KeyError("Format not understood.")

