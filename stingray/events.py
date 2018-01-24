"""
Definition of :class:`EventList`.

:class:`EventList` is used to handle photon arrival times.
"""
from __future__ import absolute_import, division, print_function

from .io import read, write
from .utils import simon, assign_value_if_none
from .gti import cross_gtis, append_gtis, check_separate

from .lightcurve import Lightcurve
from stingray.simulator.base import simulate_times

import numpy as np
import numpy.random as ra

__all__ = ['EventList']


class EventList(object):
    def __init__(self, time=None, energy=None, ncounts=None, mjdref=0, dt=0,
                 notes="", gti=None, pi=None):
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

        energy: iterable
            A list of array of photon energy values

        mjdref : float
            The MJD used as a reference for the time array.

        ncounts: int
            Number of desired data points in event list.

        gtis: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            Good Time Intervals

        pi : integer, numpy.ndarray
            PI channels

        notes : str
            Any useful annotations

        Attributes
        ----------
        time: numpy.ndarray
            The array of event arrival times, in seconds from the reference
            MJD (self.mjdref)

        energy: numpy.ndarray
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

        self.energy = None if energy is None else np.array(energy)
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

        if (time is not None) and (energy is not None):
            if len(time) != len(energy):
                raise ValueError('Lengths of time and energy must be equal.')

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

        if tstart is None and self.gti is not None:
            tstart = self.gti[0][0]
            tseg = self.gti[-1][1] - tstart

        return Lightcurve.make_lightcurve(self.time, dt, tstart=tstart,
                                          gti=self.gti, tseg=tseg,
                                          mjdref=self.mjdref)

    @staticmethod
    def from_lc(lc):
        """
        Loads eventlist from light curve.

        Parameters
        ----------
        lc: lightcurve.Lightcurve object
            Light curve data to load from

        Returns
        -------
        ev: events.EventList object
            Event List
        """

        # Multiply times by number of counts
        times = [[i] * int(j) for i,j in zip(lc.time, lc.counts)]
        # Concatenate all lists
        times = [i for j in times for i in j]

        return EventList(time=times, gti=lc.gti)

    def simulate_times(self, lc, use_spline=False, bin_time=None):
        """
        Assign (simulate) photon arrival times to event list, using the
        acceptance-rejection method.

        Parameters
        ----------
        lc: `Lightcurve` object

        Other Parameters
        ----------------
        use_spline : bool
            Approximate the light curve with a spline to avoid binning effects
        bin_time : float
            The bin time of the light curve, if it needs to be specified for
            improved precision

        Return
        ------
        times : array-like
            Simulated photon arrival times
        """
        self.time = simulate_times(lc, use_spline=use_spline,
                                   bin_time=bin_time)
        self.gti = lc.gti
        self.ncounts = len(self.time)

    def simulate_energies(self, spectrum):
        """
        Assign (simulate) energies to event list.

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

            energy = np.array(spectrum)[0]
            fluxes = np.array(spectrum)[1]

            if not isinstance(energy, np.ndarray):
                raise IndexError("Spectrum must be a 2-d array or list")

        else:
            raise TypeError("Spectrum must be a 2-d array or list")

        # Create a set of probability values
        prob = fluxes / float(sum(fluxes))

        # Calculate cumulative probability
        cum_prob = np.cumsum(prob)

        # Draw N random numbers between 0 and 1, where N is the size of event
        # list
        R = ra.uniform(0, 1, self.ncounts)

        # Assign energies to events corresponding to the random numbers drawn
        self.energy = np.array([energy[np.argwhere(cum_prob ==
            np.min(cum_prob[(cum_prob - r) > 0]))] for r in R])

    def join(self, other):
        """
        Join two ``EventList`` objects into one.

        If both are empty, an empty ``EventList`` is returned.

        GTIs are crossed if the event lists are over a common time interval,
        and appended otherwise.

        PI and PHA remain None if they are None in both. Otherwise, 0 is used
        as a default value for the ``EventList``s where they were None.

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

        if self.dt != other.dt:
            simon("The time resolution is different."
                  " Using the rougher by default")
            ev_new.dt = np.max([self.dt, other.dt])

        if self.time is None and other.time is None:
            return ev_new

        if (self.time is None):
            simon("One of the event lists you are concatenating is empty.")
            self.time = np.asarray([])

        elif (other.time is None):
            simon("One of the event lists you are concatenating is empty.")
            other.time = np.asarray([])

        ev_new.time = np.concatenate([self.time, other.time])
        order = np.argsort(ev_new.time)
        ev_new.time = ev_new.time[order]

        if (self.pi is None) and (other.pi is None):
            ev_new.pi = None
        elif (self.pi is None) or (other.pi is None):
            self.pi = assign_value_if_none(self.pi, np.zeros_like(self.time))
            other.pi = assign_value_if_none(other.pi,
                                             np.zeros_like(other.time))

        if (self.pi is not None) and (other.pi is not None):
            ev_new.pi = np.concatenate([self.pi, other.pi])
            ev_new.pi = ev_new.pi[order]

        if (self.energy is None) and (other.energy is None):
            ev_new.energy = None
        elif (self.energy is None) or (other.energy is None):
            self.energy = assign_value_if_none(self.energy,
                                               np.zeros_like(self.time))
            other.energy = assign_value_if_none(other.energy,
                                             np.zeros_like(other.time))

        if (self.energy is not None) and (other.energy is not None):
            ev_new.energy = np.concatenate([self.energy, other.energy])
            ev_new.energy = ev_new.energy[order]

        if self.gti is None and other.gti is not None and len(self.time) > 0:
            self.gti = \
                assign_value_if_none(
                    self.gti, np.asarray([[self.time[0] - self.dt / 2,
                                           self.time[-1] + self.dt / 2]]))
        if other.gti is None and self.gti is not None and len(other.time) > 0:
            other.gti = \
                assign_value_if_none(
                    other.gti, np.asarray([[other.time[0] - other.dt / 2,
                                            other.time[-1] + other.dt / 2]]))

        if (self.gti is None) and (other.gti is None):
            ev_new.gti = None

        elif (self.gti is not None) and (other.gti is not None):
            if check_separate(self.gti, other.gti):
                ev_new.gti = append_gtis(self.gti, other.gti)
                simon('GTIs in these two event lists do not overlap at all.'
                    'Merging instead of returning an overlap.')
            else:
                ev_new.gti = cross_gtis([self.gti, other.gti])

        return ev_new

    @staticmethod
    def read(filename, format_='pickle'):
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
        attributes = ['time', 'energy', 'ncounts', 'mjdref', 'dt',
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

            return EventList(time=values[0], energy=values[1],
                             ncounts=values[2], mjdref=values[3], dt=values[4],
                             notes=values[5], gti=values[6], pi=values[7])

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
            write(self, filename, format_, tnames=['EVENTS', 'GTI'],
                  colsassign={'gti':'GTI'})

        else:
            raise KeyError("Format not understood.")

