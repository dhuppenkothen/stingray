"""
Definition of :class:`EventList`.

:class:`EventList` is used to handle photon arrival times.
"""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)
import numpy as np


class EventList(object):
    def __init__(self, times, mjdref=0, dt=0, notes="", gti=None, pi=None,
                 pha=None):
        """
        Make an event list object from an array of time stamps

        Parameters
        ----------
        times: iterable
            A list or array of time stamps

        Other Parameters
        ----------------
        dt: float
            The time resolution of the events. Only relevant when using events
            to produce light curves with similar bin time.

        mjdref : float
            The MJD used as a reference for the time array.

        gtis: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            Good Time Intervals

        pi : integer, numpy.ndarray
            PI channels

        pha : float, numpy.ndarray
            Photon energies

        Attributes
        ----------
        time: numpy.ndarray
            The array of event arrival times, in seconds from the reference
            MJD (self.mjdref)

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

        pha : float, numpy.ndarray
            Photon energies

        """
        self.time = np.array(times, dtype=np.longdouble)
        self.ncounts = len(times)
        self.notes = notes
        self.dt = dt
        self.mjdref = mjdref
        self.gti = gti
        self.pi = pi
        self.pha = pha

    @staticmethod
    def from_fits(fname, **kwargs):
        from .utils import _assign_value_if_none
        from .io import load_events_and_gtis

        ret = load_events_and_gtis(fname, **kwargs)

        times = ret.ev_list
        gti = ret.gtis

        pi = _assign_value_if_none(ret.additional_data, "PI", None)
        pha = _assign_value_if_none(ret.additional_data, "PHA", None)
        dt = ret.dt
        mjdref = ret.mjdref

        return EventList(times, gti=gti, pi=pi, pha=pha, dt=dt, mjdref=mjdref)
