"""Functions to simulate event lists data."""
from __future__ import division, print_function, absolute_import
import numpy as np
import numpy.random as ra
from stingray.simulator.utils import _assign_value_if_none
import logging
import warnings

def gen_events_from_lc(
        times, lc, use_spline=False, bin_time=None):
    """
    Create events from a light curve.

    Parameters
    ----------
    times : array-like
        the center time of each light curve bin
    lc : array-like
        light curve, in units of counts/bin

    Returns
    -------
    event_list : array-like
        Simulated arrival times
    """
    try:
        import scipy.interpolate as sci
    except:
        if use_spline:
            warnings.warn("Scipy not available. ",
                          "use_spline option cannot be used.")
            use_spline = False

    # Cast as numpy arrays in case inputs are in list format
    times = np.asarray(times)
    lc = np.asarray(lc)

    bin_time = _assign_value_if_none(bin_time, times[1] - times[0])
    n_bin = len(lc)

    bin_start = 0

    maxlc = np.max(lc)
    intlc = maxlc * n_bin

    n_events_predict = int(intlc + 10 * np.sqrt(intlc))

    # Max number of events per chunk must be < 100000
    events_per_bin_predict = n_events_predict / n_bin
    if use_spline:
        max_bin = np.max([4, 1000000 / events_per_bin_predict])
        logging.debug("Using splines")
    else:
        max_bin = np.max([4, 5000000 / events_per_bin_predict])

    ev_list = np.zeros(n_events_predict)

    nev = 0

    while bin_start < n_bin:
        t0 = times[bin_start]
        bin_stop = min([bin_start + max_bin, n_bin + 1])
        lc_filt = lc[bin_start:bin_stop]
        t_filt = times[bin_start:bin_stop]
        logging.debug("{} {}".format(t_filt[0] - bin_time / 2,
                                     t_filt[-1] + bin_time / 2))

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

        logging.debug(random_ts[random_ts < 0])

        random_amps = ra.uniform(0, max(lc_filt), n_to_simulate)
        if use_spline:
            # print("Creating spline representation")
            lc_spl = sci.splrep(t_filt, lc_filt, s=np.longdouble(0), k=1)

            pts = sci.splev(random_ts, lc_spl)
        else:
            rough_bins = np.rint((random_ts - t0) / bin_time)
            rough_bins = rough_bins.astype(int)

            pts = lc_filt[rough_bins]
            #pts = [lc_filt[bin] for bin in rough_bins]

        good = random_amps < pts
        len1 = len(random_ts)
        random_ts = random_ts[good]
        len2 = len(random_ts)
        logging.debug("Max LC, nbin: {0} {1}".format(max(lc_filt), n_bin_filt))
        logging.debug("{0} Events generated".format(len1))
        logging.debug("{0} Events predicted".format(n_predict))
        logging.debug("{0} Events rejected".format(len1 - len2))
        random_ts = random_ts[:n_predict]
        random_ts.sort()
        new_nev = len(random_ts)
        ev_list[nev:nev + new_nev] = random_ts[:]
        nev += new_nev
        logging.debug(
            "{0} good events created ({1} ev/s)".format(new_nev,
                                                        new_nev / length))
        bin_start += max_bin

    # Discard all zero entries at the end!
    ev_list = ev_list[:nev]
    ev_list.sort()
    return ev_list


def gen_lc_from_events(event_list, bin_time, start_time=None,
           stop_time=None, center_time=True):
    """
    From a list of event times, extract a lightcurve.

    Parameters
    ----------
    event_list : array-like
        Times of arrival of events
    bin_time : float
        Binning time of the light curve

    Returns
    -------
    time : array-like
        The time bins of the light curve
    lc : array-like
        The light curve

    Other Parameters
    ----------------
    start_time : float
        Initial time of the light curve
    stop_time : float
        Stop time of the light curve
    center_time: bool
        If False, time is the start of the bin. Otherwise, the center
    """

    start_time = _assign_value_if_none(start_time, np.floor(event_list[0]))
    stop_time = _assign_value_if_none(stop_time, np.floor(event_list[-1]))

    logging.debug("lcurve: Time limits: %g -- %g" %
                  (start_time, stop_time))

    new_event_list = event_list[event_list >= start_time]
    new_event_list = new_event_list[new_event_list <= stop_time]

    # To compute the histogram, the times array must specify the bin edges.
    # therefore, if nbin is the length of the lightcurve, times will have
    # nbin + 1 elements

    new_event_list = ((new_event_list - start_time) / bin_time).astype(int)
    times = np.arange(start_time, stop_time, bin_time)
    lc = np.bincount(new_event_list, minlength=len(times))
    logging.debug("lcurve: Length of the lightcurve: %g" % len(times))
    logging.debug("Times, kind: %s, %s" % (repr(times), type(times[0])))
    logging.debug("Lc, kind: %s, %s" % (repr(lc), type(lc[0])))
    logging.debug("bin_time, kind: %s, %s" % (repr(bin_time), type(bin_time)))

    if center_time:
        times = times + bin_time / 2.

    return times, lc.astype(np.float)

def assign_energies(N, spectrum):
    '''
    Assign energies to an event list.

    Parameters
    ----------
    N: int
        Length/size of event list

    spectrum: 2-d array or list
        Energies versus corresponding fluxes. The 2-d array or list must
        have energies across the first dimension and fluxes across the
        second one.

    Returns
    -------
    assigned_energies: array-like
        Energies assigned to all events in event list
    '''

    # Cast spectrum as numpy arrays
    if isinstance(spectrum, list):
        try:
            energies = np.array(spectrum[0])
            fluxes = np.array(spectrum[1])
        except:
            assert False, "Spectrum must be a 2-d array or list"

    else:
        assert spectrum.shape[0] == 2, "Spectrum must be a 2-d array or list"
        energies = spectrum[0]
        fluxes = spectrum[1]

    # Create a set of probability values
    prob = fluxes / float(sum(fluxes))

    # Calculate cumulative probability
    cum_prob = np.cumsum(prob)

    # Draw N random numbers between 0 and 1, where N is the size of event list
    R = ra.uniform(0, 1, N)

    # Assign energies to events corresponding to the random numbers drawn
    return np.array([energies[np.argwhere(cum_prob == min(cum_prob[(cum_prob - r) > 0]))] for r in R])
