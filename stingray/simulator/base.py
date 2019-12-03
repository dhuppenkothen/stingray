
import numpy as np
import numpy.random as ra
import scipy.interpolate as sci
from ..utils import assign_value_if_none


def simulate_times(lc, use_spline=False, bin_time=None):
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

    Returns
    -------
    times : array-like
        Simulated photon arrival times
    """

    times = lc.time
    counts = lc.counts

    bin_time = assign_value_if_none(bin_time, lc.dt)
    n_bin = len(counts)
    bin_start = 0
    maxlc = np.max(counts)
    intlc = maxlc * n_bin
    n_events_predict = int(intlc + 10 * np.sqrt(intlc))

    # Max number of events per chunk must be < 100000
    events_per_bin_predict = n_events_predict / n_bin

    if use_spline:
        max_bin = np.long(np.max([4, 1000000 / events_per_bin_predict]))

    else:
        max_bin = np.long(np.max([4, 5000000 / events_per_bin_predict]))

    ev_list = np.zeros(n_events_predict)
    nev = 0

    while bin_start < n_bin:

        t0 = times[bin_start]
        bin_stop = np.min([bin_start + max_bin, n_bin + 1])

        lc_filt = counts[bin_start:bin_stop]
        t_filt = times[bin_start:bin_stop]
        length = t_filt[-1] - t_filt[0]

        n_bin_filt = len(lc_filt)
        n_to_simulate = n_bin_filt * np.max(lc_filt)
        safety_factor = 10

        if n_to_simulate > 10000:
            safety_factor = 4.

        n_to_simulate += safety_factor * np.sqrt(n_to_simulate)
        n_to_simulate = int(np.ceil(n_to_simulate))
        n_predict = ra.poisson(np.sum(lc_filt))

        random_ts = ra.uniform(t_filt[0] - bin_time / 2,
                               t_filt[-1] + bin_time / 2, n_to_simulate)

        random_amps = ra.uniform(0, np.max(lc_filt), n_to_simulate)

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

    return time
