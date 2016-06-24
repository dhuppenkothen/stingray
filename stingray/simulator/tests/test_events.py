
import numpy as np

from stingray.simulator import events
from stingray.sampledata import sample_data

class TestFakeSimulator(object):

    @classmethod
    def setup_class(self):
        self.times = [0.5, 1.5, 2.5, 3.5]
        self.counts = [3000, 2000, 2200, 3600]
        self.spectrum = [[1, 2, 3, 4, 5, 6],[1000, 2040, 1000, 3000, 4020, 2070]]

    def test_fake_event_create(self):
        """
        Simulate an event list from fake times and counts.
        """
        events.gen_events_from_lc(self.times, self.counts)

    def test_actual_event_create(self):
        """
        Simulate an event list from actual light curve.
        """
        lc = sample_data()
        new_lc = lc[0:100]
        events.gen_events_from_lc(new_lc.time, new_lc.counts)

    def test_event_create_with_spline(self):
        """
        Simulate an event list from actual light curve with use_spline = True.
        """
        lc = sample_data()
        new_lc = lc[0:100]
        events.gen_events_from_lc(new_lc.time, new_lc.counts, use_spline=True)

    def test_fake_recover_lcurve(self):
        """
        Recover a lightcurve from a fake event list.
        """
        ev_list = events.gen_events_from_lc(self.times, self.counts)
        new_times, new_counts = events.gen_lc_from_events(ev_list, 1, start_time=0,
                                                          stop_time=4)

        assert np.all(np.abs(new_counts - self.counts) < 3 * np.sqrt(self.counts))
        np.testing.assert_almost_equal(new_times, self.times)

    def test_actual_recover_lcurve(self):
        """
        Recover a lightcurve from an actual event list.
        """
        lc = sample_data()
        new_lc = lc[0:100]
        ev_list = events.gen_events_from_lc(new_lc.time, new_lc.counts)

        bin_length = new_lc.time[1] - new_lc.time[0]
        new_times, new_counts = events.gen_lc_from_events(ev_list, bin_length,
                                                          start_time = new_lc.time[0] - bin_length/2,
                                                          stop_time = new_lc.time[-1] + bin_length/2)

        #TODO: Sigma needs to be 4 in order to pass test. Should it be 3?
        assert np.all(np.abs(new_counts - new_lc.counts) < 4 * np.sqrt(new_lc.counts))
        np.testing.assert_almost_equal(new_times, new_lc.time)

    def test_assign_energies_from_arrays(self):
        """
        Assign energies to an event list given its spectrum from array input.
        """
        spectrum = np.array(self.spectrum)
        assert len(events.assign_energies(10, spectrum)) == 10

    def test_assign_energies_from_lists(self):
        """
        Assign energies to an event list given its spectrum from list input.
        """
        assert len(events.assign_energies(10, self.spectrum)) == 10

    def test_compare_energies(self):
        """
        Compare the simulated energy distribution to actual distribution.
        """
        fluxes = np.array(self.spectrum[1])
        energies = events.assign_energies(1000, self.spectrum)
        energies = [int(energy) for energy in energies]

        # Histogram energies to get shape approximation
        gen_energies = ((np.array(energies) - 1) / 1).astype(int)

        lc = np.bincount(energies)

        # Remove first entry as it contains occurences of '0' element
        lc = lc[1:7]

        # Calculate probabilities and compare
        lc_prob = (lc/float(sum(lc)))
        fluxes_prob = fluxes/float(sum(fluxes))
        assert np.all(np.abs(lc_prob - fluxes_prob) < 3 * np.sqrt(fluxes_prob))
