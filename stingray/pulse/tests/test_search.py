from __future__ import division, print_function
from stingray.pulse.search import epoch_folding_search, z_n_search
from stingray.pulse.search import _profile_fast, phaseogram, plot_phaseogram
from stingray.pulse.search import plot_profile
from stingray.pulse.pulsar import fold_events
import numpy as np
from stingray import Lightcurve
from stingray.events import EventList

np.random.seed(20150907)

class TestAll(object):
    """Unit tests for the stingray.pulse.search module."""
    @classmethod
    def setup_class(cls):
        cls.pulse_frequency = 1/0.101
        cls.tstart = 0
        cls.tend = 25.25
        cls.tseg = cls.tend - cls.tstart
        cls.dt = 0.01212
        cls.times = np.arange(cls.tstart, cls.tend, cls.dt) + cls.dt / 2
        cls.counts = \
            100 + 20 * np.cos(2 * np.pi * cls.times * cls.pulse_frequency)
        lc = Lightcurve(cls.times, cls.counts, gti=[[cls.tstart, cls.tend]])
        events = EventList()
        events.simulate_times(lc)
        cls.event_times = events.time

    def test_prepare(self):
        pass

    def test_phaseogram(self):
        phaseogr, phases, times, additional_info = \
            phaseogram(self.event_times, self.pulse_frequency)
        assert np.all(times < 25.6)
        assert np.any(times > 25)
        assert np.all((phases >= 0)&(phases <= 2))

    def test_phaseogram_mjdref(self):
        phaseogr, phases, times, additional_info = \
            phaseogram(self.event_times, self.pulse_frequency,
                       mjdref=57000, out_filename='phaseogram_mjdref.png')
        assert np.all(times >= 57000)
        assert np.all((phases >= 0)&(phases <= 2))

    def test_phaseogram_mjdref_pepoch(self):
        phaseogr, phases, times, additional_info = \
            phaseogram(self.event_times, self.pulse_frequency,
                       mjdref=57000, out_filename='phaseogram_mjdref.png',
                       pepoch=57000)
        assert np.all(times >= 57000)
        assert np.all((phases >= 0)&(phases <= 2))

    def test_plot_phaseogram_fromfunc(self):
        import matplotlib.pyplot as plt
        fig = plt.figure('Phaseogram from func')
        ax = plt.subplot()
        phaseogr, phases, times, additional_info = \
            phaseogram(self.event_times, self.pulse_frequency, mjdref=57000,
                       pepoch=57000, phaseogram_ax=ax, plot=True)
        plt.savefig('phaseogram_fromfunc.png')
        plt.close(fig)

    def test_plot_phaseogram_direct(self):
        import matplotlib.pyplot as plt
        phaseogr, phases, times, additional_info = \
            phaseogram(self.event_times, self.pulse_frequency)
        plot_phaseogram(phaseogr, phases, times)
        plt.savefig('phaseogram_direct.png')
        plt.close(plt.gcf())

    def test_plot_profile(self):
        import matplotlib.pyplot as plt
        phase, prof, _ = fold_events(self.event_times,
                                            self.pulse_frequency)
        ax = plot_profile(phase, prof)
        plt.savefig('profile_direct.png')
        plt.close(plt.gcf())

    def test_plot_profile_existing_ax(self):
        import matplotlib.pyplot as plt
        fig = plt.figure('Pulse profile')
        ax = plt.subplot()
        phase, prof, _ = fold_events(self.event_times,
                                     self.pulse_frequency, ax=ax)
        ax = plot_profile(phase, prof, ax=ax)
        plt.savefig('profile_existing_ax.png')
        plt.close(fig)

    def test_plot_profile_errorbars(self):
        import matplotlib.pyplot as plt
        fig = plt.figure('Pulse profile')
        ax = plt.subplot()
        phase, prof, err = fold_events(self.event_times,
                                       self.pulse_frequency, ax=ax)

        ax = plot_profile(phase, prof, err=err, ax=ax)
        plt.savefig('profile_errorbars.png')
        plt.close(fig)

    def test_profile_fast(self):
        test_phase = np.arange(0, 1, 1/16)
        prof = _profile_fast(test_phase, nbin=16)
        assert np.all(prof == np.ones(16))

    def test_epoch_folding_search(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.85, 9.95, 0.1/self.tseg)
        freq, stat = epoch_folding_search(self.event_times, frequencies,
                                          nbin=16)

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq[np.argmax(stat)]
        assert maxstatbin == frequencies[minbin]

    def test_epoch_folding_search_expocorr(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.89, 9.91, 0.1/self.tseg)
        freq, stat = epoch_folding_search(self.event_times, frequencies,
                                          nbin=16, expocorr=True)

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq[np.argmax(stat)]
        assert maxstatbin == frequencies[minbin]

    def test_z_n_search(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.85, 9.95, 0.3/self.tseg)
        freq, stat = z_n_search(self.event_times, frequencies, nbin=16,
                                nharm=1)

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq[np.argmax(stat)]
        assert maxstatbin == frequencies[minbin]

    def test_z_n_search_expocorr(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.89, 9.91, 0.1/self.tseg)
        freq, stat = z_n_search(self.event_times, frequencies, nbin=16,
                                nharm=1, expocorr=True)

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq[np.argmax(stat)]
        assert maxstatbin == frequencies[minbin]
