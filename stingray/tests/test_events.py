
import numpy as np
import os
import pytest

from ..events import EventList
from ..lightcurve import Lightcurve

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, 'data')

_H5PY_INSTALLED = True

try:
    import h5py
except ImportError:
    _H5PY_INSTALLED = False


class TestEvents(object):

    @classmethod
    def setup_class(self):
        self.time = [0.5, 1.5, 2.5, 3.5]
        self.counts = [3000, 2000, 2200, 3600]
        self.counts_flat = [3000, 3000, 3000, 3000]
        self.spectrum = [[1, 2, 3, 4, 5, 6],
                         [1000, 2040, 1000, 3000, 4020, 2070]]
        self.gti = [[0, 4]]

    def test_inequal_length(self):
        """Check that exception is raised in case of
        disparity in length of 'time' and 'energy'
        """
        with pytest.raises(ValueError):
            EventList(time=[1, 2, 3], energy=[10, 12])

    def test_to_lc(self):
        """Create a light curve from event list."""
        ev = EventList(self.time, gti=self.gti)
        lc = ev.to_lc(1)
        assert np.all((lc.time == [0.5, 1.5, 2.5, 3.5]))
        assert (lc.gti == self.gti).all()

    def test_from_lc(self):
        """Load event list from lightcurve"""
        lc = Lightcurve(time=[0.5, 1.5, 2.5], counts=[2, 1, 2])
        ev = EventList.from_lc(lc)

        assert (ev.time == np.array([0.5, 0.5, 1.5, 2.5, 2.5])).all()

    def test_simulate_times(self):
        """Simulate photon arrival times for an event list
        from light curve.
        """
        lc = Lightcurve(self.time, self.counts_flat, gti=self.gti)
        ev = EventList()
        ev.simulate_times(lc)
        lc_sim = ev.to_lc(dt=lc.dt, tstart=lc.tstart, tseg=lc.tseg)
        assert np.all((lc - lc_sim).counts < 3 * np.sqrt(lc.counts))

    def test_simulate_times_with_spline(self):
        """Simulate photon arrival times, with use_spline option
        enabled.
        """
        lc = Lightcurve(self.time, self.counts_flat, gti=self.gti)
        ev = EventList()
        ev.simulate_times(lc, use_spline=True)
        lc_sim = ev.to_lc(dt=lc.dt, tstart=lc.tstart, tseg=lc.tseg)
        assert np.all((lc - lc_sim).counts < 3 * np.sqrt(lc.counts))

    def test_simulate_energies(self):
        """Assign photon energies to an event list.
        """
        ev = EventList(ncounts=100)
        ev.simulate_energies(self.spectrum)

    def test_simulate_energies_with_1d_spectrum(self):
        """Test that simulate_energies() method raises index
        error exception is spectrum is 1-d.
        """
        ev = EventList(ncounts=100)
        with pytest.raises(IndexError):
            ev.simulate_energies(self.spectrum[0])

    def test_simulate_energies_with_wrong_spectrum_type(self):
        """Test that simulate_energies() method raises type error
        exception when wrong sepctrum type is supplied.
        """
        ev = EventList(ncounts=100)
        with pytest.raises(TypeError):
            ev.simulate_energies(1)

    def test_simulate_energies_with_counts_not_set(self):
        ev = EventList()
        ev.simulate_energies(self.spectrum)

    def test_compare_energy(self):
        """Compare the simulated energy distribution to actual distribution.
        """
        fluxes = np.array(self.spectrum[1])
        ev = EventList(ncounts=1000)
        ev.simulate_energies(self.spectrum)
        energy = [int(p) for p in ev.energy]

        # Histogram energy to get shape approximation
        gen_energy = ((np.array(energy) - 1) / 1).astype(int)
        lc = np.bincount(energy)

        # Remove first entry as it contains occurences of '0' element
        lc = lc[1:len(lc)]

        # Calculate probabilities and compare
        lc_prob = (lc/float(sum(lc)))
        fluxes_prob = fluxes/float(sum(fluxes))

        assert np.all(np.abs(lc_prob - fluxes_prob) < 3 * np.sqrt(fluxes_prob))

    def test_join_without_times_simulated(self):
        """Test if exception is raised when join method is
        called before first simulating times.
        """
        ev = EventList()
        ev_other = EventList()

        assert ev.join(ev_other).time is None

    def test_join_empty_lists(self):
        """Test if an empty event list can be concatenated
        with a non-empty event list.
        """
        ev = EventList(time=[1, 2, 3])
        ev_other = EventList()
        ev_new = ev.join(ev_other)
        assert np.all(ev_new.time == [1, 2, 3])

        ev = EventList()
        ev_other = EventList(time=[1, 2, 3])
        ev_new = ev.join(ev_other)
        assert np.all(ev_new.time == [1, 2, 3])

        ev = EventList()
        ev_other = EventList()
        ev_new = ev.join(ev_other)
        assert ev_new.time == None
        assert ev_new.gti == None
        assert ev_new.pi == None
        assert ev_new.energy == None

        ev = EventList(time=[1, 2, 3])
        ev_other = EventList([])
        ev_new = ev.join(ev_other)
        assert np.all(ev_new.time == [1, 2, 3])
        ev = EventList([])
        ev_other = EventList(time=[1, 2, 3])
        ev_new = ev.join(ev_other)
        assert np.all(ev_new.time == [1, 2, 3])

    def test_join_different_dt(self):
        ev = EventList(time=[10, 20, 30], dt = 1)
        ev_other = EventList(time=[40, 50, 60], dt = 3)
        ev_new = ev.join(ev_other)
        assert ev_new.dt == 3

    def test_join_without_energy(self):
        ev = EventList(time=[1, 2, 3], energy=[3, 3, 3])
        ev_other = EventList(time=[4, 5])
        ev_new = ev.join(ev_other)

        assert np.all(ev_new.energy == [3, 3, 3, 0, 0])

    def test_join_without_pi(self):
        ev = EventList(time=[1, 2, 3], pi=[3, 3, 3])
        ev_other = EventList(time=[4, 5])
        ev_new = ev.join(ev_other)

        assert np.all(ev_new.pi == [3, 3, 3, 0, 0])

    def test_join_with_gti_none(self):
        ev = EventList(time=[1, 2, 3])
        ev_other = EventList(time=[4, 5], gti=[[3.5, 5.5]])
        ev_new = ev.join(ev_other)

        assert np.all(ev_new.gti == [[1, 3], [3.5, 5.5]])

        ev = EventList(time=[1, 2, 3], gti=[[0.5, 3.5]])
        ev_other = EventList(time=[4, 5])
        ev_new = ev.join(ev_other)

        assert np.all(ev_new.gti == [[0.5, 3.5], [4, 5]])

        ev = EventList(time=[1, 2, 3])
        ev_other = EventList(time=[4, 5])
        ev_new = ev.join(ev_other)

        assert ev_new.gti == None

    def test_non_overlapping_join(self):
        """Join two overlapping event lists.
        """
        ev = EventList(time=[1, 1, 2, 3, 4],
                        energy=[3, 4, 7, 4, 3], gti=[[1, 2],[3, 4]])
        ev_other = EventList(time=[5, 6, 6, 7, 10],
                            energy=[4, 3, 8, 1, 2], gti=[[6, 7]])
        ev_new = ev.join(ev_other)

        assert (ev_new.time ==
                np.array([1, 1, 2, 3, 4, 5, 6, 6, 7, 10])).all()
        assert (ev_new.energy ==
                np.array([3, 4, 7, 4, 3, 4, 3, 8, 1, 2])).all()
        assert (ev_new.gti ==
                np.array([[1, 2],[3, 4],[6, 7]])).all()

    def test_overlapping_join(self):
        """Join two non-overlapping event lists.
        """
        ev = EventList(time=[1, 1, 10, 6, 5],
                        energy=[10, 6, 3, 11, 2], gti=[[1, 3],[5, 6]])
        ev_other = EventList(time=[5, 7, 6, 6, 10],
                            energy=[2, 3, 8, 1, 2], gti=[[5, 7],[8, 10]])
        ev_new = ev.join(ev_other)

        assert (ev_new.time ==
                np.array([1, 1, 5, 5, 6, 6, 6, 7, 10, 10])).all()
        assert (ev_new.energy ==
                np.array([10, 6, 2, 2, 11, 8, 1, 3, 3, 2])).all()
        assert (ev_new.gti == np.array([[5, 6]])).all()

    def test_io_with_ascii(self):
        ev = EventList(self.time)
        ev.write('ascii_ev.txt',format_='ascii')
        ev = ev.read('ascii_ev.txt', format_='ascii')
        assert np.all(ev.time == self.time)
        os.remove('ascii_ev.txt')

    def test_io_with_pickle(self):
        ev = EventList(self.time)
        ev.write('ev.pickle', format_='pickle')
        ev = ev.read('ev.pickle',format_='pickle')
        assert np.all(ev.time == self.time)
        os.remove('ev.pickle')

    def test_io_with_hdf5(self):
        ev = EventList(time=self.time)
        ev.write('ev.hdf5', format_='hdf5')

        if _H5PY_INSTALLED:
            ev = ev.read('ev.hdf5',format_='hdf5')
            assert np.all(ev.time == self.time)
            os.remove('ev.hdf5')

        else:
            ev = ev.read('ev.pickle',format_='pickle')
            assert np.all(ev.time == self.time)
            os.remove('ev.pickle')

    def test_io_with_fits(self):
        ev = EventList(time = self.time)
        ev.write('ev.fits', format_='fits')
        ev = ev.read('ev.fits', format_='fits')
        assert np.all(ev.time == self.time)
        os.remove('ev.fits')

    def test_fits_with_standard_file(self):
        """Test that fits works with a standard event list
        file.
        """
        fname = os.path.join(datadir, 'monol_testA.evt')
        ev = EventList()
        ev = ev.read(fname, format_='fits')

    def test_io_with_wrong_format(self):
        """Test that io methods raise Key Error when
        wrong format is provided.
        """
        ev = EventList()
        with pytest.raises(KeyError):
            ev.write('ev.pickle', format_="unsupported")

        with pytest.raises(KeyError):
            ev.read('ev.pickle', format_="unsupported")

