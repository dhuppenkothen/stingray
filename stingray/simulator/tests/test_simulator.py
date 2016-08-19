import numpy as np
import os

from astropy.tests.helper import pytest
from stingray import Lightcurve, Crossspectrum, sampledata
from stingray.simulator import simulator, models

_H5PY_INSTALLED = True

try:
    import h5py
except ImportError:
    _H5PY_INSTALLED = False


class TestSimulator(object):

    @classmethod
    def setup_class(self):
        self.simulator = simulator.Simulator(N=1024, mean=0.5, dt=0.125)

    def calculate_lag(self, lc, h, delay):
        """
        Class method to calculate lag between two light curves.
        """
        s = lc.counts
        output = self.simulator.simulate(s, h, 'same')[delay:]
        s = s[delay:]
        time = lc.time[delay:]

        lc1 = Lightcurve(time, s)
        lc2 = Lightcurve(time, output)
        cross = Crossspectrum(lc1, lc2)
        cross = cross.rebin(0.0075)

        return np.angle(cross.power) / (2 * np.pi * cross.freq)

    def test_simulate_with_seed(self):
        """
        Simulate with a random seed value.
        """
        self.simulator = simulator.Simulator(N=1024, seed=12)
        assert len(self.simulator.simulate(2).counts), 1024

    def test_simulate_with_incorrect_arguments(self):
        with pytest.raises(ValueError):
            self.simulator.simulate(1, 2, 3, 4)

    def test_simulate_channel(self):
        """
        Simulate an energy channel.
        """
        self.simulator.simulate_channel('3.5-4.5', 'lorenzian', [1, 2, 3, 4])
        self.simulator.delete_channel('3.5-4.5')

    def test_incorrect_simulate_channel(self):
        """Test simulating a channel that already exists."""
        self.simulator.simulate_channel('3.5-4.5', 2)
        with pytest.raises(KeyError):
            self.simulator.simulate_channel('3.5-4.5', 2)
        self.simulator.delete_channel('3.5-4.5')

    def test_get_channel(self):
        """
        Retrieve an energy channel after it has been simulated.
        """
        self.simulator.simulate_channel('3.5-4.5', 2)
        lc = self.simulator.get_channel('3.5-4.5')
        self.simulator.delete_channel('3.5-4.5')

    def test_get_channels(self):
        """
        Retrieve multiple energy channel after it has been simulated.
        """
        self.simulator.simulate_channel('3.5-4.5', 2)
        self.simulator.simulate_channel('4.5-5.5', 'smoothbknpo', [1, 2, 3, 4])
        lc = self.simulator.get_channels(['3.5-4.5', '4.5-5.5'])

        self.simulator.delete_channels(['3.5-4.5', '4.5-5.5'])

    def test_count_channels(self):
        """
        Count energy channels after they have been simulated.
        """
        self.simulator.simulate_channel('3.5-4.5', 2)
        self.simulator.simulate_channel('4.5-5.5', 1)

        assert self.simulator.count_channels() == 2
        self.simulator.delete_channels(['3.5-4.5', '4.5-5.5'])

    def test_delete_incorrect_channel(self):
        """
        Test if deleting incorrect channel raises a
        keyerror exception.
        """
        with pytest.raises(KeyError):
            self.simulator.delete_channel('3.5-4.5')

    def test_delete_incorrect_channels(self):
        """
        Test if deleting incorrect channels raises a
        keyerror exception.
        """
        with pytest.raises(KeyError):
            self.simulator.delete_channels(['3.5-4.5', '4.5-5.5'])

    def test_simulate_powerlaw(self):
        """
        Simulate light curve from power law spectrum.
        """
        assert len(self.simulator.simulate(2).counts), 1024

    def test_compare_powerlaw(self):
        """
        Compare simulated power spectrum with actual one.
        """
        B, N, red_noise, dt = 2, 1024, 10, 1

        self.simulator = simulator.Simulator(N=N, dt=dt, mean=5, rms=1,
                                             red_noise=red_noise)
        lc = [self.simulator.simulate(B) for i in range(1, 30)]
        simulated = self.simulator.powerspectrum(lc, lc[0].tseg)

        w = np.fft.rfftfreq(N, d=dt)[1:]
        actual = np.power((1/w), B/2)[:-1]

        actual_prob = actual/float(sum(actual))
        simulated_prob = simulated/float(sum(simulated))

        assert np.all(
            np.abs(actual_prob - simulated_prob) < 3*np.sqrt(actual_prob)
               )

    def test_simulate_powerspectrum(self):
        """
        Simulate light curve from any power spectrum.
        """
        s = np.random.rand(1024)
        assert len(self.simulator.simulate(s)), 1024

    def test_simulate_lorenzian(self):
        """
        Simulate light curve using lorenzian model.
        """
        assert len(self.simulator.simulate('lorenzian', [1, 2, 3, 4])), 1024

    def test_compare_lorenzian(self):
        """
        Compare simulated lorenzian spectrum with original spectrum.
        """
        N, red_noise, dt = 1024, 10, 1

        self.simulator = simulator.Simulator(N=N, dt=dt, mean=0.1,
                                             rms=0.4, red_noise=red_noise)
        lc = [self.simulator.simulate('lorenzian', [0.3, 0.9, 0.6, 0.5])
              for i in range(1, 30)]
        simulated = self.simulator.powerspectrum(lc, lc[0].tseg)

        w = np.fft.rfftfreq(N, d=dt)[1:]
        actual = models.lorenzian(w, [0.3, 0.9, 0.6, 0.5])[:-1]

        actual_prob = actual/float(sum(actual))
        simulated_prob = simulated/float(sum(simulated))

        assert np.all(
            np.abs(actual_prob - simulated_prob) < 3*np.sqrt(actual_prob))

    def test_simulate_smoothbknpo(self):
        """
        Simulate light curve using smooth broken power law model.
        """
        assert len(self.simulator.simulate('smoothbknpo', [1, 2, 3, 4])), 1024

    def test_compare_smoothbknpo(self):
        """
        Compare simulated smooth broken power law spectrum with original
        spectrum.
        """
        N, red_noise, dt = 1024, 10, 1

        self.simulator = simulator.Simulator(N=N, dt=dt, mean=0.1, rms=0.7,
                                             red_noise=red_noise)
        lc = [self.simulator.simulate('smoothbknpo', [0.6, 0.2, 0.6, 0.5])
              for i in range(1, 30)]

        simulated = self.simulator.powerspectrum(lc, lc[0].tseg)

        w = np.fft.rfftfreq(N, d=dt)[1:]
        actual = models.smoothbknpo(w, [0.6, 0.2, 0.6, 0.5])[:-1]

        actual_prob = actual/float(sum(actual))
        simulated_prob = simulated/float(sum(simulated))

        assert np.all(
            np.abs(actual_prob - simulated_prob) < 3*np.sqrt(actual_prob))

    def test_simulate_wrong_model(self):
        """
        Simulate with a model that does not exist.
        """
        with pytest.raises(ValueError):
            self.simulator.simulate('unsupported', [0.6, 0.2, 0.6, 0.5])

    def test_construct_simple_ir(self):
        """
        Construct simple impulse response.
        """
        t0, w = 100, 500
        assert len(self.simulator.simple_ir(t0, w)), (t0+w)/self.simulator.dt

    def test_construct_relativistic_ir(self):
        """
        Construct relativistic impulse response.
        """
        t1, t3 = 3, 10
        assert len(self.simulator.relativistic_ir(t1=t1, t3=t3)),\
                  (t1+t3)/self.simulator.dt

    def test_simulate_simple_impulse(self):
        """
        Simulate light curve from simple impulse response.
        """
        lc = sampledata.sample_data()
        s = lc.counts
        h = self.simulator.simple_ir(10, 1, 1)
        output = self.simulator.simulate(s, h)

    def test_powerspectrum(self):
        """
        Create a power spectrum from light curve.
        """
        self.simulator.simulate(2)
        self.simulator.powerspectrum(self.simulator.lc)

    def test_simulate_relativistic_impulse(self):
        """
        Simulate light curve from relativistic impulse response.
        """
        lc = sampledata.sample_data()
        s = lc.counts

        h = self.simulator.relativistic_ir()
        output = self.simulator.simulate(s, h)

    def test_filtered_simulate(self):
        """
        Simulate light curve using 'filtered' mode.
        """
        lc = sampledata.sample_data()
        s = lc.counts

        h = self.simulator.simple_ir()
        output = self.simulator.simulate(s, h, 'filtered')

    def test_simple_lag_spectrum(self):
        """
        Simulate light curve from simple impulse response and
        compute lag spectrum.
        """
        lc = sampledata.sample_data()
        h = self.simulator.simple_ir(start=14, width=1)
        delay = int(15/lc.dt)

        lag = self.calculate_lag(lc, h, delay)
        v_cutoff = 1.0/(2*15.0)
        h_cutoff = lag[int((v_cutoff-0.0075)*1/0.0075)]

        assert np.abs(15-h_cutoff) < np.sqrt(15)

    def test_relativistic_lag_spectrum(self):
        """
        Simulate light curve from relativistic impulse response and
        compute lag spectrum.
        """
        lc = sampledata.sample_data()
        h = self.simulator.relativistic_ir(t1=3, t2=4, t3=10)
        delay = int(4/lc.dt)

        lag = self.calculate_lag(lc, h, delay)
        v_cutoff = 1.0/(2*4)
        h_cutoff = lag[int((v_cutoff-0.0075)*1/0.0075)]

        assert np.abs(4-h_cutoff) < np.sqrt(4)

    def test_position_varying_channels(self):
        """
        Tests lags for multiple energy channels with each channel
        having same intensity and varying position.
        """
        lc = sampledata.sample_data()
        s = lc.counts
        h = []
        h.append(self.simulator.simple_ir(start=4, width=1))
        h.append(self.simulator.simple_ir(start=9, width=1))

        delays = [int(5/lc.dt), int(10/lc.dt)]

        outputs = []
        for i in h:
            lc2 = self.simulator.simulate(s, i)
            lc2 = lc2.shift(-lc2.time[0] + lc.time[0])
            outputs.append(lc2)

        cross = [Crossspectrum(lc, lc2).rebin(0.0075) for lc2 in outputs]
        lags = [np.angle(c.power) / (2 * np.pi * c.freq) for c in cross]

        v_cutoffs = [1.0/(2.0*5), 1.0/(2.0*10)]
        h_cutoffs = [lag[int((v-0.0075)*1/0.0075)]
                     for lag, v in zip(lags, v_cutoffs)]

        assert np.abs(5-h_cutoffs[0]) < np.sqrt(5)
        assert np.abs(10-h_cutoffs[1]) < np.sqrt(10)

    def test_intensity_varying_channels(self):
        """
        Tests lags for multiple energy channels with each channel
        having same position and varying intensity.
        """
        lc = sampledata.sample_data()
        s = lc.counts
        h = []
        h.append(self.simulator.simple_ir(start=4, width=1, intensity=10))
        h.append(self.simulator.simple_ir(start=4, width=1, intensity=20))

        delay = int(5/lc.dt)

        outputs = []
        for i in h:
            lc2 = self.simulator.simulate(s, i)
            lc2 = lc2.shift(-lc2.time[0] + lc.time[0])
            outputs.append(lc2)

        cross = [Crossspectrum(lc, lc2).rebin(0.0075) for lc2 in outputs]
        lags = [np.angle(c.power) / (2 * np.pi * c.freq) for c in cross]

        v_cutoff = 1.0/(2.0*5)
        h_cutoffs = [lag[int((v_cutoff-0.0075)*1/0.0075)] for lag in lags]

        assert np.abs(5-h_cutoffs[0]) < np.sqrt(5)
        assert np.abs(5-h_cutoffs[1]) < np.sqrt(5)

    def test_powerspectrum(self):
        """
        Create a powerspectrum from light curve.
        """
        lc = self.simulator.simulate(2)
        self.simulator.powerspectrum(lc)

    def test_io(self):
        sim = simulator.Simulator(N=1024)
        sim.write('sim.pickle')
        sim = sim.read('sim.pickle')
        assert sim.N == 1024
        os.remove('sim.pickle')

    def test_io_with_unsupported_format(self):
        sim = simulator.Simulator(N=1024)
        with pytest.raises(KeyError):
            sim.write('sim.hdf5', format_='hdf5')
        with pytest.raises(KeyError):
            sim.write('sim.pickle', format_='pickle')
            sim.read('sim.pickle', format_='hdf5')
        os.remove('sim.pickle')
