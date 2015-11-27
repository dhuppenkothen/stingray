__all__ = ["Powerspectrum", "AveragedPowerspectrum"]

import numpy as np
import scipy
import scipy.stats
import scipy.fftpack
import scipy.optimize

import stingray.lightcurve as lightcurve
import stingray.utils as utils


class Powerspectrum(object):

    def __init__(self, lc=None, norm='rms'):
        """
        Make a Periodogram (power spectrum) from a (binned) light curve.
        Periodograms can be Leahy normalized or fractional rms normalized.
        You can also make an empty Periodogram object to populate with your
        own fourier-transformed data (this can sometimes be useful when making
        binned periodograms).

        Parameters
        ----------
        lc: lightcurve.Lightcurve object, optional, default None
            The light curve data to be Fourier-transformed.

        norm: {"leahy" | "rms"}, optional, default "rms"
            The normaliation of the periodogram to be used. Options are
            "leahy" or "rms", default is "rms".


        Attributes
        ----------
        norm: {"leahy" | "rms"}
            the normalization of the periodogram

        freq: numpy.ndarray
            The array of mid-bin frequencies that the Fourier transform samples

        ps: numpy.ndarray
            The array of normalized squared absolute values of Fourier
            amplitudes

        df: float
            The frequency resolution

        m: int
            The number of averaged periodograms

        n: int
            The number of data points in the light curve

        nphots: float
            The total number of photons in the light curve


        """
        assert isinstance(norm, str), "norm is not a string!"

        assert norm.lower() in ["rms", "leahy"], \
                "norm must be either 'rms' or 'leahy'!"

        self.norm = norm.lower()

        ## check if input data is a Lightcurve object, if not make one or
        ## make an empty Periodogram object if lc == time == counts == None
        if lc is not None:
            pass
        else:
            self.freq = None
            self.ps = None
            self.df = None
            self.nphots = None
            self.m = 1
            self.n = None
            return

        self._make_powerspectrum(lc)

    def _make_powerspectrum(self, lc):

        ## make sure my inputs work!
        assert isinstance(lc, lightcurve.Lightcurve), \
                        "lc must be a lightcurve.Lightcurve object!"


        ## total number of photons is the sum of the
        ## counts in the light curve
        self.nphots = np.sum(lc.counts)

        ## the number of data points in the light curve
        self.n = lc.counts.shape[0]

        ## the frequency resolution
        self.df = 1.0/lc.tseg

        ## the number of averaged periodograms in the final output
        ## This should *always* be 1 here
        self.m = 1

        ## make the actual Fourier transform
        self.unnorm_powers = self._fourier_transform(lc)

        ## normalize to either Leahy or rms normalization
        self.ps = self._normalize_periodogram(self.unnorm_powers, lc)

        ## make a list of frequencies to go with the powers
        self.freq = np.arange(self.ps.shape[0])*self.df + self.df/2.


    def _fourier_transform(self, lc):
        """
        Fourier transform the light curve, then square the
        absolute value of the Fourier amplitudes.

        Parameters
        ----------
        lc: lightcurve.Lightcurve object
            The light curve to be Fourier transformed

        Returns
        -------
        fr: numpy.ndarray
            The squared absolute value of the Fourier amplitudes

        """
        fourier= scipy.fftpack.fft(lc.counts) ### do Fourier transform
        fr = np.abs(fourier[:self.n/2+1])**2.
        return fr

    def _normalize_periodogram(self, unnorm_powers, lc):
        """
        Normalize the periodogram to either Leahy or RMS normalization.
        In Leahy normalization, the periodogram is normalized in such a way
        that a flat light curve of Poissonian data will make a realization of
        the power spectrum in which the powers are distributed as Chi^2 with
        two degrees of freedom (with a mean of 2 and a variance of 4).

        In rms normalization, the periodogram will be normalized such that
        the integral of the periodogram will equal the total variance in the
        light curve divided by the mean of the light curve squared.

        Parameters
        ----------
        unnorm_powers: numpy.ndarray
            The squared absolute value of the Fourier amplitudes

        lc: lightcurve.Lightcurve object
            The input light curve


        Returns
        -------
        ps: numpy.nd.array
            The normalized periodogram
        """
        if self.norm.lower() == 'leahy':
            p = unnorm_powers
            ps =  2.*p/self.nphots

        elif self.norm.lower() == 'rms':
            p = unnorm_powers/np.float(self.n**2.)
            ps = p*2.*lc.tseg/(np.mean(lc.counts)**2.0)

        else:
            raise Exception("Normalization not recognized!")

        return ps

    def rebin(self, df, method="mean"):
        """
        Rebin the periodogram to a new frequency resolution df.

        Parameters
        ----------
        df: float
            The new frequency resolution

        Returns
        -------
        bin_ps = Periodogram object
            The newly binned periodogram
        """

        ## rebin power spectrum to new resolution
        binfreq, binps, step_size = utils.rebin_data(self.freq[1:],
                                                     self.ps[1:], df,
                                                     method=method)

        ## make an empty periodogram object
        bin_ps = Powerspectrum()

        ## store the binned periodogram in the new object
        bin_ps.norm = self.norm
        bin_ps.freq = np.hstack([binfreq[0]-self.df, binfreq])
        bin_ps.ps = np.hstack([self.ps[0], binps])
        bin_ps.df = df
        bin_ps.n = self.n
        bin_ps.nphots = self.nphots
        bin_ps.m = int(step_size)

        return bin_ps


    def rebin_log(self, f=0.01):
        """
        Logarithmic rebin of the periodogram.
        The new frequency depends on the previous frequency
        modified by a factor f:

        dnu_j = dnu_{j-1}*(1+f)

        Parameters
        ----------
        f: float, optional, default 0.01
            parameter that steers the frequency resolution


        Returns
        -------
        binfreq: numpy.ndarray
            the binned frequencies

        binps: numpy.ndarray
            the binned powers

        nsamples: numpy.ndarray
            the samples of the original periodogramincluded in each
            frequency bin
        """

        minfreq = self.freq[1]*0.5 ## frequency to start from
        maxfreq = self.freq[-1] ## maximum frequency to end
        binfreq = [minfreq, minfreq + self.df] ## first
        df = self.freq[1] ## the frequency resolution of the first bin

        ## until we reach the maximum frequency, increase the width of each
        ## frequency bin by f
        while binfreq[-1] <= maxfreq:
            binfreq.append(binfreq[-1] + df*(1.+f))
            df = binfreq[-1]-binfreq[-2]

        ## compute the mean of the powers that fall into each new frequency
        ## bin
        binps, bin_edges, binno = scipy.stats.binned_statistic(self.freq,
                                                               self.ps,
                                                               statistic="mean",
                                                               bins=binfreq)

        ## compute the number of powers in each frequency bin
        nsamples = np.array([len(binno[np.where(binno == i)[0]]) \
                             for i in xrange(np.max(binno))])

        ## the frequency resolution
        df = np.diff(binfreq)

        ## shift the lower bin edges to the middle of the bin and drop the
        ## last right bin edge
        binfreq = binfreq[:-1]+df/2.
        
        return binfreq, binps, nsamples

    def compute_rms(self, min_freq, max_freq):
        """
        Compute the fractional rms amplitude in the periodgram
        between two frequencies.

        Parameters
        ----------
        min_freq: float
            The lower frequency bound for the calculation

        max_freq: float
            The upper frequency bound for the calculation


        Returns
        -------
        rms: float
            The fractional rms amplitude contained between min_freq and
            max_freq

        """
        #assert min_freq >= self.freq[0], "Lower frequency bound must be " \
        #                                 "larger or equal the minimum " \
        #                                 "frequency in the periodogram!"

        #assert max_freq <= self.freq[-1], "Upper frequency bound must be " \
        #                                 "smaller or equal the maximum " \
        #                                 "frequency in the periodogram!"

        minind = self.freq.searchsorted(min_freq)
        maxind = self.freq.searchsorted(max_freq)
        powers = self.ps[minind:maxind]
        if self.norm.lower() == 'leahy':
            rms = np.sqrt(np.sum(powers)/self.nphots)
        elif self.norm.lower() == "rms":
            rms = np.sqrt(np.sum(powers*self.df))
        else:
            raise Exception("Normalization not recognized!")

        rms_err = self._rms_error(powers)

        return rms, rms_err

    def _rms_error(self, powers):
        """
        Compute the error on the fractional rms amplitude using error
        propagation.
        Note: this uses the actual measured powers, which is not
        strictly correct. We should be using the underlying power spectrum,
        but in the absence of an estimate of that, this will have to do.

        Parameters
        ----------
        powers: iterable
            The list of powers used to compute the fractional rms amplitude.


        Returns
        -------
        delta_rms: float
            the error on the fractional rms amplitude
        """
        p_err = scipy.stats.chi2(2.*self.m).var()*powers/self.m
        drms_dp = 1./(2.*np.sqrt(np.sum(powers)*self.df))
        delta_rms = np.sum(p_err*drms_dp*self.df)
        return delta_rms

class AveragedPowerspectrum(Powerspectrum):

    def __init__(self, lc, segment_size, norm="rms"):
        """
        Make an averaged periodogram from a light curve by segmenting the light
        curve, Fourier-transforming each segment and then averaging the
        resulting periodograms.
        Parameters
        ----------
        lc: lightcurve.Lightcurve object OR
            iterable of lightcurve.Lightcurve objects
            The light curve data to be Fourier-transformed.

        segment_size: float
            The size of each segment to average. Note that if the total duration
            of each Lightcurve object in lc is not an integer multiple of the
            segment_size, then any fraction left-over at the end of the
            time series will be lost.

        norm: {"leahy" | "rms"}, optional, default "rms"
            The normaliation of the periodogram to be used. Options are
            "leahy" or "rms", default is "rms".


        Attributes
        ----------
        norm: {"leahy" | "rms"}
            the normalization of the periodogram

        freq: numpy.ndarray
            The array of mid-bin frequencies that the Fourier transform samples

        ps: numpy.ndarray
            The array of normalized squared absolute values of Fourier
            amplitudes

        df: float
            The frequency resolution

        m: int
            The number of averaged periodograms

        n: int
            The number of data points in the light curve

        nphots: float
            The total number of photons in the light curve


        """


        assert np.isfinite(segment_size), "segment_size must be finite!"

        self.norm = norm.lower()
        self.segment_size = segment_size

        Powerspectrum.__init__(self, lc, norm)

        return


    def _make_segment_psd(self, lc, segment_size):

        assert isinstance(lc, lightcurve.Lightcurve)

        ## number of bins per segment
        nbins = int(segment_size/lc.dt)

        start_ind = 0
        end_ind = nbins

        ps_all = []
        nphots_all = []
        while end_ind <= lc.counts.shape[0]:
            time = lc.time[start_ind:end_ind]
            counts = lc.counts[start_ind:end_ind]
            lc_seg = lightcurve.Lightcurve(time, counts)
            ps_seg = Powerspectrum(lc_seg, norm=self.norm)
            ps_all.append(ps_seg)
            nphots_all.append(np.sum(lc_seg.counts))
            start_ind += nbins
            end_ind += nbins

        return ps_all, nphots_all

    def _make_powerspectrum(self, lc):

        ## chop light curves into segments
        if isinstance(lc, lightcurve.Lightcurve):
            ps_all, nphots_all = self._make_segment_psd(lc,
                                                        self.segment_size)
        else:
            ps_all, nphots_all = [], []
            for lc_seg in lc:
                ps_sep, nphots_sep = self._make_segment_psd(lc_seg,
                                                            self.segment_size)

                ps_all.append(ps_sep)
                nphots_all.append(nphots_sep)

            ps_all = np.hstack(ps_all)
            nphots_all = np.hstack(nphots_all)

        m = len(ps_all)
        nphots = np.mean(nphots_all)
        ps_avg = np.zeros_like(ps_all[0].ps)
        for ps in ps_all:
            ps_avg += ps.ps

        ps_avg /= np.float(m)

        self.freq = ps_all[0].freq
        self.ps = ps_avg
        self.m = m
        self.df = ps_all[0].df
        self.n = ps_all[0].n
        self.nphots = nphots
