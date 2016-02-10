__all__ = ["Crossspectrum", "AveragedCrossspectrum"]

import numpy as np
import scipy
import scipy.stats
import scipy.fftpack
import scipy.optimize

import stingray.lightcurve as lightcurve
import stingray.utils as utils


class Crossspectrum(object):

    def __init__(self, lc_1=None, lc_2=None, norm='none'):
        """
        Make a cross spectrum from a (binned) light curve.
        You can also make an empty Crossspectrum object to populate with your
        own fourier-transformed data (this can sometimes be useful when making
        binned periodograms).

        Parameters
        ----------
        lc_1: lightcurve.Lightcurve object, optional, default None
            The first light curve data for the channel/band of interest.
            
        lc_2: lightcurve.Lightcurve object, optional, default None
            The light curve data for the reference band.

        norm: {'frac', 'abs', 'leahy', 'none'}, default 'none'
            The normalization of the (real part of the) cross spectrum.

        Attributes
        ----------
        freq: numpy.ndarray
            The array of mid-bin frequencies that the Fourier transform samples

        cs: numpy.ndarray
            The array of cross spectra (complex numbers)

        df: float
            The frequency resolution

        m: int
            The number of averaged cross-spectra amplitudes in each bin.

        n: int
            The number of data points/time bins in one segment of the light 
            curves.

        nphots_1: float
            The total number of photons in light curve 1
            
		nphots_2: float
            The total number of photons in light curve 2
        """

        assert isinstance(norm, str), "norm is not a string!"

        assert norm.lower() in ["frac", "abs", "leahy", "none"], \
                "norm must be 'frac', 'abs', 'leahy', or 'none'!"

        self.norm = norm.lower()

        ## check if input data is a Lightcurve object, if not make one or
        ## make an empty Crossspectrum object if lc_1 == None or lc_2 == None
        ## TODO: handle only one lc input better?
        if lc_1 is not None and lc_2 is not None:
            pass

        elif lc_1 is not None and lc_2 is None:
            print("lc_2 is None. You can't do a cross spectrum with just one "\
                  "light curve!")
            self.freq = None
            self.cs = None
            self.df = None
            self.nphots_1 = None
            self.nphots_2 = None
            self.m = 1
            self.n = None

            return

        elif lc_2 is not None and lc_1 is None:
            print("lc_1 is None. You can't do a cross spectrum with just one "\
                  "light curve!")
            self.freq = None
            self.cs = None
            self.df = None
            self.nphots_1 = None
            self.nphots_2 = None
            self.m = 1
            self.n = None

            return

        else:
            self.freq = None
            self.cs = None
            self.df = None
            self.nphots_1 = None
            self.nphots_2 = None
            self.m = 1
            self.n = None

            return

        self._make_crossspectrum(lc_1, lc_2)

    def _make_crossspectrum(self, lc_1, lc_2):

        ## make sure the inputs work!
        assert isinstance(lc_1, lightcurve.Lightcurve), \
                        "lc_1 must be a lightcurve.Lightcurve object!"
        assert isinstance(lc_2, lightcurve.Lightcurve), \
                        "lc_2 must be a lightcurve.Lightcurve object!"


        ## total number of photons is the sum of the
        ## counts in the light curve
        self.nphots_1 = np.sum(lc_1.counts)
        self.nphots_2 = np.sum(lc_2.counts)


        ## the number of data points in the light curve
        assert lc_1.counts.shape[0] == lc_2.counts.shape[0], \
        		"Light curves do not have same number of time bins per segment."
        self.n = lc_1.counts.shape[0]

        assert lc_1.tseg == lc_2.tseg, "Light curves do not have same tseg."
		
        ## the frequency resolution
        self.df = 1.0/lc_1.tseg

        ## the number of averaged periodograms in the final output
        ## This should *always* be 1 here
        self.m = 1

        ## make the actual Fourier transform and compute cross spectrum
        self.unnorm_cross = self._fourier_transform(lc_1, lc_2)

        ## If co-spectrum is desired, normalize here. Otherwise, get raw back
        ## with the imaginary part still intact.
        self.cs = self._normalize_crossspectrum(self.unnorm_cross, lc_1.tseg)
        
        ## make a list of frequencies to go with the cross spectrum
        self.freq = np.arange(self.cs.shape[0])*self.df + self.df/2.


    def _fourier_transform(self, lc_1, lc_2):
        """
        Fourier transform the two light curves, then compute the cross spectrum.
        computed as CS = lc_1 x lc_2* (where lc_2 is the one that gets 
        complex-conjugated)

        Parameters
        ----------
        lc_1: lightcurve.Lightcurve object
            One light curve to be Fourier transformed. Ths is the band of 
            interest or channel of interest.

        lc_2: lightcurve.Lightcurve object
            Another light curve to be Fourier transformed. This is the reference
            band.

        Returns
        -------
        fr: numpy.ndarray
            The squared absolute value of the Fourier amplitudes

        """
        fourier_1 = scipy.fftpack.fft(lc_1.counts) ### do Fourier transform 1
        fourier_2 = scipy.fftpack.fft(lc_2.counts) ### do Fourier transform 2
		
        cross = fourier_1[:self.n/2+1] * np.conj(fourier_2[:self.n/2+1])
        return cross


    def rebin(self, df, method="mean"):
        """
        Rebin the cross spectrum to a new frequency resolution df.

        Parameters
        ----------
        df: float
            The new frequency resolution

        Returns
        -------
        bin_cs = Crossspectrum object
            The newly binned cross spectrum
        """

        ## rebin cross spectrum to new resolution
        binfreq, bincs, step_size = utils.rebin_data(self.freq[1:],
                                                     self.cs[1:], df,
                                                     method=method)

        ## make an empty cross spectrum object
        bin_cs = Crossspectrum()

        ## store the binned periodogram in the new object
        bin_cs.freq = np.hstack([binfreq[0]-self.df, binfreq])
        bin_cs.cs = np.hstack([self.cs[0], bincs])
        bin_cs.df = df
        bin_cs.n = self.n
        bin_cs.norm = self.norm
        bin_cs.nphots_1 = self.nphots_1
        bin_cs.nphots_2 = self.nphots_2
        bin_cs.m = int(step_size)

        return bin_cs

    def _normalize_crossspectrum(self, unnorm_cs, tseg):
        """
        Normalize the real part of the cross spectrum to Leahy, absolute rms^2,
        fractional rms^2 normalization, or not at all.

        Parameters
        ----------
        unnorm_cs: numpy.ndarray
            The unnormalized cross spectrum.

        tseg: int
            The length of the Fourier segment, in seconds.

        Returns
        -------
        cs: numpy.nd.array
            The normalized co-spectrum (real part of the cross spectrum). For
            'none' normalization, imaginary part is returned as well.
        """

        ## The "effective" count rate is the geometrical mean of the count rates
        ## of the two light curves; need to divide by tseg to have count rate
        actual_mean = np.sqrt(self.nphots_1 * self.nphots_2 / tseg)
        # print("Actual mean:", actual_mean)

        if self.norm.lower() == 'leahy':
            c = unnorm_cs.real
            cs = c * 2. / actual_mean

        elif self.norm.lower() == 'frac':
            c = unnorm_cs.real / np.float(self.n**2.)
            cs = c * 2. * tseg / (actual_mean**2.0)

        elif self.norm.lower() == 'abs':
            c = unnorm_cs.real / np.float(self.n**2.)
            cs = c * (2. * tseg)

        elif self.norm.lower() == 'none':
            cs = unnorm_cs

        else:
            raise Exception("Normalization not recognized!")

        return cs


class AveragedCrossspectrum(Crossspectrum):

    def __init__(self, lc_1, lc_2, segment_size=1, norm='none'):
        """
        Make an averaged cross spectrum from a light curve by segmenting two 
        light curves, Fourier-transforming each segment and then averaging the
        resulting cross spectra.
        
        Parameters
        ----------
        lc_1: lightcurve.Lightcurve object OR
            iterable of lightcurve.Lightcurve objects
            One light curve data to be Fourier-transformed. This is the band
            of interest or channel of interest.

        lc_2: lightcurve.Lightcurve object OR
            iterable of lightcurve.Lightcurve objects
            Second light curve data to be Fourier-transformed. This is the 
            reference band.
            
        segment_size: float, default 1
            The size of each segment to average. Note that if the total duration
            of each Lightcurve object in lc_1 or lc_2 is not an integer multiple
            of the segment_size, then any fraction left-over at the end of the
            time series will be lost. Otherwise you introduce artefacts.

        norm: {'frac', 'abs', 'leahy', 'none'}, default 'none'
            The normalization of the (real part of the) cross spectrum.

        Attributes
        ----------
        freq: numpy.ndarray
            The array of mid-bin frequencies that the Fourier transform samples

        cs: numpy.ndarray
            The array of cross spectra

        df: float
            The frequency resolution

        m: int
            The number of averaged cross spectra

        n: int
            The number of time bins per segment of light curve?
            
        nphots_1: float
            The total number of photons in the first (interest) light curve
        
        nphots_2: float
            The total number of photons in the second (reference) light curve

        """

        assert isinstance(norm, str), "norm is not a string!"

        assert norm.lower() in ["frac", "abs", "leahy", "none"], \
                "norm must be 'frac', 'abs', 'leahy', or 'none'!"

        self.norm = norm.lower()

        assert np.isfinite(segment_size), "segment_size must be finite!"

        self.segment_size = segment_size

        Crossspectrum.__init__(self, lc_1, lc_2, self.norm)

        return


    def _make_segment_csd(self, lc_1, lc_2, segment_size):
		
		
        ## TODO: need to update this for making cross spectra.
        assert isinstance(lc_1, lightcurve.Lightcurve)
        assert isinstance(lc_2, lightcurve.Lightcurve)

        assert lc_1.dt == lc_2.dt, \
        		"Light curves do not have same time binning dt."
        assert lc_1.tseg == lc_2.tseg, "Lightcurves do not have same tseg."

        ## number of bins per segment
        nbins = int(segment_size/lc_1.dt)

        start_ind = 0
        end_ind = nbins

        cs_all = []
        nphots_1_all = []
        nphots_2_all = []

        while end_ind <= lc_1.counts.shape[0]:
            time_1 = lc_1.time[start_ind:end_ind]
            counts_1 = lc_1.counts[start_ind:end_ind]
            time_2 = lc_2.time[start_ind:end_ind]
            counts_2 = lc_2.counts[start_ind:end_ind]
            lc_1_seg = lightcurve.Lightcurve(time_1, counts_1)
            lc_2_seg = lightcurve.Lightcurve(time_2, counts_2)
            cs_seg = Crossspectrum(lc_1_seg, lc_2_seg, norm=self.norm)
            cs_all.append(cs_seg)
            nphots_1_all.append(np.sum(lc_1_seg.counts))
            nphots_2_all.append(np.sum(lc_2_seg.counts))
            start_ind += nbins
            end_ind += nbins

        return cs_all, nphots_1_all, nphots_2_all

    def _make_crossspectrum(self, lc_1, lc_2):

        ## chop light curves into segments
        if isinstance(lc_1, lightcurve.Lightcurve) and isinstance(lc_2, lightcurve.Lightcurve):
            cs_all, nphots_1_all, nphots_2_all = self._make_segment_csd(lc_1, 
            										lc_2, self.segment_size)
        else:
            cs_all, nphots_1_all, nphots_2_all = [], [], []
            ## TODO: should be using izip from iterables if lc_1 or lc_2 could
            ## be long
            for lc_1_seg, lc_2_seg in zip(lc_1, lc_2):
                cs_sep, nphots_1_sep, nphots_2_sep = self._make_segment_csd(lc_1_seg, lc_2_seg,
                                                            self.segment_size)

                cs_all.append(cs_sep)
                nphots_1_all.append(nphots_1_sep)
                nphots_2_all.append(nphots_2_sep)


            cs_all = np.hstack(cs_all)
            nphots_1_all = np.hstack(nphots_1_all)
            nphots_2_all = np.hstack(nphots_2_all)


        m = len(cs_all)
        nphots_1 = np.mean(nphots_1_all)
        nphots_2 = np.mean(nphots_2_all)

        cs_avg = np.zeros_like(cs_all[0].cs)
        for cs in cs_all:
            cs_avg += cs.cs

        cs_avg /= np.float(m)

        self.freq = cs_all[0].freq
        self.cs = cs_avg
        self.m = m
        self.df = cs_all[0].df
        self.n = cs_all[0].n
        self.nphots_1 = nphots_1
        self.nphots_2 = nphots_2

