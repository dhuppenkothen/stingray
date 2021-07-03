import copy

from scipy.signal.spectral import lombscargle

from stingray.gti import check_gtis, cross_two_gtis
from stingray.crossspectrum import Crossspectrum, normalize_crossspectrum
from stingray.crossspectrum import normalize_crossspectrum_gauss
from stingray.powerspectrum import Powerspectrum
import warnings

import numpy as np
import scipy
import scipy.optimize
import scipy.stats
from scipy import signal, interpolate
import nfft

from .events import EventList
from .lightcurve import Lightcurve
from .utils import rebin_data, simon

__all__ = [
    "Multitaper"
]

# Inspired from nitime (https://nipy.org/nitime/)


class Multitaper(Powerspectrum):
    """
    Class to calculate the multitaper periodogram from lightcurve data.

    Parameters
    ----------
    data: :class:`stingray.Lightcurve` object, optional, default ``None``
        The light curve data to be Fourier-transformed.

    norm: {``leahy`` | ``frac`` | ``abs`` | ``none`` }, optional, default ``frac``
        The normaliation of the power spectrum to be used. Options are
        ``leahy``, ``frac``, ``abs`` and ``none``, default is ``frac``.

    NW: float, optional, default ``4``
        The normalized half-bandwidth of the data tapers, indicating a
        multiple of the fundamental frequency of the DFT (Fs/N).
        Common choices are n/2, for n >= 4.

    adaptive: boolean, optional, default ``False``
        Use an adaptive weighting routine to combine the PSD estimates of
        different tapers.

    jackknife: boolean, optional, default ``True``
        Use the jackknife method to make an estimate of the PSD variance
        at each point.

    low_bias: boolean, optional, default ``True``
        Rather than use 2NW tapers, only use the tapers that have better than
        90% spectral concentration within the bandwidth (still using
        a maximum of 2NW tapers)

    Other Parameters
    ----------------
    gti: 2-d float array
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care!

    Attributes
    ----------
    norm: {``leahy`` | ``frac`` | ``abs`` | ``none`` }
        the normalization of the power spectrun

    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples

    power: numpy.ndarray
        The array of normalized squared absolute values of Fourier
        amplitudes

    unnorm_power: numpy.ndarray
        The array of unnormalized values of Fourier amplitudes

    multitaper_norm_power: numpy.ndarray
        The array of normalized values of Fourier amplitudes, normalized
        according to the scheme followed in nitime, that is, by the length and
        the sampling frequency.

    power_err: numpy.ndarray
        The uncertainties of ``power``.
        An approximation for each bin given by ``power_err= power/sqrt(m)``.
        Where ``m`` is the number of power averaged in each bin (by frequency
        binning, or averaging power spectrum). Note that for a single
        realization (``m=1``) the error is equal to the power.

    df: float
        The frequency resolution

    m: int
        The number of averaged powers in each bin

    n: int
        The number of data points in the light curve

    nphots: float
        The total number of photons in the light curve

    jk_var_deg_freedom: numpy.ndarray
        Array differs depending on whether
        the jackknife was used. It is either
        * The jackknife estimated variance of the log-psd, OR
        * The degrees of freedom in a chi2 model of how the estimated
          PSD is distributed about the true log-PSD (this is either
          2*floor(2*NW), or calculated from adaptive weights)

    Notes
    -----
    The bandwidth of the windowing function will determine the number of
    tapers to use. This parameter (NW) represents trade-off between frequency
    resolution (lower main lobe BW for the taper) and variance reduction
    (higher BW and number of averaged estimates). Typically, the number of
    tapers is calculated as 2x the bandwidth-to-fundamental-frequency
    ratio (NW), as these eigenfunctions have the best energy concentration.

    """

    def __init__(self, data=None, norm="frac", gti=None, dt=None, lc=None,
                 NW=4, adaptive=False, jackknife=True, low_bias=True,
                 lombscargle=False):

        if lc is not None:
            warnings.warn("The lc keyword is now deprecated. Use data "
                          "instead", DeprecationWarning)
        if data is None:
            data = lc

        if isinstance(norm, str) is False:
            raise TypeError("norm must be a string")

        if norm.lower() not in ["frac", "abs", "leahy", "none"]:
            raise ValueError("norm must be 'frac', 'abs', 'leahy', or 'none'!")

        self.norm = norm.lower()

        if isinstance(data, EventList) and dt is None:
            raise ValueError(
                "If using event lists, please specify "
                "the bin time to generate lightcurves.")

        if data is None:
            self.freq = None
            self.power = None
            self.multitaper_norm_power = None
            self.eigvals = None
            self.power_err = None
            self.df = None
            self.m = 1
            self.n = None
            self.nphots = None
            self.jk_var_deg_freedom = None
            return
        elif not isinstance(data, EventList):
            lc = data
        else:
            lc = data.to_lc(dt)

        self.gti = gti
        self.lc = lc
        self.power_type = 'real'
        self.fullspec = False

        self._make_multitaper_periodogram(lc, NW=NW, adaptive=adaptive,
                                          jackknife=jackknife,
                                          low_bias=low_bias,
                                          lombscargle=lombscargle)

    def _make_multitaper_periodogram(self, lc, NW=4, adaptive=False,
                                     jackknife=True, low_bias=True,
                                     lombscargle=False):
        """Compute the normalized multitaper spectral estimate.

        This includes checking for the presence of and applying Good Time Intervals,
        computing the a nitime inspired normalized power spectrum, unnormalizing it,
        and then renormalizing it using the required normalization.

        Parameters
        ----------
        lc : :class:`stingray.Lightcurve` objects
            Two light curves used for computing the cross spectrum.

        NW: float, optional, default ``4``
            The normalized half-bandwidth of the data tapers, indicating a
            multiple of the fundamental frequency of the DFT (Fs/N).
        Common choices are n/2, for n >= 4.

        adaptive: boolean, optional, default ``False``
            Use an adaptive weighting routine to combine the PSD estimates of
            different tapers.

        jackknife: boolean, optional, default ``True``
            Use the jackknife method to make an estimate of the PSD variance
            at each point.

        low_bias: boolean, optional, default ``True``
            Rather than use 2NW tapers, only use the tapers that have better than
            90% spectral concentration within the bandwidth (still using
            a maximum of 2NW tapers)
        """

        if not isinstance(lc, Lightcurve):
            raise TypeError("lc must be a lightcurve.Lightcurve object")

        if self.gti is None:
            self.gti = cross_two_gtis(lc.gti, lc.gti)

        check_gtis(self.gti)

        if self.gti.shape[0] != 1:
            raise TypeError("Non-averaged Spectra need "
                            "a single Good Time Interval")

        lc = lc.split_by_gti()[0]

        self.meancounts = lc.meancounts
        self.nphots = np.float64(np.sum(lc.counts))

        self.err_dist = 'poisson'
        if lc.err_dist == 'poisson':
            self.var = lc.meancounts
        else:
            self.var = np.mean(lc.counts_err) ** 2
            self.err_dist = 'gauss'

        self.dt = lc.dt
        self.n = lc.n

        # the frequency resolution
        self.df = 1.0 / lc.tseg

        # the number of averaged periodograms in the final output
        # This should *always* be 1 here
        self.m = 1

        if lombscargle:
            self.freq, self.multitaper_norm_power = \
                self._multitaper_lomb_scargle(lc, NW=NW, low_bias=low_bias)

            # Same for the timebeing until normalization discrepancy is resolved
            self.unnorm_power = self.multitaper_norm_power

        else:
            self.freq, self.multitaper_norm_power = \
                self._fourier_multitaper(lc, NW=NW, adaptive=adaptive,
                                         jackknife=jackknife, low_bias=low_bias)

            self.unnorm_power = self.multitaper_norm_power * lc.n / lc.dt

        self.power = \
            self._normalize_multitaper(self.unnorm_power, lc.tseg)

        if lc.err_dist.lower() != "poisson":
            simon("Looks like your lightcurve statistic is not poisson."
                  "The errors in the Powerspectrum will be incorrect.")

        self.power_err = self.power / np.sqrt(self.m)

    def _fourier_multitaper(self, lc, NW=4, adaptive=False,
                            jackknife=True, low_bias=True):
        """Compute the multitaper periodogram.

        Auxiliary method to apply the multitaper algorithm by first computing
        the DPSS windows, then fourier transforming the data tapered with these
        DPSS windows, and then calculating the power spectral density using a
        desired scheme (a combination of adaptive and low_bias parameters).
        Also set some of the multitaper specific attributes.

        Parameters
        ----------
        lc : :class:`stingray.Lightcurve` objects
            Two light curves used for computing the cross spectrum.

        NW: float, optional, default ``4``
            The normalized half-bandwidth of the data tapers, indicating a
            multiple of the fundamental frequency of the DFT (Fs/N).
            Common choices are n/2, for n >= 4.

        adaptive: boolean, optional, default ``False``
            Use an adaptive weighting routine to combine the PSD estimates of
            different tapers.

        jackknife: boolean, optional, default ``True``
            Use the jackknife method to make an estimate of the PSD variance
            at each point.

        low_bias: boolean, optional, default ``True``
            Rather than use 2NW tapers, only use the tapers that have better than
            90% spectral concentration within the bandwidth (still using
            a maximum of 2NW tapers)

        Returns
        -------
        freq_multitaper: numpy.ndarray
            The frequency mid-bins of the PSD amplitudes

        psd_multitaper: numpy.ndarray
            The value of the PSD amplitudes at the given frequency mid-bins
        """

        if NW < 0.5:
            raise ValueError("The value of normalized half-bandwidth "
                             "should be greater than 0.5")

        Kmax = int(2 * NW)

        dpss_tapers, eigvals = \
            signal.windows.dpss(M=lc.n, NW=NW, Kmax=Kmax,
                                sym=False, return_ratios=True)

        if low_bias:
            selected_tapers = (eigvals > 0.9)
            if not selected_tapers.any():
                simon("Could not properly use low_bias, "
                      "keeping the lowest-bias taper")
                selected_tapers = [np.argmax(eigvals)]

            eigvals = eigvals[selected_tapers]
            dpss_tapers = dpss_tapers[selected_tapers, :]

        print(f"Using {len(eigvals)} DPSS windows for "
              "multitaper spectrum estimator")

        data_multitaper = lc.counts - np.mean(lc.counts)  # De-mean
        data_multitaper = np.tile(data_multitaper, (len(eigvals), 1))
        data_multitaper = np.multiply(data_multitaper, dpss_tapers)

        freq_response = scipy.fft.rfft(data_multitaper, n=lc.n)

        # Adjust DC and maybe Nyquist, depending on one-sided transform
        freq_response[..., 0] /= np.sqrt(2.)
        if lc.n % 2 == 0:
            freq_response[..., -1] /= np.sqrt(2.)

        freq_multitaper = scipy.fft.rfftfreq(lc.n, d=lc.dt)

        if adaptive:
            psd_multitaper, weights_multitaper = \
                self._get_adaptive_psd(freq_response, eigvals)
        else:
            weights_multitaper = np.sqrt(eigvals)[:, np.newaxis]
            psd_multitaper = \
                self.psd_from_freq_response(freq_response, weights_multitaper)

        psd_multitaper *= lc.dt  # /= sampling_freq
        self.eigvals = eigvals

        if jackknife:
            self.jk_var_deg_freedom = \
                self.jackknifed_sdf_variance(freq_response, eigvals, adaptive)
        else:
            if adaptive:
                self.jk_var_deg_freedom = \
                    2 * (weights_multitaper ** 2).sum(axis=-2)
            else:
                self.jk_var_deg_freedom = np.full((len(freq_multitaper)), 2 * len(eigvals))

        return freq_multitaper, psd_multitaper

    def psd_from_freq_response(self, freq_response, weights):
        """Calculate the weighted PSD from the Fourier transformed data.

        Calculate the weighted PSD from the Fourier transformed data by
        combining the frequences responses of these tapered data using the given
        weights.

        Parameters
        ----------
        freq_response: numpy.ndarray
            The frequency responses or the Fourier transforms of the tapered data

        weights: numpy.ndarray
            The weights to combine the different Fourier transforms of the data
            tapered with different windows to produce the weighted PSD.

        Returns
        -------
        psd: numpy.ndarray
            The weighted power spectral density of the provided Fourier
            transformed data.

        Notes
        -----
        The shape of ``freq_response`` and ``weights`` must either be same or
        broadcastable.
        """

        psd = freq_response * weights
        psd *= psd.conj()
        psd = psd.real.sum(axis=-2)  # Sum all rows
        psd *= 2 / (weights * weights.conj()).real.sum(axis=-2)
        return psd

    def _get_adaptive_psd(self, freq_response, eigvals, max_iter=150):
        r"""Compute the PSD from tapered data.

        Performs an iterative procedure to compute the PSD from tapered data
        by finding the optimal weights for the direct spectral estimators
        of the DPSS tapered signals.

        Parameters
        ----------
        freq_response: numpy.ndarray
            The frequency responses or the Fourier transforms of the tapered data

        eigvals: numpy.ndarray
            The eigenvalues of the DPSS tapers

        max_iter: int, optional, default ``150``
            Maximum number of iterations for weight computation

        Returns
        -------
        psd : numpy.ndarray
            The adaptively computed PSD

        weights : numpy.ndarray
            The weights used to combine the tapered spectra

        Notes
        -----
        The weights to use for making the multitaper estimate, such that
        :math:`S_{mt} = \sum_{k} |w_k|^2S_k^{mt} / \sum_{k} |w_k|^2`

        If the number of tapers are less than 3, no adaptive weights are
        calculated and the square roots of eigenvalues are used as weights
        to produce the weighted PSD
        """

        n_tapers = len(eigvals)
        n_freqs = freq_response.shape[-1]

        sqrt_eigvals = np.sqrt(eigvals)

        if n_tapers < 3:
            simon("Not adaptively combining, number of tapers < 3")
            weights = sqrt_eigvals[:, np.newaxis]
            return self.psd_from_freq_response(freq_response, weights), weights

        psd_est = \
            self.psd_from_freq_response(freq_response, sqrt_eigvals[:, np.newaxis])

        var = np.trapz(psd_est, dx=np.pi / n_freqs) / (2 * np.pi)
        del psd_est

        psd = np.empty(n_freqs)

        weights = np.empty((n_tapers, n_freqs))

        # combine the SDFs in the traditional way in order to estimate
        # the variance of the timeseries

        # The process is to iteratively switch solving for the following
        # two expressions:
        # (1) Adaptive Multitaper SDF:
        # S^{mt}(f) = [ sum |d_k(f)|^2 S_k(f) ]/ sum |d_k(f)|^2
        #
        # (2) Weights
        # d_k(f) = [sqrt(lam_k) S^{mt}(f)] / [lam_k S^{mt}(f) + E{B_k(f)}]
        #
        # Where lam_k are the eigenvalues corresponding to the DPSS tapers,
        # and the expected value of the broadband bias function
        # E{B_k(f)} is replaced by its full-band integration
        # (1/2pi) int_{-pi}^{pi} E{B_k(f)} = sig^2(1-lam_k)

        # start with an estimate from incomplete data--the first 2 tapers
        psd_iter = \
            self.psd_from_freq_response(freq_response[:2],
                                        sqrt_eigvals[:2, np.newaxis])

        err = np.zeros_like(freq_response)

        for ite in range(max_iter):
            d_k = (psd_iter / (eigvals[:, np.newaxis] *
                   psd_iter + (1 - eigvals[:, np.newaxis]) * var))
            d_k *= sqrt_eigvals[:, np.newaxis]
            # Test for convergence -- this is overly conservative, since
            # iteration only stops when all frequencies have converged.
            # A better approach is to iterate separately for each freq, but
            # that is a nonvectorized algorithm.
            # Take the RMS difference in weights from the previous iterate
            # across frequencies. If the maximum RMS error across freqs is
            # less than 1e-10, then we're converged

            err -= d_k
            if np.max(np.mean(err ** 2, axis=0)) < 1e-10:
                break

            # update the iterative estimate with this d_k
            psd_iter = self.psd_from_freq_response(freq_response, d_k)
            err = d_k
        if ite == max_iter - 1:
            simon('Iterative multi-taper PSD computation did not converge.')

        return psd_iter, d_k

    def _normalize_multitaper(self, unnorm_power, tseg):
        """Apply one of the allowed normalizations to the periodogram.

        Normalize the real part of the multitaper spectrum estimate to Leahy,
        absolute rms^2, fractional rms^2 normalization, or not at all.

        Parameters
        ----------
        unnorm_power: numpy.ndarray
            The unnormalized spectrum estimate.

        tseg: int
            The length of the Fourier segment, in seconds.

        Returns
        -------
        power: numpy.nd.array
            The normalized spectrum estimate (real part of the spectrum).
        """

        if self.err_dist == 'poisson':
            return normalize_crossspectrum(
                unnorm_power, tseg, self.n, self.nphots, self.nphots, self.norm,
                self.power_type)

        return normalize_crossspectrum_gauss(
            unnorm_power, np.sqrt(self.meancounts * self.meancounts),
            np.sqrt(self.var * self.var),
            dt=self.dt,
            N=self.n,
            norm=self.norm,
            power_type=self.power_type)

    def jackknifed_sdf_variance(self, freq_response, eigvals, adaptive):
        r"""Compute the variance of the spectrum through jack-knife.

        Returns the variance of the log-sdf estimated through jack-knifing
        a group of independent sdf estimates.

        Parameters
        ----------

        freq_response: numpy.ndarray
            The frequency responses or the Fourier transforms of the tapered data

        eigvals: numpy.ndarray
            The eigenvalues of the DPSS tapers

        adaptive: boolean
            Whether or not to use the adaptive weighting method to calculate
            the log-sdf through jack-knife method

        Returns
        -------

        var : The estimate for log-sdf variance

        Notes
        -----

        The jackknifed mean estimate is distributed about the true mean as
        a Student's t-distribution with (K-1) degrees of freedom, and
        standard error equal to sqrt(var). However, Thompson and Chave [1]
        point out that this variance better describes the sample mean.
        [1] Thomson D J, Chave A D (1991) Advances in Spectrum Analysis and Array
        Processing (Prentice-Hall, Englewood Cliffs, NJ), 1, pp 58-113.
        """

        K = len(eigvals)

        all_indices = set(range(K))
        jk_sdk = []

        for i in range(K):
            selected_indices = list(all_indices.difference([i]))
            selected_data = np.take(freq_response, selected_indices, axis=0)
            selected_eigvals = np.take(eigvals, selected_indices)

            if adaptive:
                selected_psd, selected_weights = \
                    self._get_adaptive_psd(selected_data, selected_eigvals)

            else:
                selected_weights = np.sqrt(selected_eigvals)[:, np.newaxis]
                selected_psd = \
                    self.psd_from_freq_response(selected_data, selected_weights)

            jk_sdk.append(selected_psd)

        jk_sdk = np.log(jk_sdk)
        jk_var = jk_sdk - jk_sdk.mean(axis=0)

        np.power(jk_var, 2, jk_var)
        jk_var = jk_var.sum(axis=0)

        # Thompson's recommended factor, eq 18
        # Jackknifing Multitaper Spectrum Estimates
        # IEEE SIGNAL PROCESSING MAGAZINE [20] JULY 2007
        K = float(K)
        f = (K - 1)**2 / K / (K - 0.5)
        jk_var *= f
        return jk_var

    def rebin(self, df=None, f=None, method="mean"):
        """Rebin the multitaper spectrum to a new frequency resolution ``df``.

        Parameters
        ----------
        df: float
            The new frequency resolution

        Other Parameters
        ----------------
        f: float
            the rebin factor. If specified, it substitutes df with ``f*self.df``

        Returns
        -------
        bin_cs = :class:`Multitaper` (or one of its subclasses) object
            The newly binned multitaper spectrum.
            Note: this object will be of the same type as the object
            that called this method. For example, if this method is called
            from :class:`AveragedPowerspectrum`, it will return an object of class
            :class:`AveragedPowerspectrum`, too.
        """

        if f is None and df is None:
            raise ValueError('You need to specify at least one between f and '
                             'df')
        elif f is not None:
            df = f * self.df

        # rebin cross spectrum to new resolution
        binfreq, binmtp, binerr, step_size = \
            rebin_data(self.freq, self.power, df, self.power_err,
                       method=method, dx=self.df)
        # make an empty cross spectrum object
        # note: syntax deliberate to work with subclass Powerspectrum
        bin_mtp = copy.copy(self)

        # store the binned periodogram in the new object
        bin_mtp.freq = binfreq
        bin_mtp.power = binmtp
        bin_mtp.eigvals = self.eigvals
        bin_mtp.df = df
        bin_mtp.n = self.n
        bin_mtp.norm = self.norm
        bin_mtp.nphots = self.nphots
        bin_mtp.power_err = binerr

        if hasattr(self, 'unnorm_power'):
            _, binpower_unnorm, _, _ = \
                rebin_data(self.freq, self.unnorm_power, df,
                           method=method, dx=self.df)

            bin_mtp.unnorm_power = binpower_unnorm

        if hasattr(self, 'multitaper_norm_power'):
            _, bin_multitaper_norm_power, _, _ = \
                rebin_data(self.freq, self.multitaper_norm_power, df,
                           method=method, dx=self.df)

            bin_mtp.multitaper_norm_power = bin_multitaper_norm_power

        if hasattr(self, 'jk_var_deg_freedom'):
            _, bin_jk_var_deg_freedom, _, _ = \
                rebin_data(self.freq, self.jk_var_deg_freedom, df,
                           method=method, dx=self.df)

            bin_mtp.jk_var_deg_freedom = bin_jk_var_deg_freedom

        bin_mtp.m = np.rint(step_size * self.m)

        return bin_mtp

    def compute_rms(self, min_freq, max_freq, white_noise_offset):
        return Powerspectrum.compute_rms(self, min_freq, max_freq,
                                         white_noise_offset=white_noise_offset)

    def classical_significances(self, threshold, trial_correction):
        return Powerspectrum.classical_significances(self, threshold=threshold,
                                                     trial_correction=trial_correction)

    def nfft(self, times, counts, num_freq=None):
        series_len = times.shape[-1]

        if num_freq == None:
            num_freq = 2 * series_len

        tseg = times[-1] - times[0]
        new_range = (-0.5 + 1e-5, -0.5 + (1 - 1e-5))
        times_scaled = np.interp(times, (times.min(), times.max()), new_range)

        f_hat = nfft.nfft_adjoint(times_scaled, counts, num_freq)

        f_hat = np.conjugate(f_hat[f_hat.shape[0]//2:])  # Only take 2nd half

        temp = np.arange(1, f_hat.shape[0]+1)
        odds = (temp % 2 == 0)
        del temp

        f_hat[odds] = -f_hat[odds]

        # evens = np.invert(odds)
        # del odds

        # f_hat[evens] = f_hat[evens].conj()

        return f_hat

    def lomb_scargle(self, times, counts, weight=1):

        t_length = times.shape[-1]

        # compute nfft of irregular time series
        sp = self.nfft(times, counts) * weight
        # compute nfft of 1's at frequencies 2w
        spf = self.nfft(times, np.ones(t_length), num_freq=4*t_length)
        spf = spf[1::2]

        # compute components of LS periodogram from Press & Rybicki
        Sh = -sp.imag    # \sum_i x_i sin(2\pi f t_i)
        Ch = sp.real     # \sum_i x_i cos(2\pi f t_i)
        S2 = -spf.imag   # \sum_i 1 sin(2\pi 2f)
        C2 = spf.real    # \sum_i 1 cos(2\pi 2f)

        cos2wt = C2/np.absolute(spf)
        sin2wt = S2/np.absolute(spf)
        coswt = np.sign(sin2wt)*np.sqrt(0.5*(1+cos2wt))
        sinwt = np.sqrt(0.5*(1-cos2wt))

        A1 = Ch*coswt + Sh*sinwt
        B1 = Sh*coswt - Ch*sinwt
        A2 = t_length/2 + 0.5*C2*cos2wt + 0.5*S2*sin2wt
        B2 = t_length - A2

        psd = 0.5 * (A1*A1/A2 + B1*B1/B2)

        tseg = times[-1] - times[0]
        psd *= 0.5 * tseg / t_length

        return psd

    def _multitaper_lomb_scargle(self, lc, NW=4, low_bias=True):

        if NW < 0.5:
            raise ValueError("The value of normalized half-bandwidth "
                             "should be greater than 0.5")

        Kmax = int(2 * NW)

        dpss_tapers, eigvals = \
            signal.windows.dpss(M=lc.n, NW=NW, Kmax=Kmax,
                                sym=False, return_ratios=True)

        if low_bias:
            selected_tapers = (eigvals > 0.9)
            if not selected_tapers.any():
                simon("Could not properly use low_bias, "
                      "keeping the lowest-bias taper")
                selected_tapers = [np.argmax(eigvals)]

            eigvals = eigvals[selected_tapers]
            dpss_tapers = dpss_tapers[selected_tapers, :]

        print(f"Using {len(eigvals)} DPSS windows for "
              "multitaper spectrum estimator")

        dpss_data_interpolated = []

        data_irregular = lc.counts - np.mean(lc.counts)  # De-mean
        times_regular = np.linspace(lc.time[0], lc.time[-1], lc.n)

        weights = np.sqrt(eigvals)

        for dpss_taper in dpss_tapers:
            cubic_spline_interp = \
                interpolate.InterpolatedUnivariateSpline(times_regular,
                                                         dpss_taper, k=3)
            # interpolating DPSS tapers to IRREGULAR times
            dpss_interpolated = cubic_spline_interp(lc.time)
            dpss_interpolated /= np.sum(dpss_interpolated**2)  # Re normalizing
            dpss_interpolated *= np.sqrt(lc.time.shape[0])  # From Springford R implementation
            dpss_interpolated *= data_irregular
            dpss_data_interpolated.append(dpss_interpolated)

        psd_ls = []

        tseg = lc.time[-1] - lc.time[0]
        for data, weight in zip(dpss_data_interpolated, weights):

            psd = self.lomb_scargle(lc.time, data)
            # psd *= 0.5 * tseg / lc.n
            psd_ls.append(psd)

        psd_ls = np.array(psd_ls)
        psd_ls = np.mean(psd_ls, axis=-2)

        # freq_ls = np.arange(0, length)*step + start
        freq_ls = np.arange(0, psd_ls.shape[-1])*(1/tseg)

        self.jk_var_deg_freedom = None
        self.eigvals = eigvals

        return freq_ls, psd_ls
