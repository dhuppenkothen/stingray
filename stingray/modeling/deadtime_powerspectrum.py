from astropy.modeling.models import custom_model
import numpy as np


def P1(td, tb, r0):
    """
        This function supports the `dead_time_ps` method by calculating its P1 parameter.
        Parameters
        ----------
        td : float
            Dead time of the instrument.
        tb : float
            Bin time or time resolution.
        r0 : float
            Detected Event rate of counts by the instrument.
        Returns
        -------
        P1 : float
            Parameter for dead time correction of Power spectrum used in following formula.
            P(f) = P1 - P2 * cos(pi*f/fnyq)
    """

    return 2 * [1 - 2 * r0 * td * (1 - (td / 2 * tb))]


def P2(td, tb, r0, N):
    """
           This function supports the `dead_time_ps` method by calculating its P1 parameter.
           Parameters
           ----------
           td : float
               Dead time of the instrument.
           tb : float
               Bin time or time resolution.
           r0 : float
               Detected Event rate of counts by the instrument.
            N : int
                No. of frequencies in power spectrum
           Returns
           -------
           P2 : float
               Parameter for dead time correction of Power spectrum used in following formula.
               P(f) = P1 - P2 * cos(pi*f/fnyq)
            
       """
    return 2 * r0 * td * ((N - 1) / N) * (td / tb)


@custom_model
def dead_time_ps(f, fnyq, P1=1., P2=1.):
    """
        Custom astropy.modeling model intended to model dead_time effects in power spectrum. Implements:
            P(f) = P1 - P2 * cos(pi*f/fnyq)
            where,   
                   P1 = 2[1 - 2*r0*td(1-(td/2tb))]
             and   P2 = 2*r0*td(((N-1)/N)*(td/tb))
        See references for more details.
        Inputs
        ----------
        f : numpy.ndarray
            Array of frequencies of power spectrum
        fnyq : float
            The Nyquist frequency (Zhang et al. 1995) 
        
        Parameter
        ----------
        P1 : float, default 1.0
            P1 parameter of dead time corrected power spectrum
        P2 : float, default 1.0
            P2 parameter of dead time corrected power spectrum
        Returns
        -------
        power : numpy.ndarray
            Array of power spectrum taking into account dead time.    
              
        References
        ----------     
        [1] CALIBRATION OF THE ROSSI X-RAY TIMING EXPLORER PROPORTIONAL COUNTER ARRAY by 
        Keith Jahoda, Craig B. Markwardt, Yana Radeva, Arnold H. Rots, Michael J. Stark,
        Jean H. Swank, Tod E. Strohmayer and William Zhang.
        [2] Dead Time Modifications to the Fast Fourier Transform Power Spectra.
        W. Zhang, K. Jahoda, J. H. Swank, E. H. Morgan and A. B. Giles. 
    """

    power = P1 - [P2 * np.cos((np.pi * f) / fnyq)]
    return power