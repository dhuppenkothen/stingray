import copy
import warnings

import numpy as np
import scipy
import scipy.optimize
import scipy.stats

import stingray.utils as utils
from stingray.crossspectrum import AveragedCrossspectrum, Crossspectrum
from stingray.gti import bin_intervals_from_gtis, check_gtis
from stingray.largememory import createChunkedSpectra, saveData
from stingray.stats import pds_probability
from stingray.utils import genDataPath

from .events import EventList
from .gti import cross_two_gtis
from .lightcurve import Lightcurve

from nitime.lazy import matplotlib_mlab as mlab
from nitime.lazy import scipy_linalg as linalg
from nitime.lazy import scipy_signal as sig
from nitime.lazy import scipy_interpolate as interpolate
from nitime.lazy import scipy_fftpack as fftpack

import nitime.utils as utils
from nitime.utils import tapered_spectra, dpss_windows

# To support older versions of numpy that don't have tril_indices:
from nitime.index_utils import tril_indices, triu_indices

try:
    from tqdm import tqdm as show_progress
except ImportError:
    def show_progress(a, **kwargs):
        return a

__all__ = ["Powerspectrum", "AveragedPowerspectrum", "DynamicalPowerspectrum"]


class Multitaper(Crossspectrum):
    pass
