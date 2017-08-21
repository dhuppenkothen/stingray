from astropy.modeling.models import custom_model
import numpy as np


def P1(td, tb, r0):

	 return 2 * [1 - 2 * r0 * td * (1 - (td / 2 * tb))]


def P2(td, tb, r0, N):
	return 2 * r0 * td * ((N - 1) / N) * (td / tb)


@custom_model
def dead_time_ps(f, fnyq, P1=1., P2=1.):
	power = P1 - [P2 * np.cos((np.pi * f) / fnyq)]
    return power
