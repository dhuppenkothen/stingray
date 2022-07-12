import os
from pathlib import Path
from scipy.signal import periodogram
import numpy as np
from typing import Union
from stingray.modeling.QPO_estimation.utils.get_priors import _get_mean_prior

import matplotlib
import matplotlib.pyplot as plt


def _truncate_data(times: np.ndarray, counts: np.ndarray, start: float, stop: float, yerr: np.ndarray = None) -> tuple:
    indices = np.where(np.logical_and(times > start, times < stop))[0]
    if yerr is None:
        return times[indices], counts[indices]
    else:
        return times[indices], counts[indices], yerr[indices]


def _rebin(times: np.ndarray, counts: np.ndarray, rebin_factor: int) -> tuple:
    new_times = []
    new_counts = []
    for i in range(0, len(times), rebin_factor):
        if len(times) - i < rebin_factor:
            break
        c = 0
        for j in range(rebin_factor):
            c += counts[i + j]
        new_times.append(times[i])
        new_counts.append(c)
    return np.array(new_times), np.array(new_counts)


def _get_all_binned_data(file_path: str,
                         rebin_factor: Union[int, None] = 8,
                         subtract_t0: bool = True) -> tuple:
    data = np.loadtxt(file_path)
    if rebin_factor is not None:
        times, counts = _rebin(data[:, 0], data[:, 1], rebin_factor=rebin_factor)

    else:
        times, counts = data[:, 0], data[:, 1]
    if subtract_t0:
        times -= times[0]
    return times, counts


def _get_segmented_magnetar_data(
        file_path: str,
        start_time: float,
        end_time: float,
        subtract_t0: bool = True,
        rebin_factor: Union[int, None] = 8) -> tuple:

    times, y = _get_all_binned_data(file_path=file_path,
                                    rebin_factor=rebin_factor,
                                    subtract_t0=subtract_t0)

    return _truncate_data(times=times, counts=y, start=start_time, stop=end_time)


def get_magnetar_binned_data(file_path: str,
                             run_mode: str = 'entire_segment',
                             start_time: float = 0,
                             end_time: float = 1,
                             rebin_factor: int = 8,
                             subtract_t0: bool = True) -> tuple:
    """
    Get the magnetar binned data.
    """
    if run_mode == 'entire_segment':
        times, y = _get_all_binned_data(file_path=file_path,
                                        rebin_factor=rebin_factor,
                                        subtract_t0=subtract_t0)

    elif run_mode == 'segment':
        times, y = _get_segmented_magnetar_data(file_path=file_path,
                                                start_time=start_time,
                                                end_time=end_time,
                                                rebin_factor=rebin_factor,
                                                subtract_t0=subtract_t0)
    else:
        raise ValueError('run_mode must be either "entire_segment" or "segment"')

    yerr = np.sqrt(y)
    yerr[np.where(yerr == 0)[0]] = 1

    return times, y, yerr


if __name__ == '__main__':
    # Get the data
    file_path = r'D:\Stingray_QPO\stingray\stingray\datasets\magnetar_flares\SGR_0501\080823478_lcobs_data.txt'
    times, y, yerr = get_magnetar_binned_data(file_path=file_path,
                                              run_mode='entire_segment',
                                              start_time=0,
                                              end_time=1,
                                              rebin_factor=8,
                                              subtract_t0=True)

    segment_length = times[-1] - times[0]
    sampling_frequency = 1 / (times[1] - times[0])





    t_0_min = times[0] - 0.1 * segment_length


    t_0_max = times[-1] + 0.1 * segment_length


    sigma_min = 0.5 * 1 / sampling_frequency


    sigma_max = 2 * segment_length

    l = _get_mean_prior(times,y,yerr,model_type='gaussian',t_0_max = t_0_max,t_0_min = t_0_min,sigma_max = sigma_max,sigma_min = sigma_min)

    print(l)
