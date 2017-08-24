import heapq
import numpy as np

import CONFIG as C
from data_handling import DataLoader, DataLoaderDeprecated
from db import d_models
import filters


def get_processed_data_loaders(trials, vel_filt, batch_1=False):
    """
    Return a set of data loaders for the given recording sessions. Each data loader
    will have already done the velocity filtering and time-series resampling.
    """

    if vel_filt == 'default':
        vel = C.VEL_FILT_DEFAULT

    dls = []

    for trial in trials:

        if not batch_1:
            dl = DataLoader(trial)
        else:
            dl = DataLoaderDeprecated(trial)

        dl.filt_vel_exp(vel_filt)
        dl.resample_behavioral_data_gcamp_matched()
        dls.append(dl)

    return dls


def array_to_str(a): return '{' + ','.join(a) + '}'


def make_chunked_training_masks(ts, chunk_size, train_fraction, n_masks):
    """
    Generate a set of training masks by selecting random chunks of a time-series to include.
    """

    # divide time-series up into chunks
    chunks = np.arange(int((ts[-1] - ts[0])/chunk_size))
    chunk_masks = [
        (chunk*chunk_size <= ts)*(ts < (chunk+1)*chunk_size)
        for chunk in chunks
    ]

    # specify training and test sets
    n_training_chunks = int(len(chunks) * train_fraction)
    training_masks = []

    for _ in range(n_masks):
        training_chunks = np.random.choice(chunks, n_training_chunks, replace=False)
        training_masks.append(
            np.sum([chunk_masks[chunk] for chunk in training_chunks], axis=0).astype(bool))

    return training_masks


def scale_speeds(speeds):
    """
    Scale a speed time-series to have a mean of 1.
    :param speeds: 1D array of speeds
    :return: scaled 1D array
    """
    return speeds / np.mean(speeds)


def subtract_running_avg(x, window):

    temp = np.convolve(x, np.ones(window, dtype=float)/window, mode='same')
    temp[:(window - int(window/2))] = np.nan
    temp[-int(window/2):] = np.nan

    return x - temp
