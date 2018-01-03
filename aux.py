from __future__ import division, print_function
import logging
import numpy as np
import os


class Generic(object):
    """Class for generic object."""
    
    def __init__(self, **kwargs):
        
        for k, v in kwargs.items():
            self.__dict__[k] = v
            
    def __setattr__(self, k, v):
        
        raise Exception('Attributes may only be set at instantiation.')
        
        
def save_table(save_file, df, header=True, index=False):
    """
    Save a pandas DataFrame instance to disk.
    """
    
    # make sure save directory exists
    save_dir = os.path.dirname(save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    df.to_csv(save_file, header=header, index=index)


def find_segs(x):
    """
    Return a list of index pairs corresponding to groups of data where x is True.
    :param x: 1D array
    :return: list of index pairs
    """
    assert x.dtype == bool

    # find the indices of starts and ends
    diff = np.diff(np.concatenate([[0], x, [0]]))

    starts = (diff == 1).nonzero()[0]
    ends = (diff == -1).nonzero()[0]

    return np.array([starts, ends]).T


def split_data(x, n_bins):
    """
    Return n_bins logical masks splitting the data into ordered partitions.
    """
    n_valid = np.sum(~np.isnan(x))
    
    idxs_all = np.argsort(x)
    idxs_valid = idxs_all[:n_valid]
    
    bounds = np.round(np.arange(n_bins + 1) * (n_valid/n_bins)).astype(int)
    
    masks = []
    
    for lb, ub in zip(bounds[:-1], bounds[1:]):
        
        mask = np.zeros(len(x), dtype=bool)
        mask[idxs_valid[lb:ub]] = True
        
        masks.append(mask.copy())
        
    return masks


def nansem(x, axis=None):
    """
    Calculate the standard error of the mean ignoring nans.
    :param x: data array
    :param axis: what axis to calculate the sem over
    :return: standard error of the mean
    """

    std = np.nanstd(x, axis=axis, ddof=1)
    sqrt_n = np.sqrt((~np.isnan(x)).sum(axis=axis))

    return std / sqrt_n
