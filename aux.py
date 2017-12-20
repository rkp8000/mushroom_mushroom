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


def make_extended_predictor_matrix(vs, windows, order):
    """
    Make a predictor matrix that includes offsets at multiple time points.
    For example, if vs has 2 keys 'a' and 'b', windows is {'a': (-1, 1),
    'b': (-1, 2)}, and order = ['a', 'b'], then result rows will look like:
    
        [v['a'][t-1], v['a'][t], v['b'][t-1], v['b'][t], v['b'][t+1]]
        
    :param vs: dict of 1-D array of predictors
    :param windows: dict of (start, end) time point tuples, rel. to time point of 
        prediction, e.g., negative for time points before time point of
        prediction
    :param order: order to add predictors to final matrix in
    :return: extended predictor matrix, which has shape
        (n, (windows[0][1]-windows[0][0]) + (windows[1][1]-windows[1][0]) + ...)
    """
    if not np.all([w[1] - w[0] >= 0 for w in windows.values()]):
        raise ValueError('Windows must all be non-negative.')
        
    n = len(list(vs.values())[0])
    if not np.all([v.ndim == 1 and len(v) == n for v in vs.values()]):
        raise ValueError(
            'All values in "vs" must be 1-D arrays of the same length.')
        
    # make extended predictor array
    vs_extd = []
    
    # loop over predictor variables
    for key in order:
        
        start, end = windows[key]
        
        # make empty predictor matrix
        v_ = np.nan * np.zeros((n, end-start))

        # loop over offsets
        for col_ctr, offset in enumerate(range(start, end)):

            # fill in predictors offset by specified amount
            if offset < 0:
                v_[-offset:, col_ctr] = vs[key][:offset]
            elif offset == 0:
                v_[:, col_ctr] = vs[key][:]
            elif offset > 0:
                v_[:-offset, col_ctr] = vs[key][offset:]

        # add offset predictors to list
        vs_extd.append(v_)

    # return full predictor matrix
    return np.concatenate(vs_extd, axis=1)


def calc_r2(y, y_hat):
    """
    Calculate an R^2 value between a true value
    and a prediction.
    """
    valid = (~np.isnan(y)) & (~np.isnan(y_hat))
    
    if valid.sum() == 0:
        return np.nan
    else:
        u = np.nansum((y[valid] - y_hat[valid]) ** 2)
        v = np.nansum((y[valid] - y_hat[valid].mean()) ** 2)
        
        return 1 - (u/v)
