from __future__ import division, print_function
import logging
import numpy as np
import os


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
    
        [1, v['a'][t-1], v['a'][t], v['b'][t-1], v['b'][t], v['b'][t+1]]
        
    :param vs: dict of 1-D array of predictors
    :param windows: dict of (start, end) time point tuples, rel. to time point of 
        prediction, e.g., negative for time points before time point of
        prediction
    :param order: order to add predictors to final matrix in
    :return: extended predictor matrix, which has shape
        (n, 1 + (windows[0][1]-windows[0][0]) + (windows[1][1]-windows[1][0]) + ...)
    """
    if not np.all([w[1] - w[0] >= 0 for w in windows.items()]):
        raise ValueError('Windows must all be non-negative.')
        
    n = len(list(vs.values())[0])
    if not np.all([v.ndim == 1 and len(v) == n for v in vs.values()]):
        raise ValueError('All values in "vs" must be 1-D arrays of the same length.')
        
    # make extended predictor array
    vs_extd = [np.ones((n, 1))]
    
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
        v_extd.append(v_)

    # return full predictor matrix
    return np.concatenate(v_extd, axis=1)
