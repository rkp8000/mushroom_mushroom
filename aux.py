from __future__ import division, print_function
import logging
import numpy as np
import os
import pandas as pd


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


# load data
def load_data(expt, base, cols, normed_cols, odor):
    data_dir = os.path.join('data_', expt)
    
    trials = []  # list of trials
    data_u = {}  # unnormalized data
    data_n = {}  # normalized data
    d_odor = {}  # dfs of odor times

    for fly in os.listdir(data_dir):
        fly_path = os.path.join(data_dir, fly)

        for trial in os.listdir(fly_path):
            trial_path = os.path.join(fly_path, trial)

            # load original data
            data_o_ = pd.read_csv(os.path.join(trial_path, base))

            # select out data (unnormalized) to store in renamed columns
            data_u_ = pd.DataFrame()
            for col in cols:
                if len(col) == 2:
                    data_u_[col[0]] = data_o_[col[1]]
                elif len(col) == 3:
                    f = col[2]  # func to apply 
                    data_u_[col[0]] = f(data_o_[col[1]])

            # make odor mask
            odor_mask = np.zeros(len(data_u_['Time']), bool)

            if odor is None:  # no odor
                df_odor = pd.DataFrame(data={'Start': [], 'Stop': []}, columns=['Start', 'Stop'])
            
            elif hasattr(odor, 'upper'):  # is string
                
                # load file
                df_odor = pd.read_csv(os.path.join(trial_path, odor))
                df_odor.columns = ['Start', 'Stop']
                
                # get starts and stops
                starts = df_odor['Start']
                stops = df_odor['Stop']

                # fill in mask
                for start, stop in zip(starts, stops):
                    odor_mask[(start <= data_u_['Time']) & (data_u_['Time'] < stop)] = True
                    
            else:  # single odor presentation
                start, stop = odor
                df_odor = pd.DataFrame(data={'Start': [start], 'Stop': [stop]}, columns=['Start', 'Stop'])
                
                odor_mask[(start <= data_u_['Time']) & (data_u_['Time'] < stop)]

            data_u_['Odor'] = odor_mask.astype(float)

            # normalize data
            data_n_ = data_u_.copy()
            data_n_[normed_cols] -= data_n_[normed_cols].mean()
            data_n_[normed_cols] /= data_n_[normed_cols].std()

            # store all results
            data_u[trial] = data_u_
            data_n[trial] = data_n_
            d_odor[trial] = df_odor
            
            trials.append(trial)
            
    return Generic(trials=trials, d_u=data_u, d_n=data_n, d_odor=d_odor)
