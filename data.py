"""
Classes and functions for handling data.
"""
import numpy as np
import os
import pandas as pd
import re
from scipy import signal
from scipy import stats

from aux import find_segs
from db import d_models, make_session

import CONFIG as C
import PARAMS as P
import LOCAL as L


class DataLoader(object):
    """
    Class for loading data files corresponding to closed loop experiments.
    
    :param trial: trial object for which to load data
    :param vel_filt: velocity filtering dict (or None)
    :param sfx: suffix to append to cleaned data file
        if no save file exists with this suffix a new one will be created
    :return: DataLoader object for quick data access and transforming
    """
    def __init__(self, trial, sfx, vel_filt, verbose=False):
        
        def alert(msg):
            if verbose:
                print(msg)
        
        if vel_filt is not None:
            raise NotImplementedError('Velocity filtering not implemented yet.')
            
        path_clean = os.path.join(
            L.DATA_ROOT, trial.path, '{}_{}.csv'.format(trial.pfx_clean, sfx))
        
        if os.path.exists(path_clean):
            # check for clean data file and load it if found
            alert(
                'Loading clean data from file "{}"...'.format(
                os.path.basename(path_clean)))
            data = pd.read_csv(path_clean)
            alert('Loaded.')
        else:
            alert(
                'Loading and cleaning data from directory "{}"...'.format(
                os.path.dirname(path_clean)))
                
            # create raw time vector and data matrix for each data component
            # (move these all to their own functions later)
            t_gcamp, gcamp = load_gcamp(trial)
            t_behav, behav = load_behav(trial)
            t_air, air = load_air(trial, t_behav, behav)
            t_odor, odor = load_odor(trial, t_behav)
            
            first_ts = np.array([t_gcamp[0], t_behav[0], t_air[0], t_odor[0]])
            
            if not np.all(first_ts == 0):
                raise Exception('Problem generating relative time vectors.')
            
            # truncate any data with time vector longer than max trial time
            t_mask_gcamp = t_gcamp < C.T_MAX
            t_mask_behav = t_behav < C.T_MAX
            t_mask_air = t_air < C.T_MAX
            t_mask_odor = t_odor < C.T_MAX
            
            t_gcamp = t_gcamp[t_mask_gcamp]
            t_behav = t_behav[t_mask_behav]
            t_air = t_air[t_mask_air]
            t_odor = t_odor[t_mask_odor]
             
            gcamp = gcamp[t_mask_gcamp]
            behav = behav[t_mask_behav]
            air = air[t_mask_air]
            odor = odor[t_mask_odor]
           
            # initialize final data structure
            t = np.arange(0, C.T_MAX, C.DT)
            data_ = np.nan * np.zeros((len(t), C.N_COLS_FINAL))
            data_[:, 0] = t

            # downsample data
            ## loop over all timepoints one-by-one, filling in their corresponding
            ## values by averaging/interpolating the raw data
            for t_ctr, t_ in enumerate(t):
                w = (t_ - C.DT/2, t_ + C.DT/2)
                
                data_[t_ctr, C.COL_SLICE_GCAMP] = avg_or_interp(
                    t_gcamp, gcamp, w, cols_ang=None)
                data_[t_ctr, C.COL_SLICE_BEHAV] = avg_or_interp(
                    t_behav, behav, w, cols_ang=[C.COLS_BEHAV['HEADING']])
                data_[t_ctr, C.COL_SLICE_AIR] = avg_or_interp(
                    t_air, air, w, cols_ang=[0])
                data_[t_ctr, C.COL_SLICE_ODOR] = avg_or_interp(
                    t_odor, odor, w, cols_ang=None)
                
            # calc w_air from downsampled air if air recorded separately
            if trial.expt in C.EXPTS_W_AIR:
                
                col_air = dict(C.COLS_FINAL)['AIR']
                col_w_air = dict(C.COLS_FINAL)['W_AIR']
                
                w_air = np.gradient(data_[:, col_air]) / np.gradient(t)
                
                data_[:, col_w_air] = w_air
                
            # convert to data frame
            data = pd.DataFrame()
            
            for vbl, col in C.COLS_FINAL:
                data[vbl] = data_[:, col]
            
            alert('Data loaded.')
            
            # save clean file for easy access next time
            data.to_csv(path_clean, index=False)
            
            alert('Cleaned data saved at "{}".'.format(path_clean))
        
        # bind data frame to data loader object
        self.data = data
        
        # store walking speed threshold
        self.walking_threshold = trial.walking_threshold
    
    def __getattr__(self, attr):
        """
        Modify attribute selection so that we can select attributes offset in time.
        """

        if attr.startswith('abs_'):
            remaining = attr[4:]
            return np.abs(getattr(self, remaining))

        if attr.startswith('ddt_'):

            remaining = attr[4:]
            if remaining in C.WRAPPED_ANG_VARS:
                # unwrap before taking gradient
                data_unwrapped = unwrap_signal(getattr(self, remaining), -180, 180)
                return np.gradient(data_unwrapped) / C.DT
            else:
                return np.gradient(getattr(self, remaining)) / C.DT
            
        if attr.startswith('neg_'):
            remaining = attr[4:]
            return -getattr(self, remaining)
        
        if attr.startswith('control'):
            return np.random.normal(0, 1, len(self.t))

    @property
    def t(self): return self.data['TIME'].as_matrix()
    
    @property
    def v_lat(self): return self.data['V_LAT'].as_matrix()
    
    @property
    def v_fwd(self): return self.data['V_FWD'].as_matrix()
    
    @property
    def v_ang(self): return self.data['V_ANG'].as_matrix()
    
    @property
    def heading(self): return self.data['HEADING'].as_matrix()
    
    @property
    def air(self): return self.data['AIR'].as_matrix()
    
    @property
    def w_air(self): return self.data['W_AIR'].as_matrix()
    
    @property
    def odor_binary(self): return self.data['ODOR_BINARY'].as_matrix()
    
    @property
    def odor_pid(self): return self.data['ODOR_PID'].as_matrix()
    
    @property
    def speed(self): return np.sqrt(self.v_lat**2 + self.v_fwd**2)
    
    @property
    def ball(self): return np.sqrt(self.v_lat**2 + self.v_fwd**2 + self.v_ang**2)
    
    @property
    def g2r(self): 
        return norm_by_col(
            (self.data['G2R_GREEN']/self.data['G2R_RED']).as_matrix())
     
    @property
    def g3r(self): 
        return norm_by_col(
            (self.data['G3R_GREEN']/self.data['G3R_RED']).as_matrix())
     
    @property
    def g4r(self): 
        return norm_by_col(
            (self.data['G4R_GREEN']/self.data['G4R_RED']).as_matrix())
     
    @property
    def g5r(self): 
        return norm_by_col(
            (self.data['G5R_GREEN']/self.data['G5R_RED']).as_matrix())
     
    @property
    def g2l(self): 
        return norm_by_col(
            (self.data['G2L_GREEN']/self.data['G2L_RED']).as_matrix())
     
    @property
    def g3l(self): 
        return norm_by_col(
            (self.data['G3L_GREEN']/self.data['G3L_RED']).as_matrix())
     
    @property
    def g4l(self): 
        return norm_by_col(
            (self.data['G4L_GREEN']/self.data['G4L_RED']).as_matrix())
     
    @property
    def g5l(self): 
        return norm_by_col(
            (self.data['G5L_GREEN']/self.data['G5L_RED']).as_matrix())
    
    @property
    def g2s(self):
        return norm_by_col(self.g2r + self.g2l)
 
    @property
    def g3s(self):
        return norm_by_col(self.g3r + self.g3l)
 
    @property
    def g4s(self):
        return norm_by_col(self.g4r + self.g4l)
 
    @property
    def g5s(self):
        return norm_by_col(self.g5r + self.g5l)

    @property
    def g2d(self):
        return norm_by_col(self.g2r - self.g2l)
 
    @property
    def g3d(self):
        return norm_by_col(self.g3r - self.g3l)
 
    @property
    def g4d(self):
        return norm_by_col(self.g4r - self.g4l)
 
    @property
    def g5d(self):
        return norm_by_col(self.g5r - self.g5l)


# auxiliary functions used by DataLoader

def load_gcamp(trial):
    """
    Load and return the gcamp time vector, in seconds and starting from 0,
    and the gcamp fluorescence matrix. The returned gcamp fluorescence matrix
    has one row per timepoint and 16 columns. The first eight are red channels
    (G2R, G3R, G4R, G5R, G2L, G3L, G4L, G5L) and the second eight are green
    channels (G2R, G3R, G4R, G5R, G2L, G3L, G4L, G5L).
    """
    if trial.expt in C.EXPTS_ASENSORY:
        
        # build path
        path_gcamp = os.path.join(L.DATA_ROOT, trial.path, trial.f_gcamp)
        
        # load data
        gcamp_ = pd.read_csv(path_gcamp, header=None).as_matrix().astype(float).T
        
        # get timestamp (first col)
        t_gcamp = gcamp_[:, 0] - gcamp_[0, 0]
        
        # make full (16-col) gcamp matrix
        gcamp = np.nan * np.zeros((len(t_gcamp), 16))
        
        # fill in G2-G5R red and G2-G5R green
        gcamp[:, :4] = gcamp_[:, 1:5]
        gcamp[:, 8:12] = gcamp_[:, 5:9]
        
    elif trial.expt in C.EXPTS_SENSORY:
        
        # build paths
        path_t_gcamp = os.path.join(L.DATA_ROOT, trial.path, trial.f_t_gcamp)
        path_gcamp = os.path.join(L.DATA_ROOT, trial.path, trial.f_gcamp)

        # load data
        t_gcamp = pd.read_csv(path_t_gcamp, header=None).as_matrix()[0]
        gcamp = pd.read_csv(
            path_gcamp, header=None).as_matrix()[:, 2:].astype(float).T

        if not len(t_gcamp) == len(gcamp):
            raise Exception('Time vector and GCaMP vector must be equal lengths.')

    # convert time vector to relative time
    t_gcamp -= t_gcamp[0]

    return t_gcamp, gcamp


def load_behav(trial):
    """
    Load and return behavioral time vector, in seconds and starting from 0,
    and behavioral data matrix including three velocity components and heading.
    """
    if trial.expt in C.EXPTS_ASENSORY:
        
        # build path
        path_behav = os.path.join(L.DATA_ROOT, trial.path, trial.f_behav)
        
        # load data
        behav_ = pd.read_csv(path_behav, header=None).as_matrix()
        
        # extract timestamp
        t_behav = behav_[:, C.COLS_FICTRAC['TIMESTAMP']] / 1000
        
        # order columns for final behav matrix
        cols = [None for _ in range(C.N_COLS_BEHAV)]
        for vbl, col in C.COLS_BEHAV.items():
            cols[col] = C.COLS_FICTRAC[vbl]

        behav = behav_[:, cols].astype(float)
        
    elif trial.expt in C.EXPTS_SENSORY:
        
        # build paths
        path_light = os.path.join(L.DATA_ROOT, trial.path, trial.f_light)
        path_behav = os.path.join(L.DATA_ROOT, trial.path, trial.f_behav)

        # get start and end frames from light times file
        start, end = pd.read_excel(path_light, header=None).as_matrix().flatten()

        # load raw behav data
        behav_ = pd.read_csv(path_behav, header=None).as_matrix()

        # create mask selecting frames between two light times
        frame_ctrs = behav_[:, C.COLS_FICTRAC['FRAME_CTR']]
        mask = (start <= frame_ctrs) & (frame_ctrs < end)

        # get time vector and behav data selected by mask
        t_behav = frame_ctrs[mask] * C.DT_FICTRAC

        # order columns for final behav matrix
        cols = [None for _ in range(C.N_COLS_BEHAV)]
        for vbl, col in C.COLS_BEHAV.items():
            cols[col] = C.COLS_FICTRAC[vbl]

        behav = behav_[mask][:, cols].astype(float)

    # convert time vector to relative time
    t_behav -= t_behav[0]

    # correct heading so 0 is uw and angles are in deg
    behav[:, C.COLS_BEHAV['HEADING']] -= np.pi
    behav[:, C.COLS_BEHAV['HEADING']] *= (180/np.pi)

    return t_behav, behav


def load_air(trial, t_behav=None, behav=None):
    """
    Load air tube time vector, in seconds and starting at 0,
    and corresponding air tube angle data.
    
    If trial.expt is 'closed', t_behav and behav must be provided.
    If trial.expt is 'no_air_motion', t_behav must be provided.
    """
    if trial.expt in C.EXPTS_W_AIR:
        
        # build path and load air_tube file
        path_air = os.path.join(L.DATA_ROOT, trial.path, trial.f_air)
        t_air, air = pd.read_csv(path_air, header=None).as_matrix()
        
        # convert time vector to seconds and add 0 s initial time
        # with corresponding NaN value for air
        t_air /= 1000.
        t_air = np.concatenate([[0], t_air])
        air = np.concatenate([[np.nan], air])
        
        # correct air tube so 0 is in front of fly and angles are in deg
        air -= np.pi/2
        air *= (180/np.pi)
        
        # don't calc air tube vel. until we've downsampled it
        w_air = np.nan * np.ones(len(t_air))
        
    elif trial.expt in ['closed']:
        
        # make air tube signal equal to behavioral heading signal
        if (t_behav is None) or (behav is None):
            raise Exception(
                '"t_behav" and "behav" must be provided for closed loop trials.')
        
        t_air = t_behav
        air = behav[:, C.COLS_BEHAV['HEADING']]
        
        # calc air tube vel
        w_air = np.gradient(unwrap(air, *C.LIMS_ANG)) / np.gradient(t_behav)
        
    elif trial.expt in (['no_air_motion'] + C.EXPTS_ASENSORY):
        
        # make air tube signal all NaNs
        if t_behav is None:
            raise Exception('"t_behav" must be provided for no_air_motion trials.')
        
        t_air = t_behav.copy()
        air = np.nan * np.ones(len(t_behav))
        w_air = np.nan * np.ones(len(t_behav))
        
    else:
        raise Exception('Expt "{}" not found in config.'.format(trial.expt))
    
    return t_air, np.array([air, w_air]).T


def load_odor(trial, t_behav=None):
    """
    Load odor time vector (in s, starting at 0), ttl reading, and pid reading.
    """
    if trial.expt in C.EXPTS_ODOR_FLUCT:
        
        # build paths and load odor files
        path_odor_binary = os.path.join(
            L.DATA_ROOT, trial.path, trial.f_odor_binary)
        t_odor_binary, odor_binary = pd.read_csv(
            path_odor_binary, header=None).as_matrix()
        
        path_odor_pid = os.path.join(
            L.DATA_ROOT, trial.path, trial.f_odor_pid)
        t_odor_pid, odor_pid = pd.read_csv(
            path_odor_pid, header=None).as_matrix()
        
        # make sure binary and pid have same timestamps
        np.testing.assert_array_almost_equal(t_odor_binary, t_odor_pid)
        t_odor = t_odor_binary
        
        # convert time vector to seconds and add initial 0 value
        t_odor /= 1000.
        
        t_odor = np.concatenate([[0], t_odor])
        
        odor_binary = np.concatenate([[np.nan], odor_binary])
        odor_pid = np.concatenate([[np.nan], odor_pid])
        
        # binarize odor_binary
        mask_low = odor_binary < C.ODOR_BINARY_CUTOFF
        mask_high = odor_binary >= C.ODOR_BINARY_CUTOFF
        
        odor_binary[mask_low] = 0
        odor_binary[mask_high] = 1
        
    else:
        
        # make odor binary signal = 1 between 90 & 150 s
        if t_behav is None:
            raise Exception('"t_behav" must be provided if no odor files.')
            
        t_odor = t_behav.copy()
        
        odor_binary = np.zeros(len(t_behav))
        odor_binary[(C.ODOR_START <= t_behav) & (t_behav < C.ODOR_END)] = 1
        
        odor_pid = np.nan * np.ones(len(t_behav))
        
    return t_odor, np.array([odor_binary, odor_pid]).T
        

def avg_or_interp(t, x, w, cols_ang):
    """
    Return the averaged or interpolated value of an array or matrix
    between two time values.
    
    If at least one non-nan value exists in x between in the time window
    specified by w[0] and w[1], the mean of these values will be returned.
    
    If no values exist in the time window, a linear interpolation is performed
    between the two nearest existing values.
    
    :param t: time vector
    :param x: data vector/matrix
    :param w: time window to average/interpolate (t_0, t_1)
    :param cols_ang: which cols in x correspond to angular variables defined only
        from -180 to 180, which must be "unwrapped" before avg'ing/interp'ing
    :return: avg'ed/interp'ed value(s) of x
    """
    
    if len(t) != len(x):
        raise Exception('"t" and "x" must be the same length')
        
    if x.ndim == 1:
        x = x[:, None]
        
    if cols_ang is None:
        cols_ang = []
    
    # select x for times between t_0 and t_1
    mask = (t >= w[0]) & (t < w[1])
    x_mask = x[mask, :]
    
    # loop over columns of x
    y = np.nan * np.ones(x.shape[1])
    
    for col_ctr, x_ in enumerate(x_mask.T):
        
        if col_ctr in cols_ang:
            x_ = unwrap(x_, *C.LIMS_ANG)
            
        if sum(~np.isnan(x_)) > 0:
            # take average if at least 1 non-nan val exists
            y_ = np.nanmean(x_)
            
        else:
            # find closest non-nan val before t_0
            t_before = t[t < w[0]]
            x_before = x[t < w[0], col_ctr]
            mask_before = ~np.isnan(x_before)
            
            try:
                t_0 = t_before[mask_before][-1]
                x_0 = x_before[mask_before][-1]
            except IndexError:
                t_0 = np.nan
                x_0 = np.nan
                
            # find closest non-nan val after t_1
            t_after = t[t >= w[1]]
            x_after = x[t >= w[1], col_ctr]
            mask_after = ~np.isnan(x_after)
            
            try:
                t_1 = t_after[mask_after][0]
                x_1 = x_after[mask_after][0]
            except IndexError:
                t_1 = np.nan
                x_1 = np.nan
                
            if any(np.isnan([x_0, x_1])):
                # if there were not two non-nan values around x at t
                y_ = np.nan
            else:
                # linearly interpolate between x_0 and x_1
                if col_ctr in cols_ang:
                    x_0, x_1 = unwrap(np.array([x_0, x_1]), *C.LIMS_ANG)
                    
                slp = (x_1 - x_0) / (t_1 - t_0)  # slope
                y_ = x_0 + slp * (np.mean(w) - t_0)
                
        if col_ctr in cols_ang:
            y_ = wrap(np.array([y_]), *C.LIMS_ANG)[0]
            
        y[col_ctr] = y_
            
    if len(y) > 1:
        return y
    else:
        return y[0]


def wrap(x, x_min, x_max):
    """
    Wrap a time-series signal so that it becomes modular.

    :param x: 1D array
    :param x_min: min value of time-series
    :param x_max: max value of time-series
    :return: wrapped signal
    """

    assert x_max > x_min

    # shift signal so that x_min is at 0
    x_shifted = x - x_min

    # mod signal by desired range
    x_shifted %= (x_max - x_min)

    # return inverse-shifted signal
    return x_shifted + x_min


def unwrap(x, x_min, x_max):
    """
    Unwrap a modular time-series signal so derivatives are still meaningful
    when signal crosses mod threshold.

    :param x: 1D array
    :param x_min: min value of time-series
    :param x_max: max value of time-series
    :return: unwrapped signal
    """
    th_jump = 0.5  # pcnt of range signal must change by to assume it jumped
    
    x = x.copy()
    assert x_max > x_min

    x_range = x_max - x_min
    diff = np.diff(x)
    cc = np.concatenate

    # get all boundary crossings
    down_jumps = cc([[0.], (diff < (-th_jump * x_range)).astype(float)])
    up_jumps = cc([[0.], (diff > (th_jump * x_range)).astype(float)])

    down_ctrs = down_jumps.cumsum()
    up_ctrs = up_jumps.cumsum()

    offset_ctr = down_ctrs - up_ctrs
    x_shifted = x - x_min
    x_shifted += offset_ctr * x_range

    return x_shifted + x_min


def get_states(speed, threshold):
    """
    Infer the fly's state at each time point from its walking speed.
    
    :param speed: walking speed time-series
    :param threshold: walking speed threshold
    """
    
    segs_paused = find_segs(speed < threshold)
    
    # eliminate paused segs less than min paused times
    segs_paused = np.array([seg for seg in segs_paused if seg[1] - seg[0] >= int(P.T_PAUSE_MIN/C.DT)])
    
    if len(segs_paused) == 0:
        return np.repeat('W', len(speed))
    
    # get walking segs
    segs_walk = get_seg_complement(segs_paused, len(speed))
    
    # buffer segments to allow for ambiguous states
    segs_paused[:, 0] += int(P.T_PAUSE_CUSH_INNER/C.DT)
    segs_paused[:, 1] -= int(P.T_PAUSE_CUSH_INNER/C.DT)
    segs_walk[:, 0] += int(P.T_PAUSE_CUSH_OUTER/C.DT)
    segs_walk[:, 1] -= int(P.T_PAUSE_CUSH_OUTER/C.DT)
    
    # ensure all segs are bounded by the time-series end points
    segs_paused[segs_paused < 0] = 0
    segs_paused[segs_paused > len(speed)] = len(speed)
    segs_walk[segs_walk < 0] = 0
    segs_walk[segs_walk > len(speed)] = len(speed)
    
    # remove any segments with a correct length <= 0
    segs_paused = np.array([seg for seg in segs_paused if seg[1] - seg[0] > 0])
    segs_walk = np.array([seg for seg in segs_walk if seg[1] - seg[0] > 0])
    
    # start with default ambiguous ('A') labels then fill in paused ('P')
    # and walking ('W') labels
    labels = np.repeat('A', len(speed))
    labels[segs_to_bool(segs_paused, len(speed))] = 'P'
    labels[segs_to_bool(segs_walk, len(speed))] = 'W'
    
    return labels


def get_seg_complement(segs, end):
    """
    Get the complement of a set of 1-D segments.
    
    E.g., get_seg_complement([[5, 9], [20, 22]], 0, 25) returns
        [[0, 5], [9, 20], [22, 25]]
        
    :param segs: N x 2 array of segment starts (1st col) and ends (2nd col)
    :param end: length of series
    """
    if not isinstance(end, int):
        raise TypeError('Args "start" and "end" must be integers.')
    if (not (0 <= segs[0, 0])) or (not (end >= segs[-1, 1])):
        raise ValueError('Segs must be bounded by 0 and end idx.')
        
    segs = np.array(segs)
        
    # flatten segs into 1D array and add a start and end idx (which effectively
    # shifts all idxs by 1 when we reshape it)
    flat = np.concatenate([[0], segs.flatten(), [end]])
    # reshape it to get new segments
    segs = flat.reshape((len(segs)+1, 2))
    
    # remove any zero-valued segments
    segs = np.array([seg for seg in segs if seg[1] - seg[0]])
    
    return segs


def segs_to_bool(segs, end):
    """
    Convert a set of segments to a logical mask.
    
    E.g., segs_to_logical([[2, 4], [5, 8]], 0, 10) returns:
        [False, False, True, True, False, True, True, True, False, False]
    
    :param segs: N x 2 array of segment starts (1st col) and ends (2nd col)
    :param end: length of series
    """
    mask = np.repeat(False, end)
    
    for seg in segs:
        if seg[1] > seg[0]:
            mask[seg[0]:seg[1]] = True
            
    return mask


def norm_by_col(data):
    """
    Normalize each column of a data matrix by subtracting its mean
    and dividing by its std.
    """
    data_normed = np.nan * np.zeros(data.shape)

    if data.ndim == 1:
        col_zeroed = data - np.nanmean(data)
        data_normed = col_zeroed / np.nanstd(col_zeroed)

    else:
        for ctr in range(data.shape[1]):
            col_zeroed = data[:, ctr] - np.nanmean(data[:, ctr])
            data_normed[:, ctr] = col_zeroed / np.nanstd(col_zeroed)

    return data_normed
    

# other auxiliary functions

def get_trial(name):
    """
    Get a trial object by name.
    """
    session = make_session()
    trials = session.query(d_models.Trial).filter_by(name=name)
    if trials.count() == 0:
        raise Exception('Trial "{}" not found.'.format(name))
    elif trials.count() > 1:
        raise Exception('Trial "{}" found multiple times.'.format(name))
    else:
        trial = trials.first()
        session.close()
        return trial


def split_at_nans(x, min_len, mode='any'):
    """
    Split a 1- or 2-d array into segments containing no nans.
    :param x: data (n_samples x n_vars)
    :param min_len: minimum length of resulting segments
    :param mode: 'any' to ignore rows with any nans, 'all' to only ignore if all nans
    :return: list of segments, list of logical masks relative to original data
    """
    dim = x.ndim
    x = x if dim != 1 else x[:, None]

    if mode == 'any':
        mask_1d = ~np.any(np.isnan(x), axis=1)
    elif mode == 'all':
        mask_1d = ~np.all(np.isnan(x), axis=1)
    else:
        raise Exception(
            'mode "{}" not recognized: please use "any" or "all"'.format(mode))

    mask_1d = mask_1d.astype(int)

    starts = (np.diff(np.concatenate([[0], mask_1d, [0]])) == 1).nonzero()[0]
    ends = (np.diff(np.concatenate([[0], mask_1d, [0]])) == -1).nonzero()[0]

    masks = []
    segs = []

    for start, end in zip(starts, ends):
        if end - start >= min_len:
            mask = np.zeros(mask_1d.shape, dtype=bool)
            mask[start:end] = True
            seg = x[mask]
            if dim == 1: seg = seg.flatten()

            masks.append(mask)
            segs.append(seg)

    return segs, masks


def shift_circular(x, shift_amount):
    """
    Shift all the rows of x by a specified amount, with wrapping.
    """

    if shift_amount == 0:
        return x[:]

    x_shifted = np.nan * np.zeros(x.shape)

    x_shifted[shift_amount:] = x[:-shift_amount]
    x_shifted[:shift_amount] = x[-shift_amount:]

    return x_shifted


def random_shift(x, return_shift_amount=False):
    """
    Shift all the rows of x by a random amount, with wrapping.
    """

    shift_amount = np.random.randint(0, len(x))

    if not return_shift_amount:
        return shift(x, shift_amount)
    else:
        return shift(x, shift_amount), shift_amount


def phase_shift(x, phase):
    """
    Shift a time-series by a delay corresponding to a specific phase
    of the maximum frequency component.
    :param x:
    :param phase:
    :return:
    """
    # find frequency with max power
    freqs, p_xx = signal.periodogram(x, 1)
    f = freqs[np.argmax(p_xx)]
    period = 1./f

    # convert phase to delay in time steps
    delay = int((phase / (2*np.pi)) * period)

    if delay == 0:
        x_shifted = x
    elif delay > 0:
        x_shifted = np.nan * np.zeros(x.shape)
        x_shifted[delay:] = x[:-delay]
    elif delay < 0:
        x_shifted[:delay] = x[-delay:]

    return x_shifted


def slice_by_time(t, x, t_ranges):
    """
    Slice a series of data according to a set of time-ranges.
    :param t: time vector
    :param x: data (2D array with rows as time points)
    :param t_ranges: list of time ranges for slicing x
    :return: list of slices of x
    """

    x_sliced = []

    for t_range in t_ranges:
        x_sliced.append(x[(t >= t_range[0]) * (t < t_range[1])])

    return x_sliced


def norm_by_col_multi(xs):
    """
    Normalize a data set by calculating mean and std across multiple data sets.
    :param xs: list of 1D or 2D arrays
    :return: list of arrays where columns have been normalized
    """

    xs_cc_normed = normalize_by_column(np.concatenate(xs, axis=0))

    if len(xs) == 1:
        return [xs_cc_normed]

    else:
        xs_lens = [len(x) for x in xs]
        split_idxs = np.cumsum(xs_lens)[:-1]

        return np.split(xs_cc_normed, split_idxs, axis=0)


def exp_filt(t, x, amp, tau, t_max_filt):
    """
    Exponentially filter a signal.

    :param x: signal (1D array)
    :param amp: amplitude of exponential filter
    :param tau: time constant of exponential filter (s)
    :param dt: signal time step (s)
    :param t_max_filt: upper limit of filter domain (s)

    :return filtered signal, filter time vector, filter
    """

    dt = np.mean(np.diff(t))
    
    # build filter
    
    t_filt = np.arange(0, t_max_filt, dt)
    filt = amp * np.exp(-t_filt / tau)
    
    # zero-pad filter
    
    zero_pad = np.zeros((len(filt) - 1,))
    x_padded = np.concatenate([zero_pad, x])
    
    # convolve filter with signal
    
    y = signal.fftconvolve(x_padded, filt, mode='valid') * dt

    # return filtered signal, filter time vector, and filter
    
    return y, t_filt, filt
