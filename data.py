"""
Classes and functions for handling data.
"""
import numpy as np
import os
import pandas as pd
import re
from scipy import signal
from scipy import stats

import CONFIG as C
import PARAMS as P
import LOCAL_SETTINGS as L


class DataLoaderBase(object):
    """
    Base class for loading data files.
    """
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
                return np.gradient(data_unwrapped) / self.dt_gcamp
            else:
                return np.gradient(getattr(self, remaining)) / self.dt_gcamp
            
        if attr.startswith('neg_'):
            remaining = attr[4:]
            return -getattr(self, remaining)
        
        if attr.startswith('control'):
            return self.filt_random_noise()
        
    @property
    def fs(self):
        return round(1/self.dt_gcamp)

    # useful behavioral covariates
    @property
    def fictive_x(self): return self.data_behav[:, 14]

    @property
    def fictive_y(self): return self.data_behav[:, 15]

    @property
    def v(self): return self.data_behav[:, 5:8]

    @property
    def v_fwd(self): return self.data_behav[:, 6]

    @property
    def v_lat(self): return self.data_behav[:, 5]

    @property
    def v_ang(self): return self.data_behav[:, 7]

    @property
    def speed(self): return np.sqrt(np.sum(self.data_behav[:, 5:7] ** 2, axis=1))

    @property
    def speed_mean_1(self): return self.speed / self.speed.mean()

    @property
    def ball_speed(self):
        return np.sqrt(np.sum(self.data_behav[:, 5:8] ** 2, axis=1))

    @property
    def a_fwd(self): return np.gradient(self.v_fwd) / self.dt_gcamp

    @property
    def a_lat(self): return np.gradient(self.v_lat) / self.dt_gcamp

    @property
    def a_ang(self): return np.gradient(self.v_ang) / self.dt_gcamp

    @property
    def heading(self): return self.data_behav[:, 16]

    @property
    def wind_dir(self):

        wind_dir = self.heading.copy()

        wind_dir[wind_dir > C.AIR_FLOW_OFF_ANGLE] = np.nan
        wind_dir[wind_dir < -C.AIR_FLOW_OFF_ANGLE] = np.nan

        wind_dir[wind_dir > C.AIR_FLOW_MAX_ANGLE] = C.AIR_FLOW_MAX_ANGLE
        wind_dir[wind_dir < -C.AIR_FLOW_MAX_ANGLE] = -C.AIR_FLOW_MAX_ANGLE

        return wind_dir

    @property
    def wind_dir_front(self):

        wind_dir = self.heading.copy()

        wind_dir[wind_dir > C.AIR_FLOW_MAX_ANGLE] = np.nan
        wind_dir[wind_dir < -C.AIR_FLOW_MAX_ANGLE] = np.nan

        return wind_dir

    def filt_vel_exp(self, filt_params):
        """
        Filter the vel data with an exponential filter. This modifies the
        variable self.data_behav.

        :param filt_params: dictionary of filter params:
            {'AMP': ..., 'TAU': ..., 'T_MAX_FILT': ...}
        """

        # build filtering function
        if filt_params is not None:

            # build a filter function
            def filt_func(x):

                result = exp_filter(
                    self.timestamp_behav, x,
                    filt_params['AMP'],
                    filt_params['TAU'],
                    filt_params['T_MAX_FILT'])

                return result[0]

        else:
            def filt_func(x): return x

        self.filt_func = filt_func

        # filter vels (columns 5, 6, 7)
        self.data_behav[:, 5] = filt_func(self._data_behav[:, 5])
        self.data_behav[:, 6] = filt_func(self._data_behav[:, 6])
        self.data_behav[:, 7] = filt_func(self._data_behav[:, 7])

    def resample_behavioral_data_gcamp_matched(self):
        """
        Resample behavioral data to match GCAMP data time steps.
        """

        if self.timestamp_behav[-1] < self.timestamp_gcamp[-1]:
            self.data_gcamp = self.data_gcamp[
                self.data_gcamp[:, 0] <= self.timestamp_behav[-1], :
            ]

        data_behav = np.nan * np.zeros(
            (len(self.timestamp_gcamp), self._data_behav.shape[1]))

        # handle heading signal separately since it will jump when crossing +- pi
        heading = self.heading.copy()

        # resample fictrac data
        data_behav[:, 1:], data_behav[:, 0] = signal.resample(
            self.data_behav[:, 1:], num=len(self.timestamp_gcamp),
            t=self.timestamp_behav.copy())
        # resample heading
        heading_unwrapped = unwrap_signal(heading, -180, 180)
        heading_unwrapped_resampled, _ = signal.resample(
            heading_unwrapped, num=len(self.timestamp_gcamp),
            t=self.timestamp_behav.copy())

        # re-wrap heading and store it
        heading_resampled = wrap_signal(heading_unwrapped_resampled, -180, 180)
        data_behav[:, 16] = heading_resampled
        self.data_behav = data_behav

        assert np.abs(data_behav[-1, 0] - self.timestamp_gcamp[-1]) < self.dt_gcamp

        # resample air tube motion if necessary
        if self.contains_air_tube_motion:
            # unwrap it
            air_tube_unwrapped = unwrap_signal(self.air_tube, -180, 180)
            # resample it
            air_tube_unwrapped_resampled = np.nan * np.zeros(self.timestamp_gcamp.shape)
            for ctr, t_gcamp in enumerate(self.timestamp_gcamp):
                t_min = t_gcamp - self.dt_gcamp
                t_max = t_gcamp + self.dt_gcamp
                mask = (t_min <= self.timestamp_air_tube) \
                    * (self.timestamp_air_tube < t_max)
                if np.sum(mask):
                    air_tube_unwrapped_resampled[ctr] \
                        = air_tube_unwrapped[mask].mean()

            # linearly interpolate nans
            for start, end in find_segs(np.isnan(air_tube_unwrapped_resampled)):
                if start == 0 or end == len(air_tube_unwrapped_resampled):
                    continue
                x_before = air_tube_unwrapped_resampled[start - 1]
                x_after = air_tube_unwrapped_resampled[end]

                x_middle = np.linspace(x_before, x_after, end-start + 2)[1:-1]
                air_tube_unwrapped_resampled[start:end] = x_middle

            # rewrap it
            air_tube_resampled = wrap_signal(
                air_tube_unwrapped_resampled, -180, 180)
            self.data_air_tube = air_tube_resampled
        else:
            self.data_air_tube = np.repeat(np.nan, len(self.timestamp_gcamp))
    
    def filt_random_noise(self):
        """
        Generate a Gaussian white noise signal filtered with an exponential filter.
        """

        filt_noise = self.filt_func(
            np.random.normal(0, 1, (len(self.timestamp_behav),)))

        return signal.resample(
            filt_noise, num=len(self.timestamp_gcamp), t=self.timestamp_behav)[0]


class DataLoader(DataLoaderBase):
    """
    Class for loading data files corresponding to closed loop experiments.
    """
    def __init__(self, trial, vel_filt):

        super(self.__class__, self).__init__()

        # load light time data
        path_light_times = os.path.join(
            L.DATA_ROOT, trial.path, trial.file_name_light_times)
        df_light_times = pd.read_excel(path_light_times, header=None)
        light_times = list(df_light_times.loc[:, 0])

        # load behavioral data
        path_behav = os.path.join(L.DATA_ROOT, trial.path, trial.file_name_behav)
        df_behav = pd.read_csv(path_behav, header=None)

        behav_array_raw = df_behav.as_matrix()[:, :-1].astype(float)
        frames_behav = behav_array_raw[:, 0]
        n_cols_behav = behav_array_raw.shape[1]

        # get the section of the behavioral matrix within the light times boundaries
        within_light_times_mask = (frames_behav >= light_times[0]) * \
            (frames_behav <= light_times[1])
        behav_array_raw_cut = behav_array_raw[within_light_times_mask, :]

        # create a new behavioral matrix that will have the proper size
        data_behav = np.nan * np.zeros(
            (light_times[1] - light_times[0] + 1, n_cols_behav),)
        data_behav[behav_array_raw_cut[:, 0].astype(int) - light_times[0], :] = \
            behav_array_raw_cut

        # now we have a matrix with the theoretical number of frames between the
        # two light times,
        # and which has nans for rows corresponding to skipped frames;
        # we now reset the frame counter relative to the first light time
        # and fill in the frame numbers
        # for the missing frames and convert frame counters to time stamps

        data_behav[:, 0] = np.arange(light_times[1] - light_times[0] + 1) * 1./60

        # interpolate nan values for vels and heading
        data_behav = _interpolate_nans(data_behav)

        # adjust heading so that 0 is directly upwind and convert to degrees
        data_behav[:, 16] -= np.pi
        data_behav[:, 16] *= (180/np.pi)

        # load gcamp roi data
        path_gcamp = os.path.join(L.DATA_ROOT, trial.path, trial.file_name_gcamp)
        df_gcamp = pd.read_csv(path_gcamp, header=None)
        gcamp_array = df_gcamp.as_matrix()[:, 2:].astype(float).T

        assert gcamp_array.shape[1] == 16

        # load gcamp timestamp data
        path_gcamp_timestamp = os.path.join(
            L.DATA_ROOT, trial.path, trial.file_name_gcamp_timestamp)
        df_gcamp_timestamp = pd.read_csv(path_gcamp_timestamp, header=None)
        gcamp_timestamp_array = df_gcamp_timestamp.as_matrix().flatten()

        # convert to relative time
        gcamp_timestamp_array -= gcamp_timestamp_array[0]

        # put them together into a single matrix
        data_gcamp = np.nan * np.zeros((len(gcamp_array), 17))

        data_gcamp[:, 0] = gcamp_timestamp_array
        data_gcamp[:, 1:] = gcamp_array

        # load air tube motion data
        if trial.file_name_air_tube:
            path_air_tube = os.path.join(
                L.DATA_ROOT, trial.path, trial.file_name_air_tube)
            df_air_tube = pd.read_csv(path_air_tube, header=None)
            temp = df_air_tube.as_matrix().T
            timestamp_air_tube = temp[:, 0] / 1000.  # convert to s from ms

            data_air_tube = temp[:, 1] - np.pi/2
            data_air_tube = data_air_tube * 180/np.pi
            self.contains_air_tube_motion = True
        elif trial.expt != 'motionless':
            df_air_tube = None
            # assume air tube is negative heading
            timestamp_air_tube = data_behav[:, 0].copy()
            data_air_tube = data_behav[:, 16].copy()
            self.contains_air_tube_motion = True
        else:
            # make air tube time-series all nans
            df_air_tube = None
            timestamp_air_tube = data_behav[:, 0].copy()
            data_air_tube = np.nan * np.zeros(data_behav[:, 16].shape)
            self.contains_air_tube_motion = False

        # line up the final timestamps for the gcamp, behavior, and air tube
        last_t = C.MAX_TRIAL_TIME

        # make sure no data is longer than last_t
        data_gcamp = data_gcamp[data_gcamp[:, 0] <= last_t, :]
        data_behav = data_behav[data_behav[:, 0] <= last_t, :]
        data_air_tube = data_air_tube[timestamp_air_tube <= last_t]

        timestamp_air_tube = timestamp_air_tube[timestamp_air_tube <= last_t]

        # store everything
        self.df_light_times = df_light_times
        self.df_behav = df_behav
        self.df_gcamp = df_gcamp
        self.df_gcamp_timestamp = df_gcamp_timestamp
        self.df_air_tube = df_air_tube

        self._data_behav = data_behav
        self.data_behav = data_behav.copy()
        self.data_gcamp = data_gcamp
        self._data_air_tube = data_air_tube
        self.data_air_tube = data_air_tube.copy()
        self.timestamp_air_tube = timestamp_air_tube

        self.dt_gcamp = np.mean(np.diff(self.data_gcamp[:, 0]))

        # store gcamp labels
        self.gcamp_labels = [
            'G2R', 'G3R', 'G4R', 'G5R',  # gcamp signal on right brain
            'G2L', 'G3L', 'G4L', 'G5L',  # gcamp signal on left brain
        ]

        # smooth data with velocity filter
        self.filt_vel_exp(vel_filt)
        
        # resample behavioral data to match gcamp timestamps
        self.resample_behavioral_data_gcamp_matched()
        
        # store walking speed threshold
        self.walking_threshold = trial.walking_threshold
        
    @property
    def timestamp_behav(self): return self._data_behav[:, 0]

    @property
    def timestamp_gcamp(self): return self.data_gcamp[:, 0]

    @property
    def red(self): return self.data_gcamp[:, 1:9]

    @property
    def green(self): return self.data_gcamp[:, 9:]

    @property
    def green_red_ratio(self): return self.green / self.red

    @property
    def gcamp(self): return self.green_red_ratio

    @property
    def G2L(self): return normalize_by_column(self.gcamp[:, 4])

    @property
    def G3L(self): return normalize_by_column(self.gcamp[:, 5])

    @property
    def G4L(self): return normalize_by_column(self.gcamp[:, 6])

    @property
    def G5L(self): return normalize_by_column(self.gcamp[:, 7])

    @property
    def G2R(self): return normalize_by_column(self.gcamp[:, 0])

    @property
    def G3R(self): return normalize_by_column(self.gcamp[:, 1])

    @property
    def G4R(self): return normalize_by_column(self.gcamp[:, 2])

    @property
    def G5R(self): return normalize_by_column(self.gcamp[:, 3])

    @property
    def G2S(self): return self.G2R + self.G2L
    
    @property
    def G3S(self): return self.G3R + self.G3L

    @property
    def G4S(self): return self.G4R + self.G4L

    @property
    def G5S(self): return self.G5R + self.G5L

    @property
    def G2D(self): return self.G2R - self.G2L

    @property
    def G3D(self): return self.G3R - self.G3L

    @property
    def G4D(self): return self.G4R - self.G4L

    @property
    def G5D(self): return self.G5R - self.G5L

    @property
    def air_tube(self): return self.data_air_tube
    
    @property
    def states(self): return get_states(self.speed, self.walking_threshold)
    

def _interpolate_nans(data):
    """
    Interpolate the values of the nans for vel (cols 5, 6, 7) and heading (col 16).
    :param data: data array
    :return: data array with nans interpolated
    """

    # find all nan segments
    nan_segs = find_segs(np.isnan(data[:, 1]))

    # loop over all nan groups and interpolate
    for nan_seg in nan_segs:

        # make up x-coordinates of surrounding data points
        xp = np.array([nan_seg[0] - 1, nan_seg[1]])

        # interpolate vel
        for col in range(5, 8):

            # get y-coordinates of surrounding data points
            fp = np.nan * np.zeros((2,))

            if xp[0] >= 0: fp[0] = data[xp[0], col]
            else: fp[0] = data[xp[1], col]

            if xp[1] < len(data): fp[1] = data[xp[1], col]
            else: fp[1] = data[xp[0], col]

            data[nan_seg[0]:nan_seg[1], col] = np.interp(np.arange(*nan_seg), xp, fp)

        # interpolate heading
        # get y-coordinates of surrounding data points
        fp = np.nan * np.zeros((2,))

        if xp[0] >= 0: fp[0] = data[xp[0], 16]
        else: fp[0] = data[xp[1], 16]

        if xp[1] < len(data): fp[1] = data[xp[1], 16]
        else: fp[1] = data[xp[0], 16]

        # if heading jumps between near pi and near 2*pi, transform before interpolating
        if np.abs(fp[1] - fp[0]) > (3 * np.pi / 2):
            fp[fp < 0] += (2 * np.pi)
            interp = np.interp(np.arange(*nan_seg), xp, fp)
            interp[interp > np.pi] -= (2 * np.pi)

        else:
            interp = np.interp(np.arange(*nan_seg), xp, fp)

        data[nan_seg[0]:nan_seg[1], 16] = interp

    return data


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


def normalize_by_column(data):
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


def standardize_std_by_column(data):
    """
    Divide each column of a data matrix by its std. Note that the mean
    is left as it is.
    """
    data_normed = np.nan * np.zeros(data.shape)

    if data.ndim == 1:
        data_normed = data / np.nanstd(data)

    else:
        for ctr in range(data.shape[1]):
            data_normed[:, ctr] = data[:, ctr] / np.nanstd(data[:, ctr])

    return data_normed


def normalize_if_not_heading(data, label):
    """
    Normalize an array if its label does not include 'heading' or 'wind_dir'.
    :param data: 1D array
    :param label: string
    """
    is_heading_var = ('heading' in label) or ('wind_dir' in label)

    if not is_heading_var: sig = normalize_by_column(data[:, None])[:, 0]
    else: sig = data.copy()

    return sig


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


def wrap_signal(x, x_min, x_max):
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


def unwrap_signal(x, x_min, x_max, jump_threshold=0.75):
    """
    Unwrap a modular time-series signal, e.g., so derivatives are meaningful.

    :param x: 1D array
    :param x_min: min value of time-series
    :param x_max: max value of time-series
    :param jump_threshold: threshold (as proportion of min/max range) for defining
        a min/max boundary crossing
    :return: unwrapped signal
    """

    x = x.copy()
    assert x_max > x_min

    x_range = x_max - x_min
    diff = np.diff(x)
    cc = np.concatenate

    # get all boundary crossings
    down_jumps = cc([[0.], (diff < (-jump_threshold * x_range)).astype(float)])
    up_jumps = cc([[0.], (diff > (jump_threshold * x_range)).astype(float)])

    down_ctrs = down_jumps.cumsum()
    up_ctrs = up_jumps.cumsum()

    offset_ctr = down_ctrs - up_ctrs
    x_shifted = x - x_min
    x_shifted += offset_ctr * x_range

    return x_shifted + x_min


def slice_by_time(t, x, t_ranges):
    """
    Slice a series of data according to a set of time-ranges.
    :param t: time vector
    :param x: data (2D array with rows as time points)
    :param t_ranges: time range for slicing each x
    :return: list of slices of x
    """

    x_sliced = []

    for t_range in t_ranges:
        x_sliced.append(x[(t >= t_range[0]) * (t < t_range[1])])

    return x_sliced


def normalize_by_column_multi(xs):
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


def lowpass_params_from_attr(attr):

    r = re.search('lowpass_([\d.]+)s_(.*)', attr)

    cut_t = float(r.group(1))
    remaining = r.group(2)

    return cut_t, remaining


def bandpass_params_from_attr(attr):

    r = re.search('bandpass_([\d.]+)s_([\d.]+)s_(.*)', attr)

    low = float(r.group(1))
    high = float(r.group(2))
    remaining = r.group(3)

    return low, high, remaining


def lmp_params_from_attr(attr):

    r = re.search('lmp\((.+)\|(.+)\)', attr)

    target = r.group(1)
    predictors = r.group(2).split(',')

    return target, predictors


def ar_params_from_attr(attr):

    r = re.search('ARP_(\d*)_(.*)', attr)

    n_steps = int(r.group(1))
    remaining = r.group(2)

    return n_steps, remaining


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

   
#### DEPRECATED (FOR LOADING OLD 4-COMPARTMENT DATASETS) ####

class DataLoaderDeprecated(DataLoaderBase):
    """
    Class for loading data files.
    """
    def __init__(self, trial):

        super(self.__class__, self).__init__()

        path_behav = os.path.join(L.DATA_ROOT, trial.path, trial.file_name_behav)
        self.df_behav = pd.read_csv(path_behav, header=None)
        self._data_behav = self.df_behav.as_matrix()[:, :-1].astype(float)

        # convert behavioral time stamps to seconds
        self._data_behav[:, 21] /= 1000

        path_gcamp = os.path.join(L.DATA_ROOT, trial.path, trial.file_name_gcamp)
        self.df_gcamp = pd.read_csv(path_gcamp, header=None)
        self.data_gcamp = self.df_gcamp.as_matrix().T.astype(float)

        # make sure that last behavioral time stamp is as close as possible to last
        # gcamp time stamp

        self.data_gcamp = self.data_gcamp[self.data_gcamp[:, 0] < C.MAX_TRIAL_TIME, :]
        self._data_behav = self._data_behav[self._data_behav[:, 21] < C.MAX_TRIAL_TIME, :]
        self.dt_gcamp = np.mean(np.diff(self.data_gcamp[:, 0]))

        # normalize heading to have 0 be "upwind" and to be in degrees
        self._data_behav[:, 16] -= np.pi
        self._data_behav[:, 16] *= (180 / np.pi)

        # make copy of data for public access
        self.data_behav = self._data_behav.copy()
        self.contains_air_tube_motion = False
        
        # smooth data with velocity filter
        self.filt_vel_exp(vel_filt)
        
        # resample behavioral data to match gcamp timestamps
        self.resample_behavioral_data_gcamp_matched()


    @property
    def timestamp_behav(self): return self._data_behav[:, 21]
    
    @property
    def timestamp_gcamp(self): return self.data_gcamp[:, 0]

    @property
    def red(self): return self.data_gcamp[:, 1:5]

    @property
    def green(self): return self.data_gcamp[:, 5:9]

    @property
    def green_red_ratio(self): return self.green / self.red

    @property
    def gcamp(self): return self.green_red_ratio

    @property
    def G2(self): return normalize_by_column(self.gcamp[:, 0])

    @property
    def G3(self): return normalize_by_column(self.gcamp[:, 1])

    @property
    def G4(self): return normalize_by_column(self.gcamp[:, 2])

    @property
    def G5(self): return normalize_by_column(self.gcamp[:, 3])

    @property
    def G2R(self): return normalize_by_column(self.gcamp[:, 0])

    @property
    def G3R(self): return normalize_by_column(self.gcamp[:, 1])

    @property
    def G4R(self): return normalize_by_column(self.gcamp[:, 2])

    @property
    def G5R(self): return normalize_by_column(self.gcamp[:, 3])
