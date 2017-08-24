"""
Classes and functions for handling data.
"""
import numpy as np
import os
import pandas as pd
import re
from scipy import signal
from scipy import stats
from sklearn import linear_model

import CONFIG as C
import filters
from models import make_extended_predictor_matrix

import LOCAL_SETTINGS as L


def _interpolate_nans(data):
    """
    Interpolate the values of the nans for vel (cols 5, 6, 7) and heading (col 16).
    :param data: data array
    :return: data array with nans interpolated
    """

    # find all nan segments
    nan_segs = find_segments(np.isnan(data[:, 1]))

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


def find_segments(x):
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


class DataLoaderBase(object):
    """
    Base class for loading data files.
    """

    air_flow_max_angle = 90
    air_flow_off_angle = 2.67 * 180 / np.pi

    odor_state_times = [
        ('before_short', [30, 90]),
        ('before', [0, 90]),
        ('during', [90, 150]),
        ('after_short', [150, 210]),
        ('after', [150, 300]),
    ]

    def __init__(self): pass

    def __getattr__(self, attr):
        """
        Modify attribute selection so that we can select attributes offset in time.
        """

        if len(attr) > 6 and attr.startswith('bouts_'):
            # set all values to nan if not in previously determined bouts
            if self.bout_mask is None:
                raise Exception('Trial has no marked bouts.')
            else:
                arr = getattr(self, attr[6:])
                arr[~self.bout_mask] = np.nan
                return arr

        if len(attr) > 11 and attr[:11] == 'odor_state_':
            temp = attr[11:]

            for odor_state, mask in self.odor_state_masks:
                if temp.startswith(odor_state):

                    arr = getattr(self, temp[len(odor_state)+1:])  # +1 for the "_"
                    arr[~mask] = np.nan
                    return arr

        if len(attr) > 4 and attr[:4] == 'wnm_':
            # white noise mask (second and fourth minutes)
            remaining = attr[4:]
            temp = getattr(self, remaining).copy()
            t_ = self.timestamp_gcamp
            mask = np.zeros(t_.shape, dtype=bool)
            # mask in second minute
            mask[((t_>=60)*(t_<120)).astype(bool)] = True
            # mask in fourth minute
            mask[((t_>=180)*(t_<240)).astype(bool)] = True
            # set values outside of mask to nans
            temp[~mask] = np.nan

            return temp

        if len(attr) > 5 and attr[:4] == 'ARP_':
            # perform autoregressive prediction

            n_steps, remaining = ar_params_from_attr(attr)

            # get original signal
            targ = getattr(self, remaining).copy()

            # build extended predictor matrix
            predictors = make_extended_predictor_matrix(
                targ[:, None], start=-n_steps, stop=0)

            # remove nans
            not_nan_0 = np.all(~np.isnan(predictors), axis=1)
            not_nan_1 = ~np.isnan(targ)
            not_nan = (not_nan_0 * not_nan_1).astype(bool)

            # fit model
            lm = linear_model.LinearRegression(fit_intercept=False)
            lm.fit(predictors[not_nan], targ[not_nan])

            # make predictions
            predictions = np.nan * np.zeros(targ.shape)
            predictions[not_nan] = lm.predict(predictors[not_nan])

            return predictions

        if len(attr) > 4 and attr[:4] == 'clm_':
            # closed loop mask (first, third, and fifth minutes)
            remaining = attr[4:]
            temp = getattr(self, remaining).copy()
            t_ = self.timestamp_gcamp
            mask = np.zeros(t_.shape, dtype=bool)
            # mask in first minute
            mask[(t_<60).astype(bool)] = True
            # mask in third minute
            mask[((t_>=120)*(t_<180)).astype(bool)] = True
            # mask in fifth minute
            mask[(t_>=240).astype(bool)] = True
            # set values outside of mask to nans
            temp[~mask] = np.nan

            return temp

        if len(attr) > 7 and attr[:7] == 'normed_':

            remaining = attr[7:]
            return normalize_by_column(getattr(self, remaining))

        if len(attr) > 4 and attr[:4] == 'abs_':

            remaining = attr[4:]
            return np.abs(getattr(self, remaining))

        if len(attr) > 4 and attr[:4] == 'ddt_':

            remaining = attr[4:]
            if remaining in ['heading', 'air_tube']:
                # unwrap before taking gradient
                data_unwrapped = unwrap_signal(getattr(self, remaining), -180, 180)
                return np.gradient(data_unwrapped) / self.dt_gcamp
            else:
                return np.gradient(getattr(self, remaining)) / self.dt_gcamp

        if len(attr) > 4 and attr[:4] == 'int_':

            remaining = attr[4:]
            return np.cumsum(getattr(self, remaining)) * self.dt_gcamp

        if len(attr) > 4 and attr[:4] == 'mean':
            # get difference between two normalized attributes

            r = re.search('mean\((.+),(.+)\)', attr)
            vals_0 = getattr(self, r.group(1))
            vals_1 = getattr(self, r.group(2))

            # set mean to 0 and std to 1
            vals_0_normed = (vals_0 - np.nanmean(vals_0)) / np.nanstd(vals_0)
            vals_1_normed = (vals_1 - np.nanmean(vals_1)) / np.nanstd(vals_1)

            return .5 * (vals_0_normed + vals_1_normed)

        if len(attr) > 4 and attr[:4] == 'diff':
            # get difference between two normalized attributes

            r = re.search('diff\((.+),(.+)\)', attr)
            vals_0 = getattr(self, r.group(1))
            vals_1 = getattr(self, r.group(2))

            # set mean to 0 and std to 1
            vals_0_normed = (vals_0 - np.nanmean(vals_0)) / np.nanstd(vals_0)
            vals_1_normed = (vals_1 - np.nanmean(vals_1)) / np.nanstd(vals_1)

            return vals_0_normed - vals_1_normed

        if len(attr) > 7 and attr[:7] == 'lowpass':
            # low pass filter the signal using a 5th order butterworth filter
            low, remaining = lowpass_params_from_attr(attr)

            high_cut = 1 / low

            return filters.butter_bandpass(
                getattr(self, remaining), lowcut=0, highcut=high_cut,
                fs=self.fs, order=5)

        if len(attr) > 8 and attr[:8] == 'bandpass':
            # bandpass filter the signal between two correlation times
            # using a 5th order butterworth filter

            # get bandpass parameters from attribute
            low, high, remaining = bandpass_params_from_attr(attr)

            # convert to low and high frequency cutoffs
            low_cut = 1 / high
            high_cut = 1 / low if low > 0 else np.inf

            # bandpass original signal
            return filters.butter_bandpass(
                getattr(self, remaining), low_cut, high_cut, self.fs, order=5)

        if len(attr) > 6 and attr[:6] == 'minus_':

            return -getattr(self, attr[6:])

        if len(attr) > 3 and attr[:3] == 'lmp':
            # get the best fit prediction from a linear model
            target, predictors = lmp_params_from_attr(attr)

            vals_t = getattr(self, target)
            vals_p = np.array([getattr(self, p) for p in predictors]).T

            mask = np.all(~np.isnan(np.concatenate(
                [vals_t[:, None], vals_p], axis=1)), axis=1)

            prediction = np.nan * np.zeros(vals_t.shape)

            lm = linear_model.LinearRegression()
            lm.fit(vals_p[mask], vals_t[mask])

            prediction[mask] = lm.predict(vals_p[mask])

            return prediction

        if len(attr) > 3 and attr[:2] == 'r(':
            # get moving correlation between two signals
            r = re.search('r\((.+),(.+),(\d+)s\)', attr)
            vals_0 = getattr(self, r.group(1))
            vals_1 = getattr(self, r.group(2))
            window = int(round(float(r.group(3)) / self.dt_gcamp))

            # compute the moving correlation in the slow, naive way
            back = int(np.floor(window/2))
            forw = int(np.ceil(window/2))

            rs = np.nan * np.zeros(vals_0.shape)
            for ts in range(len(vals_0)):

                start = max(0, ts - back)
                end = ts + forw

                vals_0_windowed = vals_0[start:end]
                vals_1_windowed = vals_1[start:end]
                r = stats.pearsonr(vals_0_windowed, vals_1_windowed)[0]

                rs[ts] = r

            return rs

        if '+' in attr:

            attr_name, offset = attr.split('+')[:2]
            offset = int(np.round(float(offset) / self.dt_gcamp))

            base_attr = getattr(self, attr_name)

            if offset == 0:
                return base_attr

            arr = np.nan * np.zeros(base_attr.shape)
            arr[:-offset] = base_attr[offset:]

            return arr

        elif '-' in attr:

            attr_name, offset = attr.split('-')[:2]
            offset = int(np.round(float(offset) / self.dt_gcamp))

            base_attr = getattr(self, attr_name)

            if offset == 0:
                return base_attr

            arr = np.nan * np.zeros(base_attr.shape)
            arr[offset:] = base_attr[:-offset]

            return arr

        elif 'control' in attr:
            return self.filt_random_noise()

        else:
            raise AttributeError(
                'Class \'DataLoader\' has no attribute \'{}\''.format(attr))

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

        wind_dir[wind_dir > self.air_flow_off_angle] = np.nan
        wind_dir[wind_dir < -self.air_flow_off_angle] = np.nan

        wind_dir[wind_dir > self.air_flow_max_angle] = self.air_flow_max_angle
        wind_dir[wind_dir < -self.air_flow_max_angle] = -self.air_flow_max_angle

        return wind_dir

    @property
    def wind_dir_front(self):

        wind_dir = self.heading.copy()

        wind_dir[wind_dir > self.air_flow_max_angle] = np.nan
        wind_dir[wind_dir < -self.air_flow_max_angle] = np.nan

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

                result = filters.exp(
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
            for start, end in find_segments(np.isnan(air_tube_unwrapped_resampled)):
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

    def make_bout_mask(self, trial):
        """
        Generate a mask from Ari's bout filters.
        """
        if trial.file_name_bouts is None: return None

        # load bout times and create mask from them
        path_bouts = os.path.join(L.DATA_ROOT, trial.path, trial.file_name_bouts)
        bouts = pd.read_csv(path_bouts, header=None).as_matrix().T

        t = self.timestamp_gcamp
        bout_mask = np.zeros(t.shape, dtype=bool)

        for bout in bouts: bout_mask[(t >= bout[0]) * (t < bout[1])] = True

        return bout_mask

    def make_odor_state_masks(self):
        """
        Generate masks for different odor states.
        """
        t = self.timestamp_gcamp

        masks = []
        for odor_state, (start, end) in self.odor_state_times:

            mask = np.zeros(t.shape, dtype=bool)
            mask[(start <= t) * (t < end)] = True
            masks.append((odor_state, mask))

        return masks

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

    def __init__(self, trial):

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

        self.bout_mask = self.make_bout_mask(trial)
        self.odor_state_masks = self.make_odor_state_masks()

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
    def G2D(self): return self.G2R - self.G2L

    @property
    def G3S(self): return self.G3R + self.G3L

    @property
    def G3D(self): return self.G3R - self.G3L

    @property
    def G4S(self): return self.G4R + self.G4L

    @property
    def G4D(self): return self.G4R - self.G4L

    @property
    def G5S(self): return self.G5R + self.G5L

    @property
    def G5D(self): return self.G5R - self.G5L

    @property
    def GLdsum(self): return self.G2Ld + self.G3Ld + self.G4Ld + self.G5Ld

    @property
    def Gdsum(self): return self.GRdsum + self.GLdsum

    @property
    def air_tube(self): return self.data_air_tube
    
    
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

        self.bout_mask = self.make_bout_mask(trial)
        self.odor_state_masks = self.make_odor_state_masks()

        self.contains_air_tube_motion = False

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

    @property
    def G2Rd(self): return signal.detrend(self.G2R)

    @property
    def G3Rd(self): return signal.detrend(self.G3R)

    @property
    def G4Rd(self): return signal.detrend(self.G4R)

    @property
    def G5Rd(self): return signal.detrend(self.G5R)

    @property
    def GRdsum(self): return self.G2Rd + self.G3Rd + self.G4Rd + self.G5Rd

    @property
    def G2Rd_std_1(self): return self.G2Rd / self.G2Rd.std()

    @property
    def G3Rd_std_1(self): return self.G3Rd / self.G3Rd.std()

    @property
    def G4Rd_std_1(self): return self.G4Rd / self.G4Rd.std()

    @property
    def G5Rd_std_1(self): return self.G5Rd / self.G5Rd.std()
