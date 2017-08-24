from __future__ import division
import numpy as np
from scipy import signal


def exp(t, x, amp, tau, t_max_filt):
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


# from scipy cookbook
def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if lowcut == 0:
        # low pass filter
        b, a = signal.butter(order, high, btype='low')
    elif highcut >= fs:
        # high pass filter
        b, a = signal.butter(order, low, btype='high')
    else:
        # bandpass filter
        b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass(data, lowcut, highcut, fs, order=5):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y
