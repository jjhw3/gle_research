import numpy as np


def fast_auto_correlate(arr):
    shape = arr.shape[0]
    arr = np.pad(arr, ((0, shape - 1),))
    correlated = np.abs(np.fft.ifft(np.abs(np.fft.fft(arr))**2)[:shape])
    return correlated / np.arange(shape, 0, -1)
