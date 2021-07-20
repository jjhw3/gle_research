import numpy as np


def fast_auto_correlate(arr):
    shape = arr.shape[0]
    arr = np.pad(arr, ((0, shape - 1),))
    correlated = np.abs(np.fft.ifft(np.abs(np.fft.fft(arr))**2)[:shape])
    return correlated / np.arange(shape, 0, -1)


def fast_calculate_isf(positions, delta_K):
    amplitudes = np.exp(-1j * np.tensordot(positions, delta_K, axes=(0, 0)))
    isf = fast_auto_correlate(amplitudes)
    isf /= isf[0]
    return isf
