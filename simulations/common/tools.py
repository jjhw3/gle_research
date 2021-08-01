import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from common.lattice_tools.common import mag
from common.lattice_tools.plot_tools import cla


def fast_auto_correlate(arr):
    shape = arr.shape[0]
    arr = np.pad(arr, ((0, shape - 1),))
    correlated = np.abs(np.fft.ifft(np.abs(np.fft.fft(arr))**2)[:shape])
    return correlated / np.arange(shape, 0, -1)


def fast_correlate(arr1, arr2):
    shape = arr1.shape[0]
    arr1 = np.pad(arr1, ((0, shape - 1),))
    arr2 = np.pad(arr2, ((0, shape - 1),))
    correlated = np.fft.ifft(np.fft.fft(arr1) * np.fft.fft(arr2).conj())[:shape].real
    return correlated / np.arange(shape, 0, -1)


def fast_calculate_isf(positions, delta_K):
    amplitudes = np.exp(-1j * np.tensordot(positions, delta_K, axes=(0, 0)))
    isf = fast_auto_correlate(amplitudes)
    isf /= isf[0]
    return isf


def get_alpha(
    times,
    positions,
    delta_K,
    fit_times,
    plot=False,
):
    isf = fast_calculate_isf(positions, delta_K)
    fit_mask = (times > fit_times[0]) & (times < fit_times[1])

    p0 = np.polyfit(times[fit_mask], np.log(isf[fit_mask]), 1)

    def fit_exp(t, A, alpha):
        return A * np.exp(-alpha * t)

    popt, pcov = curve_fit(
        fit_exp,
        times[fit_mask],
        isf[fit_mask],
        p0=[np.exp(p0[1]), -p0[0]],
        bounds=[(-np.inf, -np.inf), (np.inf, np.inf)]
    )

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(times, isf)
        plt.plot(times[fit_mask], isf[fit_mask])
        plt.plot(times[fit_mask], fit_exp(times[fit_mask], *popt))
        plt.xlim(0, fit_times[1] + (fit_times[1] - fit_times[0]))
        # plt.yscale('log')
        plt.show()

    return popt[1]


def stable_fit_alpha(
    times,
    positions,
    delta_K,
    t_final,
    step,
    t_0=0,
    tol=0.01,
    plot_dir=None
):
    prev_alpha = None

    isf = fast_calculate_isf(positions, delta_K)

    while t_0 < t_final:
        fit_mask = (times > t_0) & (times < t_final)
        p0 = np.polyfit(times[fit_mask], np.log(isf[fit_mask]), 1)

        # popt, pcov = curve_fit(
        #     fit_exp,
        #     times[fit_mask],
        #     isf[fit_mask],
        #     p0=[np.exp(p0[1]), -p0[0]],
        # )

        alpha = - p0[0]

        if prev_alpha is not None:
            print(np.abs(alpha / prev_alpha - 1))
            if np.abs(alpha / prev_alpha - 1) < tol:
                if plot_dir is not None:
                    plot_mask = times < 2.0 * t_final
                    plt.plot(times[plot_mask], isf[plot_mask])
                    plt.plot(times[fit_mask], np.exp(- alpha * times[fit_mask] + p0[1]))
                    plt.savefig(plot_dir / f'{mag(delta_K):.2}.png')
                    plt.yscale('log')
                    plt.savefig(plot_dir / f'log/{mag(delta_K):.2}.png')
                    cla()

                return alpha

        prev_alpha = alpha
        t_0 += step
    raise Exception('Unable to fit alpha')