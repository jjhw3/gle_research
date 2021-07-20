import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def calculate_kernel_temperature_normalization(config, w0=0):
    if config.tau == 0:
        return 1

    kernel_length_ps = 100 * config.tau
    num_steps = int(np.round(kernel_length_ps / config.dt))
    kernel = np.real(np.power(config.discrete_decay_factor, np.arange(num_steps))) / config.memory_kernel_normalization
    kernel_fft = np.fft.fft(kernel)
    ws = 2 * np.pi * np.fft.fftfreq(num_steps, config.dt)
    dw = ws[1] - ws[0]

    def inner(norm):
        greens_fft = kernel_fft / norm / (- ws ** 2 + 1j * config.eta * ws * kernel_fft / norm + w0**2)
        pre_integral = ws ** 2 * np.abs(greens_fft) ** 2
        if w0 == 0:
            pre_integral[0] = 1 / config.eta**2

        temp_mult = 2 * config.eta * np.sum(dw * pre_integral) / 2 / np.pi
        return np.abs(temp_mult - 1)

    return fsolve(inner, 1)[0]


def abs_real_polynomial(poly):
    roots = poly.roots()
    roots = np.append(roots, roots.conj())
    if len(roots) == 0:
        return np.polynomial.Polynomial([np.abs(poly.trim().coef[-1]) ** 2])
    return np.abs(poly.coef[-1]) ** 2 * np.polynomial.Polynomial.fromroots(roots)


def divide_out_root(poly, root_i):
    all_roots = list(poly.roots())
    new_roots = all_roots[:root_i] + all_roots[root_i+1:]
    if len(new_roots) == 0:
        return np.polynomial.Polynomial([poly.trim().coef[-1]])
    return poly.coef[-1] * np.polynomial.Polynomial(new_roots)

def calculate_theoretical_msd(config):
    times = config.times

    tau = config.tau
    eta = config.eta
    w_1 = config.w_1

    K_1 = np.polynomial.Polynomial([1, 1j * tau])
    abs_K_1 = abs_real_polynomial(K_1)
    K_2 = np.polynomial.Polynomial([1 + (w_1*tau)**2, 2 * 1j * tau, -tau**2])
    P = np.polynomial.Polynomial([0, 1]) * K_2 - 1j * eta * K_1

    msd = np.zeros_like(times, dtype=np.complex128)
    roots = list(P.roots())

    for i, z in enumerate(roots):
        conjugate_root = np.polynomial.Polynomial([- np.conj(z), 1])
        root_free_P = divide_out_root(P, i)
        pole_free_abs_P = abs_real_polynomial(root_free_P) * conjugate_root

        z_term = np.real((times / z - (np.exp(1j * z * times) - 1) / 1j / z**2) / pole_free_abs_P(z))

        # z_term = np.real(times / z / pole_free_abs_P(z))
        z_term = z_term * abs_K_1(z)

        msd += z_term

    msd *= - 2 * config.noise_stddev**2 / config.absorbate_mass**2

    return msd


if __name__ == '__main__':
    from gle.configuration import ComplexTauGLEConfig

    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    config = ComplexTauGLEConfig.load(working_dir)

    plot_time = 20
    plot_mask = config.times < 10
    msd = calculate_theoretical_msd(config)
    plt.plot(config.times[plot_mask], msd[plot_mask])
    plt.show()
