import numpy as np
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

    return fsolve(inner, 1)
