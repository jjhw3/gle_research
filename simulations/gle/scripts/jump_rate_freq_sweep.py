import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from gle.result_checks import plot_power_spectrum, plot_path_on_crystal
from common.thermodynamics import sample_temperature, jump_count
from gle.configuration import ComplexTauGLEConfig
from gle.run_le import run_gle
from gle.theoretics import calculate_kernel_temperature_normalization

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    config = ComplexTauGLEConfig.load(working_dir)
    initial_position = np.asarray([1.2783537, 0.72844401])
    w_1s = np.linspace(0, 10, 10)
    # w_1s = [4.444444444444445]

    jump_rates = np.zeros_like(w_1s)
    temps = np.zeros_like(w_1s)

    for i, w_1 in enumerate(w_1s):
        config.w_1 = w_1
        print('w_1:', w_1)

        if config.tau == 0:
            config.discrete_decay_factor = 1.0
            config.memory_kernel_normalization = 0.0
        else:
            config.discrete_decay_factor = np.exp(- config.dt / config.tau) * np.exp(1j * w_1 * config.dt)
            config.memory_kernel_normalization = np.real(1 / (1 - config.discrete_decay_factor))
            config.memory_kernel_normalization *= calculate_kernel_temperature_normalization(config)

        kernel = np.real(np.power(config.discrete_decay_factor, np.arange(config.num_iterations)))
        kernel /= config.memory_kernel_normalization
        kernel_fft = np.fft.fft(kernel)
        ws = 2 * np.pi * np.fft.fftfreq(config.num_iterations, config.dt)
        dw = ws[1] - ws[0]

        greens_fft = kernel_fft / (-ws**2 + 1j * config.eta * ws * kernel_fft)
        pre_integral = ws**2 * np.abs(greens_fft) ** 2
        pre_integral[0] = np.abs(kernel_fft[0] / 1j * config.eta * kernel_fft[0])**2 # Fixes limit at origin

        temp_mult = 2 * config.eta * np.sum(dw * pre_integral) / 2 / np.pi
        print('Uncorrected temperature estimate', temp_mult * config.temperature)

        # config.memory_kernel_normalization *= temp_mult

        results = run_gle(
            config,
        )

        temps[i] = sample_temperature(results)
        print('Temp:', temps[i])
        jump_rates[i] = jump_count(results) / config.run_time
        print('jump rate:', jump_rates[i])

        plt.axvline(2 * np.pi * jump_rates[i], color='red')
        plt.axvline(w_1)
        plot_power_spectrum(results, show=True)
        plot_path_on_crystal(results)

    plt.plot(w_1s, temps)
    plt.show()
    plt.plot(w_1s, jump_rates)
    plt.show()
