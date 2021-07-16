import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from gle.result_checks import plot_power_spectrum
from common.thermodynamics import sample_temperature
from gle.configuration import ComplexTauGLEConfig
from gle.run_le import run_gle

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    config = ComplexTauGLEConfig.load(working_dir)
    initial_position = np.asarray([1.2783537, 0.72844401])
    num_iterations = 100

    cum_fft = np.zeros((2, config.num_iterations))

    for i in range(num_iterations):
        results = run_gle(
            config,
            initial_position,
            'complex'
        )
        print('Observed temperature: ', sample_temperature(config, results))
        ws, fft = plot_power_spectrum(config, results)
        cum_fft += fft

    plt.scatter(ws, cum_fft[0] / num_iterations, s=3)
    plt.scatter(ws, cum_fft[1] / num_iterations, s=3)
    plt.plot(ws, config.noise_stddev ** 2 * config.num_iterations * 1 / np.abs(1 + 1j * ws * config.tau) ** 2)
    plt.xlim(- 10 / config.tau, 10 / config.tau)
    plt.show()

    # results.save()

    print()
