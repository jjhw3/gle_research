import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from gle.configuration import ComplexTauGLEConfig
from gle.theoretics import calculate_kernel_temperature_normalization

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    config = ComplexTauGLEConfig.load(working_dir)
    config.run_time = 10000
    config.eta = 1.0
    config.tau = 1.0
    config.w_1 = 0.0
    taus = np.linspace(0, 50, 10)
    w_1s = np.linspace(0, 20, 20)

    norm_adjustments = np.zeros_like(taus)

    for i, tau in enumerate(taus):
        print(i)
        config.tau = tau
        config.calculate_time_quantities()
        config.memory_kernel_normalization = 1 / np.real(1 - config.discrete_decay_factor)
        norm_adjustments[i] = calculate_kernel_temperature_normalization(config, w0=10)

    plt.plot(taus, np.abs(norm_adjustments))
    plt.ylim(0, 1.1)
    plt.show()
