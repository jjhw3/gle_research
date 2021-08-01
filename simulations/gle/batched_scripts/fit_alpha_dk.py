import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common.lattice_tools.common import norm
from gle.batched_tools import calculate_isf_batched, calculate_alpha_dk_batched
from gle.configuration import ComplexTauGLEConfig
from gle.run_le import run_gle_batched

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    config = ComplexTauGLEConfig.load(working_dir)
    num_fitting_iterations = 10

    dk_unit = norm(np.asarray(config.in_plane_basis[:, 0]))
    dk_mags = np.load(working_dir / 'dk_mags.npy')
    target_alpha_dks = np.load(working_dir / 'target_alphas_dk.npy')
    plt.plot(dk_mags, target_alpha_dks, label='target')
    plt.legend()
    plt.show()

    has_overshot = False
    etas = np.zeros(num_fitting_iterations)
    etas[0] = config.eta
    errors = np.zeros_like(etas)

    for i in range(num_fitting_iterations):
        config.eta = etas[i]
        print(f'Running eta={etas[i]} {i+1} / {num_fitting_iterations}')

        run_gle_batched(
            config,
            10000,
        )

        gle_alphas = calculate_alpha_dk_batched(
            config,
            dk_unit,
            dk_mags,
        )

        err = gle_alphas - target_alpha_dks
        err[np.isnan(err)] = 0

        errors[i] = err.sum()
        plt.plot(dk_mags, target_alpha_dks, label='target')
        plt.plot(dk_mags, gle_alphas, label=f'iter {i}, eta={etas[i]:.2}')
        plt.legend()
        plt.show()

        if i == num_fitting_iterations - 1:
            break

        print(f'Error {errors[i]}')
        if not has_overshot and errors[i] < 0:
            etas[i+1] = etas[i] * 2
            continue

        has_overshot = True
        if i == 0:
            etas[i + 1] = etas[i] / 2
        elif np.sign(errors[i]) == np.sign(errors[i-1]):
            etas[i + 1] = etas[i - 1] + 2 * (etas[i] - etas[i - 1])
        else:
            grad = (errors[i] - errors[i-1]) / (etas[i] - etas[i-1])
            etas[i + 1] = etas[i - 1] + errors[i-1] / grad

        if etas[i + 1] < 0:
            etas[i + 1] = etas[i] / 2

    print('etas tried:', etas)
    print('errors:', errors)

    plt.legend()
    plt.show()
