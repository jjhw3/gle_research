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
    num_fitting_iterations = int(sys.argv[2])
    fit_dir = working_dir / 'alpha_dk_fit'
    if fit_dir.exists():
        fit_dir.unlink()
    fit_dir.mkdir()

    print(sys.argv[1])
    config = ComplexTauGLEConfig.load(working_dir)

    dk_unit = norm(np.asarray(config.in_plane_basis[:, 0]))
    dk_mags = np.load(working_dir / 'dk_mags.npy')
    target_alpha_dks = np.load(working_dir / 'target_alphas.npy')
    # plt.plot(dk_mags, target_alpha_dks, label='target')
    # plt.legend()
    # plt.show()

    has_overshot = False
    etas = np.zeros(num_fitting_iterations)
    etas[0] = config.eta
    errors = np.zeros_like(etas)

    alpha_dks = np.zeros((num_fitting_iterations, dk_mags.shape[0]))

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

        errors[i] = err.mean()
        # plt.plot(dk_mags, target_alpha_dks, label='target')
        # plt.plot(dk_mags, gle_alphas, label=f'iter {i}, eta={etas[i]:.2} err={errors[i]:.2}')
        # plt.legend()
        # plt.show()

        alpha_dks[i] = gle_alphas

        if i == num_fitting_iterations - 1:
            break

        print(f'Error {errors[i]}')
        if i < 1:
            etas[i + 1] = etas[i] / 2
        else:
            m, c = np.polyfit(etas[max(0, i-5):i], errors[max(0, i-5):i], 1)
            etas[i + 1] = - c / m

        if etas[i + 1] < 0:
            etas[i + 1] = etas[i] / 2

    print('etas tried:', etas)
    print('errors:', errors)

    plt.plot(dk_mags, target_alpha_dks, label='target', c='black')
    for i in range(num_fitting_iterations):
        plt.plot(dk_mags, alpha_dks[i], label=f'iter {i}, eta={etas[i]:.2} err={errors[i]:.2}')

    np.save(fit_dir / 'alpha_dks.npy', alpha_dks)
    np.save(fit_dir / 'etas.npy', etas)
    np.save(fit_dir / 'errors.npy', errors)

    plt.legend()
    plt.savefig(fit_dir / 'all_fits.png')
    # plt.show()
    print()
