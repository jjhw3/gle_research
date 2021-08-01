import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common.constants import amu_K_ps_to_eV, hbar
from common.lattice_tools.common import mag, norm
from common.tools import fast_calculate_isf, stable_fit_alpha
from md.configuration import MDConfig

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    config = MDConfig.load(working_dir)

    times = config.times[::100]
    positions = np.load(working_dir / 'absorbate_positions.npy')[:, ::100]

    unit_dK = norm(config.in_plane_basis[:, 0])
    dK_mags = np.arange(0.00, 2.5, 0.05)
    alphas = np.zeros_like(dK_mags)

    for i, dK_mag in enumerate(dK_mags):
        isf = fast_calculate_isf(positions, unit_dK * dK_mag)
        np.save(config.isf_directory / f'{mag(dK_mag):.2}.npy', isf)

        # plt.plot(times, isf)
        # plt.xlim(0, 300)
        # plt.show()

        try:
            alphas[i] = stable_fit_alpha(
                times,
                isf,
                dK_mag * norm(unit_dK),
                1,
                t_0=0,
                tol=0.01,
                # plot_dir=config.isf_directory
            )
        except Exception as e:
            print(e)
            alphas[i] = np.nan

    np.save(working_dir / 'dk_mags.npy', dK_mags)
    np.save(working_dir / 'alphas_dk.npy', alphas)

    plt.plot(dK_mags, hbar * amu_K_ps_to_eV(alphas) * 1e6)
    plt.xlabel('dK')
    plt.ylabel('alpha')
    plt.savefig(working_dir / 'alpha_dk.png')
