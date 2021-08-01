import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common.constants import amu_K_ps_to_eV, planck_constant, hbar
from common.lattice_tools.common import norm
from common.tools import get_alpha, stable_fit_alpha
from gle.configuration import ComplexTauGLEConfig
from md.configuration import MDConfig

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    config = MDConfig.load(working_dir)

    times = config.times[::100]
    positions = np.load(working_dir / 'absorbate_positions.npy')[:, ::100]

    unit_dK = config.in_plane_rotation_matrix.dot(config.conventional_cell)[:, [0, 1]].sum(axis=1)
    # unit_dK = config.in_plane_basis[:, 0]
    dK_mags = np.arange(0.00, 2.5, 0.05)[1:-1]
    alphas = np.zeros_like(dK_mags)

    for i, dK_mag in enumerate(dK_mags):
        print(i, '/', len(dK_mags))
        alphas[i] = stable_fit_alpha(
            times,
            positions,
            dK_mag * norm(unit_dK),
            200,
            5,
            t_0=50,
            tol=0.01,
            plot_dir=config.isf_directory
        )

    plt.plot(dK_mags, amu_K_ps_to_eV(hbar * alphas) * 1e6)
    plt.show()

    print()
