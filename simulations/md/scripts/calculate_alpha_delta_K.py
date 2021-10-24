import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from common.constants import hbar, amu_K_ps_to_eV
from common.lattice_tools.common import norm
from common.tools import stable_fit_alpha


def get_alpha_dk(working_dir):
    times = np.arange(1000000) * 0.01
    dk_alphas = []

    for fil in working_dir.glob('*.npy'):
        print(fil)
        if fil.name[:-4] == 'alphas':
            continue

        dk = float(fil.name[:-4])
        isf = np.load(fil)
        isf /= isf[0]

        np.argmin(np.abs(isf - 0.1))

        alpha = stable_fit_alpha(
            times,
            isf,
            norm(np.asarray([1, 1, 0])) * dk,
            1,
            t_0=1,
            t_final=None,
            tol=0.1,
            # plot_dir=working_dir / 'plots'
        )

        dk_alphas.append((dk, alpha))

    dk_alphas.sort(key=lambda x: x[0])
    dks, alphas = zip(*dk_alphas)
    dks = np.array(dks)
    # alphas = 1e6 * amu_K_ps_to_eV(hbar * np.array(alphas))
    alphas = np.array(alphas)
    return dks, alphas


if __name__ == '__main__':
    for temp in [140, 160, 180, 200, 225, 250, 275, 300]:
        working_dir = Path('/Users/jeremywilkinson/research_data/md_data/32_combined_isfs') / f'{temp}' / '1.00_0.00_0.00'
        dks, alphas = get_alpha_dk(working_dir)
        np.save(working_dir / 'alphas.npy', alphas)
        plt.plot(dks, alphas, label=f'{temp}')

    plt.legend(loc='upper right')
    plt.show()
