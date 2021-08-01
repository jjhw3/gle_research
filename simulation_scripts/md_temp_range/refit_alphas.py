import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common.constants import hbar, amu_K_ps_to_eV
from common.tools import stable_fit_alpha

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])

    times = np.arange(0, 100000, 1)
    dK_unit = np.asarray([1.0, 0, 0])

    for temp_dir in working_dir.glob('*/'):
        if not temp_dir.is_dir():
            continue

        for ind_dir in temp_dir.glob('*/'):
            if not ind_dir.is_dir():
                continue

            dK_mags = []
            alphas = []

            for isf_fil in ind_dir.glob('ISFs/*.npy'):
                print(isf_fil)
                isf = np.load(isf_fil)
                dk_mag = float(isf_fil.stem)
                try:
                    alpha = stable_fit_alpha(
                        times,
                        isf,
                        dK_unit * dk_mag,
                        1.0,
                        t_0=0,
                        tol=0.01,
                        plot_dir=isf_fil.parent
                    )
                except:
                    alpha = np.nan

                dK_mags.append(dk_mag)
                alphas.append(alpha)

            alphas = np.asarray(alphas)[np.argsort(dK_mags)]
            dK_mags = np.asarray(dK_mags)[np.argsort(dK_mags)]

            np.save(ind_dir / 'dk_mags.npy', dK_mags)
            np.save(ind_dir / 'alphas_dk.npy', alphas)

            plt.plot(dK_mags, hbar * amu_K_ps_to_eV(alphas) * 1e6)
            plt.xlabel('dK')
            plt.ylabel('alpha')
            plt.savefig(temp_dir / 'alpha_dk.png')
            plt.show()

            print()
