import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common.constants import hbar, amu_K_ps_to_eV
from common.tools import stable_fit_alpha

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])

    times = np.arange(0, 10000, 1)
    dK_unit = np.asarray([1.0, 0, 0])

    for temp_dir in working_dir.glob('*/'):
        if not temp_dir.is_dir():
            continue

        isfs = defaultdict(list)

        for ind_dir in temp_dir.glob('*/'):
            if not ind_dir.is_dir():
                continue

            for isf_fil in ind_dir.glob('ISFs/*.npy'):
                print(isf_fil)
                isf = np.load(isf_fil)
                dk_mag = float(isf_fil.stem)

                isfs[dk_mag].append(isf)

        dk_mags = np.asarray(list(isfs.keys()))
        dk_mags.sort()
        mean_isfs = np.asarray([np.mean(isfs[dk_mag], axis=0) for dk_mag in dk_mags])
        alphas = np.zeros(mean_isfs.shape[0])

        if not (temp_dir / 'ISFs').exists():
            (temp_dir / 'ISFs').mkdir()

        if not (temp_dir / 'ISFs/log').exists():
            (temp_dir / 'ISFs/log').mkdir()

        for i, dk_mag in enumerate(dk_mags):
            try:
                alphas[i] = stable_fit_alpha(
                    times,
                    mean_isfs[i],
                    dK_unit * dk_mag,
                    1,
                    t_0=0,
                    tol=0.01,
                    # plot_dir=temp_dir / 'ISFs'
                )
            except:
                print(f'unable to fit {dk_mag}')
                alphas[i] = np.nan

        np.save(temp_dir / 'dk_mags.npy', dk_mags)
        np.save(temp_dir / 'alphas.npy', alphas)
        plt.plot(dk_mags, hbar * amu_K_ps_to_eV(alphas) * 1e6, label=temp_dir.name)
        # plt.show()
        # print()

    plt.legend()
    plt.ylim(0, 120)
    plt.xlim(0, 3.5)
    plt.show()

    print()