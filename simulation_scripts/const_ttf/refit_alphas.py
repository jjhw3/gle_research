import sys
from pathlib import Path

import numpy as np

from common.lattice_tools.common import norm
from common.tools import stable_fit_alpha, FitFailedException

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])

    isfs_dir = working_dir / 'combined_isfs'

    times = np.arange(0, 5000, 0.01)

    dk_unit = norm(np.array([1, 0]))
    dk_mags = np.linspace(0, 2.46, 50)

    isfs = {}
    for i, dk in enumerate(dk_mags):
        isfs[dk] = np.load(isfs_dir / f"{dk}.npy")

    alphas = np.zeros(len(isfs))

    for i, dk in enumerate(dk_mags):
        try:
            alpha = stable_fit_alpha(
                times,
                isfs[dk],
                dk,
                0,
                t_0=1,
                t_final=None,
                tol=0.1,
                plot_dir=isfs_dir / 'plots'
            )
        except FitFailedException as e:
            alpha = np.nan
        print(i, dk, alpha)
        alphas[i] = alpha

    np.save(isfs_dir / 'alphas.npy', alphas)
