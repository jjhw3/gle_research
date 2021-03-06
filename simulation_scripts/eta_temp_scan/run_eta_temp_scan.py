import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common.constants import hbar, amu_K_ps_to_eV
from common.lattice_tools.common import norm
from common.tools import fast_calculate_isf, stable_fit_alpha, FitFailedException
from gle.configuration import ComplexTauGLEConfig
from gle.run_le import run_gle_batched, run_gle


if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    config = ComplexTauGLEConfig.load(working_dir)

    dk_unit = norm(np.array([1, 0]))
    dk_mags = np.linspace(0, 2.46, 50)
    times = config.times[::10]
    save_mask = times < 5000

    isfs = {}

    for i in range(100):
        print(f'{i} / 100')
        results = run_gle(config)
        positions = results.positions[:, ::10]
        del results

        for i, dk_mag in enumerate(dk_mags):
            print(i)
            dk = dk_mag * dk_unit

            if dk_mag not in isfs:
                isfs[dk_mag] = np.zeros_like(times)

            isfs[dk_mag] += fast_calculate_isf(positions, dk)

    isfs_dir = config.working_directory / 'combined_isfs'
    isfs_dir.mkdir()

    for dk in isfs:
        isfs[dk] /= isfs[dk][0]
        np.save(isfs_dir / f"{dk}.npy", isfs[dk][save_mask])

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
                plot_dir=config.isf_directory
            )
        except FitFailedException as e:
            alpha = np.nan
        alphas[i] = alpha

    np.save(isfs_dir / 'alphas.npy', alphas)
