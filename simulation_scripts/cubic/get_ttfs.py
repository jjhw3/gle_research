from pathlib import Path

import numpy as np

from common.tools import stable_fit_alpha

topdir = Path('/home/jjhw3/rds/hpc-work/gle/cubic')
times = np.arange(0, 5000, 0.01)

alphas = []

for i in range(1, 2):
    e_auto = np.load(topdir / f'{i}/total_energy_autocorrelation.npy')
    alphas.append(stable_fit_alpha(
        times,
        e_auto,
        np.array([1.0, 0]),
        0,
        plot_dir=topdir / f'{i}/plots'
    ))

print(alphas)
