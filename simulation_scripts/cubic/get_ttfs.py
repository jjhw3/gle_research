from pathlib import Path

import numpy as np

from common.tools import stable_fit_alpha

topdir = Path('/home/jjhw3/rds/hpc-work/gle/cubic')
times = np.arange(0, 5000, 0.01)

ttfs = []

for i in range(0, 26):
    e_auto = np.load(topdir / f'{i}/total_energy_autocorrelation.npy')
    e_auto -= np.mean(e_auto[(times > 50) & (times < 100)])
    e_auto /= e_auto[0]

    ttfs.append(1 / times[np.where(e_auto < 1 / np.e)[0][0]])
    # ttfs.append(stable_fit_alpha(
    #     times,
    #     e_auto,
    #     np.array([1.0, 0]),
    #     0,
    #     plot_dir=topdir / f'{i}/plots'
    # ))

print(ttfs)
np.save(topdir / 'cubic_ttfs.npy', np.asarray(ttfs))
