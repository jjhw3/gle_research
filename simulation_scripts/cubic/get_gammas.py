from pathlib import Path

import numpy as np

from common.tools import stable_fit_alpha

topdir = Path('/home/jjhw3/rds/hpc-work/gle/cubic')
times = np.arange(0, 5000, 0.01)

gammas = []
temperatures = []
dks = np.linspace(0, 2.46, 50)
fit_mask = (dks > 1.23 - 0.5) & (dks < 1.23 + 0.5)
fit_dks = dks[fit_mask]

for i in range(0, 26):
    alphas = np.load(topdir / f'{i}/combined_isfs/alphas.npy')
    temps = np.load(topdir / f'{i}/temperatures.npy')

    poly = np.polyfit(fit_dks, alphas[fit_mask], 2)
    gammas.append(np.max(np.polyval(poly, fit_dks)))
    temperatures.append(np.mean(temps))

print(gammas)
np.save(topdir / 'cubic_gamma_maxes.npy', np.asarray(gammas))
np.save(topdir / 'temperatures.npy', np.asarray(temperatures))
