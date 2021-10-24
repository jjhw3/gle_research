from pathlib import Path

import numpy as np

dir = Path('/home/jjhw3/rds/hpc-work/gle/eta_temp_scan')
dks = np.linspace(0, 2.46, 50)
fit_mask = (dks > 1.23 - 0.5) & (dks < 1.23 + 0.5)
fit_dks = dks[fit_mask]
temps = np.array([140, 160, 180, 200, 225, 250, 275, 300])

hopping_rates = {}

for fil in dir.glob('*/*/combined_isfs/alphas.npy'):
    temp = int(fil.parents[2].name)
    eta = int(fil.parents[1].name)

    print(temp, eta)

    if eta not in hopping_rates:
        hopping_rates[eta] = np.zeros_like(temps)

    alphas = np.load(fil)
    poly = np.polyfit(fit_dks, alphas[fit_mask], 2)
    hopping_rates[eta][list(temps).index(temp)] = np.max(np.polyval(poly, fit_dks)) / 4