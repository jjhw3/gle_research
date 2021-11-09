from pathlib import Path

import numpy as np

dir = Path('/Users/jeremywilkinson/research_data/gle_data/eta_temp_scan')
dks = np.linspace(0, 2.46, 50)
fit_mask = (dks > 1.23 - 0.5) & (dks < 1.23 + 0.5)
fit_dks = dks[fit_mask]
temps = np.array([140, 160, 180, 200, 225, 250, 275, 300])

dephasing_rates = {}

for fil in dir.glob('*/*/combined_isfs/alphas.npy'):
    temp = int(fil.parents[2].name)
    eta = float(fil.parents[1].name)

    print(temp, eta)

    if eta not in dephasing_rates:
        dephasing_rates[eta] = np.zeros(temps.shape[0])

    alphas = np.load(fil)
    poly = np.polyfit(fit_dks, alphas[fit_mask], 2)
    dephasing_rates[eta][list(temps).index(temp)] = np.max(np.polyval(poly, fit_dks))

sorted_lists = list(dephasing_rates.items())
sorted_lists.sort(key=lambda x: x[0])
etas = np.array(list(zip(*sorted_lists))[0])
dephasing_rate_grid = np.array(list(zip(*sorted_lists))[1])

np.save(dir / 'etas.npy', etas)
np.save(dir / 'dephasing_rate_grid.npy', dephasing_rate_grid)
