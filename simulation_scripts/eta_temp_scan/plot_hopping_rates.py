from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLines
from scipy.interpolate import interp2d

from common.constants import amu_K_ps_to_eV, hbar
from md.scripts.extract_jump_rates import get_jump_rates

temps = np.array([140, 160, 180, 200, 225, 250, 275, 300])
sizes = np.array([8, 16, 32])
etas = np.load('/Users/jeremywilkinson/research_data/gle_data/eta_temp_scan/etas.npy')
hopping_grid = np.load('/Users/jeremywilkinson/research_data/gle_data/eta_temp_scan/hopping_rate_grid.npy')
TEMP, ETA = np.meshgrid(temps, etas)
interp = interp2d(TEMP, hopping_grid, ETA)
size_jump_rates = {}
eta_jump_rate_fits = {}


for i, temp in enumerate(temps):
    plt.scatter(etas, hopping_grid[:, i], s=2)
    poly = np.polyfit(etas, hopping_grid[:, i], 1)
    plt.plot(etas, np.polyval(poly, etas), label=f"LE {temp}K")
    eta_jump_rate_fits[temp] = poly

size_fit_etas = {}

for size in sizes:
    size_jump_rates[size] = get_jump_rates(Path(f'/Users/jeremywilkinson/research_data/md_data/{size}_combined_isfs'))[1]

    fit_etas = np.zeros(temps.shape)
    for i, temp in enumerate(temps):
        # plt.axhline(size_jump_rates[size][i], label=f'{temp}', color='black')
        m, c = eta_jump_rate_fits[temp]
        fit_etas[i] = (size_jump_rates[size][i] - c) / m

    size_fit_etas[size] = fit_etas
    plt.plot(fit_etas, size_jump_rates[size], 'o-', color='black', label=f'size {size}^3')

labelLines(plt.gca().get_lines(), zorder=2.5)
plt.xlabel('eta (THz)')
plt.ylabel('jump rate (THz)')
plt.show()
print()


for size in sizes:
    plt.scatter(temps, size_fit_etas[size], label=f'size {size}^3')
    m, c = np.polyfit(temps, size_fit_etas[size], 1)
    plt.plot(temps, m * temps + c)
    print(size, m)

plt.legend()
plt.show()
