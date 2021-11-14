from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLines
from scipy.interpolate import interp2d

from common.constants import amu_K_ps_to_eV, hbar
from md.scripts.extract_jump_rates import get_dephasing_rates

temps = np.array([140, 160, 180, 200, 225, 250, 275, 300])
sizes = np.array([8, 16, 32])
etas = np.load('/Users/jeremywilkinson/research_data/gle_data/eta_temp_scan/etas.npy')
hopping_grid = np.load('/Users/jeremywilkinson/research_data/gle_data/eta_temp_scan/dephasing_rate_grid.npy')
TEMP, ETA = np.meshgrid(temps, etas)
interp = interp2d(TEMP, hopping_grid, ETA)
dephasing_rates = {}
eta_jump_rate_fits = {}


for i, temp in enumerate(temps):
    plt.scatter(etas, hopping_grid[:, i], s=2)
    poly = np.polyfit(etas, hopping_grid[:, i], 1)
    plt.plot(etas, np.polyval(poly, etas), label=f"LE {temp}K")
    eta_jump_rate_fits[temp] = poly

size_fit_etas = {}

for size in sizes:
    dephasing_rates[size] = get_dephasing_rates(Path(f'/Users/jeremywilkinson/research_data/md_data/{size}_combined_isfs'))[1]

    fit_etas = np.zeros(temps.shape)
    for i, temp in enumerate(temps):
        # plt.axhline(size_jump_rates[size][i], label=f'{temp}', color='black')
        m, c = eta_jump_rate_fits[temp]
        fit_etas[i] = (dephasing_rates[size][i] - c) / m

    size_fit_etas[size] = fit_etas
    plt.plot(fit_etas, dephasing_rates[size], 'o-', color='black', label=rf'${size}^3$')

labelLines(plt.gca().get_lines(), zorder=2.5)
plt.xlabel('eta (THz)')
plt.ylabel(r'Peak long-time ISF decay rate $\Gamma_{MAX}$ (THz)')
plt.subplots_adjust(left=0.1, bottom=0.13, right=0.99, top=0.986, wspace=0.105)
plt.gcf().set_size_inches(8, 3.5)
# plt.savefig('/Users/jeremywilkinson/research/gle/drafts/coloured_noise/images/md_vs_gle_gamma.eps', format='eps')
plt.show()
print()

for size in sizes:
    plt.scatter(temps, size_fit_etas[size], label=f'Simulation size ${size}^3$')
    m, c = np.polyfit(temps, size_fit_etas[size], 1)
    plt.plot(temps, m * temps + c)
    print(size, m)

plt.gcf().set_size_inches(8, 3.5)
plt.xlabel('Simulation temperature (K)')
plt.ylabel(r'Best fit $\eta$ (ps$^{-1}$)')
plt.legend()
plt.subplots_adjust(left=0.083, bottom=0.13, right=0.99, top=0.986, wspace=0.105)
# plt.savefig('/Users/jeremywilkinson/research/gle/drafts/coloured_noise/images/md_temp_vs_eta.eps', format='eps')
plt.show()
