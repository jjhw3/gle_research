import numpy as np
import matplotlib.pyplot as plt

from common.constants import cm

gle_etas = np.arange(0.4, 0.8, 0.01)
gle_taus = np.arange(0.0, 0.4, 0.01)
gle_ETAs, gle_TAUs = np.meshgrid(gle_etas, gle_taus, indexing='ij')
eta_tau_ttf_grid = np.load('../eta_tau_ttf_grid_300.npy')
markovian_mask = gle_TAUs == 0

plt.subplot2grid((2, 2), (0, 0), colspan=2)
plt.plot(gle_etas, gle_etas, c='red', label=r'$\phi^{-1}=\eta$')
plt.scatter(gle_ETAs, eta_tau_ttf_grid, s=2, c=gle_TAUs, label='non-Markov. Langevin eqn')
plt.scatter(gle_ETAs[markovian_mask], eta_tau_ttf_grid[markovian_mask], s=20, marker='^', c='black', label='Markov. Langevin eqn')
plt.legend(loc='upper left', prop={'size': 8}, frameon=False)
plt.colorbar(label=r'Noise correlation time, $\tau$ (ps)', fraction=0.1, pad=0.01)
plt.clim(0, 0.4)
plt.ylabel('Energy exchange\nrate, $\\phi^{-1}$ (ps$^{-1}$)')
plt.xlabel('Friction constant, $\eta$ (ps$^{-1}$)')

plt.subplot2grid((2, 2), (1, 0))

xhis = np.arange(5e-6, 3e-5, 1e-6)
cubic_ttfs = np.load('../cubic_ttfs.npy')
plt.scatter(xhis * 1e5, cubic_ttfs, s=2, label='Cubic friction')
plt.xlabel(r'$\zeta$ ($10^{-5}$ ps$/\AA^{2}$)')
plt.ylabel('Energy exchange\nrate, $\\phi^{-1}$ (ps$^{-1}$)')
plt.legend(loc='upper left', prop={'size': 8}, frameon=False)

plt.subplot2grid((2, 2), (1, 1))

plt.yticks([], [])
MD_ttf = 0.718866407668944
plt.scatter([300], MD_ttf, marker='x', c='r', s=80, label='3D MD simulation')
plt.legend(loc='upper left', prop={'size': 8}, frameon=False)

plt.gcf().set_size_inches(18.3 * cm, 13 * cm)
plt.subplots_adjust(left=0.104, bottom=0.096, right=0.992, top=0.987, wspace=0.021, hspace=0.25)
plt.savefig('../../energy_exchange_rates.pdf')

plt.show()
