import numpy as np
import matplotlib.pyplot as plt

from common.constants import cm

gle_etas = np.arange(0.4, 0.8, 0.01)
gle_taus = np.arange(0.0, 0.4, 0.01)
gle_ETAs, gle_TAUs = np.meshgrid(gle_etas, gle_taus, indexing='ij')
eta_tau_gamma_grid = np.load('../eta_tau_gamma_grid_300.npy')
# eta_tau_gamma_grid = np.load('/Users/jeremywilkinson/research_data/gle_data/eta_tau_gamma_grid.npy')
eta_tau_ttf_grid = np.load('../eta_tau_ttf_grid_300.npy')
# eta_tau_ttf_grid = np.load('/Users/jeremywilkinson/research_data/gle_data/eta_tau_ttf_grid.npy')
markovian_mask = gle_TAUs == 0

plt.scatter(eta_tau_ttf_grid, eta_tau_gamma_grid, s=2, c=gle_ETAs, label='non-Markovian Langevin Equation', cmap='inferno')
plt.colorbar(label=r'Friction constant, $\eta$ (ps$^{-1}$)', fraction=0.1, pad=0.01)
plt.scatter(eta_tau_ttf_grid[markovian_mask], eta_tau_gamma_grid[markovian_mask], s=20, marker='^', c='green', label='Markovian Langevin Equation')

xhis = np.arange(5e-6, 3e-5, 1e-6)
cubic_ttfs = np.load('../cubic_ttfs.npy')
cubic_gammas = np.load('../cubic_gamma_maxes.npy')
plt.scatter(cubic_ttfs, cubic_gammas, marker='D', s=20, c='b', label='Cubic friction')

MD_ttf = 0.6838884274372572
MD_gamma = 0.113
plt.scatter(MD_ttf, MD_gamma, marker='x', c='r', s=80, label='Full MD simulation')

plt.xlabel(r'Total energy decorrelation rate, $\phi^{-1}$ (ps$^{-1}$)')
plt.ylabel(r'Max ISF dephasing rate, $\Gamma_{\tt{max}}$ (ps$^{-1}$)')

plt.gcf().set_size_inches(12 * cm, 8 * cm)
plt.subplots_adjust(left=0.137, bottom=0.15, right=0.912, top=0.987)

plt.legend(loc='upper left', frameon=False)
plt.savefig('../../gamma_ttf.pdf')
plt.show()

print()
