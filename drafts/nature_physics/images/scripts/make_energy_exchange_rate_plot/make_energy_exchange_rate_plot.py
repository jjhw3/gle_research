import numpy as np
import matplotlib.pyplot as plt

from common.constants import cm
from common.tools import stable_fit_alpha
from gle.theoretics import get_greens_function_parameters

gle_etas = np.arange(0.4, 0.8, 0.01)
gle_taus = np.arange(0.0, 0.4, 0.01)
gle_ETAs, gle_TAUs = np.meshgrid(gle_etas, gle_taus, indexing='ij')
eta_tau_ttf_grid = np.load('../eta_tau_ttf_grid_300.npy')
markovian_mask = gle_TAUs == 0

plt.subplot2grid((2, 2), (0, 0))
plt.plot(gle_etas, gle_etas, c='red', label=r'$\phi^{-1}=\eta$')
plt.scatter(gle_ETAs, eta_tau_ttf_grid, s=2, c=gle_TAUs, label='non-Markov. Langevin eqn')
plt.scatter(gle_ETAs[markovian_mask], eta_tau_ttf_grid[markovian_mask], s=20, marker='^', c='black', label='Markov. Langevin eqn')
plt.legend(loc='upper left', prop={'size': 8}, frameon=False)
plt.colorbar(label=r'Noise correlation time, $\tau$ (ps)', fraction=0.1, pad=0.01, location='top')
plt.clim(0, 0.4)
plt.ylabel('Energy exchange\nrate, $\\phi^{-1}$ (ps$^{-1}$)')
plt.xlabel('Friction constant, $\eta$ (ps$^{-1}$)')
plt.ylim(0.05, 0.95)


plt.subplot2grid((2, 2), (0, 1))
w0 = 7.7
eta = 0.6
theoretical_Is = np.array([2 * get_greens_function_parameters(w0, eta, tau)[1] / eta for tau in gle_taus])
plt.plot(gle_taus, 1 / (1 + (w0 * gle_taus)**2), c='grey', label=r'$I = \frac{1}{1+(\omega_0\tau)^2}$')
plt.scatter(gle_taus, theoretical_Is, s=10, c='black', marker='o', label=r'Equivalent harmonic well')
plt.scatter(gle_TAUs, eta_tau_ttf_grid / gle_ETAs, s=2, c='orange', label='non-Markov. Langevin eqn')
plt.xlabel(r'Noise correlation time, $\tau$ (ps)')
plt.ylabel('Energy exchange\nsuppression factor, $I = \\phi^{-1} / \\eta$')
# plt.colorbar(label=r'Friction constant, $\eta$ (ps$^{-1}$)', fraction=0.1, pad=0.01, location='top')
plt.legend(loc='upper right', prop={'size': 8}, frameon=False)


plt.subplot2grid((2, 2), (1, 0))
xhis = np.arange(5e-6, 3e-5, 1e-6)
cubic_ttfs = np.load('../cubic_ttfs.npy')
plt.scatter(xhis * 1e5, cubic_ttfs, s=2, label='Cubic friction', color='b')
plt.xlabel(r'$\zeta$ ($10^{-5}$ ps$/\AA^{2}$)')
plt.ylabel('Energy exchange\nrate, $\\phi^{-1}$ (ps$^{-1}$)')
plt.legend(loc='upper left', prop={'size': 8}, frameon=False)

plt.subplot2grid((2, 2), (1, 1))

times = np.arange(0, 1000, 0.01)
e_auto = np.load('3D_md_8^3_300K_total_energy_autocorrelation.npy')[:times.shape[0]]
e_auto -= np.mean(e_auto[(times > 200) & (times < 1000)])
e_auto /= e_auto[0]
ttf = 1 / times[np.where(e_auto < 1 / np.e)[0][0]]
fit_ttf = stable_fit_alpha(
    times,
    e_auto,
    np.array([1.0, 0]),
    0,
    t_0=0.63,
    t_final=2.8,
)
# fit_ttf = 0.6838884274372572
plt.plot(
    times,
    e_auto[np.abs(times - 0.63) == 0][0] * np.exp(-times * fit_ttf) / np.exp(- 0.63 * fit_ttf),
    c='black',
)
plt.text(2.25, 0.18, f'$e^{{-{fit_ttf:.2f}t}}$')
plt.scatter(times, e_auto, marker='o', c='r', s=3, label='3D MD simulation')
plt.xlim(0, 8)
# plt.yscale('log')
plt.xlabel('Time (ps)')
plt.ylabel('Normalised total energy\nautocorrelation function')
plt.legend(loc='upper right', prop={'size': 8}, frameon=False)

plt.gcf().set_size_inches(18.3 * cm, 13 * cm)
plt.subplots_adjust(left=0.104, bottom=0.096, right=0.992, top=0.924, wspace=0.315, hspace=0.315)
plt.savefig('../../energy_exchange_rates.pdf')

plt.show()
