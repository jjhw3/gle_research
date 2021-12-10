import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

from common.constants import cm
from common.tools import stable_fit_alpha
from gle.theoretics import get_greens_function_parameters
import matplotlib.transforms as mtransforms


gle_etas = np.arange(0.4, 0.8, 0.01)
gle_taus = np.arange(0.0, 0.4, 0.01)
gle_ETAs, gle_TAUs = np.meshgrid(gle_etas, gle_taus, indexing='ij')
eta_tau_ttf_grid = np.load('../eta_tau_ttf_grid_300.npy')
markovian_mask = gle_TAUs == 0


fig, axs = plt.subplot_mosaic([['a)', 'b)'], ['c)', 'd)']])


plt.sca(axs['a)'])
plt.plot(gle_etas, gle_etas, c='red', label=r'$\phi^{-1}=\eta$')
plt.scatter(gle_ETAs, eta_tau_ttf_grid, s=2, c=gle_TAUs, label='non-Markov. Langevin eqn.')
plt.scatter(gle_ETAs[markovian_mask], eta_tau_ttf_grid[markovian_mask], s=20, marker='^', c='black', label='Markov. Langevin eqn')
plt.legend(loc='upper left', prop={'size': 8}, frameon=False, bbox_to_anchor=(0.61, 0.9), bbox_transform=fig.transFigure)
plt.colorbar(label=r'Noise correlation time, $\tau$ (ps)', fraction=0.1, pad=0.01, location='top')
plt.clim(0, 0.4)
plt.ylabel('Energy exchange\nrate, $\\phi^{-1}$ (ps$^{-1}$)')
plt.xlabel('Friction constant, $\eta$ (ps$^{-1}$)')
plt.ylim(0.05, 0.95)
trans = mtransforms.ScaledTranslation(0, -0.68, fig.dpi_scale_trans)
plt.text(0.5, 0.0, 'a)', transform=plt.gca().transAxes + trans, fontsize='medium', va='bottom', fontfamily='serif')


plt.sca(axs['b)'])
w0 = 8.8
eta = 0.4
theoretical_Is = np.array([2 * get_greens_function_parameters(w0, eta, tau)[1] / eta for tau in gle_taus])
plt.plot(gle_taus, 1 / (1 + (w0 * gle_taus)**2), c='grey', label=r'$I = \frac{1}{1+(\omega_0\tau)^2}$')
plt.scatter(gle_taus, theoretical_Is, s=10, c='black', marker='o', label=r'Equivalent harmonic well')
plt.scatter(gle_TAUs, eta_tau_ttf_grid / gle_ETAs, s=2, c='orange', label='non-Markov. Langevin eqn.')
plt.xlabel(r'Noise correlation time, $\tau$ (ps)')
plt.ylabel('Energy exchange\nsuppression factor, $I = \\phi^{-1} / \\eta$')
# plt.colorbar(label=r'Friction constant, $\eta$ (ps$^{-1}$)', fraction=0.1, pad=0.01, location='top')
plt.legend(loc='upper right', prop={'size': 8}, frameon=False)
trans = mtransforms.ScaledTranslation(0, -0.68, fig.dpi_scale_trans)
plt.text(0.5, 0.0, 'b)', transform=plt.gca().transAxes + trans, fontsize='medium', va='bottom', fontfamily='serif')


plt.sca(axs['c)'])
times = np.arange(0, 100, 0.01)
freqs = np.fft.fftfreq(times.shape[0], times[1] - times[0])
mean_psd = np.load('mean_psd.npy') / 3.8e5 * 2
tdomain_corr = np.load('tdomain_corr.npy')
plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(mean_psd))
plt.plot(np.fft.fftshift(freqs), 1 / (1 + (np.fft.fftshift(freqs) / 7.4)**2))
plt.xlim(-0.5, 20)
plt.xlabel('Frequency (THz)')
plt.ylabel('Force power\nspectrum (arb. units)')
plt.text(0.5, 0.0, 'c)', transform=plt.gca().transAxes + trans, fontsize='medium', va='bottom', fontfamily='serif')

ax2 = plt.axes([0, 0, 1, 1])
ip = InsetPosition(axs['c)'], [0.42, 0.41, 0.53, 0.54])
ax2.set_axes_locator(ip)
ax2.plot(times, tdomain_corr)
ax2.set_xlim(0, 1.5)
ax2.set_xlabel('Time (ps)', labelpad=0)
ax2.set_ylabel('Auto-correlation', labelpad=0)


plt.sca(axs['d)'])
xhis = np.arange(5e-6, 3e-5, 1e-6)
cubic_ttfs = np.load('../cubic_ttfs.npy')
plt.scatter(xhis * 1e5, cubic_ttfs, s=2, label='Cubic friction', color='b')
plt.xlabel(r'$\zeta$ ($10^{-5}$ ps$/\AA^{2}$)')
plt.ylabel('Energy exchange\nrate, $\\phi^{-1}$ (ps$^{-1}$)')
plt.legend(loc='upper left', prop={'size': 8}, frameon=False)
trans = mtransforms.ScaledTranslation(0, -0.68, fig.dpi_scale_trans)
plt.text(0.5, 0.0, 'd)', transform=plt.gca().transAxes + trans, fontsize='medium', va='bottom', fontfamily='serif')


plt.gcf().set_size_inches(18.3 * cm, 14 * cm)
plt.subplots_adjust(left=0.104, bottom=0.129, right=0.983, top=0.924, wspace=0.315, hspace=0.5)
plt.savefig('../../energy_exchange_rates.pdf')

plt.show()
