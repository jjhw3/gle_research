import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from gle.theoretics import get_greens_function_parameters, get_harmonic_gle_poles

etas = np.arange(0.4, 0.8, 0.01)
taus = np.arange(0.0, 0.4, 0.01)

eta_tau_gamma_grid = np.load('/Users/jeremywilkinson/research_data/gle_data/eta_tau_gamma_grid_300.npy')
ETAs, TAUs = np.meshgrid(etas, taus, indexing='ij')


x_contours = [2, 7, 12, 18, 23, 29, 36]
y_contours = [5, 8, 11, 14, 18, 22, 29]

gamma_levels = sorted(list(eta_tau_gamma_grid[0, y_contours]) + list(eta_tau_gamma_grid[x_contours, 0]))

plt.gcf().set_size_inches(8, 3.5)

ax1 = plt.subplot(1, 2, 1)
plt.ylabel('Long-time ISF decay rate $\Gamma$ (ps$^{-1}$)')
plt.xlabel(r"Friction constant $\eta$ (ps$^{-1}$)")
plt.scatter(ETAs, eta_tau_gamma_grid, c=TAUs, s=4)
plt.colorbar(location='top', label=r'$\tau$ (ps)')

plt.subplot(1, 2, 2, sharey=ax1)

plt.xlabel(r"Noise correlation time $\tau$ (ps)")
plt.scatter(TAUs, eta_tau_gamma_grid, c=ETAs, s=4)
plt.colorbar(location='top', label=r'$\eta$ (ps$^{-1}$)')

plt.subplots_adjust(left=0.087, bottom=0.143, right=0.99, top=0.929, wspace=0.138)
plt.savefig('/Users/jeremywilkinson/research/gle/drafts/coloured_noise/images/eta_tau_gamma.eps', format='eps')
plt.show()
