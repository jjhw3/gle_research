import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from common.constants import cm
from gle.theoretics import get_greens_function_parameters, get_harmonic_gle_poles

etas = np.arange(0.4, 0.8, 0.01)
taus = np.arange(0.0, 0.4, 0.01)

eta_tau_ttf_grid = np.load('/Users/jeremywilkinson/research_data/gle_data/eta_tau_ttf_grid_300.npy')
eta_tau_gamma_grid = np.load('/Users/jeremywilkinson/research_data/gle_data/eta_tau_gamma_grid_300.npy')

ETAs, TAUs = np.meshgrid(etas, taus, indexing='ij')


x_contours = [2, 7, 12, 18, 23, 29, 36]
y_contours = [5, 8, 11, 14, 18, 22, 29]

ttf_levels = sorted(list(eta_tau_ttf_grid[0, y_contours]) + list(eta_tau_ttf_grid[x_contours, 0]))
gamma_levels = sorted(list(eta_tau_gamma_grid[0, y_contours]) + list(eta_tau_gamma_grid[x_contours, 0]))

plt.gcf().set_size_inches(18.3 * cm, 13 * cm)

ax1 = plt.subplot(2, 2, 1)
plt.ylabel('Total energy decorrelation\nrate $\\phi^{-1}$ (ps$^{-1}$)')
plt.scatter(ETAs, eta_tau_ttf_grid, c=TAUs, s=2)
plt.colorbar(location='top', label=r'$\tau$ (ps)')
plt.xticks([], [])

plt.subplot(2, 2, 2)

plt.scatter(TAUs, eta_tau_ttf_grid, c=ETAs, s=2, cmap='plasma')
plt.colorbar(location='top', label=r'$\eta$ (ps$^{-1}$)')
plt.xticks([], [])
plt.yticks([], [])

ax1 = plt.subplot(2, 2, 3)
plt.ylabel('Long-time ISF decay\nrate $\\Gamma$ (ps$^{-1}$)')
plt.xlabel(r"Friction constant $\eta$ (ps$^{-1}$)")
plt.scatter(ETAs, eta_tau_gamma_grid, c=TAUs, s=2)

plt.subplot(2, 2, 4)
plt.scatter(TAUs, eta_tau_gamma_grid, c=ETAs, s=2, cmap='plasma')
plt.xlabel(r"Noise correlation time $\tau$ (ps)")
plt.yticks([], [])

plt.subplots_adjust(left=0.13, bottom=0.092, right=0.99, top=0.945, wspace=0.02, hspace=0.03)
plt.savefig('/Users/jeremywilkinson/research/gle/drafts/nature_physics/images/eta_tau_ttf_gamma.pdf')
plt.show()


# w0s = np.linspace(8, 15, 40)
#
# k = 1800
# m = 50
#
# theoretical_ttf = np.zeros_like(TAUs)
# for i in range(ETAs.shape[0]):
#     for j in range(ETAs.shape[1]):
#         chi1, chi2, eta1 = get_greens_function_parameters(8.84, ETAs[i, j], TAUs[i, j])
#         theoretical_ttf[i, j] = 2 * chi2
#
# plt.scatter(TAUs, eta_tau_ttf_grid / ETAs, s=4, label=r'Simulated $\phi^{-1} / \eta$')
# plt.scatter(TAUs, theoretical_ttf / ETAs, s=4, label=r'Theory $\phi^{-1} / \eta$')
# plt.subplots_adjust(left=0.087, bottom=0.143, right=0.99, top=0.929, wspace=0.138)
# plt.gcf().set_size_inches(8, 3.5)
# plt.xlabel(r'$\tau$ (ps$^{-1}$)')
# plt.ylabel(r'$\phi^{-1} / \eta$')
# plt.legend()
# plt.savefig('/Users/jeremywilkinson/research/gle/drafts/coloured_noise/images/theoretical_tau_ttf.pdf')
# plt.show()
#
# W0s, ETAs, TAUs = np.meshgrid(w0s, etas, taus, indexing='ij')
#
# w0_theoretical_ttf = np.zeros_like(TAUs)
# for i in range(W0s.shape[0]):
#     for j in range(W0s.shape[1]):
#         for k in range(W0s.shape[2]):
#             chi1, chi2, eta1 = get_greens_function_parameters(W0s[i, j, k], ETAs[i, j, k], TAUs[i, j, k])
#             w0_theoretical_ttf[i, j] = 2 * chi2
#
# # plt.scatter(TAUs, w0_theoretical_ttf, c=W0s, s=4)
# # plt.scatter(W0s, w0_theoretical_ttf, c=TAUs, s=4)
# plt.contour(TAUs, W0s, w0_theoretical_ttf, levels=20)
#
# plt.show()
