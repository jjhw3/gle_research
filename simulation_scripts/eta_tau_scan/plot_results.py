import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from gle.theoretics import get_greens_function_parameters

etas = np.arange(0.4, 0.8, 0.01)
taus = np.arange(0.0, 0.4, 0.01)

eta_tau_gamma_grid = np.load('/Users/jeremywilkinson/research_data/gle_data/eta_tau_gamma_grid_160.npy')
eta_tau_ttf_grid = np.load('/Users/jeremywilkinson/research_data/gle_data/eta_tau_ttf_grid_160.npy')

ETAs, TAUs = np.meshgrid(etas, taus, indexing='ij')


# for j, tau in enumerate(taus):
#     plt.scatter(etas, eta_tau_gamma_grid[:, j], s=2, color='red')
# plt.show()


# for j, tau in enumerate(taus):
#     plt.scatter(etas, eta_tau_ttf_grid[:, j], s=2, color='blue')
# plt.show()

# plt.scatter(eta_tau_ttf_grid.flatten(), eta_tau_gamma_grid.flatten(), c=TAUs)
# plt.show()


theoretical_ttf = np.zeros_like(TAUs)
for i in range(ETAs.shape[0]):
    for j in range(ETAs.shape[1]):
        chi1, chi2, eta1 = get_greens_function_parameters(8.84, ETAs[i, j], TAUs[i, j])
        theoretical_ttf[i, j] = 2 * chi2

x_contours = [2, 7, 12, 18, 23, 29, 36]
y_contours = [5, 8, 11, 14, 18, 22, 29]

ttf_levels = sorted(list(eta_tau_ttf_grid[0, y_contours]) + list(eta_tau_ttf_grid[x_contours, 0]))
# ttf_levels = 15
gamma_levels = sorted(list(eta_tau_gamma_grid[0, y_contours]) + list(eta_tau_gamma_grid[x_contours, 0]))
# gamma_levels = 15

plt.gcf().set_size_inches(8, 3.5)
plt.subplot(1, 2, 1)

plt.title(r'Long-time ISF decay rate $\Gamma$ (ps$^{-1}$)')
plt.xlabel(r"$\eta$ (ps$^{-1}$)")
plt.ylabel(r"$\tau$ (ps)")
plt.scatter(ETAs, TAUs, c=eta_tau_gamma_grid, s=2)
contours = plt.contour(
    ETAs,
    TAUs,
    eta_tau_gamma_grid,
    levels=gamma_levels,
)
plt.clabel(contours, inline=True, fontsize=12, colors='black')

plt.subplot(1, 2, 2)

plt.title(r'Total energy decorrelation rate $\chi_2$ (ps$^{-1}$)')
plt.xlabel(r"$\eta$ (ps$^{-1}$)")
plt.scatter(ETAs, TAUs, c=eta_tau_gamma_grid, s=4)
contours = plt.contour(ETAs, TAUs, eta_tau_ttf_grid, levels=ttf_levels)
plt.clabel(contours, inline=True, fontsize=12, colors='black')

plt.subplots_adjust(left=0.087, bottom=0.143, right=0.99, top=0.929, wspace=0.138)
# plt.savefig('/Users/jeremywilkinson/research/gle/drafts/coloured_noise/images/gamma_and_ttf_contours.eps', format='eps')
plt.show()

# U = np.gradient(eta_tau_gamma_grid, axis=0)
# V = np.gradient(eta_tau_gamma_grid, axis=1)
# norm = np.sqrt(U**2 + V**2)
# plt.quiver(
#     ETAs,
#     TAUs,
#     U / norm,
#     V / norm,
#     color='r',
#     scale=40
# )

plt.scatter(eta_tau_ttf_grid, eta_tau_gamma_grid, c=ETAs, s=2)
plt.gcf().set_size_inches(8, 4)
plt.subplots_adjust(left=0.083, bottom=0.125, right=1.0, top=0.97, wspace=0.105)
plt.xlabel(r'Total energy decorrelation rate $\chi_2$ (ps$^{-1}$)')
plt.ylabel(r'Long-time ISF decay rate $\Gamma$ (ps$^{-1}$)')
plt.colorbar(label=r'$\eta$ (ps$^{-1}$)')
# plt.savefig('/Users/jeremywilkinson/research/gle/drafts/coloured_noise/images/gamma_and_ttf_scatter.eps', format='eps')
plt.show()
print()

plt.gcf().set_size_inches(8, 3.5)

plt.scatter(TAUs, theoretical_ttf / eta_tau_ttf_grid, s=2, label='Simulations Values', c=ETAs)
# plt.plot([0, np.max(theoretical_ttf)], [0, np.max(theoretical_ttf)], label=r'$\phi^{-1} = \phi_0^{-1}$', c='red')
plt.xlabel(r'Noise correlation time, $\tau$ (ps)')
plt.ylabel('Ratio of theoretical to simulated total\nenergy decorrelation rates, $\\phi_0^{-1} / \\phi^{-1}$')
plt.subplots_adjust(left=0.1, bottom=0.155, right=1.0, top=0.986, wspace=0.105)
# plt.savefig('/Users/jeremywilkinson/research/gle/drafts/coloured_noise/images/theoretical_ttf_comparison.eps', format='eps')
plt.ylim(0, 1.1)
plt.colorbar(label=r'Friction parameter $\eta$ (ps$^{-1}$)')
plt.show()


# plt.gcf().set_size_inches(8, 3.5)
#
# plt.scatter(ETAs, theoretical_ttf, s=2, c=TAUs, label='Simulations Values')
#
# plt.xlabel(r'$\eta$ (ps$^-1$)')
# plt.ylabel('Theoretical total energy\ndecorrelation rate, $\\phi_0^{-1}$ (ps$^-1$)')
# plt.subplots_adjust(left=0.1, bottom=0.155, right=0.986, top=0.986, wspace=0.105)
# plt.savefig('/Users/jeremywilkinson/research/gle/drafts/coloured_noise/images/theoretical_ttf_comparison.eps', format='eps')
# plt.show()


# plt.gcf().set_size_inches(8, 3.5)
# plt.subplot(1, 2, 1)
#
# plt.title(r'Simulated total energy decorrelation rate $\phi^{-1}$ (ps$^{-1}$)')
# plt.xlabel(r"$\eta$ (ps$^{-1}$)")
# plt.ylabel(r"$\tau$ (ps)")
# plt.scatter(ETAs, TAUs, c=eta_tau_gamma_grid, s=2)
# contours = plt.contour(
#     ETAs,
#     TAUs,
#     eta_tau_gamma_grid,
#     levels=gamma_levels,
# )
# plt.clabel(contours, inline=True, fontsize=12, colors='black')
#
# plt.subplot(1, 2, 2)
#
# plt.title(r'Simulated total energy decorrelation rate $\phi^{-1}$ (ps$^{-1}$)')
# plt.xlabel(r"$\eta$ (ps$^{-1}$)")
# plt.scatter(ETAs, TAUs, c=eta_tau_gamma_grid, s=4)
# contours = plt.contour(ETAs, TAUs, eta_tau_ttf_grid, levels=ttf_levels)
# plt.clabel(contours, inline=True, fontsize=12, colors='black')
