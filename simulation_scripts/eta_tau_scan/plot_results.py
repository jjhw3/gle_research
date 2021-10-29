import numpy as np
import matplotlib.pyplot as plt


etas = np.arange(0.4, 0.8, 0.01)
taus = np.arange(0.0, 0.4, 0.01)

eta_tau_gamma_grid = np.load('/Users/jeremywilkinson/research_data/gle_data/eta_tau_gamma_grid.npy')
eta_tau_ttf_grid = np.load('/Users/jeremywilkinson/research_data/gle_data/eta_tau_ttf_grid.npy')

ETAs, TAUs = np.meshgrid(etas, taus, indexing='ij')


# for j, tau in enumerate(taus):
#     plt.scatter(etas, eta_tau_gamma_grid[:, j], s=2, color='red')
# plt.show()


# for j, tau in enumerate(taus):
#     plt.scatter(etas, eta_tau_ttf_grid[:, j], s=2, color='blue')
# plt.show()

# plt.scatter(eta_tau_ttf_grid.flatten(), eta_tau_gamma_grid.flatten(), c=TAUs)
# plt.show()


plt.contour(ETAs, TAUs, eta_tau_gamma_grid, colors='red', levels=20)
plt.contour(ETAs, TAUs, eta_tau_ttf_grid, colors='blue', levels=20)
U = np.gradient(eta_tau_gamma_grid, axis=0)
V = np.gradient(eta_tau_gamma_grid, axis=1)
norm = np.sqrt(U**2 + V**2)
plt.quiver(
    ETAs,
    TAUs,
    U / norm,
    V / norm,
    color='r',
)

U = np.gradient(eta_tau_ttf_grid, axis=0)
V = np.gradient(eta_tau_ttf_grid, axis=1)
norm = np.sqrt(U**2 + V**2)
plt.quiver(
    ETAs,
    TAUs,
    U / norm,
    V / norm,
    color='b',
)

plt.show()

print()