import numpy as np
import matplotlib.pyplot as plt

gle_etas = np.arange(0.4, 0.8, 0.01)
gle_taus = np.arange(0.0, 0.4, 0.01)
gle_ETAs, gle_TAUs = np.meshgrid(gle_etas, gle_taus, indexing='ij')
eta_tau_gamma_grid = np.load('eta_tau_gamma_grid_300.npy')
eta_tau_ttf_grid = np.load('eta_tau_ttf_grid_300.npy')
markovian_mask = gle_TAUs == 0

plt.scatter(eta_tau_ttf_grid, eta_tau_gamma_grid, s=2, c=gle_TAUs, label='non-Markovian Langevin Equation')
plt.scatter(eta_tau_ttf_grid[markovian_mask], eta_tau_gamma_grid[markovian_mask], s=20, marker='^', c='g', label='Markovian Langevin Equation')

xhis = np.arange(5e-6, 3e-5, 1e-6)
cubic_ttfs = np.load('cubic_ttfs.npy')
cubic_gammas = np.load('cubic_gamma_maxes.npy')
plt.scatter(cubic_ttfs, cubic_gammas, marker='D', c=xhis, cmap='plasma', s=20)

MD_ttf = 0.7249310279836322
MD_gamma = 0.115
plt.scatter(MD_ttf, MD_gamma, marker='x', c='r', s=80, label='Full MD simulation')

plt.show()
