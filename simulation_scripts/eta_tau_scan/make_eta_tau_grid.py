from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common.lattice_tools.plot_tools import cla
from common.tools import stable_fit_alpha, FitFailedException

times = np.arange(0, 5000, 0.01)


def get_time_to_forget(path, eta, tau):
    energies = np.load(path / 'total_energy_autocorrelation.npy')
    energies -= np.mean(energies[(times > 50) & (times < 100)])
    energies /= energies[0]

    alpha = 1 / times[np.where(energies < 1 / np.e)[0][0]]

    # plt.scatter(times, energies, s=2)
    # plt.axhline(1 / np.e)
    # plt.axvline(1 / alpha)
    # plt.xlim(0, 5)
    # plt.ylim(0, 1)
    # plt.axhline(0)
    # plt.savefig(f'/home/jjhw3/rds/hpc-work/gle_300/eta_tau_scan/ttf_plots/{eta}_{tau}.pdf')
    # cla()

    return alpha


def get_gamma(path):
    isf = np.load(path / 'combined_isfs/1.23.npy')
    gamma = stable_fit_alpha(
        times,
        isf,
        np.array([1.0, 0]),
        0,
        t_0=5,
        tol=0.01,
        plot_dir=path / 'plots'
    )

    return gamma


taus = np.arange(0, 0.40, 0.01)
etas = np.arange(0.4, 0.8, 0.01)

eta_tau_ttf_grid = np.zeros((etas.shape[0], taus.shape[0]))
eta_tau_gamma_grid = np.zeros((etas.shape[0], taus.shape[0]))

top_dir = Path(f'/home/jjhw3/rds/hpc-work/gle_300/eta_tau_scan')


def str_remove_zeros(flt):
    strfloat = f'{flt:.2f}'
    while strfloat[-1] in ['0', '.'] and len(strfloat) > 1:
        strfloat = strfloat[:-1]
    return strfloat


for i, eta in enumerate(etas):
    for j, tau in enumerate(taus):
        print(eta, tau)
        working_directory = top_dir / f'{str_remove_zeros(eta)}/{str_remove_zeros(tau)}'

        try:
            eta_tau_gamma_grid[i, j] = get_gamma(working_directory)
            eta_tau_ttf_grid[i, j] = get_time_to_forget(working_directory, eta, tau)
        except FitFailedException as e:
            print(eta, tau, e)
            eta_tau_ttf_grid[i, j] = np.nan
            eta_tau_gamma_grid[i, j] = np.nan


np.save(top_dir / 'eta_tau_ttf_grid.npy', eta_tau_ttf_grid)
np.save(top_dir / 'eta_tau_gamma_grid.npy', eta_tau_gamma_grid)
