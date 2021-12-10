from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

from cle import eval_force_from_pot
from scipy.stats import binned_statistic

from gle.configuration import ComplexTauGLEConfig
from gle.interpolation_tools import get_coefficient_matrix_grid
from md.configuration import MDConfig

eta = 0.5
mass = 23
spline_coefficient_matrix_grid = get_coefficient_matrix_grid(np.load('/home/jjhw3/code/gle_research/simulations/high_res_potential_grid.npy'))
# spline_coefficient_matrix_grid = get_coefficient_matrix_grid(np.load('/Users/jeremywilkinson/research/gle/simulations/high_res_potential_grid.npy'))
cob = np.array([[ 3.93931354e-01, -1.67762832e-17], [-1.67762832e-17, 3.93931354e-01]])


def windowed_mean(arr, window_size):
    conv = scipy.signal.convolve(arr, np.ones(window_size) / window_size, mode='same')
    return conv[::window_size]


# def diff(arr):
#     d = np.roll(arr, -1) - arr
#     d[-1] = 0
#     return d


def diff(arr):
    d = arr - np.roll(arr, 1)
    d[0] = 0
    return np.roll(d, -1)


def get_vels_and_stochastic_force(positions, dt=0.01):
    # config = ComplexTauGLEConfig.load(dir)
    # velocities = np.gradient(positions[0]) / dt
    # forces = np.roll(mass * np.gradient(velocities) / dt, 0)
    velocities = diff(positions[0]) / dt
    forces = diff(mass * velocities / dt)
    background_forces = np.zeros_like(positions)

    # velocities = np.load('/Users/jeremywilkinson/research_data/gle_data/friction_test/absorbate_velocities.npy')[0]
    # forces = np.load('/Users/jeremywilkinson/research_data/gle_data/friction_test/absorbate_forces.npy')[0]

    for i in range(positions.shape[1]):
        eval_force_from_pot(
            background_forces[:, i],
            cob,
            spline_coefficient_matrix_grid,
            positions[:, i],
        )

    stochastic_forces = forces - background_forces[0] #- np.load('/Users/jeremywilkinson/research_data/gle_data/friction_test/absorbate_noise_forces.npy').real[0]
    return velocities, stochastic_forces


def get_force_vs_vel(velocities, stochastic_forces, vel_bins, window_size):
    mean_velocities = windowed_mean(velocities, window_size)
    mean_stochastic_forces = windowed_mean(stochastic_forces, window_size)

    means = binned_statistic(mean_velocities, mean_stochastic_forces, statistic='mean', bins=vel_bins)[0]
    vars = binned_statistic(mean_velocities, mean_stochastic_forces, statistic='std', bins=vel_bins)[0]**2
    counts = binned_statistic(mean_velocities, mean_stochastic_forces, statistic='count', bins=vel_bins)[0]

    return means, vars / counts

# window_sizes = [1, 3, 5, 10, 20, 30, 70]
# vel_range = (-5, 5)
# vel_bins = np.linspace(vel_range[0], vel_range[1], 20)
# vel_centers = (vel_bins[1:] + vel_bins[:-1]) / 2
#
# for window_size in window_sizes:
#     means, vars = get_force_vs_vel(Path('/Users/jeremywilkinson/research_data/md_data/isf_0'), vel_bins, window_size)
#     plt.errorbar(vel_centers, means, yerr=np.sqrt(vars), fmt='o', capsize=5, ls='none', markersize=5)
#     plt.plot(vel_bins, - mass * eta * vel_bins, c='r')
# plt.show()

print()

if __name__ == '__main__':
    window_sizes = [1, 2, 3, 4, 5, 10, 20, 30, 50]
    vel_range = (-10, 10)
    vel_bins = np.linspace(vel_range[0], vel_range[1], 20)
    base_dir = Path('/home/jjhw3/rds/hpc-work/md/calculate_md_isf/8/300')
    # base_dir = Path('/Users/jeremywilkinson/research_data/md_data/test')

    cum_means = {}
    cum_vars = {}
    N = 0
    for dir in base_dir.glob('*'):
        print(dir)
        if not dir.is_dir() or dir.name == 'friction':
            continue
        positions = np.load(dir / 'absorbate_positions.npy')[:2]
        velocities, stochastic_forces = get_vels_and_stochastic_force(positions)

        for window_size in window_sizes:
            means, vars = get_force_vs_vel(velocities, stochastic_forces, vel_bins, window_size)
            if window_size not in cum_means:
                cum_means[window_size] = np.zeros_like(means)
                cum_vars[window_size] = np.zeros_like(means)
            cum_means[window_size] += means
            cum_vars[window_size] += vars
        N += 1
        # break

    for window_size in window_sizes:
        mean_force = cum_means[window_size] / N / window_size
        mean_std = np.sqrt(cum_vars[window_size] / N) / window_size
        np.save(base_dir / f'friction/{window_size}_force.npy', mean_force)
        np.save(base_dir / f'friction/{window_size}_std.npy', mean_force)
