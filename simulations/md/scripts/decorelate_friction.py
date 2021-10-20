import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats

from cle import eval_force_from_pot
from common.constants import boltzmann_constant
from gle.configuration import TauGLEConfig

from md.configuration import MDConfig


def windowed_mean(arr, window_size):
    conv = scipy.signal.convolve(arr, np.ones(window_size) / window_size, mode='same')
    return conv[::window_size]


if __name__ == '__main__':
    # working_dir = Path(sys.argv[1])
    # working_dir = Path('/Users/jeremywilkinson/research_data/md_data/friction')
    working_dir = Path('/Users/jeremywilkinson/research_data/md_data/final/extract_friction/')
    # print(sys.argv[1])
    # config = TauGLEConfig.load(working_dir)
    config = MDConfig.load(working_dir)
    length = 10000000
    subsample = 1

    basis_2D = config.in_plane_basis[:2, :2]
    basis_2D_inv = np.linalg.inv(basis_2D)

    times = config.times[:length:subsample]
    config.dt *= subsample
    positions = np.load(config.working_directory / 'absorbate_positions.npy')[:2, :length:subsample]
    velocities = np.load(config.working_directory / 'absorbate_velocities.npy')[:2, :length:subsample]
    forces = np.load(config.working_directory / 'absorbate_forces.npy')[:2, :length:subsample]
    pot_surface = np.load('/Users/jeremywilkinson/research/gle/simulations/high_res_potential_grid.npy')

    background_forces = np.zeros((2, forces.shape[1]))

    for window_size in range(times.shape[0]):
        eval_force_from_pot(
            background_forces[:, window_size],
            basis_2D_inv,
            pot_surface,
            positions[:2, window_size],
        )

    stochastic_noise = forces - background_forces

    xdot = velocities[0]
    run_temp = np.mean(np.sum(velocities**2, axis=0)) * config.absorbate_mass / boltzmann_constant / velocities.shape[0]
    print('run temp:', run_temp)
    s = stochastic_noise[0]
    mass = config.absorbate_mass

    vacf = scipy.signal.correlate(xdot, xdot, mode='same')
    sacf = scipy.signal.correlate(s, s, mode='same')
    plt.plot(times, np.fft.fftshift(vacf / np.max(vacf)))
    plt.plot(times, np.fft.fftshift(sacf / np.max(sacf)))
    plt.show()

    etas = []
    noise_etas = []

    window_sizes = np.asarray(list(range(1, 100 + 1, 4)))
    plot_shape = int(np.ceil(np.sqrt(window_sizes.shape[0])))

    ax = None
    for i, window_size in enumerate(window_sizes):
        ax = plt.subplot(plot_shape, plot_shape, i + 1, sharex=ax, sharey=ax)

        mask = np.arange(0, xdot.shape[0], 1, dtype=int)
        xdot_mean = windowed_mean(xdot, window_size)
        s_mean = windowed_mean(np.roll(s, - window_size), window_size)
        # s_mean = np.roll(s_mean, -1)

        m, c = np.polyfit(xdot_mean, s_mean / mass, 1)
        print(window_size, f"eta={-m}, sigma={np.sqrt(- 2 * boltzmann_constant * run_temp * config.absorbate_mass * m)}")
        plt.scatter(xdot_mean, s_mean / mass, s=1)
        plt.plot(xdot_mean, m * xdot_mean + c, c='r')
        statistic, bedges, binds = scipy.stats.binned_statistic(xdot_mean, s_mean / mass, bins=100)

        std, _, _ = scipy.stats.binned_statistic(xdot_mean, s_mean, statistic='std', bins=100)
        noise_stddev = scipy.stats.binned_statistic(xdot_mean, s_mean, statistic='std', bins=100)[0][20:-20].mean()
        noise_eta = noise_stddev**2 / (2 * boltzmann_constant * run_temp * config.absorbate_mass / (window_size * config.dt))
        print(f"Noise eta: {noise_eta}")

        etas.append(-m)
        noise_etas.append(noise_eta)

        plt.scatter((bedges[1:] + bedges[:-1]) / 2, statistic, c='black')

        plt.title(f'window size {window_size}')

    plt.show()

    plt.subplot(1, 2, 1)
    plt.plot(window_sizes * config.dt, etas)
    plt.plot(window_sizes * config.dt, np.asarray(noise_etas))

    plt.subplot(1, 2, 2)
    plt.plot(times, np.fft.fftshift(vacf / np.max(vacf)))
    plt.plot(times, np.fft.fftshift(sacf / np.max(sacf)))
    plt.xlim(0, 1)

    plt.show()

    print()
