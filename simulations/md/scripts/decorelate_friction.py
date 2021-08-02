import sys
from pathlib import Path

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.stats

from common.lattice_tools.common import change_basis, mag, norm
from common.lattice_tools.extract_potential_surface import extract_potential_surface_3D
from common.tools import fast_correlate
from gle.configuration import ComplexTauGLEConfig
from md.configuration import MDConfig
from cle import eval_force_from_pot

# /Users/jeremywilkinson/research_data/md_data/vel_auto

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    # config = MDConfig.load(working_dir)
    config = ComplexTauGLEConfig.load(working_dir)
    skip = 10
    config.dt = skip * config.dt

    resolution = 30
    basis_2D = config.in_plane_basis[:2, :2]
    basis_2D_inv = np.linalg.inv(basis_2D)

    positions = np.load(config.working_directory / 'absorbate_positions.npy')[:, ::skip]
    velocities = np.load(config.working_directory / 'absorbate_velocities.npy')[:, ::skip]
    forces = np.load(config.working_directory / 'absorbate_forces.npy')[:, ::skip]
    noise_forces = np.load(config.working_directory / 'absorbate_noise_forces.npy')[:, ::skip]

    pot_surface = extract_potential_surface_3D(config, positions, resolution)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X, Y = change_basis(basis_2D, np.asarray(np.meshgrid(np.linspace(0, 1, resolution), np.linspace(0, 1, resolution), indexing='ij')))

    ax.plot_surface(X, Y, pot_surface)
    plt.show()

    background_forces = np.zeros((2, forces.shape[1]))

    for i in range(config.num_iterations):
        eval_force_from_pot(
            background_forces[:, i],
            basis_2D_inv,
            pot_surface,
            positions[:2, i],
        )

    stochastic_noise = forces[:2] - background_forces
    fs = np.fft.fftfreq(config.num_iterations, config.dt)
    plt.plot(fs, np.fft.fft(stochastic_noise[0]))
    plt.xlim(-2 * config.tau, 2*config.tau)
    plt.show()

    plt.plot(forces[0], label='forces')
    plt.plot(background_forces[0], label='mean background')
    plt.plot(stochastic_noise[0], label='stochastic')
    plt.legend(loc='upper left')
    plt.show()

    # vel_mags = mag(velocities[:2])[0]
    # force_components = (velocities[:2] * stochastic_noise).sum(axis=0)
    # mean_force, vel_edges, inds = scipy.stats.binned_statistic(vel_mags, force_components, bins=100)
    # vel_bin_centres = (vel_edges[1:] + vel_edges[:-1]) / 2
    #
    # plt.scatter(vel_mags, force_components, s=1)
    # plt.plot(vel_bin_centres, mean_force, c='red')
    # plt.show()

    plot_time = 10
    mask = config.times < plot_time
    vel_stochastic_auto = fast_correlate(velocities[0], stochastic_noise[0])
    plt.plot(config.times[mask], vel_stochastic_auto[mask])
    plt.plot(config.times[mask][1:], np.diff(vel_stochastic_auto)[mask[1:]])
    plt.plot(config.times[mask][1:], np.diff(vel_stochastic_auto)[mask[1:]])
    # plt.plot(config.times[mask][1:], np.diff(vel_stochastic_auto)[mask[1:]])
    plt.show()

    vel_auto = fast_correlate(velocities[0], velocities[0])
    derr_vel_stochastic_auto = np.diff(vel_stochastic_auto) / config.dt
    plt.plot(config.times, - vel_stochastic_auto[:-1] / (derr_vel_stochastic_auto + config.absorbate_mass * config.eta * vel_auto[:-1]))
    plt.ylim(0, 20)
    plt.show()

    print()