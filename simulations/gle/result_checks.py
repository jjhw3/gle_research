import numpy as np
import matplotlib.pyplot as plt

from common.constants import boltzmann_constant
from common.lattice_tools import fcc
from common.lattice_tools.common import get_lattice_points, change_basis
from common.lattice_tools.plot_tools import force_aspect
from common.thermodynamics import boltzmann_distribution_2D


def plot_path_on_crystal(results, lattice_shape=(15, 15, 4), show=False):
    config = results.config
    lattice_shape = np.asarray(lattice_shape)

    plt.plot(results.positions[0], results.positions[1])
    lattice_points = get_lattice_points(*lattice_shape)
    in_plane_basis = config.in_plane_rotate(fcc.get_fcc_111_basis(config.lattice_parameter))
    equilibrium_lattice_coordinates = change_basis(in_plane_basis, lattice_points)
    equilibrium_lattice_coordinates -= equilibrium_lattice_coordinates[:, lattice_shape[0] // 2, lattice_shape[1] // 2, -1][:, np.newaxis, np.newaxis, np.newaxis]

    for i in range(1, 4):
        plt.scatter(equilibrium_lattice_coordinates[0, :, :, -i], equilibrium_lattice_coordinates[1, :, :, -i])

    force_aspect()
    plt.xlim(equilibrium_lattice_coordinates[0, :, :, -3:].min(), equilibrium_lattice_coordinates[0, :, :, -3:].max())
    plt.ylim(equilibrium_lattice_coordinates[1, :, :, -3:].min(), equilibrium_lattice_coordinates[1, :, :, -3:].max())

    plt.savefig(config.summary_dir / 'particle_path.png')

    if show:
        plt.show()
    else:
        plt.cla()


def plot_power_spectrum(results, show=False, tau_plot_bound_multiple=10):
    config = results.config

    ws = 2 * np.pi * np.fft.fftfreq(config.num_iterations, config.dt)
    noise_forces = np.real(results.noise_forces)
    noise_spectrum = np.abs(np.fft.fft(noise_forces))**2

    plt.scatter(ws, noise_spectrum[0], label='x noise', s=3)
    plt.scatter(ws, noise_spectrum[1], label='y noise', s=3)
    plt.plot(ws, config.noise_stddev ** 2 * config.num_iterations * 1 / np.abs(1 + 1j * ws * config.tau) ** 2)
    plt.xlim(- tau_plot_bound_multiple / config.tau, tau_plot_bound_multiple / config.tau)
    plt.xlabel('angular frequency')
    plt.xlabel('|amplitude|^2')

    plt.savefig(config.summary_dir / 'noise_power_spectrum.png')
    np.save(config.summary_dir / 'noise_power_spectrum_angular_freqs.npy', ws)
    np.save(config.summary_dir / 'noise_power_spectrum.npy', noise_spectrum)

    if show:
        plt.show()
    else:
        plt.cla()

    return ws, noise_spectrum


def plot_maxwell_boltzmann_distributions(results, show=False, plot_bound_multiple=5, num_bins=50):
    config = results.config

    theoretical_mean_speed = np.sqrt(2 * boltzmann_constant * config.temperature / config.absorbate_mass)
    speeds = np.sqrt(np.sum(results.velocities**2, axis=0))

    vals, bins, _ = plt.hist(speeds, range=(0, plot_bound_multiple * theoretical_mean_speed), density=True, bins=num_bins)
    bin_centres = (bins[1:] + bins[:-1]) / 2
    theoretical_dist = boltzmann_distribution_2D(config.temperature, config.absorbate_mass, bin_centres)
    plt.plot(bin_centres, theoretical_dist)
    plt.xlabel('speed')
    plt.xlabel('probability')

    plt.savefig(config.summary_dir / 'maxwell_boltzmann_distribution.png')

    if show:
        plt.show()
    else:
        plt.cla()

    return bin_centres, vals
