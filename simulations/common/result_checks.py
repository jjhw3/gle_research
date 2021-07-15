import numpy as np
import matplotlib.pyplot as plt

from common.lattice_tools import fcc
from common.lattice_tools.common import get_lattice_points, change_basis
from common.lattice_tools.plot_tools import force_aspect


def plot_path_on_crystal(config, results, lattice_shape=(10, 10, 10)):
    plt.plot(results.positions[0], results.positions[1])
    # force_aspect()
    lattice_points = get_lattice_points(*lattice_shape)
    in_plane_basis = config.in_plane_rotate(fcc.get_fcc_111_basis(config.lattice_parameter))
    equilibrium_lattice_coordinates = change_basis(in_plane_basis, lattice_points)

    for i in range(1, 4):
        plt.scatter(equilibrium_lattice_coordinates[0, :, :, -i], equilibrium_lattice_coordinates[1, :, :, -i])

    plt.scatter(*in_plane_basis[:2].T)
    force_aspect()

    plt.show()


def plot_fourier_spectrum(config, results):
    ws = 2 * np.pi * np.fft.fftfreq(config.num_iterations, config.dt)
    noise_forces = np.real(results.noise_forces)
    noise_spectrum = np.abs(np.fft.fft(noise_forces))**2

    # plt.scatter(ws, noise_spectrum[0], label='x noise', s=3)
    # plt.scatter(ws, noise_spectrum[1], label='y noise', s=3)
    # plt.plot(ws, 2 * config.noise_stddev ** 2 * config.num_iterations * 1 / np.abs(1 + 1j * ws * config.tau) ** 2)
    # plt.xlim(- 10 / config.tau, 10 / config.tau)
    # # plt.legend()
    # plt.show()

    print()

    return ws, noise_spectrum
