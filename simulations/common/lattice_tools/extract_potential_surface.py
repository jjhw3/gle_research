import numpy as np

from common.constants import boltzmann_constant
from common.lattice_tools.common import change_basis


def extract_potential_surface_3D(config, positions, resolution):
    basis_2D = config.in_plane_basis[:2, :2]
    positions_2D = positions[:2]
    return extract_potential_surface(basis_2D, config.temperature, positions_2D, resolution)


def extract_potential_surface(basis, temperature, positions, resolution):
    in_first_cell_lattice_coords = change_basis(
        np.linalg.inv(basis),
        positions
    ) % 1

    half_bin_width = 1 / resolution / 2

    x_bin_edges = np.linspace(- half_bin_width, 1 + half_bin_width, resolution + 2)
    y_bin_edges = np.linspace(- half_bin_width, 1 + half_bin_width, resolution + 2)

    # import matplotlib.pyplot as plt
    # plt.hist2d(*change_basis(config.in_plane_basis, in_first_cell_lattice_coords), bins=60)
    # plt.show()

    pos_hist, _, _ = np.histogram2d(*in_first_cell_lattice_coords, bins=[x_bin_edges, y_bin_edges])
    pos_hist[:, 0] += pos_hist[:, -1]
    pos_hist[0, :] += pos_hist[-1, :]

    pos_hist = pos_hist[:-1, :-1]

    if (pos_hist == 0.0).any():
        print('Warning,', (pos_hist == 0.0).sum(), ', unvisited potential bins')
        pos_hist[pos_hist == 0.0] = 0.00001 * np.min(pos_hist[pos_hist > 0])

    x_points = (x_bin_edges[1:-1] + x_bin_edges[:-2]) / 2
    y_points = (y_bin_edges[1:-1] + y_bin_edges[:-2]) / 2

    boltzmann_energy = - np.log(pos_hist) * boltzmann_constant * temperature
    boltzmann_energy -= boltzmann_energy.min()

    return boltzmann_energy


def force_grid_from_potentials(config, potential_grid):
    lattice_coord_forces = np.zeros((2,) + potential_grid.shape)
    dx, dy = 1 / potential_grid.shape[0], 1 / potential_grid.shape[1]
    lattice_coord_forces[0] = - (np.roll(potential_grid, -1, axis=0) - np.roll(potential_grid, 1, axis=0)) / 2 / dx
    lattice_coord_forces[1] = - (np.roll(potential_grid, -1, axis=1) - np.roll(potential_grid, 1, axis=1)) / 2 / dy
    # cartesian_forces = change_basis(np.linalg.inv(config.in_plane_basis), lattice_coord_forces)
    cartesian_forces = change_basis(np.linalg.inv(config.in_plane_basis), lattice_coord_forces)
    return cartesian_forces


def extract_force_grid(config, positions, resolution):
    potential_surface = extract_potential_surface(config, positions, resolution)
    return force_grid_from_potentials(config, potential_surface)
