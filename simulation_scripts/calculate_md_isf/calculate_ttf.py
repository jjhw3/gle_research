from pathlib import Path

import numpy as np
from cle import eval_pot_grid
from scipy.interpolate import RegularGridInterpolator

from common.lattice_tools.common import change_basis
from common.tools import fast_auto_correlate
from gle.interpolation_tools import get_coefficient_matrix_grid
from md.configuration import MDConfig

potential_grid = np.load('/home/jjhw3/code/gle_research/simulations/high_res_potential_grid.npy')
# potential_grid = np.load('/Users/jeremywilkinson/research/gle/simulations/high_res_potential_grid.npy')
interpolation_coefficients = get_coefficient_matrix_grid(potential_grid)
# potential_grid_3D = np.load('/Users/jeremywilkinson/research_data/md_data/3D_pot_surface.npy')
potential_grid_3D = np.load('/home/jjhw3/rds/hpc-work/md/calculate_md_isf/8/300/3D_pot_surface.npy')
z_range = (13, 16)
interp = RegularGridInterpolator(
    (
        np.linspace(0, 1, potential_grid_3D.shape[0]),
        np.linspace(0, 1, potential_grid_3D.shape[1]),
        np.linspace(13, 16, potential_grid_3D.shape[2])
    ),
    potential_grid_3D,
    bounds_error=False
)

# z_k = 10557.210000649435
# z_mean = 14.351935754749448


def get_e_auto(working_dir):
    config = MDConfig.load(working_dir)
    positions = np.load(config.working_directory / 'absorbate_positions.npy')
    potentials = np.zeros(positions.shape[1])
    inv_in_plane_basis = np.linalg.inv(config.in_plane_basis)[:2, :2]

    in_first_cell_lattice_coords = np.zeros_like(positions)
    in_first_cell_lattice_coords[:2] = change_basis(
        np.linalg.inv(config.in_plane_basis[:2, :2]),
        positions[:2]
    ) % 1
    in_first_cell_lattice_coords[2] = positions[2]
    potentials = interp(in_first_cell_lattice_coords.T)

    # for i in range(potentials.shape[0]):
    #     potentials[i] = eval_pot_grid(
    #         inv_in_plane_basis,
    #         interpolation_coefficients,
    #         positions[0, i],
    #         positions[1, i],
    #     )

    velocities = np.gradient(positions, axis=1) / 0.01
    kinetic_energies = 0.5 * config.absorbate_mass * np.sum(velocities ** 2, axis=0)

    e_auto = fast_auto_correlate(potentials + kinetic_energies)
    return e_auto
    # np.save(config.working_directory / 'total_energy_autocorrelation.npy', e_auto)


if __name__ == '__main__':
    # rootdir = Path('/Users/jeremywilkinson/research_data/md_data/0')
    # get_e_auto(rootdir)
    rootdir = Path('/home/jjhw3/rds/hpc-work/md/calculate_md_isf/8/300')
    num_folders = 201

    cum_e_auto = None
    for i in range(num_folders):
        print(i)
        e_auto = get_e_auto(rootdir / str(i))
        if cum_e_auto is None:
            cum_e_auto = np.zeros_like(e_auto)
        cum_e_auto += e_auto / num_folders

    np.save(rootdir / 'total_energy_autocorrelation.npy', cum_e_auto)
