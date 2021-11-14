import sys
from pathlib import Path

import numpy as np

from common.constants import boltzmann_constant
from common.lattice_tools.common import change_basis
from md.configuration import MDConfig


def get_occupation(working_dir):
    config = MDConfig.load(working_dir)
    resolution = 100

    positions = np.load(working_dir / 'absorbate_positions.npy')[:2]

    in_first_cell_lattice_coords = change_basis(
        np.linalg.inv(config.in_plane_basis[:2, :2]),
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
    return pos_hist


if __name__ == '__main__':
    # rootdir = Path('/Users/jeremywilkinson/research_data/md_data/0')
    # get_occupation(rootdir)

    rootdir = Path('/home/jjhw3/rds/hpc-work/md/calculate_md_isf/8/300')
    num_folders = 201

    cum_pos_hist = None
    for i in range(num_folders):
        print(i)
        pos_hist = get_occupation(rootdir / str(i))
        if cum_pos_hist is None:
            cum_pos_hist = np.zeros_like(pos_hist)
        cum_pos_hist += pos_hist

    potential_surface = - boltzmann_constant * 300 * np.log(cum_pos_hist)
    np.save(rootdir / 'pot_surface.npy',potential_surface)
