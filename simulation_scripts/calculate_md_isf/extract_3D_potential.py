import sys
from pathlib import Path

import numpy as np

from common.constants import boltzmann_constant
from common.lattice_tools.common import change_basis
from md.configuration import MDConfig


def get_occupation(working_dir, z_range):
    resolutions = np.array([100, 100, 50])
    config = MDConfig.load(working_dir)

    positions = np.load(working_dir / 'absorbate_positions.npy')
    in_first_cell_lattice_coords = np.zeros_like(positions)

    in_first_cell_lattice_coords[:2] = change_basis(
        np.linalg.inv(config.in_plane_basis[:2, :2]),
        positions[:2]
    ) % 1
    in_first_cell_lattice_coords[2] = positions[2]

    half_bin_widths = 1 / resolutions / 2
    half_bin_widths[2] *= z_range[1] - z_range[0]
    bin_edges = []

    for i in range(2):
        bin_edges.append(np.linspace(- half_bin_widths[i], 1 + half_bin_widths[i], resolutions[i] + 2))
    bin_edges.append(np.linspace(z_range[0] - half_bin_widths[2], z_range[1] + half_bin_widths[2], resolutions[2] + 1))

    # import matplotlib.pyplot as plt
    # plt.hist2d(*change_basis(config.in_plane_basis, in_first_cell_lattice_coords), bins=60)
    # plt.show()

    pos_hist, _ = np.histogramdd(np.swapaxes(in_first_cell_lattice_coords, 0, 1), bin_edges)
    pos_hist[:, 0, :] += pos_hist[:, -1, :]
    pos_hist[0, :, :] += pos_hist[-1, :, :]
    pos_hist = pos_hist[:-1, :-1]

    post_hist_copy = pos_hist.copy()
    pos_hist += post_hist_copy[::-1, :, :]
    pos_hist += post_hist_copy[:, ::-1, :]
    pos_hist += np.swapaxes(pos_hist, 0, 1)

    return pos_hist


if __name__ == '__main__':
    z_range = (13, 16)
    # rootdir = Path('/Users/jeremywilkinson/research_data/md_data/0')
    # get_occupation(rootdir, z_range)

    num_folders = 201

    cum_occupation = None

    for temp in [140, 160, 180, 200, 225, 250, 275, 300]:
        rootdir = Path(f'/home/jjhw3/rds/hpc-work/md/calculate_md_isf/8/{temp}')

        cum_pos_hist = None
        for i in range(num_folders):
            print(i)
            pos_hist = get_occupation(rootdir / str(i), z_range)
            if cum_pos_hist is None:
                cum_pos_hist = np.zeros_like(pos_hist)
            cum_pos_hist += pos_hist

        if cum_occupation is None:
            cum_occupation = np.zeros_like(cum_pos_hist)

        cum_occupation += cum_pos_hist ** (temp / 300)

    cum_occupation[cum_occupation == 0] = 1 / 100
    cum_occupation /= np.max(cum_occupation)
    potential_3D = - boltzmann_constant * 300 * np.log(cum_occupation)
    np.save(rootdir.parent / '3D_pot_surface.npy', potential_3D)
