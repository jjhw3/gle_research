import sys
from pathlib import Path

import numpy as np

from common.constants import boltzmann_constant
from common.lattice_tools.common import change_basis
from md.configuration import MDConfig


if __name__ == '__main__':
    temp_dir = Path(sys.argv[1])
    print(sys.argv[1])

    cum_pos_hist = None

    for temp_ind_dir in temp_dir.glob('*'):
        if not temp_ind_dir.is_dir():
            continue
        print(temp_ind_dir)

        config = MDConfig.load(temp_ind_dir)
        resolution = 100

        positions = np.load(config.working_directory / 'absorbate_positions.npy')[:2]

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

        if cum_pos_hist is None:
            cum_pos_hist = pos_hist
        else:
            cum_pos_hist += pos_hist

    potential_grid = - np.log(cum_pos_hist) * boltzmann_constant * config.temperature
    potential_grid -= potential_grid.min()

    np.save(temp_dir / 'potential_grid.npy', potential_grid)
