import sys
from pathlib import Path

import numpy as np

from common.lattice_tools.common import change_basis
from md.configuration import MDConfig


if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
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

    np.save(working_dir / 'pos_hist.npy', pos_hist)
