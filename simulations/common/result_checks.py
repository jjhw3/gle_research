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
