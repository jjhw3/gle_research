import numpy as np


def get_fcc_basis(lattice_parameter):
    """
    Returns the canonical fcc basis in cartesian coords:

    b1 = lattice_parameter * (e_x + e_y)
    b2 = lattice_parameter * (e_y + e_z)
    b3 = lattice_parameter * (e_x + e_z)

    Parameters
    ----------
    lattice_parameter
        For a single face on the unit cell, the distance from a corner lattice site
        to the centre lattice site.

    Returns
    -------
        A 3x3 matrix with the fcc basis vectors in cartesian co-ordinates as its columns
    """
    return np.asarray([[1, 1, 0], [0, 1, 1], [1, 0, 1]]).T * lattice_parameter


def get_fcc_111_basis(lattice_parameter):
    """
    Returns basis vectors for the fcc lattice such that the first two basis vectors
    generate the 111 plane and the third jumps between planes.

    Parameters
    ----------
    lattice_parameter
        For a single face on the unit cell, the distance from a corner lattice site
        to the centre lattice site.

    Returns
    -------
        A 3x3 matrix with the fcc basis vectors in cartesian co-ordinates as its columns.
        Columns 1&2 span the 111 plane and the third jumps between planes
    """
    canonical_basis = get_fcc_basis(lattice_parameter)
    fcc_111_basis = canonical_basis.copy()
    fcc_111_basis[:, :2] -= fcc_111_basis[:, 2:]
    return fcc_111_basis
