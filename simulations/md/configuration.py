import numpy as np
import yaml

from common.constants import eV_to_amu_K_ps
from common.lattice_tools import fcc
from common.lattice_tools.common import (
    change_basis,
    get_lattice_points,
    get_basis_rotation_matrix,
    get_reciprocal_basis, get_in_plane_basis, rotate_basis, mag, get_nearest_neighbour_list
)


class MDConfig:
    def __init__(
        self,
        dt,
        run_time,
        temperature,
        lattice_shape,
        basis_vectors,
        conventional_cell,
        free_plane_indices,
        lattice_connections,
        spring_const,
        substrate_mass,
        r0_multiple_interaction_distance_cutoff,
        r0,
        D,
        a,
        absorbate_mass,
        working_directory,
    ):
        self.dt = dt
        self.run_time = run_time
        self.lattice_shape = np.asarray(lattice_shape)
        self.canonical_basis = np.asarray(basis_vectors).T
        self.conventional_cell = np.asarray(conventional_cell).T
        self.free_plane_indices = np.asarray(free_plane_indices)
        self.spring_const = spring_const
        self.substrate_mass = substrate_mass
        self.temperature = temperature
        self.working_directory = working_directory
        self.r0_multiple_interaction_distance_cutoff = r0_multiple_interaction_distance_cutoff
        self.r0 = r0
        self.D = D
        self.a = a
        self.absorbate_mass = absorbate_mass

        self.isf_directory = self.working_directory / 'ISFs'
        self.log_isf_directory = self.isf_directory / 'log'
        if not self.isf_directory.exists():
            self.isf_directory.mkdir()
        if not self.log_isf_directory.exists():
            self.log_isf_directory.mkdir()

        self.in_plane_basis_canonical_coords = get_in_plane_basis(
            self.canonical_basis,
            self.conventional_cell,
            self.free_plane_indices
        )
        self.in_plane_rotation_matrix = get_basis_rotation_matrix(self.in_plane_basis_canonical_coords)
        self.in_plane_basis = rotate_basis(self.in_plane_basis_canonical_coords)
        self.lattice_points = get_lattice_points(*lattice_shape)
        self.equilibrium_lattice_coordinates = change_basis(self.in_plane_basis, self.lattice_points)
        self.num_moveable_substrate_atoms = np.prod(self.lattice_shape - [0, 0, 1])
        self.reciprocal_basis = get_reciprocal_basis(self.canonical_basis)
        self.centre_top_point = self.equilibrium_lattice_coordinates[:, (self.lattice_shape[0] - 1) // 2, (self.lattice_shape[1] - 1) // 2, -1]
        self.absorbate_check_bubble = None
        self.cartesian_absorbate_check_bubble = None
        self.calculate_absorbate_check_bubble()

        if lattice_connections is not None:
            self.lattice_connections = np.asarray(lattice_connections).T
        else:
            self.lattice_connections = get_nearest_neighbour_list(self.in_plane_basis)

    @property
    def num_iterations(self):
        return int(np.round(self.run_time / self.dt))

    @property
    def times(self):
        return np.linspace(0, self.run_time, self.num_iterations)

    def calculate_absorbate_check_bubble(self):
        min_periodicity_dist = np.min(mag(self.in_plane_basis[:, :2]) * self.lattice_shape[:2])
        if self.r0_multiple_interaction_distance_cutoff * self.r0 > min_periodicity_dist:
            raise Exception('Interaction check bubble larger than unit cell. This is unstable.')

        interaction_distance_cutoff = self.r0_multiple_interaction_distance_cutoff * self.r0
        check_range = np.arange(-50, 50)
        candidate_points = np.asarray(np.meshgrid(check_range, check_range, check_range, indexing='ij'), dtype=int)
        cartesian_candidate_points = change_basis(self.in_plane_basis, candidate_points)
        candidate_lengths = np.sqrt(np.sum(cartesian_candidate_points ** 2, axis=0))
        in_range_mask = candidate_lengths < interaction_distance_cutoff
        self.absorbate_check_bubble = candidate_points[:, in_range_mask]
        self.cartesian_absorbate_check_bubble = cartesian_candidate_points[:, in_range_mask]

    @classmethod
    def load(cls, dir):
        return cls(
            **yaml.safe_load(open(dir / 'config.yml', "r")),
            working_directory=dir
        )

    def copy(self):
        return MDConfig(
            self.dt,
            self.run_time,
            self.temperature,
            self.lattice_shape,
            self.canonical_basis.T,
            self.conventional_cell.T,
            self.free_plane_indices,
            self.lattice_connections.T,
            self.spring_const,
            self.substrate_mass,
            self.r0_multiple_interaction_distance_cutoff,
            self.r0,
            self.D,
            self.a,
            self.absorbate_mass,
            self.working_directory,
        )
