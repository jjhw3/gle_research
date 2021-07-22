import numpy as np
import yaml

from common.lattice_tools import fcc
from common.lattice_tools.common import change_basis, get_lattice_points, get_basis_rotation_matrix, \
    get_reciprocal_basis


class Config:
    def __init__(
        self,
        run_time,
        dt,
        temperature,
        lattice_parameter,
        lattice_shape,
        lattice_connections,
        spring_const,
        substrate_mass,
        working_directory,
        r0_multiple_interaction_distance_cutoff,
        r0,
        D,
        a,
        absorbate_mass,
        beam_energy
    ):
        self.run_time = run_time
        self.dt = dt
        self.lattice_shape = np.asarray(lattice_shape)
        self.lattice_parameter = lattice_parameter
        self.lattice_connections = np.asarray(lattice_connections)
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

        self.num_iterations = int(np.round(run_time / dt))
        self.times = np.linspace(0, run_time, self.num_iterations)
        self.lattice_points = get_lattice_points(*lattice_shape)
        self.xy_plane_rotation_matrix = get_basis_rotation_matrix(fcc.get_fcc_111_basis(lattice_parameter))
        self.in_plane_basis = self.in_plane_rotate(fcc.get_fcc_111_basis(lattice_parameter))
        self.canonical_basis = self.in_plane_rotate(fcc.get_fcc_basis(lattice_parameter))
        self.equilibrium_lattice_coordinates = change_basis(self.in_plane_basis, self.lattice_points)
        self.num_moveable_substrate_atoms = np.prod(self.lattice_shape - [0, 0, 1])
        self.reciprocal_basis = get_reciprocal_basis(self.canonical_basis)
        self.centre_top_point = self.equilibrium_lattice_coordinates[:, (self.lattice_shape[0] - 1) // 2, (self.lattice_shape[1] - 1) // 2, -1]
        self.absorbate_check_bubble = None
        self.cartesian_absorbate_check_bubble = None
        self.calculate_absorbate_check_bubble()

    def in_plane_rotate(self, points):
        return change_basis(self.xy_plane_rotation_matrix, points)

    def calculate_absorbate_check_bubble(self):
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
            **yaml.load(open(dir / 'config.yml', "r")),
            working_directory=dir
        )
