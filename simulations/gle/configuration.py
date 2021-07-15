import pickle

import numpy as np
import yaml

from common.constants import boltzmann_constant, eV_to_amu_K_ps
from common.lattice_tools import fcc
from common.lattice_tools.common import change_basis, get_basis_rotation_matrix


class GLEConfig:
    def __init__(
        self,
        run_time,
        dt,
        absorbate_mass,
        eta,
        lattice_parameter,
        temperature,
        working_directory,
    ):
        self.run_time = run_time
        self.dt = dt
        self.absorbate_mass = absorbate_mass
        self.eta = eta
        self.lattice_parameter = lattice_parameter
        self.temperature = temperature
        self.working_directory = working_directory

        self.num_iterations = int(np.round(run_time / dt))
        self._times = None
        self.noise_stddev = np.sqrt(2 * boltzmann_constant * temperature * absorbate_mass * eta / dt)
        self.isf_directory = self.working_directory / 'ISFs'
        self.log_isf_directory = self.isf_directory / 'log'
        if not self.isf_directory.exists():
            self.isf_directory.mkdir()
        if not self.log_isf_directory.exists():
            self.log_isf_directory.mkdir()

        self.xy_plane_rotation_matrix = get_basis_rotation_matrix(fcc.get_fcc_111_basis(lattice_parameter))
        self.in_plane_basis = self.in_plane_rotate(fcc.get_fcc_111_basis(lattice_parameter))[:2, :2]
        self.canonical_basis = self.in_plane_rotate(fcc.get_fcc_basis(lattice_parameter))[:2]
        self.inv_in_plane_basis = np.linalg.inv(self.in_plane_basis)

        # self.digitized_background_potential = pickle.load(open(working_directory / 'in_first_cell_lattice_coords_potential_interpolator.pickle', 'rb'))
        self.potential_grid = np.load(self.working_directory / 'potential_grid.npy')

    @property
    def times(self):
        if self._times is None:
            self._times = np.linspace(0, self.run_time, self.num_iterations)
        return self._times

    def to_dict(self):
        return {
            'run_time': self.run_time,
            'dt': self.dt,
            'absorbate_mass': self.absorbate_mass,
            'eta': self.eta,
            'lattice_parameter': self.lattice_parameter,
            'temperature': self.temperature,
            'working_directory': self.working_directory,
        }

    def __str__(self):
        return str(self.to_dict())

    def in_plane_rotate(self, points):
        return change_basis(self.xy_plane_rotation_matrix, points)

    def background_potential(self, position):
        in_first_cell_lattice_coords = change_basis(self.inv_in_plane_basis, position) % 1
        return self.digitized_background_potential(*in_first_cell_lattice_coords, grid=False)

    def background_force(self, position, dx=1e-3, dy=1e-3):
        positions = np.asarray([[dx, 0], [-dx, 0], [0, dy], [0, -dy]]).T + position[:, np.newaxis]
        samples = self.background_potential(positions)
        gradx = (samples[0] - samples[1]) / 2 / dx
        grady = (samples[2] - samples[3]) / 2 / dy
        return - np.asarray([gradx, grady])

    @classmethod
    def load(cls, dir):
        return cls(
            **yaml.load(open(dir / 'config.yml', "r")),
            working_directory=dir
        )


class TauGLEConfig(GLEConfig):
    def __init__(
        self,
        run_time,
        dt,
        absorbate_mass,
        eta,
        tau,
        lattice_parameter,
        temperature,
        working_directory,
    ):
        self.tau = tau
        self.discrete_decay_factor = np.exp(- dt / tau) if tau > 0 else 0
        self.memory_kernel_normalization = 1 / (1 - self.discrete_decay_factor)

        super().__init__(
            run_time,
            dt,
            absorbate_mass,
            eta,
            lattice_parameter,
            temperature,
            working_directory,
        )

    def to_dict(self):
        dic = super(TauGLEConfig, self).to_dict()
        dic['tau'] = self.tau
        return dic


class ComplexTauGLEConfig(TauGLEConfig):
    def __init__(
        self,
        run_time,
        dt,
        absorbate_mass,
        eta,
        tau,
        w_1,
        lattice_parameter,
        temperature,
        working_directory,
    ):
        super().__init__(
            run_time,
            dt,
            absorbate_mass,
            eta,
            tau,
            lattice_parameter,
            temperature,
            working_directory,
        )

        self.discrete_decay_factor *= np.exp(1j * w_1 * dt)
        self.memory_kernel_normalization = np.real(1 / (1 - self.discrete_decay_factor))
