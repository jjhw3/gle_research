import cle
import numpy as np
import yaml

from common.constants import boltzmann_constant
from common.lattice_tools import fcc
from common.lattice_tools.common import change_basis, get_basis_rotation_matrix
from gle.results import ComplexGLEResult, GLEResult
from gle.theoretics import calculate_kernel_temperature_normalization


class GLEConfig:
    RUNNER = None
    RESULT_CLASS = None

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

        self.calculate_time_quantities()

        self.isf_directory = self.working_directory / 'ISFs'
        self.log_isf_directory = self.isf_directory / 'log'
        self.summary_dir = self.working_directory / 'run_summary'
        self.batched_results_dir = self.working_directory / 'batched_results'

        self.aux_dirs = [
            self.isf_directory,
            self.log_isf_directory,
            self.summary_dir,
            self.batched_results_dir
        ]

        for aux_dir in self.aux_dirs:
            if not aux_dir.exists():
                aux_dir.mkdir()

        self.xy_plane_rotation_matrix = get_basis_rotation_matrix(fcc.get_fcc_111_basis(lattice_parameter))
        self.in_plane_basis = self.in_plane_rotate(fcc.get_fcc_111_basis(lattice_parameter))[:2, :2]
        self.canonical_basis = self.in_plane_rotate(fcc.get_fcc_basis(lattice_parameter))[:2]
        self.inv_in_plane_basis = np.linalg.inv(self.in_plane_basis)

        self.potential_grid = np.load(self.working_directory / 'potential_grid.npy')

    def calculate_time_quantities(self):
        self.num_iterations = int(np.round(self.run_time / self.dt))
        self._times = None
        self.noise_stddev = np.sqrt(2 * boltzmann_constant * self.temperature * self.absorbate_mass * self.eta / self.dt)

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

    def get_blank_results(self):
        return self.RESULT_CLASS.blank_from_config(self)


class TauGLEConfig(GLEConfig):
    RUNNER = cle.run_gle
    RESULT_CLASS = GLEResult

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
        self.discrete_decay_factor = 1
        self.temperature_normalization = 1
        self.memory_kernel_normalization = 1


        super().__init__(
            run_time,
            dt,
            absorbate_mass,
            eta,
            lattice_parameter,
            temperature,
            working_directory,
        )

    def calculate_time_quantities(self):
        super().calculate_time_quantities()
        self.discrete_decay_factor = np.exp(- self.dt / self.tau) if self.tau > 0 else 0
        self.normalize_kernel()

    def normalize_kernel(self):
        self.memory_kernel_normalization = 1 / np.real(1 - self.discrete_decay_factor)
        self.temperature_normalization = calculate_kernel_temperature_normalization(self)
        self.memory_kernel_normalization *= self.temperature_normalization

    def to_dict(self):
        dic = super(TauGLEConfig, self).to_dict()
        dic['tau'] = self.tau
        dic['temperature_normalization'] = self.temperature_normalization
        return dic


class ComplexTauGLEConfig(TauGLEConfig):
    RUNNER = cle.run_complex_gle
    RESULT_CLASS = ComplexGLEResult

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
        self.w_1 = w_1

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

    def calculate_time_quantities(self):
        super().calculate_time_quantities()
        self.discrete_decay_factor *= np.exp(1j * self.w_1 * self.dt)
        self.normalize_kernel()

    def to_dict(self):
        dic = super().to_dict()
        dic['w_1'] = self.w_1
        return dic
