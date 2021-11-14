import cle
import matplotlib.pyplot as plt
import numpy as np
import yaml
from cle import eval_pot_grid

from common.constants import boltzmann_constant
from common.lattice_tools.common import change_basis, get_basis_rotation_matrix, rotate_basis, get_in_plane_basis
from gle.interpolation_tools import get_coefficient_matrix_grid
from gle.results import ComplexGLEResult, GLEResult


class GLEConfig:
    RUNNER = None
    RESULT_CLASS = None

    def __init__(
        self,
        run_time,
        dt,
        absorbate_mass,
        eta,
        temperature,
        basis_vectors,
        conventional_cell,
        free_plane_indices,
        working_directory,
    ):
        self.run_time = run_time
        self.dt = dt
        self.absorbate_mass = absorbate_mass
        self.eta = eta
        self.temperature = temperature
        self.basis_vectors = np.asarray(basis_vectors).T
        self.canonical_basis = self.basis_vectors[:2, :2]
        self.conventional_cell = np.asarray(conventional_cell).T
        self.free_plane_indices = np.asarray(free_plane_indices)
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

        self.in_plane_basis_canonical_coords = get_in_plane_basis(
            np.asarray(basis_vectors).T,
            np.asarray(conventional_cell).T,
            np.asarray(free_plane_indices),
        )

        self.in_plane_rotation_matrix = get_basis_rotation_matrix(self.in_plane_basis_canonical_coords)
        self.in_plane_basis = rotate_basis(self.in_plane_basis_canonical_coords)[:2, :2]
        self.inv_in_plane_basis = np.linalg.inv(self.in_plane_basis)

        self.potential_grid = np.load(self.working_directory / 'potential_grid.npy')

    @property
    def num_iterations(self):
        return int(np.round(self.run_time / self.dt))

    @property
    def noise_stddev(self):
        return np.sqrt(2 * boltzmann_constant * self.temperature * self.absorbate_mass * self.eta / self.dt)

    @property
    def times(self):
        return np.linspace(0, self.run_time, self.num_iterations)

    @property
    def unit_cell_grid_lattice_coords(self):
        xrange = np.linspace(0, 1, self.potential_grid.shape[0])
        yrange = np.linspace(0, 1, self.potential_grid.shape[0])
        return np.asarray(np.meshgrid(xrange, yrange, indexing='ij'))

    @property
    def unit_cell_grid_cartesian_coords(self):
        return change_basis(self.in_plane_basis, self.unit_cell_grid_lattice_coords)

    @property
    def centre_top_point(self):
        return self.in_plane_basis.sum(axis=1) / 2

    @property
    def interpolation_coefficients(self):
        return get_coefficient_matrix_grid(self.potential_grid)

    def to_dict(self):
        return {
            'run_time': self.run_time,
            'dt': self.dt,
            'absorbate_mass': self.absorbate_mass,
            'eta': self.eta,
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
    def load(cls, dir_or_fil):
        if dir_or_fil.is_dir():
            dir_or_fil = dir_or_fil / 'config.yml'

        return cls(
            **yaml.safe_load(open(dir_or_fil, "r")),
            working_directory=dir_or_fil.parent
        )

    def get_blank_results(self):
        return self.RESULT_CLASS.blank_from_config(self)

    def set_plot_limits_to_first_cell(self):
        diagonal = self.in_plane_basis.sum(axis=1)
        plt.xlim(-0.1, diagonal[0] + 0.1)
        plt.ylim(-0.1, diagonal[1] + 0.1)
        plt.scatter(*self.in_plane_basis)
        plt.arrow(0, 0, *self.in_plane_basis[:, 0], length_includes_head=True, width=0.02, color='black')
        plt.arrow(0, 0, *self.in_plane_basis[:, 1], length_includes_head=True, width=0.02, color='black')

    def evaluate_potentials(self, positions):
        potentials = np.zeros(positions.shape[1])
        interpolation_coefficients = self.interpolation_coefficients

        for i in range(potentials.shape[0]):
            potentials[i] = eval_pot_grid(
                self.inv_in_plane_basis,
                interpolation_coefficients,
                positions[0, i],
                positions[1, i],
            )

        return potentials


class CubicGLEConfig(GLEConfig):
    RUNNER = cle.run_gle_cubic
    RESULT_CLASS = GLEResult

    def __init__(
        self,
        run_time,
        dt,
        absorbate_mass,
        eta,
        xhi,
        temperature,
        basis_vectors,
        conventional_cell,
        free_plane_indices,
        working_directory,
    ):
        self.xhi = xhi
        self.memory_kernel_normalization = 1
        self.discrete_decay_factor = 0

        super().__init__(
            run_time,
            dt,
            absorbate_mass,
            eta,
            temperature,
            basis_vectors,
            conventional_cell,
            free_plane_indices,
            working_directory,
        )

    @property
    def noise_stddev(self):
        return 1.0

    def to_dict(self):
        dic = super().to_dict()
        dic['xhi'] = self.xhi
        return dic

    def copy(self):
        return self.__class__(
            self.run_time,
            self.dt,
            self.absorbate_mass,
            self.eta,
            self.xhi,
            self.temperature,
            self.basis_vectors.T,
            self.conventional_cell.T,
            self.free_plane_indices,
            self.working_directory,
        )

    def calculate_time_quantities(self):
        return


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
        w_1,
        temperature,
        basis_vectors,
        conventional_cell,
        free_plane_indices,
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
            temperature,
            basis_vectors,
            conventional_cell,
            free_plane_indices,
            working_directory,
        )

    def calculate_time_quantities(self):
        self.discrete_decay_factor = np.exp(- self.dt / self.tau) if self.tau > 0 else 0
        self.normalize_kernel()

    def normalize_kernel(self):
        self.memory_kernel_normalization = 1 / np.real(1 - self.discrete_decay_factor)
        # self.temperature_normalization = calculate_kernel_temperature_normalization(self)
        # self.memory_kernel_normalization *= self.temperature_normalization

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
        temperature,
        basis_vectors,
        conventional_cell,
        free_plane_indices,
        working_directory,
    ):
        self.w_1 = w_1

        super().__init__(
            run_time,
            dt,
            absorbate_mass,
            eta,
            tau,
            w_1,
            temperature,
            basis_vectors,
            conventional_cell,
            free_plane_indices,
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

    def copy(self):
        return self.__class__(
            self.run_time,
            self.dt,
            self.absorbate_mass,
            self.eta,
            self.tau,
            self.w_1,
            self.temperature,
            self.basis_vectors.T,
            self.conventional_cell.T,
            self.free_plane_indices,
            self.working_directory,
        )
