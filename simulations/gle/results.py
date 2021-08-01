import gc

import numpy as np
import yaml

from common.thermodynamics import sample_temperature
from gle.initialization import initialize_position, initialize_friction, initialize_velocities
from gle.result_checks import plot_maxwell_boltzmann_distributions, plot_power_spectrum, plot_path_on_crystal


def load_or_none(path):
    if path.exists():
        return np.load(path)
    return None


class GLEResult:
    def __init__(
        self,
        config,
        positions,
        velocities,
        forces,
        noise_forces,
        friction_forces,
        start_time,
        end_time
    ):
        self.config = config
        self.positions = positions
        self.velocities = velocities
        self.forces = forces
        self.noise_forces = noise_forces
        self.friction_forces = friction_forces
        self.start_time = start_time
        self.end_time = end_time

    @classmethod
    def blank_from_config(cls, config):
        positions = np.zeros((2, config.num_iterations))
        velocities = np.zeros_like(positions)
        forces = np.zeros_like(positions)
        noise_forces = None
        friction_forces = None
        start_time = None
        end_time = None

        obj = cls(
            config,
            positions,
            velocities,
            forces,
            noise_forces,
            friction_forces,
            start_time,
            end_time
        )

        obj.resample_noise()
        obj.friction_forces = np.zeros_like(obj.noise_forces)

        initialize_position(obj)
        initialize_velocities(obj)
        initialize_friction(obj)

        return obj

    def save(self, dir=None, postfix='', save_slice=None):
        if save_slice is None:
            save_slice = slice(0, self.config.num_iterations)

        if postfix != '':
            postfix = '_' + postfix

        if dir is None:
            dir = self.config.working_directory

        np.save(dir / f'absorbate_positions{postfix}.npy', self.positions[:, save_slice])
        np.save(dir / f'absorbate_velocities{postfix}.npy', self.velocities[:, save_slice])
        np.save(dir / f'absorbate_forces{postfix}.npy', self.forces[:, save_slice])
        np.save(dir / f'absorbate_friction_forces{postfix}.npy', self.friction_forces[:, save_slice])
        np.save(dir / f'absorbate_noise_forces{postfix}.npy', self.noise_forces[:, save_slice])

    @classmethod
    def load(cls, config, dir, postfix=''):
        if postfix != '':
            postfix = '_' + postfix

        positions = load_or_none(dir / f'absorbate_positions{postfix}.npy')
        velocities = load_or_none(dir / f'absorbate_velocities{postfix}.npy')
        forces = load_or_none(dir / f'absorbate_forces{postfix}.npy')
        friction_forces = load_or_none(dir / f'absorbate_friction_forces{postfix}.npy')
        noise_forces = load_or_none(dir / f'absorbate_noise_forces{postfix}.npy')

        return cls(
            config,
            positions,
            velocities,
            forces,
            friction_forces,
            noise_forces,
            None,
            None,
        )

    def resample_noise(self):
        del self.noise_forces
        gc.collect()

        self.noise_forces = np.random.normal(
            0,
            self.config.noise_stddev,
            size=self.positions.shape
        ) / self.config.memory_kernel_normalization
        self.friction_forces = np.zeros_like(self.noise_forces)

    def save_summary(self):
        summary = {
            'temperature': float(sample_temperature(self)),
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.end_time - self.start_time
        }

        plot_maxwell_boltzmann_distributions(self)
        plot_power_spectrum(self)
        plot_path_on_crystal(self)

        summary_file = open(self.config.summary_dir / 'run_summary.yml', 'w')
        yaml.dump(summary, summary_file)
        summary_file.close()


class ComplexGLEResult(GLEResult):
    def resample_noise(self):
        del self.noise_forces
        gc.collect()

        self.noise_forces = np.random.normal(
            0,
            self.config.noise_stddev,
            size=self.positions.shape,
        ).astype(np.complex128) / self.config.memory_kernel_normalization
