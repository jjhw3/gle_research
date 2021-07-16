import sys
import time
from pathlib import Path

import cle
import matplotlib.pyplot as plt
import numpy as np
import yaml

from common.constants import boltzmann_constant, amu_K_ps_to_eV
from common.lattice_tools.common import change_basis
from common.lattice_tools.extract_potential_surface import extract_potential_surface
from gle.result_checks import plot_path_on_crystal, plot_power_spectrum, plot_maxwell_boltzmann_distributions
from common.thermodynamics import sample_temperature
from gle.configuration import TauGLEConfig, ComplexTauGLEConfig


class BaseGLEResult:
    def __init__(self, config):
        self.positions = np.zeros((2, config.num_iterations))
        self.velocities = np.zeros_like(self.positions)
        self.forces = np.zeros_like(self.positions)
        self.config = config
        self.start_time = None
        self.end_time = None

    def save(self):
        dir = self.config.working_directory
        np.save(dir / 'positions.npy', self.positions)
        np.save(dir / 'velocities.npy', self.velocities)
        np.save(dir / 'forces.npy', self.forces)

    def save_summary(self):
        summary = {
            'temperature': float(sample_temperature(self.config, self)),
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


class GLEResult(BaseGLEResult):
    def __init__(self, config):
        super().__init__(config)
        self.friction_forces = np.zeros_like(self.positions)
        self.noise_forces = np.random.normal(
            0,
            config.noise_stddev,
            size=self.positions.shape
        ) / config.memory_kernel_normalization

    def save(self):
        super().save()
        dir = self.config.working_directory
        np.save(dir / 'friction_forces.npy', self.friction_forces)
        np.save(dir / 'noise_forces.npy', self.noise_forces)


class ComplexGLEResult(BaseGLEResult):
    def __init__(self, config):
        super().__init__(config)
        self.friction_forces = np.zeros_like(self.positions, dtype=np.complex128)
        self.noise_forces = np.random.normal(
            0,
            config.noise_stddev,
            size=self.positions.shape,
        ).astype(np.complex128) / config.memory_kernel_normalization

    def save(self):
        super().save()
        dir = self.config.working_directory
        np.save(dir / 'friction_forces.npy', self.friction_forces)
        np.save(dir / 'noise_forces.npy', self.noise_forces)


def run_gle(
    config,
    initial_position,
    type='real',
):
    if type == 'real':
        result_class = GLEResult
        runner = cle.run_gle
    elif type == 'complex':
        result_class = ComplexGLEResult
        runner = cle.run_complex_gle

    results = result_class(config)
    results.positions[:, 0] = initial_position

    results.start_time = time.time()
    print(results.start_time, ' Starting run with config:', config)

    runner(
        config,
        results.positions,
        results.forces,
        results.velocities,
        results.friction_forces,
        results.noise_forces,
        config.potential_grid,
    )

    results.end_time = time.time()
    print(results.end_time, f' Finished run, final duration {results.end_time - results.start_time:.2f} seconds')

    return results


if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    config = ComplexTauGLEConfig.load(working_dir)
    initial_position = np.asarray([1.2783537, 0.72844401])

    results = run_gle(
        config,
        initial_position,
        'complex'
    )
    results.save_summary()
    print('Observed temperature: ', sample_temperature(config, results))
    pot_surface = extract_potential_surface(config, results.positions, 60)
    plt.plot(amu_K_ps_to_eV(np.diag(pot_surface)))
    plt.show()

    print()
