import sys
import time
from pathlib import Path

import cle
import matplotlib.pyplot as plt
import numpy as np

from common.constants import boltzmann_constant, amu_K_ps_to_eV
from common.lattice_tools.common import change_basis
from common.lattice_tools.extract_potential_surface import extract_potential_surface
from common.result_checks import plot_path_on_crystal
from common.thermodynamics import sample_temperature
from gle.configuration import load


class GLEResult:
    def __init__(self, config):
        self.positions = np.zeros((2, config.num_iterations))
        self.velocities = np.zeros_like(self.positions)
        self.forces = np.zeros_like(self.positions)
        self.friction_forces = np.zeros_like(self.positions)
        self.noise_forces = np.random.normal(
            0,
            config.noise_stddev,
            size=self.positions.shape
        ) / config.memory_kernel_normalization
        self.config = config

    def save(self):
        dir = self.config.working_directory
        np.save(dir / 'positions.npy', self.positions)
        np.save(dir / 'velocities.npy', self.velocities)
        np.save(dir / 'forces.npy', self.forces)
        np.save(dir / 'friction_forces.npy', self.friction_forces)
        np.save(dir / 'noise_forces.npy', self.noise_forces)


def run_gle(config, initial_position):
    results = GLEResult(config)
    results.positions[:, 0] = initial_position

    start = time.time()
    print(start, ' Starting run with config:', config)

    cle.run_gle_force_grid_pot(
        config,
        results.positions,
        results.forces,
        results.velocities,
        results.friction_forces,
        results.noise_forces,
        config.potential_grid,
    )

    end = time.time()
    print(end, f' Finished run, final duration {end - start:.2f} seconds')

    return results


if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    config = load(working_dir)
    initial_position = np.asarray([1.2783537, 0.72844401])

    for i in range(5):
        results = run_gle(config, initial_position)
        print('Observed temperature: ', sample_temperature(config, results))
        pot_surface = extract_potential_surface(config, results.positions, 60)
        plt.plot(amu_K_ps_to_eV(np.diag(pot_surface)))
        config.pot_grid = pot_surface

    plt.show()
    print()
