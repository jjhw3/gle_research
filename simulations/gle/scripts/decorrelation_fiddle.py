import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cle import eval_force_from_pot

from common.tools import fast_correlate
from gle.configuration import ComplexTauGLEConfig

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    # config = MDConfig.load(working_dir)
    config = ComplexTauGLEConfig.load(working_dir)
    skip = 100
    config.dt = skip * config.dt

    kernel = np.real(np.power(config.discrete_decay_factor, np.arange(0, 100 // config.dt)) / config.memory_kernel_normalization)

    resolution = 30
    basis_2D = config.in_plane_basis[:2, :2]
    basis_2D_inv = np.linalg.inv(basis_2D)

    positions = np.load(config.working_directory / 'absorbate_positions.npy')[:, ::skip]
    velocities = np.load(config.working_directory / 'absorbate_velocities.npy')[:, ::skip]
    forces = np.load(config.working_directory / 'absorbate_forces.npy')[:, ::skip]
    noise_forces = np.load(config.working_directory / 'absorbate_noise_forces.npy')[:, ::skip]
    pot_surface = np.load(config.working_directory / 'potential_grid.npy')

    background_forces = np.zeros((2, forces.shape[1]))

    for i in range(config.num_iterations):
        eval_force_from_pot(
            background_forces[:, i],
            basis_2D_inv,
            pot_surface,
            positions[:2, i],
        )

    stochastic_noise = forces[:2] - background_forces

    plt.plot(config.times, forces[0])
    plt.plot(config.times, noise_forces[0])
    plt.plot(config.times, background_forces[0])
    plt.plot(config.times, stochastic_noise[0])
    plt.show()

    plt.plot(fast_correlate(noise_forces[0], velocities[0]))
    plt.plot(fast_correlate(velocities[0], noise_forces[0]))
    plt.show()

    print()