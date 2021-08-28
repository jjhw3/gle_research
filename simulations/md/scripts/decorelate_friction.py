import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

from cle import eval_force_from_pot

from md.configuration import MDConfig

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    # config = MDConfig.load(working_dir)
    config = MDConfig.load(working_dir)
    length = 1000000

    basis_2D = config.in_plane_basis[:2, :2]
    basis_2D_inv = np.linalg.inv(basis_2D)

    times = config.times[:length]
    positions = np.load(config.working_directory / 'absorbate_positions.npy')[:2, :length]
    velocities = np.load(config.working_directory / 'absorbate_velocities.npy')[:2, :length]
    forces = np.load(config.working_directory / 'absorbate_forces.npy')[:2, :length]
    pot_surface = np.load(config.working_directory / 'potential_grid.npy')

    background_forces = np.zeros((2, forces.shape[1]))

    for i in range(times.shape[0]):
        eval_force_from_pot(
            background_forces[:, i],
            basis_2D_inv,
            pot_surface,
            positions[:2, i],
        )

    stochastic_noise = forces - background_forces

    xdot = velocities[0]
    s = stochastic_noise[0]
    m = config.absorbate_mass
    eta = 1.0

    vacf = scipy.signal.correlate(xdot, xdot)
    xdot_s_corr = scipy.signal.correlate(s, xdot)

    plt.plot((- m * eta * vacf - xdot_s_corr)[:-1] / (np.diff(xdot_s_corr) / config.dt))
    plt.show()

    print()
