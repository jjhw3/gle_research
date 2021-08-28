import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

from cle import eval_force_from_pot

from common.tools import fast_correlate
from gle.configuration import ComplexTauGLEConfig

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    # config = MDConfig.load(working_dir)
    config = ComplexTauGLEConfig.load(working_dir)
    length = 10000000

    resolution = 30
    basis_2D = config.in_plane_basis[:2, :2]
    basis_2D_inv = np.linalg.inv(basis_2D)

    times = config.times[:length]
    kernel = np.exp(- times / config.tau)
    kernel /= kernel.sum()
    positions = np.load(config.working_directory / 'absorbate_positions.npy')[:, :length]
    velocities = np.load(config.working_directory / 'absorbate_velocities.npy')[:, :length]
    forces = np.load(config.working_directory / 'absorbate_forces.npy')[:, :length]
    noise_forces = np.load(config.working_directory / 'absorbate_noise_forces.npy')[:, :length]
    friction_forces = np.load(config.working_directory / 'absorbate_noise_forces.npy')[:, :length]
    raw_noise = np.load(config.working_directory / 'raw_noise.npy')[:, :length]
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

    # plt.plot(config.times, forces[0])
    # plt.plot(config.times, background_forces[0])
    # plt.plot(config.times, stochastic_noise[0])
    # plt.plot(config.times, noise_forces[0])
    # plt.plot(config.times, stochastic_noise[0] - friction_forces[0] - noise_forces[0])
    # plt.show()

    plt.plot(noise_forces[0])
    plt.plot(scipy.signal.convolve(raw_noise[0], kernel))
    plt.show()

    xdot = velocities[0]
    s = stochastic_noise[0]
    f = raw_noise[0]
    m = config.absorbate_mass
    eta = config.eta

    xdot_f_corr = scipy.signal.correlate(f, xdot)
    vacf = scipy.signal.correlate(xdot, xdot)
    xdot_s_corr = scipy.signal.correlate(s, xdot)

    plt.plot(xdot_s_corr)
    plt.plot(scipy.signal.convolve(xdot_f_corr - m * eta * vacf, np.pad(kernel, ((99999, 0),))))
    plt.show()

    plt.plot(np.diff(xdot_s_corr) / config.dt)
    plt.plot((xdot_f_corr - m * eta * vacf - xdot_s_corr) / config.tau)
    plt.show()

    plt.plot((xdot_f_corr - m * eta * vacf - xdot_s_corr)[:-1] / (np.diff(xdot_s_corr) / config.dt))
    plt.show()

    print()
