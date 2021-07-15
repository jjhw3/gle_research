from common.constants import boltzmann_constant
import numpy as np


def initialize_velocities(config):
    std = np.sqrt(2 * boltzmann_constant * config.temperature / config.substrate_mass)
    initial_velocities = np.random.normal(
        # Factor of two to account for position dof.
        scale=std,
        size=config.lattice_points.shape
    )
    initial_velocities[:, :, :, 0] = 0

    if config.temperature == 0:
        return initial_velocities

    initial_velocities[:, :, :, 1:] -= initial_velocities[:, :, :, 1:].mean(axis=(1, 2, 3), keepdims=True)
    sample_std = np.sqrt(np.mean(initial_velocities[:, :, :, 1:]**2))
    initial_velocities[:, :, :, 1:] /= sample_std / std

    return initial_velocities


def boltzmann_distribution(temperature, mass, velocity):
    return np.sqrt(2 / np.pi) * (mass / (boltzmann_constant * temperature)) ** (3 / 2) * velocity ** 2 * np.exp(- mass * velocity ** 2 / (2 * boltzmann_constant * temperature))


def boltzmann_distribution_2D(temperature, mass, velocity):
    return velocity * mass / (boltzmann_constant * temperature) * np.exp(- mass * velocity ** 2 / (2 * boltzmann_constant * temperature))


def sample_temperature(config, results):
    kinetic_energy = 0.5 * config.absorbate_mass * np.sum(results.velocities**2, axis=0)
    temp = kinetic_energy.mean() / (0.5 * results.velocities.shape[0] * boltzmann_constant)
    return temp
