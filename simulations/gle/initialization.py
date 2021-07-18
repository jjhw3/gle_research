import numpy as np

from common.constants import boltzmann_constant
from common.lattice_tools.common import change_basis, norm
from common.thermodynamics import MaxwellBoltzmannDistribution2D


def initialize_position(results):
    config = results.config
    pots = results.config.potential_grid
    probabilities = np.exp(- pots / boltzmann_constant / config.temperature)
    probabilities /= probabilities.sum()

    pos_flat_ind = np.random.choice(np.prod(pots.shape), p=probabilities.flatten())
    initial_position_lattice_coords = np.unravel_index(pos_flat_ind, pots.shape)
    initial_position = change_basis(config.in_plane_basis, initial_position_lattice_coords)

    results.positions[:, 0] = initial_position


def initialize_velocities(results):
    config = results.config
    speed_dist = MaxwellBoltzmannDistribution2D(
        config.temperature,
        config.absorbate_mass
    )
    speed = speed_dist.rvs()
    initial_velocity = speed * norm(np.random.random(2))

    results.positions[:, 0] = initial_velocity


def initialize_friction(results):
    mass = results.config.absorbate_mass
    eta = results.config.eta
    results.friction_forces[:, 0] = - mass * eta * results.velocities[:, 0]
