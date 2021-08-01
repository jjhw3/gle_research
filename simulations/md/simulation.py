import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from common.constants import boltzmann_constant
from common.lattice_tools.common import norm, mag
from .callbacks import record_temperature
from .forces import calculate_substrate_forces, calculate_absorbate_force


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


def update_substrate(
    config,
    forces,
    prev_velocities,
    displacements,
    prev_forces=None,
    damping=0,
):
    prev_forces = prev_forces if prev_forces is not None else forces

    damping_force = - prev_velocities * damping

    velocities = prev_velocities + config.dt * (forces + prev_forces) / 2 / config.substrate_mass + damping_force * config.dt
    velocities[:, :, :, 0] = 0
    new_displacements = displacements + velocities * config.dt + 0.5 * forces / config.substrate_mass * config.dt ** 2

    return velocities, new_displacements


def update_absorbate(
    config,
    force,
    prev_velocity,
    position,
    prev_force=None,
    damping=0,
):
    prev_force = prev_force if prev_force is not None else force

    damping_force = - prev_velocity * damping

    velocity = prev_velocity + config.dt * (force + prev_force) / 2 / config.absorbate_mass + damping_force * config.dt
    new_position = position + velocity * config.dt + 0.5 * force / config.absorbate_mass * config.dt ** 2

    return velocity, new_position


def simulate(
    config,
    initial_substrate_deltas=None,
    initial_substrate_velocities=None,
    initial_absorbate_position=None,
    initial_absorbate_velocity=None,
    callbacks=tuple(),
    substrate_damping=0.0,
    absorbate_damping=0.0,
    freeze_substrate=False,
    freeze_absorbate=False,
):
    prev_substrate_velocities = initialize_velocities(config) if initial_substrate_velocities is None else initial_substrate_velocities
    substrate_displacements = np.zeros_like(config.lattice_points) if initial_substrate_deltas is None else initial_substrate_deltas
    prev_substrate_forces = None

    prev_absorbate_velocity = initial_absorbate_velocity
    if prev_absorbate_velocity is None:
        prev_absorbate_velocity = norm(np.random.random(3)) * np.sqrt(3 * boltzmann_constant * config.temperature / config.absorbate_mass)

    absorbate_position = initial_absorbate_position
    if absorbate_position is None and initial_substrate_deltas is None:
        print('calculating equilibrium pos')
        absorbate_position = config.centre_top_point + config.in_plane_basis[:, 1] / 2 + config.in_plane_basis[:, 0] / 2
        absorbate_position[2] += config.r0
        substrate_displacements, absorbate_position = get_nearest_equilibrium_configuration(
            config,
            absorbate_position,
        )

    prev_absorbate_force = None

    for idx in tqdm(range(config.num_iterations - 1)):
        substrate_forces, substrate_potential = calculate_substrate_forces(config, substrate_displacements)
        substrate_forces[:, :, :, 0] = 0
        substrate_interaction_indices, absorbate_forces, absorbate_potential = calculate_absorbate_force(
            config,
            substrate_displacements,
            absorbate_position
        )

        substrate_F_net = substrate_forces
        substrate_F_net[:, substrate_interaction_indices[0], substrate_interaction_indices[1], substrate_interaction_indices[2]] -= absorbate_forces

        absorbate_F_net = absorbate_forces.sum(axis=1)

        substrate_F_net[:, :, :, 0] = 0
        if freeze_substrate:
            substrate_F_net *= 0

        if freeze_absorbate:
            absorbate_F_net[:2] *= 0

        if idx == 0:
            for callback in callbacks:
                callback(
                    idx=idx,
                    config=config,
                    substrate_forces=substrate_forces,
                    substrate_velocities=prev_substrate_velocities,
                    substrate_displacements=substrate_displacements,
                    substrate_potential=substrate_potential,
                    absorbate_velocity=prev_absorbate_velocity,
                    absorbate_position=absorbate_position,
                    absorbate_potential=absorbate_potential,
                    absorbate_force=absorbate_F_net,
                )

        substrate_velocities, new_substrate_displacements = update_substrate(
            config,
            substrate_F_net,
            prev_substrate_velocities,
            substrate_displacements,
            prev_forces=prev_substrate_forces,
            damping=substrate_damping,
        )

        absorbate_velocity, new_absorbate_position = update_absorbate(
            config,
            absorbate_F_net,
            prev_absorbate_velocity,
            absorbate_position,
            prev_force=prev_absorbate_force,
            damping=absorbate_damping,
        )

        for callback in callbacks:
            callback(
                idx=idx + 1,
                config=config,
                substrate_forces=substrate_F_net,
                substrate_velocities=substrate_velocities,
                substrate_displacements=substrate_displacements,
                substrate_potential=substrate_potential,
                absorbate_velocity=absorbate_velocity,
                absorbate_position=absorbate_position,
                absorbate_potential=absorbate_potential,
                absorbate_force=absorbate_F_net,
            )


        prev_substrate_velocities = substrate_velocities
        prev_substrate_forces = substrate_forces
        substrate_displacements = new_substrate_displacements
        prev_absorbate_force = np.sum(absorbate_forces, axis=1)
        prev_absorbate_velocity = absorbate_velocity
        absorbate_position = new_absorbate_position

    return substrate_displacements, substrate_velocities, absorbate_position, absorbate_velocity


def get_nearest_equilibrium_configuration(
    config,
    initial_absorbate_position,
):
    config = config.copy()
    config.run_time = 10
    config.temperature = 0

    absorbate_position = initial_absorbate_position
    substrate_displacements = None
    substrate_velocities = None
    absorbate_velocity = None

    while substrate_velocities is None or (mag(substrate_velocities).mean() + mag(absorbate_velocity) > 1e-3):
        substrate_displacements, substrate_velocities, absorbate_position, absorbate_velocity = simulate(
            config,
            initial_substrate_deltas=substrate_displacements,
            initial_substrate_velocities=substrate_velocities,
            initial_absorbate_position=absorbate_position,
            initial_absorbate_velocity=absorbate_velocity,
            callbacks=(
                # record_temperature(config, absorbate_window_size_ps=10),
            ),
            substrate_damping=0.4,
            absorbate_damping=0.4,
        )

    return substrate_displacements, absorbate_position