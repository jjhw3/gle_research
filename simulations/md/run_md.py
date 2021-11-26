import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from common.lattice_tools.plot_tools import force_aspect
from md.callbacks import record_absorbate, record_temperature, record_last_N_substrate_positions
from md.configuration import MDConfig
from md.forces import calculate_absorbate_force
from md.simulation import simulate, get_nearest_equilibrium_configuration

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    config = MDConfig.load(working_dir)

    absorbate_positions, absorbate_velocities, absorbate_potentials, absorbate_forces, absorbate_recorder = record_absorbate(config, auto_save=True)
    # substrate_positions, substrate_recorder = record_last_N_substrate_positions(config, -1)

    # substrate_displacements, substrate_velocities, absorbate_position, absorbate_velocity = simulate(
    #     config,
    #     callbacks=[
    #         record_temperature(config, absorbate_window_size_ps=10),
    #         # record_total_energy(config),
    #         absorbate_recorder,
    #         # substrate_recorder,
    #     ],
    #     substrate_damping=0.4,
    #     absorbate_damping=0.4,
    # )

    # simulate(
    #     config,
    #     initial_substrate_deltas=substrate_displacements,
    #     initial_substrate_velocities=substrate_velocities,
    #     initial_absorbate_position=absorbate_position + [0.0, 0, 0.01],
    #     initial_absorbate_velocity=absorbate_velocity,
    #     callbacks=[
    #         record_temperature(config, absorbate_window_size_ps=10),
    #         # record_total_energy(config),
    #         absorbate_recorder,
    #         substrate_recorder,
    #     ],
    # )

    absorbate_position = config.centre_top_point + config.in_plane_basis[:, 1] / 2 + config.in_plane_basis[:, 0] / 2
    absorbate_position[2] += config.r0 * 0.9
    substrate_displacements, absorbate_position = get_nearest_equilibrium_configuration(config, absorbate_position)

    # absorbate_position += [1, 1, 0]

    substrate_displacements, substrate_velocities, absorbate_position, absorbate_velocity = simulate(
        config,
        callbacks=[
            record_temperature(config, absorbate_window_size_ps=100),
            # record_total_energy(config),
            absorbate_recorder,
            # substrate_recorder,
        ],
        initial_substrate_deltas=substrate_displacements,
        initial_absorbate_position=absorbate_position,
        freeze_substrate=False,
        freeze_absorbate=False ##################################################################################
    )

    print()
