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

    absorbate_positions, absorbate_velocities, absorbate_potentials, absorbate_recorder = record_absorbate(config)
    substrate_positions, substrate_recorder = record_last_N_substrate_positions(config, -1)

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
    absorbate_position[2] += config.r0
    substrate_displacements, absorbate_position = get_nearest_equilibrium_configuration(config, absorbate_position)

    substrate_displacements, substrate_velocities, absorbate_position, absorbate_velocity = simulate(
        config,
        callbacks=[
            record_temperature(config, absorbate_window_size_ps=100),
            # record_total_energy(config),
            absorbate_recorder,
            # substrate_recorder,
        ],
        initial_substrate_deltas=substrate_displacements,
        initial_absorbate_position=absorbate_position + [0.01, 0, 0.0],
        freeze_substrate=False,
        freeze_absorbate=False,
    )

    plt.plot(absorbate_positions[0], absorbate_positions[1])
    force_aspect()
    plt.show()


    # absorbate_position = config.centre_top_point + config.in_plane_basis[:, 1] / 2  + config.in_plane_basis[:, 0] / 2
    # pots = []
    # zs = np.linspace(absorbate_position[0], absorbate_position[2]+10, 1000)
    # for z in zs:
    #     pos = absorbate_position.copy()
    #     pos[2] = z
    #     _, _, pot = calculate_absorbate_force(config, np.zeros_like(config.equilibrium_lattice_coordinates), pos)
    #     pots.append(pot)
    #
    # plt.plot(zs, pots)
    # plt.show()

    # min_z = zs[np.argmin(pots)]
    # pos = absorbate_position.copy()
    # pos[2] = min_z + [0.01]
    # print(np.sqrt())

    for i in [0]:
        fs = np.fft.fftfreq(absorbate_positions.shape[1], config.dt)
        fft = np.fft.fft(absorbate_positions[i])
        fft[0] = 0
        plt.plot(fs, np.abs(fft))

        plt.xlabel('THz')
        plt.xlim(0, 10)
        plt.axvline(1.34)
        plt.axvline(3.8)
        plt.show()

    plt.scatter(*config.equilibrium_lattice_coordinates[:2, :, :, -1])
    plt.scatter(*config.equilibrium_lattice_coordinates[:2, :, :, -2])
    plt.plot(*absorbate_positions[:2])
    force_aspect()
    plt.show()

    print()