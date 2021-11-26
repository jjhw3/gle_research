import sys
from pathlib import Path

import numpy as np

from md.callbacks import record_absorbate, record_temperature
from md.configuration import MDConfig
from md.simulation import simulate, get_nearest_equilibrium_configuration

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    config = MDConfig.load(working_dir)
    mass = config.absorbate_mass

    config.absorbate_mass = 23

    absorbate_positions, absorbate_velocities, absorbate_potentials, absorbate_forces, absorbate_recorder = record_absorbate(
        config, auto_save=True)
    absorbate_position = np.array([1.77695934e+01, -1.01788843e-08, 1.43491063e+01])
    substrate_displacements, absorbate_position = get_nearest_equilibrium_configuration(config, absorbate_position)

    config.absorbate_mass = mass
    config.dt = 0.001

    (
        absorbate_positions,
        absorbate_velocities,
        absorbate_potentials,
        absorbate_forces,
        absorbate_recorder
    ) = record_absorbate(config)

    substrate_displacements, substrate_velocities, absorbate_position, absorbate_velocity = simulate(
        config,
        callbacks=[
            record_temperature(config, absorbate_window_size_ps=100),
            absorbate_recorder,
        ],
    )

