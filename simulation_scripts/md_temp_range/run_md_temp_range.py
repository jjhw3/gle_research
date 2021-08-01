import sys
from pathlib import Path

import numpy as np

from md.callbacks import record_temperature, record_absorbate
from md.configuration import MDConfig
from md.simulation import simulate

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    config = MDConfig.load(working_dir)

    absorbate_positions, absorbate_velocities, absorbate_potentials, absorbate_recorder = record_absorbate(config)

    substrate_displacements, substrate_velocities, absorbate_position, absorbate_velocity = simulate(
        config,
        callbacks=[
            record_temperature(config, absorbate_window_size_ps=100),
            absorbate_recorder,
        ],
    )
