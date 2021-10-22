import os
import sys
from pathlib import Path

import numpy as np

from common.lattice_tools.common import norm
from common.tools import fast_calculate_isf
from md.callbacks import record_absorbate, record_temperature
from md.configuration import MDConfig
from md.simulation import simulate

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    config = MDConfig.load(working_dir)

    delta_ks = [norm(np.array([1, 0, 0])) * 2.46, norm(np.array([1, 1, 0])) * 2.46]

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

    del absorbate_velocities, absorbate_potentials, absorbate_forces

    for delta_k in delta_ks:
        isf = fast_calculate_isf(absorbate_positions, delta_k)
        string_delta_k = '_'.join([f"{c:.2f}" for c in delta_k])
        np.save(working_dir / f"isf_{string_delta_k}.npy", isf)

    np.save(working_dir / f"absorbate_positions.npy", absorbate_positions)
