import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common.lattice_tools.extract_potential_surface import extract_potential_surface
from md.configuration import MDConfig

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    print(sys.argv[1])
    config = MDConfig.load(working_dir)
    positions = np.load(working_dir / 'absorbate_positions.npy')
    pot_surface = extract_potential_surface(
        config.in_plane_basis[:2, :2],
        config.temperature,
        positions[:2],
        100
    )

    diag = np.diag(pot_surface)
    plt.plot(diag - diag.min())
    plt.show()

    np.save(out_dir / 'potential_grid.npy', pot_surface)
