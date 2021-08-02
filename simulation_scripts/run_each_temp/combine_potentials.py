import sys
from pathlib import Path

import numpy as np

if __name__ == '__main__':
    root_dir = Path(sys.argv[1])
    print(sys.argv[1])

    cum_potential_surface = None
    num_surfaces = 0

    for temp_dir in root_dir.glob('*'):
        print(temp_dir)
        if not temp_dir.is_dir():
            continue

        pot_surface = np.load(temp_dir / 'potential_grid.npy')

        if cum_potential_surface is None:
            cum_potential_surface = pot_surface
        else:
            cum_potential_surface += pot_surface

        num_surfaces += 1

    mean_surface = cum_potential_surface / num_surfaces
    mean_surface -= mean_surface.min()

    np.save(root_dir / 'potential_grid.npy', mean_surface)
