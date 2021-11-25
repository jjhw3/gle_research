import numpy as np
from pathlib import Path

from cle import eval_force_from_pot

from gle.interpolation_tools import get_coefficient_matrix_grid
from md.configuration import MDConfig


spline_coefficient_matrix_grid = get_coefficient_matrix_grid(np.load('/home/jjhw3/code/gle_research/simulations/high_res_potential_grid.npy'))


def get_fourier_spectrum(path):
    size = int(path.parent.name)
    temp = int(path.name)

    noise_spectrum = None
    N = 0

    for run_dir in path.glob('*'):
        print(run_dir)
        if not run_dir.name.isdecimal():
            continue

        config = MDConfig.load(run_dir)
        positions = np.load(run_dir / 'absorbate_positions.npy')[0]
        forces = np.diff(positions, 2) / 0.01**2

        background_forces = np.zeros_like((2, positions.shape[0]))

        for i in range(background_forces.shape[1]):
            eval_force_from_pot(
                background_forces[i],
                config.inv_in_plane_basis[:2, :2],
                spline_coefficient_matrix_grid,
                positions[0, i],
                positions[1, i],
            )

        fft = np.abs(np.fft.fft(forces - background_forces[0]))**2

        if noise_spectrum is None:
            noise_spectrum = fft
        else:
            noise_spectrum += fft

        N += 1

    return noise_spectrum / N


base_path = Path('/home/jjhw3/rds/hpc-work/md/calculate_md_isf')
for size in [8]:
    temp_dir = base_path / f'{size}/{300}'
    spectrum = get_fourier_spectrum(temp_dir)
    np.save(temp_dir / 'spectrum.npy', spectrum)
