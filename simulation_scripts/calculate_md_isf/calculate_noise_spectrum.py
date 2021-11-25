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

    for i, run_dir in enumerate(path.glob('*')):
        print(i, run_dir)
        if not run_dir.name.isdecimal():
            continue

        config = MDConfig.load(run_dir)
        positions = np.load(run_dir / 'absorbate_positions.npy')
        forces = np.gradient(positions[0], 2) / 0.01**2
        cob = np.linalg.inv(config.in_plane_basis[:2, :2])

        background_forces = np.zeros_like(positions)

        for i in range(positions.shape[1]):
            eval_force_from_pot(
                background_forces[:, i],
                cob,
                spline_coefficient_matrix_grid,
                positions[:, i],
            )

        fft = np.abs(np.fft.fft(forces - background_forces[0]))**2

        if noise_spectrum is None:
            noise_spectrum = fft
        else:
            noise_spectrum += fft

        N += 1

    return noise_spectrum / N


base_path = Path('/home/jjhw3/rds/hpc-work/md/calculate_md_isf')
for size in [32]:
    temp_dir = base_path / f'{size}/{300}'
    spectrum = get_fourier_spectrum(temp_dir)
    np.save(temp_dir / 'spectrum.npy', spectrum)
