import numpy as np
from pathlib import Path


def get_fourier_spectrum(path):
    size = int(path.parent.name)
    temp = int(path.name)

    noise_spectrum = None
    N = 0

    for run_dir in path.glob('*'):
        print(run_dir)
        if not run_dir.name.isdecimal():
            continue

        positions = np.load(run_dir / 'absorbate_positions.npy')[0]
        forces = np.diff(positions, 2) / 0.01**2
        fft = np.abs(np.fft.fft(forces))

        if noise_spectrum is None:
            noise_spectrum = fft
        else:
            noise_spectrum += fft

        N += 1

    return noise_spectrum / N


base_path = Path('/home/jjhw3/rds/hpc-work/md/calculate_md_isf')
for size in [8]:
    temp_dir = base_path / f'{size}/{160}'
    spectrum = get_fourier_spectrum(temp_dir)
    np.save(temp_dir / 'spectrum.npy', spectrum)
