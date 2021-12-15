import numpy as np
from pathlib import Path

path = Path('/Users/jeremywilkinson/research/gle/drafts/nature_physics/first_submission/data')

for npy in path.glob('*.npy'):
    data = np.load(npy)
    csv_file = npy.parent / f'{npy.stem}.csv'
    if not csv_file.exists():
        np.savetxt(csv_file, data, delimiter=",")
