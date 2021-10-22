import sys
from pathlib import Path

import numpy as np

from common.lattice_tools.common import norm
from common.tools import fast_calculate_isf

run_dir = Path(sys.argv[1])
dk_unit = norm(np.array([1, 0, 0]))

dk_mags = np.linspace(0, 2.46, 50)
positions = np.load(run_dir / 'absorbate_positions.npy')

isf_dir = run_dir / 'ISFs'
delta_k_dir = isf_dir / '_'.join([f"{c:.2f}" for c in dk_unit])
if not isf_dir.exists():
    isf_dir.mkdir()
if not delta_k_dir.exists():
    delta_k_dir.mkdir()

for dk_mag in dk_mags:
    print(dk_mag)
    dk = dk_mag * dk_unit
    isf = fast_calculate_isf(positions, dk)
    np.save(delta_k_dir / f"{dk_mag}.npy", isf)
