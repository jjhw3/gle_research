import os
import sys
from pathlib import Path

import numpy as np

from gle.configuration import ComplexTauGLEConfig
from gle.run_le import run_gle_batched, run_gle


if __name__ == '__main__':
    results_dir = Path('/home/jjhw3/rds/hpc-work/results/tau_scan')
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    config = ComplexTauGLEConfig.load(working_dir)

    temperature = int(os.environ['TEMP'])
    run_no = int(os.environ['SLURM_ARRAY_TASK_ID'])

    config.temperature = temperature
    results = run_gle(config, results=None)

    np.save(results_dir / f'{temperature}/{run_no}/positions.npy', results.positions[:, :10])
    np.save(results_dir / f'{temperature}/{run_no}/velocities.npy', results.velocities[:, :10])
