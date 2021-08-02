import sys
from pathlib import Path

import numpy as np

from gle.configuration import ComplexTauGLEConfig
from gle.run_le import run_gle_batched


if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    config = ComplexTauGLEConfig.load(working_dir)

    results = run_gle_batched(config, 10000)
    results.save()

