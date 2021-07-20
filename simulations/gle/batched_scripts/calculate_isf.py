import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from common.thermodynamics import get_mean_square_displacement
from common.tools import fast_calculate_isf
from gle.configuration import ComplexTauGLEConfig
from gle.results import ComplexGLEResult

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    config = ComplexTauGLEConfig.load(working_dir)
    delta_K = np.asarray([1, 0])

    batch_summary_file = open(config.batched_results_dir / 'batch_summary.yml', 'r')
    batch_summary = yaml.load(batch_summary_file)
    batch_summary_file.close()

    num_batches = batch_summary['num_batches']
    cum_isf = None

    for i in range(num_batches):
        print(i)

        results = ComplexGLEResult.load(config, config.batched_results_dir, postfix=str(i))
        if cum_isf is None:
            cum_isf = np.zeros(results.positions.shape[1])

        cum_isf += fast_calculate_isf(results.positions, delta_K)
        # plt.plot(config.times[:cum_isf.shape[0]], cum_isf / (i + 1))
        # plt.xlim(0, 10)
        # plt.show()

    # plt.plot(config.times[:cum_isf.shape[0]], cum_isf / num_batches)
    # plt.xlim(0, 10)
    # plt.ylim(0, 10)
    # plt.show()

    np.save(config.working_directory / 'isf.npy', cum_isf / num_batches)
    print()
