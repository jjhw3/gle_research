import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from common.thermodynamics import get_mean_square_displacement
from gle.configuration import ComplexTauGLEConfig
from gle.results import ComplexGLEResult

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    config = ComplexTauGLEConfig.load(working_dir)
    direction = np.asarray([1, 0])

    batch_summary_file = open(config.batched_results_dir / 'batch_summary.yml', 'r')
    batch_summary = yaml.load(batch_summary_file)
    batch_summary_file.close()

    num_batches = batch_summary['num_batches']
    cum_msd = None

    for i in range(num_batches):
        print(i)

        results = ComplexGLEResult.load(config, config.batched_results_dir, postfix=str(i))
        if cum_msd is None:
            cum_msd = np.zeros(results.positions.shape[1])

        cum_msd += get_mean_square_displacement(results, direction)
        # plt.plot(config.times[:cum_msd.shape[0]], cum_msd / (i+1))
        # plt.show()

    # plt.plot(config.times[:cum_msd.shape[0]], cum_msd / num_batches)
    # plt.xlim(0, 10)
    # plt.ylim(0, 10)
    # plt.show()

    np.save(config.working_directory / 'mean_square_distance.npy', cum_msd / num_batches)

    print()
