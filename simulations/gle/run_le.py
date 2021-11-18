import sys
import time
from pathlib import Path

import cle
import numpy as np
import matplotlib.pyplot as plt
import yaml

from common.constants import boltzmann_constant
from common.thermodynamics import sample_temperature
from gle.configuration import ComplexTauGLEConfig
from gle.interpolation_tools import get_coefficient_matrix_grid


def run_gle(config, results=None):
    if results is None:
        results = config.get_blank_results()

    results.start_time = time.time()
    print(results.start_time, ' Starting run with config:', config)

    config.RUNNER(
        config,
        results.positions,
        results.forces,
        results.velocities,
        results.friction_forces,
        results.noise_forces,
    )

    results.end_time = time.time()
    print(results.end_time, f' Finished run, final duration {results.end_time - results.start_time:.2f} seconds')

    return results


def run_gle_batched(
    config,
    batch_run_time,
):
    config = config.copy()

    for fil in config.batched_results_dir.glob('*.npy'):
        fil.unlink()

    total_run_time = config.run_time
    num_batches = int(np.ceil(total_run_time / batch_run_time))
    config.run_time = batch_run_time + config.dt
    config.calculate_time_quantities()
    results = config.get_blank_results()
    batch_temperatures = []

    print(f'Running {num_batches} batches of {batch_run_time} duration each')

    for i in range(num_batches):
        print(f'Running batch {i+1} / {num_batches}')

        run_gle(config, results)
        results.save(config.batched_results_dir, postfix=str(i), save_slice=slice(0, -1))

        batch_temperatures.append(sample_temperature(results))
        print('Batch temperature:', batch_temperatures[-1])

        if i < num_batches - 1:
            last_noise = results.noise_forces[:, -1].copy()
            results.resample_noise()
            results.noise_forces[:, 0] = last_noise
            results.positions[:, 0] = results.positions[:, -1]
            results.velocities[:, 0] = results.velocities[:, -1]
            results.friction_forces[:, 0] = results.friction_forces[:, -1]

        for fil in config.batched_results_dir.glob('*.npy'):
            if not 'position' in fil.name:
                fil.unlink()

    print('Final temp:', np.mean(batch_temperatures))

    info = {
        'num_batches': int(num_batches),
        'batch_run_time': float(batch_run_time),
        'total_run_time': float(num_batches * batch_run_time),
        'batch_temperatures': list(map(float, batch_temperatures)),
        'final_temperature': float(np.mean(batch_temperatures)),
    }

    batch_summary_file = open(config.batched_results_dir / 'batch_summary.yml', 'w')
    yaml.dump(info, batch_summary_file)
    batch_summary_file.close()

    return results


if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    config = ComplexTauGLEConfig.load(working_dir)



    # config.potential_grid *= 0 #####################################################



    results = run_gle(
        config
    )

    print('Temp:', 0.5 * config.absorbate_mass * (results.velocities**2).sum(axis=0).mean() / boltzmann_constant)
    # plt.plot(*results.positions)
    # plt.show()

    results.save()

    # last_result.save_summary()

    # from memory_profiler import memory_usage
    # mem_usage = memory_usage(f)
    # plt.plot(mem_usage)
    # plt.show()
