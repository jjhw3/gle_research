import numpy as np
import yaml

from common.lattice_tools.common import norm
from common.tools import fast_calculate_isf, stable_fit_alpha
from gle.results import ComplexGLEResult


def calculate_isf_batched(
    config,
    delta_K,
    time_skip_resolution=1,
):
    batch_summary_file = open(config.batched_results_dir / 'batch_summary.yml', 'r')
    batch_summary = yaml.load(batch_summary_file)
    batch_summary_file.close()

    num_batches = batch_summary['num_batches']
    cum_isf = None
    times = np.arange(0, batch_summary['batch_run_time'], config.dt)[::time_skip_resolution]

    for i in range(num_batches):
        results = ComplexGLEResult.load(config, config.batched_results_dir, postfix=str(i))
        positions = results.positions[:, ::time_skip_resolution]
        if cum_isf is None:
            cum_isf = np.zeros(positions.shape[1])

        cum_isf += fast_calculate_isf(positions, delta_K)

    return times, cum_isf / num_batches


def calculate_alpha_dk_batched(
    config,
    dk_unit,
    dk_mags,
    time_skip_resolution=1
):
    dk_unit = norm(dk_unit)
    alphas = np.zeros_like(dk_mags)

    for i, dk_mag in enumerate(dk_mags):
        print(f'Fitting alpha to dk={dk_mag}')

        delta_K = dk_unit * dk_mag
        times, isf = calculate_isf_batched(
            config,
            delta_K,
            time_skip_resolution=100
        )

        try:
            alphas[i] = stable_fit_alpha(
                times,
                isf,
                delta_K,
                1,
                t_0=0,
                tol=0.01,
            )
        except:
            alphas[i] = np.nan

    return alphas
