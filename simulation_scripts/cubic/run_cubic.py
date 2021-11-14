import sys
from pathlib import Path

import numpy as np
from cle import eval_pot_grid

from common.lattice_tools.common import norm
from common.tools import fast_calculate_isf, stable_fit_alpha, FitFailedException, fast_auto_correlate
from gle.configuration import CubicGLEConfig
from gle.run_le import run_gle


if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    config = CubicGLEConfig.load(working_dir)

    dk_unit = norm(np.array([1, 0]))
    dk_mags = np.linspace(0, 2.46, 50)
    times = config.times[::10]
    save_mask = times < 5000
    num_iterations = 10

    isfs = {}
    e_auto = np.zeros_like(times)

    interpolation_coeffs = config.interpolation_coefficients

    for i in range(num_iterations):
        print(f'{i} / {num_iterations}')
        results = run_gle(config)
        positions = results.positions[:, ::10]
        kinetic_energies = 0.5 * config.absorbate_mass * (results.velocities[:, ::10]**2).sum(axis=0)
        potentials = np.zeros_like(kinetic_energies)

        for i in range(potentials.shape[0]):
            potentials[i] = eval_pot_grid(
                config.inv_in_plane_basis,
                interpolation_coeffs,
                positions[0, i],
                positions[1, i],
            )

        del results

        e_auto += fast_auto_correlate(kinetic_energies + potentials) / num_iterations

        # for j, dk_mag in enumerate(dk_mags):
        #     print(j)
        #     dk = dk_mag * dk_unit
        #
        #     if dk_mag not in isfs:
        #         isfs[dk_mag] = np.zeros_like(times)
        #
        #     isfs[dk_mag] += fast_calculate_isf(positions, dk)

    # isfs_dir = config.working_directory / 'combined_isfs'
    # isfs_dir.mkdir()

    # for dk in isfs:
    #     isfs[dk] /= isfs[dk][0]
    #     np.save(isfs_dir / f"{dk}.npy", isfs[dk][save_mask])

    np.save(config.working_directory / "total_energy_autocorrelation.npy", e_auto[save_mask])

    # alphas = np.zeros(len(isfs))
    #
    # for i, dk in enumerate(dk_mags):
    #     try:
    #         alpha = stable_fit_alpha(
    #             times,
    #             isfs[dk],
    #             dk,
    #             0,
    #             t_0=1,
    #             t_final=None,
    #             tol=0.1,
    #             plot_dir=config.isf_directory
    #         )
    #     except FitFailedException as e:
    #         alpha = np.nan
    #     alphas[i] = alpha

    # np.save(isfs_dir / 'alphas.npy', alphas)
