from pathlib import Path

import numpy as np
from cle import eval_pot_grid
from scipy.interpolate import RegularGridInterpolator

from common.lattice_tools.common import change_basis
from common.tools import fast_auto_correlate
from gle.interpolation_tools import get_coefficient_matrix_grid
from md.configuration import MDConfig


def get_e_auto(working_dir):
    config = MDConfig.load(working_dir)
    positions = np.load(config.working_directory / 'absorbate_positions.npy')[:, ::10]

    velocities = np.gradient(positions, axis=1) / 0.01
    kinetic_energies = 0.5 * config.absorbate_mass * np.sum(velocities ** 2, axis=0)

    e_auto = fast_auto_correlate(kinetic_energies)
    return e_auto
    # np.save(config.working_directory / 'total_energy_autocorrelation.npy', e_auto)


if __name__ == '__main__':
    # rootdir = Path('/Users/jeremywilkinson/research_data/md_data/0')
    # get_e_auto(rootdir)

    for rootdir in Path('/home/jjhw3/rds/hpc-work/md/phonon_cutoff/').glob('*'):
        num_folders = 100
        cum_e_auto = None
        N = 0

        for i in range(num_folders):
            print(i)
            try:
                e_auto = get_e_auto(rootdir / str(i))
            except Exception as e:
                print('Error', rootdir / str(i), e)
                continue

            if cum_e_auto is None:
                cum_e_auto = np.zeros_like(e_auto)

            cum_e_auto += e_auto
            N += 1

        np.save(rootdir / 'total_energy_autocorrelation.npy', cum_e_auto / N)
