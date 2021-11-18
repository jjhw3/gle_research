from pathlib import Path

import numpy as np
from cle import eval_pot_grid

from common.constants import boltzmann_constant
from common.tools import fast_auto_correlate
from gle.interpolation_tools import get_coefficient_matrix_grid
from md.configuration import MDConfig

def get_z_bins(working_dir, bin_edges):
    config = MDConfig.load(working_dir)
    positions = np.load(config.working_directory / 'absorbate_positions.npy')
    potentials = np.zeros(positions.shape[1])
    hist, _ = np.histogram(positions[2], bins=bin_edges)

    return hist
    # np.save(config.working_directory / 'total_energy_autocorrelation.npy', e_auto)


if __name__ == '__main__':

    z_range = (13, 16)
    bin_edges = np.linspace(*z_range, 30)
    temp = 160
    # get_z_bins(Path('/Users/jeremywilkinson/research_data/md_data/0'), np.linspace(*(56.80623211808088, 57.92793274753702), 30))

    rootdir = Path(f'/home/jjhw3/rds/hpc-work/md/calculate_md_isf/8/{temp}')
    num_folders = 201

    cum_z_hist = None
    for i in range(num_folders):
        print(i)
        z_hist = get_z_bins(rootdir / str(i), bin_edges)
        if cum_z_hist is None:
            cum_z_hist = np.zeros_like(z_hist)
        cum_z_hist += z_hist

    z_potential = - boltzmann_constant * temp * np.log(cum_z_hist)
    z_potential -= np.min(z_potential)
    np.save(rootdir / 'z_potential.npy', z_potential)
