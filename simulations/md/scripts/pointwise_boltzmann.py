import sys
from pathlib import Path

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

from common.constants import boltzmann_constant
from common.thermodynamics import MaxwellBoltzmannDistribution2D
from gle.configuration import ComplexTauGLEConfig

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    config = ComplexTauGLEConfig.load(working_dir)

    positions = np.load(config.working_directory / 'absorbate_positions.npy')
    velocities = np.load(config.working_directory / 'absorbate_velocities.npy')
    kinetic_energies = 0.5 * config.absorbate_mass * (velocities**2).sum(axis=0)
    speeds = np.sqrt((velocities**2).sum(axis=0))

    print('Mean temp:', kinetic_energies.mean() / boltzmann_constant)

    means, xedges, yedges, bin_numbers = scipy.stats.binned_statistic_2d(
        positions[0],
        positions[1],
        speeds,
        bins=20,
        expand_binnumbers=False
    )

    plot_range = (0, np.sqrt(5 * boltzmann_constant * config.temperature / (0.5 * config.absorbate_mass)))
    unique_bins = set(bin_numbers)

    for i, bin in enumerate(unique_bins):
        print(i, len(unique_bins))
        bin_speeds = speeds[bin_numbers == bin]
        if len(bin_speeds) < 1000:
            continue

        vals, bins = np.histogram(bin_speeds, bins=20, range=plot_range, density=True)
        plt.plot((bins[1:] + bins[:-1]) / 2, vals)

    dist = MaxwellBoltzmannDistribution2D(config.temperature, config.absorbate_mass)
    plt.scatter(bins, dist.pdf(bins), zorder=10)

    plt.xlabel('speed')
    plt.ylabel('probability')
    plt.show()

    print()
