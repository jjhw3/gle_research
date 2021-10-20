from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.stats import norm

from common.lattice_tools.common import change_basis
from md.callbacks import record_absorbate
from md.configuration import MDConfig
from md.simulation import get_nearest_equilibrium_configuration, simulate

if __name__ == '__main__':
    working_dir = Path('/Users/jeremywilkinson/research_data/md_data/pointwise_noise')
    config = MDConfig.load(working_dir)

    pnts = np.asarray([
        [0.0, 0.0],
        # [0.0, 0.3],
        # [0.0, 0.5],
        # [0.20, 0.5],
        # [0.5, 0.5],
        # [0.21, 0.21],
    ]).T

    pot_surface = np.load('/high_res_potential_grid.npy')
    fig, ax = plt.subplots()
    ax.imshow(pot_surface, extent=(0, 1, 0, 1), origin='lower')
    ax.scatter(*pnts, c='r')
    for i, pnt in enumerate(pnts.T):
        ax.annotate(str(i), pnts[:, i])
    plt.show()

    cartesian_pnts = change_basis(config.in_plane_basis[:2, :2], pnts)

    for i, pnt in enumerate(cartesian_pnts.T):
        print(i, pnts[:, i])

        absorbate_position = np.zeros(3)
        absorbate_position[2] = config.centre_top_point[2] + config.r0
        absorbate_position[:2] = pnt
        substrate_displacements, absorbate_position = get_nearest_equilibrium_configuration(
            config,
            absorbate_position,
            freeze_absorbate=True
        )

        print(absorbate_position)

        absorbate_positions, absorbate_velocities, absorbate_potentials, absorbate_forces, absorbate_recorder = record_absorbate(
            config
        )

        substrate_displacements, substrate_velocities, absorbate_position, absorbate_velocity = simulate(
            config,
            callbacks=[
                absorbate_recorder,
            ],
            initial_substrate_deltas=substrate_displacements,
            initial_absorbate_position=absorbate_position,
            freeze_substrate=False,
            freeze_absorbate=True
        )

        window_size = 12
        absorbate_forces = absorbate_forces[:, absorbate_forces.shape[1] // 10:]
        absorbate_forces = scipy.signal.convolve(absorbate_forces, np.ones(window_size)[np.newaxis, :] / window_size, mode='same')[:, ::window_size]

        mean_force = absorbate_forces.mean(axis=1)
        random_forces = absorbate_forces - mean_force[:, np.newaxis]

        rng = (-1500, 1500)
        nbins = 100
        force_rng = np.linspace(*rng, nbins)
        cs = ['b', 'g', 'r']

        plt.subplot(pnts.shape[1], 3, 3 * i + 1)
        plt.imshow(pot_surface, extent=(0, 1, 0, 1), origin='lower')
        plt.scatter(*pnts[:, i], c='r')

        for j in range(2):
            plt.subplot(pnts.shape[1], 3, 3*i + j + 2)
            force_component = random_forces[j]
            fit = norm.pdf(force_rng, loc=force_component.mean(), scale=force_component.std())
            plt.hist(force_component, bins=nbins, alpha=0.6, density=True, color=cs[j])
            plt.xlim(rng)
            plt.plot(
                force_rng,
                fit,
                c=cs[j],
                label=rf'$\mu$={mean_force[j]:.2f} $\sigma$={force_component.std():.2f}'
            )
            if i == 0:
                plt.title(f'Component {j}')

            if j == 0:
                plt.ylabel(f'Point {i}')

            plt.legend(loc='lower center')

    plt.subplots_adjust(left=0, right=0.98, top=0.968, bottom=0.027)
    plt.show()

    print()
