import numpy as np

from common.lattice_tools.common import mag, change_basis, norm


def calculate_substrate_forces(config, substrate_deltas):
    F_net = np.zeros_like(config.equilibrium_lattice_coordinates)
    substrate_potential = 0

    for connection in config.lattice_connections.T:
        # equilibrium vector of this connection
        r0 = sum(connection[i] * config.in_plane_basis[:, i] for i in range(3))[:, np.newaxis, np.newaxis, np.newaxis]
        # equilibrium length of this connection
        l0 = mag(r0)
        extensions = r0 + np.roll(substrate_deltas, - connection, axis=(1, 2, 3)) - substrate_deltas
        # current extension vectors of this connection for each mass
        lengths = mag(extensions)
        # current length of this connection for each mass
        units = extensions / lengths
        stretch = lengths - l0

        # Enforce no periodicity in config.in_plane_basis[:, 2] direction
        if connection[2] > 0:
            stretch[:, :, :, -connection[2]:] = 0

        substrate_potential += 0.5 * config.spring_const * np.sum(stretch**2)
        connection_F = config.spring_const * stretch * units

        F_net += connection_F
        F_net -= np.roll(connection_F, connection, axis=(1, 2, 3))

    return F_net, substrate_potential


def calculate_absorbate_force(config, substrate_deltas, absorbate_position):
    absorbate_lattice_coords = change_basis(np.linalg.inv(config.in_plane_basis), absorbate_position)
    nearest_lattice_point = np.round(absorbate_lattice_coords).astype(int)
    cartesian_nearest_lattice_point = change_basis(config.in_plane_basis, nearest_lattice_point)
    interaction_points = config.absorbate_check_bubble + nearest_lattice_point[:, np.newaxis]
    within_slab_mask = (interaction_points[2, :] >= 0) & (interaction_points[2, :] < config.lattice_shape[2])
    cartesian_interaction_points = config.cartesian_absorbate_check_bubble[:, within_slab_mask]
    interaction_points = interaction_points[:, within_slab_mask]
    interaction_points_first_cell = interaction_points % config.lattice_shape[:, np.newaxis]
    relative_difference = (cartesian_nearest_lattice_point - absorbate_position)[:, np.newaxis] + cartesian_interaction_points + substrate_deltas[:, interaction_points_first_cell[0], interaction_points_first_cell[1], interaction_points_first_cell[2]]
    relative_difference_mags = mag(relative_difference)[0]

    final_distance_exclusion_map = relative_difference_mags < config.r0 * config.r0_multiple_interaction_distance_cutoff
    relative_difference = relative_difference[:, final_distance_exclusion_map]
    relative_difference_mags = relative_difference_mags[final_distance_exclusion_map]
    interaction_points_first_cell = interaction_points_first_cell[:, final_distance_exclusion_map]
    cartesian_interaction_points = cartesian_interaction_points[:, final_distance_exclusion_map]

    absorbate_F = norm(relative_difference) * morse_force_mag(config, relative_difference_mags)
    absorbate_potential = np.sum(morse_potential(config, relative_difference_mags))

    # import matplotlib.pyplot as plt
    # from common.lattice_tools.plot_tools import force_aspect
    #
    # pos = (cartesian_nearest_lattice_point)[:, np.newaxis] + cartesian_interaction_points + substrate_deltas[:, interaction_points_first_cell[0], interaction_points_first_cell[1], interaction_points_first_cell[2]]
    # components = [0, 2]
    # plt.scatter(*pos[components])
    # plt.scatter(*absorbate_position[components])
    # plt.quiver(*pos[components], *absorbate_F[components])
    # force_aspect()
    # plt.show()

    # import matplotlib.pyplot as plt
    # from common.lattice_tools.plot_tools import force_aspect
    #
    # pos = (cartesian_nearest_lattice_point)[:, np.newaxis] + cartesian_interaction_points + substrate_deltas[:, interaction_points_first_cell[0], interaction_points_first_cell[1], interaction_points_first_cell[2]]
    # components = [0, 1]
    # mask = np.sqrt(np.sum((pos - absorbate_position[:, np.newaxis])**2, axis=0)) < 5
    # plt.scatter(*pos[components])
    # plt.scatter(*absorbate_position[components])
    # plt.scatter(*cartesian_nearest_lattice_point[components])
    # plt.quiver(*(pos[components][:, mask]), *(absorbate_F[components][:, mask]
    # ))
    #
    # force_aspect()
    # plt.show()

    return interaction_points_first_cell, absorbate_F, absorbate_potential


def morse_potential(config, r):
    return config.D * (1 - np.exp(-config.a*(r - config.r0))) ** 2 - config.D


def morse_force_mag(config, r):
    return 2 * config.a * config.D * (1 - np.exp(-config.a*(r-config.r0))) * np.exp(-config.a*(r-config.r0))
