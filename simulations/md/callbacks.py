from common.constants import boltzmann_constant, amu_K_ps_to_eV
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from common.lattice_tools.common import mag


def record_temperature(config, absorbate_window_size_ps=1, evaluate_every=100):
    substrate_temperatures = np.zeros(config.num_iterations // evaluate_every)
    absorbate_temperatures = np.zeros(config.num_iterations // evaluate_every)
    absorbate_window_size = int(absorbate_window_size_ps / config.dt)

    def record(
        idx,
        config,
        substrate_velocities,
        absorbate_velocity,
        **kwargs
    ):
        eval_indx = idx // evaluate_every
        if idx % evaluate_every == 0:
            substrate_kinetics = 0.5 * config.substrate_mass * substrate_velocities ** 2
            mean_kinetic_per_dof = substrate_kinetics.sum() / (3 * config.num_moveable_substrate_atoms)
            substrate_temperature = 2 * mean_kinetic_per_dof / boltzmann_constant
            substrate_temperatures[eval_indx] = substrate_temperature
            absorbate_temperature = (absorbate_velocity**2 * config.absorbate_mass / boltzmann_constant).mean()
            absorbate_temperatures[eval_indx] = absorbate_temperature

        if idx == config.num_iterations - 1:
            expected_temp_sample_stddev = np.sqrt(5 / 3) * config.temperature

            print(f"mean substrate temperature: {substrate_temperatures.mean()}K")
            print(f"mean absorbate temperature: {absorbate_temperatures.mean()}K")
            plt.plot(config.times[::evaluate_every], substrate_temperatures, label='Substrate')

            plt.axhline(config.temperature + expected_temp_sample_stddev / np.sqrt(config.num_moveable_substrate_atoms), color='red')
            plt.axhline(config.temperature - expected_temp_sample_stddev / np.sqrt(config.num_moveable_substrate_atoms), color='red', label=r"Theoretical substrate 1$\sigma$ C.I.")

            # plt.plot(
            #     config.times,
            #     np.convolve(
            #         np.pad(absorbate_temperatures, ((absorbate_window_size - 1, 0),),
            #                mode='mean'),
            #         np.ones(absorbate_window_size) / absorbate_window_size,
            #         mode='valid'
            #     ),
            #     label='Adsorbate (windowed mean)',
            # )
            plt.xlabel('Run Time (ps)')
            plt.ylabel('Temperature (K)')
            plt.legend()
            plt.savefig(config.working_directory / 'temperature.png')
            # plt.show()
            plt.cla()

    return record


def record_last_N_substrate_positions(config, N):
    if N == -1:
        N = config.num_iterations

    N = int(np.min([N, config.num_iterations]))
    substrate_positions = np.zeros((3, N) + config.lattice_points.shape[1:])

    def record(
        idx,
        config,
        substrate_displacements,
        **kwargs
    ):
        if config.num_iterations - idx - 1 < N:
            substrate_positions[:, N - (config.num_iterations - idx)] = substrate_displacements

        if idx == config.num_iterations - 1:
            np.save(
                config.working_directory / f"substrate_last_deltas.npy",
                substrate_positions
            )

    return substrate_positions, record


def record_absorbate(config):
    absorbate_positions = np.zeros((3, config.num_iterations))
    absorbate_velocities = np.zeros((3, config.num_iterations))
    absorbate_potentials = np.zeros(config.num_iterations)
    absorbate_forces = np.zeros((3, config.num_iterations))

    def record(
        idx,
        config,
        absorbate_velocity,
        absorbate_position,
        absorbate_potential,
        absorbate_force,
        **kwargs
    ):
        absorbate_positions[:, idx] = absorbate_position
        absorbate_velocities[:, idx] = absorbate_velocity
        absorbate_potentials[idx] = absorbate_potential
        absorbate_forces[:, idx] = absorbate_force

        if idx == config.num_iterations - 1:
            np.save(
                config.working_directory / f"absorbate_positions.npy",
                absorbate_positions
            )
            np.save(
                config.working_directory / f"absorbate_velocities.npy",
                absorbate_velocities
            )
            np.save(
                config.working_directory / f"absorbate_potentials.npy",
                absorbate_potentials
            )
            np.save(
                config.working_directory / f"absorbate_forces.npy",
                absorbate_forces
            )

            sio.savemat(
                config.working_directory / 'adsorbate_trajectory.mat',
                {
                    'run_time': config.run_time,
                    'dt': config.dt,
                    'adsorbate_mass': config.absorbate_mass,
                    'temp': config.temperature,
                    'adsorbate_positions': absorbate_positions,
                    'adsorbate_velocities': absorbate_velocities,
                    'adsorbate_potentials': absorbate_potentials,
                    'adsorbate_forces': absorbate_forces,
                }
            )

    return absorbate_positions, absorbate_velocities, absorbate_potentials, record


def record_substrate(config):
    substrate_potentials = np.zeros(config.num_iterations)

    def record(
        idx,
        config,
        substrate_potential,
        **kwargs
    ):
        substrate_potentials[idx] = substrate_potential

    return substrate_potentials, record


def record_total_energy(config, evaluate_every=100):
    total_energies = np.zeros(config.num_iterations // evaluate_every)

    def record(
        idx,
        config,
        substrate_velocities,
        substrate_potential,
        absorbate_velocity,
        absorbate_potential,
        **kwargs
    ):
        if idx % evaluate_every == 0:
            total_energy = np.sum(substrate_potential) + 0.5 * config.substrate_mass * (mag(substrate_velocities)**2).sum()
            total_energy += absorbate_potential + 0.5 * config.absorbate_mass * mag(absorbate_velocity)**2
            total_energies[idx // evaluate_every] = total_energy

        if idx == config.num_iterations - 1:
            plt.plot(config.times[::evaluate_every], amu_K_ps_to_eV(total_energies))
            plt.xlabel('Run Time (ps)')
            plt.ylabel('Total Energy (eV)')
            plt.savefig(config.working_directory / 'total_energy.png')
            plt.show()
            print()

    return record
