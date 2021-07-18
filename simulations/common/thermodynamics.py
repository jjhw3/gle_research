import scipy.stats

from common.constants import boltzmann_constant
import numpy as np
import scipy.signal

from common.lattice_tools.common import change_basis, norm
from common.tools import fast_auto_correlate


def initialize_velocities(config):
    std = np.sqrt(2 * boltzmann_constant * config.temperature / config.substrate_mass)
    initial_velocities = np.random.normal(
        # Factor of two to account for position dof.
        scale=std,
        size=config.lattice_points.shape
    )
    initial_velocities[:, :, :, 0] = 0

    if config.temperature == 0:
        return initial_velocities

    initial_velocities[:, :, :, 1:] -= initial_velocities[:, :, :, 1:].mean(axis=(1, 2, 3), keepdims=True)
    sample_std = np.sqrt(np.mean(initial_velocities[:, :, :, 1:]**2))
    initial_velocities[:, :, :, 1:] /= sample_std / std

    return initial_velocities


def boltzmann_distribution(temperature, mass, velocity):
    return np.sqrt(2 / np.pi) * (mass / (boltzmann_constant * temperature)) ** (3 / 2) * velocity ** 2 * np.exp(- mass * velocity ** 2 / (2 * boltzmann_constant * temperature))


class MaxwellBoltzmannDistribution2D(scipy.stats.rv_continuous):
    def __init__(
        self,
        temperature,
        mass,
    ):
        self.temperature = temperature
        self.mass = mass

        super().__init__(a=0)

    def _pdf(self, speed):
        norm = self.mass / (boltzmann_constant * self.temperature)
        return speed * norm * np.exp(- self.mass * np.power(speed, 2) / (2 * boltzmann_constant * self.temperature))


def sample_temperature(results):
    kinetic_energy = 0.5 * results.config.absorbate_mass * np.sum(results.velocities**2, axis=0)
    temp = kinetic_energy.mean() / (0.5 * results.velocities.shape[0] * boltzmann_constant)
    return temp


def jump_count(results):
    rounded_lattice_coords = change_basis(
        np.linalg.inv(results.config.in_plane_basis),
        results.positions
    ) // 1

    different_cell_map = np.roll(rounded_lattice_coords, 1, axis=1) != rounded_lattice_coords
    different_cell_map = np.sum(different_cell_map, axis=0) > 0
    different_cell_map[0] = 0

    return different_cell_map.sum() / 2 * 3


def msd_fft(r):
    N=len(r)
    D=np.square(r).sum(axis=1)
    D=np.append(D,0)
    S2=sum([fast_auto_correlate(r[:, i]) for i in range(r.shape[1])])
    Q=2*D.sum()
    S1=np.zeros(N)
    for m in range(N):
        Q=Q-D[m-1]-D[N-m]
        S1[m]=Q/(N-m)
    return S1-2*S2


# https://www.neutron-sciences.org/articles/sfn/abs/2011/01/sfn201112010/sfn201112010.html
def get_mean_square_displacement(results, direction):
    direction = norm(direction)
    projection_1D = direction.dot(results.positions)

    num_steps = len(projection_1D)
    D = np.append(projection_1D ** 2, 0)
    auto_corr = fast_auto_correlate(projection_1D)
    Q = 2 * D.sum()
    S1 = np.zeros(num_steps)
    for m in range(num_steps):
        Q = Q - D[m - 1] - D[num_steps - m]
        S1[m] = Q / (num_steps - m)
    return S1 - 2 * auto_corr
