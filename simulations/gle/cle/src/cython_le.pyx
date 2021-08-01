#cython: language_level=3


import  numpy as np
cimport numpy as np
cimport cython
from libc.stdio cimport printf


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void vec_sum_2d(double[:] dest, double[:] src):
    dest[0] += src[0]
    dest[1] += src[1]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void real_part_vec_sum_2d(double[:] dest, complex[:] src):
    dest[0] += src[0].real
    dest[1] += src[1].real


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void vec_2d_scalar_mult(double[:] dest, double[:] vec, double scalar):
    dest[0] += vec[0] * scalar
    dest[1] += vec[1] * scalar


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void complex_vec_2d_scalar_mult(complex[:] dest, complex[:] vec, complex scalar):
    dest[0] += vec[0] * scalar
    dest[1] += vec[1] * scalar


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void complex_real_vec_2d_scalar_mult(complex[:] dest, double[:] vec, complex scalar):
    dest[0] += vec[0] * scalar
    dest[1] += vec[1] * scalar


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double mod(double a, double b):
    cdef double c = a % b

    if c < 0:
        c += b

    if c == b:
        c = 0

    return c


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double eval_pot_grid(double[:, :] lattice_coords_cob, double[:, :] potential_grid, double posx, double posy):
    cdef int grid_x_size = potential_grid.shape[0]
    cdef int grid_y_size = potential_grid.shape[1]
    cdef double first_cell_lattice_x = lattice_coords_cob[0, 0] * posx + lattice_coords_cob[0, 1] * posy
    cdef double first_cell_lattice_y = lattice_coords_cob[1, 0] * posx + lattice_coords_cob[1, 1] * posy

    cdef double float_indx = mod(first_cell_lattice_x / (1.0 / grid_x_size), grid_x_size)
    cdef double float_indy = mod(first_cell_lattice_y / (1.0 / grid_y_size), grid_y_size)
    cdef int indx = int(float_indx)
    cdef int indy = int(float_indy)
    cdef int adjacent_indx = int(mod(indx + 1, grid_x_size))
    cdef int adjacent_indy = int(mod(indy + 1, grid_y_size))
    cdef double remx = float_indx - indx
    cdef double remy = float_indy - indy

    # print(first_cell_lattice_x, first_cell_lattice_y, float_indx, float_indy, indx, indy, adjacent_indx, adjacent_indy)

    cdef double pot = potential_grid[indx, indy] * (1 - remx) * (1 - remy)
    pot += potential_grid[adjacent_indx, indy] * (remx) * (1 - remy)
    pot += potential_grid[indx, adjacent_indy] * (1 - remx) * (remy)
    pot += potential_grid[adjacent_indx, adjacent_indy] * (remx) * (remy)

    return pot


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void eval_force_from_pot(
    double[:] dest,
    double[:, :] lattice_coords_cob,
    double[:, :] potential_grid,
    double[:] vec,
    double dh = 1e-3,
):
    cdef double posx = vec[0]
    cdef double posy = vec[1]

    dest[0] = eval_pot_grid(lattice_coords_cob, potential_grid, posx + dh, posy)
    dest[0] -= eval_pot_grid(lattice_coords_cob, potential_grid, posx - dh, posy)
    dest[0] /= - 2 * dh

    dest[1] = eval_pot_grid(lattice_coords_cob, potential_grid, posx, posy + dh)
    dest[1] -= eval_pot_grid(lattice_coords_cob, potential_grid, posx, posy - dh)
    dest[1] /= - 2 * dh


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def run_gle(
    config,
    double[:, :] positions,
    double[:, :] forces,
    double[:, :] velocities,
    double[:, :] friction_forces,
    double[:, :] noise_forces,
    double[:, :] pot_grid,
):
    cdef int num_iterations = config.num_iterations
    cdef double memory_kernel_normalization = config.memory_kernel_normalization
    cdef double discrete_decay_factor = config.discrete_decay_factor
    cdef double eta = config.eta
    cdef double absorbate_mass = config.absorbate_mass
    cdef double dt = config.dt
    cdef double[:, :] lattice_coords_cob = np.linalg.inv(config.in_plane_basis)
    cdef int idx

    eval_force_from_pot(forces[:, 0], lattice_coords_cob, pot_grid, positions[:, 0])

    for idx in range(1, num_iterations):
        if idx % (num_iterations // 10) ==0:
            printf("%d / %d\n", idx, num_iterations)

        positions[:, idx] = 0
        vec_sum_2d(positions[:, idx], positions[:, idx-1])
        vec_2d_scalar_mult(positions[:, idx], velocities[:, idx-1], dt)
        vec_2d_scalar_mult(positions[:, idx], forces[:, idx-1], 0.5 / absorbate_mass * dt ** 2)

        vec_2d_scalar_mult(noise_forces[:, idx], noise_forces[:, idx-1], discrete_decay_factor)

        friction_forces[:, idx] = 0
        vec_2d_scalar_mult(friction_forces[:, idx], velocities[:, idx-1], - eta * absorbate_mass / memory_kernel_normalization)
        vec_2d_scalar_mult(friction_forces[:, idx], friction_forces[:, idx-1], discrete_decay_factor)

        forces[:, idx] = 0
        eval_force_from_pot(forces[:, idx], lattice_coords_cob, pot_grid, positions[:, idx])
        # vec_2d_scalar_mult(forces[:, idx], positions[:, idx], - absorbate_mass * 10 ** 2)
        vec_sum_2d(forces[:, idx], friction_forces[:, idx])
        vec_sum_2d(forces[:, idx], noise_forces[:, idx])

        velocities[:, idx] = 0
        vec_sum_2d(velocities[:, idx], velocities[:, idx-1])
        vec_2d_scalar_mult(velocities[:, idx], forces[:, idx-1], dt / 2 / absorbate_mass)
        vec_2d_scalar_mult(velocities[:, idx], forces[:, idx], dt / 2 / absorbate_mass)



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def run_complex_gle(
    config,
    double[:, :] positions,
    double[:, :] forces,
    double[:, :] velocities,
    complex[:, :] friction_forces,
    complex[:, :] noise_forces,
    double[:, :] pot_grid,
):
    cdef int num_iterations = config.num_iterations
    cdef double memory_kernel_normalization = config.memory_kernel_normalization
    cdef complex discrete_decay_factor = config.discrete_decay_factor
    cdef double eta = config.eta
    cdef double absorbate_mass = config.absorbate_mass
    cdef double dt = config.dt
    cdef double[:, :] lattice_coords_cob = np.linalg.inv(config.in_plane_basis)
    cdef int idx = 0

    forces[:, idx] = 0
    eval_force_from_pot(forces[:, idx], lattice_coords_cob, pot_grid, positions[:, idx])
    real_part_vec_sum_2d(forces[:, idx], friction_forces[:, idx])
    real_part_vec_sum_2d(forces[:, idx], noise_forces[:, idx])

    for idx in range(1, num_iterations):
        if idx % (num_iterations // 10) ==0:
            printf("%d / %d\n", idx, num_iterations)

        positions[:, idx] = 0
        vec_sum_2d(positions[:, idx], positions[:, idx-1])
        vec_2d_scalar_mult(positions[:, idx], velocities[:, idx-1], dt)
        vec_2d_scalar_mult(positions[:, idx], forces[:, idx-1], 0.5 / absorbate_mass * dt ** 2)

        complex_vec_2d_scalar_mult(noise_forces[:, idx], noise_forces[:, idx-1], discrete_decay_factor)

        friction_forces[:, idx] = 0
        complex_real_vec_2d_scalar_mult(friction_forces[:, idx], velocities[:, idx-1], - eta * absorbate_mass / memory_kernel_normalization)
        complex_vec_2d_scalar_mult(friction_forces[:, idx], friction_forces[:, idx-1], discrete_decay_factor)

        forces[:, idx] = 0
        eval_force_from_pot(forces[:, idx], lattice_coords_cob, pot_grid, positions[:, idx])
        # vec_2d_scalar_mult(forces[:, idx], positions[:, idx], - absorbate_mass * 10 ** 2)
        real_part_vec_sum_2d(forces[:, idx], friction_forces[:, idx])
        real_part_vec_sum_2d(forces[:, idx], noise_forces[:, idx])

        velocities[:, idx] = 0
        vec_sum_2d(velocities[:, idx], velocities[:, idx-1])
        vec_2d_scalar_mult(velocities[:, idx], forces[:, idx-1], dt / 2 / absorbate_mass)
        vec_2d_scalar_mult(velocities[:, idx], forces[:, idx], dt / 2 / absorbate_mass)
