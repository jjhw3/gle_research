#cython: language_level=3


import  numpy as np
cimport numpy as np
cimport cython
from libc.stdio cimport printf
from common.constants import boltzmann_constant
from libc.math cimport sqrt


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
cpdef double eval_pot_grid(double[:, :] lattice_coords_cob, double[:, :, :] spline_coefficient_matrix_grid, double posx, double posy):
    cdef int grid_x_size = spline_coefficient_matrix_grid.shape[0]
    cdef int grid_y_size = spline_coefficient_matrix_grid.shape[1]
    cdef double first_cell_lattice_x = lattice_coords_cob[0, 0] * posx + lattice_coords_cob[0, 1] * posy
    cdef double first_cell_lattice_y = lattice_coords_cob[1, 0] * posx + lattice_coords_cob[1, 1] * posy

    cdef double float_indx = mod(first_cell_lattice_x / (1.0 / grid_x_size), grid_x_size)
    cdef double float_indy = mod(first_cell_lattice_y / (1.0 / grid_y_size), grid_y_size)
    cdef int indx = int(float_indx)
    cdef int indy = int(float_indy)
    cdef double interp_value = 0
    cdef int i, j

    for i in range(4):
        for j in range(4):
            interp_value += spline_coefficient_matrix_grid[indx, indy, i + 4*j] * (float_indx - indx)**i * (float_indy - indy)**j

    return interp_value


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void eval_force_from_pot(
    double[:] dest,
    double[:, :] lattice_coords_cob,
    double[:, :, :] spline_coefficient_matrix_grid,
    double[:] vec,
    double dh = 1e-3,
):
    cdef double posx = vec[0]
    cdef double posy = vec[1]

    dest[0] = eval_pot_grid(lattice_coords_cob, spline_coefficient_matrix_grid, posx + dh, posy)
    dest[0] -= eval_pot_grid(lattice_coords_cob, spline_coefficient_matrix_grid, posx - dh, posy)
    dest[0] /= - 2 * dh

    dest[1] = eval_pot_grid(lattice_coords_cob, spline_coefficient_matrix_grid, posx, posy + dh)
    dest[1] -= eval_pot_grid(lattice_coords_cob, spline_coefficient_matrix_grid, posx, posy - dh)
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
):
    cdef int num_iterations = config.num_iterations
    cdef double memory_kernel_normalization = config.memory_kernel_normalization
    cdef double discrete_decay_factor = config.discrete_decay_factor
    cdef double eta = config.eta
    cdef double absorbate_mass = config.absorbate_mass
    cdef double dt = config.dt
    cdef double[:, :] lattice_coords_cob = np.linalg.inv(config.in_plane_basis)
    cdef int idx
    cdef double[:, :, :] spline_coefficient_matrix_grid = config.interpolation_coefficients

    eval_force_from_pot(forces[:, 0], lattice_coords_cob, spline_coefficient_matrix_grid, positions[:, 0])

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
        eval_force_from_pot(forces[:, idx], lattice_coords_cob, spline_coefficient_matrix_grid, positions[:, idx])
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
):
    cdef int num_iterations = config.num_iterations
    cdef double memory_kernel_normalization = config.memory_kernel_normalization
    cdef complex discrete_decay_factor = config.discrete_decay_factor
    cdef double eta = config.eta
    cdef double absorbate_mass = config.absorbate_mass
    cdef double dt = config.dt
    cdef double[:, :] lattice_coords_cob = np.linalg.inv(config.in_plane_basis)
    cdef int idx = 0
    cdef double[:, :, :] spline_coefficient_matrix_grid = config.interpolation_coefficients

    forces[:, idx] = 0
    eval_force_from_pot(forces[:, idx], lattice_coords_cob, spline_coefficient_matrix_grid, positions[:, idx])
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
        eval_force_from_pot(forces[:, idx], lattice_coords_cob, spline_coefficient_matrix_grid, positions[:, idx])
        # vec_2d_scalar_mult(forces[:, idx], positions[:, idx], - absorbate_mass * 10 ** 2)
        real_part_vec_sum_2d(forces[:, idx], friction_forces[:, idx])
        real_part_vec_sum_2d(forces[:, idx], noise_forces[:, idx])

        velocities[:, idx] = 0
        vec_sum_2d(velocities[:, idx], velocities[:, idx-1])
        vec_2d_scalar_mult(velocities[:, idx], forces[:, idx-1], dt / 2 / absorbate_mass)
        vec_2d_scalar_mult(velocities[:, idx], forces[:, idx], dt / 2 / absorbate_mass)



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def run_gle_cubic(
    config,
    double[:, :] positions,
    double[:, :] forces,
    double[:, :] velocities,
    double[:, :] friction_forces,
    double[:, :] noise_forces,
):
    cdef int num_iterations = config.num_iterations
    cdef double eta = config.eta
    cdef double xhi = config.xhi
    cdef double mass = config.absorbate_mass
    cdef double T = config.temperature
    cdef double dt = config.dt
    cdef double[:, :] lattice_coords_cob = np.linalg.inv(config.in_plane_basis)
    cdef int idx
    cdef double[:, :, :] spline_coefficient_matrix_grid = config.interpolation_coefficients

    cdef double const_noise_var = 2 * boltzmann_constant * T * mass * eta + 4 * mass**2 * xhi * (boltzmann_constant * T) ** 2
    cdef double var_noise_var = 2 * mass**3 * xhi * boltzmann_constant * T
    cdef double noise_std
    cdef double vel_mag_sq

    eval_force_from_pot(forces[:, 0], lattice_coords_cob, spline_coefficient_matrix_grid, positions[:, 0])

    for idx in range(1, num_iterations):
        if idx % (num_iterations // 10) ==0:
            printf("%d / %d\n", idx, num_iterations)

        positions[:, idx] = 0
        vec_sum_2d(positions[:, idx], positions[:, idx-1])
        vec_2d_scalar_mult(positions[:, idx], velocities[:, idx-1], dt)
        vec_2d_scalar_mult(positions[:, idx], forces[:, idx-1], 0.5 / mass * dt ** 2)

        vel_mag_sq = (velocities[0, idx-1]**2 + velocities[1, idx-1]**2)
        noise_std = sqrt(const_noise_var + var_noise_var * vel_mag_sq) / sqrt(dt)

        friction_forces[:, idx] = 0
        vec_2d_scalar_mult(friction_forces[:, idx], velocities[:, idx-1], - eta * mass)
        vec_2d_scalar_mult(friction_forces[:, idx], velocities[:, idx-1], - xhi * mass**3 * vel_mag_sq)

        forces[:, idx] = 0
        eval_force_from_pot(forces[:, idx], lattice_coords_cob, spline_coefficient_matrix_grid, positions[:, idx])

        vec_sum_2d(forces[:, idx], friction_forces[:, idx])
        vec_2d_scalar_mult(forces[:, idx], noise_forces[:, idx], noise_std)

        velocities[:, idx] = 0
        vec_sum_2d(velocities[:, idx], velocities[:, idx-1])
        vec_2d_scalar_mult(velocities[:, idx], forces[:, idx-1], dt / 2 / mass)
        vec_2d_scalar_mult(velocities[:, idx], forces[:, idx], dt / 2 / mass)
