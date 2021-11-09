from pathlib import Path

import numpy as np


bicubic_spline_inv_coeff_matrix = np.load(Path(__file__).parent / 'bicubic_spline_inverse_coefficient_matrix.npy')


def directional_circular_derrivative(grid, axis):
    return (np.roll(grid, -1, axis=axis) - np.roll(grid, 1, axis=axis)) / 2


def unit_square_walk(grid_2D, i, j):
    i_plus_1 = (i + 1) % grid_2D.shape[0]
    j_plus_1 = (j + 1) % grid_2D.shape[1]

    return [
        grid_2D[i, j],
        grid_2D[i_plus_1, j],
        grid_2D[i, j_plus_1],
        grid_2D[i_plus_1, j_plus_1]
    ]


def get_coefficient_matrix_grid(potential_grid):
    spline_coefficient_matrix_grid = np.zeros((*potential_grid.shape, 16))

    x_derrivative_grid = directional_circular_derrivative(potential_grid, 0)
    y_derrivative_grid = directional_circular_derrivative(potential_grid, 1)
    xy_derrivative_grid = directional_circular_derrivative(x_derrivative_grid, 1)

    for i in range(potential_grid.shape[0]):
        for j in range(potential_grid.shape[1]):
            x_vec = unit_square_walk(potential_grid, i, j)
            x_vec = x_vec + unit_square_walk(x_derrivative_grid, i, j)
            x_vec = x_vec + unit_square_walk(y_derrivative_grid, i, j)
            x_vec = x_vec + unit_square_walk(xy_derrivative_grid, i, j)

            alpha_vec = bicubic_spline_inv_coeff_matrix.dot(x_vec)
            spline_coefficient_matrix_grid[i, j] = alpha_vec

    return spline_coefficient_matrix_grid


def interpolate(spline_coefficient_matrix_grid, posx, posy):
    grid_x_size = spline_coefficient_matrix_grid.shape[0]
    grid_y_size = spline_coefficient_matrix_grid.shape[1]

    float_indx = (posx / (1.0 / grid_x_size)) % grid_x_size
    float_indy = (posy / (1.0 / grid_y_size)) % grid_y_size
    indx = int(float_indx)
    indy = int(float_indy)

    interp_value = 0
    coefficients = spline_coefficient_matrix_grid[indx, indy]
    for i in range(4):
        for j in range(4):
            interp_value += coefficients[i + 4*j] * (float_indx - indx)**i * (float_indy - indy)**j

    return interp_value

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from cle import eval_pot_grid_spline

    pots = np.load('/Users/jeremywilkinson/research/gle/simulations/high_res_potential_grid.npy')
    x = get_coefficient_matrix_grid(pots)

    new_pots = np.zeros((200, 200))
    for i in range(new_pots.shape[0]):
        for j in range(new_pots.shape[1]):
            new_pots[i, j] = interpolate(x, i / new_pots.shape[0], j / new_pots.shape[1])

    new_pots2 = np.zeros_like(new_pots)
    for i in range(new_pots2.shape[0]):
        for j in range(new_pots2.shape[1]):
            new_pots2[i, j] = eval_pot_grid_spline(np.identity(2), x, i / new_pots2.shape[0], j / new_pots2.shape[1])

    x1 = np.arange(pots.shape[0]) / pots.shape[0]
    x2 = np.arange(new_pots.shape[0]) / new_pots.shape[0]

    plt.scatter(x1, pots[:, 1], s=8)
    plt.plot(x2, new_pots[:, 1])
    plt.scatter(x2, new_pots2[:, 1], s=6)
    plt.show()

    print()
