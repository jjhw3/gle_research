from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common.constants import cm

fig, axs = plt.subplot_mosaic([['a)', 'b)']])

dt = 0.01
window_sizes = [1, 5, 10, 30]

base_dir = Path('./windows')
vel_range = (-10, 10)
vel_bins = np.linspace(vel_range[0], vel_range[1], 20)
vel_centers = (vel_bins[1:] + vel_bins[:-1]) / 2

for i, window_size in enumerate(window_sizes):
    force = np.load(base_dir / f'{window_size}_force.npy')
    std = np.load(base_dir / f'{window_size}_std.npy')
    axs['b)'].errorbar(vel_centers, force, yerr=std, fmt='o', ls='none', markersize=3, capsize=4)
    axs['b)'].plot(vel_centers, - 23 * 0.7 * vel_centers)

axs['b)'].set_xlabel(r'Time-averaged x-velocity ($\AA/ps$)')
axs['b)'].set_ylabel(r'Time-averaged ')

plt.gcf().set_size_inches(18.3 * cm, 9 * cm)
plt.subplots_adjust(left=0.104, bottom=0.129, right=0.983, top=0.924, wspace=0.315, hspace=0.5)
plt.show()
