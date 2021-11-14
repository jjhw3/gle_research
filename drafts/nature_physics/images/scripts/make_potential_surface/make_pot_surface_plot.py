import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from common.constants import cm

amu = 1.66e-27
picosecond = 1e-12
angstrom = 1e-10
def amu_K_ps_to_eV(x):
    return x / 1.6e-19 * amu * angstrom ** 2 / picosecond ** 2


x = np.linspace(0, 2.53851334, 101)
x = (x[1:] + x[:-1]) / 2
pot_surface = amu_K_ps_to_eV(np.fft.fftshift(np.load('high_res_potential_grid.npy'), axes=(0, 1))) * 1000


plt.gcf().set_size_inches(12 * cm, 8 * cm)

ax1 = plt.subplot2grid((1, 6), (0, 4), colspan=3)
plt.plot(pot_surface[:, 49], x, c='r', linestyle='--')
plt.yticks([0.5, 1.0, 1.5, 2.0, 2.5], [''] * 5)
plt.xlabel('Potential (meV)')
plt.annotate(r'', (0, x[0]), xytext=(67, 0), arrowprops=dict(arrowstyle='<->'))
plt.text(18, 0.05, r'$E_a=67$meV')

plt.subplot2grid((1, 6), (0, 0), colspan=4)
cont = plt.contour(x, x, pot_surface, levels=20)
plt.gca().set_aspect(1.0)
# plt.clabel(cont)
plt.xlabel(r'x coordinate ($\AA$)')
plt.ylabel(r'y coordinate ($\AA$)')
plt.axvline(x[50], c='r', linestyle='--')

plt.subplots_adjust(left=0.105, bottom=0.15, right=0.988, top=0.987, wspace=0.1)
plt.savefig(Path('../../pot_surface.eps'), format='eps')
plt.show()
