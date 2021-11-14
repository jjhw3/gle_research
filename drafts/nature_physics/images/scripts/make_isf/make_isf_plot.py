import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

from common.constants import cm

dk_mags = np.linspace(0, 2.46, 50)
times = np.arange(0, 5000, 0.01)
isf = np.load('1.23.npy')
alphas = np.load('alphas.npy')

fit_range = (5, 25)
fit_mask = (times > fit_range[0]) & (times < fit_range[1])
fit_times = times[fit_mask] - times[fit_mask][0]
m, c = np.polyfit(fit_times, np.log(isf[fit_mask]), 1)
A0 = isf[np.where(fit_mask)[0]]

plt.gcf().set_size_inches(11 * cm, 8.9*cm)
plt.plot(times, isf, c='black')
plt.plot(times[fit_mask], np.exp(m * fit_times + c), linestyle='--', c='r', linewidth=2)
plt.text(5, 0, r'$\mathrm{ISF} \sim e^{-\Gamma(1.23) t}$', c='r')
plt.axvline(3, linestyle='--', c='black', linewidth=1)
plt.xlim(-1, 60)
plt.xlabel('Time (ps)')
plt.ylabel(r'$\mathrm{ISF}(1.23, t)$ / $\mathrm{ISF}(1.23, 0)$')
plt.subplots_adjust(left=0.123, bottom=0.13, right=0.98, top=0.99, wspace=0.138)

ax1 = plt.gca()
ax2 = plt.axes([0, 0, 1, 1])
ip = InsetPosition(ax1, [0.3, 0.3, 0.66, 0.66])
ax2.set_axes_locator(ip)
plt.scatter(dk_mags, alphas, s=2)
plt.scatter(dk_mags[24], alphas[24], c='r', s=20, marker='x')
plt.xlabel(r'$|\Delta{K}|$ ($\AA^{-1}$)')
plt.ylabel(r'$\Gamma(\Delta{K})$ (ps$^{-1}$)')

plt.savefig('../../isf_dk.eps', format='eps')
plt.show()
