import numpy as np
import matplotlib.pyplot as plt

from common.constants import cm

times = np.arange(0, 5000, 0.01)
e_auto = np.load('total_energy_autocorrelation.npy')
e_auto -= np.mean(e_auto[(times > 14) & (times < 26)])
e_auto /= e_auto[0]

phi = times[np.where(e_auto < 1 / np.e)[0][0]]

fit_range = (0.2, 14)
fit_mask = (times > fit_range[0]) & (times < fit_range[1])
fit_start = np.where(fit_mask)[0]
fit_times = times[fit_mask]

m, c = np.polyfit(fit_times, np.log(e_auto[fit_mask]), 1)

plt.plot(times, e_auto, c='black')
plt.plot(fit_times, np.exp(m * fit_times + c), c='r', linestyle='--', linewidth=3)
plt.annotate('', (0, 1 / np.e), xytext=(phi, 1 / np.e), arrowprops=dict(arrowstyle='<->'))
plt.annotate('', (phi, 0), xytext=(phi, 1 / np.e), arrowprops=dict(arrowstyle='<->'))
plt.text(phi / 2 - 0.2, 1 / np.e - 0.08, r'$\phi$')
plt.text(phi + 0.1, 1 / 2 / np.e - 0.08, r'$1/e$')
plt.xlim(0, 15)
plt.xlabel('Time (ps)')
plt.ylabel(r'$\frac{\left<E(t)E(0)\right> - \left<E\right>^2}{\left<E^2\right> - \left<E\right>^2}$')
plt.ylim(-0.01, 1.05)

plt.gcf().set_size_inches(12 * cm, 8 * cm)
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.993, top=0.987)
plt.savefig('../../e_auto.pdf')
plt.show()
