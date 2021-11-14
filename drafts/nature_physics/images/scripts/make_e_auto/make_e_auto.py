import numpy as np
import matplotlib.pyplot as plt

from common.constants import boltzmann_constant, cm

times = np.arange(0, 5000, 0.01)
e_auto = np.load('total_energy_autocorrelation.npy')
e_auto -= np.mean(e_auto[(times > 14) & (times < 26)])
e_auto /= e_auto[0]

fit_range = (0.2, 14)
fit_mask = (times > fit_range[0]) & (times < fit_range[1])
fit_start = np.where(fit_mask)[0]
fit_times = times[fit_mask]

m, c = np.polyfit(fit_times, np.log(e_auto[fit_mask]), 1)

plt.plot(times, e_auto, c='black')
plt.plot(fit_times, np.exp(m * fit_times + c), c='r', linestyle='--', linewidth=3)
plt.text(5, 0.15, r"$\frac{\left<E(t)E(0)\right> - \left<E\right>^2}{\left<E^2\right> - \left<E\right>^2}$ ~ $\exp\left(-t/\phi\right)$", c='r')

plt.xlim(0, 15)
plt.xlabel('Time (ps)')
plt.ylabel('Normalised total\nenergy autocorrelation')
plt.gcf().set_size_inches(8.9 * cm, 0.7 * 8.9 * cm)
plt.subplots_adjust(left=0.2, bottom=0.18, right=0.955, top=0.99, wspace=0.138)
plt.savefig('../../e_auto.eps', format='eps')

plt.show()
