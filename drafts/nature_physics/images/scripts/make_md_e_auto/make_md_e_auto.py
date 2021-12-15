import numpy as np
import matplotlib.pyplot as plt

from common.constants import cm
from common.tools import stable_fit_alpha

times = np.arange(0, 1000, 0.01)
e_auto = np.load('3D_md_8^3_300K_total_energy_autocorrelation.npy')[:times.shape[0]]
e_auto -= np.mean(e_auto[(times > 200) & (times < 1000)])
e_auto /= e_auto[0]
ttf = 1 / times[np.where(e_auto < 1 / np.e)[0][0]]
fit_ttf = stable_fit_alpha(
    times,
    e_auto,
    np.array([1.0, 0]),
    0,
    t_0=0.63,
    t_final=2.8,
)
# fit_ttf = 0.6838884274372572
plt.plot(
    times,
    e_auto[np.abs(times - 0.63) == 0][0] * np.exp(-times * fit_ttf) / np.exp(- 0.63 * fit_ttf),
    c='black',
)
plt.text(2.25, 0.18, f'$e^{{-{fit_ttf:.2f}t}}$')
plt.scatter(times, e_auto, marker='o', c='r', s=3, label='3D MD simulation')
plt.xlim(0, 8)
# plt.yscale('log')
plt.xlabel('Time, t (ps)')
plt.ylabel(r'$\frac{\left<E(t)E(0)\right> - \left<E\right>^2}{\left<E^2\right> - \left<E\right>^2}$')
plt.gcf().set_size_inches(12 * cm, 8 * cm)
plt.subplots_adjust(left=0.154, bottom=0.145, right=0.985, top=0.987)
plt.savefig('../../md_e_auto.pdf')

plt.show()
