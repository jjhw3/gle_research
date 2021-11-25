from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from common.constants import cm

dks = np.linspace(0, 2.46, 50)
fit_mask = (dks > 1.23 - 0.7) & (dks < 1.23 + 0.7)
fit_dks = dks[fit_mask]


def plot_alpha_dk(path, norm=None, s=10, **kwargs):
    alphas = np.load(path)
    alphas[0] = 0
    alphas[-1] = 0
    if norm is not None:
        poly = np.polyfit(fit_dks, alphas[fit_mask], 2)
        alphas /= np.max(np.polyval(poly, fit_dks))
        alphas *= norm
    plt.scatter(dks, alphas, s=s, **kwargs)


dirs = [
    'markovian_alphas',
    'non_markovian_alphas',
    'cubic_alphas',
]
ttfs = [0.3, 0.5, 0.7]
norms = [0.06294573372369872, 0.091227573239961, 0.12118987650132698]
# norms = [1.0, 1.0, 1.0]
cs = ['g', '#FD01FF', 'b']
labels = [
    'Markovian Langevin Equation',
    'non-Markovian Langevin Equation',
    'Cubic friction',
]
markers = ['+', 'o', 'D']
sizes = [30, 10, 10]

for i, plot_dir in enumerate(dirs):
    plot_dir = Path(plot_dir)
    for j, ttf in enumerate(ttfs):
        label = None
        if ttf == 0.3:
            label = labels[i]
        plot_alpha_dk(plot_dir / f'{ttf}.npy', norms[j], c=cs[i], label=label, marker=markers[i], s=sizes[i])

plot_alpha_dk(Path('md_alphas.npy'), marker='x', label='3D MD simulation', c='r')

plt.gcf().set_size_inches(12 * cm, 8 * cm)
plt.subplots_adjust(left=0.137, bottom=0.15, right=0.993, top=0.987)
plt.legend(loc='lower center', frameon=False)
plt.xlabel(r'Momentum transfer, |$\Delta{K}$| ($\AA^{-1}$)')
plt.ylabel(r'Normalised ISF decay rate, $\Gamma$ (ps$^{-1}$)')

plt.text(0.9, norms[0] - 0.012, r'$\phi^{-1} \approx 0.3$ps$^{-1}$')
plt.text(0.9, norms[1] - 0.012, r'$\phi^{-1} \approx 0.5$ps$^{-1}$')
plt.text(0.9, norms[2] - 0.018, r'$\phi^{-1} \approx 0.7$ps$^{-1}$')

plt.savefig('../../jump_distribution.pdf')
plt.show()

# 06 = {tuple: 2} (6, 0.30864197530864196)
# 13 = {tuple: 2} (13, 0.5)
# 20 = {tuple: 2} (20, 0.6896551724137931)
#
#
# markov
# 11 = {tuple: 2} (0.5100000000000001, 0.5050505050505051)
# 31 = {tuple: 2} (0.7100000000000003, 0.6993006993006994)
#
#
# tau = 0.2
# (0.7900000000000004, 0.3012048192771084)
#
# tau = 0.1
# (0.7300000000000003, 0.5050505050505051)
#
# tau = 0.05
# (0.7900000000000004, 0.6993006993006994)