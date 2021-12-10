import numpy as np
import matplotlib.pyplot as plt

from common.constants import cm

x = np.linspace(-2, 2, 10000)
dx = x[1] - x[0]
T = 0.1

U = (-np.cos(2 * np.pi * x) + 1) / 2
P = np.exp(-U/T)
P /= np.sum(P) * dx

plt.subplot(3, 1, 1)

plt.plot(x, U, c='black', label=r'$U(x)$')
plt.plot(x, 2 * np.exp(-10000*x**2), c='r', label=r'$\rho(x, t)$')
plt.ylim(-0.05, 2.1)
plt.xticks([], [])
plt.yticks([], [])
plt.ylabel(r'$\rho(x, 0)$')
plt.legend(loc='upper right', frameon=False)

plt.subplot(3, 1, 2)
plt.plot(x, U, c='black')
plt.plot(x, P * np.exp(-1*x**2), c='r')
plt.ylim(-0.05, 2.1)
plt.xticks([], [])
plt.yticks([], [])
plt.ylabel(r'$\rho(x, t_1)$')

plt.subplot(3, 1, 3)
P /= np.sum(P) * dx
plt.plot(x, U, c='black')
plt.plot(x, P * np.exp(-0.1*x**2) / 1.5, c='r')
plt.ylim(-0.05, 2.1)
plt.xticks([], [])
plt.yticks([], [])
plt.ylabel(r'$\rho(x, t_2)$')
plt.xlabel('x')

plt.gcf().set_size_inches(11 * cm, 7.5 * cm)
plt.subplots_adjust(left=0.054, bottom=0.062, right=0.988, top=0.99, hspace=0.056)
plt.savefig('../../density_function.pdf')
plt.show()
