import numpy as np
import matplotlib.pyplot as plt

from common.constants import cm

chi1 = 1
chi2 = 1
eta1 = 5

plt.scatter([-chi1, 0, chi1], [chi2, eta1, chi2])
plt.ylim(-1, 7)
plt.gca().spines['bottom'].set_position('zero')
plt.gca().spines['left'].set_position('zero')
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')

plt.text(- chi1, chi2 + 0.2, r'$-\chi^*$')
plt.text(0.02, eta1 - 0.2, r'$i\eta_1$')
plt.text(chi1, chi2 + 0.2, r'$\chi$')

plt.xlabel('Re($\omega$)', x=0.9)
plt.ylabel('Im($\omega$)', y=0.9)
plt.xticks([], [])
plt.yticks([], [])

plt.gcf().set_size_inches(12 * cm, 8 * cm)
plt.subplots_adjust(left=0.013, bottom=0.012, right=0.993, top=0.987)
plt.savefig('../../F_poles.pdf')
plt.show()
