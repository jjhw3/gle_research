import numpy as np
import matplotlib.pyplot as plt

ws = np.linspace(-10, 10, 10000)

tau1 = 0.6
kernel_1 = 1 / (1 + ws**2 * tau1**2)


tau2 = 0.6
w1 = 3.0
kernel_2 = np.abs(1 / (1 + 1j * (ws - w1) * tau2) + 1 / (1 + 1j * (ws + w1) * tau2))**2
kernel_2 /= np.max(kernel_2)
wmax = np.abs(ws[np.argmax(kernel_2)])
teff = 0.7


# plt.subplot(1, 2, 1)
plt.xticks([], [])
plt.xlabel('Angular frequency $\omega$')
plt.ylabel('Amplitude')
plt.plot(ws, kernel_1)
plt.axvline(1 / tau1, c='r')
plt.text(0.2, 0.44, r'$\tau^{-1}$', c='black')
plt.text(1 / tau1 + 0.1, 0.05, r'$\omega_{c}$', c='red')
plt.plot([0, 1 / tau1], [0.5, 0.5], c='black')
plt.axvline(0, c='r')
plt.ylim(0, 1.1)
plt.title('Exponential kernel')


# plt.subplot(1, 2, 2)
# plt.xticks([], [])
# plt.xlabel('Angular frequency $\omega$')
# plt.plot(ws, kernel_2)
# plt.axvline(wmax - 1 / teff)
# plt.axvline(wmax, c='green')
# plt.text(wmax - 1 / teff - 0.02, 0.44, r'$\tau_{eff}^{-1}$', c='black')
# plt.plot([wmax - 1 / teff, wmax], [0.5, 0.5], c='black')
# plt.text(wmax + 0.1, 0.2, r'$\omega_{max}^{-1}$', c='green')
# plt.ylim(0, 1.1)
# plt.title('Exponentially dampened sinusoid kernel')

plt.gcf().set_size_inches(5, 3.5)
plt.subplots_adjust(left=0.11, bottom=0.065, right=0.992, top=0.930, wspace=0.105)
plt.savefig('/Users/jeremywilkinson/research/gle/drafts/coloured_noise/images/kernel_spectra.eps', format='eps')
plt.show()
