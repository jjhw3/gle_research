import numpy as np
import matplotlib.pyplot as plt

spectrum = np.load('md_8^3_noise_power_spectrum.npy')
spectrum = np.load('/Users/jeremywilkinson/research_data/md_data/spectrum.npy')
times = np.arange(0, spectrum.shape[0]) * 0.01
dt = times[1] - times[0]
ws = np.fft.fftfreq(times.shape[0], dt)

plt.plot(ws, spectrum)
plt.xlim(-11, 11)
plt.show()
