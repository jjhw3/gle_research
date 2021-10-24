import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def get_jump_rates(base_dir):
    dks = np.linspace(0, 2.46, 50)
    fit_mask = (dks > 1.23 - 0.5) & (dks < 1.23 + 0.5)
    fit_dks = dks[fit_mask]
    temps = np.array([140, 160, 180, 200, 225, 250, 275, 300])
    jump_rates = np.zeros(temps.shape[0])

    for i, temp in enumerate(temps):
        working_dir = base_dir / f'{temp}/1.00_0.00_0.00'
        alphas = np.load(working_dir / 'alphas.npy')
        poly = np.polyfit(fit_dks, alphas[fit_mask], 2)
        # plt.plot(dks, alphas)
        # plt.plot(fit_dks, np.polyval(poly, fit_dks))
        jump_rates[i] = np.max(np.polyval(poly, fit_dks)) / 4
    # plt.show()

    return temps, jump_rates


if __name__ == '__main__':
    for size in [8, 16, 32]:
        temps, jump_rates = get_jump_rates(Path(f'/Users/jeremywilkinson/research_data/md_data/{size}_combined_isfs'))
        plt.plot(temps, 1 / np.log(jump_rates), label=size)

    plt.legend()
    plt.show()
