import numpy as np

amu = 1.66e-27
picosecond = 1e-12
angstrom = 1e-10


boltzmann_constant = 1.38e-23 / angstrom**2 / amu * picosecond**2
planck_constant = 6.626e-34 / angstrom**2 / amu * picosecond
hbar = planck_constant / 2 / np.pi


def eV_to_amu_K_ps(x):
    return x * 1.6e-19 / amu / angstrom ** 2 * picosecond ** 2


def amu_K_ps_to_eV(x):
    return x / 1.6e-19 * amu * angstrom ** 2 / picosecond ** 2

print()
