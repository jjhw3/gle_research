import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common.lattice_tools.extract_potential_surface import extract_potential_surface
from common.thermodynamics import sample_temperature
from gle.configuration import ComplexTauGLEConfig
from gle.run_le import run_gle

if __name__ == '__main__':
    working_dir = Path(sys.argv[1])
    print(sys.argv[1])
    config = ComplexTauGLEConfig.load(working_dir)
    config.run_time = 100000
    config.calculate_time_quantities()
    num_iterations = 10

    for i in range(num_iterations):
        print(i, '/', num_iterations)
        results = run_gle(config)
        new_pot_surface = extract_potential_surface(
            config.in_plane_basis,
            config.temperature,
            results.positions,
            config.potential_grid.shape[0]
        )

        print('Temperature:', sample_temperature(results))

        diag = np.diag(new_pot_surface)
        plt.plot(diag - diag.min())
        config.potential_grid = new_pot_surface

    plt.show()
