I must apologize for a certain degree of clutter at this stage and my embarrassing misspellings of adsorbate.

The primary loops for the Langevin simulations are available in simulations/gle/cle/src/cython_le.pyx . The two main functions are 'run_gle_cubic', 'run_gle', and 'run_complex_gle'.

The primary loop for the MD simulation is in simulations/md/run_md.py and is entitled 'simulate'.

The simulation_scripts directory contains scripts used to execute experiments on the Cambridge high performance computing cluster. The relevant folders are 'cubic', 'eta_tau_scan', and 'calculate_md_isf'.