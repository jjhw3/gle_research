import numpy as np
from pathlib import Path


def check_make(path):
    if not path.exists():
        path.mkdir()


def combine_isfs(path):
    isfs = {}
    size = int(path.parent.name)
    temp = int(path.name)

    for run_dir in path.glob('*'):
        print(run_dir)
        if not run_dir.name.isdecimal():
            continue

        for fil in (run_dir / 'ISFs/1.00_0.00_0.00').glob('*.npy'):
            print(fil)
            isf = np.load(fil)
            dk = float(fil.name[:-4])

            if dk not in isfs:
                isfs[dk] = np.zeros_like(isf)

            isfs[dk] += isf

    res_dir = path.parent / f'{size}_combined_isfs'
    check_make(res_dir)
    temp_dir = res_dir / f'{temp}'
    check_make(temp_dir)
    dk_dir = temp_dir / '1.00_0.00_0.00'
    check_make(dk_dir)

    for dk in isfs:
        isfs[dk] /= isfs[dk][0]
        np.save(dk_dir / f'{dk}.npy', isfs[dk])


base_path = Path('/home/jjhw3/rds/hpc-work/md/calculate_md_isf')
for size in [16]:
    for temp in [140, 160, 180, 200, 225, 250, 275, 300]:
        combine_isfs(base_path / f'{size}/{temp}')
