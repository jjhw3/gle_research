import numpy as np
from Cython.Distutils import build_ext
from setuptools import setup, Extension
from pathlib import Path

cython_source_file = (Path(__file__).parent / 'src/cython_le.pyx').as_posix()

cython_cle_extension = Extension(
    'cle',
    sources=[cython_source_file],
    extra_compile_args=['-O3'],
    language='c++'
)

if __name__ == '__main__':
    setup(
        name='cle',
        ext_modules=[cython_cle_extension],
        zip_safe=True,
        include_dirs=[np.get_include()],
        cmdclass={'build_ext': build_ext}
    )
