from setuptools import Extension, setup
from Cython.Build import cythonize



ext_modules = [
    Extension(
        "spectral_functions",
        [r"Classes/spectral_functions.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],

    )
]

setup(
    name='spectral_functions',
    ext_modules = cythonize(ext_modules),
    language="c++",
)
