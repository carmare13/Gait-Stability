from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys, os

prefix = sys.prefix  # conda env prefix

ext_modules = [
    Pybind11Extension(
        "bayesH_py",
        ["bayesH_pybind.cpp", "bayesH_core.cpp"],
        include_dirs=[os.path.join(prefix, "include")],
        library_dirs=[os.path.join(prefix, "lib")],
        libraries=["armadillo"],
        cxx_std=17,
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="bayesH_py",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)