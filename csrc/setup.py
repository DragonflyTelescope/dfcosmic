"""
Setup script for compiling the C++ extension.

To build:
    python csrc/setup.py build_ext --inplace

This will create a compiled .so file that can be imported in Python.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="median_filter_cpp",
    ext_modules=[
        CppExtension(
            name="median_filter_cpp",
            sources=["csrc/median_filter.cpp"],
            extra_compile_args=["-fopenmp", "-O3", "-march=native"],
            extra_link_args=["-fopenmp"],
        ),
        CppExtension(
            name="dilation_cpp",
            sources=["csrc/dilation.cpp"],
            extra_compile_args=["-fopenmp", "-O3", "-march=native"],
            extra_link_args=["-fopenmp"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
