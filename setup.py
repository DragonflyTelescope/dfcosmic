from __future__ import annotations

import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


def _openmp_flags():
    if sys.platform.startswith("win"):
        return ["/openmp", "/O2"], []
    if sys.platform == "darwin":
        return ["-O3", "-Xpreprocessor", "-fopenmp"], ["-lomp"]
    return ["-O3", "-fopenmp", "-march=native"], ["-fopenmp"]


extra_compile_args, extra_link_args = _openmp_flags()

ext_modules = [
    CppExtension(
        name="median_filter_cpp",
        sources=["csrc/median_filter.cpp"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    CppExtension(
        name="dilation_cpp",
        sources=["csrc/dilation.cpp"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
