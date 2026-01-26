from __future__ import annotations

import os
import sys

from setuptools import setup


def _openmp_flags():
    if sys.platform.startswith("win"):
        return ["/openmp", "/O2"], []
    if sys.platform == "darwin":
        return ["-O3", "-Xpreprocessor", "-fopenmp"], ["-lomp"]
    return ["-O3", "-fopenmp", "-march=native"], ["-fopenmp"]


def _build_extensions():
    """
    Build torch C++ extensions only when torch is available AND we're not on Read the Docs.
    Docs builds don't need compiled extensions, and RTD doesn't have torch at metadata time.
    """
    if os.environ.get("READTHEDOCS", "").lower() == "true":
        return None

    try:
        from torch.utils.cpp_extension import BuildExtension, CppExtension  # type: ignore
    except Exception:
        return None

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
    cmdclass = {"build_ext": BuildExtension}
    return ext_modules, cmdclass


maybe = _build_extensions()
if maybe is None:
    setup()
else:
    ext_modules, cmdclass = maybe
    setup(ext_modules=ext_modules, cmdclass=cmdclass)
