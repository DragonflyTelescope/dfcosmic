from __future__ import annotations

import os
import sys

from setuptools import setup

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _openmp_flags(torch_lib_dir: str):
    if sys.platform.startswith("win"):
        return ["/openmp", "/O2"], []
    if sys.platform == "darwin":
        compile_args = ["-O3", "-Xpreprocessor", "-fopenmp"]
        # torch wheels on macOS bundle their own libomp. Linking a second OpenMP
        # runtime (e.g. Homebrew's) into the same process segfaults, so link against
        # the copy torch already loads whenever it is there.
        if os.path.exists(os.path.join(torch_lib_dir, "libomp.dylib")):
            return compile_args, [
                f"-L{torch_lib_dir}",
                f"-Wl,-rpath,{torch_lib_dir}",
                "-lomp",
            ]
        return compile_args, ["-lomp"]
    return ["-O3", "-fopenmp"], ["-fopenmp"]


def _build_extensions():
    """
    Build the optional torch C++ median filter extension.

    The extension is opt-in: it is only built when torch is importable in the build
    environment, which with pip means installing with ``--no-build-isolation``.
    Otherwise a pure-Python package is built and the torch median filter is used.

    ``DFCOSMIC_BUILD_CPP=1`` turns a missing torch into an error instead of a silent
    skip, and ``DFCOSMIC_BUILD_CPP=0`` skips the extension even if torch is present.
    Docs builds on Read the Docs never need the compiled extension.
    """
    requested = os.environ.get("DFCOSMIC_BUILD_CPP", "").lower()
    if requested in _FALSE:
        return None
    if os.environ.get("READTHEDOCS", "").lower() == "true":
        return None

    try:
        import torch  # type: ignore
        from torch.utils.cpp_extension import BuildExtension, CppExtension  # type: ignore
    except Exception as exc:
        if requested in _TRUE:
            raise RuntimeError(
                "DFCOSMIC_BUILD_CPP is set but torch could not be imported in the "
                "build environment. Install torch first and build with "
                "`pip install --no-build-isolation .`"
            ) from exc
        return None

    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    extra_compile_args, extra_link_args = _openmp_flags(torch_lib_dir)

    ext_modules = [
        CppExtension(
            name="dfcosmic._median_filter_cpp",
            sources=["csrc/median_filter.cpp"],
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
