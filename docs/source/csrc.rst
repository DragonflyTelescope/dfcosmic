CPU C++ Extension (csrc)
========================

dfcosmic includes an optional C++/OpenMP median filter for the CPU. It gives the
same results as the PyTorch median filter but is faster.

The extension is **not** part of the wheel on PyPI and is **not** built by a plain
``pip install``: it has to be compiled against the PyTorch version you have
installed. Without it, dfcosmic runs entirely on PyTorch.

Building
--------

You need a C++ compiler with OpenMP support and PyTorch installed *before* building.
On macOS the Xcode command line tools are enough: the extension links against the
OpenMP runtime bundled with the PyTorch wheel, so do not point the build at a separate
``libomp`` (e.g. Homebrew's) via ``LDFLAGS``; two OpenMP runtimes in one process will
crash.

.. code-block:: bash

    pip install torch "setuptools>=77"
    git clone https://github.com/DragonflyTelescope/dfcosmic.git
    cd dfcosmic
    DFCOSMIC_BUILD_CPP=1 pip install --no-build-isolation -e .

``--no-build-isolation`` makes your installed PyTorch visible to the build, and
``DFCOSMIC_BUILD_CPP=1`` turns a missing PyTorch into an error rather than silently
skipping the extension. To check that the extension is available:

.. code-block:: python

    from dfcosmic.utils import cpp_median_available

    print(cpp_median_available())

The extension is tied to the PyTorch version it was built against, so it has to be
rebuilt after upgrading PyTorch. If it is missing or fails to load,
``lacosmic(..., use_cpp=True)`` emits a warning (once per session) and uses the
PyTorch median filter; with the default ``use_cpp=None`` the fallback is silent.
Setting the environment variable ``DFCOSMIC_DISABLE_CPP=1`` disables the extension
at runtime.

dfcosmic._median_filter_cpp
---------------------------

``dfcosmic._median_filter_cpp.median_filter_cpu(input, kernel_size) -> torch.Tensor``

- ``input``: 2D ``float32`` CPU tensor (H, W).
- ``kernel_size``: odd integer window size.
- Behavior: uses replicate-style boundary handling by clamping indices at the
  image edges.

This is a private module; the supported entry point is
``dfcosmic.utils.median_filter_cpp_torch``, which also handles dtype conversion:

.. code-block:: python

    import torch
    from dfcosmic.utils import median_filter_cpp_torch

    image = torch.rand(512, 512, dtype=torch.float32)
    filtered = median_filter_cpp_torch(image, kernel_size=5)
