CPU C++ Extensions (csrc)
=========================

dfcosmic ships optional C++/OpenMP CPU extensions for a faster median filter. 
This is built during installation (or manually via
``python csrc/setup.py build_ext --inplace``) and exposed as Python extension
modules.

This function is CPU-only and expects 2D ``torch.Tensor`` inputs with a
floating dtype (the C++ code accesses data as ``float``). If the extensions are
not built, the module below will not be importable.

median_filter_cpp
-----------------

``median_filter_cpp.median_filter_cpu(input, kernel_size) -> torch.Tensor``

- ``input``: 2D tensor (H, W).
- ``kernel_size``: odd integer window size.
- Behavior: uses replicate-style boundary handling by clamping indices at the
  image edges.

Example:

.. code-block:: python

    import torch
    import median_filter_cpp

    image = torch.rand(512, 512, dtype=torch.float32)
    filtered = median_filter_cpp.median_filter_cpu(image, kernel_size=5)

