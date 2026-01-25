CPU C++ Extensions (csrc)
=========================

dfcosmic ships optional C++/OpenMP CPU extensions for a faster median filter
and dilation. These are built during installation (or manually via
``python csrc/setup.py build_ext --inplace``) and exposed as Python extension
modules.

These functions are CPU-only and expect 2D ``torch.Tensor`` inputs with a
floating dtype (the C++ code accesses data as ``float``). If the extensions are
not built, the modules below will not be importable.

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

dilation_cpp
------------

``dilation_cpp.dilation_cpu(input, k_h, k_w, origin_y, origin_x, border_value) -> torch.Tensor``

- ``input``: 2D tensor (H, W).
- ``k_h``, ``k_w``: kernel height/width (positive integers).
- ``origin_y``, ``origin_x``: origin offsets relative to the kernel center.
- ``border_value``: value used for out-of-bounds pixels.

Example:

.. code-block:: python

    import torch
    import dilation_cpp

    image = torch.rand(256, 256, dtype=torch.float32)
    dilated = dilation_cpp.dilation_cpu(
        image, k_h=3, k_w=3, origin_y=0, origin_x=0, border_value=0.0
    )
