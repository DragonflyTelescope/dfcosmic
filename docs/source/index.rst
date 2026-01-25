.. dfcosmic documentation master file, created by
   sphinx-quickstart on Wed Dec  3 20:37:05 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

dfcosmic documentation
======================

Welcome to the documentation for `dfcosmic` -- a PyTorch implementation of the LA Cosmic algorithm by `van Dokkum 2001 <https://ui.adsabs.harvard.edu/abs/2001PASP..113.1420V/abstract>`_.


You can quickly run `dfcosmic` with the following:

.. code:: python

    from dfcosmic import lacosmic

    clean, crmask = df_lacosmic(
            image=imdata,
            objlim=2,
            sigfrac=1,
            sigclip=6,
            gain=1,
            readnoise=10,
            niter=1,
            device="cuda",
        )


Runtime Options
---------------
There are three runtime options. We list them below in order of speed from the slowest to the fastest implementation:

1. CPU pure PyTorch: this implementation uses only PyTorch for all functions and can be called by adding the argument `use_cpp=False`.

2. CPU Pytorch & C++: this implementation uses Pytorch combined with certain function implemented in C++ for speed optimizations. This is the default behavior when `device='cpu'`.

3. GPU: this implementation uses PyTorch only and runs on the GPU. This runs when `device='cuda'` is set.


Main Parameters
---------------
There are several key parameters that a user can set depending on their specific use case:

1. `objlim`: the contrast limit between cosmic rays and underlying objects

2. `sigfrac`: the fractional detection limit for neighboring pixels

3. `sigclip`: the detection limit for cosmic rays

Furthermore, the user can supply the gain and readnoise. If a gain is not supplied, then it will be estimated at each iteration.

Additional parameters can be found in the API call to `lacosmic`.


Timing Analysis
---------------
.. image:: ./demos/comparison_dark.png
   :alt: Timing Analysis
   :width: 600px
   :align: center


.. toctree::
   :maxdepth: 1
   :caption: Examples:

   demos/Comparison.ipynb
   demos/Example.ipynb

.. toctree::
    :maxdepth: 2
    :caption: API:

    autoapi/index
    csrc
