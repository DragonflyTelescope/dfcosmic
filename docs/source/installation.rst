Installation
============

dfcosmic requires Python 3.10 or newer and `PyTorch <https://pytorch.org/>`_.

From PyPI
---------

.. code-block:: bash

    pip install dfcosmic

This gives you a pure-Python install that runs entirely on PyTorch, on the CPU or
on a GPU.

``pip`` installs the default PyTorch build for your platform, which on Linux includes
GPU support and is a large download. If you only need the CPU, you can install the
CPU-only build of PyTorch first:

.. code-block:: bash

    pip install --index-url https://download.pytorch.org/whl/cpu torch
    pip install dfcosmic

From source
-----------

For the latest development version:

.. code-block:: bash

    git clone https://github.com/DragonflyTelescope/dfcosmic.git
    cd dfcosmic
    pip install -e .

Optional extras
---------------

- ``pip install "dfcosmic[notebooks]"`` installs the packages needed to run the
  example notebooks: ``astropy``, ``matplotlib`` and ``cmcrameri``, and the two codes
  that dfcosmic is compared with, ``astroscrappy`` and ``lacosmic``.
- ``pip install "dfcosmic[docs]"`` installs the packages needed to build this
  documentation.

Running on a GPU
----------------

Pass ``device="cuda"`` to :func:`dfcosmic.lacosmic` to run on an NVIDIA GPU, or
``device="mps"`` on Apple Silicon. This requires a PyTorch build with support for
that device; you can check with ``torch.cuda.is_available()`` or
``torch.backends.mps.is_available()``.

C++ median filter for the CPU
-----------------------------

On the CPU, dfcosmic can use an optional C++ median filter that is faster than the
PyTorch one and gives identical results. It is not part of the PyPI wheel and has to
be built from source; see :doc:`csrc`.

Checking the installation
-------------------------

.. code-block:: bash

    python -c "import dfcosmic; print(dfcosmic.__version__)"
