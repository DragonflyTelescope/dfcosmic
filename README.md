# dfcosmic

[![Tests](https://github.com/DragonflyTelescope/dfcosmic/actions/workflows/test.yml/badge.svg)](https://github.com/DragonflyTelescope/dfcosmic/actions/workflows/test.yml) 
[![Documentation Status](https://readthedocs.org/projects/dfcosmic/badge/?version=latest)](https://dfcosmic.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/1109261439.svg)](https://doi.org/10.5281/zenodo.18451350)



A high-performance Python package for cosmic ray removal strictly following the procedure outlined in [van Dokkum 2001](https://ui.adsabs.harvard.edu/abs/2001PASP..113.1420V/abstract). Although several other implementations exist, their procedures differ slightly from that described in van Dokkum 2001. In this package, we use [PyTorch](https://pytorch.org/) to achieve considerable speedup over the original implementation while retaining fidelity to the algorithmic choices presented in the original paper.

## Who is dfcosmic for?

`dfcosmic` is for people who want the *original* L.A.Cosmic algorithm, as implemented in the IRAF script `lacos_im.cl`, at pipeline speed. It was written for the nightly reduction pipeline of the MOTHRA array, which has to clean tens of thousands of large CMOS frames every night using two threads per frame. On those data a true median filter turned out to be necessary to remove cosmic rays and hot pixels without also flagging the cores of bright stars.

`dfcosmic` is a good choice if:

- **You need results that match the original algorithm.** `dfcosmic` always uses a true (non-separable) median filter, and its mask agrees closely with the IRAF mask on the *HST* WFPC2 frame from van Dokkum 2001 (see the [example](#simple-example) below).
- **You need that at scale.** With the [optional C++ median filter](#optional-c-median-filter-for-the-cpu) and two or more threads, it is faster than the other true-median implementations we tested, and on a GPU it is more than an order of magnitude faster (see the [timing comparison](#timing-comparisons)).

[astroscrappy](https://github.com/astropy/astroscrappy) with its default settings (`sepmed=True`) is the better choice if:

- **CPU speed matters more to you than exact agreement with the original algorithm.** Its separable median filter is much faster on a CPU, at the cost of a different mask, most visibly in the cores of bright stars.
- **You do not want PyTorch as a dependency**, which is a large install.
- **You need features that `dfcosmic` does not have**, such as input masks, saturation handling or a background/variance image.

## Installation

### Using PyPi

If you want to install using PyPi (which is certainly the easiest way), you can simply run

```bash
pip install dfcosmic
```

### Installing from Source

For the latest development version, install directly from the GitHub repository:

```bash
git clone https://github.com/DragonflyTelescope/dfcosmic.git
cd dfcosmic
pip install -e .
```

Both of these give you a pure-Python install that runs entirely on PyTorch (CPU or GPU). The [example notebooks](#example-notebooks) need a few extra packages, which you can get with `pip install "dfcosmic[notebooks]"`.

For development installation with documentation dependencies:

```bash
pip install -e ".[docs]"
```

### Optional: C++ median filter for the CPU

On the CPU, most of the runtime is spent in the median filter. `dfcosmic` includes an optional C++/OpenMP median filter that gives identical results to the PyTorch one but is faster. It is **not** part of the PyPI wheel and is **not** built by a plain `pip install`, because it has to be compiled against the PyTorch version you have installed. To build it you need a C++ compiler with OpenMP support and PyTorch installed *before* building. On macOS the Xcode command line tools are enough: the extension links against the OpenMP runtime bundled with the PyTorch wheel, so do not point the build at a separate `libomp` (e.g. Homebrew's) via `LDFLAGS` — two OpenMP runtimes in one process will crash.

```bash
pip install torch "setuptools>=77"
git clone https://github.com/DragonflyTelescope/dfcosmic.git
cd dfcosmic
DFCOSMIC_BUILD_CPP=1 pip install --no-build-isolation -e .
```

`--no-build-isolation` is what makes your installed PyTorch visible to the build; `DFCOSMIC_BUILD_CPP=1` turns a missing PyTorch into an error rather than silently skipping the extension. You can check that it worked with

```bash
python -c "from dfcosmic.utils import cpp_median_available; print(cpp_median_available())"
```

Once built, the extension is used automatically when `device="cpu"`. A few things to keep in mind:

- The extension is tied to the PyTorch version it was built against. After upgrading PyTorch, rebuild it by re-running the `pip install` command above.
- If the extension is not available, `dfcosmic` uses the PyTorch median filter. Passing `use_cpp=True` explicitly will warn you (once per session) when that happens; `use_cpp=False` always uses the PyTorch median filter.
- If you want CPU-specific optimizations, you can pass extra compiler flags, e.g. `CXXFLAGS="-march=native"`. The resulting binary will then only run on similar CPUs.
- Setting the environment variable `DFCOSMIC_DISABLE_CPP=1` at runtime disables the extension.

## Basic Usage
We follow the same parameter naming conventions presented in the original IRAF code.

- `objlim`: The contrast limit between CR and underlying objects
- `sigfrac`: The fractional detection limit for neighboring pixels
- `sigclip`: The detection limit for cosmic rays

The following example runs as is on a CPU. It uses a synthetic image; replace it with your own 2D image, for example from `astropy.io.fits.getdata`.

```python
import numpy as np
from dfcosmic import lacosmic

# Synthetic sky frame with 50 cosmic ray hits
rng = np.random.default_rng(0)
image = rng.normal(200, 15, (512, 512)).astype(np.float32)
image[rng.integers(0, 512, 50), rng.integers(0, 512, 50)] += 2000

clean_image, crmask = lacosmic(
    image,
    sigclip=4.5,
    sigfrac=0.5,
    objlim=1,
    gain=1,
    readnoise=5,
    niter=1,
    device="cpu",
)
print(f"{crmask.sum()} pixels flagged")
```

`lacosmic` returns the cleaned image and a boolean mask of the flagged pixels.

### Running on a GPU

Set `device="cuda"` to run on an NVIDIA GPU, or `device="mps"` on Apple Silicon. This requires a PyTorch build with support for that device; asking for a device that is not available raises an error. You can check with `torch.cuda.is_available()` or `torch.backends.mps.is_available()`.

### Gain and read noise

If you do not know the gain you can leave it out or set it to 0. It is then estimated from the image at every iteration, as in the original IRAF script, assuming that the noise is dominated by the sky background.

This estimate needs the sky background to still be in the image. For background-subtracted frames it fails with `Gain determination failed` (or gives a meaningless value), so for those you have to provide the gain. The `skyval` and `statsec` parameters of the IRAF script are not implemented.

If you know the gain, providing it is also faster, since the estimate costs an additional median filter per iteration.

### Input images

- `image` must be a 2D numpy array or torch tensor. Numpy arrays of any dtype, byte order and memory layout are accepted, so data straight from `astropy.io.fits.getdata` works without conversion. The input is never modified.
- All computations are done in single precision: the cleaned image is returned as `float32`, even for `float64` input.
- Non-finite pixels (NaN and ±inf, e.g. flagged bad pixels) are ignored: they are excluded from the gain estimate, are never flagged as cosmic rays, and are returned unchanged in the cleaned image.

## Memory-Constrained CPU Runners

On small CPU runners, PyTorch CPU convolutions can request a large temporary workspace. `dfcosmic` supports two environment variables to force safer behavior:

```bash
export DFCOSMIC_CONVOLVE_DIRECT_MAX_NUMEL=0
export DFCOSMIC_MAX_MEMORY_MB=2048
```

- `DFCOSMIC_CONVOLVE_DIRECT_MAX_NUMEL=0` forces CPU convolution onto the chunked path.
- `DFCOSMIC_MAX_MEMORY_MB=2048` reduces chunk sizes in the main memory-heavy CPU steps to fit a rough 2 GB workspace budget.

For the tightest runners, also set `cpu_threads=1` when calling `lacosmic(...)`.

### Limiting CPU threads

`cpu_threads` limits the number of CPU threads for the duration of the call only. The thread settings of PyTorch and of the other OpenMP/BLAS thread pools in the process are restored when `lacosmic` returns, and no environment variables are changed, so the rest of your program is not affected. If `cpu_threads` is not given, `lacosmic` uses whatever the process is already configured to use (e.g. `torch.get_num_threads()`).

## Simple Example

![Example](demos/example_hst.png)

## Timing Comparisons
We compare our pytorch implementation running on either a CPU (torch), CPU (torch & the [optional C++ median filter](#optional-c-median-filter-for-the-cpu)) or GPU with two popular cosmic ray removal codes: [lacosmic](https://github.com/larrybradley/lacosmic) and [astroscrappy](https://github.com/astropy/astroscrappy).

In order to run this timing comparison, we use the synthetic data described (and created) in the [astroscrappy testing suite](https://github.com/astropy/astroscrappy/blob/main/astroscrappy/tests/fake_data.py). The full notebook can be found in [demos/Comparison.ipynb](./demos/Comparison.ipynb).

![Timing Comparisons](demos/comparison.png)

## Example Notebooks

The notebooks in [demos/](./demos) are also rendered in the [documentation](https://dfcosmic.readthedocs.io):

- [QuickExample.ipynb](./demos/QuickExample.ipynb): cleaning the *HST* WFPC2 frame from van Dokkum 2001 and comparing the mask with the one from IRAF.
- [HST.ipynb](./demos/HST.ipynb): the same frame cleaned with `dfcosmic`, `astroscrappy` and `lacosmic`.
- [Comparison.ipynb](./demos/Comparison.ipynb): the timing comparison shown above.

The notebooks need a few packages that `dfcosmic` itself does not depend on: `astropy`, `matplotlib` and `cmcrameri` for all of them, and `astroscrappy` and `lacosmic` for the two comparison notebooks. You can install all of them with

```bash
pip install "dfcosmic[notebooks]"
```


## Running Tests
The unit tests can be run using the following command:

```bash
pytest
```

The default settings are in the `[tool.pytest.ini_options]` section of `pyproject.toml`. The tests for the C++ median filter are skipped unless the extension has been built.

## Contributing

Contributions are welcome! [CONTRIBUTING.md](CONTRIBUTING.md) explains how to set up a development environment, run the tests and the linter, and build the documentation. Changes between releases are listed in [CHANGELOG.md](CHANGELOG.md).

## Citation
If you use this package, please include a reference to the GitHub repository and the following Zenodo DOI: 10.5281/zenodo.18451351. Citation metadata is also available in [CITATION.cff](CITATION.cff) (the "Cite this repository" button on GitHub).



## License
The License for all past and present versions is the GPL-3.0.

## AI Disclosure
Generative AI was used for the following parts of this project:

1. Claude (Claude.ai and Claude Code) was used to help write the unit tests and to understand the original IRAF implementation.
2. ChatGPT/Codex was used to write the C++ median filter and to make the code more memory efficient.
3. Claude Code was used to help implement the changes requested during the pyOpenSci review: packaging and continuous integration, input validation, the handling of non-finite and unrepairable pixels, the per-iteration gain estimate, and documentation.

The original implementation of the algorithm was written by the authors without AI. All code produced by AI was manually inspected for correctness.
