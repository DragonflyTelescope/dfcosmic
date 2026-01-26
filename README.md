# dfcosmic

[![Tests](https://github.com/DragonflyTelescope/dfcosmic/actions/workflows/test.yml/badge.svg)](https://github.com/DragonflyTelescope/dfcosmic/actions/workflows/test.yml) 
[![Documentation Status](https://readthedocs.org/projects/dfcosmic/badge/?version=latest)](https://dfcosmic.readthedocs.io/en/latest/?badge=latest)


A high-performance Python package for cosmic ray removal strictly following the procedure outlined in [van Dokkum 2001](https://ui.adsabs.harvard.edu/abs/2001PASP..113.1420V/abstract). Although several other implementations exist, their procedures differ slightly from that described in van Dokkum 2001. In this package, we use [PyTorch](https://pytorch.org/) to achieve considerable speedup over the original implementation while retaining fidelity to the algorithmic choices presented in the original paper.

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

Note: the CPU extensions (median filter + dilation) are compiled during install for speed. This requires a working C++ toolchain. If you need to pass custom OpenMP flags, you can do so via environment variables, for example:

```bash
export CXXFLAGS="-O3 -fopenmp -march=native"
export LDFLAGS="-fopenmp"
pip install -e .
```

For development installation with documentation dependencies:

```bash
pip install -e ".[docs]"
```

## Basic Usage
We follow the same parameter naming conventions presented in the original IRAF code.

- `objlim`: The contrast limit between CR and underlying objects
- `sigfrac`: The fractional detection limit for neighboring pixels
- `sigclip`: The detection limit for cosmic rays

```python
from dfcosmic import lacosmic

clean_image, crmask = lacosmic(
    image=original_data,
    objlim=2,
    sigfrac=1,
    sigclip=6,
    gain=1,
    readnoise=10,
    niter=1,
    device="cuda",
)
```

If you are unsure of either the gain or the readnoise you can leave them blank or set to 0. If so, then the gain will be estimated at each iteration.

## Simple Example

![Example](demos/example.png)

## Timing Comparisons
We compare our pytorch implementation running on either a CPU (torch), CPU (torch & c++) or GPU with two popular cosmic ray removal codes: [lacosmic](https://github.com/larrybradley/lacosmic) and [astroscrappy](https://github.com/astropy/astroscrappy).

In order to run this timing comparison, we use the synthetic data described (and created) in the [astroscrappy testing suite](https://github.com/astropy/astroscrappy/blob/main/astroscrappy/tests/fake_data.py). The full notebook can be found in [demos/Comparison.ipynb](./demos/Comparison.ipynb).

![Timing Comparisons](demos/comparison.png)


## Running Tests
The unit tests can be run using the following command:

```bash
pytest
```

The default settings are in the `pytest.ini` file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

## License
The License for all past and present versions is the GPL-3.0.

## AI Disclosure
Claude code was used to help with the unit tests *only*. ChatGPT was used to create *only* the c++ code for the median filter and dilation - these were thoroughly reviewed by the authors.
