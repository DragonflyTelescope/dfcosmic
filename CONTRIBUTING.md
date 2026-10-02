# Contributing to dfcosmic

Contributions are welcome, whether they are bug reports, documentation fixes or new features. Please open an [issue](https://github.com/DragonflyTelescope/dfcosmic/issues) to report a problem or to discuss a larger change before starting on it.

## Setting up a development environment

You need Python 3.10 or newer.

```bash
git clone https://github.com/DragonflyTelescope/dfcosmic.git
cd dfcosmic
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest pytest-cov ruff pre-commit astropy
```

`astropy` is only used by the tests, to read the *HST* demo frame for the regression test against the IRAF mask.

By default `pip` installs a PyTorch build with GPU support, which is a large download. If you only need the CPU, install PyTorch first with

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

### The optional C++ median filter

`pip install -e .` gives you a pure-Python install. To also build the C++ median filter you need a C++ compiler with OpenMP support, and PyTorch has to be installed before building:

```bash
pip install torch "setuptools>=77"
DFCOSMIC_BUILD_CPP=1 pip install --no-build-isolation -e .
```

Rebuild it with the same command after changing `csrc/median_filter.cpp` or upgrading PyTorch.

## Running the tests

```bash
pytest
```

The settings are in the `[tool.pytest.ini_options]` section of `pyproject.toml`, and a coverage report is printed at the end. The tests for the C++ median filter are skipped if the extension has not been built, and the regression test against the IRAF mask is skipped if `astropy` is not installed.

Please add tests for new behaviour and for bug fixes.

## Code style

The code in `src/` is linted and formatted with [ruff](https://docs.astral.sh/ruff/). The easiest way to run it is through [pre-commit](https://pre-commit.com/), which also checks that no secrets are committed:

```bash
pre-commit install          # once, to run the checks on every commit
pre-commit run --all-files  # to run them by hand
```

The same ruff checks run in CI.

## Building the documentation

The documentation is built with Sphinx and needs [pandoc](https://pandoc.org/installing.html) for the notebooks.

```bash
pip install -e ".[docs]"
cd docs
make html
```

The result is in `docs/build/html`. The notebooks shown in the documentation are the ones in `demos/`, with the outputs stored in them: they are copied into `docs/source/demos` when the documentation is built and are not re-executed. After changing a notebook, run it and save it with its outputs.

## Re-running the timing comparison

The timing figure and tables in the README, the documentation and the paper come from `demos/benchmark_results.json`, which is written by `demos/benchmark.py`. The script needs the C++ median filter (see above), `astroscrappy`, `lacosmic` and `matplotlib` (`pip install -e ".[notebooks]"`), and a CUDA build of PyTorch for the GPU timing.

```bash
python demos/benchmark.py run --quick                  # a minute, to check the set-up
python demos/benchmark.py run                          # niter=1, 1 to 16 threads
python demos/benchmark.py run --niter 4 --threads 1 2 --rounds 3 --repeats 2
python demos/benchmark.py run --image hst --rounds 3 --repeats 2   # a crowded image
python demos/benchmark.py report                       # tables and figures
```

The three full runs take about 80, 50 and 40 minutes and should have the machine to itself; an interrupted run continues where it stopped. Afterwards re-run `demos/Comparison.ipynb` and update the numbers quoted in `README.md`, `docs/source/index.rst` and `paper.md` from the output of `report`.

## Submitting a pull request

1. Fork the repository and create a branch for your change (`git checkout -b my-change`).
2. Make your change, with tests, and add a line to the "Unreleased" section of `CHANGELOG.md`.
3. Check that `pytest` and `pre-commit run --all-files` pass.
4. Push the branch to your fork and open a pull request against `master`.

## Making a release

1. Update the version in `pyproject.toml` and `CITATION.cff`, and move the "Unreleased" entries in `CHANGELOG.md` under the new version.
2. Publish a GitHub release with the tag `vX.Y.Z` on `master`. The "Build and publish" workflow checks that the tag matches the package version and uploads the release to PyPI.

By contributing you agree to follow the [code of conduct](https://github.com/DragonflyTelescope/dfcosmic/blob/master/CODE_OF_CONDUCT.md).
