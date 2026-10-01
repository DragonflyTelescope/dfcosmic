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

The result is in `docs/build/html`. The notebooks shown in the documentation are the copies in `docs/source/demos`.

## Submitting a pull request

1. Fork the repository and create a branch for your change (`git checkout -b my-change`).
2. Make your change, with tests, and add a line to the "Unreleased" section of `CHANGELOG.md`.
3. Check that `pytest` and `pre-commit run --all-files` pass.
4. Push the branch to your fork and open a pull request against `master`.

## Making a release

1. Update the version in `pyproject.toml` and `CITATION.cff`, and move the "Unreleased" entries in `CHANGELOG.md` under the new version.
2. Publish a GitHub release with the tag `vX.Y.Z` on `master`. The "Build and publish" workflow checks that the tag matches the package version and uploads the release to PyPI.

By contributing you agree to follow the [code of conduct](https://github.com/DragonflyTelescope/dfcosmic/blob/master/CODE_OF_CONDUCT.md).
