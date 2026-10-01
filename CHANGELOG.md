# Changelog

All notable changes to dfcosmic are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Changed

- If no gain is given, it is now estimated at every iteration, as in the original IRAF script and as documented. It used to be estimated in the first iteration only. Results with `gain=0` and `niter > 1` change slightly; results with an explicit gain are unchanged.
- The error raised when the gain cannot be estimated now mentions that the gain has to be given for background-subtracted images.

### Fixed

- The cleaned image could contain placeholder values (up to 4.4e19) at flagged pixels where more than half of the 5x5 window was flagged. These pixels are now set to the median of their unflagged neighbours. The mask is unchanged.
- A NaN or infinite pixel silently gave an empty mask when the gain was estimated, and NaN at every repaired pixel when the gain was given. Non-finite pixels are now ignored and returned unchanged.
- Big-endian arrays (e.g. from `astropy.io.fits`) and arrays with negative strides raised an error.
- Input that is not 2D now raises a clear `ValueError`.

### Documentation

- Added a statement of need, runnable examples, an installation page, a contributor guide and this changelog.
- The documentation now shows the version of the installed package.

## 0.1.0 - 2026-10-01

### Changed

- The C++ median filter is now an opt-in source build (`pip install --no-build-isolation`), and the wheel on PyPI is pure Python. Previously the documentation said that it was built by `pip install`, which was not the case.
- `use_cpp` now defaults to `None`: the C++ median filter is used if it is available. `use_cpp=True` warns once per session if it is not.
- The C++ extension is installed inside the package, as `dfcosmic._median_filter_cpp`.
- `matplotlib` and `cmcrameri` moved to the `notebooks` extra. `threadpoolctl` is now a dependency. The minimum versions are `torch>=2.1.0` and `numpy>=1.24.0`.
- The neighbour growing step uses a convolution, and the main steps are chunked to reduce memory use.

### Added

- `dfcosmic.__version__`, `CITATION.cff`, and license metadata.
- Environment variables `DFCOSMIC_MAX_MEMORY_MB` and `DFCOSMIC_CONVOLVE_DIRECT_MAX_NUMEL` for memory-constrained machines.

### Removed

- The unused C++ dilation extension.

## 0.0.1 - 2026-01-26

- Initial release.
