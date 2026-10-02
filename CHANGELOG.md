# Changelog

All notable changes to dfcosmic are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Performance

- Three of the five median filters of an iteration are only read at a few pixels: the two fine-structure medians at the candidate pixels, and the repair median at the flagged pixels. They are now evaluated only there, which gives exactly the same mask and cleaned image. On a 4000 x 6500 frame with few cosmic rays, `lacosmic` is 1.7 to 2.4 times as fast on the CPU, depending on the number of threads, and 2.9 times as fast on the GPU. If those pixels make up more than 10% of the image, the whole image is filtered as before; the fraction can be changed with the environment variable `DFCOSMIC_SPARSE_MEDIAN_MAX_FRACTION`. Suggested by Robert Vetter ([@robert-vetter](https://github.com/robert-vetter)) in his pyOpenSci review.

### Changed

- If no gain is given, it is now estimated at every iteration, as in the original IRAF script and as documented. It used to be estimated in the first iteration only. Results with `gain=0` and `niter > 1` change slightly; results with an explicit gain are unchanged.
- The error raised when the gain cannot be estimated now mentions that the gain has to be given for background-subtracted images.

### Fixed

- `cpu_threads` no longer changes settings for the whole process. The thread count of torch is restored after the call, the `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS` environment variables are no longer written, and `torch.backends.mkldnn.enabled` is no longer switched on.
- `cpu_threads` smaller than 1 now raises a `ValueError`.
- The cleaned image could contain placeholder values (up to 4.4e19) at flagged pixels where more than half of the 5x5 window was flagged. These pixels are now set to the median of their unflagged neighbours. The mask is unchanged.
- A NaN or infinite pixel silently gave an empty mask when the gain was estimated, and NaN at every repaired pixel when the gain was given. Non-finite pixels are now ignored and returned unchanged.
- Big-endian arrays (e.g. from `astropy.io.fits`) and arrays with negative strides raised an error.
- Input that is not 2D now raises a clear `ValueError`.

### Timing comparison

- The timing comparison with astroscrappy and lacosmic was redone so that it is like-for-like: every code runs the same number of iterations with the same parameters, in its own process, with a warm-up call and repeated timed calls. It is produced by the new `demos/benchmark.py`, which replaces the broken `demos/comparison.py`, and its results are stored with the hardware and package versions in `demos/benchmark_results.json`. The comparison now also covers a crowded image and `niter=4`, and `demos/benchmark_results_v0.1.0.json` holds the same measurement for dfcosmic 0.1.0.
- `demos/HST.ipynb` passed the two lacosmic thresholds the wrong way round. This is corrected, and the notebook now reports how closely every mask agrees with the IRAF mask.
- `demos/QuickExample.ipynb` no longer overwrites `demos/example_hst.png`, the six-panel figure of the paper; its own figure is now `demos/quick_example.png`.
- The notebooks shown in the documentation are now copied from `demos/` when the documentation is built, so that there is a single copy of each.

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
