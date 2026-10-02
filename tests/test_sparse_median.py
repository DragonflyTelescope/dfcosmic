"""
Tests for the median filters that are only evaluated at the pixels where their result
is used. They must give exactly the same result as filtering the whole image.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import dfcosmic.core as core
import dfcosmic.utils as utils
from dfcosmic import lacosmic
from dfcosmic.utils import (
    median_filter_at,
    median_filter_torch,
    median_of_median_at,
    use_sparse_median,
)

DEMOS = Path(__file__).resolve().parents[1] / "demos"
FRACTION = "DFCOSMIC_SPARSE_MEDIAN_MAX_FRACTION"
# With "0" every median filter runs on the full image and with "1" none of the three
# that are only read at a few pixels does. In between, the 3x3 median still runs on
# the full image when the candidates are too many for it to be evaluated around them.
NEVER, ALWAYS = "0", "1"

PARAMS = dict(sigclip=4.5, sigfrac=0.3, objlim=4, readnoise=5)


def _sky_frame(n_cosmic_rays: int = 150, seed: int = 42) -> np.ndarray:
    """300x300 sky frame (200 e- background, read noise 5) with single-pixel hits."""
    rng = np.random.default_rng(seed)
    img = rng.poisson(200, (300, 300)).astype(np.float32)
    img += rng.normal(0, 5, img.shape).astype(np.float32)
    ys, xs = rng.integers(0, 300, n_cosmic_rays), rng.integers(0, 300, n_cosmic_rays)
    img[ys, xs] += rng.uniform(500, 2000, n_cosmic_rays).astype(np.float32)
    # Hits in the corners and on the edges, where the windows are clipped
    for y, x in [(0, 0), (0, 299), (299, 0), (299, 299), (0, 150), (150, 299)]:
        img[y, x] += 3000
    return img


def _blob_frame() -> np.ndarray:
    """Sky frame with large hits, where more than half of a 5x5 window is flagged."""
    rng = np.random.default_rng(1)
    img = rng.normal(200, 5, (120, 120)).astype(np.float32)
    blobs = np.zeros(img.shape, dtype=bool)
    blobs[20:27, 20:27] = True
    blobs[60:65, 70:76] = True
    blobs[0:6, 0:6] = True
    blobs[100:104, 40:60] = True
    img[blobs] += rng.uniform(2000, 20000, blobs.sum()).astype(np.float32)
    return img


def _every_way(monkeypatch, image, middle="0.02", **kwargs):
    """Run lacosmic with the medians on the full image, and only where they are used."""
    results = []
    for fraction in (NEVER, middle, ALWAYS):
        monkeypatch.setenv(FRACTION, fraction)
        results.append(lacosmic(image, **kwargs))
    return results


def _assert_identical(results):
    reference = results[0]
    for result in results[1:]:
        np.testing.assert_array_equal(result[1], reference[1])
        np.testing.assert_array_equal(result[0], reference[0])


class TestMedianFilterAt:
    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    def test_matches_full_filter_at_every_pixel(self, kernel_size):
        image = torch.from_numpy(_sky_frame()[:40, :50].copy())
        ys, xs = torch.nonzero(torch.ones_like(image, dtype=torch.bool), as_tuple=True)
        result = median_filter_at(image, ys, xs, kernel_size=kernel_size)
        expected = median_filter_torch(image, kernel_size=kernel_size)
        assert torch.equal(result, expected.flatten())

    def test_selected_pixels_including_corners(self):
        image = torch.rand(30, 20)
        ys = torch.tensor([0, 0, 29, 29, 15, 7])
        xs = torch.tensor([0, 19, 0, 19, 10, 0])
        expected = median_filter_torch(image, kernel_size=5)[ys, xs]
        assert torch.equal(median_filter_at(image, ys, xs, kernel_size=5), expected)

    def test_replaced_pixels_enter_the_median_as_value(self):
        image = torch.rand(30, 20)
        replace = torch.rand(30, 20) > 0.7
        ys, xs = torch.nonzero(replace, as_tuple=True)
        filled = image.clone()
        filled[replace] = 1e9
        expected = median_filter_torch(filled, kernel_size=5)[ys, xs]
        result = median_filter_at(
            image, ys, xs, kernel_size=5, replace=replace, value=1e9
        )
        assert torch.equal(result, expected)

    def test_does_not_modify_the_image(self):
        image = torch.rand(30, 20)
        original = image.clone()
        replace = torch.rand(30, 20) > 0.5
        ys, xs = torch.nonzero(replace, as_tuple=True)
        median_filter_at(image, ys, xs, kernel_size=5, replace=replace, value=1e9)
        assert torch.equal(image, original)

    def test_result_independent_of_chunk_size(self, monkeypatch):
        image = torch.rand(40, 40)
        ys, xs = torch.nonzero(image > 0.3, as_tuple=True)
        expected = median_filter_at(image, ys, xs, kernel_size=7)
        monkeypatch.setattr(utils, "_SPARSE_MEDIAN_CHUNK_VALUES", 49 * 3)
        assert torch.equal(median_filter_at(image, ys, xs, kernel_size=7), expected)

    def test_no_pixels(self):
        image = torch.rand(10, 10)
        none = torch.empty(0, dtype=torch.long)
        result = median_filter_at(image, none, none, kernel_size=5)
        assert result.shape == (0,) and result.dtype == image.dtype


class TestMedianOfMedianAt:
    def _expected(self, image, ys, xs):
        inner = median_filter_torch(image, kernel_size=3)
        outer = median_filter_torch(inner, kernel_size=7)
        return inner[ys, xs], outer[ys, xs]

    def test_matches_full_filters_at_every_pixel(self):
        image = torch.from_numpy(_sky_frame()[:40, :50].copy())
        ys, xs = torch.nonzero(torch.ones_like(image, dtype=torch.bool), as_tuple=True)
        inner, outer = median_of_median_at(image, ys, xs, 3, 7)
        expected_inner, expected_outer = self._expected(image, ys, xs)
        assert torch.equal(inner, expected_inner)
        assert torch.equal(outer, expected_outer)

    def test_result_independent_of_chunk_size(self, monkeypatch):
        image = torch.rand(30, 30)
        ys, xs = torch.nonzero(image > 0.6, as_tuple=True)
        expected_inner, expected_outer = self._expected(image, ys, xs)
        monkeypatch.setattr(utils, "_SPARSE_MEDIAN_CHUNK_VALUES", 100)
        inner, outer = median_of_median_at(image, ys, xs, 3, 7)
        assert torch.equal(inner, expected_inner)
        assert torch.equal(outer, expected_outer)

    def test_no_pixels(self):
        none = torch.empty(0, dtype=torch.long)
        inner, outer = median_of_median_at(torch.rand(10, 10), none, none)
        assert inner.shape == outer.shape == (0,)


class TestThreshold:
    def test_default_fraction(self, monkeypatch):
        monkeypatch.delenv(FRACTION, raising=False)
        image = torch.zeros(100, 100)
        limit = int(utils._DEFAULT_SPARSE_MEDIAN_MAX_FRACTION * image.numel())
        assert use_sparse_median(limit, image)
        assert not use_sparse_median(limit + 1, image)

    @pytest.mark.parametrize(
        "value, n_pixels, expected",
        [("0", 1, False), ("0", 0, True), ("1", 10_000, True), ("0.5", 5001, False)],
    )
    def test_environment_variable(self, monkeypatch, value, n_pixels, expected):
        monkeypatch.setenv(FRACTION, value)
        assert use_sparse_median(n_pixels, torch.zeros(100, 100)) is expected

    @pytest.mark.parametrize("value", ["invalid", ""])
    def test_invalid_environment_variable_uses_default(self, monkeypatch, value):
        monkeypatch.setenv(FRACTION, value)
        fraction = utils._sparse_median_max_fraction()
        assert fraction == utils._DEFAULT_SPARSE_MEDIAN_MAX_FRACTION

    def test_environment_variable_is_clipped(self, monkeypatch):
        monkeypatch.setenv(FRACTION, "7")
        assert utils._sparse_median_max_fraction() == 1.0
        monkeypatch.setenv(FRACTION, "-1")
        assert utils._sparse_median_max_fraction() == 0.0


class TestSameResultAsFullFilters:
    """lacosmic must return exactly the same mask and image either way."""

    @pytest.mark.parametrize("niter", [1, 2, 4])
    @pytest.mark.parametrize("use_cpp", [None, False])
    def test_sky_frame(self, monkeypatch, niter, use_cpp):
        image = _sky_frame()
        results = _every_way(
            monkeypatch, image, gain=1, niter=niter, use_cpp=use_cpp, **PARAMS
        )
        assert results[0][1].sum() > 100
        # The hits in the corners are found, so clipped windows are exercised
        assert results[0][1][0, 0] and results[0][1][299, 299]
        _assert_identical(results)

    @pytest.mark.parametrize("niter", [1, 4])
    def test_estimated_gain(self, monkeypatch, niter):
        results = _every_way(monkeypatch, _sky_frame(), niter=niter, **PARAMS)
        assert results[0][1].sum() > 100
        _assert_identical(results)

    @pytest.mark.parametrize("niter", [1, 2, 4])
    def test_large_hits(self, monkeypatch, niter):
        """Windows that are mostly flagged, which are filled in at the end."""
        results = _every_way(
            monkeypatch, _blob_frame(), gain=1, readnoise=5, niter=niter, sigfrac=0.3
        )
        assert results[0][1][23, 23]
        _assert_identical(results)

    def test_non_finite_pixels(self, monkeypatch):
        image = _sky_frame()
        image[[10, 150, 299], [10, 150, 0]] = [np.nan, np.inf, -np.inf]
        results = _every_way(monkeypatch, image, gain=1, niter=2, **PARAMS)
        assert results[0][1].sum() > 100
        _assert_identical(results)

    def test_no_cosmic_rays(self, monkeypatch):
        image = np.full((64, 64), 100.0, dtype=np.float32)
        results = _every_way(monkeypatch, image, gain=1, readnoise=5)
        assert not results[0][1].any()
        _assert_identical(results)

    def test_many_cosmic_rays(self, monkeypatch):
        """A crowded frame, far above the default threshold."""
        image = _sky_frame(n_cosmic_rays=9000)
        results = _every_way(monkeypatch, image, gain=1, niter=2, **PARAMS)
        assert results[0][1].mean() > 0.05
        _assert_identical(results)

    @pytest.mark.parametrize("use_cpp", [None, False])
    def test_hst_frame(self, monkeypatch, use_cpp):
        fits = pytest.importorskip("astropy.io.fits")
        data = fits.getdata(DEMOS / "hst_im_ext3.fits")
        # About 2% of the pixels are candidates here
        results = _every_way(
            monkeypatch, data, middle="0.2", gain=7, niter=4, use_cpp=use_cpp, **PARAMS
        )
        assert results[0][1].sum() > 18000
        _assert_identical(results)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
    def test_gpu(self, monkeypatch):
        results = _every_way(
            monkeypatch, _sky_frame(), gain=1, niter=2, device="cuda", **PARAMS
        )
        assert results[0][1].sum() > 100
        _assert_identical(results)


class TestFullFrameMediansAreSkipped:
    """The point of the exercise: fewer median filters over the whole image."""

    @pytest.fixture
    def kernel_sizes(self, monkeypatch):
        sizes = []
        median_filter = core.median_filter_torch

        def recording_median_filter(image, kernel_size=3, **kwargs):
            sizes.append(kernel_size)
            return median_filter(image, kernel_size=kernel_size, **kwargs)

        monkeypatch.setattr(core, "median_filter_torch", recording_median_filter)
        return sizes

    def test_full_filters(self, monkeypatch, kernel_sizes):
        monkeypatch.setenv(FRACTION, NEVER)
        lacosmic(_sky_frame(), gain=1, niter=1, use_cpp=False, **PARAMS)
        assert kernel_sizes == [5, 5, 3, 7, 5]

    def test_only_where_used(self, monkeypatch, kernel_sizes):
        monkeypatch.setenv(FRACTION, ALWAYS)
        lacosmic(_sky_frame(), gain=1, niter=1, use_cpp=False, **PARAMS)
        assert kernel_sizes == [5, 5]

    def test_3x3_on_the_full_image_for_more_candidates(self, monkeypatch, kernel_sizes):
        # The candidates are below 2% of the pixels, but their 7x7 windows are not
        monkeypatch.setenv(FRACTION, "0.02")
        lacosmic(_sky_frame(), gain=1, niter=1, use_cpp=False, **PARAMS)
        assert kernel_sizes == [5, 5, 3]

    def test_default_on_a_frame_with_few_cosmic_rays(self, monkeypatch, kernel_sizes):
        monkeypatch.delenv(FRACTION, raising=False)
        image = _sky_frame(n_cosmic_rays=10)
        _, mask = lacosmic(image, gain=1, niter=1, use_cpp=False, **PARAMS)
        assert mask.sum() >= 10
        assert kernel_sizes == [5, 5]

    def test_default_on_a_crowded_frame(self, monkeypatch, kernel_sizes):
        monkeypatch.delenv(FRACTION, raising=False)
        # With a threshold this low, a large part of the image is a candidate
        params = {**PARAMS, "sigclip": 0.01}
        _, mask = lacosmic(_sky_frame(), gain=1, niter=1, use_cpp=False, **params)
        assert mask.mean() > 0.1
        assert kernel_sizes == [5, 5, 3, 7, 5]

    def test_no_candidates_needs_no_fine_structure(self, monkeypatch, kernel_sizes):
        monkeypatch.delenv(FRACTION, raising=False)
        image = np.full((64, 64), 100.0, dtype=np.float32)
        lacosmic(image, gain=1, readnoise=5, use_cpp=False)
        assert kernel_sizes == [5, 5]
