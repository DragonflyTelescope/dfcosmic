import warnings

import numpy as np
import pytest
import torch

from dfcosmic import lacosmic

PARAMS = dict(sigclip=4.5, sigfrac=0.3, objlim=4, readnoise=5)


@pytest.fixture
def frame() -> np.ndarray:
    """300x300 sky frame (200 e- background, read noise 5) with 150 cosmic rays."""
    rng = np.random.default_rng(42)
    img = rng.poisson(200, (300, 300)).astype(np.float32)
    img += rng.normal(0, 5, img.shape).astype(np.float32)
    ys, xs = rng.integers(5, 295, 150), rng.integers(5, 295, 150)
    img[ys, xs] += rng.uniform(500, 2000, 150).astype(np.float32)
    return img


@pytest.fixture
def bad_pixels() -> tuple[np.ndarray, np.ndarray]:
    return np.array([0, 150, 200, 299]), np.array([0, 150, 100, 299])


class TestNonFinitePixels:
    """Tests for images containing NaN or inf pixels."""

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("gain", [0.0, 1.0])
    @pytest.mark.parametrize("use_cpp", [None, False])
    def test_non_finite_pixels_are_ignored(
        self, frame, bad_pixels, value, gain, use_cpp
    ):
        """A few bad pixels must not change the detection elsewhere."""
        _, mask_ref = lacosmic(frame, gain=gain, use_cpp=use_cpp, **PARAMS)
        assert mask_ref.sum() > 100

        bad = frame.copy()
        bad[bad_pixels] = value
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            clean, mask = lacosmic(bad, gain=gain, use_cpp=use_cpp, **PARAMS)

        # Bad pixels are never flagged and come back unchanged
        assert not mask[bad_pixels].any()
        np.testing.assert_array_equal(clean[bad_pixels], bad[bad_pixels])
        # Nothing else in the cleaned image is non-finite
        assert (~np.isfinite(clean)).sum() == len(bad_pixels[0])

        # Detection is unchanged away from the bad pixels
        far = np.ones(frame.shape, dtype=bool)
        for y, x in zip(*bad_pixels):
            far[max(0, y - 5) : y + 6, max(0, x - 5) : x + 6] = False
        assert (mask[far] != mask_ref[far]).mean() < 1e-3
        assert mask.sum() > 0.9 * mask_ref[far].sum()

    def test_single_nan_with_estimated_gain(self, frame):
        """Regression test: one NaN pixel used to give an empty mask."""
        _, mask_ref = lacosmic(frame, **PARAMS)
        bad = frame.copy()
        bad[0, 0] = np.nan
        _, mask = lacosmic(bad, **PARAMS)
        np.testing.assert_array_equal(mask, mask_ref)

    def test_input_with_nan_not_modified(self, frame):
        bad = frame.copy()
        bad[10, 10] = np.nan
        original = bad.copy()
        lacosmic(bad, gain=1, **PARAMS)
        np.testing.assert_array_equal(bad, original)

    def test_torch_input_with_nan(self, frame):
        bad = torch.from_numpy(frame.copy())
        bad[10, 10] = torch.nan
        clean, mask = lacosmic(bad, **PARAMS)
        assert mask.sum() > 100
        assert np.isnan(clean[10, 10])
        assert np.isnan(clean).sum() == 1

    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_all_non_finite_raises(self, value):
        image = np.full((20, 20), value, dtype=np.float32)
        with pytest.raises(ValueError, match="no finite pixels"):
            lacosmic(image, gain=1, readnoise=5)


class TestArrayConversion:
    """Tests for numpy inputs that torch.from_numpy does not accept directly."""

    @pytest.mark.parametrize(
        "convert",
        [
            pytest.param(lambda a: a.astype(">f4"), id="big-endian-f4"),
            pytest.param(lambda a: a.astype(">f8"), id="big-endian-f8"),
            pytest.param(lambda a: a.astype(np.float64), id="float64"),
            pytest.param(lambda a: np.asfortranarray(a), id="fortran-order"),
            pytest.param(lambda a: a[::-1, ::-1][::-1, ::-1], id="negative-stride"),
            pytest.param(lambda a: np.repeat(a, 2, axis=1)[:, ::2], id="strided"),
        ],
    )
    def test_matches_native_float32(self, frame, convert):
        clean_ref, mask_ref = lacosmic(frame, gain=1, **PARAMS)
        arr = convert(frame)
        np.testing.assert_array_equal(np.asarray(arr, dtype=np.float32), frame)
        clean, mask = lacosmic(arr, gain=1, **PARAMS)
        np.testing.assert_array_equal(mask, mask_ref)
        np.testing.assert_array_equal(clean, clean_ref)

    def test_negative_stride_view(self, frame):
        clean_ref, mask_ref = lacosmic(frame, gain=1, **PARAMS)
        clean, mask = lacosmic(frame[::-1], gain=1, **PARAMS)
        assert mask.sum() == mask_ref.sum()
        np.testing.assert_array_equal(mask, mask_ref[::-1])

    def test_integer_input(self, frame):
        image = np.round(frame).astype(np.int32)
        clean, mask = lacosmic(image, gain=1, **PARAMS)
        assert clean.dtype == np.float32
        assert mask.sum() > 100

    def test_float64_returns_float32(self, frame):
        clean, mask = lacosmic(frame.astype(np.float64), gain=1, **PARAMS)
        assert clean.dtype == np.float32
        assert mask.dtype == np.bool_

    def test_list_input(self):
        image = np.random.default_rng(0).normal(100, 5, (20, 20)).tolist()
        clean, mask = lacosmic(image, gain=1, readnoise=5)
        assert clean.shape == (20, 20)


class TestShapeCheck:
    """Tests for the 2D input requirement."""

    @pytest.mark.parametrize("shape", [(50,), (3, 50, 50), (1, 1, 50, 50), ()])
    @pytest.mark.parametrize("as_tensor", [False, True])
    def test_non_2d_raises(self, shape, as_tensor):
        image = np.ones(shape, dtype=np.float32)
        if as_tensor:
            image = torch.from_numpy(image)
        with pytest.raises(ValueError, match="must be a 2D array"):
            lacosmic(image, gain=1, readnoise=5)
