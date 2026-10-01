from pathlib import Path

import numpy as np
import pytest
import torch

from dfcosmic import lacosmic
from dfcosmic.utils import fill_from_unflagged_neighbors

DEMOS = Path(__file__).resolve().parents[1] / "demos"

# Parameters of demos/HST.ipynb, which were also used for the IRAF reference mask
HST_PARAMS = dict(sigclip=4.5, sigfrac=0.3, objlim=4, readnoise=5, gain=7)


@pytest.fixture(scope="module")
def hst():
    fits = pytest.importorskip("astropy.io.fits")
    data = fits.getdata(DEMOS / "hst_im_ext3.fits")
    iraf_mask = fits.getdata(DEMOS / "hst_im_ext3_iraf_mask.fits") > 0
    return data, iraf_mask


def _blob_frame() -> np.ndarray:
    """Sky frame with large cosmic ray hits, where most of a 5x5 window is flagged."""
    rng = np.random.default_rng(1)
    img = rng.normal(200, 5, (120, 120)).astype(np.float32)
    blobs = np.zeros(img.shape, dtype=bool)
    blobs[20:27, 20:27] = True
    blobs[60:65, 70:76] = True
    blobs[0:6, 0:6] = True  # at the corner of the image
    blobs[100:104, 40:60] = True
    img[blobs] += rng.uniform(2000, 20000, blobs.sum()).astype(np.float32)
    return img


def _mostly_flagged(mask: np.ndarray) -> np.ndarray:
    """Flagged pixels with more than half of their 5x5 window flagged."""
    padded = np.pad(mask, 2, mode="edge").astype(int)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (5, 5))
    return mask & (windows.sum(axis=(2, 3)) >= 13)


class TestRepairedValues:
    """The cleaned image must never contain placeholder values."""

    @pytest.mark.parametrize("niter", [1, 2, 4])
    @pytest.mark.parametrize("use_cpp", [None, False])
    def test_blobs_within_input_range(self, niter, use_cpp):
        img = _blob_frame()
        clean, mask = lacosmic(
            img, gain=1, readnoise=5, niter=niter, use_cpp=use_cpp, sigfrac=0.3
        )
        # Make sure this exercises the case where most of the window is flagged
        assert _mostly_flagged(mask).sum() > 10
        assert clean.min() >= img.min()
        assert clean.max() <= img.max()
        # Unflagged pixels are untouched
        np.testing.assert_array_equal(clean[~mask], img[~mask])
        # Repaired pixels take the value of an unflagged input pixel
        assert np.isin(clean[mask], img[~mask]).all()

    def test_negative_image(self):
        """The placeholder must sort above the data also when all data are negative."""
        img = _blob_frame() - 50000
        clean, mask = lacosmic(img, gain=1, readnoise=5, sigfrac=0.3)
        assert _mostly_flagged(mask).sum() > 10
        assert clean.min() >= img.min()
        assert clean.max() <= img.max()

    @pytest.mark.parametrize("niter", [1, 2, 4])
    def test_hst_within_input_range(self, hst, niter):
        data, _ = hst
        clean, mask = lacosmic(data, niter=niter, **HST_PARAMS)
        assert _mostly_flagged(mask).sum() > 10
        assert clean.min() >= data.min()
        assert clean.max() <= data.max()
        np.testing.assert_array_equal(clean[~mask], data[~mask])
        assert np.isin(clean[mask], data[~mask]).all()


class TestIrafRegression:
    """Regression test against the mask produced by the original IRAF script."""

    @pytest.mark.parametrize("use_cpp", [None, False])
    def test_mask_matches_iraf(self, hst, use_cpp):
        data, iraf_mask = hst
        _, mask = lacosmic(data, niter=4, use_cpp=use_cpp, **HST_PARAMS)
        iou = (mask & iraf_mask).sum() / (mask | iraf_mask).sum()
        assert iou > 0.996
        assert (mask & ~iraf_mask).sum() < 50
        assert (~mask & iraf_mask).sum() < 50


class TestFillFromUnflaggedNeighbors:
    def test_median_of_unflagged(self):
        image = torch.arange(25, dtype=torch.float32).reshape(5, 5)
        flagged = torch.zeros((5, 5), dtype=torch.bool)
        flagged[2, :] = True
        flagged[1, 1:4] = True
        fill = torch.zeros((5, 5), dtype=torch.bool)
        fill[2, 2] = True
        expected = image[~flagged].median()
        fill_from_unflagged_neighbors(image, fill, flagged)
        assert image[2, 2] == expected

    def test_window_grows_until_unflagged_pixel(self):
        image = torch.full((11, 11), 1e9)
        flagged = torch.ones((11, 11), dtype=torch.bool)
        image[0, 0] = 42.0
        flagged[0, 0] = False
        fill = torch.zeros((11, 11), dtype=torch.bool)
        fill[10, 10] = True
        fill_from_unflagged_neighbors(image, fill, flagged)
        assert image[10, 10] == 42.0

    def test_only_fill_mask_is_modified(self):
        image = torch.rand(8, 8)
        original = image.clone()
        flagged = torch.zeros((8, 8), dtype=torch.bool)
        flagged[3:6, 3:6] = True
        fill = torch.zeros((8, 8), dtype=torch.bool)
        fill[4, 4] = True
        fill_from_unflagged_neighbors(image, fill, flagged)
        assert torch.equal(image[~fill], original[~fill])
        assert (original[~flagged] == image[4, 4]).any()

    def test_empty_fill_mask(self):
        image = torch.rand(8, 8)
        original = image.clone()
        empty = torch.zeros((8, 8), dtype=torch.bool)
        fill_from_unflagged_neighbors(image, empty, empty)
        assert torch.equal(image, original)
