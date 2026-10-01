import os
import warnings

import numpy as np
import pytest
import torch

import dfcosmic
import dfcosmic.utils as utils
from dfcosmic import lacosmic
from dfcosmic.utils import median_filter_cpp_torch, median_filter_torch

# Set in the CI job that builds the extension, so a silently missing build fails there.
_REQUIRE_CPP = os.environ.get("DFCOSMIC_REQUIRE_CPP", "").lower() in {"1", "true", "yes"}

requires_cpp = pytest.mark.skipif(
    not utils.cpp_median_available() and not _REQUIRE_CPP,
    reason="C++ median filter extension is not built",
)


def _image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(100, 5, (64, 64)).astype(np.float32)


def test_version():
    assert isinstance(dfcosmic.__version__, str)
    assert dfcosmic.__version__ != "0+unknown"


class TestCppFallbackWarning:
    """Tests for the behaviour when the C++ extension is not available."""

    @pytest.fixture
    def cpp_unavailable(self, monkeypatch):
        monkeypatch.setattr(utils, "_CPP_MEDIAN_AVAILABLE", False)
        monkeypatch.setattr(utils, "_CPP_MEDIAN_UNAVAILABLE_REASON", "it was not built.")
        monkeypatch.setattr(utils, "_CPP_FALLBACK_WARNED", False)

    def test_use_cpp_true_warns(self, cpp_unavailable):
        with pytest.warns(RuntimeWarning, match="C\\+\\+ median filter"):
            lacosmic(_image(), gain=1, readnoise=5, use_cpp=True)

    def test_use_cpp_true_warns_only_once(self, cpp_unavailable):
        with pytest.warns(RuntimeWarning):
            lacosmic(_image(), gain=1, readnoise=5, use_cpp=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lacosmic(_image(), gain=1, readnoise=5, use_cpp=True)

    @pytest.mark.parametrize("use_cpp", [None, False])
    def test_default_and_false_do_not_warn(self, cpp_unavailable, use_cpp):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lacosmic(_image(), gain=1, readnoise=5, use_cpp=use_cpp)

    def test_default_call_does_not_warn(self, cpp_unavailable):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lacosmic(_image(), gain=1, readnoise=5)

    def test_fallback_matches_torch_path(self, cpp_unavailable):
        image = _image()
        with pytest.warns(RuntimeWarning):
            clean_cpp, mask_cpp = lacosmic(image, gain=1, readnoise=5, use_cpp=True)
        clean_torch, mask_torch = lacosmic(image, gain=1, readnoise=5, use_cpp=False)
        np.testing.assert_array_equal(clean_cpp, clean_torch)
        np.testing.assert_array_equal(mask_cpp, mask_torch)

    def test_median_filter_cpp_torch_raises(self, cpp_unavailable):
        with pytest.raises(RuntimeError, match="not available"):
            median_filter_cpp_torch(torch.rand(8, 8))


@requires_cpp
class TestCppMedianFilter:
    """Tests for the optional C++ median filter extension."""

    def test_extension_available(self):
        assert utils.cpp_median_available(), utils._CPP_MEDIAN_UNAVAILABLE_REASON

    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    def test_matches_torch_median(self, kernel_size):
        image = torch.from_numpy(_image())
        expected = median_filter_torch(image, kernel_size=kernel_size)
        result = median_filter_cpp_torch(image, kernel_size=kernel_size)
        assert torch.equal(result, expected)

    def test_lacosmic_matches_torch_path(self):
        image = _image()
        image[20, 20] += 5000
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            clean_cpp, mask_cpp = lacosmic(image, gain=1, readnoise=5, use_cpp=True)
        clean_torch, mask_torch = lacosmic(image, gain=1, readnoise=5, use_cpp=False)
        np.testing.assert_array_equal(mask_cpp, mask_torch)
        np.testing.assert_array_equal(clean_cpp, clean_torch)
        assert mask_cpp[20, 20]
