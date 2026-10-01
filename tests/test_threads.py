import os

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_info

import dfcosmic.core as core
from dfcosmic import lacosmic

ENV_VARS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]


def _global_state() -> dict:
    return {
        "torch_threads": torch.get_num_threads(),
        "mkldnn": torch.backends.mkldnn.enabled,
        "env": {name: os.environ.get(name) for name in ENV_VARS},
        "pools": sorted(
            (pool["filepath"], pool["num_threads"]) for pool in threadpool_info()
        ),
    }


@pytest.fixture
def image() -> np.ndarray:
    return np.random.default_rng(0).normal(100, 5, (128, 128)).astype(np.float32)


@pytest.fixture
def four_torch_threads():
    """Run the test with a known torch thread count different from the limits used."""
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.mark.usefixtures("four_torch_threads")
class TestCpuThreadsDoesNotLeak:
    """cpu_threads must only apply for the duration of the call."""

    @pytest.mark.parametrize("cpu_threads", [None, 1, 2])
    @pytest.mark.parametrize("use_cpp", [None, False])
    def test_global_state_unchanged(self, image, monkeypatch, cpu_threads, use_cpp):
        for name in ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        before = _global_state()
        lacosmic(image, gain=1, readnoise=5, cpu_threads=cpu_threads, use_cpp=use_cpp)
        assert _global_state() == before

    def test_existing_environment_variables_untouched(self, image, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "3")
        before = _global_state()
        lacosmic(image, gain=1, readnoise=5, cpu_threads=1)
        assert _global_state() == before

    def test_mkldnn_setting_is_respected(self, image, monkeypatch):
        """A caller that has turned oneDNN off must not get it turned back on."""
        monkeypatch.setattr(torch.backends.mkldnn, "enabled", False)
        lacosmic(image, gain=1, readnoise=5, cpu_threads=1)
        assert torch.backends.mkldnn.enabled is False

    def test_restored_after_error(self, image):
        before = _global_state()
        with pytest.raises(ValueError, match="2D"):
            lacosmic(image[None], gain=1, readnoise=5, cpu_threads=1)
        assert _global_state() == before

    @pytest.mark.parametrize("cpu_threads", [1, 2])
    def test_limit_applies_during_call(self, image, monkeypatch, cpu_threads):
        seen = []
        median_filter = core.median_filter_torch

        def recording_median_filter(*args, **kwargs):
            seen.append(torch.get_num_threads())
            return median_filter(*args, **kwargs)

        monkeypatch.setattr(core, "median_filter_torch", recording_median_filter)
        lacosmic(image, gain=1, readnoise=5, cpu_threads=cpu_threads, use_cpp=False)
        assert seen and set(seen) == {cpu_threads}
        assert torch.get_num_threads() == 4

    def test_result_independent_of_cpu_threads(self, image):
        image[40, 40] += 5000
        clean_ref, mask_ref = lacosmic(image, gain=1, readnoise=5)
        clean, mask = lacosmic(image, gain=1, readnoise=5, cpu_threads=1)
        np.testing.assert_array_equal(mask, mask_ref)
        np.testing.assert_array_equal(clean, clean_ref)

    @pytest.mark.parametrize("cpu_threads", [0, -1])
    def test_invalid_cpu_threads_raises(self, image, cpu_threads):
        before = _global_state()
        with pytest.raises(ValueError, match="cpu_threads"):
            lacosmic(image, gain=1, readnoise=5, cpu_threads=cpu_threads)
        assert _global_state() == before
