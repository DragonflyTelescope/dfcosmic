import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_random_generators():
    """Many tests draw random images: seed the generators so every run sees the same."""
    np.random.seed(0)
    torch.manual_seed(0)
