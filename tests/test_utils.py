import numpy as np
import pytest
import torch

from dfcosmic.utils import (
    _process_block_inputs,
    block_replicate_torch,
    convolve,
    dilation_pytorch,
    median_filter_torch,
    sigma_clip_pytorch,
)


class TestProcessBlockInputs:
    """Tests for _process_block_inputs helper function."""

    def test_valid_scalar_block_size(self):
        """Test with valid scalar block size."""
        data = torch.ones((4, 4))
        block_size = torch.tensor([2])
        processed_data, processed_block_size = _process_block_inputs(data, block_size)
        assert torch.equal(processed_data, data)
        assert processed_block_size.shape[0] == 2
        assert torch.all(processed_block_size == 2)

    def test_valid_vector_block_size(self):
        """Test with valid vector block size."""
        data = torch.ones((4, 4))
        block_size = torch.tensor([2, 3])
        processed_data, processed_block_size = _process_block_inputs(data, block_size)
        assert torch.equal(processed_data, data)
        assert torch.equal(processed_block_size, torch.tensor([2, 3]))

    def test_invalid_negative_block_size(self):
        """Test that negative block size raises ValueError."""
        data = torch.ones((4, 4))
        block_size = torch.tensor([-1])
        with pytest.raises(
            ValueError, match="block_size elements must be strictly positive"
        ):
            _process_block_inputs(data, block_size)

    def test_invalid_zero_block_size(self):
        """Test that zero block size raises ValueError."""
        data = torch.ones((4, 4))
        block_size = torch.tensor([0])
        with pytest.raises(
            ValueError, match="block_size elements must be strictly positive"
        ):
            _process_block_inputs(data, block_size)

    def test_mismatched_dimensions(self):
        """Test that mismatched dimensions raise ValueError."""
        data = torch.ones((4, 4))
        block_size = torch.tensor([2, 3, 4])
        with pytest.raises(
            ValueError, match="block_size must be a scalar or have the same"
        ):
            _process_block_inputs(data, block_size)


class TestBlockReplicateTorch:
    """Tests for block_replicate_torch function."""

    def test_basic_replication_2x2(self):
        """Test basic 2x2 block replication."""
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        block_size = torch.tensor([2, 2])
        result = block_replicate_torch(data, block_size, conserve_sum=False)
        assert result.shape == (4, 4)
        assert torch.equal(result[0:2, 0:2], torch.ones((2, 2)) * 1.0)
        assert torch.equal(result[0:2, 2:4], torch.ones((2, 2)) * 2.0)

    def test_conserve_sum_true(self):
        """Test that conserve_sum=True preserves the sum."""
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        block_size = torch.tensor([2, 2])
        result = block_replicate_torch(data, block_size, conserve_sum=True)
        assert torch.allclose(result.sum(), data.sum())

    def test_conserve_sum_false(self):
        """Test that conserve_sum=False increases the sum."""
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        block_size = torch.tensor([2, 2])
        result = block_replicate_torch(data, block_size, conserve_sum=False)
        assert result.sum() > data.sum()

    def test_scalar_block_size(self):
        """Test with scalar block size for 2D array."""
        data = torch.ones((3, 3))
        block_size = torch.tensor([2])
        result = block_replicate_torch(data, block_size, conserve_sum=False)
        assert result.shape == (6, 6)

    def test_different_block_sizes(self):
        """Test with different block sizes for each dimension."""
        data = torch.ones((2, 3))
        block_size = torch.tensor([2, 3])
        result = block_replicate_torch(data, block_size, conserve_sum=False)
        assert result.shape == (4, 9)

    def test_1d_array(self):
        """Test with 1D array."""
        data = torch.tensor([1.0, 2.0, 3.0])
        block_size = torch.tensor([2])
        result = block_replicate_torch(data, block_size, conserve_sum=False)
        assert result.shape == (6,)


class TestConvolve:
    """Tests for convolve function."""

    def test_identity_kernel(self):
        """Test convolution with identity kernel produces expected output shape."""
        image = torch.randn((10, 10))
        kernel = torch.zeros((3, 3))
        kernel[1, 1] = 1.0
        result = convolve(image, kernel)
        # Check that the result has the correct shape
        assert result.shape == image.shape
        # Identity kernel should approximately preserve the image
        assert torch.allclose(result, image, rtol=1e-4, atol=1e-4)

    def test_kernel_smaller_than_image(self):
        """Test that kernel can be smaller than image."""
        image = torch.randn((20, 20))
        kernel = torch.randn((3, 3))
        result = convolve(image, kernel)
        assert result.shape == image.shape

    def test_laplacian_kernel(self):
        """Test with Laplacian kernel."""
        image = torch.ones((10, 10))
        laplacian_kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
        )
        result = convolve(image, laplacian_kernel)
        assert result.shape == image.shape
        # Laplacian of constant image should be zero in the interior (not at edges)
        assert torch.allclose(result[1:-1, 1:-1], torch.zeros(8, 8), atol=1e-5)

    def test_output_dtype(self):
        """Test that output preserves float dtype."""
        image = torch.randn((10, 10), dtype=torch.float32)
        kernel = torch.randn((3, 3), dtype=torch.float32)
        result = convolve(image, kernel)
        assert result.dtype == torch.float32


class TestMedianFilterTorch:
    """Tests for median_filter_torch function."""

    def test_basic_median_filter(self):
        """Test basic median filtering."""
        image = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = median_filter_torch(image, kernel_size=3)
        assert result.shape == image.shape

    def test_removes_outliers(self):
        """Test that median filter removes outliers."""
        image = torch.ones((5, 5))
        image[2, 2] = 100.0  # Add outlier
        result = median_filter_torch(image, kernel_size=3)
        assert result[2, 2] < 100.0

    def test_kernel_size_5(self):
        """Test with kernel size 5."""
        image = torch.randn((10, 10))
        result = median_filter_torch(image, kernel_size=5)
        assert result.shape == image.shape

    def test_preserves_shape(self):
        """Test that output shape matches input shape."""
        image = torch.randn((7, 9))
        result = median_filter_torch(image, kernel_size=3)
        assert result.shape == image.shape

    def test_uniform_image(self):
        """Test with uniform image."""
        image = torch.ones((5, 5)) * 5.0
        result = median_filter_torch(image, kernel_size=3)
        assert torch.allclose(result, image)


class TestDilationPytorch:
    """Tests for dilation_pytorch function."""

    def test_basic_dilation(self):
        """Test basic dilation operation."""
        image = torch.zeros((5, 5))
        image[2, 2] = 1.0
        strel = torch.zeros((3, 3))
        result = dilation_pytorch(image, strel)
        assert result.shape == image.shape
        assert result[2, 2] >= 1.0

    def test_preserves_shape(self):
        """Test that dilation preserves image shape."""
        image = torch.randn((10, 10))
        strel = torch.zeros((3, 3))
        result = dilation_pytorch(image, strel)
        assert result.shape == image.shape

    def test_border_value(self):
        """Test custom border value."""
        image = torch.zeros((5, 5))
        strel = torch.zeros((3, 3))
        result = dilation_pytorch(image, strel, border_value=1.0)
        assert result.shape == image.shape

    def test_custom_origin(self):
        """Test with custom origin."""
        image = torch.zeros((5, 5))
        image[2, 2] = 1.0
        strel = torch.zeros((3, 3))
        result = dilation_pytorch(image, strel, origin=(1, 1))
        assert result.shape == image.shape

    def test_increases_values(self):
        """Test that dilation generally increases or maintains values."""
        image = torch.randn((5, 5)).abs()
        strel = torch.zeros((3, 3))
        result = dilation_pytorch(image, strel)
        assert result.sum() >= image.sum()

    def test_binary_image(self):
        """Test with binary image."""
        image = torch.zeros((7, 7))
        image[3, 3] = 1.0
        strel = torch.zeros((3, 3))
        result = dilation_pytorch(image, strel)
        assert result.shape == image.shape
        assert result.max() >= 1.0


class TestSigmaClipPytorch:
    """Tests for sigma_clip_pytorch function."""

    def test_basic_sigma_clipping(self):
        """Test basic sigma clipping operation."""
        data = torch.randn(100) * 10 + 100
        clipped_data, stats = sigma_clip_pytorch(data, sigma=3.0, maxiters=5)
        assert isinstance(clipped_data, torch.Tensor)
        assert isinstance(stats, dict)
        assert "median" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "niter" in stats
        assert "npix" in stats

    def test_removes_outliers(self):
        """Test that sigma clipping removes outliers."""
        data = torch.randn(100) * 10 + 100
        # Add some extreme outliers
        data = torch.cat([data, torch.tensor([1000.0, 2000.0, -1000.0])])
        clipped_data, stats = sigma_clip_pytorch(data, sigma=3.0, maxiters=10)
        # Clipped data should have fewer pixels
        assert len(clipped_data) < len(data)
        # Stats should be reasonable (not affected by outliers)
        assert 80 < stats["median"] < 120
        assert 80 < stats["mean"] < 120

    def test_scalar_sigma(self):
        """Test with scalar sigma (symmetric clipping)."""
        data = torch.randn(100) * 10 + 100
        clipped_data, stats = sigma_clip_pytorch(data, sigma=3.0, maxiters=5)
        assert len(clipped_data) <= len(data)
        assert stats["niter"] <= 5

    def test_tuple_sigma(self):
        """Test with tuple sigma (asymmetric clipping)."""
        data = torch.randn(100) * 10 + 100
        clipped_data, stats = sigma_clip_pytorch(data, sigma=(2.0, 3.0), maxiters=5)
        assert len(clipped_data) <= len(data)
        assert stats["niter"] <= 5

    def test_convergence(self):
        """Test that sigma clipping converges."""
        data = torch.randn(100) * 10 + 100
        clipped_data, stats = sigma_clip_pytorch(data, sigma=3.0, maxiters=10)
        # Should converge before max iterations on normal data
        assert stats["niter"] <= 10

    def test_no_clipping_needed(self):
        """Test with data that doesn't need clipping."""
        data = torch.ones(100) * 100.0
        clipped_data, stats = sigma_clip_pytorch(data, sigma=3.0, maxiters=10)
        assert len(clipped_data) == len(data)
        assert stats["median"] == 100.0
        assert stats["mean"] == 100.0
        assert stats["std"] == 0.0

    def test_stats_dict_contents(self):
        """Test that stats dictionary contains correct keys and types."""
        data = torch.randn(100) * 10 + 100
        _, stats = sigma_clip_pytorch(data, sigma=3.0, maxiters=5)
        assert isinstance(stats["median"], float)
        assert isinstance(stats["mean"], float)
        assert isinstance(stats["std"], float)
        assert isinstance(stats["niter"], int)
        assert isinstance(stats["npix"], int)
        assert stats["npix"] > 0
        assert stats["niter"] >= 1

    def test_preserves_data_distribution(self):
        """Test that clipping preserves central data distribution."""
        data = torch.randn(1000) * 10 + 100
        clipped_data, stats = sigma_clip_pytorch(data, sigma=3.0, maxiters=10)
        # Most data should remain (only extreme outliers removed)
        assert len(clipped_data) > len(data) * 0.9
        # Stats should be close to true parameters
        assert 95 < stats["mean"] < 105
        assert 8 < stats["std"] < 12

    def test_maxiters_limit(self):
        """Test that maxiters limits number of iterations."""
        data = torch.randn(100) * 10 + 100
        _, stats = sigma_clip_pytorch(data, sigma=0.5, maxiters=3)
        assert stats["niter"] <= 3

    def test_2d_input_flattened(self):
        """Test that 2D input is properly flattened."""
        data = torch.randn(10, 10) * 10 + 100
        clipped_data, stats = sigma_clip_pytorch(data, sigma=3.0, maxiters=5)
        assert clipped_data.ndim == 1
        assert stats["npix"] <= 100

    def test_does_not_modify_original(self):
        """Test that original data is not modified."""
        data = torch.randn(100) * 10 + 100
        data_original = data.clone()
        sigma_clip_pytorch(data, sigma=3.0, maxiters=5)
        assert torch.equal(data, data_original)
