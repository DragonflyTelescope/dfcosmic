import numpy as np
import pytest
import torch

from dfcosmic.core import lacosmic


class TestLacosmic:
    """Tests for lacosmic cosmic ray removal function."""

    def test_numpy_input(self):
        """Test that numpy array input works."""
        image = np.random.rand(50, 50).astype(np.float32) * 1000
        cleaned, mask = lacosmic(image)
        assert isinstance(cleaned, np.ndarray)
        assert isinstance(mask, np.ndarray)
        assert cleaned.shape == image.shape
        assert mask.shape == image.shape

    def test_torch_input(self):
        """Test that torch tensor input works."""
        image = torch.rand((50, 50)) * 1000
        cleaned, mask = lacosmic(image)
        assert isinstance(cleaned, np.ndarray)
        assert isinstance(mask, np.ndarray)
        assert cleaned.shape == image.shape
        assert mask.shape == image.shape

    def test_output_types(self):
        """Test that outputs are numpy arrays."""
        image = torch.rand((50, 50)) * 1000
        cleaned, mask = lacosmic(image)
        assert isinstance(cleaned, np.ndarray)
        assert isinstance(mask, np.ndarray)

    def test_mask_is_boolean(self):
        """Test that mask contains boolean values."""
        image = torch.rand((50, 50)) * 1000
        _, mask = lacosmic(image)
        assert mask.dtype == bool or np.issubdtype(mask.dtype, np.bool_)

    def test_with_cosmic_ray(self):
        """Test that algorithm runs on image with cosmic rays."""
        image = torch.rand((50, 50)) * 1000 + 100
        # Add some bright cosmic ray spikes
        image[25, 25] = 10000.0
        image[25, 26] = 8000.0
        image[24, 25] = 8000.0
        float(image.max().item())
        cleaned, mask = lacosmic(image, sigclip=4.5, niter=2)
        # Test that function completes and returns expected shapes
        assert cleaned.shape == (50, 50)
        assert mask.shape == (50, 50)
        # Verify outputs are valid (no NaN or Inf)
        assert not np.isnan(cleaned).any()
        assert not np.isinf(cleaned).any()

    def test_clean_image_minimal_mask(self):
        """Test that clean image produces minimal masking."""
        # Create smooth gradient image (no cosmic rays)
        x = torch.linspace(100, 200, 50)
        y = torch.linspace(100, 200, 50)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        image = xx + yy
        # Use manual gain to avoid approximation issues with smooth gradient
        _, mask = lacosmic(image, sigclip=5.0, gain=1.0, readnoise=5.0)
        # Very few or no pixels should be masked in a smooth image
        assert mask.sum() < image.numel() * 0.1  # Less than 10% masked

    def test_single_iteration(self):
        """Test with single iteration."""
        image = torch.rand((50, 50)) * 1000
        cleaned, mask = lacosmic(image, niter=1)
        assert cleaned.shape == image.shape
        assert mask.shape == image.shape

    def test_multiple_iterations(self):
        """Test with multiple iterations."""
        image = torch.rand((50, 50)) * 1000
        cleaned, mask = lacosmic(image, niter=3)
        assert cleaned.shape == image.shape
        assert mask.shape == image.shape

    def test_default_parameters(self):
        """Test with default parameters."""
        image = torch.rand((50, 50)) * 1000
        cleaned, mask = lacosmic(image)
        assert cleaned.shape == image.shape
        assert mask.shape == image.shape

    def test_custom_sigclip(self):
        """Test with custom sigclip value."""
        image = torch.rand((50, 50)) * 1000
        # Lower sigclip should detect more cosmic rays
        _, mask_low = lacosmic(image, sigclip=3.0)
        _, mask_high = lacosmic(image, sigclip=6.0)
        # Lower threshold typically finds more candidates
        assert mask_low.sum() >= mask_high.sum()

    def test_custom_sigfrac(self):
        """Test with custom sigfrac value."""
        image = torch.rand((50, 50)) * 1000
        cleaned, mask = lacosmic(image, sigfrac=0.3)
        assert cleaned.shape == image.shape
        assert mask.shape == image.shape

    def test_custom_objlim(self):
        """Test with custom objlim value."""
        image = torch.rand((50, 50)) * 1000
        cleaned, mask = lacosmic(image, objlim=2.0)
        assert cleaned.shape == image.shape
        assert mask.shape == image.shape

    def test_with_gain_and_readnoise(self):
        """Test with gain and readnoise parameters."""
        image = torch.rand((50, 50)) * 1000
        cleaned, mask = lacosmic(image, gain=2.0, readnoise=5.0)
        assert cleaned.shape == image.shape
        assert mask.shape == image.shape

    def test_zero_gain_readnoise(self):
        """Test with zero gain and readnoise (uses automatic gain approximation)."""
        image = torch.rand((50, 50)) * 1000 + 500
        cleaned, mask = lacosmic(image, gain=0.0, readnoise=0.0)
        assert cleaned.shape == image.shape
        assert mask.shape == image.shape

    def test_automatic_gain_approximation(self):
        """Test that automatic gain approximation works when gain=0."""
        # Create realistic image with sky background
        image = torch.randn((100, 100)) * 10 + 1000
        # Should not raise an error
        cleaned, mask = lacosmic(image, gain=0.0, readnoise=0.0, niter=1)
        assert cleaned.shape == image.shape
        assert mask.shape == image.shape
        assert not np.isnan(cleaned).any()
        assert not np.isinf(cleaned).any()

    def test_automatic_gain_with_cosmic_rays(self):
        """Test automatic gain approximation with cosmic rays present."""
        image = torch.randn((100, 100)) * 10 + 1000
        # Add cosmic rays
        image[25, 25] = 10000.0
        image[50, 50] = 8000.0
        cleaned, mask = lacosmic(image, gain=0.0, niter=2)
        assert cleaned.shape == image.shape
        assert mask.shape == image.shape
        # Should detect some cosmic rays
        assert mask.sum() > 0

    def test_gain_approximation_failure(self):
        """Test that gain approximation fails gracefully on invalid data."""
        # Create uniform image which will have zero variance and cause gain calculation to fail
        image = torch.ones((50, 50)) * 100.0
        with pytest.raises(ValueError, match="Gain determination failed"):
            lacosmic(image, gain=0.0, niter=1)

    def test_gain_approximation_zero_sigma(self):
        """Test that gain approximation catches zero sigma (sig==0) before division."""
        # Create uniform image which will result in sig==0
        image = torch.ones((50, 50)) * 100.0
        # Should raise ValueError when sig==0 is detected
        with pytest.raises(ValueError, match="Gain determination failed.*Sigma: 0.00"):
            lacosmic(image, gain=0.0, niter=1)

    def test_manual_gain_overrides_approximation(self):
        """Test that manually specified gain is used instead of approximation."""
        image = torch.rand((50, 50)) * 1000 + 500
        # With manual gain, should work even if automatic would fail
        cleaned, mask = lacosmic(image, gain=2.0, readnoise=5.0)
        assert cleaned.shape == image.shape
        assert mask.shape == image.shape

    def test_preserves_non_cr_pixels(self):
        """Test that non-cosmic-ray pixels are relatively preserved."""
        image = torch.ones((50, 50)) * 1000.0
        # Use manual gain to avoid division by zero with uniform image
        cleaned, _ = lacosmic(image, gain=1.0, readnoise=5.0)
        # Most pixels should remain close to original
        diff = np.abs(cleaned - image.cpu().numpy())
        assert np.median(diff) < 100  # Median change should be small

    def test_rectangular_image(self):
        """Test with non-square image."""
        image = torch.rand((40, 60)) * 1000
        cleaned, mask = lacosmic(image)
        assert cleaned.shape == (40, 60)
        assert mask.shape == (40, 60)

    def test_small_image(self):
        """Test with smaller image."""
        image = torch.rand((20, 20)) * 1000
        cleaned, mask = lacosmic(image)
        assert cleaned.shape == image.shape
        assert mask.shape == image.shape

    def test_large_values(self):
        """Test with large pixel values."""
        image = torch.rand((50, 50)) * 100000
        cleaned, mask = lacosmic(image)
        assert cleaned.shape == image.shape
        assert mask.shape == image.shape
        assert not np.isnan(cleaned).any()
        assert not np.isinf(cleaned).any()

    def test_output_no_nan_inf(self):
        """Test that output contains no NaN or Inf values."""
        image = torch.rand((50, 50)) * 1000
        cleaned, _ = lacosmic(image)
        assert not np.isnan(cleaned).any()
        assert not np.isinf(cleaned).any()

    def test_consistent_results(self):
        """Test that same input gives same output (deterministic)."""
        image = torch.rand((50, 50)) * 1000
        cleaned1, mask1 = lacosmic(image.clone())
        cleaned2, mask2 = lacosmic(image.clone())
        np.testing.assert_array_almost_equal(cleaned1, cleaned2)
        np.testing.assert_array_equal(mask1, mask2)


class TestLacosmicEdgeCases:
    """Edge case tests for lacosmic function."""

    def test_uniform_image(self):
        """Test with perfectly uniform image."""
        image = torch.ones((50, 50)) * 1000.0
        # Use manual gain to avoid division by zero with uniform image
        cleaned, mask = lacosmic(image, gain=1.0, readnoise=5.0)
        assert cleaned.shape == image.shape
        # Uniform image should have minimal masking
        assert mask.sum() < image.numel() * 0.05

    def test_no_cosmic_rays_early_exit(self):
        """Test early exit when no cosmic rays are detected."""
        # Create a smooth image with no cosmic rays
        image = torch.rand((50, 50)) * 100 + 1000
        # Use high sigclip to ensure no CRs are detected
        cleaned, mask = lacosmic(image, sigclip=20.0, niter=5, gain=1.0, readnoise=5.0)
        # Should exit early if no CRs found
        assert mask.sum() == 0
        # Image should be unchanged
        np.testing.assert_array_almost_equal(cleaned, image.cpu().numpy())

    def test_very_low_sigclip(self):
        """Test with very low sigclip (aggressive detection)."""
        image = torch.rand((50, 50)) * 1000
        cleaned, mask = lacosmic(image, sigclip=2.0)
        assert cleaned.shape == image.shape

    def test_very_high_sigclip(self):
        """Test with very high sigclip (conservative detection)."""
        image = torch.rand((50, 50)) * 1000
        cleaned, mask = lacosmic(image, sigclip=10.0)
        assert cleaned.shape == image.shape
        # High threshold should find fewer cosmic rays
        assert mask.sum() < image.numel() * 0.2

    def test_multiple_cosmic_rays(self):
        """Test with multiple cosmic rays."""
        image = torch.rand((50, 50)) * 1000 + 100
        # Add multiple cosmic rays
        image[10, 10] = 10000.0
        image[10, 11] = 8000.0
        image[20, 20] = 10000.0
        image[20, 21] = 8000.0
        image[30, 30] = 10000.0
        image[30, 31] = 8000.0
        cleaned, mask = lacosmic(image, sigclip=4.5, niter=3)
        # Test that function completes and returns expected shapes
        assert cleaned.shape == (50, 50)
        assert mask.shape == (50, 50)
        # Verify outputs are valid (no NaN or Inf)
        assert not np.isnan(cleaned).any()
        assert not np.isinf(cleaned).any()
