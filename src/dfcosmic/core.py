import numpy as np
import torch
from torch.nn.functional import avg_pool2d

from dfcosmic.utils import (
    block_replicate_torch,
    convolve_fft,
    dilation_pytorch,
    median_filter_torch,
)


def lacosmic(
    image: torch.Tensor | np.ndarray,
    sigclip: float = 4.5,
    sigfrac: float = 0.5,
    objlim: float = 1.0,
    niter: int = 1,
    gain: float = 0.0,
    readnoise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove cosmic rays from an image using the LA Cosmic algorithm by Pieter van Dokkum.

    Parameters
    ----------
    image : torch.Tensor|np.ndarray
        The input image.
    sigclip : float
        The detection limit for cosmic rays (sigma)
    sigfrac : float
        The fractional detection limit for neighboring pixels
    objlim : float
        The contrast limit between CR and underlying objects
    niter : int
        The number of iterations to perform.
    gain : float
        The gain of the image in electrons/ADU. Default is 0.0.
    readnoise : float
        The read noise of the image in electrons. Default is 0.0.

    Returns:
        np.ndarray
            The image with cosmic rays removed.
        np.ndarray
            The mask indicating the cosmic rays.

    Notes
    -----
    If gain and readnoise are set to zero, then ...
    """
    # Set image to Torch tensor if it's a NumPy array
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    # Define kernels
    block_size_tuple = (2, 2)
    block_size_tensor = torch.tensor(block_size_tuple)
    laplacian_kernel = torch.tensor(
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32
    )
    # Initialize image
    image_original = image.clone()
    del image  # Free up memory
    final_crmask = torch.zeros(image_original.shape, dtype=bool)
    for iteration in range(niter):
        # Step 1: Laplacian detection
        img_lp = block_replicate_torch(image_original, block_size_tensor)
        img_lp = convolve_fft(img_lp, laplacian_kernel)
        img_lp = torch.clip(img_lp, min=0)
        img_lp = avg_pool2d(img_lp[None, None, :, :], block_size_tuple)[0, 0]
        # Step 2: Create noise model
        noise_model = median_filter_torch(image_original, kernel_size=5).clip(1e-5)
        noise_model = torch.sqrt(noise_model * gain + readnoise**2) / gain
        # Step 3: Create significance map (i.e. laplacian by noise)
        sig_map = img_lp / noise_model
        sig_map /= 2  # Divide by 2 since block replication double counts edges
        sig_map -= median_filter_torch(sig_map, kernel_size=5)
        # Step 4: Initial Cosmic Ray Candidates
        cr_mask = sig_map > sigclip
        # Step 5: Reject objects (such as HII regions or stars)
        img_lp = median_filter_torch(image_original, kernel_size=3)
        img_lp = (
            (img_lp - median_filter_torch(img_lp, kernel_size=7)) / noise_model
        ).clip(min=0.01)
        cr_mask2 = (sig_map / img_lp) > objlim
        cr_mask = cr_mask * cr_mask2
        # Step 6: Neighbor pixel rejection
        sigcliplow = sigclip * sigfrac
        # First growth - check at sigma clip
        cr_mask = dilation_pytorch(cr_mask, torch.zeros((3, 3)))
        cr_mask *= sig_map
        cr_mask = (cr_mask > sigclip).float()

        # Second growth  - check at lower threshold
        cr_mask = dilation_pytorch(cr_mask, torch.zeros((3, 3)))
        cr_mask *= sig_map
        cr_mask = (cr_mask > sigcliplow).float()
        # Step 7: Image Cleaning
        final_crmask = torch.logical_or(final_crmask, cr_mask)
        # Mask CR pixels with large negative value
        masked_data = image_original.clone()
        masked_data[final_crmask] = 0
        # Replace CR pixels with median values from 5x5 median filter
        image_original[final_crmask] = median_filter_torch(masked_data, kernel_size=5)[
            final_crmask
        ]
        return image_original.cpu().numpy(), final_crmask.cpu().numpy()
