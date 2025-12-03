import numpy as np
import torch
from torch.nn.functional import avg_pool2d

from dfcosmic.utils import (
    block_replicate_torch,
    convolve,
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
    clean_image = image.clone()
    del image  # Free up memory
    final_crmask = torch.zeros(clean_image.shape, dtype=bool)
    for iteration in range(niter):
        # Step 1: Laplacian detection
        # Reuse variable name 'temp' for intermediate calculations
        temp = block_replicate_torch(clean_image, block_size_tensor)
        temp = convolve(temp, laplacian_kernel)
        temp.clip_(min=0)  # In-place operation
        temp = avg_pool2d(temp[None, None, :, :], block_size_tuple)[0, 0]

        # Step 2: Create noise model
        noise_model = median_filter_torch(clean_image, kernel_size=5)
        noise_model.clip_(min=1e-5)  # In-place
        noise_model = torch.sqrt(noise_model * gain + readnoise**2) / gain

        # Step 3: Create significance map
        sig_map = temp / noise_model
        del temp  # Done with Laplacian
        sig_map /= 2
        sig_map -= median_filter_torch(sig_map, kernel_size=5)

        # Step 4: Initial Cosmic Ray Candidates
        cr_mask = (sig_map > sigclip).float()

        # Step 5: Reject objects
        # Reuse 'temp' for object flux calculation
        temp = median_filter_torch(clean_image, kernel_size=3)
        temp -= median_filter_torch(temp, kernel_size=7)
        temp /= noise_model
        temp.clip_(min=0.01)  # In-place
        del noise_model  # Done with noise model

        # Update cr_mask in-place
        cr_mask *= ((sig_map / temp) > objlim).float()
        del temp  # Done with object flux

        # Step 6: Neighbor pixel rejection
        sigcliplow = sigclip * sigfrac

        # First growth - reuse cr_mask
        cr_mask = dilation_pytorch(cr_mask, torch.zeros((3, 3)))
        cr_mask *= sig_map
        cr_mask = (cr_mask > sigclip).float()

        # Second growth - reuse cr_mask again
        cr_mask = dilation_pytorch(cr_mask, torch.zeros((3, 3)))
        cr_mask *= sig_map
        cr_mask = (cr_mask > sigcliplow).float()
        del sig_map  # Done with significance map

        # Check if any CRs were found
        n_crs = cr_mask.sum().item()
        if n_crs == 0:
            break

        # Step 7: Image Cleaning
        final_crmask |= cr_mask.bool()  # In-place OR operation

        # Reuse 'temp' for masked data
        temp = clean_image.clone()
        temp[final_crmask] = -9999

        # Replace CR pixels with median values
        temp = median_filter_torch(temp, kernel_size=5)
        clean_image[final_crmask] = temp[final_crmask]
        del temp, cr_mask  # Clean up iteration variables

    return clean_image.cpu().numpy(), final_crmask.cpu().numpy()
