import numpy as np
from scipy.ndimage import binary_dilation
from scipy.stats import sigmaclip
from torch.nn.functional import avg_pool2d

from dfcosmic.utils import (
    block_replicate_scipy,
    block_replicate_torch,
    convolve,
    dilation_pytorch,
    median_filter_torch,
    sigma_clip_pytorch,
)


def lacosmic(
    image: torch.Tensor | np.ndarray,
    sigclip: float = 4.5,
    sigfrac: float = 0.5,
    objlim: float = 1.0,
    niter: int = 1,
    gain: float = 0.0,
    readnoise: float = 0.0,
    device: torch.device | str = torch.device("cpu"),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove cosmic rays from an image using the LA Cosmic algorithm by Pieter van Dokkum.

    The paper can be found at the following URL https://ui.adsabs.harvard.edu/abs/2001PASP..113.1420V/abstract

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
    device : torch.device | str
        The device to use for computation. Default is torch.device("cpu").
    Returns:
        np.ndarray
            The image with cosmic rays removed.
        np.ndarray
            The mask indicating the cosmic rays.

    Notes
    -----
    If the gain is set to zero (or not provided), then we compute it assuming sky-dominated noise and poisson statistics.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is not installed. Please install it with 'pip install torch'. \n If you do not wish to use the PyTorch version, please run `lacosmic_scipy` instead.")

    # Set image to Torch tensor if it's a NumPy array
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float().to(device)
    else:
        image = image.float().to(device)
    # Define kernels
    block_size_tuple = (2, 2)
    block_size_tensor = torch.tensor(block_size_tuple, device=device)
    laplacian_kernel = torch.tensor(
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32, device=device
    )
    strel = torch.zeros((3, 3), device=device)
    # Initialize image
    clean_image = image.clone()
    del image  # Free up memory
    final_crmask = torch.zeros(clean_image.shape, dtype=bool, device=device)
    for iteration in range(niter):
        # Step 0: If gain is not set then approximate it
        if gain <= 0:
            sky_level = sigma_clip_pytorch(clean_image, sigma=5, maxiters=10)[1][
                "median"
            ]
            med7 = median_filter_torch(clean_image, kernel_size=7)
            residuals = clean_image - med7
            del med7
            abs_residuals = torch.abs(residuals)
            del residuals
            mad = sigma_clip_pytorch(abs_residuals, sigma=5, maxiters=10)[1]["median"]
            sig = 1.48 * mad
            del abs_residuals
            if sig == 0:
                raise ValueError(
                    "Gain determination failed - provide estimate of gain manually. "
                    f"Sky level: {sky_level:.2f}, Sigma: {sig:.2f}"
                )
            gain = sky_level / (sig**2)

            # Sanity check (matching IRAF behavior)
            if gain <= 0:
                raise ValueError(
                    "Gain determination failed - provide estimate of gain manually. "
                    f"Sky level: {sky_level:.2f}, Sigma: {sig:.2f}"
                )
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
        cr_mask = dilation_pytorch(cr_mask, strel)
        cr_mask *= sig_map
        cr_mask = (cr_mask > sigclip).float()

        # Second growth - reuse cr_mask again
        cr_mask = dilation_pytorch(cr_mask, strel)
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


def lacosmic_scipy(
    image: np.ndarray,
    sigclip: float = 4.5,
    sigfrac: float = 0.5,
    objlim: float = 1.0,
    niter: int = 1,
    gain: float = 0.0,
    readnoise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove cosmic rays from an image using the LA Cosmic algorithm by Pieter van Dokkum.

    The paper can be found at the following URL https://ui.adsabs.harvard.edu/abs/2001PASP..113.1420V/abstract

    Parameters
    ----------
    image : np.ndarray
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
    If the gain is set to zero (or not provided), then we compute it assuming sky-dominated noise and poisson statistics.
    """

    # Define kernels
    block_size_tuple = (2, 2)
    laplacian_kernel = np.array([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]])
    strel = np.ones((3,3))

    clean_image = image.copy()
    final_crmask = np.zeros(clean_image.shape, dtype=bool)

    for iteration in range(niter):
        # Step 0: gain estimate if not provided
        if gain <= 0:
            sky_level = sigmaclip(clean_image, low=5, high=5)
            med7 = median_filter_2d(clean_image, 7)
            residuals = clean_image - med7
            abs_residuals = np.abs(residuals)
            mad = sigmaclip(abs_residuals, low=5, high=5)
            sig = 1.48 * mad
            if sig == 0:
                raise ValueError(f"Gain determination failed. Sky: {sky_level:.2f}, Sigma: {sig:.2f}")
            gain = sky_level / (sig**2)
            if gain <= 0:
                raise ValueError(f"Gain determination failed. Sky: {sky_level:.2f}, Sigma: {sig:.2f}")

        # Step 1: Laplacian detection
        temp = block_replicate_scipy(clean_image, block_size_tuple)
        temp = convolve(temp, laplacian_kernel, mode='constant', cval=0.0)
        temp = np.clip(temp, 0, None)
        temp = block_average(temp, block_size_tuple)  # downsample

        # Step 2: Noise model
        noise_model = median_filter_2d(clean_image, 5)
        noise_model = np.clip(noise_model, 1e-5, None)
        noise_model = np.sqrt(noise_model * gain + readnoise**2) / gain

        # Step 3: Significance map
        sig_map = temp / noise_model
        sig_map /= 2
        sig_map -= median_filter_2d(sig_map, 5)

        # Step 4: Initial cosmic ray candidates
        cr_mask = (sig_map > sigclip).astype(np.float32)

        # Step 5: Reject objects
        temp = median_filter_2d(clean_image, 3)
        temp -= median_filter_2d(temp, 7)
        temp /= noise_model
        temp = np.clip(temp, 0.01, None)
        cr_mask *= ((sig_map / temp) > objlim).astype(np.float32)

        # Step 6: Neighbor pixel rejection
        sigcliplow = sigclip * sigfrac

        # First growth
        cr_mask = binary_dilation(cr_mask > 0, structure=strel).astype(np.float32)
        cr_mask = (cr_mask * sig_map > sigclip).astype(np.float32)

        # Second growth
        cr_mask = binary_dilation(cr_mask > 0, structure=strel).astype(np.float32)
        cr_mask = (cr_mask * sig_map > sigcliplow).astype(np.float32)

        # Step 7: Update final mask & clean image
        final_crmask |= cr_mask.astype(bool)
        temp = clean_image.astype(np.float32)
        temp[final_crmask] = -9999
        temp = median_filter_2d(temp, 5)
        clean_image[final_crmask] = temp[final_crmask]

    return clean_image, final_crmask



