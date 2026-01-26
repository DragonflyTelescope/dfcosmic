import os
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.functional import avg_pool2d

from dfcosmic.utils import (
    block_replicate_torch,
    convolve,
    cpp_dilation_available,
    cpp_median_available,
    dilation_cpp_torch,
    dilation_pytorch,
    median_filter_cpp_torch,
    median_filter_torch,
    sigma_clip_pytorch,
)

try:
    from threadpoolctl import threadpool_limits

    _THREADPOOLCTL_AVAILABLE = True
except Exception:
    _THREADPOOLCTL_AVAILABLE = False

_KERNEL_CACHE: dict[
    tuple[str, torch.dtype],
    tuple[tuple[int, int], torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


def _get_kernels(device: torch.device, dtype: torch.dtype):
    key = (str(device), dtype)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    block_size_tuple = (2, 2)
    block_size_tensor = torch.tensor(block_size_tuple, device=device)
    laplacian_kernel = torch.tensor(
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=dtype, device=device
    )
    strel = torch.zeros((3, 3), device=device, dtype=dtype)

    cached = (block_size_tuple, block_size_tensor, laplacian_kernel, strel)
    _KERNEL_CACHE[key] = cached
    return cached


def lacosmic(
    image: torch.Tensor | np.ndarray,
    sigclip: float = 4.5,
    sigfrac: float = 0.5,
    objlim: float = 1.0,
    niter: int = 1,
    gain: float = 0.0,
    readnoise: float = 0.0,
    device: str = "cpu",
    cpu_threads: int | None = None,
    use_cpp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove cosmic rays from an image using the LA Cosmic algorithm by Pieter van Dokkum.

    The paper can be found at the following URL https://ui.adsabs.harvard.edu/abs/2001PASP..113.1420V/abstract

    Parameters
    ----------
    image : torch.Tensor|np.ndarray
        The input image.
    sigclip : float
        The detection limit for cosmic rays (sigma). Default is 4.5.
    sigfrac : float
        The fractional detection limit for neighboring pixels. Default is 0.5.
    objlim : float
        The contrast limit between CR and underlying objects. Default is 1.0.
    niter : int
        The number of iterations to perform. Default is 1.0.
    gain : float
        The gain of the image in electrons/ADU. Default is 0.0.
    readnoise : float
        The read noise of the image in electrons. Default is 0.0.
    device : str
        The device to use for computation. Default is "cpu".
    cpu_threads : int | None
        Number of cpu threads to use. Default is None.
    use_cpp : bool
        Boolean to use cpp optimized median filter and dilation algorithms. Default is True.

    Returns
    -------
        np.ndarray
            The image with cosmic rays removed.
        np.ndarray
            The mask indicating the cosmic rays.

    Notes
    -----
    If the gain is set to zero (or not provided), then we compute it assuming sky-dominated noise and poisson statistics.

    Performance Tips
    ----------------
    For CPU performance:
    - Use gain parameter if known to avoid gain estimation overhead
    - Set niter=1 for faster processing (at cost of potentially detecting fewer cosmic rays)
    - set use_cpp=True to enable C++ implementations of the median filter and dilation functions

    For best performance, use CUDA-enabled GPU by setting device='cuda'.
    """
    device = torch.device(device)
    use_cpp_median = device.type == "cpu" and cpp_median_available()
    use_cpp_dilation = device.type == "cpu" and cpp_dilation_available()
    use_cpp_median = use_cpp and device.type == "cpu" and cpp_median_available()
    use_cpp_dilation = use_cpp and device.type == "cpu" and cpp_dilation_available()

    cpu_thread_ctx = nullcontext()
    if device.type == "cpu" and cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
        if _THREADPOOLCTL_AVAILABLE:
            cpu_thread_ctx = threadpool_limits(limits=cpu_threads)

    with cpu_thread_ctx:
        # Set image to Torch tensor if it's a NumPy array
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(device).float().contiguous()
        else:
            image = image.to(device).float().contiguous()

        block_size_tuple, block_size_tensor, laplacian_kernel, strel = _get_kernels(
            device, image.dtype
        )

        clean_image = image.clone()
        del image  # Free up memory
        final_crmask = torch.zeros(clean_image.shape, dtype=bool, device=device)

        if device.type == "cpu":
            torch.backends.mkldnn.enabled = True
            median_filter_fn = (
                median_filter_cpp_torch if use_cpp_median else median_filter_torch
            )
            dilation_fn = dilation_cpp_torch if use_cpp_dilation else dilation_pytorch
        else:
            median_filter_fn = median_filter_torch
            dilation_fn = dilation_pytorch

        with torch.no_grad():
            for iteration in range(niter):
                # Step 0: If gain is not set then approximate it
                if gain <= 0:
                    sky_level = sigma_clip_pytorch(clean_image, sigma=5, maxiters=10)[
                        1
                    ]["median"]
                    med7 = median_filter_fn(clean_image, kernel_size=7)
                    residuals = clean_image - med7
                    del med7
                    abs_residuals = torch.abs(residuals)
                    del residuals
                    mad = sigma_clip_pytorch(abs_residuals, sigma=5, maxiters=10)[1][
                        "median"
                    ]
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
                temp = block_replicate_torch(clean_image, block_size_tensor)
                temp = convolve(temp, laplacian_kernel)
                temp.clip_(min=0)  # In-place operation
                temp = avg_pool2d(temp[None, None, :, :], block_size_tuple)[0, 0]

                # Step 2: Create noise model
                noise_model = median_filter_fn(clean_image, kernel_size=5)
                noise_model.clip_(min=1e-5)  # In-place
                noise_model = torch.sqrt(noise_model * gain + readnoise**2) / gain

                # Step 3: Create significance map
                sig_map = temp / noise_model
                del temp  # Done with Laplacian
                sig_map /= 2
                sig_map -= median_filter_fn(sig_map, kernel_size=5)

                # Step 4: Initial Cosmic Ray Candidates
                cr_mask = (sig_map > sigclip).float()

                # Step 5: Reject objects
                temp = median_filter_fn(clean_image, kernel_size=3)
                temp -= median_filter_fn(temp, kernel_size=7)
                temp /= noise_model
                temp.clip_(min=0.01)  # In-place
                del noise_model  # Done with noise model

                # Update cr_mask in-place
                cr_mask *= ((sig_map / temp) > objlim).float()
                del temp  # Done with object flux

                # Step 6: Neighbor pixel rejection
                sigcliplow = sigclip * sigfrac

                # First growth - reuse cr_mask
                cr_mask = dilation_fn(cr_mask, strel)
                cr_mask *= sig_map
                cr_mask = (cr_mask > sigclip).float()

                # Second growth - reuse cr_mask again
                cr_mask = dilation_fn(cr_mask, strel)
                cr_mask *= sig_map
                cr_mask = (cr_mask > sigcliplow).float()
                del sig_map  # Done with significance map

                # Check if any CRs were found
                n_crs = cr_mask.sum().item()
                if n_crs == 0:
                    break

                # Step 7: Image Cleaning
                final_crmask |= cr_mask.bool()  # In-place OR operation

                temp = clean_image.clone()
                temp[final_crmask] = -9999

                temp = median_filter_fn(temp, kernel_size=5)
                clean_image[final_crmask] = temp[final_crmask]
                del temp, cr_mask  # Clean up iteration variables

        return clean_image.cpu().numpy(), final_crmask.cpu().numpy()
