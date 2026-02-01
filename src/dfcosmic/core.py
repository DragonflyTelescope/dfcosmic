import os
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.functional import avg_pool2d

from dfcosmic.utils import (
    block_replicate_torch,
    convolve,
    cpp_median_available,
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
    strel = torch.ones((3, 3), device=device, dtype=dtype)
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
    verbose: bool = False,
    rss_debug: bool = False,
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
    verbose : bool
        Print iteration progress. Default is False.
    rss_debug : bool
        Print RSS memory at key steps. Default is False.

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

    use_cpp_median = use_cpp and device.type == "cpu" and cpp_median_available()

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
        # Move/cast to torch
        if isinstance(image, np.ndarray):
            image_t = torch.from_numpy(image).to(device).float().contiguous()
        else:
            image_t = image.to(device).float().contiguous()

        block_size_tuple, block_size_tensor, laplacian_kernel, strel = _get_kernels(
            device, image_t.dtype
        )
        gkernel = torch.ones((3, 3), dtype=image_t.dtype, device=device)

        clean_image = image_t.clone()
        del image_t

        final_crmask = torch.zeros(clean_image.shape, dtype=torch.bool, device=device)
        if device.type == "cpu":
            torch.backends.mkldnn.enabled = True
            median_filter_fn = (
                median_filter_cpp_torch if use_cpp_median else median_filter_torch
            )
        else:
            median_filter_fn = median_filter_torch

        with torch.no_grad():
            for iteration in range(niter):
                if verbose:
                    print("")
                    print(f"{'_' * 31} Iteration {iteration + 1} {'_' * 35}")
                    print("")

                # Step 0: Gain estimation (if requested)
                if gain <= 0:
                    if verbose and iteration == 0:
                        print("Trying to determine gain automatically:")
                    elif verbose:
                        print("Improving gain estimate:")

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
                    del abs_residuals
                    sig = 1.48 * mad

                    if verbose:
                        print(f"  Approximate sky level = {sky_level:.2f} ADU")
                        print(f"  Sigma of sky = {sig:.2f}")
                        print(f"  Estimated gain = {sky_level / (sig**2):.2f}")
                        print("")

                    if sig == 0:
                        raise ValueError(
                            "Gain determination failed - provide estimate of gain manually. "
                            f"Sky level: {sky_level:.2f}, Sigma: {sig:.2f}"
                        )
                    gain = sky_level / (sig**2)
                    if gain <= 0:
                        raise ValueError(
                            "Gain determination failed - provide estimate of gain manually. "
                            f"Sky level: {sky_level:.2f}, Sigma: {sig:.2f}"
                        )

                if verbose:
                    print("Convolving image with Laplacian kernel")
                    print("")

                # Step 1: Laplacian detection
                temp = block_replicate_torch(
                    clean_image, block_size_tensor, conserve_sum=False
                )
                temp = convolve(temp, laplacian_kernel)
                temp.clamp_(min=0)
                temp = avg_pool2d(temp[None, None, :, :], block_size_tuple)[0, 0]

                if verbose:
                    print("Creating noise model using:")
                    print(f"  gain = {gain:.2f} electrons/ADU")
                    print(f"  readnoise = {readnoise:.2f} electrons")
                    print("")

                # Step 2: Noise model
                med5 = median_filter_fn(clean_image, kernel_size=5)
                med5.clamp_(min=1e-4)
                noise = torch.sqrt(med5 * gain + readnoise**2) / gain
                del med5

                # Step 3: Significance map
                temp /= noise
                sigmap = temp
                sigmap /= 2.0
                sigmap -= median_filter_fn(sigmap, kernel_size=5)

                if verbose:
                    print("Selecting candidate cosmic rays")
                    print(f"  sigma limit = {sigclip:.1f}")
                    print("")

                # Step 4: Initial CR candidates
                firstsel = (sigmap >= sigclip).to(sigmap.dtype)

                if verbose:
                    print("Removing suspected compact bright objects (e.g. stars)")
                    print(f"  selecting cosmic rays > {objlim:.1f} times object flux")
                    print("")

                # Step 5: Reject objects (fine structure)
                med3 = median_filter_fn(clean_image, kernel_size=3)
                med7 = median_filter_fn(med3, kernel_size=7)
                med3 = med3 - med7
                del med7
                med3 = med3 / noise
                med3.clamp_(min=0.01)

                starreject = (firstsel * sigmap) / med3
                del med3
                starreject = (starreject >= objlim).to(sigmap.dtype)
                firstsel = firstsel * starreject
                del starreject

                # Step 6: Neighbor pixel rejection / grow
                sigcliplow = sigclip * sigfrac

                if verbose:
                    print("Finding neighbouring pixels affected by cosmic rays")
                    print(f"  sigma limit = {sigcliplow:.1f}")
                    print("")

                # First grow: keep pixels whose (grown mask * sig_map) > sigclip
                gfirstsel = convolve(firstsel, gkernel)
                del firstsel
                gfirstsel = (gfirstsel > 0.5).to(sigmap.dtype)
                gfirstsel = gfirstsel * sigmap
                gfirstsel = (gfirstsel > sigclip).to(sigmap.dtype)

                # Second grow: threshold at sigcliplow
                finalsel = convolve(gfirstsel, gkernel)
                del gfirstsel
                finalsel = (finalsel > 0.5).to(sigmap.dtype)
                finalsel = finalsel * sigmap
                finalsel = (finalsel > sigcliplow).to(sigmap.dtype)
                del sigmap

                # Count only NEW cosmic rays found in this iteration
                new_crs = (~final_crmask).to(finalsel.dtype) * finalsel
                npix = new_crs.sum().item()

                del new_crs

                if npix == 0:
                    if finalsel.sum().item() > 0:
                        # Update mask even if no new CRs (edge case)
                        final_crmask |= finalsel.bool()
                    break

                # Step 7: Clean flagged pixels with 5x5 median replacement
                final_crmask |= finalsel.bool()

                if verbose:
                    print(f"{int(npix)} cosmic rays found in iteration {iteration + 1}")
                    print("")

                # Create cleaned output image using 5x5 median
                tmp = clean_image.clone()
                sentinel = clean_image.max() * 1e4 + 1e6
                tmp[final_crmask] = sentinel
                tmp = median_filter_fn(tmp, kernel_size=5)
                # Only use the median at CR locations
                clean_image[final_crmask] = tmp[final_crmask]

                del tmp, finalsel, noise

        return clean_image.cpu().numpy(), final_crmask.cpu().numpy()
