import math
import os
from contextlib import nullcontext
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from dfcosmic.utils import (
    convolve,
    cpp_median_available,
    fill_from_unflagged_neighbors,
    laplacian_pool_chunked,
    median_filter_at,
    median_filter_cpp_torch,
    median_of_median_at,
    median_filter_torch,
    sigma_clip_pytorch,
    use_sparse_median,
    warn_cpp_median_unavailable,
)

_KERNEL_CACHE: dict[
    tuple[str, torch.dtype],
    tuple[tuple[int, int], torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


def _current_rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        return None
    return None


def _log_rss(label: str) -> None:
    rss_mb = _current_rss_mb()
    if rss_mb is None:
        print(f"[rss] {label}: unavailable")
    else:
        print(f"[rss] {label}: {rss_mb:.1f} MiB")


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
    use_cpp: bool | None = None,
    verbose: bool = False,
    rss_debug: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove cosmic rays from an image using the LA Cosmic algorithm by Pieter van Dokkum.

    The paper can be found at the following URL https://ui.adsabs.harvard.edu/abs/2001PASP..113.1420V/abstract

    Parameters
    ----------
    image : torch.Tensor|np.ndarray
        The input image. Must be 2D. Numpy arrays of any dtype, byte order and memory
        layout are accepted (e.g. big-endian data straight from ``astropy.io.fits``).
        The input is never modified.
    sigclip : float
        The detection limit for cosmic rays (sigma). Default is 4.5.
    sigfrac : float
        The fractional detection limit for neighboring pixels. Default is 0.5.
    objlim : float
        The contrast limit between CR and underlying objects. Default is 1.0.
    niter : int
        The number of iterations to perform. Default is 1.
    gain : float
        The gain of the image in electrons/ADU. Default is 0.0, in which case the
        gain is estimated from the image at every iteration (see Notes).
    readnoise : float
        The read noise of the image in electrons. Default is 0.0.
    device : str
        The device to use for computation, e.g. "cpu", "cuda" (NVIDIA GPU) or "mps"
        (Apple Silicon GPU). Default is "cpu".
    cpu_threads : int | None
        Number of CPU threads to use when ``device="cpu"``. The limit applies only
        for the duration of the call: the thread settings of torch and of the other
        OpenMP/BLAS thread pools in the process are restored afterwards, and no
        environment variables are changed. Default is None, which uses the current
        settings of the process (e.g. ``torch.get_num_threads()``).
    use_cpp : bool | None
        Whether to use the optional C++ median filter on the CPU. The extension is not
        part of the PyPI wheel; it has to be built from source (see the installation
        instructions). Default is None, which uses the extension if it is available
        and the torch median filter otherwise. True does the same but emits a warning
        (once per session) if the extension is not available. False always uses the
        torch median filter. Ignored when running on a GPU.
    verbose : bool
        Print iteration progress. Default is False.
    rss_debug : bool
        Print RSS memory at key steps. Default is False.

    Returns
    -------
        np.ndarray
            The image with cosmic rays removed, as float32.
        np.ndarray
            The boolean mask indicating the cosmic rays.

    Raises
    ------
    ValueError
        If the image is not 2D, if it contains no finite pixels, or if the gain
        cannot be estimated.

    Notes
    -----
    If the gain is set to zero (or not provided), it is estimated at every iteration
    from the image cleaned so far, assuming sky-dominated noise and Poisson
    statistics: ``gain = sky / sigma**2``, where ``sky`` is the sigma-clipped median of
    the image and ``sigma`` its robust scatter around a 7x7 median. This only works if
    the image still contains its sky background. For background-subtracted images the
    estimate fails with a ``ValueError`` (or is meaningless), so the gain has to be
    given explicitly; the ``skyval`` and ``statsec`` parameters of the original IRAF
    script are not implemented.

    All computations are done in single precision: the input is cast to float32 and
    the cleaned image is returned as float32, whatever the input dtype (including
    float64).

    Non-finite pixels (NaN and +/-inf, e.g. bad pixels flagged in a reduced frame) are
    ignored: they are excluded from the gain estimate, are never flagged as cosmic
    rays, and are returned unchanged in the cleaned image. Internally they are
    replaced by the median of the finite pixels so that they do not affect the
    detection or the repair of neighbouring pixels.

    Flagged pixels are repaired with a 5x5 median in which flagged pixels are treated
    as very high values, so that a repaired pixel takes the value of one of its
    unflagged neighbours. When more than half of the 5x5 window is flagged (the
    interior of a large cosmic ray hit) this median is not defined. Such a pixel is
    kept marked as unrepaired between iterations and, in the returned image, is set to
    the median of the unflagged pixels in its 5x5 window (the window is grown until it
    contains at least one unflagged pixel). The returned image therefore never
    contains placeholder values: every repaired pixel is the value of an unflagged
    pixel of the input image.

    Of the five median filters of an iteration, three are only read at a few pixels:
    the two that build the fine structure image are read at the candidate pixels, and
    the one used for the repair at the flagged pixels. They are evaluated at those
    pixels only, which gives exactly the same result as filtering the whole image and
    is much faster. If the pixels make up more than 10% of the image, the whole image
    is filtered instead. The fraction can be changed with the environment variable
    ``DFCOSMIC_SPARSE_MEDIAN_MAX_FRACTION``; 0 always filters the whole image.

    Performance tips:

    - Provide the gain if it is known, to avoid the cost of estimating it at every
      iteration.
    - Use ``niter=1`` for faster processing, at the cost of potentially detecting
      fewer cosmic rays.
    - On the CPU, build the optional C++ median filter (see the installation
      instructions); it is used automatically once available.
    - For the best performance use a GPU: ``device="cuda"``, or ``device="mps"`` on
      Apple Silicon.

    Examples
    --------
    Clean a synthetic sky frame with a few cosmic ray hits on the CPU:

    >>> import numpy as np
    >>> from dfcosmic import lacosmic
    >>> rng = np.random.default_rng(0)
    >>> image = rng.normal(200, 15, (100, 100)).astype(np.float32)
    >>> image[[20, 50, 80], [30, 60, 10]] += 2000
    >>> clean, mask = lacosmic(image, gain=1.0, readnoise=5.0)
    >>> clean.shape, mask.dtype
    ((100, 100), dtype('bool'))
    >>> bool(mask[20, 30])
    True

    With data from a FITS file (requires ``astropy``):

    >>> from astropy.io import fits  # doctest: +SKIP
    >>> image = fits.getdata("image.fits")  # doctest: +SKIP
    >>> clean, mask = lacosmic(  # doctest: +SKIP
    ...     image, sigclip=4.5, sigfrac=0.3, objlim=4, gain=7, readnoise=5, niter=4
    ... )
    """

    device = torch.device(device)

    want_cpp_median = use_cpp is not False and device.type == "cpu"
    use_cpp_median = want_cpp_median and cpp_median_available()
    if use_cpp and want_cpp_median and not use_cpp_median:
        warn_cpp_median_unavailable()

    # Limit the CPU threads for the duration of this call only. torch's own pool is
    # set (and restored in the finally block below) through torch; threadpoolctl
    # covers the other OpenMP/BLAS pools in the process, including the one used by
    # the C++ median filter, and restores them when the context exits.
    cpu_thread_ctx = nullcontext()
    prev_torch_threads = None
    if device.type == "cpu" and cpu_threads is not None:
        if cpu_threads < 1:
            raise ValueError(f"cpu_threads must be at least 1, but got {cpu_threads}")
        prev_torch_threads = torch.get_num_threads()
        torch.set_num_threads(cpu_threads)
        cpu_thread_ctx = threadpool_limits(limits=cpu_threads)

    prev_rss_env = os.environ.get("DFCOSMIC_RSS_DEBUG")
    if rss_debug:
        os.environ["DFCOSMIC_RSS_DEBUG"] = "1"

    try:
        with cpu_thread_ctx:
            if rss_debug:
                _log_rss("start")
            # Move/cast to torch
            if isinstance(image, torch.Tensor):
                image_t = image.to(device).float().contiguous()
            else:
                # torch.from_numpy rejects non-native byte order (FITS data is
                # big-endian) and negative strides, so normalise the array first.
                image_np = np.ascontiguousarray(image, dtype=np.float32)
                image_t = torch.from_numpy(image_np).to(device).contiguous()
                del image_np
            if image_t.ndim != 2:
                raise ValueError(
                    "image must be a 2D array, but got an array with "
                    f"{image_t.ndim} dimension(s) and shape {tuple(image_t.shape)}. "
                    "Process each 2D frame separately."
                )
            if rss_debug:
                _log_rss("after input cast/to(device)")

            block_size_tuple, block_size_tensor, laplacian_kernel, strel = _get_kernels(
                device, image_t.dtype
            )
            gkernel = torch.ones((3, 3), dtype=image_t.dtype, device=device)

            clean_image = image_t.clone()
            del image_t
            if rss_debug:
                _log_rss("after clean_image clone")

            # Non-finite pixels (NaN/inf) would poison the gain estimate and every
            # median window they fall in. Replace them with the median of the finite
            # pixels while running, and restore them in the output.
            bad_mask = ~torch.isfinite(clean_image)
            if bad_mask.any():
                good_values = clean_image[~bad_mask]
                if good_values.numel() == 0:
                    raise ValueError("image contains no finite pixels")
                bad_values = clean_image[bad_mask]
                clean_image[bad_mask] = good_values.median()
                del good_values
                if verbose:
                    print(f"Ignoring {bad_values.numel()} non-finite pixels")
            else:
                bad_mask = None

            # Placeholder for flagged pixels in the repair median. It is fixed for
            # the whole run and only ever lives in the working image: pixels still
            # holding it at the end are filled in before returning.
            sentinel = min(
                clean_image.abs().max().item() * 1e4 + 1e6,
                torch.finfo(clean_image.dtype).max,
            )
            # Round to the working precision so it can be compared exactly later
            sentinel = torch.tensor(sentinel, dtype=clean_image.dtype).item()

            final_crmask = torch.zeros(
                clean_image.shape, dtype=torch.bool, device=device
            )
            if rss_debug:
                _log_rss("after final_crmask allocation")
            if device.type == "cpu":
                median_filter_fn = (
                    median_filter_cpp_torch if use_cpp_median else median_filter_torch
                )
            else:
                median_filter_fn = median_filter_torch

            with torch.no_grad():
                for iteration in range(niter):
                    iter_start = perf_counter()
                    if verbose:
                        print("")
                        print(f"{'_' * 31} Iteration {iteration + 1} {'_' * 35}")
                        print("")
                    if rss_debug:
                        _log_rss(f"iter {iteration + 1} start")

                    # Step 0: Gain estimation (if requested). As in the original
                    # IRAF script, the estimate is repeated at every iteration on the
                    # image cleaned so far.
                    if gain > 0:
                        usegain = gain
                    else:
                        if verbose and iteration == 0:
                            print("Trying to determine gain automatically:")
                        elif verbose:
                            print("Improving gain estimate:")

                        # Leave out non-finite input pixels and pixels that could
                        # not be repaired in the previous iteration.
                        exclude = bad_mask
                        if iteration > 0:
                            unrepaired = clean_image >= sentinel
                            if unrepaired.any():
                                exclude = (
                                    unrepaired
                                    if bad_mask is None
                                    else unrepaired | bad_mask
                                )
                            del unrepaired

                        sky_level = sigma_clip_pytorch(
                            clean_image if exclude is None else clean_image[~exclude],
                            sigma=5,
                            maxiters=10,
                        )[1]["median"]
                        med7 = median_filter_fn(clean_image, kernel_size=7)
                        residuals = clean_image - med7
                        del med7
                        abs_residuals = torch.abs(residuals)
                        del residuals
                        if exclude is not None:
                            abs_residuals = abs_residuals[~exclude]
                        del exclude
                        mad = sigma_clip_pytorch(abs_residuals, sigma=5, maxiters=10)[
                            1
                        ]["median"]
                        del abs_residuals
                        sig = 1.48 * mad

                        if sig == 0 or not math.isfinite(sig):
                            raise ValueError(
                                "Gain determination failed - provide the gain manually "
                                "(this is required for background-subtracted images). "
                                f"Sky level: {sky_level:.2f}, Sigma: {sig:.2f}"
                            )
                        usegain = sky_level / (sig**2)
                        if usegain <= 0 or not math.isfinite(usegain):
                            raise ValueError(
                                "Gain determination failed - provide the gain manually "
                                "(this is required for background-subtracted images). "
                                f"Sky level: {sky_level:.2f}, Sigma: {sig:.2f}"
                            )
                        if verbose:
                            print(f"  Approximate sky level = {sky_level:.2f} ADU")
                            print(f"  Sigma of sky = {sig:.2f}")
                            print(f"  Estimated gain = {usegain:.2f}")
                            print("")
                        if rss_debug:
                            _log_rss(f"iter {iteration + 1} after gain estimation")

                    if verbose:
                        print("Convolving image with Laplacian kernel")
                        print("")

                    # Step 1: Laplacian detection
                    temp = laplacian_pool_chunked(
                        clean_image,
                        block_size_tensor,
                        laplacian_kernel,
                    )
                    if rss_debug:
                        _log_rss(f"iter {iteration + 1} after laplacian detection")

                    if verbose:
                        print("Creating noise model using:")
                        print(f"  gain = {usegain:.2f} electrons/ADU")
                        print(f"  readnoise = {readnoise:.2f} electrons")
                        print("")

                    # Step 2: Noise model
                    med5 = median_filter_fn(clean_image, kernel_size=5)
                    med5.clamp_(min=1e-4)
                    noise = torch.sqrt(med5 * usegain + readnoise**2) / usegain
                    del med5
                    if rss_debug:
                        _log_rss(f"iter {iteration + 1} after noise model")

                    # Step 3: Significance map
                    temp /= noise
                    sigmap = temp
                    sigmap /= 2.0
                    sigmap -= median_filter_fn(sigmap, kernel_size=5)
                    if rss_debug:
                        _log_rss(f"iter {iteration + 1} after significance map")

                    if verbose:
                        print("Selecting candidate cosmic rays")
                        print(f"  sigma limit = {sigclip:.1f}")
                        print("")

                    # Step 4: Initial CR candidates
                    candidates = sigmap >= sigclip

                    if verbose:
                        print("Removing suspected compact bright objects (e.g. stars)")
                        print(
                            f"  selecting cosmic rays > {objlim:.1f} times object flux"
                        )
                        print("")

                    # Step 5: Reject objects (fine structure). The fine structure
                    # image is only read at the candidates.
                    n_candidates = int(candidates.sum())
                    if n_candidates == 0:
                        firstsel = torch.zeros_like(sigmap)
                    elif use_sparse_median(n_candidates, clean_image):
                        # Few candidates: evaluate the 7x7 median only at them. The
                        # values, and therefore the result, are the same as when
                        # filtering the full image.
                        ys, xs = torch.nonzero(candidates, as_tuple=True)
                        if use_sparse_median(49 * n_candidates, clean_image):
                            # So few that the 3x3 median, which is needed in their
                            # 7x7 windows, is not worth computing everywhere either
                            med3, med7 = median_of_median_at(clean_image, ys, xs, 3, 7)
                        else:
                            full_med3 = median_filter_fn(clean_image, kernel_size=3)
                            med3 = full_med3[ys, xs]
                            med7 = median_filter_at(full_med3, ys, xs, kernel_size=7)
                            del full_med3
                        fine = ((med3 - med7) / noise[ys, xs]).clamp_(min=0.01)
                        keep = sigmap[ys, xs] / fine >= objlim
                        firstsel = torch.zeros_like(sigmap)
                        firstsel[ys[keep], xs[keep]] = 1.0
                        del ys, xs, med3, med7, fine, keep
                    else:
                        firstsel = candidates.to(sigmap.dtype)
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
                    del candidates
                    if rss_debug:
                        _log_rss(f"iter {iteration + 1} after star rejection")

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
                    if rss_debug:
                        _log_rss(f"iter {iteration + 1} after first grow")

                    # Second grow: threshold at sigcliplow
                    finalsel = convolve(gfirstsel, gkernel)
                    del gfirstsel
                    finalsel = (finalsel > 0.5).to(sigmap.dtype)
                    finalsel = finalsel * sigmap
                    finalsel = (finalsel > sigcliplow).to(sigmap.dtype)
                    del sigmap
                    if bad_mask is not None:
                        finalsel[bad_mask] = 0
                    if rss_debug:
                        _log_rss(f"iter {iteration + 1} after second grow")

                    # Count only NEW cosmic rays found in this iteration
                    new_crs = (~final_crmask).to(finalsel.dtype) * finalsel
                    npix = new_crs.sum().item()

                    del new_crs

                    if npix == 0:
                        if finalsel.sum().item() > 0:
                            # Update mask even if no new CRs (edge case)
                            final_crmask |= finalsel.bool()
                        if rss_debug:
                            _log_rss(f"iter {iteration + 1} early exit")
                            print(
                                f"[rss] iter {iteration + 1} elapsed: "
                                f"{perf_counter() - iter_start:.2f}s"
                            )
                        break

                    # Step 7: Clean flagged pixels with 5x5 median replacement
                    final_crmask |= finalsel.bool()

                    if verbose:
                        print(
                            f"{int(npix)} cosmic rays found in iteration {iteration + 1}"
                        )
                        print("")

                    # Create cleaned output image using 5x5 median. Flagged pixels
                    # sort last, so the median is the value of an unflagged neighbour
                    # unless more than half of the window is flagged. In that case it
                    # is the sentinel itself: the pixel stays marked as unrepaired for
                    # the next iteration and is filled in after the last one.
                    if use_sparse_median(int(final_crmask.sum()), clean_image):
                        # The median is only used at the flagged pixels, so it is
                        # only evaluated there.
                        ys, xs = torch.nonzero(final_crmask, as_tuple=True)
                        clean_image[ys, xs] = median_filter_at(
                            clean_image,
                            ys,
                            xs,
                            kernel_size=5,
                            replace=final_crmask,
                            value=sentinel,
                        )
                        del ys, xs
                    else:
                        tmp = clean_image.clone()
                        tmp[final_crmask] = sentinel
                        tmp = median_filter_fn(tmp, kernel_size=5)
                        # Only use the median at CR locations
                        clean_image[final_crmask] = tmp[final_crmask]
                        del tmp
                    if rss_debug:
                        _log_rss(f"iter {iteration + 1} after repair")
                        print(
                            f"[rss] iter {iteration + 1} elapsed: "
                            f"{perf_counter() - iter_start:.2f}s"
                        )

                    del finalsel, noise

        # Repair the pixels whose 5x5 window was mostly flagged, so that the
        # sentinel never reaches the output.
        unrepaired = final_crmask & (clean_image >= sentinel)
        if unrepaired.any():
            fill_from_unflagged_neighbors(clean_image, unrepaired, final_crmask)
        del unrepaired
        if bad_mask is not None:
            clean_image[bad_mask] = bad_values
        if rss_debug:
            _log_rss("before cpu().numpy() return")
        return clean_image.cpu().numpy(), final_crmask.cpu().numpy()
    finally:
        if prev_torch_threads is not None:
            torch.set_num_threads(prev_torch_threads)
        if rss_debug:
            if prev_rss_env is None:
                os.environ.pop("DFCOSMIC_RSS_DEBUG", None)
            else:
                os.environ["DFCOSMIC_RSS_DEBUG"] = prev_rss_env
