import os
import warnings

import torch
import torch.nn.functional as F

_DISABLE_CPP = os.environ.get("DFCOSMIC_DISABLE_CPP", "").lower() in {
    "1",
    "true",
    "yes",
}
_DEFAULT_CONVOLVE_DIRECT_MAX_NUMEL = 262_144
_DEFAULT_MEMORY_BUDGET_MARGIN = 0.8
# A median filter whose result is only read at a few pixels is evaluated at those
# pixels instead of on the whole image, as long as they make up at most this fraction
# of the image. Evaluating at pixels is several times faster up to this fraction on
# one and on many threads; far above it, filtering the whole image wins.
_DEFAULT_SPARSE_MEDIAN_MAX_FRACTION = 0.1
_SPARSE_MEDIAN_CHUNK_VALUES = 2_000_000


def _rss_debug_enabled() -> bool:
    return os.environ.get("DFCOSMIC_RSS_DEBUG", "").lower() in {"1", "true", "yes"}


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
    if not _rss_debug_enabled():
        return
    rss_mb = _current_rss_mb()
    if rss_mb is None:
        print(f"[rss] {label}: unavailable")
    else:
        print(f"[rss] {label}: {rss_mb:.1f} MiB")


def _convolve_direct_max_numel() -> int:
    raw = os.environ.get("DFCOSMIC_CONVOLVE_DIRECT_MAX_NUMEL")
    if raw is None:
        return _DEFAULT_CONVOLVE_DIRECT_MAX_NUMEL
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_CONVOLVE_DIRECT_MAX_NUMEL
    return max(0, value)


def _sparse_median_max_fraction() -> float:
    raw = os.environ.get("DFCOSMIC_SPARSE_MEDIAN_MAX_FRACTION")
    if raw is None:
        return _DEFAULT_SPARSE_MEDIAN_MAX_FRACTION
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_SPARSE_MEDIAN_MAX_FRACTION
    return min(1.0, max(0.0, value))


def use_sparse_median(n_pixels: int, image: torch.Tensor) -> bool:
    """Whether a median that is needed at ``n_pixels`` pixels only is evaluated there."""
    return n_pixels <= _sparse_median_max_fraction() * image.numel()


def _memory_budget_mb() -> float | None:
    raw = os.environ.get("DFCOSMIC_MAX_MEMORY_MB")
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _budgeted_chunk_rows(
    *,
    width: int,
    dtype: torch.dtype,
    default_rows: int,
    bytes_per_row_multiplier: float,
) -> int:
    budget_mb = _memory_budget_mb()
    if budget_mb is None:
        return max(1, int(default_rows))

    element_size = torch.tensor((), dtype=dtype).element_size()
    bytes_per_row = max(1, int(width * element_size * bytes_per_row_multiplier))
    usable_budget = int(budget_mb * 1024 * 1024 * _DEFAULT_MEMORY_BUDGET_MARGIN)
    budget_rows = max(1, usable_budget // bytes_per_row)
    return max(1, min(int(default_rows), budget_rows))


def _median_filter_chunk_rows(
    image: torch.Tensor, kernel_size: int, default_rows: int = 128
) -> int:
    # Unfold materializes k*k values per output pixel, so this path needs a much
    # tighter row budget than convolution on small CPU runners.
    return _budgeted_chunk_rows(
        width=image.shape[1],
        dtype=image.dtype,
        default_rows=default_rows,
        bytes_per_row_multiplier=float(kernel_size * kernel_size + 3),
    )


_CPP_BUILD_HINT = (
    "The C++ median filter is an opt-in source build: install torch and a C++ "
    "compiler, then run `pip install --no-build-isolation .` from a dfcosmic checkout."
)

# The extension is optional and tied to the torch version it was compiled against,
# so both a missing module and a failed load (e.g. after a torch upgrade) end up here.
try:
    import dfcosmic._median_filter_cpp as median_filter_cpp

    _CPP_MEDIAN_AVAILABLE = not _DISABLE_CPP
    _CPP_MEDIAN_UNAVAILABLE_REASON = (
        "it is disabled by the DFCOSMIC_DISABLE_CPP environment variable."
        if _DISABLE_CPP
        else None
    )
except ModuleNotFoundError as exc:
    _CPP_MEDIAN_AVAILABLE = False
    if exc.name == "dfcosmic._median_filter_cpp":
        _CPP_MEDIAN_UNAVAILABLE_REASON = f"it was not built. {_CPP_BUILD_HINT}"
    else:
        _CPP_MEDIAN_UNAVAILABLE_REASON = f"it failed to load ({exc}). {_CPP_BUILD_HINT}"
except Exception as exc:
    _CPP_MEDIAN_AVAILABLE = False
    _CPP_MEDIAN_UNAVAILABLE_REASON = (
        f"it failed to load ({exc}). It must be rebuilt against the installed "
        f"torch version. {_CPP_BUILD_HINT}"
    )

_CPP_FALLBACK_WARNED = False


def _process_block_inputs(
    data: torch.Tensor, block_size: int | list[int] | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = torch.atleast_1d(torch.as_tensor(block_size))

    if torch.any(block_size <= 0):
        raise ValueError("block_size elements must be strictly positive")

    if data.ndim > 1 and len(block_size) == 1:
        block_size = torch.repeat_interleave(block_size, data.ndim)

    if len(block_size) != data.ndim:
        raise ValueError(
            "block_size must be a scalar or have the same "
            "length as the number of data dimensions"
        )

    if not torch.all(block_size == torch.floor(block_size)):
        raise ValueError("block_size elements must be integers")

    block_size_int = block_size.long()
    return data, block_size_int


def block_replicate_torch(
    data: torch.Tensor,
    block_size: int | list[int] | torch.Tensor,
    conserve_sum: bool = False,
) -> torch.Tensor:
    """
    Upsample an array by repeating every element ``block_size`` times along each axis.

    ``block_size`` is a single number, used for every axis, or one number per axis.
    If ``conserve_sum`` is True, the result is divided by the number of copies of each
    element, so that its sum is the sum of the input.
    """
    data, block_size = _process_block_inputs(data, block_size)

    if data.ndim == 2:
        h, w = data.shape
        bh, bw = int(block_size[0]), int(block_size[1])

        chunk_size = 128
        # Pre-allocate output tensor
        output = torch.empty(h * bh, w * bw, dtype=data.dtype, device=data.device)
        # Process in chunks
        for i in range(0, h, chunk_size):
            i_end = min(i + chunk_size, h)
            chunk = data[i:i_end, :]
            # Replicate this chunk
            chunk_rep = chunk.repeat_interleave(bh, dim=0).repeat_interleave(bw, dim=1)
            # Place in output
            output[i * bh : i_end * bh, :] = chunk_rep
            # Free memory
            del chunk_rep

    else:
        output = data
        for i in range(data.ndim):
            output = output.repeat_interleave(int(block_size[i]), dim=i)

    if conserve_sum:
        output = output / torch.prod(block_size).float()

    return output


def laplacian_pool_chunked(
    image: torch.Tensor,
    block_size: torch.Tensor,
    laplacian_kernel: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Exact chunked implementation of the LA Cosmic subsampled Laplacian step:
    2x block replication -> Laplacian convolution -> clamp(min=0) -> 2x2 average pool.

    At the edges, the image is extended by its nearest pixel for the convolution, as
    the IRAF ``convolve`` task does by default (``boundary="nearest"``).
    """
    _log_rss("utils.laplacian_pool_chunked start")
    h, w = image.shape
    block_h, block_w = int(block_size[0]), int(block_size[1])
    # Pixels by which the image is extended, enough to cover half the kernel after
    # the replication
    edge_h = -(-(laplacian_kernel.shape[0] // 2) // block_h)
    edge_w = -(-(laplacian_kernel.shape[1] // 2) // block_w)
    output = torch.empty_like(image)
    # Approximate workspace per core row: input slice + replicated slice +
    # convolution output + pooled output. Keep this conservative for small runners.
    core_chunk_rows = _budgeted_chunk_rows(
        width=w,
        dtype=image.dtype,
        default_rows=chunk_size,
        bytes_per_row_multiplier=12.0,
    )

    for i in range(0, h, core_chunk_rows):
        i_end = min(i + core_chunk_rows, h)
        src_y0 = max(0, i - 1)
        src_y1 = min(h, i_end + 1)
        core_offset = i - src_y0
        core_rows = i_end - i

        # Extend the image by its nearest pixel instead of by zeros, which would look
        # like an edge to the Laplacian. This is done before the replication, where
        # the copy is small; the extension is cut off again after the convolution.
        # Inside the image it only changes the rows next to the chunk, which are
        # dropped below.
        image_chunk = F.pad(
            image[src_y0:src_y1, :].unsqueeze(0).unsqueeze(0),
            (edge_w, edge_w, edge_h, edge_h),
            mode="replicate",
        )[0, 0]
        replicated = block_replicate_torch(image_chunk, block_size, conserve_sum=False)
        del image_chunk
        conv = convolve(replicated, laplacian_kernel)
        conv = conv[
            edge_h * block_h : conv.shape[0] - edge_h * block_h,
            edge_w * block_w : conv.shape[1] - edge_w * block_w,
        ]
        conv.clamp_(min=0)
        pooled = F.avg_pool2d(
            conv.unsqueeze(0).unsqueeze(0), kernel_size=(block_h, block_w)
        )[0, 0]

        output[i:i_end, :] = pooled[core_offset : core_offset + core_rows, :]
        if i == 0:
            _log_rss("utils.laplacian_pool_chunked after first chunk")
        del replicated, conv, pooled

    _log_rss("utils.laplacian_pool_chunked before return")
    return output


def convolve_chunked(
    image: torch.Tensor, kernel: torch.Tensor, chunk_size: int = 256
) -> torch.Tensor:
    """Memory-efficient chunked convolution"""
    _log_rss("utils.convolve_chunked start")
    h, w = image.shape
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    chunk_rows = _budgeted_chunk_rows(
        width=w + 2 * pad_w,
        dtype=image.dtype,
        default_rows=chunk_size,
        bytes_per_row_multiplier=4.0,
    )

    # Pre-allocate output
    output = torch.empty_like(image)

    # Prepare kernel
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)

    # Process in chunks
    for i in range(0, h, chunk_rows):
        i_end = min(i + chunk_rows, h)
        src_y0 = max(0, i - pad_h)
        src_y1 = min(h, i_end + pad_h)
        pad_top = max(0, pad_h - i)
        pad_bottom = max(0, i_end + pad_h - h)

        chunk = image[src_y0:src_y1, :].unsqueeze(0).unsqueeze(0)
        chunk = F.pad(
            chunk,
            (pad_w, pad_w, pad_top, pad_bottom),
            mode="constant",
            value=0,
        )
        if i == 0:
            _log_rss("utils.convolve_chunked after first chunk pad")

        # Convolve
        result = F.conv2d(chunk, kernel_4d, padding=0)
        output[i:i_end, :] = result.squeeze(0).squeeze(0)

        del chunk, result

    _log_rss("utils.convolve_chunked before return")
    return output


def convolve(
    image: torch.Tensor, kernel: torch.Tensor, chunk_size: int = 512
) -> torch.Tensor:
    """
    Convolve a 2D image with a 2D kernel, with zeros beyond the edges of the image.

    Large images on the CPU are convolved in chunks of ``chunk_size`` rows to limit
    memory use (see ``convolve_chunked``); the result is the same.
    """
    _log_rss("utils.convolve start")
    # oneDNN/MKL-backed CPU conv2d can request a large temporary workspace on
    # memory-constrained runners, so keep the direct path limited and tunable.
    use_direct = (
        image.device.type != "cpu" or image.numel() <= _convolve_direct_max_numel()
    )
    if use_direct:
        image_4d = image.unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        pad_h = kernel.shape[0] // 2
        pad_w = kernel.shape[1] // 2
        result = F.conv2d(image_4d, kernel_4d, padding=(pad_h, pad_w))
        _log_rss("utils.convolve small before return")
        return result.squeeze(0).squeeze(0)

    return convolve_chunked(image, kernel, chunk_size=chunk_size)


def median_filter_torch(
    image: torch.Tensor,
    kernel_size: int = 3,
    zloreject: float | None = None,
) -> torch.Tensor:
    """
    Median filter with optional IRAF-like zloreject.

    If zloreject is not None, values < zloreject are ignored when computing the median
    (mimics IRAF median(..., zloreject=...)).
    """
    _log_rss(f"utils.median_filter_torch k={kernel_size} start")
    h, w = image.shape
    pad = kernel_size // 2
    chunk_rows = _median_filter_chunk_rows(image, kernel_size)
    filtered = torch.empty_like(image)

    for i in range(0, h, chunk_rows):
        i_end = min(i + chunk_rows, h)
        src_y0 = max(0, i - pad)
        src_y1 = min(h, i_end + pad)
        pad_top = max(0, pad - i)
        pad_bottom = max(0, i_end + pad - h)

        chunk = image[src_y0:src_y1, :].unsqueeze(0).unsqueeze(0)
        chunk = F.pad(
            chunk,
            (pad, pad, pad_top, pad_bottom),
            mode="replicate",
        )
        if i == 0:
            _log_rss(f"utils.median_filter_torch k={kernel_size} after pad")

        unfolded = F.unfold(chunk, kernel_size, stride=1)
        if i == 0:
            _log_rss(f"utils.median_filter_torch k={kernel_size} after unfold")

        unfolded = unfolded.view(1, kernel_size * kernel_size, i_end - i, w)
        if zloreject is not None:
            valid = unfolded >= zloreject
            masked = unfolded.masked_fill(~valid, torch.inf)
            chunk_filtered, _ = masked.median(dim=1)
            fallback, _ = unfolded.median(dim=1)
            has_valid = valid.any(dim=1)
            chunk_filtered = torch.where(has_valid, chunk_filtered, fallback)
        else:
            chunk_filtered, _ = unfolded.median(dim=1)

        filtered[i:i_end, :] = chunk_filtered.squeeze(0)
        del chunk, unfolded, chunk_filtered

    _log_rss(f"utils.median_filter_torch k={kernel_size} before return")
    return filtered


def _window_indices(
    ys: torch.Tensor, xs: torch.Tensor, kernel_size: int, shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Row and column indices of the ``kernel_size`` window around each pixel, with shapes
    (n, k, 1) and (n, 1, k). Indices beyond the image are moved to its edge, which is
    the same as the replicate padding of the median filter.
    """
    radius = kernel_size // 2
    offsets = torch.arange(-radius, radius + 1, device=ys.device)
    yy = (ys[:, None, None] + offsets[None, :, None]).clamp(0, shape[0] - 1)
    xx = (xs[:, None, None] + offsets[None, None, :]).clamp(0, shape[1] - 1)
    return yy, xx


def median_filter_at(
    image: torch.Tensor,
    ys: torch.Tensor,
    xs: torch.Tensor,
    kernel_size: int = 3,
    replace: torch.Tensor | None = None,
    value: float | None = None,
) -> torch.Tensor:
    """
    Median filter evaluated only at the pixels ``(ys, xs)``.

    Returns the same values as ``median_filter_torch(image, kernel_size)[ys, xs]``, at
    a cost proportional to the number of pixels instead of the size of the image. If
    ``replace`` (a boolean image) is given, the pixels where it is True enter the
    medians as ``value``.
    """
    window_size = kernel_size * kernel_size
    result = torch.empty(ys.numel(), dtype=image.dtype, device=image.device)
    step = max(1, _SPARSE_MEDIAN_CHUNK_VALUES // window_size)
    for start in range(0, ys.numel(), step):
        stop = start + step
        yy, xx = _window_indices(
            ys[start:stop], xs[start:stop], kernel_size, image.shape
        )
        window = image[yy, xx].reshape(-1, window_size)
        if replace is not None:
            window.masked_fill_(replace[yy, xx].reshape(-1, window_size), value)
        result[start:stop] = window.median(dim=1).values
    return result


def median_of_median_at(
    image: torch.Tensor,
    ys: torch.Tensor,
    xs: torch.Tensor,
    inner_size: int = 3,
    outer_size: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Two nested median filters evaluated only at the pixels ``(ys, xs)``.

    Returns the values at those pixels of ``inner = median_filter(image, inner_size)``
    and of ``median_filter(inner, outer_size)``. The inner filter is only evaluated in
    the ``outer_size`` windows of the pixels, never on the whole image.
    """
    window_size = outer_size * outer_size
    inner = torch.empty(ys.numel(), dtype=image.dtype, device=image.device)
    outer = torch.empty_like(inner)
    step = max(1, _SPARSE_MEDIAN_CHUNK_VALUES // (window_size * inner_size**2))
    for start in range(0, ys.numel(), step):
        stop = start + step
        yy, xx = _window_indices(
            ys[start:stop], xs[start:stop], outer_size, image.shape
        )
        yy, xx = torch.broadcast_tensors(yy, xx)
        window = median_filter_at(
            image, yy.reshape(-1), xx.reshape(-1), kernel_size=inner_size
        ).reshape(-1, window_size)
        # The middle of the window is the pixel itself
        inner[start:stop] = window[:, window_size // 2]
        outer[start:stop] = window.median(dim=1).values
    return inner, outer


def fill_from_unflagged_neighbors(
    image: torch.Tensor,
    fill_mask: torch.Tensor,
    flagged_mask: torch.Tensor,
    kernel_size: int = 5,
) -> torch.Tensor:
    """
    Replace, in place, the pixels in ``fill_mask`` by the median of the pixels in
    their ``kernel_size`` window that are not in ``flagged_mask``.

    This is the repair rule for pixels where more than half of the window is flagged,
    for which a median over the whole window is not defined. If a window contains no
    unflagged pixel at all it is grown by one pixel on each side until it does. For an
    even number of unflagged pixels the lower of the two middle values is used, as in
    ``torch.median``. Image edges are handled by replication, as in the median filter.
    """
    h, w = image.shape
    ys, xs = torch.nonzero(fill_mask, as_tuple=True)
    values = torch.zeros(ys.shape, dtype=image.dtype, device=image.device)
    todo = torch.arange(ys.numel(), device=image.device)

    radius = kernel_size // 2
    while todo.numel() > 0 and radius < max(h, w):
        yy, xx = _window_indices(ys[todo], xs[todo], 2 * radius + 1, image.shape)
        window = image[yy, xx].reshape(todo.numel(), -1)
        flagged = flagged_mask[yy, xx].reshape(todo.numel(), -1)
        del yy, xx

        # Sort the unflagged values first, then pick their (lower) median
        n_good = (~flagged).sum(dim=1)
        window = window.masked_fill(flagged, torch.inf).sort(dim=1).values
        median_idx = ((n_good - 1) // 2).clamp(min=0)
        median = window.gather(1, median_idx[:, None])[:, 0]

        found = n_good > 0
        values[todo[found]] = median[found]
        todo = todo[~found]
        radius += 1

    image[ys, xs] = values
    return image


def median_filter_cpp_torch(
    image: torch.Tensor,
    kernel_size: int = 3,
    zloreject: float | None = None,
) -> torch.Tensor:
    """
    Fast CPU median filter using the C++ extension.

    NOTE: The current extension does not support zloreject. If zloreject is requested,
    we fall back to the torch implementation to preserve IRAF-parity behavior.
    """
    _log_rss(f"utils.median_filter_cpp_torch k={kernel_size} start")
    if zloreject is not None:
        return median_filter_torch(image, kernel_size=kernel_size, zloreject=zloreject)

    if not _CPP_MEDIAN_AVAILABLE:
        raise RuntimeError(
            "The C++ median filter extension is not available because "
            f"{_CPP_MEDIAN_UNAVAILABLE_REASON}"
        )
    if image.device.type != "cpu":
        raise ValueError("median_filter_cpp_torch requires a CPU tensor")
    if image.dtype != torch.float32:
        image = image.float()
    if not image.is_contiguous():
        image = image.contiguous()

    result = median_filter_cpp.median_filter_cpu(image, kernel_size)
    _log_rss(f"utils.median_filter_cpp_torch k={kernel_size} before return")
    return result


def sigma_clip_pytorch(
    data: torch.Tensor, sigma: tuple[float, float] | float = 3.0, maxiters: int = 10
) -> tuple[torch.Tensor, dict]:
    """
    Iteratively remove the values further than ``sigma`` standard deviations from the
    mean, until none is removed or ``maxiters`` iterations have been made.

    ``sigma`` is one number, or a (lower, upper) pair. Returns the remaining values as
    a 1D tensor, and a dictionary with their ``median``, ``mean`` and ``std``, the
    number of iterations made (``niter``) and the number of values left (``npix``).
    """
    if isinstance(sigma, (int, float)):
        sigma_low, sigma_high = sigma, sigma
    else:
        sigma_low, sigma_high = sigma

    data = data.flatten()

    for i in range(maxiters):
        mean_val = torch.mean(data)
        std_val = torch.std(data, unbiased=True)

        lower = mean_val - sigma_low * std_val
        upper = mean_val + sigma_high * std_val

        mask = (data >= lower) & (data <= upper)
        data_new = data[mask]

        if len(data_new) == len(data):
            break

        data = data_new

    stats = {
        "median": torch.median(data).item(),
        "mean": torch.mean(data).item(),
        "std": torch.std(data, unbiased=True).item(),
        "niter": i + 1,
        "npix": len(data),
    }

    return data, stats


def cpp_median_available() -> bool:
    """Whether the optional C++ median filter has been built and can be used."""
    return _CPP_MEDIAN_AVAILABLE


def warn_cpp_median_unavailable() -> None:
    """Warn, once per session, that the torch median filter is used instead of C++."""
    global _CPP_FALLBACK_WARNED
    if _CPP_MEDIAN_AVAILABLE or _CPP_FALLBACK_WARNED:
        return
    _CPP_FALLBACK_WARNED = True
    warnings.warn(
        "use_cpp=True was requested but the C++ median filter extension is not "
        f"available because {_CPP_MEDIAN_UNAVAILABLE_REASON} Falling back to the "
        "slower torch median filter; the results are identical.",
        RuntimeWarning,
        stacklevel=3,
    )
