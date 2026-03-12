import os

import torch
import torch.nn.functional as F

_DISABLE_CPP = os.environ.get("DFCOSMIC_DISABLE_CPP", "").lower() in {
    "1",
    "true",
    "yes",
}
_DEFAULT_CONVOLVE_DIRECT_MAX_NUMEL = 262_144
_DEFAULT_MEMORY_BUDGET_MARGIN = 0.8


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


try:
    import median_filter_cpp

    _CPP_MEDIAN_AVAILABLE = not _DISABLE_CPP
except Exception:
    _CPP_MEDIAN_AVAILABLE = False

try:
    _CPP_DILATION_AVAILABLE = not _DISABLE_CPP
except Exception:
    _CPP_DILATION_AVAILABLE = False


def _process_block_inputs(
    data: torch.Tensor, block_size: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = torch.atleast_1d(block_size)

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
    data: torch.Tensor, block_size: int | list[int], conserve_sum: bool = False
) -> torch.Tensor:
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
        for i in range(data.ndim):
            data = data.repeat_interleave(int(block_size[i]), dim=i)

    if conserve_sum:
        output = output / torch.prod(block_size).float()

    return output if data.ndim == 2 else data


def laplacian_pool_chunked(
    image: torch.Tensor,
    block_size: torch.Tensor,
    laplacian_kernel: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Exact chunked implementation of the LA Cosmic subsampled Laplacian step:
    2x block replication -> Laplacian convolution -> clamp(min=0) -> 2x2 average pool.
    """
    _log_rss("utils.laplacian_pool_chunked start")
    h, w = image.shape
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

        image_chunk = image[src_y0:src_y1, :]
        replicated = block_replicate_torch(image_chunk, block_size, conserve_sum=False)
        conv = convolve(replicated, laplacian_kernel)
        conv.clamp_(min=0)
        pooled = F.avg_pool2d(
            conv.unsqueeze(0).unsqueeze(0),
            kernel_size=(int(block_size[0]), int(block_size[1])),
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
        raise RuntimeError("median_filter_cpp extension is not available")
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
    return _CPP_MEDIAN_AVAILABLE
