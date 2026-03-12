import os

import torch
import torch.nn.functional as F

_DISABLE_CPP = os.environ.get("DFCOSMIC_DISABLE_CPP", "").lower() in {
    "1",
    "true",
    "yes",
}


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


def convolve_chunked(
    image: torch.Tensor, kernel: torch.Tensor, chunk_size: int = 256
) -> torch.Tensor:
    """Memory-efficient chunked convolution"""
    _log_rss("utils.convolve_chunked start")
    h, w = image.shape
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2

    # Pre-allocate output
    output = torch.empty_like(image)

    # Prepare kernel
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)

    # Process in chunks
    for i in range(0, h, chunk_size):
        i_end = min(i + chunk_size, h)
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
    if image.numel() < 1000 * 1000:  # Small images, use direct method
        image_4d = image.unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        pad_h = kernel.shape[0] // 2
        pad_w = kernel.shape[1] // 2
        result = F.conv2d(image_4d, kernel_4d, padding=(pad_h, pad_w))
        _log_rss("utils.convolve small before return")
        return result.squeeze(0).squeeze(0)

    # Large images - use chunked processing
    h, w = image.shape
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2

    output = torch.empty_like(image)
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
    for i in range(0, h, chunk_size):
        i_end = min(i + chunk_size, h)
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
            _log_rss("utils.convolve after first chunk pad")
        result = F.conv2d(chunk, kernel_4d, padding=0)
        output[i:i_end, :] = result.squeeze(0).squeeze(0)
        del chunk, result

    _log_rss("utils.convolve before return")
    return output


def median_filter_torch(
    image: torch.Tensor,
    kernel_size: int = 3,
) -> torch.Tensor:
    """
    Median filter with optional IRAF-like zloreject.

    If zloreject is not None, values < zloreject are ignored when computing the median
    (mimics IRAF median(..., zloreject=...)).
    """
    _log_rss(f"utils.median_filter_torch k={kernel_size} start")
    h, w = image.shape
    pad = kernel_size // 2

    image_padded = F.pad(
        image.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode="replicate"
    )
    _log_rss(f"utils.median_filter_torch k={kernel_size} after pad")

    unfolded = F.unfold(image_padded, kernel_size, stride=1)  # (1, k*k, h*w)
    _log_rss(f"utils.median_filter_torch k={kernel_size} after unfold")

    unfolded = unfolded.view(kernel_size * kernel_size, h, w)
    filtered, _ = unfolded.median(dim=0)
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
