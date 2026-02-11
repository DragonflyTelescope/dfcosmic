import os

import torch
import torch.nn.functional as F

_DISABLE_CPP = os.environ.get("DFCOSMIC_DISABLE_CPP", "").lower() in {
    "1",
    "true",
    "yes",
}

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
        data = data.repeat_interleave(block_size[0], dim=0).repeat_interleave(
            block_size[1], dim=1
        )
    else:
        for i in range(data.ndim):
            data = torch.repeat_interleave(data, block_size[i], dim=i)

    if conserve_sum:
        data = data / torch.prod(block_size)

    return data


def convolve(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    image_4d = image.unsqueeze(0).unsqueeze(0)
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)

    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2

    result = F.conv2d(image_4d, kernel_4d, padding=(pad_h, pad_w))
    return result.squeeze(0).squeeze(0)


def median_filter_torch(
    image: torch.Tensor,
    kernel_size: int = 3,
) -> torch.Tensor:
    """
    Median filter with optional IRAF-like zloreject.

    If zloreject is not None, values < zloreject are ignored when computing the median
    (mimics IRAF median(..., zloreject=...)).
    """
    h, w = image.shape
    pad = kernel_size // 2

    image_padded = F.pad(
        image.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode="replicate"
    )

    unfolded = F.unfold(image_padded, kernel_size, stride=1)  # (1, k*k, h*w)

    unfolded = unfolded.view(kernel_size * kernel_size, h, w)
    filtered, _ = unfolded.median(dim=0)
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

    return median_filter_cpp.median_filter_cpu(image, kernel_size)


def sigma_clip_pytorch(
    data: torch.Tensor, sigma: tuple[float, float] | float = 3.0, maxiters: int = 10
) -> tuple[torch.Tensor, dict]:
    if isinstance(sigma, (int, float)):
        sigma_low, sigma_high = sigma, sigma
    else:
        sigma_low, sigma_high = sigma

    data = data.flatten().clone()

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
