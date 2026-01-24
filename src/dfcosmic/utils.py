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
    import dilation_cpp

    _CPP_DILATION_AVAILABLE = not _DISABLE_CPP
except Exception:
    _CPP_DILATION_AVAILABLE = False


def _process_block_inputs(
    data: torch.Tensor, block_size: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function for block replication.
    """
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

    # Check if all elements are integers (e.g., 2.0 is OK, 2.5 is not)
    if not torch.all(block_size == torch.floor(block_size)):
        raise ValueError("block_size elements must be integers")

    block_size_int = block_size.long()
    return data, block_size_int


def block_replicate_torch(
    data: torch.Tensor, block_size: int | list[int], conserve_sum: bool = True
) -> torch.Tensor:
    """
    Upsample a data array by block replication.

    Parameters
    ----------
    data : array-like
        The data to be block replicated.

    block_size : int or array-like (int)
        The integer block size along each axis.  If ``block_size`` is a
        scalar and ``data`` has more than one dimension, then
        ``block_size`` will be used for for every axis.

    conserve_sum : bool, optional
        If `True` (the default) then the sum of the output
        block-replicated data will equal the sum of the input ``data``.

    Returns
    -------
    output : torch.Tensor
        The block-replicated data. Note that when ``conserve_sum`` is
        `True`, the dtype of the output array will be float.

    """
    data, block_size = _process_block_inputs(data, block_size)

    # Optimized version using repeat for 2D case (most common)
    if data.ndim == 2:
        # Use repeat which is faster than repeat_interleave for block replication
        data = data.repeat_interleave(block_size[0], dim=0).repeat_interleave(
            block_size[1], dim=1
        )
    else:
        # General case for any dimensionality
        for i in range(data.ndim):
            data = torch.repeat_interleave(data, block_size[i], dim=i)

    if conserve_sum:
        # in-place division can fail due to dtype casting rule
        data = data / torch.prod(block_size)

    return data


def convolve(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Applies 2D convolution using spatial domain (direct convolution)
    Parameters
    ----------
        image : torch.tensor
            Input for convolution
        kernel : torch.tensor
            Kernel for convolution
    Returns
    -------
        convolved image (torch.tensor)
    """
    # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
    image_4d = image.unsqueeze(0).unsqueeze(0)
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)

    # Calculate padding to maintain image size (same as IRAF's behavior)
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2

    # Convolve using torch.nn.functional.conv2d
    result = F.conv2d(image_4d, kernel_4d, padding=(pad_h, pad_w))

    # Remove batch and channel dimensions: (1, 1, H, W) -> (H, W)
    return result.squeeze(0).squeeze(0)


def median_filter_torch(image: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Applies a median filter using torch operations."""
    h, w = image.shape
    pad = kernel_size // 2

    image_padded = torch.nn.functional.pad(
        image.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode="replicate"
    )

    unfolded = torch.nn.functional.unfold(image_padded, kernel_size, stride=1)
    unfolded = unfolded.view(kernel_size * kernel_size, h, w)

    filtered, _ = unfolded.median(dim=0)
    return filtered


def median_filter_cpp_torch(image: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Fast CPU median filter using the C++ extension."""
    if not _CPP_MEDIAN_AVAILABLE:
        raise RuntimeError("median_filter_cpp extension is not available")
    if image.device.type != "cpu":
        raise ValueError("median_filter_cpp_torch requires a CPU tensor")
    if image.dtype != torch.float32:
        image = image.float()
    if not image.is_contiguous():
        image = image.contiguous()

    return median_filter_cpp.median_filter_cpu(image, kernel_size)


# Definition of the dilation using PyTorch
def dilation_pytorch(
    image: torch.Tensor,
    strel: torch.Tensor,
    origin: tuple[int, int] = (0, 0),
    border_value: float = 0,
):
    """
    Taken from https://stackoverflow.com/questions/56235733/is-there-a-tensor-operation-or-function-in-pytorch-that-works-like-cv2-dilate
    """
    # first pad the image to have correct unfolding; here is where the origins is used
    image_pad = F.pad(
        image,
        [
            origin[0],
            strel.shape[0] - origin[0] - 1,
            origin[1],
            strel.shape[1] - origin[1] - 1,
        ],
        mode="constant",
        value=border_value,
    )
    # Unfold the image to be able to perform operation on neighborhoods
    image_unfold = F.unfold(
        image_pad.unsqueeze(0).unsqueeze(0), kernel_size=strel.shape
    )
    # Flatten the structural element since its two dimensions have been flatten when unfolding
    strel_flatten = torch.flatten(strel).unsqueeze(0).unsqueeze(-1)
    # Perform the greyscale operation; sum would be replaced by rest if you want erosion
    sums = image_unfold + strel_flatten
    # Take maximum over the neighborhood
    result, _ = sums.max(dim=1)
    # Reshape the image to recover initial shape
    return torch.reshape(result, image.shape)


def sigma_clip_pytorch(
    data: torch.Tensor, sigma: tuple[float, float] | float = 3.0, maxiters: int = 10
) -> tuple[torch.Tensor, dict]:
    """
    Compute iterative sigma clipping (mimics IRAF iterstat)


    Returns
    -------
    clipped_data : torch.Tensor
        Data with outliers removed
    stats : dict
        Dictionary with 'median', 'mean', 'std', 'niter'
    """
    if isinstance(sigma, (int, float)):
        sigma_low, sigma_high = sigma, sigma
    else:
        sigma_low, sigma_high = sigma

    data = data.flatten().clone()

    for i in range(maxiters):
        torch.median(data)
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


def dilation_cpp_torch(
    image: torch.Tensor,
    strel_shape: tuple[int, int],
    origin: tuple[int, int] = (0, 0),
    border_value: float = 0.0,
) -> torch.Tensor:
    if not _CPP_DILATION_AVAILABLE:
        raise RuntimeError("dilation_cpp extension is not available")
    if image.device.type != "cpu":
        raise ValueError("dilation_cpp_torch requires a CPU tensor")
    if image.dtype != torch.float32:
        image = image.float()
    if not image.is_contiguous():
        image = image.contiguous()

    if isinstance(strel_shape, torch.Tensor):
        if strel_shape.dim() == 2:
            strel_shape = (int(strel_shape.shape[0]), int(strel_shape.shape[1]))
        else:
            strel_shape = tuple(int(x) for x in strel_shape.tolist())

    return dilation_cpp.dilation_cpu(
        image, strel_shape[0], strel_shape[1], origin[0], origin[1], border_value
    )


def cpp_median_available() -> bool:
    return _CPP_MEDIAN_AVAILABLE


def cpp_dilation_available() -> bool:
    return _CPP_DILATION_AVAILABLE
