import torch
import torch.nn.functional as F


def _process_block_inputs(data, block_size):
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

    block_size_int = torch.tensor(block_size, dtype=torch.int)
    if torch.any(block_size_int != block_size):  # e.g., 2.0 is OK, 2.1 is not
        raise ValueError("block_size elements must be integers")

    return data, block_size_int


def block_replicate_torch(data, block_size, conserve_sum=True):
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
    output : array-like
        The block-replicated data. Note that when ``conserve_sum`` is
        `True`, the dtype of the output array will be float.

    """
    data, block_size = _process_block_inputs(data, block_size)
    for i in range(data.ndim):
        data = torch.repeat_interleave(data, block_size[i], dim=i)

    if conserve_sum:
        # in-place division can fail due to dtype casting rule
        data = data / torch.prod(block_size)

    return data


def convolve(image, kernel):
    """
    Applies 2D convolution using spatial domain (direct convolution)
    Args:
        image: image (torch.tensor) - 2D
        kernel: kernel (torch.tensor) - 2D
    Returns:
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


def median_filter_torch(image, kernel_size=3):
    """Applies a median filter using torch operations."""
    # b, c, h, w = image.shape
    h, w = image.shape
    pad = kernel_size // 2
    image = image.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, H, W)
    # Pad the image
    image_padded = torch.nn.functional.pad(image, (pad, pad, pad, pad), mode="reflect")

    # Use unfold to extract sliding window patches
    unfolded = torch.nn.functional.unfold(image_padded, kernel_size, stride=1)
    # unfolded = unfolded.view(b, c, kernel_size * kernel_size, h, w)
    unfolded = unfolded.squeeze(0).squeeze(0)
    unfolded = unfolded.view(kernel_size * kernel_size, h, w)
    # Compute median along the window dimension
    filtered, _ = unfolded.median(dim=0)

    return filtered


# Definition of the dilation using PyTorch
def dilation_pytorch(image, strel, origin=(0, 0), border_value=0):
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
