import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import Union

def _to_numpy_image(img: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Casts an image torch tensor to numpy array format. If numpy array is 
    passed, returns it directly. If the input is a normalized tensor in the 
    range [0.0, 1.0], it will be coerced to a RGB uint8.

    Args:
        img (Union[np.ndarray, torch.Tensor]): image in numpy array or torch
            tensor format.

    Raises:
        TypeError: if the input image is not in numpy or torch format.

    Returns:
        [type]: image as numpy array.
    """

    if not isinstance(img, (np.ndarray, torch.Tensor)):
        raise TypeError("img must be a numpy array or a torch tensor.")

    if isinstance(img, torch.Tensor):
        if img.dtype == torch.float:
            img = np.uint8(((img*255).permute(1, 2, 0)
                            if img.ndim == 3 else img).numpy())
        else:
            img = (img.permute(1, 2, 0) if img.ndim == 3 else img).numpy()

    return img


def imshow(img: Union[np.ndarray, torch.Tensor]) -> None:
    """Shows an image.

    Args:
        img (Union[np.ndarray, torch.Tensor]): Either an image in numpy array 
            or in torch tensor format (normalized or not).
    """
    img = _to_numpy_image(img)
    cmap = "gray" if img.ndim == 2 else None

    plt.imshow(img, cmap=cmap)
    plt.show()

