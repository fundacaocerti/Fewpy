import random
import torch
import numpy as np
from typing import Dict, List

def fix_randseed(seed: int) -> None:
    """
    Set seeds for reproducibility.

    :param seed: The integer number chosen for reproducibility.
    :type seed: int
    :return: None
    """

    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def mean(x: List[float]) -> float:
    """
    Calculate the mean of a list of numbers.

    :param x: A list of numbers.
    :type x: List[float]
    :return: The mean of the numbers in the list.
    :rtype: float
    """
    return sum(x) / len(x) if len(x) > 0 else 0.0

def to_cuda(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Move the elements of a dictionary containing tensors to the GPU.

    :param batch: A dictionary where the values are PyTorch tensors.
    :type batch: Dict[str, torch.Tensor]
    :return: A dictionary with tensors moved to the GPU.
    :rtype: Dict[str, torch.Tensor]
    """

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch

def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """
    Create a tensor on the CPU.

    :param tensor: The tensor on GPU.
    :type tensor: torch.Tensor
    :return: The tensor on CPU.
    :rtype: torch.Tensor
    """

    return tensor.detach().clone().cpu()