import torch
import numpy as np
from PIL.Image import Image
from dataset import Dataset
from ..models.fsl_model import FSLModel


def DataLoader(data: list[list[Image]], model: FSLModel) -> Dataset:

    # Resizes the images if the model requires a specific image size
    if model.img_size != (0, 0):
        for i in range(len(data)):
            for k in range(len(data[i])):
                data[i][k] = data[i][k].resize(model.img_size)

    data = [[np.array(img) for img in imagelist] for imagelist in data]    
    # normalizes pixel value
    if model.pixel_norm:
        min_v, max_v, std = model.pixel_norm
        div = (max_v - min_v) * std
        data = [(in_features - min_v) / div for in_features in data]

    # convert np.ndarray into toch.Tensor
    data = [[torch.from_numpy(arr).permute(2, 0, 1) for arr in imagelist] for imagelist in data]
    data = torch.stack([torch.stack(in_features) for in_features in data])

    return Dataset(data)
