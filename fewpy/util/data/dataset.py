import torch
import numpy as np
from PIL.Image import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from ..models.fsl_model import FSLModel


class FSLDataset(Dataset):

    def __init__(self,
                 data: list[list[Image]],
                 model) -> None:
        super().__init__()

        # set transform composition that turns pillow image objects into datapoints compatible with the library
        transfs = []
        if model.img_size:
            transfs.append(transforms.Resize(model.img_size))

        transfs.append(transforms.ToTensor())

        if model.pixel_norm:
            mean, std = model.pixel_norm
            transfs.append(transforms.Normalize(mean, std))

        self.data = data
        self.transf = transforms.Compose(transfs)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> list[torch.Tensor]:
        return torch.stack(list((self.transf(img) for img in self.data[index])))
