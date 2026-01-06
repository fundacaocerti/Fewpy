import torch
import numpy as np
from PIL.Image import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FSLDataset(Dataset):

    def __init__(self,
                 data: list[list[Image]],
                 img_size: tuple=None,
                 max_size: int=None,
                 pixel_norm: tuple=None) -> None:
        super().__init__()

        # set transform composition that turns pillow image objects into datapoints compatible with the library
        transfs = []
        if img_size is not None:
            if max_size is not None:
                transfs.append(transforms.Resize(size=img_size, max_size=max_size))
            else:
                transfs.append(transforms.Resize(img_size))

        transfs.append(transforms.ToTensor())

        if pixel_norm is not None:
            mean, std = pixel_norm
            transfs.append(transforms.Normalize(mean, std))

        self.data = data
        self.transf = transforms.Compose(transfs)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> list[torch.Tensor]:
        return torch.stack(list((self.transf(img) for img in self.data[index])))
 Meus itens
Fewpy Models 