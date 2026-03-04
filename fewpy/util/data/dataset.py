import torch
import numpy as np
from PIL.Image import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FSLDataset(Dataset):

    def __init__(self,
                 x: list[Image],
                 s_x: list[Image]=None,
                 s_y: list[dict] | torch.Tensor=None,
                 img_size: tuple[int]=None,
                 max_size: int=None,
                 pixel_norm: tuple=None,
                 norm_annot: bool=False,
                 transform_datapoints: bool=True) -> None:
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

        self.data = x
        self.transf = transforms.Compose(transfs)
        self.support_set = (not s_x is None) and (not s_y is None)
        self.s_x = s_x
        self.s_y = s_y
        self.support_set_preproc = False
        self.norm_annot = norm_annot
        self.transform_datapoints = transform_datapoints

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int):

        xi = self.data[index]
        if self.transform_datapoints:
            xi = self.transf(xi)

        if not self.support_set:
            return xi
        
        if self.support_set_preproc:
            return xi, self.s_x, self.s_y
        
        s_x = []
        s_y = []
        for img, annot in zip(self.s_x, self.s_y):
            new_img = self.transf(img)

            if self.norm_annot:
                old_h, old_w = img.size

                if "bboxes" not in annot.keys():
                   raise KeyError("Bounding Box annotations should be named bboxes and be a list of bounding boxes [xmin, ymin, xmax, ymax]")

                new_bboxes = []
                for xmin, ymin, xmax, ymax in annot["bboxes"]:
                    new_bboxes.append([
                        xmin / old_w,
                        ymin / old_h,
                        xmax / old_w,
                        ymax / old_h,
                    ])
                
                annot["bboxes"] = new_bboxes
            
            s_y.append(annot)
            s_x.append(new_img)

        if len(s_x) > 0:
            first_shape = s_x[0].shape
            all_same_shape = all(t.shape == first_shape for t in s_x)
            if all_same_shape:
                self.s_x = torch.stack(s_x)
            else:
                self.s_x = s_x
        else:
            self.s_x = s_x
        is_tensor = isinstance(self.s_y, torch.Tensor)
        self.s_y = torch.stack(s_y) if is_tensor else s_y

        self.support_set_preproc = True

        return xi, self.s_x, self.s_y
    

def fsl_collate(batch):

    batch, s_x, s_y = zip(*batch)
    batch = torch.stack(batch)
    s_x = s_x[0]
    s_y = s_y[0]

    return batch, s_x, s_y

def qwen_collate(batch):
    
    return batch