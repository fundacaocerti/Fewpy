import torch
import numpy as np
from PIL.Image import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision.transforms.functional as F


def d2_tensor_transform(image):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    return torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

class FSLDataset(Dataset):

    def __init__(self,
                 x: list[Image],
                 s_x: list[Image]=None,
                 s_y: list[dict] | list[torch.Tensor] | torch.Tensor=None,
                 img_size: tuple[int]=None,
                 max_size: int=None,
                 pixel_norm: tuple=None,
                 support_set_preprocessing_method: str="standard",
                 transform_datapoints: bool=True,
            ) -> None:
        super().__init__()

        # set transform composition that turns pillow image objects into datapoints compatible with the library
        transfs = []
        if img_size is not None:
            if max_size is not None:
                transfs.append(T.Resize(size=img_size, max_size=max_size))
            else:
                transfs.append(T.Resize(img_size))
 
        transfs.append(T.ToTensor())
        if pixel_norm is not None:
            mean, std = pixel_norm
            transfs.append(T.Normalize(mean, std))

        self.data = x
        self.transf = T.Compose(transfs)
        self.support_set = (not s_x is None) and (not s_y is None)
        self.s_x = s_x
        self.s_y = s_y
        self.support_set_preproc = False
        self.transform_datapoints = transform_datapoints
        self.img_size = img_size
        self.method = support_set_preprocessing_method.lower()

    def __len__(self) -> int:
        return len(self.data)
    
    def _check_support_set(self):

        if self.s_x is None or self.s_y is None:
            raise ValueError("Operation requires support set and s_x or s_y are None!")
        
    def _stack_sx(self, s_x):

        if len(s_x) > 0:
            first_shape = s_x[0].shape
            all_same_shape = all(t.shape == first_shape for t in s_x)
            if all_same_shape:
                self.s_x = torch.stack(s_x)
            else:
                self.s_x = s_x
        else:
            self.s_x = s_x

        self.support_set_preproc = True

    def _padded_crop(self, image, bbox, output_size=(320, 320), padding_ratio=0.5):
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            
            pad_w, pad_h = w * padding_ratio, h * padding_ratio
            
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(image.width if not isinstance(image, torch.Tensor) else image.shape[2], x2 + pad_w)
            y2 = min(image.height if not isinstance(image, torch.Tensor) else image.shape[1], y2 + pad_h)
            
            if isinstance(image, torch.Tensor):
                support_patch = image[:, int(y1):int(y2), int(x1):int(x2)]
            else:
                support_patch = image.crop((x1, y1, x2, y2))
        
            support_patch = F.resize(support_patch, output_size)
        
            if not isinstance(support_patch, torch.Tensor):
                support_patch = d2_tensor_transform(support_patch)
                
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            support_patch = F.normalize(support_patch, mean=mean, std=std)
        
            return support_patch
    
    def transform_s_x(self):

        self._check_support_set()

        s_x = []
        for img in self.s_x:
            if isinstance(img, Image):
                img = self.transf(img)
            s_x.append(img)

        self._stack_sx(s_x)

    def resize_annotations(self):

        self.transform_s_x()

        if isinstance(self.s_y[0], torch.Tensor):

            if self.img_size is None:
                raise ValueError("Support Set cannot be resized if img_size is None!")

            new_s_y = []
            for s_yi in self.s_y:
                new_s_y.append(T.functional.resize(
                    s_yi, 
                    self.img_size,
                    interpolation=T.functional.InterpolationMode.NEAREST,
                ))
            self.s_y = torch.stack(new_s_y).squeeze(1)
    
    def normalize_annotations(self):

        self._check_support_set()

        s_x = []
        s_y = []
        for img, annot in zip(self.s_x, self.s_y):
            if isinstance(img, Image):
                img = self.transf(img)

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
            s_x.append(img)

        self.s_y = s_y

        self._stack_sx(s_x)

    def detection_crop(self):

        if self.img_size is None:
            raise ValueError("Support Set cannot be resized if img_size is None!")

        permuted_indexes = np.random.permutation(len(self.s_x))
        s_x = []
        s_y = []
        n_per_class = [0] * self.c
        for n in permuted_indexes:
            xn, yn = self.s_x[n], self.s_y[n]

            if all([k >= self.k for k in n_per_class]):
                break
            
            if n_per_class[yn["class_ids"][0]] < self.k:
                n_per_class[yn["class_ids"][0]] += 1
                s_y.append({
                    "bboxes": torch.tensor(yn["bboxes"]).squeeze(1),
                    "cls": yn["class_ids"][0]+1,
                })
                xn = self._padded_crop(xn, yn["bboxes"][0], self.img_size)
                s_x.append(xn)

        s_y = torch.stack(s_y).squeeze(1)
        self.s_x = torch.stack(s_x), s_y
    
    def __getitem__(self, index: int):

        xi = self.data[index]
        if self.transform_datapoints:
            xi = self.transf(xi)

        if not self.support_set:
            return xi
        
        if self.support_set_preproc:
            return xi, self.s_x, self.s_y
        
        match self.method:

            case "standard":
                self.transform_s_x()

            case "resize_annotations":
                self.resize_annotations()

            case "normalize_annotations":
                self.normalize_annotations()

            case "detection_crop":
                self.detection_crop()

            case "none":
                self.support_set_preproc = True

            case _:
                raise ValueError("Unsuported support_set_preprocessing_method configured!") 

        return xi, self.s_x, self.s_y
    

def fsl_collate(batch):

    batch, s_x, s_y = zip(*batch)
    batch = torch.stack(batch)
    s_x = s_x[0]
    s_y = s_y[0]

    return batch, s_x, s_y