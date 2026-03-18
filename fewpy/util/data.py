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
                 s_y: list[dict] | list[torch.Tensor]=None,
                 labels: list[dict]=None,
                 img_size: tuple[int] | int=None,
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
        self.transform_datapoints = transform_datapoints
        self.img_size = img_size
        self.method = support_set_preprocessing_method.lower()
        self.labels = labels
        self.pixel_norm = pixel_norm

        match self.method:

            case "standard":
                self.transform_s_x()

            case "resize_annotations":
                self.resize_annotations()

            case "normalize_annotations":
                self.normalize_annotations()

            case "detection_crop":
                self.detection_crop()

            case "norm_detection_crop":
                self.detection_crop(True)

            case "resize_labels":
                self.resize_labels()

            case "none":
                pass

            case _:
                raise ValueError("Unsuported support_set_preprocessing_method configured!") 

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

    def _padded_crop(self, image, bbox, output_size=(320, 320), norm=False, padding_ratio=0.5):
            if isinstance(output_size, int):
                output_size = (output_size, output_size)

            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            
            pad_w, pad_h = w * padding_ratio, h * padding_ratio
            
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(image.width if not isinstance(image, torch.Tensor) else image.shape[2], x2 + pad_w)
            y2 = min(image.height if not isinstance(image, torch.Tensor) else image.shape[1], y2 + pad_h)
            w, h = x2 - x1, y2 - y1
        
            bbox = [
                (bbox[0] - x1) / w,
                (bbox[1] - y1) / h,
                (bbox[2] - x1) / w,
                (bbox[3] - y1) / h,
            ]

            if isinstance(image, torch.Tensor):
                support_patch = image[:, int(y1):int(y2), int(x1):int(x2)]
            else:
                support_patch = image.crop((x1, y1, x2, y2))

            H, W = output_size
            bbox = [bbox[0] * W, bbox[1] * H, bbox[2] * W, bbox[3] * H]
        
            support_patch = F.resize(support_patch, output_size)
        
            if not isinstance(support_patch, torch.Tensor):
                support_patch = d2_tensor_transform(support_patch)
                
            if norm:
                mean, std = self.pixel_norm
                support_patch = F.normalize(support_patch, mean=mean, std=std)
        
            return support_patch, bbox
    
    def _transform(self, img, gt):

        W, H = img.size
        xi = self.transf(img)
        if isinstance(gt, dict) and "bbboxes" in gt.keys():
            new_gt = {k: gt[k] for k in gt.keys()}
            new_gt["bboxes"] = list()
            w, h = xi.shape[-1], xi.shape[-2]
            bboxes = torch.tensor(gt["bboxes"])
            scale = torch.tensor([w / W, h / H, w / W, h / H])
            new_gt["bboxes"] = (bboxes * scale).tolist()
        
            return xi, new_gt

        return xi, gt
    
    def transform_s_x(self):

        self._check_support_set()

        s_x = []
        s_y = []

        for img, gt in zip(self.s_x, self.s_y):
            img, gt = self._transform(img, gt)
            s_x.append(img)
            s_y.append(gt)

        self._stack_sx(s_x)
        self.s_y = s_y

    def resize_annotations(self):

        if self.img_size is None:
            raise ValueError("Support Set cannot be resized if img_size is None!")

        self.transform_s_x()

        if isinstance(self.s_y[0], torch.Tensor):
            new_s_y = []
            for s_yi in self.s_y:
                new_s_y.append(T.functional.resize(
                    s_yi, 
                    self.img_size,
                    interpolation=T.functional.InterpolationMode.NEAREST,
                ))
            self.s_y = torch.stack(new_s_y).squeeze(1)

    def resize_labels(self):

        self.resize_annotations()

        if self.labels is None:
            raise ValueError("Cannot resize None!")
        
        if isinstance(self.labels[0], torch.Tensor):
            new_labels = []
            for yi in self.labels:
                new_labels.append(T.functional.resize(
                    yi,
                    self.img_size,
                    interpolation=T.functional.InterpolationMode.NEAREST,
                ))
            self.labels = new_labels

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

    def detection_crop(self, norm=False):

        if self.img_size is None:
            raise ValueError("Support Set cannot be resized if img_size is None!")
        
        s_x = []
        s_y = []
        for xn, yn in zip(self.s_x, self.s_y):

            xn, bbox = self._padded_crop(xn, yn["bboxes"][0], self.img_size, norm)
            s_y.append({
                "bboxes": torch.tensor([bbox]).squeeze(1),
                "cls": yn["cls"][0]+1,
            })
            s_x.append(xn)

        self.s_x = torch.stack(s_x)
        self.s_y = s_y
    
    def __getitem__(self, index: int):

        xi = self.data[index]
        yi = dict()
        if self.transform_datapoints:
            if self.labels is not None:
                xi, yi = self._transform(xi, self.labels[index])

        if not self.support_set:
            return xi
        
        if not yi is None:
            return xi, yi, self.s_x, self.s_y

        return xi, self.s_x, self.s_y
    

def fsl_collate(batch):
    has_query_labels = len(batch[0]) == 4

    if has_query_labels:
        xi, yi, s_x, s_y = zip(*batch)
    
        batch_xi = torch.stack(xi)
        batch_yi = list(yi)
    
        return batch_xi, batch_yi, s_x[0], s_y[0]
        
    else:
        xi, s_x, s_y = zip(*batch)
        
        batch_xi = torch.stack(xi)
        
        return batch_xi, s_x[0], s_y[0]
