import torch
import numpy as np
import random
import copy
from PIL.Image import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision.transforms.functional as F
from enum import Enum


def d2_tensor_transform(image):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    return torch.as_tensor(image.astype("float32").transpose(2, 0, 1))


class PreprocessingMethod(Enum):

    NONE = "none"
    STANDARD = "standard"
    NORMALIZE_SUPPORT_GT = "normalize_support_gt"
    RESIZE_SUPPORT_GT = "resize_support_gt"
    DETECTION_CROP = "detection_crop"
    NORM_DETECTION_CROP = "norm_detection_crop"
    AUGMENT_SUPPORT_IMAGES = "augment_support_images"


class FSLDataset(Dataset):

    def __init__(self,
                 x: list[Image],
                 s_x: list[Image]=None,
                 s_y: list[dict] | list[torch.Tensor]=None,
                 labels: list[dict] | list[torch.Tensor]=None,
                 img_size: tuple[int] | int=None,
                 max_size: int=None,
                 antialias: bool=True,
                 interpolation: F.InterpolationMode=F.InterpolationMode.BICUBIC,
                 pixel_norm: tuple=None,
                 center_crop: int=None,
                 support_set_preprocessing_method: PreprocessingMethod=PreprocessingMethod.STANDARD,
                 transform_datapoints: bool=True,
                 **kwargs
            ) -> None:
        super().__init__()

        # set transform composition that turns pillow image objects into datapoints compatible with the library
        transfs = []
        if img_size is not None:
            transfs.append(
                T.Resize(
                    size=img_size, 
                    max_size=max_size, 
                    antialias=antialias,
                    interpolation=interpolation
                    )
                )

        if center_crop is not None and center_crop != 0:
            transfs.append(T.CenterCrop(center_crop))
 
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
        self.method = support_set_preprocessing_method
        self.labels = labels
        self.pixel_norm = pixel_norm

        if not self.support_set:
            self.method = PreprocessingMethod.NONE

        self.epochs = kwargs["epochs"] if "epochs" in kwargs else kwargs.get("augment_epochs", 12)

        self._preprocess()

    def __len__(self) -> int:
        return len(self.data)

    def _preprocess(self):

        match self.method:

            case PreprocessingMethod.STANDARD:
                self.transform_s_x()

            case PreprocessingMethod.RESIZE_SUPPORT_GT:
                self.resize_support_gt()

            case PreprocessingMethod.NORMALIZE_SUPPORT_GT:
                self.normalize_annotations()

            case PreprocessingMethod.DETECTION_CROP:
                self.detection_crop()

            case PreprocessingMethod.NORM_DETECTION_CROP:
                self.detection_crop(True)

            case PreprocessingMethod.AUGMENT_SUPPORT_IMAGES:

                self.augment_support_images(self.epochs)

            case PreprocessingMethod.NONE:
                pass

            case _:
                raise ValueError("Unsuported support_set_preprocessing_method configured!") 

        
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
            gt = new_gt
        elif isinstance(gt, torch.Tensor) and len(gt.shape) > 2:
            new_gt = T.functional.resize(
                gt, self.img_size,
                interpolation=T.functional.InterpolationMode.NEAREST,
            )
            gt = new_gt

        return xi, gt
    
    def transform_s_x(self):

        s_x = []
        s_y = []

        for img, gt in zip(self.s_x, self.s_y):
            img, gt = self._transform(img, gt)
            s_x.append(img)
            s_y.append(gt)

        self._stack_sx(s_x)
        self.s_y = s_y

    def resize_support_gt(self):

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

    def normalize_annotations(self):

        s_x = []
        s_y = []
        for img, annot in zip(self.s_x, self.s_y):
            if isinstance(img, Image):
                old_w, old_h = img.size
                img = self.transf(img)

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

    def augment_support_images(self, epochs):

        _transform = T.Compose([
            T.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        augmented_set = []
        for _ in range(epochs):
            for img in self.s_x:
                augmented_set.append(_transform(img))

        self.s_x = torch.stack(augmented_set)
        self.s_y = torch.stack(self.s_y)
    
    def __getitem__(self, index: int):

        xi = self.data[index]
        yi = None
        if self.transform_datapoints:
            if self.labels is not None:
                xi, yi = self._transform(xi, self.labels[index])
            else:
                xi = self.transf(xi)

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


class EpisodicDataset(FSLDataset):
    def __init__(self,
                 n_way: int,
                 n_shot: int,
                 n_query: int,
                 n_episodes: int,
                 x: list[Image],
                 labels: list,
                 **kwargs):
        
        kwargs_clean = kwargs.copy()
        self.intended_method = kwargs_clean.pop('support_set_preprocessing_method', PreprocessingMethod.STANDARD)
        self.kwargs = kwargs_clean
        
        super().__init__(
            x=x, 
            s_x=None, 
            s_y=None, 
            labels=labels, 
            support_set_preprocessing_method=PreprocessingMethod.NONE, 
            **kwargs_clean
        )
        
        self.method = self.intended_method
        self.support_set = True 

        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        
        # Safe Multi-Class Map
        self.class_to_indices = {}
        for idx, label in enumerate(self.labels):
            unique_classes = self._extract_all_class_ids(label)
            for cls_id in unique_classes:
                if cls_id not in self.class_to_indices:
                    self.class_to_indices[cls_id] = []
                self.class_to_indices[cls_id].append(idx)
            
        self.classes = list(self.class_to_indices.keys())
        if len(self.classes) < n_way:
            raise ValueError(f"Dataset only has {len(self.classes)} unique classes, but n_way={n_way}.")

    def _extract_all_class_ids(self, label) -> set:
        if isinstance(label, dict):
            if "cls" in label and label["cls"] is not None:
                val = label["cls"]
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    return set(v.item() if isinstance(v, torch.Tensor) else v for v in val)
                elif isinstance(val, torch.Tensor) and val.numel() > 0:
                    return set(val.flatten().tolist())
                elif not isinstance(val, (list, tuple, torch.Tensor)):
                    return {val}
            
            if "mask" in label and isinstance(label["mask"], torch.Tensor):
                return set(label["mask"].unique().tolist()) - {0}
            
            # Fallback if dictionary has NO class data (e.g., background-only image)
            return {0} 
            
        if isinstance(label, torch.Tensor):
            return {label.item()} if label.numel() == 1 else set(label.unique().tolist()) - {0}
            
        if isinstance(label, (list, tuple)):
            return set(label) if len(label) > 0 else {0}

        try:
            return {label}
        except TypeError:
            return {0}

    def __len__(self) -> int:
        return self.n_episodes
    
    def _remap_label(self, label, global_to_local, ignore_index=-1):
        label_copy = copy.deepcopy(label)
        
        if isinstance(label_copy, dict):
            if "cls" in label_copy and label_copy["cls"] is not None:
                val = label_copy["cls"]
                if isinstance(val, torch.Tensor):
                    new_val = torch.full_like(val, ignore_index)
                    for g_id, l_id in global_to_local.items():
                        new_val[val == g_id] = l_id
                    label_copy["cls"] = new_val
                elif isinstance(val, (list, tuple)):
                    label_copy["cls"] = [global_to_local.get(v.item() if isinstance(v, torch.Tensor) else v, ignore_index) for v in val]
                else:
                    v_clean = val.item() if isinstance(val, torch.Tensor) else val
                    label_copy["cls"] = global_to_local.get(v_clean, ignore_index)
                    
            if "mask" in label_copy and isinstance(label_copy["mask"], torch.Tensor):
                mask = label_copy["mask"]
                new_mask = torch.full_like(mask, ignore_index)
                # If 0 represents structural background, preserve it safely
                if 0 not in global_to_local:
                    new_mask[mask == 0] = 0
                for g_id, l_id in global_to_local.items():
                    new_mask[mask == g_id] = l_id
                label_copy["mask"] = new_mask
                
            return label_copy
            
        if isinstance(label_copy, torch.Tensor):
            if label_copy.numel() == 1:
                v_clean = label_copy.item()
                return torch.tensor(global_to_local.get(v_clean, ignore_index), dtype=label_copy.dtype, device=label_copy.device)
            else:
                new_label = torch.full_like(label_copy, ignore_index)
                if 0 not in global_to_local:
                    new_label[label_copy == 0] = 0
                for g_id, l_id in global_to_local.items():
                    new_label[label_copy == g_id] = l_id
                return new_label
                
        if isinstance(label_copy, (list, tuple)):
            return [global_to_local.get(v, ignore_index) for v in label_copy]
            
        return global_to_local.get(label_copy, ignore_index)

    def __getitem__(self, index: int):
        if hasattr(self, 'seed_interval') and self.seed_interval is not None:
            current_seed = self.base_seed + (index // self.seed_interval) * 100000 + (index % self.seed_interval)
            random.seed(current_seed)
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(current_seed)
                torch.cuda.manual_seed_all(current_seed)

        episode_classes = random.sample(self.classes, self.n_way)
        
        global_to_local = {g_id: l_idx for l_idx, g_id in enumerate(episode_classes)}
        
        s_x_raw, s_y_raw, s_targets = [], [], []
        q_x_raw, q_y_raw = [], []
        used_episode_indices = set()

        episode_q_x_idx, episode_s_x_idx = [], []
        
        for cls in episode_classes:
            available_indices = self.class_to_indices[cls]
            valid_indices = [i for i in available_indices if i not in used_episode_indices]
            
            if len(valid_indices) < (self.n_shot + self.n_query):
                valid_indices = available_indices 
                
            sampled_indices = random.sample(valid_indices, self.n_shot + self.n_query)
            used_episode_indices.update(sampled_indices)
            
            for idx in sampled_indices[:self.n_shot]:
                s_x_raw.append(self.data[idx])
                s_y_raw.append(copy.deepcopy(self.labels[idx]))
                s_targets.append(global_to_local[cls]) 
                
            for idx in sampled_indices[self.n_shot:]:
                q_x_raw.append(self.data[idx])
                q_y_raw.append(copy.deepcopy(self.labels[idx]))

            episode_q_x_idx.extend(sampled_indices[self.n_shot:])
            episode_s_x_idx.extend(sampled_indices[:self.n_shot])

        s_y_raw = [self._remap_label(y, global_to_local) for y in s_y_raw]
        q_y_raw = [self._remap_label(y, global_to_local) for y in q_y_raw]

        processed_q_x, processed_q_y = [], []
        for q_x, q_y in zip(q_x_raw, q_y_raw):
            xi, yi = q_x, q_y
            if self.transform_datapoints:
                xi, yi = self._transform(xi, yi)
            processed_q_x.append(xi)
            processed_q_y.append(yi)
        
        q_x_out = torch.stack(processed_q_x) if isinstance(processed_q_x[0], torch.Tensor) else processed_q_x

        episode_s_x, episode_s_y = self._preprocess_episode_statelessly(s_x_raw, s_y_raw, s_targets)

        return q_x_out, processed_q_y, episode_q_x_idx, episode_s_x, episode_s_y, episode_s_x_idx, episode_classes

    def _preprocess_episode_statelessly(self, s_x, s_y, s_targets):
        
        match self.method:
            case PreprocessingMethod.STANDARD:
                return self._stateless_transform_s_x(s_x, s_y)

            case PreprocessingMethod.RESIZE_SUPPORT_GT:
                if self.img_size is None:
                    raise ValueError("Support Set cannot be resized if img_size is None!")
                s_x_proc, s_y_proc = self._stateless_transform_s_x(s_x, s_y)
                if isinstance(s_y_proc[0], torch.Tensor):
                    new_s_y = [
                        torch.nn.functional.interpolate(
                            s_yi.unsqueeze(0) if s_yi.ndim == 2 else s_yi, 
                            size=self.img_size, mode='nearest'
                        ).squeeze(0) for s_yi in s_y_proc
                    ]
                    return s_x_proc, torch.stack(new_s_y)
                return s_x_proc, s_y_proc

            case PreprocessingMethod.NORMALIZE_SUPPORT_GT:
                s_x_out, s_y_out = [], []
                for img, annot in zip(s_x, s_y):
                    old_w, old_h = img.size if isinstance(img, Image) else (1, 1)
                    if isinstance(img, Image):
                        img = self.transf(img)
                    
                    # Safe normalization check
                    if "bboxes" in annot and len(annot["bboxes"]) > 0:
                        annot["bboxes"] = [[b[0]/old_w, b[1]/old_h, b[2]/old_w, b[3]/old_h] for b in annot["bboxes"]]
                    s_y_out.append(annot)
                    s_x_out.append(img)
                return torch.stack(s_x_out) if all(t.shape == s_x_out[0].shape for t in s_x_out) else s_x_out, s_y_out

            case PreprocessingMethod.DETECTION_CROP | PreprocessingMethod.NORM_DETECTION_CROP:
                if self.img_size is None:
                    raise ValueError("Support Set cannot be resized if img_size is None!")
                norm = (self.method == PreprocessingMethod.NORM_DETECTION_CROP)
                s_x_out, s_y_out = [], []
                
                for i, (xn, yn) in enumerate(zip(s_x, s_y)):
                    bboxes = yn.get("bboxes", [])
                    cls_list = yn.get("cls", [])
                    target_cls = s_targets[i]
                    
                    # Convert labels to standard primitives to locate the target index safely
                    cls_list_clean = [c.item() if isinstance(c, torch.Tensor) else c for c in cls_list]
                    
                    # Find exactly which index contains our target class object
                    if target_cls in cls_list_clean:
                        target_idx = cls_list_clean.index(target_cls)
                    else:
                        target_idx = 0 # Fallback if class not found
                    
                    # Secure bounding box extraction
                    if len(bboxes) > target_idx:
                        chosen_bbox = bboxes[target_idx]
                        chosen_cls = cls_list[target_idx]
                    elif len(bboxes) > 0:
                        chosen_bbox = bboxes[0]
                        chosen_cls = cls_list[0]
                    else:
                        # Complete fallback if image contains zero annotations
                        w, h = xn.size if hasattr(xn, 'size') else (224, 224)
                        chosen_bbox = [0, 0, w, h]
                        chosen_cls = target_cls

                    xn_proc, bbox = self._padded_crop(xn, chosen_bbox, self.img_size, norm)
                    
                    out_cls = chosen_cls.item() if isinstance(chosen_cls, torch.Tensor) else chosen_cls
                    s_y_out.append({
                        "bboxes": torch.tensor([bbox]),
                        "cls": out_cls + 1,
                    })
                    s_x_out.append(xn_proc)
                return torch.stack(s_x_out), s_y_out

            case PreprocessingMethod.AUGMENT_SUPPORT_IMAGES:
                epochs = self.kwargs.get("epochs", self.kwargs.get("augment_epochs", 12))
                from torchvision import transforms as T
                local_aug = T.Compose([
                    T.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=T.InterpolationMode.BICUBIC),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                ])
                augmented_x = [local_aug(img) for _ in range(epochs) for img in s_x]
                augmented_y = [copy.deepcopy(y) for _ in range(epochs) for y in s_y]
                if isinstance(augmented_y[0], torch.Tensor):
                    augmented_y = torch.stack(augmented_y)
                return torch.stack(augmented_x), augmented_y

            case PreprocessingMethod.NONE:
                return s_x, s_y
            case _:
                raise ValueError("Unsupported preprocessing method configuration!")

    def _stateless_transform_s_x(self, s_x, s_y):
        s_x_out, s_y_out = [], []
        for img, gt in zip(s_x, s_y):
            img_proc, gt_proc = self._transform(img, gt)
            s_x_out.append(img_proc)
            s_y_out.append(gt_proc)
        if len(s_x_out) > 0 and all(t.shape == s_x_out[0].shape for t in s_x_out):
            return torch.stack(s_x_out), s_y_out
        return s_x_out, s_y_out

def episodic_collate(batch):
    has_query_labels = len(batch[0]) == 7

    if has_query_labels:
        xi, yi, q_idx, s_x, s_y, s_idx, episode_classes = zip(*batch)
    
        batch_xi = torch.stack(xi).squeeze()
        batch_yi = list(yi)[0]

        return batch_xi, batch_yi, q_idx[0], s_x[0], torch.tensor(s_y[0]).squeeze(), s_idx[0], episode_classes[0]
        