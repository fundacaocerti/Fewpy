'''
Script to generate Dataset e Dataloader for inference in app
'''
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import PIL.Image as Image
import numpy as np
from typing import Dict, Union, List

class Dataset:
    """
    A class that define hyperparameters will be used for 
    construct a dataloader object to inference
    """
    @classmethod
    def initialize(cls, img_size: int= 384) -> None:
        """
        Initialize parameters used in class Visualizer
        :param img_size: image size
        :type img_size: int
        
        :returns: None
        """
        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.transform = A.Compose([
                    A.Resize(img_size, img_size),
                    A.Normalize(cls.img_mean, cls.img_std),
                    A.pytorch.transforms.ToTensorV2(),
                ])
        
    @classmethod
    def build_dataloader(cls,
     query: str, 
     experiment: str, 
     nworker: int, 
     shuffle: bool, 
     bsz: int, 
     nshot: int) -> DataLoader:
        '''
        This function create a object of the class Dataset and instanciate a 

        :param query: image directory
        :type query: str
        :param experiment: experiment directory
        :type query: experiment
        :param nworker: number of workers
        :type nworker: int
        :param shuffle: determine if shuffle the dataset
        :type shuffle: bool
        :param bsz:  batch size 
        :type bsz: int
        :param nshot: number of shots
        :type nshot: int
        :returns: A Dataloader 
        :rtype: Dataloader                
        '''
        dataset = DatasetPredict(query, experiment, cls.transform, nshot)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader


class DatasetPredict(Dataset):
    def __init__(self, query: str, experiment: str, transform: A.Compose, shot: int) -> None:
        """
        Generate dataset

        :param query: query image directory
        :type query: str
        :param experiment: experiment directory
        :type experiment: str
        :param transform: transform to tensor, normalize and resize
        :type transform: A.compose
        :param shot: number of shots in the support
        :type shot: int
        :returns: None
        """

        self.shot = shot
        # support directory
        self.base_path_support = os.path.join(experiment, 'support') 
        # support_mask directory
        self.base_path_mask = os.path.join(experiment, 'support_mask')
        # support_mask images
        self.support_images = os.listdir(self.base_path_support)

        self.transform = transform
        self.shot = shot
        self.query = query

        self.query_name = query.split('/')[-1]

    def __len__(self):
        return len(range(1))

    def __getitem__(self, idx: int) -> Dict[str,Union[torch.Tensor,List[str]]]:
        """
        Create batch for inference model

        :returns: (dict) dict with values 
                 tensor (query_img, support_imgs, support_masks) and List[str] (query_name)
        :rtype: Dict[str,Union[tensor.Torch,List[str]]]
        """

        support_name_list = []
        query_img = self.query

        if self.shot > len(self.support_images):
            raise Exception(f"Insira pelo menos {self.shot} imagens de suporte em {self.base_path_support}")

        # select k imagens of support
        support_name_list = self.support_images[:self.shot]

        # concatenate name of the support image/mask with her path
        support_path_images = [os.path.join(self.base_path_support, support_name) for support_name in support_name_list]
        support_path_masks = [os.path.join(self.base_path_mask, support_name.split('.')[0]+'.png') for support_name in support_name_list]

        # read query and support images
        query_img = np.array(Image.open(query_img).convert('RGB'))
        support_imgs = [np.array(Image.open(support_path).convert('RGB')) for support_path in support_path_images]

        # read mask support images
        support_mask_imgs = [self.read_mask(mask_path) for mask_path in support_path_masks]
        
        # transform images
        query_img = self.transform(image=query_img)["image"]
        support_transformed = [self.transform(image=support_img)["image"] for support_img in support_imgs]
        # concatenate support images in first dimension
        support_imgs = torch.stack(support_transformed)

        # resize masks and concatenate everything in first dimension
        support_masks_tmp = []
        for smask in support_mask_imgs:
            # support_imgs.size() == [shot, 3, H, W] 
            # smask.size() == [h,w]
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            # smask.size() == [H,W]
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)
        # support_masks.size() == [shot,H,W]

        batch = {'query_img': query_img, # [3, H, W]
                 'support_imgs': support_imgs, #[shot,3,H,W]
                 'support_masks': support_masks, #[shot,H,W]
                 'query_name': self.query_name  #1

                 #'query_mask': query_mask,
                 #'support_name': support_name,
                 #'class_id': torch.tensor(class_sample)
                 }
        
        print(query_img.size(), support_imgs.size(), support_masks.size(), self.query_name)

        return batch

    def read_mask(self, mask_path: str) -> torch.Tensor:
        '''
        Transform mask to a binary tensor 
        :param mask_path: path of the support mask 
        :type mask_path: str
        :returns: tensor with 0 and 1 values
        :rtype: tensor.Torch
        '''
        mask = torch.tensor(np.array(Image.open(mask_path).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask


class FewShotDataset(Dataset):
    def __init__(self, query_path: str, shot: int = 5, img_size: int = 700, transform=None):
        self.query_path = query_path
        self.shot = shot
        self.img_size = img_size

        if transform is None:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = transform

        self.query_dir = os.path.dirname(query_path)
        self.query_name = os.path.basename(query_path)

        self.defect_class = os.path.basename(self.query_dir)

        all_images = sorted(os.listdir(self.query_dir))
        self.support_images = [img for img in all_images if img != self.query_name]

        if len(self.support_images) < self.shot:
            raise ValueError(f"Tem {len(self.support_images)} imagens suporte, mas shot={self.shot}")

        self.mask_dir = self.query_dir.replace("test", "ground_truth")

    def __len__(self):
        return 1  

    def __getitem__(self, idx):
        query_img = np.array(Image.open(self.query_path).convert('RGB'))
        query_img = self.transform(image=query_img)['image']

        import random
        selected_support_names = random.sample(self.support_images, self.shot)

        support_img_tensors = []
        raw_support_masks = [] 

        for sup_name in selected_support_names:
            sup_path = os.path.join(self.query_dir, sup_name)
            sup_img = np.array(Image.open(sup_path).convert('RGB'))
            
            support_img_tensors.append(self.transform(image=sup_img)['image'])

            mask_name = sup_name.rsplit('.',1)[0] + "_mask.png"
            mask_path = os.path.join(self.mask_dir, mask_name)
            
            if os.path.exists(mask_path):
                mask = torch.tensor(np.array(Image.open(mask_path).convert('L')))
                mask[mask < 128] = 0
                mask[mask >= 128] = 1
                raw_support_masks.append(mask)
            else:
                mask = np.zeros((sup_img.shape[0], sup_img.shape[1]), dtype=bool)
                raw_support_masks.append(torch.from_numpy(mask))

        support_imgs = torch.stack(support_img_tensors)

        support_mask_tensors = []
        target_size = support_imgs.size()[-2:]

        for smask in raw_support_masks:
            smask_resized = F.interpolate(
                smask.unsqueeze(0).unsqueeze(0).float(), 
                size=target_size, 
                mode='nearest'
            ).squeeze()
            support_mask_tensors.append(smask_resized)
        
        support_masks = torch.stack(support_mask_tensors)

        return {
            'query_img': query_img,         # [3, H, W]
            'support_imgs': support_imgs,   # [shot, 3, H, W]
            'support_masks': support_masks, # [shot, H, W]
            'query_name': self.query_name,
            'defect_class': self.defect_class,
        }