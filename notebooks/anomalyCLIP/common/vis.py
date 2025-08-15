import numpy as np
import torchvision.transforms as transforms
from .utils import to_cpu
import torch
from typing import List, Tuple, Union
from PIL import Image

class Visualizer:
    """
    Class to construct the image with the mask predicted merged in image of query
    """
    @classmethod
    def initialize(cls):
        """
        Initialize parameters used in class Visualizer

        :return: None
        """

        cls.colors = {'red': (255, 50, 50), 'blue': (102, 140, 255)}
        for key, value in cls.colors.items():
            cls.colors[key] = tuple([c / 255 for c in cls.colors[key]])

        cls.mean_img = [0.485, 0.456, 0.406]
        cls.std_img = [0.229, 0.224, 0.225]
        cls.to_pil = transforms.ToPILImage()  
        

    @classmethod
    def visualize_prediction_batch(
        cls, 
        qry_img_b: torch.Tensor, 
        pred_mask_b: torch.Tensor, 
        query_name_b: Union[List[torch.Tensor],None] =None
        ) -> List[Image.Image]:
        """
        Apply mask to the given batch of images.

        :param qry_img_b: Query image tensor with dimensions [batch, 3, h, w].
        :type qry_img_b: torch.Tensor
        :param pred_mask_b: Predicted mask tensor with dimensions [batch, h, w].
        :type pred_mask_b: torch.Tensor
        :param query_name_b: List of strings with the query image directories.
        :type query_name_b: Union[List[torch.Tensor], None]
        :return: List of PIL Images with the query images merged with predicted masks.
        :rtype: List[Image.Image]
        """

        merged_list = []
        # tensor moved from GPU to CPU 
        qry_img_b = to_cpu(qry_img_b)
        pred_mask_b = to_cpu(pred_mask_b)

        #apply mask to the qry_img
        for qry_img, pred_mask in zip(qry_img_b, pred_mask_b):
            merge_image_mask = cls.visualize_prediction(qry_img, pred_mask)
            merged_list.append(merge_image_mask)
        return merged_list
    
    @classmethod
    def visualize_prediction(cls, 
        qry_img:torch.Tensor, 
        pred_mask:torch.Tensor
        ) -> List[Image.Image]:
        """
        Apply mask to the given image.

        :param qry_img: Query image tensor with dimensions [3, h, w].
        :type qry_img: torch.Tensor
        :param pred_mask: Predicted mask tensor with dimensions [h, w].
        :type pred_mask: torch.Tensor
        :param query_name: Query image directory.
        :type query_name: str
        :return: List with batch_size elements of query PIL images merged with predicted masks.
        :rtype: List[Image.Image]
        """

        pred_color = cls.colors['red']

        qry_img = cls.to_numpy(qry_img, 'query')
        pred_mask = cls.to_numpy(pred_mask, 'mask')

        pred_masked_pil = Image.fromarray(cls.apply_mask(qry_img, pred_mask, pred_color))

        return pred_masked_pil

    @classmethod
    def to_numpy(cls, tensor:torch.Tensor, type:str) -> np.array:
        """
        Convert tensor (query and mask) to ndarray. 
        :param tensor: (tensor) query or mask tensor
        :param type: (str) type of tensor: query or mask
        :return: (ndarray)
        """

        if type == 'query':
            # to_pil change the order [C,H,W] to [H,W,C] e rescale values to range [0,255]
            return np.array(cls.to_pil(cls.unnormalize(tensor))).astype(np.uint8)
        elif type == 'mask':
            return np.array(tensor).astype(np.uint8)
        else:
            raise Exception('Undefined tensor type: %s' % type)
        
    @classmethod
    def unnormalize(cls, img: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize image with ImageNet weights.

        :param img: Image normalized tensor.
        :type img: torch.Tensor
        :return: Image unnormalized tensor.
        :rtype: torch.Tensor
        """

        img = img.clone()
        for im_channel, mean, std in zip(img, cls.mean_img, cls.std_img):
            im_channel.mul_(std).add_(mean)
        return img

    @classmethod
    def apply_mask(cls, image:np.ndarray, 
        mask: np.ndarray, 
        color: Tuple[int], 
        alpha: float = 0.5
        ) -> np.ndarray:
        """
         Apply mask to the given image. 

        :param image: The query image.
        :type image: np.ndarray
        :param mask: The mask image.
        :type mask: np.ndarray
        :param color: Tuple of integers representing the color of the pixel
        :type color: Tuple of integers
        :param alpha: The alpha value
        :type alpha: float
        :return: The query image merged with mask.
        :rtype: np.ndarray
        """

        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image
