from fewpy.models.register import register_constructor
from fewpy.util.download import download, BackboneFactory
from .config import PrototypicalHeadConfig

import sys
import torch
import warnings
import numpy as np

import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.ops import batched_nms

from torch import nn

from pathlib import Path


class PrototypicalHead(torch.nn.Module):

    def __init__(self, config):
        super(PrototypicalHead, self).__init__()
        self.config = config
        self.device = None
        self.backbone = None
        self.adapter = None 
        self.temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.class_scales = nn.Parameter(torch.zeros(
            self.config.n_classes, 
            self.config.embedding_size)) if self.config.task == "classification" else None
        self.regressor = None
    
    def _apply_nms(self, bboxes, scores, class_ids, iou_threshold):

        keep = batched_nms(bboxes, scores, class_ids, iou_threshold)
        return bboxes[keep], scores[keep], class_ids[keep]
    
    def train(self, mode = True):

        super().train(mode)
        self.backbone.eval()

        if self.class_scales is not None:
            self.class_scales.requires_grad_(mode)
        
        self.temp.requires_grad_(mode)

        return self
    
    def parameters(self, recurse = True):
        params = []

        params += self.temp.parameters(recurse)
        params += self.class_scales.parameters(recurse) if self.class_scales is not None else []

        return params
        
    def named_parameters(self, prefix = "", recurse = True, remove_duplicate = True):
        
        params = []

        params += self.temp.named_parameters(prefix, recurse)
        params += [(prefix + "class_scales", self.class_scales)] if self.class_scales is not None else []

        return params

    def gen_prototypes(self, support_features, support_gt):

        prototypes = []
        unique_labels = torch.unique(support_gt["labels"])

        if self.config.cache_prototypes:
            cache_dir = Path(self.config.cache_dir)
            prototypes_path = cache_dir / f"{self.config.kshot}_shot.pt"
            if prototypes_path.exists():
                return torch.load(prototypes_path), unique_labels

        num_aug = self.config.augment_epochs
        support_set_size = support_gt["labels"].shape[0]
        assert support_features.shape[0] == num_aug * support_set_size
        if num_aug > 0:
            support_features = support_features.view(num_aug, support_set_size, -1).mean(dim=0)
        
        for label in unique_labels:
            mask = (support_gt["labels"] == label)
            feature_subset = support_features[mask]
            
            proto = feature_subset.mean(dim=0)

            prototypes.append(proto)

        prototypes = torch.stack(prototypes)

        if self.config.cache_prototypes:
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save(prototypes, prototypes_path)

        return prototypes, unique_labels

    def forward(self, query, support_images, support_gt):

        query = query.to(self.device).to(self.config.torch_dtype)
        support_images = support_images.to(self.device).to(self.config.torch_dtype)
        
        query_features = self.backbone(query)
        support_features = self.backbone(support_images)

        # L2 Normalization of support features
        support_features = F.normalize(support_features, p=2, dim=1)

        prototypes, labels = self.gen_prototypes(support_features, support_gt)
        # labels = torch.unique(support_gt["labels"])

        query_features_norm = F.normalize(query_features, p=2, dim=1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)

        # [B, 1, D] - [1, K, D] -> [B, K, D]
        diff = query_features_norm.unsqueeze(1) - prototypes_norm.unsqueeze(0)
        
        precision = torch.exp(self.class_scales) # [K, D]
        
        dist_sq = torch.sum((diff ** 2) * precision.unsqueeze(0), dim=-1) # [B, K]
        logits = -dist_sq * self.temp.exp() # [B, K]
        # logits = (query_features_norm @ prototypes_norm.t()) * self.temp.exp()
        # k = prototypes_norm
        # k = support_features.view(
        #     self.config.augment_epochs, support_gt["labels"].shape[0], -1,
        # ).mean(dim=0)
        # k /= k.norm(dim=-1, keepdim=True)
        # k = k.permute(1, 0)
        # k = k.contiguous()
        # affinity = query_features_norm @ k
        # v = F.one_hot(support_gt["labels"], num_classes=labels.shape[0]).to(self.device).to(self.config.torch_dtype)
        # logits = (2 * affinity - 2).exp() @ v
            
        return logits, labels

    def predict(self, x, s_x, s_y):
        logits, labels = self.forward(x, s_x, s_y)
        out = None
        results = []

        if self.training:
            return logits

        out = labels[logits.argmax(dim=-1)]
        for label in out:
            results.append({
                "data": label,
                "task": "classification",
            })

        return results


@register_constructor(name="PrototypicalHead", config_cls=PrototypicalHeadConfig)
class constructor_PrototypicalHead():
    def __init__(self, args):
            
        self.args = args

    def instantiate_model(self):

        model = PrototypicalHead(self.args)
        return self.load_weights(model)

    def load_weights(self, model):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.device = device

        if self.args.backbone == "":
            raise ValueError("Backbone should be set through the backbone parameter!")
        
        main_dir = Path(__file__).resolve().parent
        model_path = main_dir / "weights" / f"{self.args.backbone}.pt"
        if not model_path.exists():
            main_dir = Path(sys.path[0])
            model_path = main_dir / "weights" / f"{self.args.backbone}.pt"
        if not model_path.exists():
            model.backbone = BackboneFactory.get_backbone(
                self.args.backbone, 
                keep_avg_pool=self.args.task == "classification",
                cache_dir=main_dir,
            )
        else:
            full_backbone = torch.jit.load(model_path, map_location=device)
            model.backbone = BackboneFactory.extract_visual_encoder(full_backbone, keep_avg_pool=self.args.task == "classification", device=device)

        if self.args.training:
        
            return model.train(), device

        return model.eval(), device
