from fewpy.models.register import register_constructor
from fewpy.util.download import download, model2url
from .config import PrototypicalHeadConfig

import sys
import torch
import warnings

import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.ops import batched_nms

from torch import nn

from pathlib import Path


class PrototypicalHead(torch.nn.Module):

    def __init__(self, config):
        super(PrototypicalHead, self).__init__()
        self.config = config
        self.backbone = None
        self.adapter = None 
        self.temp = getattr(config, 'temperature', 10.0)
        self.regressor = None
        self.nms = None
    
    def _apply_nms(self, bboxes, scores, class_ids, iou_threshold):

        keep = batched_nms(bboxes, scores, class_ids, iou_threshold)
        return bboxes[keep], scores[keep], class_ids[keep]
    
    def train(self, mode = True):

        super().train(mode)
        self.backbone.eval()

        if self.adapter is not None:
            self.adapter.train(mode)

        return self

    def gen_prototypes(self, support_features, support_gt):

        prototypes = []
        unique_labels = torch.unique(support_gt["labels"])

        if self.config.cache_prototypes:
            cache_dir = Path(self.config.cache_dir)
            prototypes_path = cache_dir / f"{self.config.kshot}_shot.pt"
            if prototypes_path.exists():
                return torch.load(prototypes_path), unique_labels
        
        for label in unique_labels:
            mask = (support_gt["labels"] == label)
            feat_subset = support_features[mask]
            
            match self.config.task:
                case "detection":
                    boxes_subset = support_gt["bboxes"][mask]
                    roi_feat = roi_align(feat_subset, boxes_subset, output_size=(7, 7), spatial_scale=1.0)
                    proto = roi_feat.mean(dim=[0, 2, 3]) 
                
                case "segmentation":
                    mask_subset = support_gt["masks"][mask].unsqueeze(1) # [K, 1, H, W]
                    mask_subset = F.interpolate(mask_subset, size=feat_subset.shape[-2:], mode='nearest')
                    proto = (feat_subset * mask_subset).sum(dim=[0, 2, 3]) / (mask_subset.sum() + 1e-6)
                
                case "classification":
                    proto = feat_subset.mean(dim=[0, 2, 3])

            prototypes.append(proto)

        if self.config.cache_prototypes:
            cache_dir.parent.mkdir(parents=True, exist_ok=True)
            torch.save(prototypes, prototypes_path)

        return torch.stack(prototypes), unique_labels

    def forward(self, query, support_images, support_gt):
        query_features = self.backbone.encode_image(query)
        support_features = self.backbone.encode_image(support_images)

        if self.training or getattr(self.config, 'load_adapter', False):
            query_features = self.adapter(query_features)
            support_features = self.adapter(support_features)

        prototypes, labels = self.gen_prototypes(support_features, support_gt)

        q_feat_norm = F.normalize(query_features, p=2, dim=1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)

        logits = F.conv2d(q_feat_norm, prototypes_norm.view(-1, query_features.shape[1], 1, 1)) * self.temp

        return logits, labels

    def predict(self, x, s_x, s_y):
        logits, labels = self.forward(x, s_x, s_y)
        out = None
        results = []

        match self.config.task:

            case "classification":
                pooled_logits = F.adaptive_avg_pool2d(logits, (1, 1)).flatten(1)
                out = labels[pooled_logits.argmax(dim=-1)].cpu().numpy()

            case "segmentation":
                full_res_logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
                out = full_res_logits.argmax(dim=1).cpu().numpy()

            case "detection":
                out = logits.cpu().numpy()
                bboxes, scores, class_ids = self.regressor(out)
                bboxes, scores, class_ids = self._apply_nms(
                    bboxes,
                    scores,
                    class_ids,
                    self.config.detection_threshold,
                )
            
            case _:
                raise ValueError("Task Should be classification, segmentation or detection")
        
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

        if self.args.backbone == "":
            raise ValueError("Backbone should be set either through the backbone parameter!")
        
        current_dir = Path(__file__).resolve().parent
        model_path = current_dir / "weights" / f"{self.args.backbone}.pt"
        if not model_path.exists():
            model_path = Path(download(self.args.backbone))

        self.args.backbone = torch.jit.load(model_path, map_location=device)
        model.backbone = self.args.backbone

        if self.args.load_adapter:
            current_dir = Path(__file__).resolve().parent
            model_path = current_dir / "weights"
            checkpoint_path = model_path.parent / "tip_adapter.pth"

            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Adapter weights not found at {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path)
            k1, k2 = "adapter.weight", "weight"
            if k1 in checkpoint.keys():
                adapter_weights = checkpoint[k1]
            elif k2 in checkpoint.keys():
                adapter_weights = checkpoint[k2]
            else:
                e = f"Could not find adapter weights in state_dict! Weights must be under {k1} or {k2}"
                raise KeyError(e)
            
            out_dim, in_dim = adapter_weights.shape
            model.adapter = nn.Conv2d(out_dim, in_dim, kernel_size=1).to(x.device)
            model.adapter.weight = nn.Parameter(adapter_weights)
            target_device = next(model.backbone.parameters()).device
            target_dtype = next(model.backbone.parameters()).dtype

            model.adapter.to(device=target_device, dtype=target_dtype)

        if self.args.task == "regression":
            # TODO - load regressor
            ...

        if self.args.training:
        
            return model.train(), device

        return model.eval(), device
