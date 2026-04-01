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
    
    def _apply_nms(self, bboxes, scores, class_ids, iou_threshold):

        keep = batched_nms(bboxes, scores, class_ids, iou_threshold)
        return bboxes[keep], scores[keep], class_ids[keep]
    
    def train(self, mode = True):

        super().train(mode)
        self.backbone.eval()

        if self.adapter is not None:
            self.adapter.train(mode)

        return self
    
    def parameters(self, recurse = True):
        params = []
        if self.adapter is not None:
            params += self.adapter.parameters(recurse)

        if self.regressor is not None:
            params += self.regressor.parameters(recurse)

        return params
        
    def named_parameters(self, prefix = "", recurse = True, remove_duplicate = True):
        
        params = []
        if self.adapter is not None:
            params += self.adapter.named_parameters(prefix, recurse, remove_duplicate)

        if self.regressor is not None:
            params += self.regressor.pnamed_arameters(prefix, recurse, remove_duplicate)

        return params

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
            feature_subset = support_features[mask]

            is4d = len(feature_subset.shape) == 4
            
            match self.config.task:
                case "detection":
                    if is4d and "bboxes" in support_gt:
                        boxes_subset = support_gt["bboxes"][mask]
                        roi_feat = roi_align(feature_subset, boxes_subset, output_size=(7, 7), spatial_scale=1.0)
                        proto = roi_feat.mean(dim=[0, 2, 3]) 
                    else:
                        proto = feature_subset.mean(dim=0) if not is4d else feature_subset.mean(dim=[0, 2, 3])
                case "segmentation":
                    if is4d and "masks" in support_gt:
                        mask_subset = support_gt["masks"][mask].unsqueeze(1) 
                        mask_subset = F.interpolate(mask_subset, size=feature_subset.shape[-2:], mode='nearest')
                        proto = (feature_subset * mask_subset).sum(dim=[0, 2, 3]) / (mask_subset.sum() + 1e-6)
                    else:
                        proto = feature_subset.mean(dim=0) if not is4d else feature_subset.mean(dim=[0, 2, 3])
                case "classification":
                    proto = feature_subset.mean(dim=[0, 2, 3]) if is4d else feature_subset.mean(dim=0)

            prototypes.append(proto)

        prototypes = torch.stack(prototypes)

        if self.config.cache_prototypes:
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save(prototypes, prototypes_path)

        return prototypes, unique_labels

    def forward(self, query, support_images, support_gt):
        query_features = self.backbone(query)
        support_features = self.backbone(support_images)

        # print("images shape", support_images.shape)
        # print("features shape", support_features.shape)

        if self.training or getattr(self.config, 'load_adapter', False):
            query_features = self.adapter(query_features)
            support_features = self.adapter(support_features)

        prototypes, labels = self.gen_prototypes(support_features, support_gt)

        query_features_norm = F.normalize(query_features, p=2, dim=1)
        # print(type(prototypes))
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)

        if query_features_norm.ndim == 2:
            logits = (query_features_norm @ prototypes_norm.t()) * self.temp
        else:
            logits = F.conv2d(
                query_features_norm, 
                prototypes_norm.view(-1, query_features.shape[1], 1, 1)
            ) * self.temp

        return logits, labels

    def predict(self, x, s_x, s_y):
        logits, labels = self.forward(x, s_x, s_y)
        out = None
        results = []

        # print("LOGITS:", logits.shape)

        match self.config.task:

            case "classification":
                pooled_logits = F.adaptive_avg_pool2d(logits, (1, 1)).flatten(1) if len(logits.shape) > 2 else logits
                out = labels[pooled_logits.argmax(dim=-1).cpu()].cpu().numpy()
                for label in out:
                    results.append({
                        "data": label,
                        "task": "classification",
                    })

            case "segmentation":
                full_res_logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
                out = full_res_logits.argmax(dim=1).cpu().numpy()
                for segment in out:
                    results.append({
                        "data": segment,
                        "task": "segmentation",
                    })

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

        def extract_visual_encoder(model):
            
            visual = None
            if hasattr(model, "visual"): 
                visual = model.visual
            elif hasattr(model, "visual_tower"): 
                visual = model.visual
            dummy_input = torch.randn(1, 3, self.args.img_h, self.args.img_w)
            backbone = torch.jit.trace(visual, dummy_input)

            del model
            torch.cuda.empty_cache()

            return backbone

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.args.backbone == "":
            raise ValueError("Backbone should be set either through the backbone parameter!")
        
        current_dir = Path(__file__).resolve().parent
        model_path = current_dir / "weights" / f"{self.args.backbone}.pt"
        if not model_path.exists():
            model_path = Path(download(self.args.backbone))

        full_backbone = torch.jit.load(model_path, map_location=device)
        model.backbone = extract_visual_encoder(full_backbone)

        print(model.backbone)

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
