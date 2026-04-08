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

        # print("Labels shape:")
        # print(support_gt["labels"].shape)
        # print(support_gt["labels"])
        # print("Unique Labels shape:")
        # print(unique_labels.shape)
        # print(unique_labels)

        if self.config.cache_prototypes:
            cache_dir = Path(self.config.cache_dir)
            prototypes_path = cache_dir / f"{self.config.kshot}_shot.pt"
            if prototypes_path.exists():
                return torch.load(prototypes_path), unique_labels

        num_aug = self.config.augement_epochs
        support_set_size = support_gt["labels"].shape[0]
        # print(support_features.shape)
        # print(num_aug, support_set_size)
        assert support_features.shape[0] == num_aug * support_set_size
        if num_aug > 0:
            support_features = support_features.view(num_aug, support_set_size, -1).mean(dim=0)
        
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

        query = query.to(self.device).to(self.config.torch_dtype)
        support_images = support_images.to(self.device).to(self.config.torch_dtype)
        
        query_features = self.backbone(query)
        support_features = self.backbone(support_images)

        # query_features = F.normalize(query_features, p=2, dim=1)
        # support_features = F.normalize(support_features, p=2, dim=1)

        # print("support images shape", support_images.shape)
        # print("support features shape", support_features.shape)

        prototypes, labels = self.gen_prototypes(support_features, support_gt)

        query_features_norm = F.normalize(query_features, p=2, dim=1)
        # print(type(prototypes))
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)

        # print(prototypes.shape, prototypes_norm.shape)

        if query_features_norm.ndim == 2:
            logits = (query_features_norm @ prototypes_norm.t()) * self.temp.exp()
            # print("logits shape", logits.shape)
            # a = support_features.mean(dim=0)
            # k = support_features
            k = prototypes_norm
            # print(query_features_norm.shape, a.shape, support_features.shape)
            k /= k.norm(dim=-1, keepdim=True)
            k = k.permute(1, 0)
            k = k.contiguous()
            affinity = query_features_norm @ k
            # v = F.one_hot(support_gt["labels"]).to(self.device).to(self.config.torch_dtype)
            v = F.one_hot(labels, num_classes=labels.shape[0]).to(self.device).to(self.config.torch_dtype)
            # print("labels shape", labels.shape)
            # print("v shape", v.shape)
            alt_logits = (2 * affinity - 2).exp() @ v
            # print("alt logits shape", alt_logits.shape)
            # print(f"v: {v}, logits: {alt_logits}")
            # print("=============================")
        else:
            logits = F.conv2d(
                query_features_norm, 
                prototypes_norm.view(-1, query_features.shape[1], 1, 1)
            ) * self.temp

        return logits, labels, alt_logits

    def predict(self, x, s_x, s_y):
        logits, labels, *a = self.forward(x, s_x, s_y)
        out = None
        results = []

        # print("LOGITS:", logits.shape)

        match self.config.task:

            case "classification":
                pooled_logits = F.adaptive_avg_pool2d(logits, (1, 1)).flatten(1) if len(logits.shape) > 2 else logits
                out = labels[pooled_logits.argmax(dim=-1).cpu()].numpy()
                # alt_out = labels[a[0].argmax(dim=-1).cpu()].numpy()
                # alt_out2 = labels[(a[0] * 1.1 + logits).argmax(dim=-1).cpu()].numpy()
                # for label, alt_label, alt_label2 in zip(out, alt_out, alt_out2):
                for label in out:
                    results.append({
                        "data": label,
                        # "alt_labels": (alt_label, alt_label2),
                        # "logits": (logits, a[0], a[0] * 0.3 + logits),
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.device = device

        if self.args.backbone == "":
            raise ValueError("Backbone should be set either through the backbone parameter!")
        
        current_dir = Path(__file__).resolve().parent
        model_path = current_dir / "weights" / f"{self.args.backbone}.pt"
        if not model_path.exists():
            # print(f"Fewpy: '{self.args.backbone}' is a standard backbone, downloading and loading the model...")
            model.backbone = BackboneFactory.get_backbone(self.args.backbone, keep_avg_pool=self.args.task == "classification")
        else:
            # print(f"Fewpy: Found backbone weights at '{model_path}', loading the model...")
            full_backbone = torch.jit.load(model_path, map_location=device)
            model.backbone = BackboneFactory.extract_visual_encoder(full_backbone, keep_avg_pool=self.args.task == "classification", device=device)

        # if self.args.load_adapter:
        #     current_dir = Path(__file__).resolve().parent
        #     model_path = current_dir / "weights"
        #     checkpoint_path = model_path.parent / "tip_adapter.pth"

        #     if not checkpoint_path.exists():
        #         raise FileNotFoundError(f"Adapter weights not found at {checkpoint_path}")
            
        #     checkpoint = torch.load(checkpoint_path)
        #     k1, k2 = "adapter.weight", "weight"
        #     if k1 in checkpoint.keys():
        #         adapter_weights = checkpoint[k1]
        #     elif k2 in checkpoint.keys():
        #         adapter_weights = checkpoint[k2]
        #     else:
        #         e = f"Could not find adapter weights in state_dict! Weights must be under {k1} or {k2}"
        #         raise KeyError(e)
            
        #     out_dim, in_dim = adapter_weights.shape
        #     model.adapter = nn.Conv2d(out_dim, in_dim, kernel_size=1).to(x.device)
        #     model.adapter.weight = nn.Parameter(adapter_weights)

        #     model.adapter.to(device=device, dtype=self.args.torch_dtype)

        if self.args.task == "regression":
            # TODO - load regressor
            ...

        if self.args.training:
        
            return model.train(), device

        return model.eval(), device
