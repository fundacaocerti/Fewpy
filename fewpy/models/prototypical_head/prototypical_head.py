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

from fewpy.metrics import DistanceMetric

from fewpy.models.tipAdapter.tip_adapter import tokenize

from torch import nn

from pathlib import Path


class PrototypicalHead(torch.nn.Module):

    def __init__(self, config):
        super(PrototypicalHead, self).__init__()
        self.config = config
        self.device = None
        self.backbone = None
        self.textual = None 
        self.temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.class_scales = nn.Parameter(torch.zeros(
            self.config.n_classes, 
            self.config.embedding_size)) if self.config.task == "classification" else None

        self.textual_scale = self.config.textual_scale
        
    # def _apply_nms(self, bboxes, scores, class_ids, iou_threshold):

    #     keep = batched_nms(bboxes, scores, class_ids, iou_threshold)
    #     return bboxes[keep], scores[keep], class_ids[keep]
    
    def train(self, mode = True):

        super().train(mode)
        self.backbone.eval()

        if self.class_scales is not None:
            self.class_scales.requires_grad_(mode)
        
        self.temp.requires_grad_(mode)

        return self
    
    def parameters(self, recurse=True):
        
        for name, param in self.named_parameters(recurse=recurse):
            if "backbone" not in name and "textual" not in name:
                yield param

    def gen_prototypes(self, support_features, support_gt):

        prototypes = []
        unique_labels = torch.unique(support_gt)

        if self.config.cache_prototypes:
            cache_dir = Path(self.config.cache_dir)
            prototypes_path = cache_dir / f"{self.config.kshot}_shot.pt"
            if prototypes_path.exists():
                return torch.load(prototypes_path), unique_labels

        num_aug = self.config.augment_epoch
        support_set_size = unique_labels.shape[0] * self.config.kshot
        # print(support_features.shape[0], num_aug, support_set_size)
        assert support_features.shape[0] == num_aug * support_set_size
        if num_aug > 0:
            support_features = support_features.view(num_aug, support_set_size, -1).mean(dim=0)

        # print("proto gen sup features", support_features.shape)
        if support_gt.shape[0] > support_set_size: support_gt = support_gt[:support_set_size]
        for label in unique_labels:
            mask = (support_gt == label)
            feature_subset = support_features[mask]
            
            proto = feature_subset.mean(dim=0)

            prototypes.append(proto)

        prototypes = torch.stack(prototypes)

        if self.config.cache_prototypes:
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save(prototypes, prototypes_path)

        return prototypes, unique_labels

    def encode_image(self, batch: torch.Tensor):

        return self.backbone(batch.to(self.device).to(self.config.torch_dtype)).float()

    def gen_visual_logits(self, query_features, support_features, prototypes, distance_metric):

        query_features_norm = F.normalize(query_features, p=2, dim=1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)
        
        match distance_metric:
            case DistanceMetric.EUCLIDEAN:
                logits = -torch.cdist(query_features, prototypes, p=2)**2 * self.temp.exp()
                logits = F.softmax(logits, dim=1)

            case DistanceMetric.COSINE_SIMILARITY:
                logits = (query_features_norm @ prototypes_norm.t()) * self.temp.exp()
                logits = F.softmax(logits, dim=1)

            case DistanceMetric.LEARNABLE_MAHALANOBIS:
                diff = query_features.unsqueeze(1) - prototypes.unsqueeze(0)
                precision = torch.exp(
                    self.class_scales
                ).to(self.device).to(self.config.torch_dtype) # [K, D]

                dist_sq = torch.sum((diff ** 2) * precision.unsqueeze(0), dim=-1) # [B, K]
                logits = -dist_sq * self.temp.exp()
                logits = F.softmax(logits, dim=1)

            case DistanceMetric.FULL_MATRIX_MAHALANOBIS:
                diff = query_features_norm.unsqueeze(1) - prototypes_norm.unsqueeze(0) # [B, K, D]
                support_features_norm = F.normalize(support_features, p=2, dim=1)
                if support_features_norm.shape == prototypes_norm.shape: 
                    all_diffs = support_features_norm - prototypes_norm
                else: 
                    all_diffs = support_features_norm - prototypes_norm.unsqueeze(1)       # [..., D]
                B, K, D = diff.shape
                all_diffs_flat = all_diffs.view(-1, D)
                N = all_diffs_flat.shape[0]
                
                global_covariance = (all_diffs_flat.T @ all_diffs_flat) / N 
                reg_matrix = torch.eye(D, device=all_diffs.device, dtype=all_diffs.dtype) * self.config.precision_regularizer
                global_covariance = global_covariance + reg_matrix # [D, D]

                diff_flat_T = diff.view(-1, D).T 
                precision_diff_T = torch.linalg.solve(global_covariance, diff_flat_T)
                precision_diff = precision_diff_T.T.view(B, K, D)
                
                dist_sq = torch.sum(diff * precision_diff, dim=-1) # [B, K]
                logits = -dist_sq * self.temp.exp()
                logits = F.softmax(logits, dim=1)

            case DistanceMetric.DIAGONAL_MAHALANOBIS:
                diff = query_features_norm.unsqueeze(1) - prototypes_norm.unsqueeze(0)
                # print(support_features.shape, prototypes.shape)
                support_features_norm = F.normalize(support_features, p=2, dim=1)
                # print(support_features_norm == prototypes_norm)
                if support_features_norm.shape == prototypes_norm.shape: 
                    all_diffs = support_features_norm - prototypes_norm
                else: 
                    all_diffs = support_features_norm - prototypes_norm.unsqueeze(1)       # [..., D]
                # print(all_diffs)
                global_variance = all_diffs.pow(2).mean(dim=(0, 1)) # [D]
                
                # print(global_variance)
                # exit(1)
                
                precision = 1.0 / (global_variance + self.config.precision_regularizer)
                dist_sq = torch.sum((diff ** 2) * precision.unsqueeze(0), dim=-1) # [B, K]
                logits = -dist_sq * self.temp.exp()
                logits = F.softmax(logits, dim=1)

            case DistanceMetric.KV:
                k = prototypes_norm
                affinity = query_features_norm @ k.t()
                logits = (2 * affinity - 2).exp()

            case DistanceMetric.SOFTMAX_KV:
                k = prototypes_norm
                affinity = query_features_norm @ k.t()
                logits = (2 * affinity - 2).exp()
                logits = F.softmax(logits, dim=1)

            case _:
                raise ValueError(f"Distance metric cannot be specified [{distance_metric}]!")

        return logits, query_features_norm

    def forward(
            self, 
            query: torch.Tensor, 
            support_images: torch.Tensor, 
            support_gt: torch.Tensor,
            query_features: torch.Tensor,
            support_features: torch.Tensor,
            prompts: list[str],
            classnames: list[str],
            distance_metric: DistanceMetric,
        ):

        # print(query.shape, support_images.shape, support_gt.shape)
        if query_features is None:
            assert query is not None
            query = query.to(self.device).to(self.config.torch_dtype)
            query_features = self.backbone(query).float()
        if support_features is None:
            assert support_features is not None
            support_images = support_images.to(self.device).to(self.config.torch_dtype)
            support_features = self.backbone(support_images).float()

        prototypes, labels = self.gen_prototypes(support_features, support_gt)
        # labels = torch.unique(support_gt)

        logits, query_features_norm = self.gen_visual_logits(query_features, support_features, prototypes, distance_metric)

        if self.textual is not None and len(prompts) > 0 and len(classnames) > 0 and self.config.textual_scale > 0.0:
            with torch.no_grad():
                textual_features = []
                for classname in classnames:
                    classname = classname.replace('_', ' ')
                    prompt = [t.format(classname) for t in prompts]
                    text_tokens = tokenize(prompt).to(self.device)
                    text_features = self.textual(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_features = text_features.mean(dim=0).squeeze()
                    text_features /= text_features.norm()
                    textual_features.append(text_features)
                textual_features = torch.stack(textual_features, dim=1).to(self.device).float()
            textual_logits = (query_features_norm @ textual_features) # * self.temp.exp()
            logits += F.softmax(textual_logits, dim=1) * self.textual_scale
            # logits += textual_logits * self.textual_scale
            
        return logits, labels

    def predict(
            self, 
            x: torch.Tensor=None, 
            s_x: torch.Tensor=None, 
            s_y: torch.Tensor=None,
            query_features: torch.Tensor=None,
            support_features: torch.Tensor=None,
            prompts: list[str]=[],
            classnames: list[str]=[],
            distance_metric: str=DistanceMetric.EUCLIDEAN,
            textual_scale: int=None,
        ):

        assert s_y is not None
        
        if self.class_scales is not None:
            self.class_scales.to(self.device).to(self.config.torch_dtype)

        if textual_scale is not None:
            self.textual_scale = textual_scale
        else:
            self.textual_scale = self.config.textual_scale

        logits, labels = self.forward(
            x, s_x, s_y, 
            query_features,
            support_features,
            prompts, 
            classnames, 
            distance_metric
        )
        
        if self.training:
            return logits

        out = labels[logits.argmax(dim=-1).detach().cpu()]
        results = []
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
        bacbone_name = self.args.backbone.replace("/", "-")
        model_path = main_dir / "weights" / f"{bacbone_name}.pt"
        if not model_path.exists():
            main_dir = Path(sys.path[0])
            model_path = main_dir / "weights" / f"{bacbone_name}.pt"
        if not model_path.exists():
            model.backbone, model.textual = BackboneFactory.get_backbone(
                self.args.backbone, 
                keep_avg_pool=self.args.task == "classification",
                cache_dir=main_dir / "weights",
            )
        else:
            full_backbone = torch.jit.load(model_path, map_location=device)
            state_dict = full_backbone.state_dict()
            embed_dim = state_dict["text_projection"].shape[1]
            transformer_width = state_dict["ln_final.weight"].shape[0]
            context_length = state_dict["positional_embedding"].shape[0]
            
            model.backbone, model.textual = BackboneFactory.extract_encoders(full_backbone, keep_avg_pool=self.args.task == "classification", device=device)

            model.textual.text_proj = torch.nn.Parameter(torch.empty(transformer_width, embed_dim)).to(device)
            model.textual.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width)).to(device)
            with torch.no_grad():
                model.textual.text_proj.copy_(state_dict["text_projection"])
                model.textual.text_proj = model.textual.text_proj.to(self.args.torch_dtype)
                model.textual.positional_embedding.copy_(state_dict["positional_embedding"])
                model.textual.positional_embedding = model.textual.positional_embedding.to(self.args.torch_dtype)

        if not isinstance(model.textual, torch.jit.ScriptModule):
            model.textual = torch.jit.script(model.textual)
            model.textual = torch.jit.freeze(model.textual.eval())

        parent_path = model_path.parent
        for state_dict_path in model_path.parent.glob("*.pth"):
            state_dict = torch.load(state_dict_path)
            if "temp" in state_dict or "class_scales" in state_dict:
                try:
                    model.load_state_dict(state_dict)
                    # print("Success with", state_dict_path)
                    break
                except Exception as e:
                    continue
        
        if self.args.training:
        
            return model.train(), device

        return model.eval(), device