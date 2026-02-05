from .base.vit import VisionTransformer, Transformer, LayerNorm
from fewpy.util.inference.register import register_constructor

from typing import Tuple, Union
from .config import AnomalyCLIPConfig

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import cv2

from scipy.ndimage import gaussian_filter
from .base.prompt_ensemble import AnomalyCLIP_PromptLearner

import sys
import os


def get_similarity_map(sm, shape):
    side = int(sm.shape[1] ** 0.5)
    sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
    sm = torch.nn.functional.interpolate(sm, shape, mode="bilinear")
    sm = sm.permute(0, 2, 3, 1)
    return sm


def compute_similarity(image_features, text_features, t=2):
    prob_1 = image_features[:, :1, :] @ text_features.t()
    b, n_t, n_i, c = (
        image_features.shape[0],
        text_features.shape[0],
        image_features.shape[1],
        image_features.shape[2],
    )
    feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(
        1, 1, n_t, c
    )
    similarity = feats.sum(-1)
    return (similarity / 0.07).softmax(-1), prob_1


class AnomalyCLIP(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        config: AnomalyCLIPConfig,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        design_details=None
    ):
        
        super().__init__()

        self.config = config if config is not None else AnomalyCLIPConfig()
        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            text_layer=True,
            design_details=design_details,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, embed_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cls_id = config.cls_id
        self.prompt_learner = None

        self.initialize_parameters()

    def initialize_parameters(self):

        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(
                self.text_projection, std=self.transformer.width**-0.5
            )

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):

        return self.visual.conv1.weight.dtype


    def encode_image(
        self,
        image,
        feature_list=[],
        ori_patch=False,
        proj_use=True,
        DPAM_layer=None,
        ffn=False,
    ):
        
        return self.visual(
            image.type(self.dtype),
            self.config.feature_list,
            ori_patch=ori_patch,
            proj_use=proj_use,
            DPAM_layer=DPAM_layer,
            ffn=ffn,
        )

    def encode_text(self, text):

        x = self.token_embedding(text).type(
            self.dtype
        )  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            @ self.text_projection
        )

        return x

    def encode_text_learn(
        self,
        prompts,
        tokenized_prompts,
        deep_compound_prompts_text=None
        # normalize: bool = False,
    ):
        
        cast_dtype = self.transformer.get_cast_dtype()

        # x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        # x = x + self.positional_embedding.to(cast_dtype)
        x = prompts + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if deep_compound_prompts_text is None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, deep_compound_prompts_text, 0])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(
            self.dtype
        )  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(
            dim=1, keepdim=True
        )
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        
        # return logits_per_image, logits_per_text

    def predict(self,
                x,
                s_x,
                s_y,
                user_tknized_prompts):
        """
        self.model.predict:
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * x: Tensor, batch of images in (B, C, H, W) format.
                * s_x: Tensor, batch of support images in (B, C, H, W) format.
                * s_y: Tensor, batch of ground truth images in (B, H, W) format.
                * user_tknized_prompts: list[int], tokenized text. Ideally open_clip.tokenize is
                used with fewpy.util.inference.preprocessor.Preprocessor so that you only need
                to pass a list of strings to FewShotModel 
        Returns:
                list[dict]:
                    Each dict is corresponds to the output of a single input image.
                    The dict contains a string "segmentation" under the key "task" to specify the task type,
                    a "data" mask, Tensor of format (H, W) and a "postproc_data" mask, Tensor of format (H, W)
        """

        with torch.no_grad():

            epsilon = 1e-8

            prompts, tknized_prompts, compound_prompts_text = self.prompt_learner(cls_id=self.cls_id)
            text_features = self.encode_text_learn(
                prompts,
                tknized_prompts,
                compound_prompts_text
            ).float()                                                                                           # -> [K, D]
            K, D = text_features.shape
            text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)                     # -> [1, K, D]
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + epsilon)                # -> [1, K, D]
            text_normal_ref = text_features[:, 0]                                                               # -> [D]
            text_anomalous_ref = text_features[:, 1]                                                            # -> [D]

            user_anomalous_ref = None
            if user_tknized_prompts.numel():
                all_text_features = self.encode_text_learn(
                    prompts=self.token_embedding(user_tknized_prompts).type(self.dtype),
                    tokenized_prompts=user_tknized_prompts,
                    deep_compound_prompts_text=[]
                ).float()                                                                                       # -> [P, D]
                user_anomalous_ref = all_text_features.mean(dim=0, keepdim=True)                                # -> [1, D]
                user_anomalous_ref = user_anomalous_ref / user_anomalous_ref.norm(dim=-1, keepdim=True)         # -> [1, D]
            
            _, support_patch_features_multiscale = self.encode_image(s_x, self.config.feature_list, DPAM_layer=20) # -> list[Tensor] of L tensors [S, N, D]

            final_prototypes_per_layer = []
            for support_patch_features in support_patch_features_multiscale:
                S, N, D = support_patch_features.shape
                grid_size = int(np.sqrt(N - 1))

                resized_masks = F.interpolate(s_y.unsqueeze(1).float(), size=(grid_size, grid_size), mode='nearest').squeeze(1) # -> [S, G, G]
                resized_masks_flat = resized_masks.view(S, -1) > 0                                              # -> [S, G*G] (bool)
                
                patches_only = support_patch_features[:, 1:, :]                                                 # -> [S, N-1, D]
                patches_only_flat = patches_only.reshape(-1, D)                                                 # -> [S*(N-1), D]
                mask_flat = resized_masks_flat.reshape(-1)                                                      # -> [S*(N-1)] (bool)

                anomalous_patches = patches_only_flat[mask_flat]                                                # -> [NumAnom, D]
                normal_patches = patches_only_flat[~mask_flat]                                                  # -> [NumNorm, D]

                if anomalous_patches.shape[0] == 0: raise ValueError("No anomalous patches in the suppport set.")
                if normal_patches.shape[0] == 0: raise ValueError("No normal patches in the support set.")

                if user_anomalous_ref is not None and anomalous_patches.numel() > 0:
                    anomalous_patches_norm = anomalous_patches / anomalous_patches.norm(dim=-1, keepdim=True)   # -> [NumAnom, D]
                    similarities_to_prompt = anomalous_patches_norm @ user_anomalous_ref.T                      # -> [NumAnom, 1]
                    weights = F.softmax(similarities_to_prompt / self.config.softmax_temp, dim=0)                           # -> [NumAnom, 1]
                    visual_anomalous_reference = (weights * anomalous_patches).sum(dim=0)                       # -> [D]
                else:
                    visual_anomalous_reference = anomalous_patches.mean(dim=0)                                  # -> [D]

                visual_normal_reference = normal_patches.mean(dim=0)                                            # -> [D]

                final_normal_ref = (self.config.alpha * visual_normal_reference) + ((1 - self.config.alpha) * text_normal_ref)          # -> [D]
                final_anomalous_ref = (self.config.beta * visual_anomalous_reference) + ((1 - self.config.beta) * text_anomalous_ref)   # -> [D]

                final_normal_ref = final_normal_ref / (final_normal_ref.norm(dim=-1, keepdim=True) + epsilon)               # -> [D]
                final_anomalous_ref = final_anomalous_ref / (final_anomalous_ref.norm(dim=-1, keepdim=True) + epsilon)      # -> [D]


                layer_prototypes = torch.stack([final_normal_ref, final_anomalous_ref])                         # -> [K, D]
                final_prototypes_per_layer.append(layer_prototypes)                                             # -> list[Tensor] of L tensors [K, D]

            _, query_patch_features_multiscale = self.encode_image(x, self.config.feature_list, DPAM_layer=20)              # -> list[Tensor] pf L tensors [B, N, D]
            
            anomaly_map_list = []
            for i, query_patch_feature in enumerate(query_patch_features_multiscale):
                patch_feature_normalized = query_patch_feature / (query_patch_feature.norm(dim=-1, keepdim=True) + epsilon) # -> [B, N, D]
                prototypes_for_this_layer = final_prototypes_per_layer[i]                                       # -> [K, D]
                prototypes_for_this_layer = prototypes_for_this_layer.squeeze(dim=1)

                similarity, _ = compute_similarity(patch_feature_normalized, prototypes_for_this_layer)         # -> [B, N, K]
                similarity = similarity / self.config.contrast                                                              # -> [B, N, K]
                
                similarity_map = get_similarity_map(similarity[:, 1:, :], self.config.image_size)                           # -> [B, H, W, K]
                anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0                       # -> [B, H, W]
                anomaly_map_list.append(anomaly_map)                                                            # -> list[Tensor] of L tensors [B, H, W]
                        
            if len(self.config.scale_weights) != len(anomaly_map_list):
                raise ValueError("The number of weights and scales must be equal.")
            
            stacked_maps = torch.stack(anomaly_map_list)                                                        # -> [L, B, H, W]
            weights = torch.tensor(self.config.scale_weights)                                                               # -> [L]
            weights = weights.view(len(self.config.scale_weights), 1, 1, 1)                                                 # -> [L, 1, 1, 1]
            anomaly_map = (stacked_maps * weights).sum(dim=0)                                                   # -> [B, H, W]
            
            anomaly_map_device = anomaly_map.detach()
            object_mask = anomaly_map_device > self.config.obj_threshold
            object_pixels = anomaly_map_device[object_mask]
            
            if object_pixels.numel() > 0:
                median_val = torch.median(object_pixels) 
                
                map_processed = anomaly_map_device - median_val
                map_processed = torch.clamp(map_processed, min=0)
                max_val = map_processed.max()
                
                if max_val.item() > 1e-6: 
                    map_processed = map_processed / max_val
                    
                map_processed = map_processed ** self.config.gamma
                final_map = map_processed * object_mask.float() 
            else:
                final_map = anomaly_map_device                                                                 # -> [B, H, W]
                        
            final_map_filtered = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=self.config.sigma)) for i in final_map])    # -> [B, H, W]

            results = []
            final_map_filtered = final_map_filtered.unsqueeze(1)
            for map in final_map_filtered:

                mask = map.squeeze().cpu().detach().numpy()
                mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
                mask = cv2.GaussianBlur(mask, (7, 7), 0)
                mask = torch.from_numpy((mask > 0.5).astype(np.uint8) * 255)
        
                results.append({
                    "task": "segmentation",
                    "raw_data": map,
                    "data": mask
                })

            return results


@register_constructor(name="anomalyCLIP", config_cls=AnomalyCLIPConfig)
class contructor_AnomalyCLIP:

    model_cls_mame = "anomalyCLIP"
 
    def __init__(self, config: AnomalyCLIPConfig):
        
        self.config = config

    def instantiate_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # if self.config.checkpoint:
        #     state_dict = self.config.checkpoint
        # else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "weights", "ViT-L-14-336px.pt")
        if not os.path.exists(model_path):
            main_dir = sys.path[0]
            model_path = os.path.join(main_dir, "weights", "ViT-L-14-336px.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError("ViT weights not found!")
        
        model = torch.jit.load(model_path, map_location=device)
        state_dict = model.state_dict()

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith("visual.")
                and k.endswith(".attn.in_proj_weight")
            ]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5
        )
        image_resolution = vision_patch_size * grid_size
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(
            set(
                k.split(".")[2]
                for k in state_dict
                if k.startswith("transformer.resblocks")
            )
        )

        dsgn_details = {"Prompt_length": self.config.n_ctx, "learnabel_text_embedding_depth": self.config.depth, "learnabel_text_embedding_length": self.config.t_n_ctx}

        model = AnomalyCLIP(
            embed_dim,
            self.config,
            image_resolution,
            vision_layers,
            vision_width,
            vision_patch_size,
            context_length,
            vocab_size,
            transformer_width,
            transformer_heads,
            transformer_layers,
            design_details=dsgn_details,
        )

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in state_dict:
                del state_dict[key]

        model.load_state_dict(state_dict)
        model.prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), dsgn_details)
        model.prompt_learner.to(device)

        model.to(device)
        model.visual.DAPM_replace(DPAM_layer=20)

        return model.eval(), device
