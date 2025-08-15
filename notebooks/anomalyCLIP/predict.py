import os
import torch
import torch.nn.functional as F
from model import AnomalyCLIP_lib 
from common.dataset import Dataset, FewShotDataset
from scipy.ndimage import gaussian_filter
import cv2
import random
import clip
import numpy as np
from model.utils import normalize
from model.prompt_ensemble import AnomalyCLIP_PromptLearner
import logging
from pydantic import BaseModel, Field
from PIL import Image
from torch.utils.data import DataLoader
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class AnomalyCLIPConfig(BaseModel):
    query_image_path: str = Field(
        ...,
    )
    checkpoint_path: str = Field(
        './model/checkpoints/9_12_4_multiscale/epoch_15.pth',
        description="Model checkpoint path"
    )
    features_list: list[int] = Field(
        default_factory=lambda: [6, 12, 18, 24],
        description="List of feature indices"
    )
    image_size: int = 700
    depth: int = 9
    n_ctx: int = 12
    t_n_ctx: int = 4

    kshot: int = 5
    alpha: float = Field(1.0, description="Visual weight for normal prototype")
    beta: float = Field(1.0, description="Visual weight for anomalous prototype")
    scale_weights: list[float] = Field(
        default_factory=lambda: [0.5, 1.0, 2.0, 3.0],
        description="Weights for the scales"
    )

    user_prompts: list[str] | None = Field(
        None,
        description="List of prompts to guide detection"
    )
    softmax_temp: float = Field(0.07, description="Softmax temperature for attention")
    seed: int = 111
    sigma: int = 4
    output_dir: str = './results'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def visualizer(path, anomaly_map, img_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(path)
    
    vis_bgr = cv2.imread(path)
    if vis_bgr is None:
        raise FileNotFoundError(f"It was not possible to read image {path}")
    
    vis = cv2.cvtColor(cv2.resize(vis_bgr, (img_size, img_size)), cv2.COLOR_BGR2RGB)
    mask = normalize(anomaly_map[0])
    vis_heatmap = apply_ad_scoremap(vis, mask)
    
    save_path = os.path.join(output_dir, f'anomaly_map_{filename}')
    cv2.imwrite(save_path, cv2.cvtColor(vis_heatmap, cv2.COLOR_RGB2BGR))

    pil_image = Image.fromarray(vis_heatmap)
    
    return vis_heatmap, save_path, pil_image


###### DICTIONARY #######
# B: Batch size (in this case, 1)
# S: Number of support images (k-shots, e.g., 4)
# C: Image channels (3 for RGB)
# H, W: Image height and width (e.g., 518)
# P: Number of user prompts (e.g., 3)
# L: Number of feature scales (e.g., 4)
# N: Number of Vision Transformer patches (e.g., 1370)
# D: Embedding dimension (768)
# G: Patch grid size (e.g., 37)
# K: Number of prototypes (2 for normal/anomalous)


def predict_anomalyclip(config: AnomalyCLIPConfig):
    # try:
    #     paths = ExperimentPaths(**config.model_dump())
    #     logger.info(f"Paths successfully validated for the experiment: {paths.root.name}")
    # except (ValidationError, FileNotFoundError, ValueError) as e:
    #     logger.error("Error configuring paths: {e}")
    #     return None

    setup_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    AnomalyCLIP_parameters = {"Prompt_length": config.n_ctx, "learnabel_text_embedding_depth": config.depth, "learnabel_text_embedding_length": config.t_n_ctx}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)
    model.eval()

    # query_file_path = os.path.join(config.experiment_path, "test", config.query_image_filename)

    Dataset.initialize(img_size=config.image_size)
    # dataloader = Dataset.build_dataloader(
    #     query=query_file_path, 
    #     experiment=config.experiment_path, 
    #     nworker=1, 
    #     shuffle=False, 
    #     bsz=1, 
    #     nshot=config.kshot
    # )
    dataset = FewShotDataset(query_path=config.query_image_path, shot=config.kshot, img_size=config.image_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    batch = next(iter(dataloader))
    query_img = batch['query_img'].to(device)                                                              # -> [B, C, H, W]
    support_imgs = batch['support_imgs'].squeeze(0).to(device)                                             # -> [S, C, H, W]
    support_masks = batch['support_masks'].squeeze(0).to(device)                                           # -> [S, H, W]

    print(query_img.size(), support_imgs.size(), support_masks.size())

    with torch.no_grad():
        epsilon = 1e-8

        prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)                    # prompts: [K, 77, D], tokenized_prompts: [K, 77]
        text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float() # -> [K, D]
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)                    # -> [1, K, D]
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + epsilon)               # -> [1, K, D]
        text_normal_ref = text_features[0, 0]                                                              # -> [D]
        text_anomalous_ref = text_features[0, 1]                                                           # -> [D]

        user_anomalous_ref = None
        if config.user_prompts:
            logger.debug(f"Encoding user prompts: '{config.user_prompts}'")
            text_inputs = clip.tokenize(config.user_prompts).to(device)                                    # -> [P, 77]
            all_text_features = model.encode_text_learn(
                prompts=model.token_embedding(text_inputs).type(model.dtype),
                tokenized_prompts=text_inputs,
                deep_compound_prompts_text=[]
            ).float()                                                                                      # -> [P, D]
            user_anomalous_ref = all_text_features.mean(dim=0, keepdim=True)                               # -> [1, D]
            user_anomalous_ref = user_anomalous_ref / user_anomalous_ref.norm(dim=-1, keepdim=True)        # -> [1, D]

        logger.info("Generating and refining multi-scale prototypes...")
        _, support_patch_features_multiscale = model.encode_image(support_imgs, config.features_list, DPAM_layer=20) # -> list[Tensor] of L tensors [S, N, D]

        final_prototypes_per_layer = []
        for support_patch_features in support_patch_features_multiscale:                                   # support_patch_features -> [S, N, D]
            S, N, D = support_patch_features.shape
            grid_size = int(np.sqrt(N - 1))

            resized_masks = F.interpolate(support_masks.unsqueeze(1).float(), size=(grid_size, grid_size), mode='nearest').squeeze(1) # -> [S, G, G]
            resized_masks_flat = resized_masks.view(S, -1) > 0                                             # -> [S, G*G] (bool)
            
            patches_only = support_patch_features[:, 1:, :]                                                # -> [S, N-1, D]
            patches_only_flat = patches_only.reshape(-1, D)                                                # -> [S*(N-1), D]
            mask_flat = resized_masks_flat.reshape(-1)                                                     # -> [S*(N-1)] (bool)

            anomalous_patches = patches_only_flat[mask_flat]                                               # -> [NumAnom, D]
            normal_patches = patches_only_flat[~mask_flat]                                                 # -> [NumNorm, D]

            if anomalous_patches.shape[0] == 0: raise ValueError("No anomalous patches in the support.")
            if normal_patches.shape[0] == 0: raise ValueError("No normal patches in the support.")

            if user_anomalous_ref is not None and anomalous_patches.numel() > 0:
                anomalous_patches_norm = anomalous_patches / anomalous_patches.norm(dim=-1, keepdim=True)    # -> [NumAnom, D]
                similarities_to_prompt = anomalous_patches_norm @ user_anomalous_ref.T                       # -> [NumAnom, 1]
                weights = F.softmax(similarities_to_prompt / config.softmax_temp, dim=0)                     # -> [NumAnom, 1]
                visual_anomalous_reference = (weights * anomalous_patches).sum(dim=0)                        # -> [D]
            else:
                visual_anomalous_reference = anomalous_patches.mean(dim=0)                                   # -> [D]

            visual_normal_reference = normal_patches.mean(dim=0)                                             # -> [D]
            
            final_normal_ref = (config.alpha * visual_normal_reference) + ((1 - config.alpha) * text_normal_ref)         # -> [D]
            final_anomalous_ref = (config.beta * visual_anomalous_reference) + ((1 - config.beta) * text_anomalous_ref)  # -> [D]

            final_normal_ref = final_normal_ref / (final_normal_ref.norm(dim=-1, keepdim=True) + epsilon)          # -> [D]
            final_anomalous_ref = final_anomalous_ref / (final_anomalous_ref.norm(dim=-1, keepdim=True) + epsilon) # -> [D]
            
            layer_prototypes = torch.stack([final_normal_ref, final_anomalous_ref], dim=0)                   # -> [K, D]
            final_prototypes_per_layer.append(layer_prototypes)                                              # -> list[Tensor] of L tensors [K, D]

        logger.info(f"Performing inference on {str(config.query_image_path.split('/')[-1])}...")
        _, query_patch_features_multiscale = model.encode_image(query_img, config.features_list, DPAM_layer=20) # -> list[Tensor] of L tensors [B, N, D]
        
        anomaly_map_list = []
        for i, query_patch_feature in enumerate(query_patch_features_multiscale):                          # query_patch_feature -> [B, N, D]
            patch_feature_normalized = query_patch_feature / (query_patch_feature.norm(dim=-1, keepdim=True) + epsilon) # -> [B, N, D]
            prototypes_for_this_layer = final_prototypes_per_layer[i]                                      # -> [K, D]
            
            similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature_normalized, prototypes_for_this_layer) # -> [B, N, K]
            similarity = similarity // 0.07                                                                         # -> [B, N, K]
            
            similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], config.image_size)   # -> [B, H, W, K]
            anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0                      # -> [B, H, W]
            anomaly_map_list.append(anomaly_map)                                                           # -> list[Tensor] of L tensors [B, H, W]
                    
        if len(config.scale_weights) != len(anomaly_map_list):
            raise ValueError("The number of weights and scales must be equal.")
        
        stacked_maps = torch.stack(anomaly_map_list)                                                       # -> [L, B, H, W]
        weights = torch.tensor(config.scale_weights, device=device)                                        # -> [L]
        weights = weights.view(len(config.scale_weights), 1, 1, 1)                                         # -> [L, 1, 1, 1]
        anomaly_map = (stacked_maps * weights).sum(dim=0)                                                  # -> [B, H, W]
        
        logger.debug("Starting advanced post-processing of anomaly map")
        anomaly_map_cpu = anomaly_map.detach().cpu()                                                       # -> [B, H, W]
        threshold = 0.1
        gamma = 1.5
        object_mask = anomaly_map_cpu > threshold                                                          # -> [B, H, W] (bool)
        object_pixels = anomaly_map_cpu[object_mask]                                                       # -> [NumObjPixels]
        
        if object_pixels.numel() > 0:
            median_val = torch.median(object_pixels)                                                       # -> [1] (scalar tensor)
            map_processed = anomaly_map_cpu - median_val                                                   # -> [B, H, W]
            map_processed = torch.clamp(map_processed, min=0)                                              # -> [B, H, W]
            max_val = map_processed.max()                                                                  # -> [1] (scalar tensor)
            
            if max_val > 1e-6:
                map_processed = map_processed / max_val                                                    # -> [B, H, W]
                
            map_processed = map_processed ** gamma                                                         # -> [B, H, W]
            final_map = map_processed * object_mask.float()                                                # -> [B, H, W]
        else:
            logger.warning("No objects detected with the current threshold.")
            final_map = anomaly_map_cpu                                                                    # -> [B, H, W]
            
        final_map_filtered = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=config.sigma)) for i in final_map]) # -> [B, H, W]
        
        logger.debug("Post-processing complete. Generating preview")
        vis_image, save_path, pil_image = visualizer(config.query_image_path, final_map_filtered.numpy(), config.image_size, config.output_dir)

        logger.info(f"Visualization saved to {save_path}")
        return {
            "final_map": final_map_filtered.squeeze().numpy(),
            "visualization": vis_image,
            "save_path": save_path,
            "pil_image": pil_image
        }
