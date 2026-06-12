from fewpy.inference import FewShotModel
from fewpy.metrics import DistanceMetric
from fewpy.util.data import EpisodicDataset, PreprocessingMethod, episodic_collate

from pathlib import Path
from PIL import Image

import xml.etree.ElementTree as ET
import json
import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from tqdm import tqdm
from copy import deepcopy

import random
import pickle
from scipy import stats

from extract_data import get_data

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

verbose = False
E = 600
M = 5 
K = 1
Q = 15
batch_size = 75
IMG_SIZE = 224

cache_dir = Path("/home") / "Fewpy" / "cache"
key_cache = Path("/home") / "Fewpy" / "keys"
value_cache = Path("/home") / "Fewpy" / "values"
feature_cache_path = Path("/home") / "Fewpy" / "features" / "features.pt"

cache_dirs = [cache_dir, key_cache, value_cache]

textual_scale = 130
aug_epoch = 12

# GET DATASET
data = get_data("cub", K, M, Q, E)

cls2int = data["cls2int"]
CLASSES = data["classes"]

data = data["data"]

images = []
labels = data["cls"]

for path in data["path"]:
    images.append(Image.open(path).convert("RGB"))

feature_cache = dict()
# if not feature_cache_path.exists():
#     feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
# else:
#     feature_cache = torch.load(feature_cache_path)

def get_features(imgs, ids):
    missing_images = []
    missing_ids = []
    
    for img, id_ in zip(imgs, ids):
        if id_ not in feature_cache:
            missing_images.append(img)
            missing_ids.append(id_)
            
    if len(missing_images) > 0:
        missing_tensors = torch.stack(missing_images)
        new_features = model.encode_image(missing_tensors)
        
        feature_cache.update({
            id_: feat for id_, feat in zip(missing_ids, new_features)
        })
        
    features = torch.stack([feature_cache[id_] for id_ in ids])
    return features

def print_fsl_results(episodic_results, metrics):
    print("\n" + "="*60)
    print(f"{'METRIC':<25} | {'ACCURACY (Mean ± 95% CI)':<25}")
    print("="*60)

    all_metrics = metrics + ["TipAdapter"]
    metric_accuracies = {metric: [] for metric in all_metrics}
    
    for ep_id, ep_data in episodic_results.items():
        for metric in all_metrics:
            if metric in ep_data:
                acc_percentage = ep_data[metric]["accuracy"] * 100
                metric_accuracies[metric].append(acc_percentage)
                
    for metric in all_metrics:
        accs = np.array(metric_accuracies[metric])
        
        if len(accs) == 0:
            print(f"{metric:<25} | No data found.")
            continue
            
        mean_acc = np.mean(accs)
        
        n_episodes = len(accs)
        if n_episodes > 1:
            std_err = stats.sem(accs)
            ci_margin = std_err * stats.t.ppf((1 + 0.95) / 2., n_episodes - 1)
        else:
            ci_margin = 0.0
            
        print(f"{metric:<25} | {mean_acc:6.2f}% ± {ci_margin:4.2f}%")
        
    print("="*60 + "\n")

metrics = [
    # DistanceMetric.EUCLIDEAN,
    DistanceMetric.COSINE_SIMILARITY,
    # DistanceMetric.LEARNABLE_MAHALANOBIS,
    # DistanceMetric.FULL_MATRIX_MAHALANOBIS,
    DistanceMetric.DIAGONAL_MAHALANOBIS,
    DistanceMetric.SOFTMAX_KV,
    # DistanceMetric.KV,
]

args = {
    "kshot": K,
    "backbone": "ViT-B/32",
    "task": "classification",
    "cache_prototypes": True,
    "cache_dir": str(cache_dir),
    "dtype": "float16",
    "augment_epoch": aug_epoch,
    "n_classes": M,
    "embedding_size": 512,
    "textual_scale": textual_scale,
    # "precision_regularizer": 1e-8,
}

# print("GAMMA:", args["precision_regularizer"])

model = FewShotModel("PrototypicalHead", config=args)
print("TEXTUAL SCALE:", args["textual_scale"])

args = {
    "kshot": K,
    "clip_model": "ViT-B/32",
    "augment_epoch": aug_epoch,
    "show_logits": True,
    "alpha": textual_scale,
}
tip_adapter = FewShotModel("TipAdapter", config=args)

ds = EpisodicDataset(
    M,    # M-way
    K,    # K-shot
    Q,    # Number of query images / episode
    E,    # Number of episodes
    images,
    labels,
    seed_interval=120,
    base_seed=42,
    img_size=IMG_SIZE,
    center_crop=IMG_SIZE,
    pixel_norm=((0.48145466, 0.48145466, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),       # (mean, std))
    support_set_preprocessing_method=PreprocessingMethod.AUGMENT_SUPPORT_IMAGES,
    # support_set_preprocessing_method=PreprocessingMethod.STANDARD,
    augment_epochs=args["augment_epoch"]
)


print("Test set size:", len(images))
# for i in set(labels):
#     count = labels.count(i)
#     if count < K + Q:
#         print("cls:", i, "occur:", count) 

#     # print("cls:", i, "occur:", count) 

# # print(ds.classes)
# exit(1)

dl = torch.utils.data.DataLoader(
    dataset=ds,
    batch_size=1,
    collate_fn=episodic_collate,
    shuffle=False
)

class_counter = [0] * len(CLASSES)
class_correct_prediction_counter = [0] * len(CLASSES)

print("starting experiment...")
    
episodic_results = dict()
episode = 0
with torch.no_grad():
    for batch, labels, query_ids, s_x, s_y, support_ids, episode_classes in tqdm(dl):
        # episode setup
        episode += 1
        episodic_results[episode] = dict()
        # clear cache dir
        for cache_dir in cache_dirs:
            for file in cache_dir.glob("*shot*"):
                file.unlink()
        # seperate batch
        num_batches = (len(batch) + batch_size - 1) // batch_size
        support_features = model.encode_image(s_x)
        for selected_metric in metrics:
            # results setup
            correct_predictions = 0
            correct_predictions_per_cls = torch.tensor([0] * M)
            predictions_per_cls = torch.tensor([0] * M)
            # inference
            for i in range(num_batches):
                s, e = i*batch_size, i*batch_size+batch_size
                query_features = get_features(batch[s:e], query_ids[s:e])
                batched_results = model.predict(
                    query_features=query_features,
                    support_features=support_features,
                    s_y=s_y,
                    classnames=[CLASSES[int(i)] for i in episode_classes],
                    prompts=["a {}"],
                    distance_metric=selected_metric,
                )
                for r, label in zip(batched_results, labels[i*batch_size:i*batch_size+batch_size]):
                    if isinstance(r["data"], torch.Tensor):
                        pred_correctness = int(r["data"].detach().cpu().item() == label)
                    else:
                        pred_correctness = int(r["data"] == label)
                    correct_predictions += pred_correctness
                    correct_predictions_per_cls[label] += pred_correctness
                    predictions_per_cls[label] += 1
            # storing results 
            episodic_results[episode][selected_metric] = {
                "accuracy": correct_predictions / (Q * M),
                "accuracy/class": (correct_predictions_per_cls / predictions_per_cls).tolist()
            }
        # correct_predictions = 0
        # correct_predictions_per_cls = torch.tensor([0] * M)
        # predictions_per_cls = torch.tensor([0] * M)
        # for i in range(num_batches): 
        #     s, e = i*batch_size, i*batch_size+batch_size
        #     query_features = get_features(batch[s:e], query_ids[s:e])
        #     results = tip_adapter.predict(
        #         s_x=s_x,
        #         s_y=s_y[:K*M],
        #         query_features=query_features,
        #         classnames=[CLASSES[int(i)] for i in episode_classes],
        #         prompts=["a {}"],
        #     )
        #     for r, label in zip(results, labels[i*batch_size:i*batch_size+batch_size]):
        #         if isinstance(r["data"], torch.Tensor):
        #             pred_correctness = int(r["data"].detach().cpu().item() == label)
        #         else:
        #             pred_correctness = int(r["data"] == label)
        #         correct_predictions += pred_correctness
        #         correct_predictions_per_cls[label] += pred_correctness
        #         predictions_per_cls[label] += 1

        # result = {
        #     "TipAdapter": {
        #         "accuracy": correct_predictions / (Q * M),
        #         "accuracy/class": (correct_predictions_per_cls / predictions_per_cls).tolist()
        #     }
        # }
        # episodic_results[episode].update(result)
        
print_fsl_results(episodic_results, metrics)
torch.save(feature_cache, feature_cache_path)