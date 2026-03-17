from fewpy.inference import FewShotModel
from fewpy.util.data import FSLDataset

from torch.utils.data import DataLoader
from torch.optim import AdamW

from detectron2.utils.events import EventStorage
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.detection_utils as utils

from PIL import Image
from pathlib import Path

import copy
import json

import xml.etree.ElementTree as ET
import torch.nn as nn
import numpy as np
import torch


def load_data(path, n=30, k=10):
    
    with open(path, "r") as f:
        data = json.loads(f.read())

    annotations = data["annotations"]
    annotations = np.random.permutation(annotations)
    n_selected_img = [0, 0]
    n_support_img = [0, 0]
    
    x = []
    y = []
    s_x = []
    s_y = []

    for annot in annotations:
        
        class_id = annot["category_id"] - 1

        if class_id > 1:
            continue
        
        if all([ni >= n for ni in n_selected_img]) \
            and all([ki >= k for ki in n_support_img]):
            break

        if n_selected_img[class_id] >= n:
            if n_support_img[class_id] >= k:
                continue

            n_support_img[class_id] += 1
            n_selected_img[class_id] += 1
            img_path = Path("images") / f"{annot["image_id"]}.jpg"
            s_x.append(Image.fromarray(utils.read_image(img_path, format="BGR")))
            xmin, ymin, w, h = annot["bbox"]
            bbox = xmin, ymin, xmin + w, ymin + h
            s_y.append({
                "cls": [class_id],
                "bboxes": [bbox],
            })

        n_selected_img[class_id] += 1
        img_path = Path("images") / f"{annot["image_id"]}.jpg"
        x.append(Image.fromarray(utils.read_image(img_path, format="BGR")))
        xmin, ymin, w, h = annot["bbox"]
        bbox = xmin, ymin, xmin + w, ymin + h
        y.append({
            "cls": [class_id],
            "bboxes": [bbox],
        })

    return x, y, s_x, s_y

def fsl_collate(batch):

    batch, labels, s_x, s_y = zip(*batch)

    return batch, labels, s_x, s_y

classnames = ["1", "2"]
N, K = 20, 5 # number of examples, number of support examples per class
epochs = 10
batch_size = 4
learning_rate = 2e-6
SHOW_WEIGHT_CHANGE = False

mapper = {
    1: 0,
    2: 1,
}

args = {
    "datasetname": 'SeaDronesSeeFineTuning',
    "classnames": classnames,
    "confidence_threshold": 0.5,
    "mapping_to_contiguous_ids": mapper
}

# config = AnomalyCLIPConfig(**args)
model = FewShotModel(
    model="AirShot",
    config=args
)

# print(model)

# freeze backbone
for param in model.model.model.backbone.parameters():
    param.requires_grad = False
    
# for model in [model.model.roi_heads, model.model.proposal_generator, model.model.fuser, model.model.apn]:
#     for _, param in model.named_parameters():
#         param.requires_grad = False
    
params = [p for p in model.parameters() if p.requires_grad]

optimizer = AdamW(params, lr=learning_rate)

x, labels, s_x, s_y = load_data("./data/annotations.json", n=N, k=K)
ds = FSLDataset(
    x=x,
    labels=labels,
    s_x=s_x,
    s_y=s_y,
    img_size=600,
    max_size=1000,
    pixel_norm=((103.530, 116.280, 123.675), (1.0, 1.0, 1.0)),       # (mean, std)
    support_set_preprocessing_method="detection_crop",
)

dl = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=fsl_collate
)

previous_weights = None

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    model.train()

    if SHOW_WEIGHT_CHANGE:
        old_state = copy.deepcopy(model.state_dict())
    
    total_loss = 0

    epoch_loss_dict = dict()
    
    for batch_idx, batch in enumerate(dl):
        
        with EventStorage() as storage:
            
            batch, labels, s_x, s_y = batch
            batch = [xi.to(model.device) for xi in batch]
            
            optimizer.zero_grad()
            
            loss_dict = model.predict(x=batch, y=labels, s_x=s_x, s_y=s_y)
            loss = sum(loss for loss in loss_dict.values())
            
            loss.backward()
            total_loss += loss
            
            optimizer.step()

            for k in loss_dict.keys():
                if k not in epoch_loss_dict:
                    epoch_loss_dict[k] = loss_dict[k]
                else:
                    epoch_loss_dict[k] += loss_dict[k]
            
            print(f"  Batch {batch_idx + 1} | Loss: {loss:.4f}")

    for k in epoch_loss_dict.keys():
        print(f"{k}: {epoch_loss_dict[k] / len(dl)}") 

    if SHOW_WEIGHT_CHANGE:
        new_state = model.state_dict()
        for layer_name in old_state:
            diff = torch.abs(old_state[layer_name] - new_state[layer_name]).sum()
            if diff < 1e-4: continue
            print(f"{layer_name} changed by: {diff.item()}")
        
    avg_loss = total_loss / len(dl)
    print(f"Average Loss for Epoch {epoch + 1}: {avg_loss:.4f}")

print("\nFine-tuning complete!")

checkpointer = DetectionCheckpointer(model, save_dir="output/")
checkpointer.save(f"Airshot_{N}_{K}_{epochs}_{batch_size}_{learning_rate:.0e}")
