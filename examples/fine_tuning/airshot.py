from fewpy.inference import FewShotModel
from fewpy.util.data import FSLDataset

from torch.utils.data import DataLoader
from torch.optim import AdamW

from detectron2.utils.events import EventStorage
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.detection_utils as utils

from PIL.Image import Image
from pathlib import Path

import copy

import xml.etree.ElementTree as ET
import torch.nn as nn
import numpy as np
import torch


def load_xml_data(CLASSES: str, n=30, k=10):
    
    dataset = Path("./testdata2").expanduser()
    annotations = dataset / "selected_annot" 
    images = dataset / "selected_img"

    name2id = {CLASSES[i]: i for i in range(len(CLASSES))}
    
    counter = {cls: 0 for cls in CLASSES}
    counter_query = 0
    
    annotations = list(annotations.glob("*.xml"))
    # print("retrieving", len(annotations), "annotations")
    # print()
    
    indices = np.random.permutation(len(annotations))
    x = [] # PIL.Image.Image
    s_x = [] # PIL.Image.Image
    s_y = [] # dict: {"class_ids: [class_id], "bboxes": [bbox]}
    y = [] # dict: {"class_ids: [class_id], "bboxes": [bbox]}
    
    for i in indices:
        annotation = annotations[i]
    
        tree = ET.parse(annotation)
        root = tree.getroot()
        image = images / root.find("filename").text
        s_yi = dict()
        first = True
        cls = None
    
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
    
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in CLASSES:
                continue
    
            if cls is not None and name != cls:
                continue
    
            bndbox = obj.find("bndbox")
            bbox = [
                int(bndbox.find("xmin").text) / width,
                int(bndbox.find("ymin").text) / height,
                int(bndbox.find("xmax").text) / width,
                int(bndbox.find("ymax").text) / height
            ]
            # bbox = torch.as_tensor(bbox, dtype=torch.float32).unsqueeze(0)
    
            if first: 
                first = False
                s_yi["class_ids"] = [name2id[name]]
                s_yi["bboxes"] = [bbox]
                cls = name
                break
    
            s_yi["bboxes"].append(bbox)
            s_yi["class_ids"].append(name2id[name])
    
        if counter[cls] < k:
            s_y.append(s_yi)
            s_x.append(Image.fromarray(utils.read_image(image, format='BGR')))
    
            counter[cls] += 1
        elif counter_query < n:
            x += [Image.fromarray(utils.read_image(image, format='BGR'))]
            s_yi["height"] = height
            s_yi["width"] = width
            y += [s_yi]

            counter_query += 1

    return x, y, s_x, s_y

def fsl_collate(batch):

    batch, labels, s_x, s_y = zip(*batch)

    return batch, labels, s_x, s_y

classnames = ["1", "2"]
N, K = 20, 10 # number of examples, number of support examples per class
epochs = 10
batch_size = 4
learning_rate = 2e-6
SHOW_WEIGHT_CHANGE = False

mapper = {
    "1": 0,
    "2": 1,
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

# freeze backbone
for param in model.model.backbone.parameters():
    param.requires_grad = False
    
# for model in [model.model.roi_heads, model.model.proposal_generator, model.model.fuser, model.model.apn]:
#     for _, param in model.named_parameters():
#         param.requires_grad = False
    
params = [p for p in model.parameters() if p.requires_grad]

optimizer = AdamW(params, lr=learning_rate)

CLASSES = ("bottle", "sofa")
x, labels, s_x, s_y = load_xml_data(CLASSES, n=N, k=K)
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
            batch = [xi.to(device) for xi in batch]
            
            optimizer.zero_grad()
            
            loss_dict = model.predict(batch, y=labels, s_x=s_x, s_y=s_y)
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