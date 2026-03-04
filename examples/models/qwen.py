from fewpy.util.inference.FewShotModel import FewShotModel
from fewpy.util.data.dataset import FSLDataset, qwen_collate
from torch.utils.data import DataLoader

from pathlib import Path
import xml.etree.ElementTree as ET

import torch

from PIL import Image
import numpy as np


# prepare support and query data 
K = 5
n = 1
CLASSES = ("bottle", "sofa")

dataset = Path("./testdata2").expanduser()
annotations = dataset / "selected_annot" 
images = dataset / "selected_img"

counter = {cls: 0 for cls in CLASSES}

annotations = list(annotations.glob("*.xml"))

indices = np.random.permutation(len(annotations))
query_targets = []
query_images = []
support_images = []
support_ground_truth = []

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
        bbox = torch.as_tensor(bbox, dtype=torch.float32).unsqueeze(0)

        if first: 
            first = False
            s_yi["cls"] = name
            s_yi["bboxes"] = bbox
            cls = name
            continue

        torch.stack([s_yi["bboxes"], bbox], dim=0)

    if counter[cls] < K:
        support_ground_truth.append(s_yi)
        support_images.append(Image.open(image).convert("RGB"))

        counter[cls] += 1
    elif len(query_images) < n:
        query_images += [Image.open(image).convert("RGB")]
        s_yi["height"] = height
        s_yi["width"] = width
        query_targets = [s_yi]
    
    if all(ki >= K for ki in counter.values()) and \
        len(query_images) >= n:
        break

ds = FSLDataset(query_images, transform_datapoints=False)
dl = DataLoader(
    ds,
    batch_size=1,
    shuffle=True,
    collate_fn=qwen_collate,
)

args = {
    "classnames": ["bottle", "sofa"],
}

model = FewShotModel(
    model="Qwen",
    config=args
)

results = []
for batch in dl:

    results += model.predict(
        x=batch,
        s_x=support_images,
        s_y=support_ground_truth,
    )

print(results)

for i, image_results in enumerate(results):
    print(f"Image {i+1} detections:")
    for detection in image_results:
        print(f"confidence: {detection["conf"]}\nbbox: {detection["data"]}")
