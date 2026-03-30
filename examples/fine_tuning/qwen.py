from fewpy.inference import FewShotModel
from fewpy.util.data import FSLDataset
from torch.utils.data import DataLoader

from pathlib import Path

from torch.optim import AdamW
import torch

import xml.etree.ElementTree as ET

from PIL import Image
import numpy as np


def qwen_collate(batch):
    return batch

# prepare support and query data 
K = 1
n = 8
CLASSES = ("bottle", "sofa")
epochs = 5
batch_size = 4
learning_rate = 1e-5

dataset = Path("./testdata2").expanduser()
annotations = dataset / "selected_annot" 
images = dataset / "selected_img"
print(dataset)

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
    "lora": True,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.05,
    "lora_alpha": 32,
    "lora_rank": 16,
    "lora_bias": "none",
    "gradient_checkpointing": True,
    "quantization": True,
    "compute_dtype": "bfloat16",
    "max_pixels": 512 * 512,
    "min_pixels": 224 * 224,
}

model = FewShotModel(
    model="Qwen",
    config=args
)
    
params = [p for p in model.parameters() if p.requires_grad]

optimizer = AdamW(params, lr=learning_rate)

for epoch in range(epochs):
    """
            self.model.forward:
            Args:
                x: a list of Image objects, the batched query.
                s_x: a list of Image objects, the support images.
                s_y: a list of dictionaries containing the gorund truth for each of the support images.
            Returns:
                list[list[dict]]:
                    Each list[dict] is a list of detections from a single image
                    Each dict is the output of one detection from a single image.
                    The dict contains the following keys:
                    key "task" that specifies the task the model is trained on (always "detection")
    """
    model.train()
    total_loss = 0
    for batch in dl:
        optimizer.zero_grad()
        loss = model.predict(
            x=batch,
            s_x=support_images,
            s_y=support_ground_truth,
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(dl)
    print(f"Average Loss for Epoch {epoch + 1}: {avg_loss / epochs:.4f}")

print("\nFine-tuning complete!")
torch.save(model.state_dict(), "qwen.pth")
