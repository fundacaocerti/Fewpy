from fewpy.inference import FewShotModel
from fewpy.util.data import FSLDataset, PreprocessingMethod, fsl_collate

from pathlib import Path
from PIL import Image

import xml.etree.ElementTree as ET
import numpy as np
import torch

from torch.nn.functional import cross_entropy
from tqdm import tqdm


# prepare support and query data 
K = 10
n = 4
CLASSES = ("bottle", "sofa")
epochs = 5
batch_size = 2
IMG_SIZE = 224
cls2int = {
    "bottle": 0,
    "sofa": 1,
}
learning_rate = 1e-3
eps = 1e-4
epochs = 5

dataset = Path("./testdata2").expanduser()
annotations = dataset / "selected_annot" 
images = dataset / "selected_img"

counter = [0] * 2

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
    cls = None

    skip = False
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASSES or (cls is not None and name != cls):
            skip = True
            break

        first = False
        cls = cls2int[name]

    if counter[cls] < K:
        support_ground_truth.append(torch.tensor(cls).squeeze())
        support_images.append(Image.open(image).convert("RGB"))

        counter[cls] += 1
    elif len(query_images) < n:
        query_images += [Image.open(image).convert("RGB")]
        query_targets += [torch.tensor(cls).squeeze()]
    
    if all(ki >= K for ki in counter) and \
        len(query_images) >= n:
        break

"""
kshot: int, Number of examples per class in the support set
clip_model: str, Name of the CLIP model used
cache_kv: bool, Flag for KV caching 
cache_dir: str, Directory to save the cached KV
augment_epoch: int, Number of epochs in the support set images augmentation process,
used to calculate the expected number of support images and transform the label tensor accordingly
load_adapter: bool, flag for loading previouly tuned adapter from a .pth file
training: bool, training mode flag
beta: float, sharpness for cache_logits calculation
alpha: float, scales clip logits when combining the logits to gen the final prediction
show_logits: bool, flag for outputing logits along with final prediction
"""
args = {
    "kshot": 10,
    "clip_model": "ViT-B/32",
    "augment_epoch": 12,
    "show_logits": True,
    "training": True,
    "n_novel_classes": len(CLASSES),
    "cache_kv": True,
    "cache_dir": "./cache"
}

model = FewShotModel("TipAdapter", config=args)

ds = FSLDataset(
    query_images,
    support_images,
    support_ground_truth,
    labels=query_targets,
    img_size=IMG_SIZE,
    center_crop=IMG_SIZE,
    pixel_norm=((0.48145466, 0.48145466, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),       # (mean, std))
    support_set_preprocessing_method=PreprocessingMethod.AUGMENT_SUPPORT_IMAGES,
    epochs=args["augment_epoch"],
)

dl = torch.utils.data.DataLoader(
    dataset=ds,
    batch_size=batch_size,
    collate_fn=fsl_collate,
    shuffle=False
)

# creating optimizer
# no layer freezing, TipAdapter only returns the adapters parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=eps)
T_max = epochs * len(dl)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)

"""
self.model.predict:
Args:
    batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
        Each item in the list contains the inputs for one image.
        For now, each item in the list is a dict that contains:
        * x: Tensor, batch of images in (B, C, H, W) format.
        * s_x: Tensor, batch of support images in (B, C, H, W) format.
        * s_y: Tensor, batch of ground truth classes in (B) format.
        * prompts: list[str], prompt templates for the classnames to be formatted into.
        * classnames: list[str], name of all novel classes. 
Returns:
        list[dict]:
            Each dict is corresponds to the output of a single input image.
            The dict contains a string "classification" under the key "task" to specify the task type and
            a "data" class prediction, Tensor of format (1)
"""
results = []
for epoch in range(epochs):
    loss_list = []
    for batch, labels, s_x, s_y in tqdm(dl):

        labels = torch.stack(labels).to(model.device)

        logits = model.predict(
            x=batch,
            s_x=s_x,
            s_y=s_y,
            classnames=CLASSES,
            prompts=["an image of one or more objects of type {}"]
        )

        loss = cross_entropy(logits, labels)
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | Loss: {sum(loss_list) / len(loss_list):.8f}")

adapter_state_dict = {k: v for k, v in model.state_dict().items() if k.startswith("adapter")}
torch.save(adapter_state_dict, "tip_adapter.pth")
