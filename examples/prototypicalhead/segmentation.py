from fewpy.inference import FewShotModel
from fewpy.util.data import FSLDataset, PreprocessingMethod, fsl_collate

from pathlib import Path
from PIL import Image

import xml.etree.ElementTree as ET
import numpy as np
import torch
from tqdm import tqdm


from pathlib import Path
import xml.etree.ElementTree as ET

import PIL
import numpy as np

import torch

import torchvision.transforms.functional as F


def fptrans_collate(batch):

    batch, s_x, s_y = zip(*batch)
    batch = torch.stack(batch)
    s_x = torch.stack(s_x)
    s_y = torch.stack(s_y)

    return batch, s_x, s_y

# prepare support and query data 
K = 1
N = 4
CLASSES = ["bottle",]
IMG_SIZE = 700

dataset = Path("./ds").expanduser()
annotations = dataset / "Annotations" 
images = dataset / "JPEGImages"
segmentation = dataset / "SegmentationClass"

annotations = list(annotations.glob("*.xml"))
# print("retrieving", len(annotations), "annotations")
# print()

indices = np.random.permutation(len(annotations))
query_targets = []
query_images = []
support_images = []
support_ground_truth = []
sizes = []

for i in indices:
    annotation = annotations[i]
    # print(f"looking into {annotation}")

    tree = ET.parse(annotation)
    root = tree.getroot()
    image = images / root.find("filename").text
    s_yi = segmentation / f"{root.find("filename").text[:-4]}.png"
    if not s_yi.exists():
        # print(f"Segmentation {s_yi} does not exist!\n\n")
        continue
    first = True
    cls = None

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASSES:
            first = True
            break

        first = False
        cls = name

    if first == True:
        # print("no object from class list found!")
        continue

    if len(support_images) < K:
        mask = PIL.Image.open(s_yi).convert("L")
        mask_tensor = torch.from_numpy(np.array(mask))
        mask_tensor = (mask_tensor > 0).float().unsqueeze(0)
        support_images.append(PIL.Image.open(image).convert("RGB"))
        support_ground_truth.append(mask_tensor)
    
    elif len(query_images) < N:
        query_images += [PIL.Image.open(image).convert("RGB")]
        mask = PIL.Image.open(s_yi).convert("L")
        mask_tensor = torch.from_numpy(np.array(mask))
        mask_tensor = (mask_tensor > 0).float().unsqueeze(0)
        query_targets.append(mask_tensor)
        sizes.append(query_images[-1].size)

    if len(support_images) >= K and len(query_images) >= N:
        break

ds = FSLDataset(
    x=query_images,
    s_x=support_images,
    s_y=support_ground_truth,
    img_size=(IMG_SIZE, IMG_SIZE),
    pixel_norm=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),       # (mean, std)
    support_set_preprocessing_method=PreprocessingMethod.RESIZE_SUPPORT_GT,
)

dl = torch.utils.data.DataLoader(
    ds, 
    batch_size=2,
    shuffle=False,
    collate_fn=fptrans_collate      # fptrans collate function
)

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
    "kshot": K,
    "backbone": "ViT-B/32",
    "task": "classification",
    "cache_prototypes": True,
    "cache_dir": "./cache",
}

model = FewShotModel("PrototypicalHead", config=args)

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
for batch, s_x, s_y in dl:
    # print(f"shapes: batch = {batch.shape}, s_x = {s_x.shape}, s_y = {s_y.shape}")
    results += model.predict(
        x=batch,
        s_x=s_x,
        s_y=s_y,
    )

# saving the masks as images
diff = 0
total_size = 0
for i, yi in enumerate(results):
    mask = yi["data"]
    mask = F.resize(
        sizes[i],
        F.InterpolationMode.NEAREST,
    )
    diff += torch.abs(query_targets[i] - mask).sum()
    total_size += query_targets.sum()

print(F"{diff} / {total_size} wrong pixels, {diff / total_size: 0.4f}%")
