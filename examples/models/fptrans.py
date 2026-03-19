from fewpy.inference import FewShotModel
from fewpy.util.data import FSLDataset, PreprocessingMethod

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
        query_targets += [PIL.Image.open(s_yi).convert("L")]
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
kshot: int, Number of support examples per class in the few-shot setup (e.g., 1-shot learning)
dataset: str, The dataset being used for training or evaluation (e.g., Pascal VOC)
backbone: str, The neural network backbone used for feature extraction (Vision Transformer Base)
split: int, The specific dataset split used for cross-validation/testing
checkpoint: dict | str | None, Dictionary or path containing pre-trained model weights to load
Probs_return: bool, Whether the model should return soft probability maps instead of hard logits/masks
drop_dim: int, The specific tensor dimension along which dropout is applied
drop_rate: float, The dropout probability rate used to prevent overfitting
block_size: int, The patch/block resolution for the Vision Transformer (e.g., 16x16 pixel patches)
height: int, The input image height (often paired with img_size)
pretrained: str, Path or identifier for the backbone's pre-trained weights (e.g., ImageNet weights)
SAHI: bool, Flag to enable Slicing Aided Hyper Inference (used for detecting/segmenting small objects)
bg_num: int, Number of background prototypes or background tokens used to model background context
bsz: int, The batch size used in SAHI inference
img_size: int, The overall spatial dimensions to resize the input image to (e.g., 700x700)
training: bool, Boolean flag indicating whether the network is in training mode or evaluation mode
vit_depth: int, The specific depth (number of transformer layers) extracted from the ViT backbone
vit_stride: int, The stride used for ViT patch extraction (affects the sequence length and overlap)
coco2pascal: bool, Flag for cross-dataset evaluation (e.g., training on COCO, evaluating on Pascal VOC)
num_prompt: int, The number of learnable visual prompt tokens injected into the transformer
pt_std: float, Standard deviation for the normal distribution used to initialize the prompt tokens
"""

args = {
    "kshot": 1,
    "dataset": "pascal",
    "backbone": "VIT-B",
    "checkpoint": None,
    "Probs_return": False,
    "drop_dim": 1,
    "drop_rate": 0.3,
    "block_size": 16,
    "height": IMG_SIZE,
    "pretrained": "",
    "SAHI": False,
    "bg_num": 5,
    "bsz": 32,
    "img_size": IMG_SIZE,
    "training": False,
    "vit_depth": 10,
    "vit_stride": 23,
    "num_prompt": 72,
}

model = FewShotModel(
    model="FPTRANS",
    config=args
)

"""
FPTRANS.predict:
Args:
    batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
        Each item in the list contains the inputs for one image.
        For now, each item in the list is a dict that contains:
        * x: Tensor, batch of images in (B, C, H, W) format.
        * s_x: Tensor, batch of support images in (B, K, C, H, W) format.
        * s_y: Tensor, batch of ground truth images in (B, K, H, W) format.
Returns:
        list[dict]:
            Each dict is corresponds to the output of a single input image.
            The dict contains a string "segmentation" under the key "task" to specify the task type,
            a "data" mask, Tensor of format (H, W) and a "postproc_data" mask, Tensor of format (H, W)
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
for i, yi in enumerate(results):
    mask = yi["data"]
    mask = F.to_pil_image(mask.to(torch.uint8) * 255)
    mask = mask.resize(sizes[i], resample=PIL.Image.NEAREST)
    mask.save(f"output{i}.png")
    query_targets[i].save(f"expected{i}.png")
