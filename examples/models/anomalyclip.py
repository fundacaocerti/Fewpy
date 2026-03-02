from fewpy.util.inference.FewShotModel import FewShotModel
from fewpy.util.inference.preprocessor import Preprocessor
from fewpy.util.data.dataset import FSLDataset, fsl_collate
from torch.utils.data import DataLoader


import torch
import PIL
import open_clip as clip

from torchvision.transforms import ToPILImage, ToTensor, Resize
from pathlib import Path


# The following example uses the KolektorSDD dataset

MAX_SUBSETS = 2     # each subset contains 8 examples, therefore k = 16 for MAX_SUBSETS =  2
IMG_SIZE = 448

totensor = ToTensor()
converter = ToPILImage()
resize = Resize((IMG_SIZE, IMG_SIZE))

# load images
root = Path("./KolektorSDD").expanduser()
query_path = root / "kos50"

query_images = []
support_images = []
support_ground_truth = []

for i, subset in enumerate(root.iterdir()):
    if subset.name == query_path.name:
        continue

    if i >= MAX_SUBSETS:
        break
    support_images += [PIL.Image.open(img_path).convert("RGB") for img_path in subset.glob("*.jpg")]
    support_ground_truth += [resize(torch.Tensor(totensor(PIL.Image.open(img_path)))) for img_path in subset.glob("*.bmp")]

support_ground_truth = torch.stack(support_ground_truth).squeeze(1)
query = list(query_path.glob("*.jpg"))
query_images = [PIL.Image.open(img_path).convert("RGB") for img_path in query]
W, H = query_images[0].size

print("len query:", len(query_images))

# initialize dataset
ds = FSLDataset(
    x=query_images,
    s_x=support_images,
    s_y=support_ground_truth,
    img_size=(IMG_SIZE, IMG_SIZE),
    pixel_norm=((0.485, 0.456, 0.0406), (0.229, 0.224, 0.225))       # (mean, std)
)

dl = DataLoader(
    ds, 
    batch_size=8,
    shuffle=False,
    collate_fn=fsl_collate      # Fewpy collate function
)

# expects H == W
"""
feature_list: list[int], A list of feature (for scaling) indices, default=[6, 12, 18, 24]
image_size: int, default=700 (700x700)
depth: int, Depth of ViT, default=9
n_ctx: int, Number of context tokens, default=12
t_n_ctx: int, default=4

kshot: int, default=5
alpha: float, Visual weight for normal prototype, default=1.0
beta: float, Visual weight for anomalous prototype, default=1.0
scale_weights: list[float], Weights for the scales, default=[0.5, 1.0, 2.0, 3.0]

obj_threshold: float, Pixel intensity threshold for object detection, default=0.1
gamma: float, Anomaly map intesity, default=2.0
contrast: float, Controls the sharpness between anomaly patches. 
Larger values tend to give less self.config.contrast between anomalies, default = 0.07
softmax_temp: float, Softmax temperature for user text prompt attention. 
Lower values pay more attention to visual anomalies described by the user prompt, default = 0.07
seed: int, default=111
sigma: int, default=4
cls_id: int | None, default=None

checkpoint: Dict | None, Checkpoint dictionary with model state, default=None
"""
args = {
    "image_size": IMG_SIZE,
    "n_ctx": 12,
    "t_n_ctx": 4,
}

model = FewShotModel(
    model="anomalyCLIP",
    config=args,
    preprocessors=[
        Preprocessor(
            function=clip.tokenize,
            input_keys=["user_prompts"],
            output_key="user_tknized_prompts",
            is_tokenizer=True
        )
    ]
)

"""
AnomalyCLIP.predict:
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
result = []
for batch, s_x, s_y in dl:
    # print(f"shapes: batch = {batch.shape}, s_x = {s_x.shape}, s_y = {s_y.shape}")
    result += model.predict(
        x=batch,
        s_x=s_x,
        s_y=s_y,
        user_prompts=["crack", "fissure"]
    )

# saving the masks as images
for i, yi in enumerate(result):
    print(f"example: {query[i]}")
    mask = yi["raw_data"]
    mask = converter(mask)
    mask = mask.resize((W, H), resample=PIL.Image.NEAREST)
    mask.save(f"output{i}.png")
    postproc_mask = yi["data"]
    postproc_mask = converter(postproc_mask)
    postproc_mask = postproc_mask.resize((W, H), resample=PIL.Image.NEAREST)
    postproc_mask.save(f"output{i}_alt.png")
