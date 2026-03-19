from fewpy.inference import Preprocessor, FewShotModel
from fewpy.util.data import FSLDataset, PreprocessingMethod
from fewpy.util.loss import BinaryDiceLoss, FocalLoss
from torch.utils.data import DataLoader

import torch
import PIL
import open_clip as clip

from torchvision.transforms import ToPILImage, ToTensor
from torch.optim import AdamW
from pathlib import Path

import torch.nn.functional as F


# The following example uses the KolektorSDD dataset
SUBSET_SIZE = 8
MAX_SUBSETS = 2     # each subset contains 8 examples, therefore k = 16 for MAX_SUBSETS =  2
DATASET_SIZE = 16
IMG_SIZE = 448
epochs = 5
batch_size = 4
learning_rate = 1e-5
loss_focal = FocalLoss()
loss_dice = BinaryDiceLoss()
lam = 4

def collate_fn(batch):
    batch, labels, s_x, s_y = zip(*batch)
    batch = torch.stack(batch)
    labels = torch.stack(labels)

    s_x = s_x[0]
    s_y = s_y[0]

    return batch, labels, s_x, s_y


totensor = ToTensor()
converter = ToPILImage()

# load images
root = Path("./KolektorSDD").expanduser()
query_path = root / "kos50"

query_images = []
labels = []
support_images = []
support_ground_truth = []

for i, subset in enumerate(root.iterdir()):

    if i < MAX_SUBSETS:
        support_images += [PIL.Image.open(img_path).convert("RGB") for img_path in subset.glob("*.jpg")]
        support_ground_truth += [totensor(PIL.Image.open(img_path)) for img_path in subset.glob("*.bmp")]
    elif i < (MAX_SUBSETS + DATASET_SIZE / SUBSET_SIZE):
        query_images += [PIL.Image.open(img_path).convert("RGB") for img_path in subset.glob("*.jpg")]
        labels += [totensor(PIL.Image.open(img_path)) for img_path in subset.glob("*.bmp")]
    else:
        break

# support_ground_truth = torch.stack(support_ground_truth).squeeze(1)
W, H = query_images[0].size

# initialize dataset
ds = FSLDataset(
    x=query_images,
    s_x=support_images,
    s_y=support_ground_truth,
    labels=labels,
    img_size=(IMG_SIZE, IMG_SIZE),
    pixel_norm=((0.485, 0.456, 0.0406), (0.229, 0.224, 0.225)),       # (mean, std)
    support_set_preprocessing_method=PreprocessingMethod.RESIZE_SUPPORT_GT,
)

dl = DataLoader(
    ds, 
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,      # Fewpy collate function
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

# freezing the backbone
# for param in model.model.backbone.parameters():
#     param.requires_grad = False
    
params = [p for p in model.parameters() if p.requires_grad]

optimizer = AdamW(params, lr=learning_rate)

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
total_loss = 0
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    model.train()

    epoch_loss = 0
    for batch, labels, s_x, s_y in dl:
        
        batch = batch.to(model.device)
        optimizer.zero_grad()

        sim_map_list, text_probs = model.predict(
            x=batch,
            s_x=s_x, 
            s_y=s_y,
            user_prompts=["crack", "fissure"],
        )

        image_level_label = (labels.view(labels.size(0), -1).sum(dim=1) > 0).long()
        image_loss = F.cross_entropy(text_probs.view(-1, 2), image_level_label)
        # image_loss = F.cross_entropy(text_probs.squeeze(), labels.long().to(model.device))
        loss = 0
        for i in range(len(sim_map_list)):
            loss += loss_focal(sim_map_list[i], labels)
            loss += loss_dice(sim_map_list[i][:, 1, :, :], labels)
            loss += loss_dice(sim_map_list[i][:, 0, :, :], 1-labels)

        loss = lam * loss
        (loss + image_loss).backward()
        total_loss += loss.item()

        optimizer.step()
        print(f"  Loss: {loss:.4f}")

    avg_loss = total_loss / len(dl)
    print(f"Average Loss for Epoch {epoch + 1}: {avg_loss / epochs:.4f}")

print("\nFine-tuning complete!")
torch.save(model.state_dict(), "anomaly_clip.pth")
