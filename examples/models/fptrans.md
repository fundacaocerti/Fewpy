# Running FPTRANS with fewpy

In order to run the FPTRANS implementation within fewpy you will need:
    - to import fewpy's FewShotModel class
    - a checkpoint (weight) file for the model named "fptrans.pth"
    - to import fewpy's FSLDataset (optional)

## Obtaining the models weights

The weights were made available through the paper [Official Github Repository](https://github.com/Jarvis73/FPTrans?tab=readme-ov-file). 

The easiest way to make the weights available to fewpy is to download them, rename them and move them to a "weights" folder in your projects' root directory.

## Importing necessary fewpy tooling

In order to instatiate the model FewShotModel is necessary, so import it as in the following snippet.

```python
from fewpy.util.inference.FewShotModel import FewShotModel
```

You may or may not use fewpy's FSLDataset class. Notice that it is a recommendation to use torch's DataLoader with FSLDataset but any compatible implementation should work. Fewpy also contains a collate function that should be used with torch.util.data.DataLoader.

```python
from fewpy.util.data.dataset import FSLDataset, fsl_collate
from torch.utils.data import DataLoader
```

## Instatiating FPTRANS

Now that you imported FewShotModel all you will need to instatiate the model is a configuration dictionary, name it whatever works best for as it will be passed to the FewShotModel contructor.

An example of configuration dictionary would be:

```python
args = {
    "kshot": 1,                 # number of support examples per class in the few-shot setup (e.g., 1-shot learning)
    "dataset": "pascal",        # the dataset being used for training or evaluation (e.g., Pascal VOC)
    "backbone": "VIT-B",        # the neural network backbone used for feature extraction (Vision Transformer Base)
    "split": 9,                 # the specific dataset split used for cross-validation/testing
    "checkpoint": None,         # dictionary or path containing pre-trained model weights to load
    "Probs_return": True,       # whether the model should return soft probability maps instead of hard logits/masks
    "drop_dim": 1,              # the specific tensor dimension along which dropout is applied
    "drop_rate": 0.3,           # the dropout probability rate used to prevent overfitting
    "block_size": 16,           # the patch/block resolution for the Vision Transformer (e.g., 16x16 pixel patches)
    "height": 700,              # the input image height (often paired with img_size)
    "pretrained": "",           # path or identifier for the backbone's pre-trained weights (e.g., ImageNet weights)
    "SAHI": False,              # flag to enable Slicing Aided Hyper Inference (used for detecting/segmenting small objects)
    "bg_num": 5,                # number of background prototypes or background tokens used to model background context
    "bsz": 32,                  # the batch size used during training or inference runs
    "img_size": 700,            # the overall spatial dimensions to resize the input image to (e.g., 700x700)
    "training": False,          # boolean flag indicating whether the network is in training mode or evaluation mode
    "vit_depth": 10,            # the specific depth (number of transformer layers) extracted from the ViT backbone
    "vit_stride": 23,           # the stride used for ViT patch extraction (affects the sequence length and overlap)
    "coco2pascal": False,       # flag for cross-dataset evaluation (e.g., training on COCO, evaluating on Pascal VOC)
    "num_prompt": 72,           # the number of learnable visual prompt tokens injected into the transformer
    "pt_std": 0.02,             # standard deviation for the normal distribution used to initialize the prompt tokens
}
```

Notice that some configurations alter the models layers. Therefore your configuration should be compatible with the downloaded weights.

## Preparing your input

FPTRANS expects a query_batch (x: torch.Tensor), a support image tensor (s_x: torch.Tensor) and a support ground truth tensor (s_y: torch.Tensor).

```python
model.predict(
        x=batch,
        s_x=s_x,
        s_y=s_y,
)
```

## Model output

In fewpy every model has standardized output dictionary. Every model outputs a dictionary containing a "task" value and a "data" value. Those are the specific task the model is designed for and the prediction data. In addition FPTRANS's output also contains the raw predictions (no postprocessing) as the value for "raw_data".

As previouly mentioned you may or may not use fewpy's dataset class and collate function. A full example of FPTRANS inference using fewpy is availble in [Fewpy/examples/models/fptrans.py]().
