# Running TipAdapter with fewpy

In order to run the AirShot implementation within fewpy you will need:
    - to import fewpy's FewShotModel class
    - CLIP backbone weights (e.g., ViT-B/32) (optional)
    - Adapter Weights in a '.pth' file (optional)
    - to import fewpy's FSLDataset and PreprocessingMethod (optional)

## Obtaining the models weights

TipAdapter relies on CLIP weights. These can be downloaded directly through fewpy. The implementation is designed to work with standard CLIP architectures such as ViT-B/32.

```python
from fewpy.util.download import download

# Example for downloading ViT-B/32 weights
print(download("ViT-B/32"))
```

The model will download its weights automatically if not weight file is found.

## Importing necessary fewpy tooling

In order to instantiate the model, FewShotModel is necessary. For data handling, especially when using support image augmentation, you should import FSLDataset and PreprocessingMethod.

```python
from fewpy.inference import FewShotModel
from fewpy.util.data import FSLDataset, PreprocessingMethod, fsl_collate
from torch.utils.data import DataLoader
```

## Instatiating AirShot

To instantiate the model, you need a configuration dictionary. This dictionary controls the CLIP backbone, the k-shot settings, and whether to show logit outputs.

An example configuration dictionary:

```python
args = {
    "kshot": 10,             # Number of examples per class in the support set
    "clip_model": "ViT-B/32", # Name of the CLIP model used
    "cache_kv": True,        # Flag for KV caching 
    "augment_epoch": 12,     # Number of epochs for support set augmentation
    "alpha": 1.0,            # Scales clip logits when combining for final prediction
    "beta": 1.0,             # Sharpness for cache_logits calculation
    "show_logits": True      # Flag for outputting logits along with final prediction
}

model = FewShotModel("TipAdapter", config=args)
```

## Preparing your input

TipAdapter expects a query batch (x), a support image tensor (s_x), and a support ground truth tensor (s_y). Additionally, it requires a list of class names and prompt templates.

```python
# The model formats classnames into the {} placeholder in the prompts
results = model.predict(
    x=batch,
    s_x=s_x,
    s_y=s_y,
    classnames=["bottle", "sofa"],
    prompts=["an image of one or more objects of type {}"]
)
```

Notice that it is recommended to use PreprocessingMethod.AUGMENT_SUPPORT_IMAGES within the FSLDataset to properly prepare the support set according to the augment_epoch setting.

## Model output

In fewpy, every model has a standardized output dictionary containing a "task" value and a "data" value. For TipAdapter, the task is "classification" and the data is the predicted class tensor. If show_logits is enabled, the output also contains the final logits.

As previouly mentioned you may or may not use fewpy's dataset class and collate function. A full example of AirShot inference using fewpy is availble in [Fewpy/examples/models/airshot.py](./tipadapter.py).
