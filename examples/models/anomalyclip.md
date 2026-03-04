# Running AnomalyCLIP with fewpy

In order to run the AnomalyCLIP implementation within fewpy you will need:
    - to import fewpy's FewShotModel class
    - a checkpoint (weight) file for the model named "anomalyClip.pth"
    - ViT-L-14-336px weights
    - to import fewpy's FSLDataset (optional)

## Obtaining the models weights

The weights were made available through the paper [Official Github Repository](). While the ViT weights are available through [OpenAI](https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt).

The easiest way to make the weights available to fewpy is to download them, rename them and move them to a "weights" folder in your projects' root directory.

ViT weights can be downloaded through fewpy:

```python
import fewpy.util.download as d

print(d.download("ViT-L/14@336px"))
```

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

Additionaly, AnomalyCLIP also requires a tokenizer to be added as a preprocessor. Use open_clip.tokenize as the model was trained with it. Add the Preprocessor object when initializing the model.

```python
from fewpy.util.inference.preprocessor import Preprocessor
import open_clip as clip
```

## Instatiating AnomalyCLIP

Now that you imported FewShotModel all you will need to instatiate the model is a configuration dictionary, name it whatever works best for as it will be passed to the FewShotModel contructor.

An example of configuration dictionary would be:

```python
args = {
    "image_size": 488,     # the size of image to be feed to the model (H = W)
    "n_ctx": 12,                # number of context tokens
    "t_n_ctx": 4,
    "feature_list": [6, 12, 18, 24],    # used to scale indices
    "kshot": 5,                 # number of support examples of each class
    "depth": 9,                 # depth of the ViT backbone
    "beta": 1,                  # scaling factor for anomalous prototype
    "alpha": 1,                 # scaling factor for normal prototype
    "obj_threshold": 0.1,       # pixel intensity threshold for object detection
    "scale_weights": [0.5, 1.0, 2.0, 3.0],  # scaling weights for the anomaly maps
    "gamma": 2,                 # anomaly map intensity
    "constrast": 0.07,          # controls the sharpness between anomaly patches
    "softmax_temp": 0.07,       # softmax temperature for user text prompt attention
    "sigma": 4,                 # anomaly map filtering scaling factor
}
```

Notice that some configurations alter the models layers. Therefore your configuration should be compatible with the downloaded weights.

## Preparing your input

AnomalyCLIP expects a query_batch (x: torch.Tensor), a support image tensor (s_x: torch.Tensor), a support ground truth tensor (s_y: torch.Tensor) and textual prompts (user_prompts: list[str]).

```python
model.predict(
        x=batch,
        s_x=s_x,
        s_y=s_y,
        user_prompts=["crack"]
)
```

## Model output

In fewpy every model has standardized output dictionary. Every model outputs a dictionary containing a "task" value and a "data" value. Those are the specific task the model is designed for and the prediction data. In addition AnomalyCLIP's output also contains the raw predictions (no postprocessing) as the value for "raw_data".

As previouly mentioned you may or may not use fewpy's dataset class and collate function. A full example of AnomalyCLIP inference using fewpy is availble in [Fewpy/examples/models/anomalyclip.py]().
