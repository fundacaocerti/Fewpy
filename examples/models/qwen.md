# Running Qwen with fewpy

In order to run the Qwen implementation within fewpy you will need:
    - to import fewpy's FewShotModel class
    - to import fewpy's FSLDataset (optional)

## Importing necessary fewpy tooling

In order to instatiate the model FewShotModel is necessary, so import it as in the following snippet.

```python
from fewpy.util.inference.FewShotModel import FewShotModel
```

You may or may not use fewpy's FSLDataset class. Notice that it is a recommendation to use torch's DataLoader with FSLDataset but any compatible implementation works. Fewpy also contains a collate function that should be used with torch.util.data.DataLoader.

Since Qwen takes list[PIL.Image.Image] as input, you will need a collate function that does not stack the batch. If using FSLDataset initialize with transform_datapoints=False.

```python
from fewpy.util.data.dataset import FSLDataset, qwen_collate
from torch.utils.data import DataLoader
```

## Instatiating Qwen

Now that you imported FewShotModel all you will need to instatiate the model is a configuration dictionary, name it whatever works best for as it will be passed to the FewShotModel contructor.

An example of configuration dictionary would be:

```python
args = {
    "classnames": ["bottle", "sofa"],   # a list containing the name of each class in your dataset
}
```

## Preparing your input

Qwen expects a query_batch (x: list[PIL.Image.Image]), a support image tensor (s_x: list[PIL.Image.Image]), a support ground truth tensor (s_y: list[dict]) and textual prompts (user_prompts: list[str]).

```python
model.predict(
        x=batch,
        s_x=support_images,
        s_y=support_ground_truth,
)
```

## Model output

In fewpy every model has standardized output dictionary. Every model outputs a dictionary containing a "task" value and a "data" value. Those are the specific task the model is designed for and the prediction data. In addition Qwen's output also contains confidence scores as the value for "conf".

As previouly mentioned you may or may not use fewpy's dataset class and collate function. A full example of Qwen inference using fewpy is availble in [Fewpy/examples/models/qwen.py]().
