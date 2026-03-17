# Running Qwen with fewpy

In order to run the Qwen implementation within fewpy you will need:
    - to import fewpy's FewShotModel class
    - to import fewpy's FSLDataset (optional)

## Importing necessary fewpy tooling

In order to instatiate the model FewShotModel is necessary, so import it as in the following snippet.

```python
from fewpy.inference import FewShotModel
```

You may or may not use fewpy's FSLDataset class. Notice that it is a recommendation to use torch's DataLoader with FSLDataset but any compatible implementation works. Fewpy also contains a collate function that should be used with torch.util.data.DataLoader.

Since Qwen takes list[PIL.Image.Image] as input, you will need a collate function that does not stack the batch. If using FSLDataset initialize with transform_datapoints=False.

```python
from fewpy.util.data import FSLDataset
from torch.utils.data import DataLoader
```

## Instatiating Qwen

Now that you imported FewShotModel all you will need to instatiate the model is a configuration dictionary, name it whatever works best for as it will be passed to the FewShotModel contructor.

An example of configuration dictionary would be:

```python
args = {
    "classnames": ["bottle", "sofa"],   # a list containing the name of each class in your dataset
    # fine tuning:
    "lora": True,                       # flag for Low-Rank Adaptation usage during training (recommended True for fine tuning)
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],                       # list of target modules for LoRA
    "lora_dropout": 0.05,               # LoRA dropout probability
    "lora_alpha": 32,                   # LoRA scaling factor (recommended 2 * LoRA rank)
    "lora_rank": 16,                    # LoRA rank (higher values increase learning capabilities at the cost of computational resources)
    "lora_bias": "none",                # bias type for LoRA
    "gradient_checkpointing": True,     # grandient checkpointing flag
    "quantization": True,               # quantization flag
    "quantization_bits": 8              # quantization precision 8 for better precision, 4 for low cost
    "compute_dtype": "bfloat16",        # use bf16 for Ampere/Ada cards
    "max_pixels": 512 * 512,            # max pixels per image when loaded to the model
    "min_pixels": 224 * 224,            # min pixels per image when loaded to the model
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

As previouly mentioned you may or may not use fewpy's dataset class and collate function. A full example of Qwen inference using fewpy is availble in [Fewpy/examples/models/qwen.py](./qwen.py).

