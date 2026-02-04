# CVAT auto annotation with Fewpy and cvat_sdk

## Annotation script

When you annotate images with cvat_sdk the annotation step runs in you machine and not at the server, making it a great tool to use with CVAT Online. But also creating performance limitations depending on the machine you run the script from. So you must carefully consider if the machine you intend to run the annotations from has the computational resources to do so.

### cvat_sdk

The way cvat_sdk works simplifies a lot your setup, all you need a a CVAT server to connect to and a simple script containing a detect method compatible with what cvat_sdk expects. Fewpy also has fewpy.util.cvat.adapter.CVATAdapter that converts dictionaries of format {"task": task, "data": data} to the correct return values expected by cvat_sdk from your detect method. The cvat_sdk library deals with batching so all your script needs to implement is a setup and a detect method that deals with a single image at a time.

```python
mport cvat_sdk.auto_annotation as cvataa
import torch
import numpy as np
import sys
import os

from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET
import open_clip as clip
import numpy as np

# script_dir = Path(__file__).parent.resolve()
# sys.path.append(str(script_dir))
sys.path.append(os.getcwd())

from fewpy.util.inference.FewShotModel import FewShotModel
from fewpy.util.inference.preprocessor import Preprocessor
from fewpy.util.cvat.adapter import CVATAdapter
import torchvision.transforms as T


preprocess = T.Compose([
    T.Resize((448, 448)),              
    T.CenterCrop(448),          
    T.ToTensor(),
    T.Normalize(                
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])

preprocess_sy = T.Compose([
    T.Resize((448, 448)),              
    T.CenterCrop(448),          
    T.ToTensor()
])

# load support set
dataset = Path(__file__).parent.resolve() / "support_set"
annotations = dataset / "selected_annot"
images = dataset / "selected_img"

s_x = []
s_y = []

annotations = list(annotations.glob(f"*.png"))

indices = np.random.permutation(len(annotations))
for i in indices:
    annotation = annotations[i]

    s_yi = Image.open(annotation).convert("L")
    arr = np.asarray(s_yi)
    arr = (arr > 127).astype(np.uint8) * 255
    s_yi = Image.fromarray(arr)
    s_xi = Image.open((images / annotation.name).with_suffix(".jpg")).convert("RGB")

    s_xi = preprocess(s_xi)
    s_yi = preprocess_sy(s_yi).squeeze(0)

    s_x.append(s_xi)
    s_y.append(s_yi)

s_x = torch.stack(s_x)
s_y = torch.stack(s_y)

args = {
        "image_size": 512,
        "n_ctx": 4,
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

print("sx", s_x.shape)
print("sy", s_y.shape)

spec = cvataa.DetectionFunctionSpec(
    labels=[cvataa.label_spec("bottle", 0)] #  cvataa.label_spec("sofa", 1)
)
LABEL_MAP = {label.name: label.id for label in spec.labels}

def detect(context, image):
    img_tensor = preprocess(image).unsqueeze(0)
    print("x", img_tensor.shape)

    predictions = model.predict(
        x=img_tensor,
        s_x=s_x.to(model.device),
        s_y=s_y.to(model.device),
        user_prompts=[label.name for label in spec.labels]
    )

    return CVATAdapter.to_cvat(predictions, LABEL_MAP)
```

## Running auto annotation

In order to run auto annotation as in the example you must have cvat_sdk (for the script) and cvat_cli installed in your active enviroment. After guaranteeing the installation of both and that the task is correctly configured in your server run the following command:

```shell
cvat-cli --server-host <YOUR_SERVER_URL> \
         --auth <YOUR_USERNAME>:<YOUR_PASSWORD> \
         task auto-annotate <TASK_ID> \
         --function-file <PATH_TO_YOUR_FILE>
```