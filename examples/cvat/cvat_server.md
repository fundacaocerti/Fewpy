# Deploying a Fewpy Model to CVAT

In this example we deploy anomalyCLIP to be used a Cvat Community server. In order to run this test after downloading CVAT Community, inside its root directory you will need to set up a directory with a function.yaml and a main.py files and a support_set. In our example we also clone the Fewpy repository, but in your case it should be installed through docker directives in your function.yaml file.

## Setting up a CVAT server

CVAT Community should be installed following the CVAT documentation bellow:
[Cvat Community](https://docs.cvat.ai/docs/administration/community/basics/installation/)

To deploy the server use the following shell command:
´´´shell
docker compose -f docker-compose.yml -f docker-compose.dev.yml -f components/serverless/docker-compose.serverless.yml up -d
´´´

## Setting up your model

In order to deploy your models as a nuclio serverless function there are two scripts inside the CVAT folder, ./cvat/serversless/deploy_cpu.sh and ./cvat/serversless/deploy_gpu.sh.

The shell command bellow runs the script that will look for a function.yaml and a main.py files inside /cvat/fewpy_test/ and deploy your serverless function
```shell
cvat$ ./serverless/deploy_cpu.sh ./fewpy_test/
```

### function.yaml

Your function.yaml file will be used by the deployment shell script to produce a dockerfile. In the case of this example we use the following function.yaml:

```yaml
metadata:
  name: Fewpy-CVAT
  namespace: cvat
  labels:
    author: cvat
    type: detector
    framework: pytorch
  annotations:
    name: Fewpy AnomalyCLIP
    type: detector
    framework: pytorch
    spec: |
      [
        { "id": 0, "name": "object", "type": "mask" }
      ]

spec:
  # resources:
  #   limits:
  #     nvidia.com/gpu: 1
  description: "AnomalyCLIP, loaded from Fewpy"
  handler: main:handler
  runtime: python:3.11
  eventTimeout: 60s

  env:
    - name: PYTHONPATH
      value: "/opt/nuclio:/opt/nuclio/.."

    - name: UV_LINK_MODE
      value: "copy"

    - name: UV_PYTHON_CACHE_DIR
      value: "/root/.cache/uv/python"

  build:
    image: cvat/fewpy:latest
    baseImage: python:3.11-slim

    directives:

      preCopy:
        - kind: RUN
          value: |
            apt-get update && apt-get install -y --no-install-recommends \
            curl git build-essential pkg-config \
            libgl1 libglx-mesa0 libglib2.0-0 \
            python3-dev ninja-build && \
            rm -rf /var/lib/apt/lists/*

        - kind: RUN
          value: curl -LsSf https://astral.sh/uv/install.sh | sh && mv /root/.local/bin/uv /usr/local/bin/uv

      postCopy:
        - kind: RUN
          value: |
            --mount=type=cache,target=/root/.cache/uv \
            uv pip install --system torch torchvision setuptools wheel

        - kind: RUN
          value: |
            --mount=type=cache,target=/root/.cache/uv \
            uv pip install --system --no-build-isolation /opt/nuclio/Fewpy/

        - kind: RUN
          value: rm -rf /opt/nuclio/Fewpy

        - kind: RUN
          value: uv pip install --system debugpy

  triggers:
  myHttpTrigger:
    numWorkers: 1
    kind: 'http'
    workerAvailabilityTimeoutMilliseconds: 10000
    attributes:
      maxRequestBodySize: 33554432 # 32MB

  platform:
    attributes:
      restartPolicy:
        name: always
        maximumRetryCount: 3
      mountMode: volume
```

### main.py

The main.py file will define two functions, init_context and handler. In our init_context function we load the model and support_set and save them on context.user_data. The handler function will be called by CVAT in order to annotate an image.

```python
import json
import torch
from fewpy.util.inference.FewShotModel import FewShotModel
from fewpy.util.inference.preprocessor import Preprocessor

import base64
import io
import numpy as np
from PIL import Image
from pathlib import Path

import open_clip as clip
import torchvision.transforms as T


import debugpy
debugpy.listen(5678)


"""
Example code for AnomalyClip
Some small changes will be necessary to run other models on CVAT
Mainly in the initialization of the model, event handling and function.yaml class definitions
"""
def init_context(context):
    # resize and normalize pixels
    context.user_data.preprocess = T.Compose([
        T.Resize((448, 488)),
        T.CenterCrop(448),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])
    # resize and make it a tensor
    preprocess_sy = T.Compose([
        T.Resize((448, 488)),
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

        s_xi = context.user_data.preprocess(s_xi)
        s_yi = preprocess_sy(s_yi).squeeze(0)

        s_x.append(s_xi)
        s_y.append(s_yi)

    s_x = torch.stack(s_x)
    s_y = torch.stack(s_y)

    context.user_data.s_x = s_x
    context.user_data.s_y = s_y

    context.logger.info(f"tensor shape: s_x: {s_x.shape}; s_y: {s_y.shape}")
    context.logger.info("\n---------Loading Fewpy model...---------\n")

    # model config
    args = {
        "image_size": 448,
        "n_ctx": 4,
        "t_n_ctx": 4,
    }
    # model loading
    context.user_data.model = FewShotModel(
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

    context.logger.info("\n---------Model loaded to context!---------\n")

def handler(context, event):

    context.logger.info("\n---------Event Handler called!---------\n")

    encoded_img = event.body["image"]
    # extra_params = event.body.get("extra_params", {})
    # prompts = extra_params.get("prompts", ["object"])
    mapping = event.body.get("mapping", {})
    prompts = list(mapping.values())
    img_bytes = base64.b64decode(encoded_img)
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    W, H = pil_image.size
    img = context.user_data.preprocess(pil_image).unsqueeze(0)

    context.logger.info("\n---------Image preprocessed!---------\n")
    context.logger.info(f"\n---------Image size: {img.shape}---------\n")

    predictions = context.user_data.model.predict(
        x=img,
        s_x=context.user_data.s_x,
        s_y=context.user_data.s_y,
        user_prompts=prompts
    )

    results = []
    for pred in predictions:
        # resize and flatten mask
        mask = pred["postproc_data"].squeeze(0).numpy().astype(np.uint8)
        mask = Image.fromarray(mask)
        mask = mask.resize((W, H), resample=Image.NEAREST)
        mask = np.array(mask).flatten().tolist()    # the CVAT ui expects a flattened mask

        # format json output
        results.append(
            {
                "type": "mask",
                "mask": mask,
                "label": "object"
            }
        )

    context.logger.info("\n---------Got output from model!---------\n")

    return context.Response(body=json.dumps(results), content_type='application/json', status_code=200)
```

Notice that in this example, since we work with a segementation model, which implies one output per image the result variable holds a list of dictionaries. But in the case of object detection, where a single image could result in several bounding boxes each of these detections should be encapsulated in one dictionary. Therefore your output should be a nested list of dicctionaries (list[list[dict]]).

## Extra Tips

### Deployment Bugs

It is important to point out that if you deploy many nuclio functions and there is a bug in one of them you might not be able to use any models that you deployed. If the bug is caught during model registering that outcome is likely.

### Nuclio functions stuck at 'building' status

If for some reason the thread building your function was stopped. Then their status will be stuck at 'building'. In that case, this function will cause trouble for future deployments. Functions stuck with the 'building' status cannot be removed with 'nuctl delete function'. They have to be deleted directly from the cache directory. That directory runs separately. So in order to remove your function do:

List the nuclio processes
```shell
docker ps -a | grep nuclio
```

Access that process
```shell
docker exec -it <id> /bin/sh
```

Find the cache directory and delete the function .json
```shell
cd etc/nuclio/store/functions/nuclio/
rm -rf <your function name>.json
exit
```

Finally, restard the dashboard processes
```shell
docker restart <dashboard id>
```

### Read the CVAT/nuclio documentation

It will be very helpful to take a look at the documentation. LLMs will not always be able to precisely diagnose the problem. If using an LLM remember to give detailed context.
