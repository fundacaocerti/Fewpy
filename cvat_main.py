import json
import torch
from fewpy.util.inference.FewShotModel import FewShotModel
from fewpy.util.inference.preprocessor import Preprocessor

import base64
import io
import numpy as np
from PIL import Image
from pathlib import Path


"""
Example code for AnomalyClip
Some small changes will be necessary to run other models on cvat
Mainly in the initialization of the model, event handling and function.yaml class definitions
"""
def init_context(context):

    def load_support_set(
            k: int=5, 
            classes: list=["bottle", "sofa"],
            query_cls: str="sofa"   
        ):

        dataset = Path("./support_set").expanduser()
        annotations = dataset / "selected_annot"
        images = dataset / "selected_img"

        counter = {cls: 0 for cls in classes}

        s_x = []
        s_y = []
        x = None

        annotations = list(annotations.glob("*.xml"))

        indices = np.random.permutation(len(annotations))
        for i in indices:
            annotation = annotations[i]

            tree = ET.parse(annotation)
            root = tree.getroot()
            image = images / root.find("filename").text
            s_yi = dict()
            first = True
            cls = None

            size = root.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)

            for obj in root.findall("object"):
                name = obj.find("name").text
                if name not in classes:
                    continue

                if cls is not None and name != cls:
                    continue

                bndbox = obj.find("bndbox")
                bbox = [
                    int(bndbox.find("xmin").text),
                    int(bndbox.find("ymin").text),
                    int(bndbox.find("xmax").text),
                    int(bndbox.find("ymax").text)
                ]

                if first: 
                    first = False
                    # s_yi["cls"] = mapper[name]
                    s_yi["cls"] = name
                    s_yi["bboxes"] = bbox
                    cls = name
                    continue

            if counter[cls] < k:
                s_x.append(Image.open(image).convert("RGB"))
                s_y.append(s_yi)
        
                counter[cls] += 1

            if all((c >= k for c in counter.values())):
                break

        return s_x, s_y

    context.logger.info("Loading Fewpy model...")

    import open_clip as clip
    # initialize your model
    args = {
        "image_size": 16,
        "n_ctx": 4,
        "t_n_ctx": 4,
    }
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

    context.user_data.s_x, context.user_data.s_y = load_support_set()

def handler(context, event):
    device = context.user_data.model.device
    encoded_img = event.body["image"]
    img_bytes = base64.b64decode(encoded_img)
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    numpy_image = np.asrray(pil_image)  # may need to make it BGR using [:, :, ::-1]
    img = torch.Tensor(numpy_image).to(device)
    
    predictions = context.user_data.model.predict(
        x=img,
        s_x=context.user_data.s_x,
        s_y=context.user_data.s_y,
        user_prompts=["some user prompts", "just a test", "som epr", "ewqeq"]
    )
    
    # post processing + making it a json
    results = [
        {
            "type": "mask",
            "mask": predictions
        }
    ]

    return context.Response(body=json.dumps(results), content_type='application/json')