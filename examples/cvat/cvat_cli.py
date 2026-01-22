import cvat_sdk.auto_annotation as cvataa
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


def load_support_set(
        k: int=5, 
        classes: list=["bottle", "sofa"],
        annotation_type: str="xml",

        device: str="cpu"
    ):

    dataset = Path(__file__).parent.resolve() / "support_set"
    annotations = dataset / "selected_annot"
    images = dataset / "selected_img"

    counter = {cls: 0 for cls in classes}

    s_x = []
    s_y = []

    annotations = list(annotations.glob(f"*.{annotation_type}"))

    indices = np.random.permutation(len(annotations))
    for i in indices:
        annotation = annotations[i]


        match annotation_type:

            case "xml":
                tree = ET.parse(annotation)
                root = tree.getroot()
                image = images / root.find("filename").text
                s_yi = dict()
                first = True
                cls = None

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
                        s_yi["cls"] = name
                        s_yi["bboxes"] = bbox
                        cls = name
                        continue

                if counter[cls] < k:
                    img = Image.open(image).convert("RGB")
                    img = torch.Tensor(np.array(img)).to(device)
                    s_x.append(img)
                    s_y.append(s_yi)
            
                    counter[cls] += 1

                if all((c >= k for c in counter.values())):
                    break

            case "png":

                s_yi = Image.open(annotation).convert("L")
                arr = np.asarray(s_yi)
                arr = (arr > 127).astype(np.uint8) * 255
                s_yi = Image.fromarray(arr)
                s_xi = Image.open((images / annotation.name).with_suffix(".jpg")).convert("RGB")

                s_xi = preprocess(s_xi).to(device)
                s_yi = preprocess_sy(s_yi).to(device).squeeze(0)

                s_x.append(s_xi)
                s_y.append(s_yi)

    return s_x, s_y

preprocess = T.Compose([
    T.Resize(448),              
    T.CenterCrop(448),          
    T.ToTensor(),
    T.Normalize(                
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])

preprocess_sy = T.Compose([
    T.Resize(448),              
    T.CenterCrop(448),          
    T.ToTensor()
])

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

s_x, s_y = load_support_set(
    k=1,
    annotation_type="png"
)

s_x = torch.stack(s_x)
print("sx", s_x.shape)
s_y = torch.stack(s_y)
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
        user_prompts=["some user prompts", "just a test", "som epr", "ewqeq"]
    )

    return CVATAdapter.to_cvat(predictions, LABEL_MAP)