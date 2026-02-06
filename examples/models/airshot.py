from pathlib import Path
import xml.etree.ElementTree as ET

from fewpy.util.inference.FewShotModel import FewShotModel
from fewpy.util.data.dataset import FSLDataset, fsl_collate
from torch.utils.data import DataLoader
import torch

from PIL import Image
import numpy as np
import detectron2.data.detection_utils as utils


# prepare support and query data 
K = 5
QUERY_CLASS = "sofa"
CLASSES = ("bottle", "sofa")

dataset = Path("./testdata2").expanduser()
annotations = dataset / "selected_annot" 
images = dataset / "selected_img"

counter = {cls: 0 for cls in CLASSES}

annotations = list(annotations.glob("*.xml"))
# print("retrieving", len(annotations), "annotations")
# print()

indices = np.random.permutation(len(annotations))
query_targets = []
query_images = None
support_images = []
support_ground_truth = []

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

    # print("looking at file", image)
    for obj in root.findall("object"):
        name = obj.find("name").text
        # print("object", name)
        if name not in CLASSES:
            continue

        if cls is not None and name != cls:
            continue

        bndbox = obj.find("bndbox")
        bbox = [
            int(bndbox.find("xmin").text) / width,
            int(bndbox.find("ymin").text) / height,
            int(bndbox.find("xmax").text) / width,
            int(bndbox.find("ymax").text) / height
        ]
        bbox = torch.as_tensor(bbox, dtype=torch.float32).unsqueeze(0)

        if first: 
            first = False
            # s_yi["cls"] = mapper[name]
            s_yi["cls"] = name
            s_yi["bboxes"] = bbox
            cls = name
            continue

        torch.stack([s_yi["bboxes"], bbox], dim=0)

    if counter[cls] < K:
        # print("bboxes shape", s_yi["bboxes"].shape)
        # print("bboxes", s_yi["bboxes"])
        support_ground_truth.append(s_yi)
        support_images.append(Image.fromarray(utils.read_image(image, format='BGR')))

        counter[cls] += 1
        # print("added support image of class", cls)
        # print("File", annotation)
    elif cls == QUERY_CLASS and query_images is None:
        query_images = [Image.fromarray(utils.read_image(image, format='BGR'))]
        s_yi["height"] = height
        s_yi["width"] = width
        query_targets = [s_yi]

        # print("added query file")
        # print("File", annotation)


ds = FSLDataset(
    x=query_images,
    s_x=support_images,
    s_y=support_ground_truth,
    img_size=600,
    max_size=1000,
    pixel_norm=((103.530, 116.280, 123.675), (1.0, 1.0, 1.0))       # (mean, std)
)

dl = DataLoader(
    ds, 
    batch_size=1,
    shuffle=False,
    collate_fn=fsl_collate      # Fewpy collate function
)

args = {
    "DATASETNAME": 'JJ',
    "CLASSNAMES": ["bottle", "sofa"],
    "confidence_threshold": 0.5,
    "mapping_to_contiguous_ids": {"bottle": 0, "sofa": 1}
}


print("fewshot config1", args)
# config = AnomalyCLIPConfig(**args)
model = FewShotModel(
    model="AirShot",
    config=args
)

results = []
for batch, s_x, s_y in dl:
    print("batch shape:", batch.shape)
    print("support set shape:", len(s_x), "s_x[0] shape", s_x[0].shape)
    print("groud truth type:", type(s_y))
    results += model.predict(
        x=batch,
        s_x=s_x,
        s_y=s_y
    )

labelid2label = {1: "sofa", 0: "bottle"}

for i, img_predictions in enumerate(results):
    print("image", f"{i+1}:")
    for pred in img_predictions:
        label = labelid2label.get(pred.get("labed_id"))
        conf = pred.get("conf")
        bbox = pred.get("data")
        print(f"label: {label}\nconfidence: {conf}\nbbox: {bbox}")
        