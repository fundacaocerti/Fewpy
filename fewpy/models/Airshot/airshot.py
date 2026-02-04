"""
Copyright (c) 2024, Zihan Wang

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from pathlib import Path
import os

from detectron2.checkpoint import DetectionCheckpointer

from fewpy.models.Airshot.config import AirShotConfig

import torch
from fewpy.models.Airshot.fewx.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from typing import List

from fewpy.util.inference.register import register_constructor


class AirShot(torch.nn.Module):

    def __init__(self, cfg, device="cpu"):
        super(AirShot, self).__init__()

        self.model = build_model(cfg)
        self.model.to(device)
        self.model.eval()

        self.device = device

        DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).load(
            cfg.MODEL.WEIGHTS
        )
        self.cached = False
        # self.predictor = Predictor(cfg)

    def forward(self, x: dict):

        return self.model(x)
    
    def predict(self, x: List[torch.Tensor], s_x: List[torch.Tensor] | List[str], s_y: List[dict]):
        """
        self.model.forward:
        Args:
            x: a list of Tensors of fomat (C, H, W), the batched query.
            s_x: a list of Tensors of fomat (C, H, W), the support images.
            s_y: a list of dictionaries containing the gorund truth for each of the support images.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The dict contains the following keys:
                key "task" that specifies the task the model is trained on (always "detection")
                key "label_id", contains the id of the detected object
        """

        with torch.no_grad():

            if not self.cached:
                support_set = [
                    {
                        "support_box": s_yi["bboxes"],
                        "category_id": s_yi["cls"],
                        "image": s_xi
                    } for s_xi, s_yi in zip(s_x, s_y)]
            else:
                support_set = None


            batched_inputs = [{
                "image": xi.to(self.device),
                "height": xi.size(-2),
                "width": xi.size(-1),
            } for xi in x]

            x = {
                "batched_inputs": batched_inputs,
                "support_set": support_set
            }

            self.cached = True

            results = []
            for item in self(x):
                instances = item.get("instances")
                top_instance = instances[instances.scores.argmax()]
                score = top_instance.score.item()
                bboxes = top_instance.pred_boxes.tensor.cpu().tolist()
                labels = top_instance.pred_classes.cpu().tolist()
                img_results = []
                for bbox, label in zip(bboxes, labels):
                    img_results.append({
                        "task": "detection",
                        "label_id": label,
                        "conf": score,
                        "data": bbox        # [xmin, ymin, xmax, ymax] == [xtl, ytl, xbr, ybr]
                    })
                results.append(img_results)

            return results

@register_constructor(name="AirShot", config_cls=AirShotConfig)
class constructor_AirShot:

    def __init__(self, cfg):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, "weights", "checkpoint.pth")
        if not os.path.exists(weights_path):
            weights_path = os.path.join("./weights", "checkpoint.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError("Model weights not found!")
        
        cfg_path = os.path.join(current_dir, "configs", "fsod", "R101", "test_R_101_C4_1x_subt3_a.yaml")
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join("./configs", "fsod", "R101", "test_R_101_C4_1x_subt3_a.yaml")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError("Model config not found!")

        DatasetCatalog.register(cfg.DATASETNAME, lambda : [])
        metadata = MetadataCatalog.get(cfg.DATASETNAME)
        metadata.set(thing_classes=cfg.CLASSNAMES)
        metadata.set(thing_dataset_id_to_contiguous_id = cfg.mapping_to_contiguous_ids)

        # print("cfg", cfg)

        self.cfg = get_cfg()
        self.cfg.merge_from_file(cfg_path)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.confidence_threshold
        self.cfg.DATASETS.TEST = [cfg.DATASETNAME]
        self.cfg.MODEL.WEIGHTS = str(weights_path)
        self.cfg.freeze()
        
    def instantiate_model(self, device=None):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = AirShot(self.cfg, device)
        model.eval()
        model.to(device)

        return model, device
