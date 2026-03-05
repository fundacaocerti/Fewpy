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

from detectron2.checkpoint import DetectionCheckpointer

from fewpy.models.Airshot.config import AirShotConfig
from fewpy.models.Airshot.fewx.data.dataset_mapper import DatasetMapperWithSupport
from fewpy.models.Airshot.fewx.data.build import build_detection_train_loader, build_detection_test_loader
from fewpy.models.Airshot.fewx.solver import build_optimizer
from fewpy.models.Airshot.fewx.evaluation import COCOEvaluator

import torch
from fewpy.models.Airshot.fewx.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Instances, Boxes, ImageList
from typing import List

from fewpy.util.inference.register import register_constructor


# class Predictor(DefaultPredictor):

    # @classmethod
    # def build_train_loader(cls, cfg):
    #     """
    #     Returns:
    #         iterable
    #     It calls :func:`detectron2.data.build_detection_train_loader` with a customized
    #     DatasetMapper, which adds categorical labels as a semantic mask.
    #     """
    #     mapper = DatasetMapperWithSupport(cfg)
    #     return build_detection_train_loader(cfg, mapper)

    # @classmethod
    # def build_test_loader(cls, cfg, dataset_name):
    #     """
    #     Returns:
    #         iterable
    #     It now calls :func:`detectron2.data.build_detection_test_loader`.
    #     Overwrite it if you'd like a different data loader.
    #     """
    #     return build_detection_test_loader(cfg, dataset_name)

    # @classmethod
    # def build_optimizer(cls, cfg, model):
    #     """
    #     Returns:
    #         torch.optim.Optimizer:
    #     It now calls :func:`detectron2.solver.build_optimizer`.
    #     Overwrite it if you'd like a different optimizer.
    #     """
    #     return build_optimizer(cfg, model)

    # @classmethod
    # def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    #     if output_folder is None:
    #         output_folder = Path(cfg.OUTPUT_DIR) / "inference"
    #     return COCOEvaluator(dataset_name, cfg, True, output_folder)


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
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
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

            result = self(x)
            print("result")
            return result

@register_constructor(name="AirShot", config_cls=AirShotConfig)
class constructor_AirShot:

    def __init__(self, cfg):

        cfg_path = Path("fewpy").expanduser() / "models" / "Airshot" / "configs" \
            / "fsod" / "R101" / f"test_R_101_C4_1x_subt3_a.yaml"
        weights_path = Path("fewpy").expanduser() / "models" / "Airshot" / "weights" \
            / "checkpoint.pth"

        DatasetCatalog.register(cfg.DATASETNAME, lambda : [])
        metadata = MetadataCatalog.get(cfg.DATASETNAME)
        metadata.set(thing_classes=cfg.CLASSNAMES)
        metadata.set(thing_dataset_id_to_contiguous_id = cfg.mapping_to_contiguous_ids)

        print("cfg", cfg)

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
