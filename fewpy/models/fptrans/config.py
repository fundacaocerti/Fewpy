from pydantic import BaseModel, Field
from typing import Dict, Tuple

args = {
    "kshot": 1,
    "dataset": "pascal",
    "backbone": "VIT-B",
    "split": 0,
    "checkpoint": None,
    "Probs_return": False,
    "drop_dim": 1,
    "drop_rate": 0.3,
    "block_size": 16,
    "height": 700,
    "pretrained": "",
    "SAHI": False,
    "bg_num": 5,
    "bsz": 32,
    "img_size": 700,
    "training": False,
    "vit_depth": 10,
    "vit_stride": 23,
    "num_prompt": 12 * 6,
}


class FPTRANSConfig(BaseModel):

    kshot: int = Field(default=1)
    dataset: str = Field(default="pascal")
    backbone: str = Field(default="VIT-B")
    split: int = Field(default=0)
    checkpoint: Dict | None = Field(default=None)

    Probs_return: bool = Field(default=True)
    drop_dim: int = Field(default=1)
    drop_rate: float = Field(default=0.3)
    block_size: int = Field(default=16)
    height: int = Field(default=700)
    pretrained: str = Field(default="")
    
    SAHI: bool = Field(default=False)
    bg_num: int = Field(default=5)
    bsz: int = Field(defaul=32)
    img_size: int = Field(default=700)
    training: bool = Field(default=False)

    vit_depth: int | None = Field(default=10)
    vit_stride: int = Field(default=23)
    coco2pascal: bool = Field(default=False)
    num_prompt: int = Field(default=72)
    pt_std: float = Field(default=0.02)
