from pydantic import BaseModel, Field
from typing import List
from torch.nn import Module as Module


class PrototypicalHeadConfig(BaseModel):

    kshot: int = Field(default=1)
    backbone: str = Field(default="")
    training: bool = Field(default=False)
    load_adapter: bool = Field(default=False)
    cache_prototypes: bool = Field(default=False)
    cache_dir: str = Field(default="")
    detection_threshold: float = Field(default=0.5)
    task: str = Field(default="segmentation")
    img_h: int = Field(default=224)
    img_w: int = Field(default=224)
