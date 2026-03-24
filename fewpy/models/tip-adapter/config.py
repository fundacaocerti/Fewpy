from pydantic import BaseModel, Field
from typing import List

class TipAdapterConfig(BaseModel):

    kshot: int = Field(default=1)
    clip_model: str = Field(default="RN50")
    cache_kv: bool = Field(default=False)
    cache_dir: str = Field(default="")
    augment_epoch: int = Field(default=1)
    load_adapter: bool = Field(default=False)
    training: bool = Field(default=False)
    beta: float = Field(default=1.0)
    alpha: float = Field(default=1.17)
    show_logits: bool = Field(default=False)
    n_novel_classes: int = Field(default=16)