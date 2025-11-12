from pydantic import BaseModel, Field
from typing import Dict, Tuple


class FPTRANSConfig(BaseModel):

    kshot: int = Field(default=1)
    data_set: str = Field(default="pascal")
    backbone: str = Field(default="VIT-B")
    split: Tuple[int] = Field(default_factory=lambda: (0.9, 0.1))
    checkpoint: Dict | None = Field(default=None)

    Probs_return: bool = Field(default=True)
    drop_dim: int = Field(default=1)
    drop_rate: float = Field(default=0.3)
    block_size: int = Field(default=16)
    height: int = Field(default=700)
    pretrained: str = Field(default="")
    
    SAHI: bool = Field(default=None)
    bg_num: int = Field(default=16)
    bsz: int = Field(defaul=32)
    img_size: int = Field(default=700)
    training: bool = Field(default=False)
