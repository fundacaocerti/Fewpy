from pydantic import BaseModel, Field
from typing import List
import torch


class PrototypicalHeadConfig(BaseModel):

    kshot: int = Field(default=1)
    backbone: str = Field(default="")
    training: bool = Field(default=False)
    # load_adapter: bool = Field(default=False)
    n_classes: int = Field(default=0)
    embedding_size: int = Field(default=512)
    cache_prototypes: bool = Field(default=False)
    cache_dir: str = Field(default="")
    detection_threshold: float = Field(default=0.5)
    task: str = Field(default="segmentation")
    dtype: str = Field(
        default="float32", 
        description="Precision to use: 'float32', 'float16', or 'bfloat16'"
    )
    textual_scale: float = Field(default=1.0, description="Scale for textual features when combining with visual features")
    augment_epoch: int = Field(default=0, description="Number of epochs the support set is augmented")

    @property
    def torch_dtype(self) -> torch.dtype:
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        dtype_str = self.dtype.lower()
        if dtype_str not in mapping:
            raise ValueError(f"Unsupported dtype: {self.model_dtype}. Use 'float32', 'float16', or 'bfloat16'.")
        return mapping[dtype_str]