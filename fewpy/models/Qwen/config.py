from pydantic import BaseModel, Field
from typing import List, Literal
import torch


class QwenConfig(BaseModel):

    classnames: List[str] = Field(
        description="Names of known and novel classes"
    )

    lora: bool = Field(
        default=False,
        description="Instantiates the model with Low-Rank Adaptation"
    )

    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="Target modules for LoRA training"
    )

    lora_dropout: float = Field(
        default=0.05,
        description="Dropout probability for LoRA"
    )

    lora_alpha: int = Field(
        default=32,
        description="Scaling factor for LoRA"
    )

    lora_rank: int = Field(
        default=16,
        description="Rank of the LoRA matrix, higher values increase training cost"
    )

    lora_bias: str = Field(
        default="none",
        description="Bias type for LoRA"
    )

    gradient_checkpointing: bool = Field(
        deafault=False,
        description="Grandient checkpointing flag"
    )

    quantization: bool = Field(
        default=False,
        description="Model quantization flag"
    )

    quantization_bits: Literal[4, 8] = Field(
        default=4,
        description="Weight precision for quantization"
    )
    
    compute_dtype: str = Field(
        default="bfloat16",
        description="Use bf16 for Ampere/Ada cards"
    )

    max_pixels: int = Field(
        default=512 * 512, 
        description="Max pixels per image to limit VRAM usage"
    )
    
    min_pixels: int = Field(
        default=224 * 224,
        description="Min pixels to ensure small objects aren't lost"
    )

    @property
    def torch_dtype(self):
        return torch.bfloat16 if self.compute_dtype == "bfloat16" else torch.float16
