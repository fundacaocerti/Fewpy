from pydantic import BaseModel, Field
from typing import List


class AirShotConfig(BaseModel):

    datasetname: str = Field(default="TEST", description="Name of the inference dataset")
    classnames: List[str] = Field(
        description="name (str) of each class in the dataset"
    ) 

    mapping_to_contiguous_ids: dict = Field(
        descriptio="id mapper in case dataset does not have contiguous ids"
    )

    confidence_threshold: float = Field(
        default=0.5,
        description="lower bound of confidence for accepted proposals"
    )

    support_shot: int = Field(default=5, description="Number of examples per class in the support set")
    support_way: int = Field(default=5, description="Number of novel classes in the support set")
