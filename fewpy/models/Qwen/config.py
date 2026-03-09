from pydantic import BaseModel, Field
from typing import List


class QwenConfig(BaseModel):

    classnames: List[str] = Field(
        description="Names of known and novel classes"
    )
