from pydantic import BaseModel, Field, PrivateAttr
from typing import Callable, List, Dict, Any


class Preprocessor(BaseModel):

    input_keys: List[str] = Field(..., description="List of keys in the context dictionary required as input.")
    output_key: str = Field(..., description="The key to store the function's output in the context dictionary.")
    
    description: str = Field("", description="A brief description of this pre-processing step.")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Keyword arguments passed to the function.")

    _function: Callable = PrivateAttr()
    is_tokenizer: bool = Field(default=False)

    def __init__(self, function: Callable, **data):
        super().__init__(**data)
        if not callable(function):
            raise TypeError("The 'function' provided must be a callable object (e.g., a function or method).")
        self._function = function

    @property
    def function(self) -> Callable:
        return self._function
        
    class Config:
        arbitrary_types_allowed = True
