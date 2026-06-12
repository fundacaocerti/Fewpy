from torch import Tensor
from inspect import signature, Parameter

from pydantic import BaseModel, Field, PrivateAttr
from typing import Callable, List, Dict, Any

from fewpy.metrics import DistanceMetric
from fewpy.models.register import REGISTRY, CONFIG, CONSTRUCTOR
import fewpy.models
    

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


class FewShotModel:

    def __init__(self, model: str, config: dict=None, preprocessors: list[Preprocessor]=[]) -> None:

        self.model, self.device = self.__load_model(model, config)
        params = signature(self.model.predict).parameters
        self.default_params = []
        self.params = []
        for k, p in params.items():
            if p.default is not Parameter.empty:
                self.default_params.append(k)
            else:
                self.params.append(k)
        self.preprocessors = []
        if isinstance(preprocessors, list):
            self.preprocessors += preprocessors

    @staticmethod
    def __load_model(model: str, config: dict):

        return REGISTRY[model][CONSTRUCTOR](REGISTRY[model][CONFIG](**config)).instantiate_model()
    
    @staticmethod
    def get_available_models():
        return REGISTRY.keys()
    
    @property
    def training(self):
        
        return self.model.training

    def encode_image(self, batch):

        if hasattr(self.model, "encode_image") and callable(getattr(self.model, "encode_image")):
            return self.model.encode_image(batch)
        else:
            return []

    def get_model_in_features(self):

        return self.params
    
    def train(self):

        self.model.train()

    def eval(self):

        self.model.eval()

    def parameters(self):

        return self.model.parameters()

    def named_parameters(self):

        return self.model.named_parameters()
    
    def state_dict(self):

        return self.model.state_dict()
    
    def __str__(self):

        return str(self.model)

    def predict(self,*args, **kwargs) -> Tensor:
        
        for step in self.preprocessors:
            
            try:
                step_inputs = [kwargs[key] for key in step.input_keys]
            except KeyError as e:
                raise ValueError(
                    f"Missing input key '{e.args[0]}' for preprocessor '{step.function.__name__}'. "
                    f"Available keys: {list(kwargs.keys())}"
                )
            
            if step.is_tokenizer:
                result = step.function(*step_inputs, **step.kwargs).to(self.device)
            else:
                result = step.function(*step_inputs, **step.kwargs)

            kwargs[step.output_key] = result

        try:   
            model_inputs = {k: kwargs[k] for k in self.params}
            for k in self.default_params:
                if k in kwargs:
                    model_inputs[k] = kwargs[k]
        except KeyError as e:
            raise ValueError(
                    f"Missing input key '{e.args[0]}' for model. "
                    f"Available keys: {list(kwargs.keys())}"
                )

        return self.model.predict(*args, **model_inputs)
    
    def add_preprocessor(self, preprocessor: list[Preprocessor] | Preprocessor):

        if isinstance(preprocessor, list):
            self.preprocessors += preprocessor
        elif isinstance(preprocessor, Preprocessor):
            self.preprocessors.append(preprocessor)