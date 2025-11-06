from torch import Tensor
from inspect import signature, Parameter
from pydantic import BaseModel

from .preprocessor import Preprocessor
from .register import REGISTRY, CONFIG, CONSTRUCTOR


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

    def get_model_in_features(self):

        return self.params

    def predict(self, **kwargs) -> Tensor:
        
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

        return self.model.predict(**model_inputs)
    
    def add_preprocessor(self, preprocessor: list[Preprocessor] | Preprocessor):

        if isinstance(preprocessor, list):
            self.preprocessors += preprocessor
        elif isinstance(preprocessor, Preprocessor):
            self.preprocessors.append(preprocessor)
        