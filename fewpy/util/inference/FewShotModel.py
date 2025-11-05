from torch import Tensor
from inspect import signature
from pydantic import BaseModel

from .preprocessor import Preprocessor
from .register import CONSTRUCTOR_REGISTRY


class FewShotModel:

    def __init__(self, model: str, config: BaseModel=None, preprocessors: list[Preprocessor]=[]) -> None:
        
        self.model, self.device = self.__load_model(model, config)
        self.params = signature(self.model.predict)
        if isinstance(preprocessors, list):
            self.preprocessors += preprocessors

    @staticmethod
    def __load_model(model: str, config: dict):
        
        return CONSTRUCTOR_REGISTRY[model](config).instantiate_model()

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
            model_inputs = [kwargs[k] for k in self.params]
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
        