from typing import Dict, Any, Callable, Tuple


REGISTRY: Dict[str, Tuple[Any]] = {}
CONFIG = "config"
CONSTRUCTOR = "constructor"

def register_constructor(name: str, config_cls):

    def wrapper(constructor_cls) -> Callable:
        # Check for duplicate registrations
        if name in REGISTRY:
            raise ValueError(f"Model with name '{name}' is already registered.")
        
        # Add the model and config to the registry
        REGISTRY[name] = {
            "constructor": constructor_cls,
            "config": config_cls
        }

        return constructor_cls
    return wrapper
