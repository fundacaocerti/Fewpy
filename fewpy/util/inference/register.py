from typing import Dict, Any, Callable


CONSTRUCTOR_REGISTRY: Dict[str, Any] = {}

def register_constructor(name: str):

    def wrapper(constructor_cls) -> Callable:
        # Check for duplicate registrations
        if name in CONSTRUCTOR_REGISTRY:
            raise ValueError(f"Model with name '{name}' is already registered.")
        
        # Add the model and config to the registry
        CONSTRUCTOR_REGISTRY[name] = constructor_cls

        return constructor_cls
    return wrapper
