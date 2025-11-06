import pkgutil
import importlib
import os
import sys

invalid_dir = ["__pycache__", "weights"]

def check_for_dep(path):

    for root, dirnames, _ in os.walk(str(path)):
        for dir in dirnames:
            if dir not in invalid_dir: check_inside(os.path.join(root, dir))

def check_inside(path):

    pkg_name = __name__

    for _, name, ispkg in pkgutil.iter_modules([path]):
        try:
            if not ispkg:
                parent_module = ""
                split_path = path.split(os.sep)
                for p in split_path[split_path.index('models')+1:]:
                    parent_module += p + "."
                importlib.import_module(f"{pkg_name}.{parent_module}{name}")
        except Exception as e:
            print(f"Failed to import module {name}.", e)

check_for_dep(os.path.dirname(__file__))
