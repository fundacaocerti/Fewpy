import pkgutil
import importlib
import os
import sys


__path__ = [os.path.abspath(os.path.dirname(__file__))]
# TODO - solve imports
# __path__[0] = os.path.join(__path__[0], "anomalyCLIP" )
# print(__path__)
# print(list(pkgutil.walk_packages(__path__)))

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    if not is_pkg and module_name.count(".") == 2:
        try:
            importlib.import_module(module_name)
            print(f"imported: {module_name}")
        except Exception as e:
            print(f"Error loading model module {module_name}: {e}", file=sys.stderr)