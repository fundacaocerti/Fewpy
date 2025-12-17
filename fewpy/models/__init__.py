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
    import_errors = 0
    failed_imports = []
    errors = []

    for _, name, ispkg in pkgutil.iter_modules([path]):
        # print("looking at module", name)
        try:
            if not ispkg:
                parent_module = ""
                split_path = path.split(os.sep)
                for p in split_path[split_path.index('models')+1:]:
                    parent_module += p + "."
                    module = f"{pkg_name}.{parent_module}{name}"
                    # print("importing", module, "parent", parent_module, "pkg name", pkg_name)
                    importlib.import_module(module)
                    # print("imported module", module)
        except Exception as e:
            import_errors += 1
            errors.append(e)
            failed_imports.append(name)
            
        print(f"Failed to import {import_errors} modules. See ./import.log")
        
        log = ""
        for error, module in zip(errors, failed_imports):
            log += f"Failed to import module {module}. {error}\n"
        with open("./import.log", "w") as f:
            f.write(log)

print("importing models")
check_for_dep(os.path.dirname(__file__))
