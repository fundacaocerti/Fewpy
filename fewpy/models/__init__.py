import pkgutil
import importlib
import os

print("--- Starting Model Registration ---")
pkg_path = os.path.dirname(__file__)
pkg_name = __name__

# walk_packages is more robust for nested directories than iter_modules
for loader, module_name, is_pkg in pkgutil.walk_packages([pkg_path], pkg_name + "."):
    # print(module_name)
    if not is_pkg:
        try:
            importlib.import_module(module_name)
            # print(f"\n\nSuccessfully registered: {module_name}\n\n")
        except Exception as e:
            # We PRINT the error so it shows up in your docker logs/terminal
            print(f"CRITICAL: Failed to import {module_name}. Error: {e}")

print("--- Registration Finished ---")