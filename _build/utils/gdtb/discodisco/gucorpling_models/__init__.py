# These imports are important for making the configuration files find the classes that you wrote.
# If you don't have these, you'll get errors about allennlp not being able to find
# "simple_classifier", or whatever name you registered your model with.  These imports and the
# contents of .allennlp_plugins makes it so you can just use `allennlp train`, and we will find your
# classes and use them.  If you change the name of `gucorpling_models`, you'll also need to change it in
# the same way in the .allennlp_plugins file.
# from gucorpling_models.rel.baseline_model import *
# from gucorpling_models.rel.dataset_reader import *

# https://stackoverflow.com/questions/3365740/how-to-import-all-submodules
import importlib
import pkgutil


def import_submodules(package, recursive=True):
    """Import all submodules of a module, recursively, including subpackages

    :param package: package (name or actual module)
    :type package: str | module
    :rtype: dict[str, types.ModuleType]
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + "." + name
        results[full_name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results


import_submodules(__name__)
