import math
from typing import Tuple, Dict, List, Any, Union

import torch
from allennlp.common import FromParams, Registrable
import scipy.stats as stats


from allennlp.data import Vocabulary, Field
from allennlp.data.fields import TextField, TensorField, SequenceLabelField, LabelField


class TransformationFunction(Registrable):
    def __call__(self, xs, tokenwise):
        raise NotImplementedError("Please use a class that inherits from TransformationFunction")


@TransformationFunction.register("natural_log")
class NaturalLog(TransformationFunction):
    def __call__(self, xs, tokenwise):
        if tokenwise:
            return [math.log(x) if x != 0 else x for x in xs]
        else:
            return math.log(xs)


@TransformationFunction.register("abs_natural_log")
class AbsNaturalLog(TransformationFunction):
    def __call__(self, xs, tokenwise):
        if tokenwise:
            return [math.log(abs(x)) if x != 0 else x for x in xs]
        else:
            return math.log(abs(xs))


@TransformationFunction.register("bins")
class Bins(TransformationFunction):
    def __init__(self, bins: List[Tuple[Union[int, float], Union[int, float]]]):
        # bins is a list of ints such that any value falls into it iff `bin[0] <= x < bin[1]`
        self.bins = bins

    def _bin(self, x):
        for i, (start, end) in enumerate(self.bins):
            if start <= x < end:
                return str(i)
        raise Exception(f"No bin found for value {x}! Bins: {self.bins}")

    def __call__(self, xs, tokenwise):
        if tokenwise:
            return [self._bin(x) for x in xs]
        else:
            return self._bin(xs)


class Feature(FromParams):
    def __init__(self, source_key: str, label_namespace: str = None, xform_fn: TransformationFunction = None):
        self.source_key = source_key
        self.label_namespace = label_namespace
        self.xform_fn = xform_fn


class FeatureBundle(FromParams):
    def __init__(self, features: Dict[str, Feature], corpus: str, corpus_configs: Dict[str, List[str]]):
        self.features = features
        self.corpus = corpus
        # Use a corpus-specific subset of features if provided, else the default
        self.corpus_keys = corpus_configs[corpus] if corpus in corpus_configs else list(features.keys())
        print("Using keys " + str(self.corpus_keys) + " for " + corpus)


def get_feature_field(feature_config: Feature, features: Union[List[Any], Any], sentence: TextField = None) -> Field:
    """
    Returns an AllenNLP `Field` suitable for use on an AllenNLP `Instance` for a given token-level feature.
    If the type of the data in `features` is int or float, we will use TensorField; if it is str, we will
    use SequenceLabelField; other data types are currently unsupported.

    Args:
        feature_config: a Feature for the Field
        features: either a list of data (token-wise features) or a single piece of data
        sentence: the TextField the Field is associated with. If present, some fields will be associated with it.

    Returns:
        a Field for the feature.
    """
    tokenwise = (isinstance(features, list) or isinstance(features, tuple))
    if tokenwise and not (len(features) == len(sentence.tokens)):
        raise Exception(f"Token-level features must match the number of tokens")

    if feature_config.xform_fn is not None:
        features = feature_config.xform_fn(features, tokenwise=tokenwise)

    py_type = type(features[0]) if tokenwise else type(features)
    if py_type in [int, float]:
        return TensorField(torch.tensor(features))
    elif py_type == str and tokenwise:
        return SequenceLabelField(features, sentence, label_namespace=feature_config.label_namespace or "labels")
    elif py_type == str:
        return LabelField(features, label_namespace=feature_config.label_namespace or "labels")
    elif py_type == bool and not tokenwise:
        return LabelField("true" if features else "false", label_namespace=feature_config.label_namespace or "labels")
    else:
        raise Exception(f"Unsupported type for feature: {py_type}")


def get_feature_modules(
    features: Union[Dict[str, Feature], FeatureBundle], vocab: Vocabulary
) -> Tuple[torch.nn.ModuleDict, int]:
    """
    Returns a PyTorch `ModuleDict` containing a module for each feature in `token_features`.
    This function tries to be smart: if the feature is numeric, it will not do anything, but
    if it is categorical (as indicated by the presence of a `label_namespace`), then the module
    will be a `torch.nn.Embedding` with size equal to the ceiling of the square root of the
    categorical feature's vocabulary size. We could be a lot smarter of course, but this will
    get us going.

    Args:
        features: a dict of `TokenFeatures` describing all the categorical features to be used
        vocab: the initialized vocabulary for the model

    Returns:
        A 2-tuple: the ModuleDict, and the summed output dimensions of every module, for convenience.
    """
    modules: Dict[str, torch.nn.Module] = {}
    total_dims = 0
    if isinstance(features, FeatureBundle):
        keys = features.corpus_keys
        config_dict = features.features
    else:
        keys = features.keys()
        config_dict = features
    for key in keys:
        config = config_dict[key]
        ns = config.label_namespace
        if ns is None:
            modules[key] = torch.nn.Identity()
            total_dims += 1
        else:
            size = vocab.get_vocab_size(ns)
            # if size <= 5:
            #    modules[key] = torch.nn.Identity()
            #    total_dims += size
            # else:
            edims = math.ceil(math.sqrt(size))
            total_dims += edims
            modules[key] = torch.nn.Embedding(size, edims, padding_idx=(0 if vocab.is_padded(ns) else None))

    return torch.nn.ModuleDict(modules), total_dims


def get_combined_feature_tensor(self, kwargs):
    output_tensors = []
    for module_key, module in self.feature_modules.items():
        output_tensor = module(kwargs[module_key])
        if len(output_tensor.shape) == 1:
            output_tensor = output_tensor.unsqueeze(-1)
        output_tensors.append(output_tensor)

    combined_feature_tensor = torch.cat(output_tensors, dim=1)
    return combined_feature_tensor
