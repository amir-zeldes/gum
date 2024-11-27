from typing import Dict, Tuple

from allennlp.common.checks import ConfigurationError


def detect_encoding(index_to_label: Dict[int, str]) -> Tuple[str, Dict[str, int]]:
    """
    Given a label vocabulary used to encode spans, infer its encoding.
    Returns the encoding along with a dict mapping from units in the encoding
    to indexes in the label space.
    """
    label_to_index = {v: k for k, v in index_to_label.items()}
    description = {}

    # detect encoding
    if 2 <= len(index_to_label) <= 3:
        begin_token = None
        in_token = None
        out_token = None
        for index, label in index_to_label.items():
            if label.startswith("B"):
                if begin_token is not None:
                    raise ConfigurationError(
                        f"Already saw a BEGIN token, {begin_token}, but got a new begin token {label}. "
                        f"Rename your tokens so that only one of them begins with B."
                    )
                begin_token = label
            elif label.startswith("O"):
                if out_token is not None:
                    raise ConfigurationError(
                        f"Already saw an OUT token, {out_token}, but got a new out token {label}. "
                        f"Rename your tokens so that only one of them begins with O."
                    )
                out_token = label
            elif label.startswith("I"):
                if in_token is not None:
                    raise ConfigurationError(
                        f"Already saw an IN token, {in_token}, but got a new out token {label}. "
                        f"Rename your tokens so that only one of them begins with I."
                    )
                in_token = label
            else:
                raise ConfigurationError(
                    "Received a label vocabulary with size 2 or size 3. This makes me think it's a BIO encoding. But "
                    f"one of the labels doesn't look like a BIO tag: {index}\n"
                    "Please make sure your BIO tags start with B, I, or O appropriately."
                )
        if begin_token is None:
            raise ConfigurationError("BIO encoding must have a BEGIN token beginning with B.")
        if out_token is None:
            raise ConfigurationError("BIO encoding must have an OUT token beginning with B.")
        description["B"] = label_to_index[begin_token]
        description["O"] = label_to_index[out_token]
        if in_token is not None:
            description["I"] = label_to_index[in_token]
    else:
        raise ConfigurationError("Only BIO encoding is currently supported.")

    return "BIO", description
