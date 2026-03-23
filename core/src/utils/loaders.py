import keras as ks

from typing import Any
from yaml import safe_load


def load_params(filepath: str) -> dict[str, Any]:
    """
    Load a yaml file used to configure data cleaning, data processing, model training
    and model evaluation. It is recommended to have this file named "params.yaml" and
    put in the root of the `core` directory.

    Args:
        filepath (str): Path to the yaml file.

    Returns:
        dict[str, Any]: A dict reprensenting the yaml file.
    """

    with open(filepath, "r") as f:
        params: dict[str, Any] = safe_load(f)
    return params


def load_model(filepath: str) -> Any:
    """
    Load a Keras model.

    Args:
        filepath (str): The path to the .keras model.

    Returns:
        Model: The loaded Keras model.
    """

    return ks.models.load_model(filepath)
