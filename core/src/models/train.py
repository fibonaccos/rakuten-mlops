import json
import os
import random
from collections import Counter
from pathlib import Path

import keras as ks
import numpy as np
import pandas as pd
from box import Box
from keras.callbacks import ModelCheckpoint
from keras.metrics import AUC

from ..utils.loaders import load_params  # type: ignore[misc]


def set_seed(seed: int) -> None:
    """
    Fix all random seeds for reproducibility across python, numpy et keras/TF

    Args:
        seed (int): the seed value to use everywhere
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    ks.utils.set_random_seed(seed)


def create_model(n_features: int, n_classes: int) -> ks.models.Model:
    """
    Create a Keras Model instance with input shape `n_features` and output shape
    `n_classes`.

    Args:
        n_features (int): The input dimension of the neural network.
        n_classes (int): The output dimension of the neural network.

    Returns:
        ks.models.Model: A neural network with the corresponding input and output
            shape.
    """

    inputs = ks.layers.Input(shape=(n_features,), name="input")

    x = ks.layers.Dense(units=256, activation="relu", name="dense1")(inputs)
    x = ks.layers.Dropout(rate=0.4, name="drop1")(x)

    output = ks.layers.Dense(units=n_classes, activation="softmax", name="output")(x)

    model = ks.models.Model(inputs=inputs, outputs=output)
    return model


def compute_class_weight(labels: pd.DataFrame) -> dict[str, float]:
    """
    Compute weights for each classes to rebalance the importance of loss on each class.
    Weights are computed on the train set. The less important a class is in the
    dataset, the higher its weight is, resulting to a bigger loss fore less
    representative classes.

    Args:
        labels (pd.DataFrame): The labels of the train set.

    Returns:
        dict[str, float]: A dict mapping each label to its corresponding weight.
    """

    counts = Counter(labels.to_numpy()[:, 0])
    N = labels.shape[0]
    K = labels.nunique().values[0]
    weights = {str(cls): float(N / (K * count)) for cls, count in counts.items()}
    return weights


def train_model() -> None:
    """
    Main function to train the model built using `create_model`. It uses the parameters
    loaded from the `params.yaml` file at the root of the `core` directory.

    Stateful artefacts and some metadata are saved in a dedicated folder.

    The fields used for training the model can be found under the `train` key of the
    `params.yaml` file.
    """

    CORE_DIR: Path = Path(__file__).parent.parent.parent
    params: Box = Box(load_params(str(CORE_DIR / "params.yaml"))).train

    seed: int = params.get("seed", 42)
    set_seed(seed)

    X_train = pd.read_parquet(CORE_DIR / params.input.x_train)
    X_test = pd.read_parquet(CORE_DIR / params.input.x_test)
    y_train = pd.read_parquet(CORE_DIR / params.input.y_train)
    y_test = pd.read_parquet(CORE_DIR / params.input.y_test)[params.target]

    X = X_train.drop(columns=["productid"]).to_numpy()
    Xt = X_test.drop(columns=["productid"]).to_numpy()
    y_base = y_train[params.target].to_numpy()

    class_weights = compute_class_weight(y_train)

    codes_uniques = [int(c) for c in np.unique(y_base)]
    code_to_index = {code: i for i, code in enumerate(codes_uniques)}
    y = np.array([code_to_index[code] for code in y_base])
    yt = np.array([code_to_index[code] for code in y_test.to_numpy()])

    y_onehot = ks.utils.to_categorical(y)
    yt_onehot = ks.utils.to_categorical(yt)

    model = create_model(n_features=X.shape[1], n_classes=params.n_classes)

    checkpoint_cb = ModelCheckpoint(
        CORE_DIR / params.output.model,
        monitor="val_auc",
        save_best_only=True,
        mode="max",
        verbose=0,
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[AUC(multi_label=True, name="auc"), "accuracy"],
    )

    history = model.fit(
        X,
        y_onehot,
        epochs=20,
        batch_size=128,
        class_weight=class_weights,
        validation_data=(Xt, yt_onehot),
        callbacks=[checkpoint_cb],
    )

    model.save(CORE_DIR / params.output.model)
    """ This function creates an image with the model graph, but requires graphviz
        to be installed (out of the project dependencies). Uncomment it if you have
        graphviz installed.

    plot_model(
        model,
        to_file=CORE_DIR / params.output.architecture,
        show_shapes=True,
        show_layer_names=True
    )
    """

    with open(CORE_DIR / params.output.history, "w") as f:
        json.dump(history.history, f, indent=2)

    with open(CORE_DIR / params.output.labels_map, "w") as f:
        json.dump(code_to_index, f, indent=2)

    with open(CORE_DIR / params.output.class_weights, "w") as f:
        json.dump(class_weights, f, indent=2)


if __name__ == "__main__":
    train_model()
