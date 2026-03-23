import json
import joblib
import numpy as np
import pandas as pd
import unicodedata

from box import Box
from pathlib import Path
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
)
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Any

from ..utils.loaders import load_params
from .utils import load, save


def split(
        filepath: str,
        n_use: int,
        target: str,
        train_size: float,
        random_state: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into ML usable datasets for training and testing.

    Args:
        filepath (str): The path to the original data.
        n_use (int): The number of samples to use. If negative, uses all the samples.
        target (str): The name of the target column.
        train_size (float): The proportion of samples to use for training.
        random_state (int): The seed for randomness control.

    Returns:
        tuple: A tuple of pd.DataFrame ordered as *(X_train, X_test, y_train, y_test)*.
    """

    df: pd.DataFrame = load(filepath, format="parquet")

    n_sample = min(n_use, df.shape[0]) if n_use > 0 else df.shape[0]
    X = df.drop(columns=[target]).iloc[:n_sample]
    y = df[[target]].iloc[:n_sample]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y[target], stratify=y[target],
        train_size=train_size, random_state=random_state
    )

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    y_train = pd.DataFrame(y_train, columns=y.columns)
    y_test = pd.DataFrame(y_test, columns=y.columns)

    return X_train, X_test, y_train, y_test


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Utility function to chunk texts before encoding as some of the data may be larger
    than the maximum supported tokens of the embedding model.

    Args:
        text (str): The text to chunk.
        chunk_size (int): The size (number of words) of each chunks.
        overlap (int): The number of words overlaping between each chunk.

    Returns:
        list[str]: The chunks of the given text.
    """

    if not isinstance(text, str):
        return [""]
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks: list[str] = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks


def _embed(
        model: SentenceTransformer,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> np.ndarray:
    """
    Utility function to embed a chunked text using a SentenceTransformer model.
    Whenever the text is splitted into multiple chunks, the final embedding is computed
    by averaging the embedding of each chunk.

    Args:
        model (SentenceTransformer): The embedding model.
        text (str): The text to encode.
        chunk_size (int): The size of each chunk. Must fit with the maximum supported
            tokens of the model.
        overlap (int): The number of words overlaping between each chunk.

    Returns:
        np.ndarray: The embedding of the text.
    """

    chunks = _chunk_text(text, chunk_size, overlap)
    chunk_embeddings = model.encode(chunks)
    document_embedding = np.mean(chunk_embeddings, axis=0)
    return document_embedding


def make_embeddings(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        columns: list[str],
        product_id: str,
        chunk_size: int,
        overlap: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create embeddings of the train and test dataset. As the embedding model is **not**
    fitted on the data, the processes applied on the train and the test set are equal.
    The embeddings are made on the given columns and keep the product id to ensure
    tracability.

    Args:
        X_train (pd.DataFrame): The train DataFrame.
        X_test (pd.DataFrame): The test DataFrame.
        columns (list[str]): The columns to embed.
        product_id (str): The product id column. 
        chunk_size (int): The chunk size (see `_chunk_text` for more details).
        overlap (int): The overlap between chunks (see `_chunk_text` for more
            details).

    Returns:
        tuple: A tuple of DataFrame containing product ids and the embeddings, ordered
            as *(X_train, X_test)*.
    """

    X_train["combined"] = ""
    X_test["combined"] = ""
    for col in columns:
        X_train["combined"] += " " + X_train[col]
        X_test["combined"] += " " + X_test[col]

    X_train["combined"] = X_train["combined"]
    X_test["combined"] = X_test["combined"]

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        TextColumn("[progress.percentage]{task.percentage:.2f}%"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Embedding train", total=X_train.shape[0])
        embedding_list = []
        for text in X_train["combined"]:
            embedding_list.append(_embed(model, text, chunk_size, overlap))
            progress.update(task, advance=1)

    embeddings_train = np.array(embedding_list)
    embed_train = pd.DataFrame(embeddings_train)
    embed_train.columns = [f"emb_{i+1}" for i in range(embeddings_train.shape[1])]
    out_train = pd.concat([
        X_train[[product_id]].reset_index(drop=True),
        embed_train.reset_index(drop=True)
    ], axis=1)

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        TextColumn("[progress.percentage]{task.percentage:.2f}%"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Embedding test", total=X_test.shape[0])
        embedding_list = []
        for text in X_test["combined"]:
            embedding_list.append(_embed(model, text, chunk_size, overlap))
            progress.update(task, advance=1)

    embeddings_test = np.array(embedding_list)
    embed_test = pd.DataFrame(embeddings_test)
    embed_test.columns = [f"emb_{i+1}" for i in range(embeddings_test.shape[1])]
    out_test = pd.concat([
        X_test[[product_id]].reset_index(drop=True),
        embed_test.reset_index(drop=True)
    ], axis=1)

    return out_train, out_test


def create_features(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        columns: list[str],
        product_id: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build statistical features from the textual columns. Covered statistics :
    - Number of characters.
    - Number of words.
    - Average length of a word.
    - Maximum length of a word.
    - Number of digits.
    - Number of punctuations.

    Args:
        X_train (pd.DataFrame): The train DataFrame.
        X_test (pd.DataFrame): The test DataFrame.
        columns (list[str]): The textual columns to process.
        product_id (str): The product id column.

    Returns:
        tuple: A tuple of DataFrame containing product ids and the computed statistics,
            ordered as *(X_train, X_test)*.
    """

    def mean_word_length(text: str) -> float:
        words = [len(w) for w in text.split()]
        if words and len(words) > 0:
            return float(np.mean(words))
        else:
            return 0.

    def max_word_length(text: str) -> int:
        words = [len(w) for w in text.split()]
        if words and len(words) > 0:
            return max(words)
        else:
            return 0

    def num_digits(text: str) -> int:
        return sum([unicodedata.category(c).startswith("N") for c in text])

    def num_punctuation(text: str) -> int:
        return sum([unicodedata.category(c).startswith("P") for c in text])

    to_keep = {"train": [product_id], "test": [product_id]}
    for name, df in zip(("train", "test"), (X_train, X_test)):
        for col in columns:
            df[f"length_{col}"] = df[col].apply(lambda s: len(s))
            df[f"num_words_{col}"] = df[col].apply(lambda s: len(s.split()))
            df[f"mean_word_length_{col}"] = df[col].apply(
                lambda s: mean_word_length(s)
            )
            df[f"max_word_length_{col}"] = df[col].apply(lambda s: max_word_length(s))
            df[f"num_digits_{col}"] = df[col].apply(lambda s: num_digits(s))
            df[f"num_punctuation_{col}"] = df[col].apply(
                lambda s: num_punctuation(s)
            )
            to_keep[name] += [
                f"length_{col}",
                f"num_words_{col}",
                f"mean_word_length_{col}",
                f"max_word_length_{col}",
                f"num_digits_{col}",
                f"num_punctuation_{col}"
            ]

    return X_train[to_keep["train"]], X_test[to_keep["test"]]


def scaler(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame, 
        product_id: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Utility that wraps the sklearn StandardScaler to apply it on the train and test
    dataset, keeping the product id for tracability.

    Args:
        X_train (pd.DataFrame): The train DataFrame.
        X_test (pd.DataFrame): The test DataFrame.
        product_id (str): The product id column.

    Returns:
        tuple: A tuple containing the scaled DataFrame and the fitted scaler, ordered
            as *(X_train, X_test, scaler)*.
    """

    sc = StandardScaler()

    train_scaled = sc.fit_transform(X_train.drop(columns=[product_id]))
    test_scaled = sc.transform(X_test.drop(columns=[product_id]))

    X_train_scaled = pd.concat([
        X_train[[product_id]].reset_index(drop=True),
        pd.DataFrame(train_scaled).reset_index(drop=True)
    ], axis=1)
    X_test_scaled = pd.concat([
        X_test[[product_id]].reset_index(drop=True),
        pd.DataFrame(test_scaled).reset_index(drop=True)
    ], axis=1)

    X_train_scaled.columns = X_train.columns
    X_test_scaled.columns = X_test.columns

    return X_train_scaled, X_test_scaled, sc


def reducer(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        product_id: str,
        n_components: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """
    Utility that wraps the sklearn PCA to apply it on the train and test
    dataset, keeping the product id for tracability.

    Args:
        X_train (pd.DataFrame): The train DataFrame.
        X_test (pd.DataFrame): The test DataFrame.
        product_id (str): The product id column.
        n_components (int): The number of components of the PCA.

    Returns:
        tuple: A tuple containing the scaled DataFrame and the fitted PCA, ordered
            as *(X_train, X_test, pca)*.
    """

    pca = PCA(n_components=n_components)

    train_reduced = pca.fit_transform(X_train.drop(columns=[product_id]))
    test_reduced = pca.transform(X_test.drop(columns=[product_id]))

    X_train_reduced = pd.concat([
        X_train[[product_id]].reset_index(drop=True),
        pd.DataFrame(train_reduced).reset_index(drop=True)
    ], axis=1)
    X_test_reduced = pd.concat([
        X_test[[product_id]].reset_index(drop=True),
        pd.DataFrame(test_reduced).reset_index(drop=True)
    ], axis=1)

    X_train_reduced.columns = [product_id] \
        + [f"feat_{i+1}" for i in range(n_components)]
    X_test_reduced.columns = [product_id] \
        + [f"feat_{i+1}" for i in range(n_components)]

    return X_train_reduced, X_test_reduced, pca


def build() -> None:
    """
    Main function to prepare data for ML/DL models. It applies embeddings, statistical
    computations, scaling, and dimension reduction on the data. It follows the
    `params.yaml` file in the rot of the `core` directory.

    Stateful artefacts and some metadata are saved in a dedicated folder.

    The fields used for building features can be found under the `features` key of the
    `params.yaml` file.

    Returns:
        None:
    """

    CORE_DIR = Path(__file__).parent.parent.parent
    params: Box = Box(load_params(str(CORE_DIR / "params.yaml"))).features

    X_train, X_test, y_train, y_test = split(
        str(CORE_DIR / params.input),
        params.split.n_use,
        params.split.target,
        params.split.train_size,
        params.split.random_state
    )

    embed_train, embed_test = make_embeddings(
        X_train, X_test,
        params.embedding.columns,
        params.product_id,
        params.embedding.chunk_size,
        params.embedding.overlap
    )

    created_train, created_test = create_features(
        X_train, X_test, params.statistic.columns, params.product_id
    )

    X_train_grouped = created_train.merge(
        right=embed_train, how="left", on=params.product_id
    )

    X_test_grouped = created_test.merge(
        right=embed_test, how="left", on=params.product_id
    )

    X_train_scaled, X_test_scaled, sc = scaler(
        X_train_grouped, X_test_grouped, params.product_id
    )

    X_train_reduced, X_test_reduced, pca = reducer(
        X_train_scaled, X_test_scaled, params.product_id, params.pca.n_components
    )

    save(X_train_reduced, str(CORE_DIR / params.output / "x_train.parquet"))
    save(X_test_reduced, str(CORE_DIR / params.output / "x_test.parquet"))

    save(y_train, str(CORE_DIR / params.output / "y_train.parquet"))
    save(y_test, str(CORE_DIR / params.output / "y_test.parquet"))

    with open(CORE_DIR / params.scale.artifact, "wb") as f:
        joblib.dump(sc, f)

    with open(CORE_DIR / params.pca.artifact, "wb") as f:
        joblib.dump(pca, f)

    metadata: dict[str, Any] = {}
    metadata["embedder"] = {
        "path": "",
        "model": "paraphrase-multilingual-MiniLM-L12-v2"
    }
    metadata["scaler"] = {
        "path": str(CORE_DIR / params.scale.artifact)
    }
    metadata["pca"] = {
        "path": str(CORE_DIR / params.pca.artifact),
        "params": {
            "n_components": pca.components_.shape[0],
            "explained_variance": pca.explained_variance_ratio_.sum()
        }
    }
    with open(CORE_DIR / params.output, "w") as f:
        json.dump(metadata, f, indent=2)

    return None


if __name__ == "__main__":
    build()
