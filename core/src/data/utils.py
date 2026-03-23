import pandas as pd

from typing import Literal


def load(filepath: str, format: Literal["csv", "parquet"]) -> pd.DataFrame:
    """
    Load a .csv or .parquet file into a pandas DataFrame.

    Args:
        filepath (str): The path to the file to load.
        format (Literal[&quot;csv&quot;, &quot;parquet&quot;]): Either csv or parquet.

    Returns:
        pd.DataFrame: The loaded file into a pd.DataFrame.
    """

    if format == "csv":
        return pd.read_csv(filepath)
    if format == "parquet":
        return pd.read_parquet(filepath)


def save(df: pd.DataFrame, filepath: str) -> None:
    """
    Save a pandas DataFrame into either a csv or parquet file.

    Args:
        df (pd.DataFrame): The pd.DataFrame to save.
        filepath (str): The file path for saving. Must end by either .csv or .parquet. 

    Returns:
        None:
    """

    if filepath.endswith(".csv"):
        df.to_csv(filepath, index=False)
    if filepath.endswith(".parquet"):
        df.to_parquet(filepath, index=False)
    return None
