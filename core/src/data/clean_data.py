import pandas as pd
import re
import unicodedata

from box import Box
from lxml import html
from pathlib import Path

from ..utils.loaders import load_params
from .utils import load, save


def remove_html(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Remove HTML components from columns of a pandas DataFrame.

    Args:
        df (pd.DataFrame): The pd.DataFrame to process.
        columns (list[str]): The columns from which the cleaning will apply.

    Returns:
        pd.DataFrame: A clean pd.DataFrame.
    """

    def apply_clean(text):
        if not isinstance(text, str):
            return "NULL"
        tree = html.fromstring(text)
        text = tree.text_content()
        return text

    for col in columns:
        df[col] = df[col].apply(lambda s: apply_clean(s))
    return df


def remove_patterns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Remove patterns from columns of a pandas DataFrame. Patterns include URLs and
    emails.

    Args:
        df (pd.DataFrame): The pd.DataFrame to process.
        columns (list[str]): The columns from which the cleaning will apply.

    Returns:
        pd.DataFrame: A clean pd.DataFrame.
    """

    patterns = {
        "URL": r'https?://\S+|www\.\S+',
        "MAIL": r'\b[\w\.-]+@[\w\.-]+\.\w+\b',
    }
    for col in columns:
        for name, pattern in patterns.items():
            df[col] = df[col].apply(lambda s: re.sub(pattern, name, s))
            df[col] = df[col].apply(lambda s: re.sub(r'\s+', ' ', s).strip())
    return df


def keep_characters(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Keep standardized characters from columns of a pandas DataFrame. It keeps letters,
    numbers, punctuation, separators based on the unicode norm.

    Args:
        df (pd.DataFrame): The pd.DataFrame to process.
        columns (list[str]): The columns from which the cleaning will apply.

    Returns:
        pd.DataFrame: A clean pd.DataFrame.
    """

    def apply_clean(text):
        cleaned = []
        for char in text:
            cat = unicodedata.category(char)
            if cat.startswith("L") \
            or cat.startswith("N") \
            or cat.startswith("P") \
            or cat == "Sc" \
            or char.isspace():
                cleaned.append(char)
        text = "".join(cleaned)
        text = re.sub(r"\s+", " ", text).strip()
        text = unicodedata.normalize("NFKC", text)
        return text

    for col in columns:
        df[col] = df[col].apply(lambda s: apply_clean(s))
    return df


def clean() -> None:
    """
    Main function to apply different cleaning processes on the raw data. It uses the
    parameters loaded from the `params.yaml` file at the root of the `core` directory.
    Applies HTML and pattern cleaning and characters filtering on the textual columns
    which are the designation and the description of the products.

    The cleaned data is automatically saved in the corresponding directory. The fields
    used for this step can be found under the `clean` key in the `params.yaml` file.

    Returns:
        None:
    """

    CORE_DIR = Path(__file__).parent.parent.parent
    params: Box = Box(load_params(str(CORE_DIR / "params.yaml"))).clean

    df: pd.DataFrame = load(CORE_DIR / params.input, format="csv")

    df = df[params.columns.keep]
    df = remove_html(df, params.columns.clean)
    df = remove_patterns(df, params.columns.clean)
    df = keep_characters(df, params.columns.clean)

    save(df, str(CORE_DIR / params.output))
    return None


if __name__ == "__main__":
    clean()
