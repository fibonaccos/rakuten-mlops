"""
API configuration loaded from environment variables.

Settings are validated at startup via pydantic-settings. All artifact paths can be
overridden through environment variables or a .env file, making the service portable
across local development, CI, and Docker environments.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings sourced from environment variables.

    All path values are relative to the working directory unless overridden.
    In Docker, mount the core/artifacts volume and set the *_PATH variables
    in the env_file accordingly.
    """

    # -- Application metadata --------------------------------------------------
    APP_NAME: str = "Rakuten Predict API"
    APP_VERSION: str = "0.1.0"
    APP_DESCRIPTION: str = "Product category classification for Rakuten catalogue."

    # -- ML artifact paths -----------------------------------------------------
    MODEL_PATH: str = "core/artifacts/model.keras"
    SCALER_PATH: str = "core/models/artifacts/scaler.joblib"
    PCA_PATH: str = "core/models/artifacts/pca.joblib"
    LABELS_MAP_PATH: str = "core/artifacts/labels_map.json"

    # -- Embedding model -------------------------------------------------------
    EMBEDDER_MODEL_NAME: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # -- Feature pipeline hyper-parameters (must match training) ---------------
    EMBED_CHUNK_SIZE: int = 150
    EMBED_OVERLAP: int = 30

    # -- Stub mode (dev / CI without ML artifacts) ----------------------------
    # Set STUB_MODE=true in .env to boot the server without any real artifact.
    # The stub returns random plausible predictions drawn from the known label set.
    STUB_MODE: bool = False

    # -- Training job configuration -------------------------------------------
    # Command executed by TrainingService to run the real training pipeline.
    # The command runs from the project root directory where pyproject.toml lives.
    TRAIN_COMMAND: str = "uv run python -m core.src.models.train"

    # Path to the metrics JSON file written by the training pipeline.
    # Read after a successful training run to populate job.metrics.
    METRICS_PATH: str = "core/artifacts/metrics.json"

    # Maximum number of completed/failed jobs kept in the in-memory history.
    TRAIN_MAX_JOBS_HISTORY: int = 50

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def model_path_resolved(self) -> Path:
        """Return MODEL_PATH as an absolute Path."""
        return Path(self.MODEL_PATH).resolve()

    def scaler_path_resolved(self) -> Path:
        """Return SCALER_PATH as an absolute Path."""
        return Path(self.SCALER_PATH).resolve()

    def pca_path_resolved(self) -> Path:
        """Return PCA_PATH as an absolute Path."""
        return Path(self.PCA_PATH).resolve()

    def labels_map_path_resolved(self) -> Path:
        """Return LABELS_MAP_PATH as an absolute Path."""
        return Path(self.LABELS_MAP_PATH).resolve()


@lru_cache
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    Using lru_cache ensures the .env file is read exactly once at startup.
    To reset the cache in tests, call get_settings.cache_clear().

    Returns:
        Settings: The validated application settings.
    """
    return Settings()
