"""
Front-end configuration, loaded from environment variables.

Every variable is prefixed with ``UI_`` so it never collides with the API
settings (prefixed ``API_``) when both services share the same ``.env`` file.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: services/streamlit/settings.py -> services/streamlit -> services -> root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class UISettings(BaseSettings):
    """Settings for the Streamlit interface."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="UI_",
        extra="ignore",
    )

    # -- Upstream services -----------------------------------------------------
    api_url: str = "http://localhost:8000"
    mlflow_url: str = "http://localhost:5000"
    airflow_url: str = "http://localhost:8080"

    # -- Credentials used to pre-fill the login forms --------------------------
    # Convenience for local demos only; nothing is stored or sent automatically.
    default_username: str = "admin"
    default_password: str = "changeme"
    airflow_username: str = "admin"
    airflow_password: str = "admin"

    # -- Behaviour -------------------------------------------------------------
    request_timeout: float = 30.0
    artifacts_dir: str = str(PROJECT_ROOT / "core" / "artifacts")
    assets_dir: str = str(Path(__file__).resolve().parent / "assets")
    mlruns_dir: str = str(PROJECT_ROOT / "tracking" / "mlflow" / "mlruns")
    params_file: str = str(PROJECT_ROOT / "core" / "params.yaml")
    features_metadata_file: str = str(
        PROJECT_ROOT / "core" / "data" / "features" / "metadata.json"
    )

    @property
    def artifacts_path(self) -> Path:
        """Return the artifacts directory as a Path."""
        return Path(self.artifacts_dir)

    @property
    def assets_path(self) -> Path:
        """Return the front-end reference assets directory as a Path."""
        return Path(self.assets_dir)

    @property
    def mlruns_path(self) -> Path:
        """Return the local MLflow file store as a Path."""
        return Path(self.mlruns_dir)

    @property
    def params_path(self) -> Path:
        """Return the pipeline parameter file as a Path."""
        return Path(self.params_file)

    @property
    def features_metadata_path(self) -> Path:
        """Return the feature-pipeline metadata file as a Path."""
        return Path(self.features_metadata_file)


@lru_cache
def get_ui_settings() -> UISettings:
    """
    Return the cached settings singleton.

    Returns:
        UISettings: Validated configuration for the front-end.
    """
    return UISettings()
