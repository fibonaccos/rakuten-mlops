"""
API configuration loaded from environment variables.

Settings are validated at startup via pydantic-settings. All variables are
prefixed with ``API_`` in the environment (e.g. ``API_MODEL_PATH``).
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="API_",
        extra="ignore",
    )

    # -- Application metadata --------------------------------------------------
    app_name: str = "Rakuten Predict API"
    app_version: str = "0.1.0"
    app_description: str = "Product category classification for Rakuten catalogue."

    # -- JWT settings ----------------------------------------------------------
    jwt_secret: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # -- Admin user bootstrapped at startup ------------------------------------
    # Default hash is bcrypt("changeme") — override in production via API_ADMIN_PASSWORD_HASH.
    admin_username: str = "admin"
    admin_password_hash: str = "$2b$12$.SFcvNZnowY3gVm76LEhD.g9exvyLWMBhiHJAIyvcBWMnjhIL51Qy"

    # -- ML artifact paths -----------------------------------------------------
    model_path: str = "core/artifacts/model.keras"
    scaler_path: str = "core/models/artifacts/scaler.joblib"
    pca_path: str = "core/models/artifacts/pca.joblib"
    labels_map_path: str = "core/artifacts/labels_map.json"

    # -- Embedding model -------------------------------------------------------
    embedder_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # -- Feature pipeline hyper-parameters (must match training) ---------------
    embed_chunk_size: int = 150
    embed_overlap: int = 30

    # -- Stub mode (dev / CI without ML artifacts) ----------------------------
    stub_mode: bool = False

    # -- Training job configuration -------------------------------------------
    train_command: str = "uv run python -m core.src.models.train"
    metrics_path: str = "core/artifacts/metrics.json"
    train_max_jobs_history: int = 50


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


# Module-level instance for direct imports (e.g. in services/auth.py).
settings = get_settings()
