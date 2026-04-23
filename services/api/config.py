from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="API_")

    jwt_secret: str = "dev-secret-change-me" #openssl rand -hex 32
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60


settings = Settings()