from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="API_")

    # JWT settings
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Default admin user (bootstrapped at startup)
    admin_username: str = "admin"
    admin_password_hash: str


settings = Settings()