from pydantic import BaseModel, Field


class Token(BaseModel):
    """Schema for the access token returned after a successful login."""

    access_token: str = Field(..., description="JWT access token.")
    token_type: str = Field(default="bearer", description="Type of the token.")


class TokenPayload(BaseModel):
    """Schema for the decoded JWT payload."""

    sub: str | None = Field(None, description="Subject of the token (username).")
    exp: int | None = Field(None, description="Expiration timestamp (epoch seconds).")


class BaseUser(BaseModel):
    """Base user fields exposed publicly."""

    username: str = Field(..., description="Unique username of the user.")
    disabled: bool = Field(default=False, description="Whether the user is disabled.")


class User(BaseUser):
    """Schema for a user returned to clients (no credentials)."""


class UserInDB(BaseUser):
    """Schema for a user as stored internally, including the hashed password."""

    hashed_password: str = Field(..., description="Bcrypt hash of the user password.")