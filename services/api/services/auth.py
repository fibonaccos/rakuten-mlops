from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from services.api.config import settings
from services.api.schemas.auth import TokenPayload, User, UserInDB

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    """Hash a plain password with bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check a plain password against its bcrypt hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


# Temporary in-memory user store. Replace with a DB query later.
_FAKE_USERS_DB: dict[str, UserInDB] = {
    settings.admin_username: UserInDB(
        username=settings.admin_username,
        hashed_password=settings.admin_password_hash,
        disabled=False,
    ),
}


def get_user(username: str) -> UserInDB | None:
    """Fetch a user by username from the user store."""
    return _FAKE_USERS_DB.get(username)


def authenticate_user(username: str, password: str) -> UserInDB | None:
    """Return the user if credentials are valid, otherwise None."""
    user = get_user(username)
    if user is None or not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(subject: str) -> str:
    """Create a signed JWT access token for the given subject (username)."""
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """FastAPI dependency resolving the authenticated user from the Bearer token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        raw_payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        payload = TokenPayload(**raw_payload)
    except JWTError:
        raise credentials_exception

    if payload.sub is None:
        raise credentials_exception

    user = get_user(payload.sub)
    if user is None or user.disabled:
        raise credentials_exception

    return User(username=user.username, disabled=user.disabled)