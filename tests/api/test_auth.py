"""Tests for the authentication endpoints."""

from fastapi.testclient import TestClient

from services.api.main import app

client = TestClient(app)

VALID_USERNAME = "admin"
VALID_PASSWORD = "changeme"  # matches the hash in .env for local testing


def _login(username: str, password: str):
    """Helper to call the login endpoint."""
    return client.post(
        "/auth/login",
        data={"username": username, "password": password},
    )


def test_login_success_returns_token():
    response = _login(VALID_USERNAME, VALID_PASSWORD)

    assert response.status_code == 200
    body = response.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert len(body["access_token"]) > 20


def test_login_wrong_password_returns_401():
    response = _login(VALID_USERNAME, "wrong-password")

    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"


def test_login_unknown_user_returns_401():
    response = _login("ghost", "any-password")

    assert response.status_code == 401


def test_me_without_token_returns_401():
    response = client.get("/auth/me")

    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


def test_me_with_invalid_token_returns_401():
    response = client.get(
        "/auth/me",
        headers={"Authorization": "Bearer this-is-not-a-valid-jwt"},
    )

    assert response.status_code == 401


def test_me_with_valid_token_returns_user():
    token = _login(VALID_USERNAME, VALID_PASSWORD).json()["access_token"]

    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["username"] == VALID_USERNAME
    assert body["disabled"] is False