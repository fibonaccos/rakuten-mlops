"""
Pytest configuration and shared fixtures for the API test suite.

Fixtures defined here are automatically available to all tests under tests/api/
without explicit imports (pytest discovers conftest.py automatically).
"""

import pytest
from fastapi.testclient import TestClient

from services.api.main import app


@pytest.fixture(scope="session")
def test_client() -> TestClient:
    """
    Session-scoped synchronous test client.

    Uses scope="session" so the client (and FastAPI app) is created once for
    the entire test session. Individual test modules that need dependency
    overrides use their own module-scoped fixtures.
    """
    return TestClient(app, raise_server_exceptions=False)
