# API service

FastAPI service for Rakuten product classification inference and authentication.

## Run locally

From the repo root:

```bash
uv sync --group api
cp .env.example .env
# Edit .env to fill in secrets (see below)
uv run uvicorn services.api.main:app --reload --port 8000
```

Open http://localhost:8000/docs for the Swagger UI.

## Environment variables

See `.env.example` at the repo root.

| Variable                           | Required | Description                        |
| ---------------------------------- | -------- | ---------------------------------- |
| `API_JWT_SECRET`                   | yes      | Secret key for signing JWT tokens  |
| `API_ADMIN_PASSWORD_HASH`          | yes      | Bcrypt hash of the admin password  |
| `API_ADMIN_USERNAME`               | no       | Admin username (default: `admin`)  |
| `API_ACCESS_TOKEN_EXPIRE_MINUTES`  | no       | Token lifetime (default: `60`)     |

Generate a JWT secret:

```bash
openssl rand -hex 32
```

Generate a bcrypt password hash:

```bash
uv run python -c "import bcrypt; print(bcrypt.hashpw(b'your-password', bcrypt.gensalt()).decode())"
```

## Authentication

The API uses OAuth2 Password Flow with JWT bearer tokens.

```bash
# Get a token
curl -X POST http://localhost:8000/auth/login \
  -d "username=admin&password=<your-password>"

# Use the token
curl http://localhost:8000/auth/me \
  -H "Authorization: Bearer <token>"
```

To protect an endpoint, add the `get_current_user` dependency:

```python
from fastapi import Depends
from services.api.schemas.auth import User
from services.api.services.auth import get_current_user

@router.post("/predict")
def predict(..., current_user: User = Depends(get_current_user)):
    ...
```