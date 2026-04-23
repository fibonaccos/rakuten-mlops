from fastapi import FastAPI

from services.api.routes import auth

app = FastAPI(title="Rakuten Inference API")

app.include_router(auth.router)


@app.get("/health")
def health() -> dict[str, str]:
    """Simple healthcheck."""
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Rakuten API is running"}