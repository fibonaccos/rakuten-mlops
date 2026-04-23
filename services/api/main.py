from fastapi import FastAPI

app = FastAPI(title="Rakuten Inference API")


@app.get("/health")
def health() -> dict[str, str]:
    """Simple healthcheck."""
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Rakuten API is running"}