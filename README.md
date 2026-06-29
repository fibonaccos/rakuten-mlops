## Docker services

The local MLOps stack is defined in `docker-compose.yml`.

Services:

- `api`: FastAPI service for authentication, prediction and training routes.
- `mlflow`: MLflow tracking server on port `5000`.
- `airflow-webserver`: Airflow UI on port `8080`.
- `airflow-scheduler`: Airflow scheduler.
- `airflow-postgres`: Airflow metadata database.

Start the stack:

```bash
cp .env.example .env
docker compose up -d --build
```

Useful URLs:

- API health: http://localhost:8000/health
- API docs: http://localhost:8000/docs
- MLflow: http://localhost:5000
- Airflow: http://localhost:8080

By default, Compose starts the API with `API_STUB_MODE=true` so the service can run
without local model artifacts. Set `API_STUB_MODE=false` in `.env` when
`core/artifacts/model.keras`, `core/models/artifacts/scaler.joblib` and
`core/models/artifacts/pca.joblib` are available.

Stop the stack:

```bash
docker compose down
```
