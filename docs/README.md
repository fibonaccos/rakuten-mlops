# Documentation

Ce dossier documente les parties du projet liées à l'industrialisation de l'API : conteneurisation, intégration continue et sécurité.

- [`dockerisation.md`](./dockerisation.md) — comment et pourquoi chaque service (API, MLflow, Airflow) est conteneurisé, ce que fait chaque Dockerfile, comment lancer la stack.
- [`ci-cd.md`](./ci-cd.md) — le pipeline GitHub Actions (`.github/workflows/ci.yml`) : lint, tests, build Docker, et comment le reproduire en local avant de pousser.
- [`securite-api.md`](./securite-api.md) — l'authentification JWT et comment les routes `/predict` et `/train` sont protégées, avec les limites connues.

Ces documents partent du principe que le lecteur connaît Python et FastAPI, mais pas forcément Docker ni GitHub Actions en détail — l'idée est de pouvoir comprendre le *pourquoi* de chaque choix, pas seulement le *quoi*.
