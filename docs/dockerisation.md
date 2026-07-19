# Dockerisation du projet

## Pourquoi on a fait ça

Au départ, le projet tournait uniquement en local avec un environnement `uv` installé à la main. Ça marchait pour développer, mais ça posait deux problèmes dès qu'on voulait le faire tourner ailleurs (chez un autre membre de l'équipe, sur une VM, ou plus tard en prod) :

- il fallait recréer exactement le même environnement Python (version, dépendances, artefacts du modèle) à chaque fois, à la main ;
- l'API ne vit pas seule : elle a besoin de MLflow pour le tracking, et à terme d'Airflow pour orchestrer le ré-entraînement. Faire démarrer ces trois briques dans le bon ordre, avec la bonne config réseau, devient vite pénible sans outillage.

Docker résout les deux : chaque service tourne dans une image qui contient exactement ce dont il a besoin, et `docker-compose.yml` décrit comment les faire causer entre eux.

## Vue d'ensemble des services

Le fichier `docker-compose.yml` à la racine du repo définit six services :

| Service | Rôle | Port exposé |
|---|---|---|
| `api` | Le serveur FastAPI (`/predict`, `/train`, `/auth`) | 8000 |
| `mlflow` | Tracking des runs d'entraînement | 5000 |
| `airflow-postgres` | Base de données pour Airflow | — |
| `airflow-init` | Initialise la DB Airflow + crée l'utilisateur admin (tourne une fois puis s'arrête) | — |
| `airflow-webserver` | Interface web Airflow | 8080 |
| `airflow-scheduler` | Ordonnanceur qui déclenche les DAGs | — |

Les trois premiers Dockerfiles vivent dans `infra/docker/` : `Dockerfile.api`, `Dockerfile.mlflow`, `Dockerfile.airflow`. Airflow, lui, n'a pas de logique custom particulière — on part de l'image officielle `apache/airflow:2.10.5-python3.12` et on installe juste nos dépendances par-dessus (`services/airflow/requirements.txt`).

C'est le Dockerfile de l'API qui mérite le plus d'explications, donc c'est sur celui-là qu'on va passer le plus de temps.

## Le Dockerfile de l'API, étape par étape

```dockerfile
# infra/docker/Dockerfile.api
FROM python:3.12-slim AS builder
WORKDIR /build
COPY services/api/requirements.txt requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim AS runner
RUN useradd --create-home appuser
WORKDIR /app
COPY --from=builder /install /usr/local
COPY services/ ./services/
COPY core/ ./core/
USER appuser
EXPOSE 8000
ENTRYPOINT ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Un build en deux étapes (multi-stage build)

La première étape (`builder`) ne sert qu'à installer les paquets Python avec pip, dans un dossier `/install` séparé plutôt que dans le système. La deuxième étape (`runner`) repart d'une image toute propre et ne récupère que ce dossier `/install`, pas le cache pip ni les outils de compilation qui ont pu être utilisés en route.

Résultat concret : l'image finale ne contient pas les traces du build (cache pip, `.tar.gz` téléchargés, etc.), juste les paquets installés et le code. Pour un projet qui embarque `sentence-transformers`, `torch` et `tensorflow`, ça compte — sans cette séparation, l'image ferait facilement quelques centaines de Mo de plus.

### Un utilisateur non-root

```dockerfile
RUN useradd --create-home appuser
...
USER appuser
```

Par défaut, un conteneur Docker tourne en `root` si on ne précise rien. C'est pratique pour développer mais dangereux en prod : si quelqu'un arrive à exécuter du code arbitraire dans le conteneur (faille dans une dépendance, par exemple), il aurait les pleins pouvoirs *dans le conteneur*. En tournant en tant qu'`appuser` sans privilèges, on limite les dégâts possibles. C'est une recommandation assez classique de durcissement des images Docker, donc autant l'avoir dès le départ plutôt que de la rajouter plus tard sous la pression.

### `requirements.txt`, pas `pyproject.toml` directement

On aurait pu faire `COPY pyproject.toml` et laisser `uv` résoudre les dépendances au moment du build. On a préféré passer par un `requirements.txt` généré à l'avance (`services/api/requirements.txt`), pour deux raisons :

1. **Reproductibilité** : ce fichier fige les versions exactes de tout ce qui va être installé (y compris les dépendances transitives). Deux builds à des dates différentes installent rigoureusement la même chose.
2. **Simplicité du Dockerfile** : pas besoin d'installer `uv` dans l'image juste pour résoudre des dépendances, `pip install -r` suffit.

Ce fichier est régénéré avec :

```bash
uv export -qq --group api --no-dev --no-hashes -o services/api/requirements.txt
```

Le `--no-dev` compte : sans lui, `uv export` embarque aussi le groupe `dev` (pytest, httpx…) par défaut, ce qui n'a rien à faire dans une image de prod. On l'a découvert en creusant ce sujet — la commande documentée dans le `Makefile` (`make add PACKAGE=... GROUP=api`) ne le précisait pas, ce qui fait que `pytest` se retrouvait silencieusement dans les requirements de prod à chaque régénération. À garder en tête si vous retouchez à cette commande.

**Remarque historique** : il existait avant un second fichier, `requirements.docker.txt`, maintenu à la main en parallèle de `requirements.txt`, censé être une version « allégée » pour Docker. Le problème, c'est qu'il avait été composé à la main un jour et plus jamais mis à jour depuis : il ne contenait ni `scikit-learn`, ni `sentence-transformers`, ni `keras`/`tensorflow`, alors que `PredictorService` en a besoin dès qu'on sort du mode stub (voir plus bas). Avoir deux fichiers de dépendances qui doivent rester synchronisés à la main, c'est exactement le genre de piège qui finit par casser un jour sans que personne ne s'en rende compte — c'est ce qui s'est passé ici. On a supprimé ce fichier et fait pointer le Dockerfile directement sur `requirements.txt`, qui lui est généré automatiquement et donc toujours à jour avec `pyproject.toml`.

### Les artefacts ML ne sont pas dans l'image

```dockerfile
# Artifacts (model.keras, scaler.joblib, pca.joblib, labels_map.json) are
# mounted at runtime via docker-compose volume. They are NOT baked into the
# image so that model updates do not require an image rebuild.
```

C'est un choix de design important. Le modèle entraîné (`model.keras`), le scaler, la PCA et le mapping des labels ne sont pas copiés dans l'image au moment du build. Ils sont montés en volume au démarrage du conteneur, via `docker-compose.yml` :

```yaml
volumes:
  - ./core/artifacts:/app/core/artifacts
  - ./core/data:/app/core/data
  - ./core/models/artifacts:/app/core/models/artifacts
```

L'intérêt : si on ré-entraîne le modèle et qu'on obtient un nouveau `model.keras`, on n'a pas besoin de reconstruire toute l'image Docker (ce qui prendrait du temps et ferait redescendre `torch`/`tensorflow` inutilement). Il suffit de redémarrer le conteneur `api` — le nouveau fichier sera repris automatiquement au prochain chargement des artefacts, au démarrage du serveur.

## Configuration par variables d'environnement

L'API ne lit aucune valeur en dur dans le code : tout passe par `services/api/config.py`, qui utilise `pydantic-settings`. Chaque variable d'environnement doit être préfixée par `API_` (par exemple `API_JWT_SECRET`). Le fichier `.env.example` liste toutes les variables attendues et sert de modèle pour créer son propre `.env` :

```bash
cp .env.example .env
```

Dans `docker-compose.yml`, ces variables sont passées au conteneur `api` avec des valeurs par défaut de secours (syntaxe `${VAR:-défaut}`) :

```yaml
environment:
  API_JWT_SECRET: ${API_JWT_SECRET:-dev-secret-change-me}
  API_STUB_MODE: ${API_STUB_MODE:-true}
  ...
```

Le point important ici, c'est `API_STUB_MODE`. Par défaut il vaut `true`, ce qui veut dire que l'API démarre **sans charger le vrai modèle** et renvoie des prédictions aléatoires (voir `services/api/services/stub_predictor.py`). C'est volontaire : ça permet de faire tourner toute la stack Docker pour tester l'architecture, l'auth, les routes, sans avoir besoin d'avoir déjà entraîné un modèle. Pour lancer une vraie inférence, il faut mettre `API_STUB_MODE=false` dans son `.env` et s'assurer que les artefacts (`model.keras`, `scaler.joblib`, `pca.joblib`, `labels_map.json`) sont bien présents dans les dossiers montés.

## Healthchecks et ordre de démarrage

Chaque service critique a un `healthcheck` dans `docker-compose.yml`. Par exemple pour l'API :

```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
  interval: 30s
  timeout: 5s
  retries: 3
  start_period: 30s
```

Ça sert à deux choses. D'abord, Docker peut signaler qu'un conteneur est « unhealthy » s'il ne répond plus, ce qui est utile pour du monitoring. Ensuite, et c'est le plus important ici, ça permet d'enchaîner le démarrage des services dans le bon ordre grâce à `depends_on: ... condition: service_healthy`. Par exemple, `api` déclare qu'il dépend de `mlflow` en bonne santé avant de démarrer :

```yaml
api:
  depends_on:
    mlflow:
      condition: service_healthy
```

Sans ça, Docker Compose démarrerait les deux conteneurs en même temps, et l'API risquerait de tenter de contacter MLflow avant qu'il soit prêt à répondre.

La route `/health` utilisée par ce healthcheck est volontairement très simple (voir `services/api/monitoring/health.py`) : elle répond `200 OK` tant que le processus tourne, sans vérifier si le modèle est chargé. Il existe une deuxième route, `/ready`, qui elle vérifie que `PredictorService` a bien fini de charger ses artefacts et répond `503` sinon. La distinction correspond aux probes *liveness* / *readiness* de Kubernetes : on ne veut pas qu'un modèle en cours de chargement fasse redémarrer le conteneur en boucle (ce que ferait une liveness probe qui échoue), mais on ne veut pas non plus lui envoyer du trafic avant qu'il soit prêt (rôle de la readiness probe). Le `docker-compose.yml` actuel n'utilise que `/health` pour son healthcheck, mais `/ready` est déjà prête à être branchée si le projet passe un jour sous Kubernetes.

## Lancer la stack

```bash
# Build + démarrage de tous les services en arrière-plan
make docker-up
# équivalent à : docker compose up -d

# Suivre les logs
make docker-logs

# Tout arrêter
make docker-down
```

Pour ne démarrer que l'API (utile en dev quand on n'a pas besoin d'Airflow) :

```bash
docker compose up api --build
```

Une fois lancé, la documentation interactive de l'API (Swagger) est disponible sur `http://localhost:8000/docs`.

## Ce qui reste perfectible

Pour être honnête sur l'état actuel du projet plutôt que de faire comme si tout était figé :

- **Pas d'image poussée sur un registre.** Le pipeline CI (voir `docs/ci-cd.md`) vérifie que les images se construisent, mais rien n'est publié sur Docker Hub ou un autre registre pour l'instant. Il n'y a pas non plus de déploiement automatisé vers un environnement cible.
- **Le mot de passe admin par défaut d'Airflow est `admin`/`admin`** (voir `.env.example`), à changer avant tout usage réel.
- **Le service `mlflow` utilise SQLite** comme backend de tracking (`--backend-store-uri sqlite:///...`), ce qui est très bien pour un projet étudiant mais ne tiendrait pas la charge d'un vrai usage en production avec plusieurs utilisateurs simultanés.
