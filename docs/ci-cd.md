# Pipeline CI/CD

## Pourquoi

Avant ce travail, il n'y avait aucune vérification automatique sur le code poussé sur GitHub. Ça veut dire que rien n'empêchait un commit avec du code mal formaté, une erreur de typage, ou pire, une régression sur les routes d'authentification, d'arriver sur `master` sans que personne ne s'en aperçoive avant de le tester à la main. Sur un projet évalué par un jury, c'est aussi un problème de crédibilité : si le jury clone le repo et que rien ne garantit que ce qui est décrit dans la doc fonctionne réellement, ça affaiblit tout le travail fait à côté.

Le but du pipeline, donc, c'est de vérifier automatiquement, à chaque push et à chaque pull request, que :

1. le code respecte les règles de style et de typage du projet (lint) ;
2. la suite de tests de l'API passe (test) ;
3. les trois images Docker se construisent sans erreur (build).

Rien de plus pour l'instant — pas de déploiement automatique, parce qu'il n'y a pas d'environnement cible (serveur, registre d'images) vers lequel déployer. Mieux vaut un pipeline honnête qui fait ce qu'il dit, plutôt qu'un pipeline qui prétend déployer vers un endroit qui n'existe pas.

## Où le trouver

Le workflow vit dans `.github/workflows/ci.yml`. C'est un fichier GitHub Actions standard, déclenché sur :

```yaml
on:
  push:
    branches: [master, main]
  pull_request:
    branches: [master, main]
  workflow_dispatch:
```

`workflow_dispatch` permet en plus de le relancer manuellement depuis l'onglet **Actions** de GitHub, ce qui est pratique pour vérifier que tout fonctionne encore sans avoir à faire un commit vide.

Il y a aussi un `concurrency` group configuré :

```yaml
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true
```

Ça annule automatiquement une run en cours si on pousse un nouveau commit sur la même branche pendant qu'elle tourne encore. Sans ça, chaque petit commit de correction pendant qu'on itère finirait sa propre run jusqu'au bout, ce qui gaspille des minutes CI pour rien — on ne s'intéresse de toute façon qu'au résultat du dernier commit.

## Job 1 — Lint

```yaml
lint:
  steps:
    - uses: actions/checkout@v4
    - uses: astral-sh/setup-uv@v5
      with:
        enable-cache: true
    - run: uv tool install pre-commit
    - run: uv tool run pre-commit run --all-files --show-diff-on-failure
```

Ce job exécute exactement les mêmes hooks que ceux définis dans `.pre-commit-config.yaml`, celui qu'on est censé avoir installé en local via `pre-commit install` (voir la cible `check-precommit` du `Makefile`). Trois outils tournent l'un après l'autre :

- **`ruff`** — linter Python rapide, écrit en Rust. Il détecte les erreurs évidentes (variable inutilisée, import mort, etc.) et peut corriger automatiquement une partie d'entre elles (`args: [--fix]`).
- **`ruff-format`** — le formatteur associé (équivalent de `black`, en plus rapide). Il uniformise l'indentation, les guillemets, le retour à la ligne des appels trop longs.
- **`mypy`** — vérifie la cohérence des annotations de type. Le projet impose des type hints sur toutes les fonctions (voir `CLAUDE.md`), donc autant s'assurer qu'ils sont corrects et pas juste décoratifs.

### Une galère qu'on a eue en la mettant en place : `isort` contre `ruff-format`

À l'origine, `.pre-commit-config.yaml` avait un quatrième hook : `isort`, pour trier les imports. Sur le papier ça a l'air raisonnable, sauf qu'en pratique ce hook et `ruff-format` n'étaient pas d'accord entre eux sur la mise en page de certains blocs d'import multi-lignes. Résultat : à chaque exécution, `isort` réécrivait un fichier dans un format, puis `ruff-format` le réécrivait dans un autre au run suivant, et ainsi de suite — le pipeline n'atteignait jamais un état stable, il aurait échoué indéfiniment même sur du code parfaitement propre.

Plutôt que de bricoler une configuration pour forcer les deux outils à se mettre d'accord, on a supprimé le hook `isort` et activé directement les règles de tri d'imports intégrées à `ruff` (le jeu de règles `I`, compatible `isort`) :

```toml
# pyproject.toml
[tool.ruff.lint]
extend-select = ["I"]
```

`ruff` gère maintenant le tri des imports *et* le formatage dans le même outil, donc il ne peut plus être en désaccord avec lui-même. C'est aussi une dépendance en moins à maintenir. La leçon plus générale : faire tourner deux formatteurs différents sur le même périmètre de code est une source classique de flakiness en CI, autant s'y prendre avec un seul outil dès que c'est possible.

### Un fichier volontairement exclu

`services/api/schemas/inference.py` est explicitement documenté comme un contrat d'entrée/sortie figé, à ne pas modifier (voir `CLAUDE.md`). Pour que le formatteur ne le retouche pas automatiquement à chaque run, il est exclu explicitement dans `pyproject.toml` :

```toml
[tool.ruff]
extend-exclude = ["services/api/schemas/inference.py"]
```

## Job 2 — Tests API

```yaml
test-api:
  steps:
    - uses: actions/checkout@v4
    - uses: astral-sh/setup-uv@v5
      with:
        enable-cache: true
    - run: uv sync --group api --group dev
    - run: uv run pytest tests/api -v
```

Ce job installe uniquement les groupes de dépendances `api` et `dev` (pas `core`, qui contient des outils comme `matplotlib` ou `seaborn` qui n'ont rien à voir avec l'API), puis lance la suite de tests dans `tests/api/`. C'est le seul dossier de tests réellement peuplé aujourd'hui — `tests/core/` existe mais est vide, donc il n'y a pas encore de tests automatisés sur la pipeline de features/entraînement (`core/src/`). Si ce dossier se remplit un jour, il faudra ajouter le groupe `core` et une étape équivalente ici.

Les tests eux-mêmes ne nécessitent aucun artefact ML réel (pas de `model.keras`, pas de `.env`) : ils mockent `PredictorService` et `BaseTrainingService` via `app.dependency_overrides` (voir par exemple `tests/api/test_routes_predict.py`). C'est précieux pour la CI, ça évite d'avoir à embarquer ou télécharger un modèle entraîné juste pour vérifier que les routes répondent avec les bons codes HTTP.

## Job 3 — Build Docker

```yaml
docker-build:
  needs: [lint, test-api]
  strategy:
    matrix:
      include:
        - image: api
          dockerfile: infra/docker/Dockerfile.api
        - image: mlflow
          dockerfile: infra/docker/Dockerfile.mlflow
        - image: airflow
          dockerfile: infra/docker/Dockerfile.airflow
  steps:
    - uses: actions/checkout@v4
    - uses: docker/setup-buildx-action@v3
    - uses: docker/build-push-action@v6
      with:
        context: .
        file: ${{ matrix.dockerfile }}
        push: false
        tags: rakuten-${{ matrix.image }}:ci
        cache-from: type=gha,scope=${{ matrix.image }}
        cache-to: type=gha,mode=max,scope=${{ matrix.image }}
```

Ce job attend que `lint` et `test-api` soient passés (`needs: [lint, test-api]`) avant de démarrer — pas la peine de perdre du temps à construire trois images Docker si le code ne passe même pas le lint. Il utilise une matrice (`strategy.matrix`) pour construire les trois images en parallèle plutôt que dans un seul job séquentiel, ce qui va plus vite.

`push: false` est volontaire : on vérifie juste que le `docker build` se termine sans erreur, on ne publie rien nulle part (il n'y a pas de compte sur un registre configuré pour ce projet). Le cache GitHub Actions (`type=gha`) permet de ne pas retélécharger toutes les couches à chaque run — utile vu que l'image `api` embarque `torch` et `tensorflow`, qui sont volumineux.

## Reproduire le pipeline en local

Avant de pousser, on peut rejouer exactement ce que fait la CI :

```bash
# Lint (identique au job "lint")
uv tool run pre-commit run --all-files

# Tests (identique au job "test-api")
uv sync --group api --group dev
uv run pytest tests/api -v

# Build Docker (identique au job "docker-build", en séquentiel)
docker build -f infra/docker/Dockerfile.api -t rakuten-api:local .
docker build -f infra/docker/Dockerfile.mlflow -t rakuten-mlflow:local .
docker build -f infra/docker/Dockerfile.airflow -t rakuten-airflow:local .
```

Encore mieux : installer les hooks pre-commit une bonne fois pour toutes, pour que le lint tourne automatiquement à chaque `git commit` sans y penser :

```bash
pre-commit install
# ou : make check-precommit (cible définie dans le Makefile)
```

## Limites actuelles

- **Pas de déploiement continu.** Le "CD" du nom est un peu ambitieux pour l'instant — il n'y a que de l'intégration continue (CI). Ajouter un vrai déploiement suppose de choisir une cible (un serveur, un cluster, un service cloud) qui n'existe pas encore pour ce projet.
- **`tests/core/` est vide.** La pipeline de feature engineering et d'entraînement (`core/src/`) n'est couverte par aucun test automatisé, contrairement à l'API.
- **Pas de scan de vulnérabilités des dépendances ni des images** (type `pip-audit` ou `trivy`). Vu le nombre de dépendances tierces (transformers, torch...), ce serait une amélioration naturelle à ajouter.
