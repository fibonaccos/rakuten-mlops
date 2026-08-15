# Front-end Streamlit

Interface de démonstration et de pilotage du stack Rakuten MLOps. Elle ne
contient aucune logique de machine learning : elle parle à l'API en HTTP, lit
les artefacts du pipeline sur disque et interroge MLflow et Airflow par leurs
API REST.

## Démarrage

### En local

```bash
uv sync --group streamlit
uv run streamlit run services/streamlit/app.py
```

Ou, plus court :

```bash
make ui
```

L'interface est servie sur <http://localhost:8501>. Elle suppose que l'API
tourne sur <http://localhost:8000> ; sinon, les points d'accès sont modifiables
dans la barre latérale, section « Points d'accès ».

### Dans le stack Docker

```bash
docker compose up -d --build streamlit
```

Le conteneur pointe vers les noms de services internes (`api`, `mlflow`,
`airflow-webserver`) et monte `core/artifacts/` en lecture seule.

## Configuration

Toutes les variables sont préfixées `UI_` et se placent dans le `.env` à la
racine (voir `.env.example`).

| Variable | Défaut | Rôle |
| --- | --- | --- |
| `UI_API_URL` | `http://localhost:8000` | API de prédiction |
| `UI_MLFLOW_URL` | `http://localhost:5000` | Serveur de tracking MLflow |
| `UI_AIRFLOW_URL` | `http://localhost:8080` | Webserver Airflow |
| `UI_DEFAULT_USERNAME` | `admin` | Identifiant pré-rempli dans le formulaire |
| `UI_DEFAULT_PASSWORD` | `changeme` | Mot de passe pré-rempli (confort local) |
| `UI_AIRFLOW_USERNAME` | `admin` | Basic auth Airflow |
| `UI_AIRFLOW_PASSWORD` | `admin` | Basic auth Airflow |
| `UI_ARTIFACTS_DIR` | `core/artifacts` | Métriques, historique, matrice de confusion |
| `UI_PARAMS_FILE` | `core/params.yaml` | Paramètres du pipeline |
| `UI_MLRUNS_DIR` | `tracking/mlflow/mlruns` | Magasin MLflow local (mode dégradé) |

Le thème est fixé dans `.streamlit/config.toml` à la racine : palette de
graphiques validée sur fond clair, en contraste et en séparation pour les
daltonismes.

## Organisation du code

```
services/streamlit/
├── app.py                  # Point d'entrée : navigation et assemblage des pages
├── settings.py             # Configuration UI_* (pydantic-settings)
├── assets/                 # Extraits du catalogue brut (voir plus bas)
├── clients/                # Accès HTTP : API, MLflow, Airflow
├── domain/                 # Artefacts, catalogue, libellés de catégories
├── ui/                     # Thème, graphiques, mise en page, état de session
└── views/                  # Une page = un module exposant render()
```

## Données de référence

Les fichiers du challenge Rakuten sont trop volumineux pour être versionnés. Deux
extraits sont donc embarqués dans `assets/` et régénérables :

```bash
uv run python scripts/build_ui_assets.py --raw-dir <dossier des CSV Rakuten>
```

- `categories.json` — pour chaque `prdtypecode` : nombre réel de produits, part du
  corpus, taux de descriptions remplies, termes caractéristiques et vrais titres.
- `demo_products.json` — un produit réel par catégorie, pris dans le jeu de test du
  modèle, avec sa vraie catégorie : la démonstration est donc vérifiable.

Chaque page est un module autonome : ajouter une page revient à écrire un
`render()` et à l'inscrire dans `PAGES` (`app.py`).

## Les six pages

1. **Vue d'ensemble** — problème métier, architecture, état des services.
2. **Données & features** — catalogue réel, déséquilibre, contenu de chaque code, chaîne de features.
3. **Prédiction** — démonstration unitaire vérifiable, par lot, et contrat d'API.
4. **Performance** — métriques globales et par catégorie, courbes, confusion.
5. **MLOps** — Airflow, MLflow, réentraînement piloté, CI/CD.
6. **Monitoring** — disponibilité, temps de réponse mesurés, feuille de route Grafana.

## Tests

```bash
uv run pytest tests/streamlit -v
```

Les tests couvrent le client HTTP, la lecture des artefacts et le rendu des six
pages — y compris lorsqu'aucun service n'est joignable, cas le plus embêtant en
démonstration.

## Principes de conception

- **Aucun import de `services.api`.** Ce que l'interface affiche, elle l'a
  obtenu en HTTP, comme n'importe quel client.
- **Aucune page ne casse quand un service est éteint.** Elle explique ce qui
  manque et comment le démarrer.
- **Rien n'est recalculé, rien n'est inventé.** Les métriques viennent de
  `metrics.json`, les runs de MLflow, les états de DAG d'Airflow, et les chiffres
  du catalogue des extraits versionnés dans `assets/`.
