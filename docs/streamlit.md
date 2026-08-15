# Front-end Streamlit

## Pourquoi

Jusqu'ici, montrer le projet voulait dire ouvrir trois onglets : Swagger pour l'API, l'UI MLflow pour les runs, l'UI Airflow pour les DAG. Chacune de ces interfaces est faite pour un développeur qui sait déjà ce qu'il cherche. Aucune ne raconte ce que fait le projet, ni pourquoi les choix qu'on a faits tiennent debout.

Le front-end Streamlit répond à ce besoin-là. Il a deux usages :

1. **Démontrer** — un jury, ou n'importe qui d'extérieur, peut envoyer un produit et voir la catégorie sortir, sans écrire une seule ligne de `curl`.
2. **Piloter** — l'interface expose les fonctions réelles du stack : lancer un entraînement, déclencher un DAG, consulter les métriques, mesurer les temps de réponse.

Un point de conception est structurant : **l'interface n'importe jamais le code de l'API**. Elle passe par HTTP, comme n'importe quel client. Si l'API tombe, l'interface le dit ; elle ne peut pas afficher un résultat que le service n'a pas produit. C'est ce qui rend la démonstration honnête.

## Où le trouver

```
services/streamlit/
├── app.py                  # Point d'entrée : navigation, assemblage
├── settings.py             # Configuration UI_* (pydantic-settings)
├── clients/                # api_client.py, mlflow_client.py, airflow_client.py
├── domain/                 # artifacts.py, labels.py, samples.py
├── ui/                     # theme.py, charts.py, layout.py, state.py
└── views/                  # une page = un module exposant render()
```

Le thème est dans `.streamlit/config.toml`, à la racine, parce que Streamlit lit ce fichier depuis le répertoire de travail.

## Lancer l'interface

En local, sans Docker :

```bash
uv sync --group streamlit
make ui                     # ou : uv run streamlit run services/streamlit/app.py
```

Dans le stack complet :

```bash
docker compose up -d --build
```

L'interface est alors sur <http://localhost:8501>. Elle suppose l'API sur le port 8000, MLflow sur 5000, Airflow sur 8080 — ces trois adresses sont modifiables à chaud dans la barre latérale, section « Points d'accès », ce qui évite de redémarrer le conteneur pour pointer ailleurs.

Pour une démonstration sans modèle entraîné sous la main, l'API peut tourner en mode bouchon :

```bash
API_STUB_MODE=true uv run uvicorn services.api.main:app --port 8000
```

L'interface affiche alors un bandeau explicite : le contrat HTTP est réel, les prédictions ne le sont pas. Autant que ce soit dit à l'écran plutôt que découvert par le jury.

## Les six pages

| Page | Ce qu'elle montre | Ce qu'elle lit |
| --- | --- | --- |
| Vue d'ensemble | Problème métier, architecture, état des services | `metrics.json`, sondes `/health` et `/ready` |
| Données & features | Répartition des 27 catégories, déséquilibre, chaîne de features | `metrics.json`, `class_weights.json`, `params.yaml` |
| Prédiction | Démonstration unitaire, traitement par lot, contrat d'API | `POST /predict`, `POST /predict/batch`, `/openapi.json` |
| Performance | Métriques globales et par classe, courbes, matrice de confusion | `metrics.json`, `history.json`, `confusion_matrix.png` |
| MLOps | DAG Airflow, runs MLflow, réentraînement, CI/CD | API REST Airflow et MLflow, `POST /train` |
| Santé | Disponibilité, temps de réponse, feuille de route supervision | Mesures client, sondes `/health` |

## Authentification

Les routes `/predict` et `/train` sont protégées par JWT (voir [`securite-api.md`](./securite-api.md)). La barre latérale contient donc un formulaire de connexion : il appelle `POST /auth/login` en OAuth2 password flow et garde le jeton dans l'état de session Streamlit.

Le jeton vit dans la session serveur de Streamlit, pas dans le navigateur, et disparaît à la déconnexion ou à la fermeture de l'onglet. Les identifiants pré-remplis (`UI_DEFAULT_USERNAME`, `UI_DEFAULT_PASSWORD`) sont un confort de démonstration locale ; en dehors de ça, il n'y a aucune raison de les renseigner.

## Modes dégradés

Chaque dépendance externe a un comportement défini quand elle est absente :

| Service absent | Comportement |
| --- | --- |
| API | Pastille rouge dans la barre latérale, message d'erreur explicite sur les pages d'inférence, les pages hors ligne (données, performance) restent complètes |
| MLflow | Bascule sur la lecture directe du magasin de fichiers `tracking/mlflow/mlruns` produit par les entraînements lancés hors Docker |
| Airflow | Message expliquant la commande à lancer, le reste de la page MLOps continue de fonctionner |
| Artefacts manquants | Avertissement nommant le fichier attendu et la commande qui le régénère |

C'est testé : `tests/streamlit/test_pages_render.py` rend les six pages sans qu'aucun service ne tourne et vérifie qu'aucune ne lève d'exception.

## Choix de visualisation

Trois règles suivies partout, parce qu'elles évitent les graphiques trompeurs :

- **Jamais deux axes verticaux.** Une perte et une exactitude sur le même graphique donnent une illusion de corrélation. Les courbes d'apprentissage sont donc trois graphiques séparés.
- **Une couleur = une entité, jamais un rang.** Filtrer par famille de produits ne repeint pas les barres restantes.
- **Palette validée, pas choisie à l'œil.** Les trois couleurs de séries passent les seuils de séparation pour les daltonismes (deutéranopie, tritanopie) et de contraste sur fond blanc. Le thème est verrouillé en clair pour cette raison — un basculement automatique en sombre invaliderait la validation.

Les libellés de catégories (`domain/labels.py`) sont une lecture métier faite par l'équipe : Rakuten ne publie que les codes numériques. C'est écrit à l'écran, sous le graphique de répartition.

## Tests

```bash
uv run pytest tests/streamlit -v
```

Trois familles :

- `test_api_client.py` — construction des requêtes, gestion des erreurs HTTP et réseau, enregistrement des latences ;
- `test_domain.py` — lecture des artefacts, cohérence entre `labels_map.json` et les libellés, dégradation propre quand les fichiers manquent ;
- `test_pages_render.py` — rendu des six pages via `streamlit.testing.v1.AppTest`, services éteints.

Le job `test-ui` de la CI joue cette suite à chaque push, et `docker-build` construit aussi l'image `streamlit`.

## Déroulé de soutenance (20 minutes)

L'interface est organisée pour être parcourue dans l'ordre des pages. Répartition indicative :

| Temps | Page | Points à faire passer |
| --- | --- | --- |
| 0–3 min | Vue d'ensemble | Le problème métier, le schéma d'architecture, les quatre services qui tournent devant le jury |
| 3–6 min | Données & features | Le déséquilibre des classes, la chaîne 396 → 120, l'exigence de rejouer la même chaîne en inférence |
| 6–11 min | Prédiction | **Le moment fort** : un produit classé en direct, la lecture de la confiance, un lot de 10 produits, puis le `curl` équivalent |
| 11–14 min | Performance | Métriques globales, les catégories qui coincent, la matrice de confusion, ce qu'on ferait ensuite |
| 14–18 min | MLOps | Airflow, un run MLflow avec son commit Git, un entraînement déclenché depuis l'interface, la CI |
| 18–20 min | Santé | Temps de réponse mesurés en direct, et ce qui reste à faire côté Prometheus/Grafana |

Deux réflexes utiles le jour J : lancer le stack au moins cinq minutes avant (le premier appel à `/predict` paie le chargement du modèle de plongement), et se connecter dès l'ouverture pour ne pas buter sur l'authentification au milieu de la démonstration.
