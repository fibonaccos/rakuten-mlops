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
├── assets/                 # categories.json, demo_products.json (extraits du dataset)
├── clients/                # api_client.py, mlflow_client.py, airflow_client.py
├── domain/                 # artifacts.py, catalog.py, labels.py
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
| Vue d'ensemble | Problème métier, architecture, état des services | `categories.json`, `metrics.json`, sondes `/health` et `/ready` |
| Données & features | Répartition réelle des 27 catégories, ce que contient chacune, chaîne de features | `categories.json`, `class_weights.json`, `params.yaml`, `metadata.json` |
| Prédiction | Démonstration unitaire, traitement par lot, contrat d'API | `demo_products.json`, `POST /predict`, `POST /predict/batch`, `/openapi.json` |
| Performance | Métriques globales et par classe, courbes, matrice de confusion | `metrics.json`, `history.json`, `confusion_matrix.png` |
| MLOps | DAG Airflow, runs MLflow, réentraînement, CI/CD | API REST Airflow et MLflow, `POST /train` |
| Monitoring | Disponibilité, temps de réponse, feuille de route supervision | Mesures client, sondes `/health` |

## D'où viennent les chiffres

Aucune donnée n'est inventée ni codée en dur dans les pages. Les valeurs affichées ont trois origines :

1. **Les artefacts du pipeline** — `core/artifacts/` (`metrics.json`, `history.json`, `class_weights.json`, `labels_map.json`, `confusion_matrix.png`) et `core/data/features/metadata.json` pour la variance conservée par la PCA.
2. **Les services, en direct** — prédictions, jobs d'entraînement, runs MLflow, DAG Airflow, temps de réponse.
3. **Deux extraits du catalogue brut**, versionnés dans `services/streamlit/assets/`.

Les fichiers du challenge sont trop volumineux pour être versionnés (60 Mo de CSV, 2,4 Go d'images). Le front-end embarque donc deux extraits, régénérables :

```bash
uv run python scripts/build_ui_assets.py --raw-dir <dossier des CSV Rakuten>
```

`categories.json` contient, pour chacun des 27 codes : le nombre réel de produits, sa part du corpus, le taux de descriptions réellement remplies, les termes nettement sur-représentés par rapport au reste du catalogue, et trois vrais titres de fiches.

`demo_products.json` contient un produit réel par catégorie, **restreint au jeu de test du modèle** (`core/data/features/x_test.parquet`) : aucun n'a été vu à l'entraînement, et chacun porte son vrai `prdtypecode`. C'est ce qui permet à la démonstration d'afficher « prédit 2583 / réel 2583 » plutôt que de demander au jury de faire confiance.

### Le cas des libellés de catégories

Rakuten n'a jamais publié la signification de ses codes : le jeu de données ne contient que des nombres. Les libellés utilisés dans l'application ont donc été établis en lisant le catalogue, et l'onglet « Les 27 catégories » affiche à côté de chaque nom les termes et les titres qui le justifient — le jury peut vérifier plutôt que croire.

Quelques lectures que les données tranchent, et qu'on aurait mal devinées :

- `1160` est dominé par `pokemon`, `mtg`, `panini`, `foil`, `rare` : ce sont des **cartes à collectionner**, malgré une numérotation qui pourrait faire penser à des livres ;
- `2905` a 100 % de descriptions remplies autour de `dlc`, `telechargement`, `extension` : des **jeux dématérialisés** ;
- `1301` mélange fléchettes (`flechette`, `ailettes`, `harrows`), billard (`aramith`, `bce`) et baby-foot, d'où un libellé volontairement large.

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

## Tests

```bash
uv run pytest tests/streamlit -v
```

Trois familles :

- `test_api_client.py` — construction des requêtes, gestion des erreurs HTTP et réseau, enregistrement des latences ;
- `test_domain.py` — lecture des artefacts et des extraits du catalogue, cohérence entre `labels_map.json` et les libellés, vérification que les 27 produits de démonstration couvrent bien toutes les catégories, dégradation propre quand les fichiers manquent ;
- `test_pages_render.py` — rendu des six pages via `streamlit.testing.v1.AppTest`, services éteints.

Le job `test-ui` de la CI joue cette suite à chaque push, et `docker-build` construit aussi l'image `streamlit`.

## Déroulé de soutenance (20 minutes)

L'interface est organisée pour être parcourue dans l'ordre des pages. Répartition indicative :

| Temps | Page | Points à faire passer |
| --- | --- | --- |
| 0–3 min | Vue d'ensemble | Le problème métier, le schéma d'architecture, les quatre services qui tournent devant le jury |
| 3–6 min | Données & features | Le déséquilibre des classes (2583 pèse 12 % à lui seul), ce que contient réellement chaque code, la chaîne 396 → 120 |
| 6–11 min | Prédiction | **Le moment fort** : une vraie fiche du jeu de test classée en direct et confrontée à sa catégorie réelle, la lecture de la confiance, un lot de 12 produits, puis le `curl` équivalent |
| 11–14 min | Performance | Métriques globales, les catégories qui coincent, la matrice de confusion, ce qu'on ferait ensuite |
| 14–18 min | MLOps | Airflow, un run MLflow avec son commit Git, un entraînement déclenché depuis l'interface, la CI |
| 18–20 min | Monitoring | Temps de réponse mesurés en direct, et ce qui reste à faire côté Prometheus/Grafana |

Deux réflexes utiles le jour J : lancer le stack au moins cinq minutes avant (le premier appel à `/predict` paie le chargement du modèle de plongement), et se connecter dès l'ouverture pour ne pas buter sur l'authentification au milieu de la démonstration.
