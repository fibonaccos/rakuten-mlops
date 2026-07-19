# Sécurisation de l'API — routes `/predict` et `/train`

## Pourquoi protéger ces routes

Sans authentification, n'importe qui connaissant l'URL de l'API pourrait :

- lancer des prédictions en boucle (`/predict`), ce qui consomme des ressources de calcul (le pipeline d'inférence charge un modèle d'embedding multilingue et fait tourner un réseau de neurones à chaque appel) ;
- surtout, déclencher un **entraînement complet** via `/train`, qui lance un vrai processus d'entraînement en arrière-plan. Un entraînement, ça prend du temps machine, potentiellement du GPU, et ça peut aussi écraser des artefacts existants (`model.keras`, métriques) si on n'y prend pas garde.

`/train` est donc la route la plus sensible du projet : ce n'est pas juste de la lecture, c'est une action qui a un coût réel et des effets de bord durables. Les deux routes sont donc protégées par la même mécanique d'authentification, décrite ci-dessous.

## Le principe : OAuth2 + JWT

Le projet utilise le flux `OAuth2PasswordBearer` de FastAPI, qui est en réalité une implémentation simple de JWT (JSON Web Token) — pas un vrai serveur OAuth2 avec plusieurs clients, scopes, etc. C'est le bon niveau de complexité pour une API interne avec un seul type d'utilisateur (l'admin), sans sur-ingénierer avec un système d'autorisation complet dont on n'a pas l'usage.

Le flux, en résumé :

1. Le client envoie son login/mot de passe à `POST /auth/login`.
2. Si les identifiants sont valides, le serveur renvoie un **token JWT signé**, avec une durée de validité limitée.
3. Pour chaque appel à une route protégée (`/predict`, `/train`), le client doit joindre ce token dans l'en-tête `Authorization: Bearer <token>`.
4. Le serveur vérifie la signature et l'expiration du token à chaque requête, sans avoir besoin de conserver de session côté serveur — c'est tout l'intérêt d'un JWT : il est auto-suffisant.

### `POST /auth/login`

Défini dans `services/api/routes/auth.py` :

```python
@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()) -> Token:
    user = authenticate_user(form_data.username, form_data.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Incorrect username or password", ...)
    return Token(access_token=create_access_token(subject=user.username))
```

`authenticate_user` (dans `services/api/services/auth.py`) va chercher l'utilisateur, puis compare le mot de passe fourni avec le hash stocké via `bcrypt.checkpw`. On ne stocke jamais le mot de passe en clair, ni même son hash n'est jamais renvoyé au client — la fonction retourne un objet `UserInDB` en interne, mais seul un `User` (sans le champ `hashed_password`) transite dans les réponses HTTP (voir `services/api/schemas/auth.py`).

Si l'authentification réussit, `create_access_token` construit le JWT :

```python
def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
```

Le payload est minimal : juste le nom d'utilisateur (`sub`, pour *subject*) et la date d'expiration (`exp`). Le token est signé avec `API_JWT_SECRET` (algorithme HS256 par défaut) — sans connaître ce secret, impossible de forger un token valide ou d'en modifier le contenu sans que la signature ne devienne invalide.

### Vérification du token — `get_current_user`

C'est le cœur du système de protection, dans `services/api/services/auth.py` :

```python
def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    try:
        raw_payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        payload = TokenPayload(**raw_payload)
    except JWTError:
        raise credentials_exception

    if payload.sub is None:
        raise credentials_exception

    user = get_user(payload.sub)
    if user is None or user.disabled:
        raise credentials_exception

    return User(username=user.username, disabled=user.disabled)
```

C'est une dépendance FastAPI (`Depends`) : n'importe quelle route peut demander `current_user: User = Depends(get_current_user)` dans sa signature, et FastAPI se charge d'exécuter cette vérification *avant* d'entrer dans le corps de la fonction. Si le token est absent, expiré, mal signé, ou que l'utilisateur associé n'existe plus (ou a été désactivé), la fonction lève une `HTTPException 401` et la route n'est jamais exécutée.

## Comment `/predict` et `/train` sont protégées — deux styles différents

C'est un détail qui vaut la peine d'être noté parce que les deux fichiers de routes ne branchent pas la protection de la même façon, pour une raison précise.

### `predict.py` — dépendance par endpoint

```python
# services/api/routes/predict.py
@router.post("", response_model=SinglePredictionOutput, ...)
async def predict_single(
    body: SinglePredictionInput,
    predictor: PredictorService = Depends(get_predictor),
    current_user: User = Depends(get_current_user),
) -> SinglePredictionOutput:
    ...
```

Ici, `get_current_user` est injecté individuellement dans chaque fonction de route (`predict_single` et `predict_batch`).

### `train.py` — dépendance au niveau du routeur

```python
# services/api/routes/train.py
router = APIRouter(
    prefix="/train",
    tags=["Training"],
    dependencies=[Depends(get_current_user)],
)
```

Ici, la dépendance est déclarée une seule fois, directement sur l'`APIRouter`. Elle s'applique alors automatiquement à *toutes* les routes définies dans ce routeur (`POST /train`, `GET /train/jobs`, `GET /train/{job_id}`, `DELETE /train/{job_id}/cancel`), sans avoir à la répéter dans la signature de chaque fonction.

Ce deuxième style est plus sûr par construction : il est structurellement impossible d'oublier de protéger une nouvelle route ajoutée sous `/train`, puisque la protection est portée par le routeur et pas par chaque fonction individuellement. Pour `/train`, où l'oubli aurait un coût réel (un déclenchement d'entraînement non autorisé), ce choix a du sens. Pour `/predict`, la déclaration par fonction reste explicite et lisible, et le risque d'oubli est moindre vu qu'il n'y a que deux endpoints à surveiller. Les deux approches sont correctes ; ce sont deux façons différentes de répondre au même besoin, avec un niveau de garantie légèrement différent.

## Le hachage des mots de passe

```python
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
```

`bcrypt` est utilisé directement (pas via `passlib`, qui posait un problème de compatibilité avec les versions récentes de `bcrypt` — c'est documenté dans l'historique des commits du projet). `bcrypt.gensalt()` génère un sel aléatoire différent à chaque hachage, donc même si deux utilisateurs choisissaient le même mot de passe, leurs hashes stockés seraient différents — ça protège contre les attaques par table arc-en-ciel (*rainbow table*).

## Configuration et secrets

Toute la configuration sensible passe par des variables d'environnement (`services/api/config.py`), jamais en dur dans le code métier :

```python
jwt_secret: str = "dev-secret-change-in-production"
access_token_expire_minutes: int = 60
admin_username: str = "admin"
admin_password_hash: str = "$2b$12$.SFcvNZnowY3gVm76LEhD.g9exvyLWMBhiHJAIyvcBWMnjhIL51Qy"
```

Ces valeurs par défaut ne servent qu'au développement local — le hash par défaut correspond au mot de passe `changeme`. En production, il faut impérativement les redéfinir via `.env` (voir `.env.example`) :

```bash
API_JWT_SECRET=$(openssl rand -hex 32)
API_ADMIN_PASSWORD_HASH=$(uv run python -c "import bcrypt; print(bcrypt.hashpw(b'un-vrai-mot-de-passe', bcrypt.gensalt()).decode())")
```

## Comment c'est testé

`tests/api/test_auth.py` couvre le flux de login (bons/mauvais identifiants, token valide/invalide/expiré, `/auth/me`). Pour les routes elles-mêmes, on trouve par exemple dans `tests/api/test_routes_train.py` :

```python
def test_submit_without_token_returns_401(...):
    response = client.post("/train")
    assert response.status_code == 401
```

Ce test vérifie concrètement qu'un appel sans token à `/train` est bien rejeté — c'est le genre de test qui aurait immédiatement détecté une régression si quelqu'un avait par erreur retiré la dépendance `Depends(get_current_user)` du routeur.

Pour tester `/predict` et `/train` manuellement une fois l'API lancée :

```bash
# 1. Récupérer un token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -d "username=admin&password=changeme" \
  -H "Content-Type: application/x-www-form-urlencoded" | jq -r .access_token)

# 2. L'utiliser sur une route protégée
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"designation": "Livre de cuisine"}}'
```

## Ce qui n'est pas (encore) sécurisé

Autant le dire clairement plutôt que de laisser croire que tout est parfait :

- **Un seul utilisateur, stocké en mémoire.** `_FAKE_USERS_DB` dans `services/api/services/auth.py` est un simple dictionnaire Python construit au démarrage à partir des variables `API_ADMIN_USERNAME` / `API_ADMIN_PASSWORD_HASH`. Il n'y a ni base de données d'utilisateurs, ni possibilité de créer plusieurs comptes, ni de révoquer un utilisateur sans redéployer l'API. C'est suffisant pour un projet à un seul opérateur, pas pour une vraie mise en production avec plusieurs personnes.
- **Pas de distinction de rôles.** Le même utilisateur peut appeler `/predict` et `/train` — il n'y a pas de notion de scope JWT ou de rôle qui permettrait, par exemple, de donner à quelqu'un le droit de prédire sans lui donner le droit de lancer un entraînement.
- **Pas de révocation de token.** Un JWT reste valide jusqu'à son expiration (60 minutes par défaut) même si on voudrait le invalider immédiatement (déconnexion forcée, compromission suspectée). Il n'y a pas de liste noire de tokens côté serveur — c'est une limite connue de l'approche JWT "stateless", à comparer avec des sessions côté serveur si ce besoin devient réel.
- **CORS ouvert à tout le monde.** Dans `services/api/main.py` :

  ```python
  app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
  ```

  N'importe quel site web pourrait faire des requêtes vers cette API depuis le navigateur d'un utilisateur. C'est explicitement commenté comme un choix de développement (« Allow all origins in development ») à restreindre avant toute mise en production réelle.
- **Pas de limitation de débit (rate limiting).** Rien n'empêche aujourd'hui un utilisateur authentifié d'appeler `/train` en boucle pour saturer les ressources de calcul, en dehors de la protection déjà en place au niveau métier (un seul job d'entraînement actif à la fois, voir `TrainingConflictError` dans `services/api/services/training.py`, qui renvoie une erreur 409 si un job tourne déjà).

Ces limites ne sont pas des oublis honteux — elles reflètent le niveau de maturité attendu pour un projet de fin d'études, où l'objectif est de démontrer une compréhension solide des mécanismes de sécurité de base (hachage, JWT, injection de dépendances) plutôt que de reproduire un système d'authentification de niveau entreprise. Mais les lister clairement ici évite de laisser penser que l'API est prête pour un usage grand public en l'état.
