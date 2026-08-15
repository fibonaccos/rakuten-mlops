"""
Example products used to drive the live demonstration.

They are written the way real catalogue entries are: abbreviations, brand
names, sometimes a missing description. Each carries the category the team
expects, which makes it easy to comment the result out loud — including when
the model gets it wrong, which is part of an honest demo.
"""

from typing import Final, TypedDict


class SampleProduct(TypedDict):
    """One catalogue entry ready to be sent to the API."""

    designation: str
    description: str
    expected: str


SAMPLES: Final[list[SampleProduct]] = [
    {
        "designation": "Zelda Breath Of The Wild Nintendo Switch",
        "description": (
            "Jeu d'aventure en monde ouvert pour console Nintendo Switch, "
            "version française, boîtier neuf sous blister."
        ),
        "expected": "40",
    },
    {
        "designation": "Manette sans fil DualSense compatible PS5 - noire",
        "description": (
            "Manette de jeu sans fil avec retour haptique et gâchettes adaptatives, "
            "batterie rechargeable, câble USB-C fourni."
        ),
        "expected": "50",
    },
    {
        "designation": "Lot de 3 romans policiers de poche - occasion bon état",
        "description": (
            "Trois romans policiers format poche, couvertures légèrement usées, "
            "pages complètes et propres. Vendus en lot."
        ),
        "expected": "2403",
    },
    {
        "designation": "Table basse scandinave chêne clair 110x60 cm",
        "description": (
            "Table basse en bois massif, pieds compas, finition huilée, "
            "montage rapide, livrée avec visserie."
        ),
        "expected": "1560",
    },
    {
        "designation": "Bâche de protection piscine ronde 4m avec œillets",
        "description": (
            "Bâche d'hivernage pour piscine hors-sol, polyéthylène 180 g/m², "
            "résistante aux UV, œillets renforcés tous les 50 cm."
        ),
        "expected": "2583",
    },
    {
        "designation": "Puzzle 1000 pièces paysage de montagne",
        "description": "Puzzle adulte 1000 pièces, dimensions 68 x 48 cm, poster inclus.",
        "expected": "1281",
    },
    {
        "designation": "Coffret 20 capsules café expresso intensité 9",
        "description": (
            "Capsules compatibles machines à expresso, café arabica torréfaction "
            "intense, boîte de 20 capsules."
        ),
        "expected": "1940",
    },
    {
        "designation": "Chaise haute évolutive bébé avec harnais 5 points",
        "description": (
            "Chaise haute pliable, hauteur réglable sur 6 positions, tablette "
            "amovible lavable, harnais de sécurité 5 points."
        ),
        "expected": "1320",
    },
    {
        "designation": "Perceuse visseuse sans fil 18V + 2 batteries et coffret",
        "description": (
            "Perceuse à percussion 18V, couple 45 Nm, mandrin auto-serrant 13 mm, "
            "deux batteries lithium-ion et chargeur rapide."
        ),
        "expected": "2585",
    },
    {
        "designation": "Figurine collector Dragon Ball Goku Super Saiyan 25 cm",
        "description": "",
        "expected": "1140",
    },
]

BATCH_SAMPLE_CSV: Final = """designation,description
Écouteurs bluetooth intra-auriculaires réduction de bruit,Autonomie 30h avec le boîtier de charge
Housse de couette 220x240 coton lavé terracotta,Parure 2 personnes avec deux taies d'oreiller
Nintendo Switch OLED console blanche,Console avec écran OLED 7 pouces et station d'accueil
Salon de jardin résine tressée 4 places,Table basse en verre trempé et coussins déhoussables
Album panini Ligue 1 saison complète,Classeur rigide avec 300 vignettes collées
"Gamelle inox antidérapante pour chien 1,5 L",Base en caoutchouc pour éviter le glissement
Livre de cuisine italienne 250 recettes,Édition reliée illustrée en couleurs
Aspirateur balai sans fil 22000 Pa,Autonomie 45 minutes et brosse motorisée
Maquette avion RC électrique envergure 1 m,"Kit à monter avec radiocommande 2,4 GHz"
Ramette papier A4 80g 500 feuilles,Papier blanc multiusage pour imprimante laser
"""
