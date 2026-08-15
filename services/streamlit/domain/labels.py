"""
Human-readable names for the 27 Rakuten ``prdtypecode`` values.

Rakuten never published the meaning of these codes — the dataset ships numbers
only. The names below were established by reading the actual catalogue: for
each code, the terms that appear far more often than in the rest of the corpus,
and a sample of real product titles. That evidence lives in
``assets/categories.json`` and is displayed next to each name in the app, so
the reading can be checked rather than taken on trust.

A few examples of what the evidence settles:

* ``1160`` is dominated by ``pokemon``, ``mtg``, ``panini``, ``foil``, ``rare``
  — collectible cards, not books as the codes' numbering might suggest.
* ``2905`` is 100 % filled descriptions around ``dlc``, ``telechargement``,
  ``extension`` — downloadable games.
* ``1301`` mixes darts (``flechette``, ``ailettes``, ``harrows``), billiards
  (``aramith``, ``bce``) and table football, hence its broader name.

The model itself only ever sees and returns the numeric code.
"""

from typing import Final

CATEGORY_NAMES: Final[dict[str, str]] = {
    "10": "Livres et guides",
    "40": "Jeux vidéo import",
    "50": "Accessoires gaming",
    "60": "Consoles de jeux",
    "1140": "Figurines et produits dérivés",
    "1160": "Cartes à collectionner",
    "1180": "Jeux de rôle et figurines de wargame",
    "1280": "Jouets et peluches",
    "1281": "Jeux de société et jeux éducatifs",
    "1300": "Modélisme et drones",
    "1301": "Jeux d'adresse et accessoires de jeu",
    "1302": "Jeux et loisirs de plein air",
    "1320": "Puériculture",
    "1560": "Mobilier d'intérieur",
    "1920": "Linge de maison",
    "1940": "Épicerie et boissons",
    "2060": "Décoration et luminaires",
    "2220": "Accessoires pour animaux",
    "2280": "Magazines et presse",
    "2403": "Lots de livres et revues",
    "2462": "Jeux vidéo d'occasion",
    "2522": "Papeterie et fournitures",
    "2582": "Mobilier et équipement de jardin",
    "2583": "Piscine et accessoires",
    "2585": "Outillage et jardinage",
    "2705": "Romans et littérature",
    "2905": "Jeux vidéo dématérialisés",
}

# Coarse families, used to filter the class-level tables and charts.
CATEGORY_FAMILIES: Final[dict[str, str]] = {
    "10": "Culture",
    "2280": "Culture",
    "2403": "Culture",
    "2705": "Culture",
    "40": "Gaming",
    "50": "Gaming",
    "60": "Gaming",
    "2462": "Gaming",
    "2905": "Gaming",
    "1140": "Loisirs",
    "1160": "Loisirs",
    "1180": "Loisirs",
    "1281": "Loisirs",
    "1300": "Loisirs",
    "1301": "Loisirs",
    "1280": "Enfance",
    "1302": "Enfance",
    "1320": "Enfance",
    "1560": "Maison",
    "1920": "Maison",
    "1940": "Maison",
    "2060": "Maison",
    "2220": "Maison",
    "2522": "Maison",
    "2582": "Jardin",
    "2583": "Jardin",
    "2585": "Jardin",
}


def category_name(code: str | int) -> str:
    """
    Return the readable name of a ``prdtypecode``.

    Args:
        code: Product type code, as a string or an integer.

    Returns:
        str: The category name, or ``"Catégorie <code>"`` for unknown codes.
    """
    key = str(code)
    return CATEGORY_NAMES.get(key, f"Catégorie {key}")


def category_family(code: str | int) -> str:
    """
    Return the coarse family a ``prdtypecode`` belongs to.

    Args:
        code: Product type code, as a string or an integer.

    Returns:
        str: Family name, or ``"Autre"`` for unknown codes.
    """
    return CATEGORY_FAMILIES.get(str(code), "Autre")


def category_label(code: str | int) -> str:
    """
    Return the display label combining the code and its name.

    Args:
        code: Product type code, as a string or an integer.

    Returns:
        str: For example ``"1560 · Mobilier d'intérieur"``.
    """
    return f"{code} · {category_name(code)}"
