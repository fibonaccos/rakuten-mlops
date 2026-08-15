"""
Human-readable names for the 27 Rakuten ``prdtypecode`` values.

The dataset ships numeric codes only. Rakuten never published an official
mapping, so the wording below is the reading of the categories the team agreed
on after browsing the products of each code. It exists to make the demo
readable — the model itself only ever sees and returns the numeric code.
"""

from typing import Final

CATEGORY_NAMES: Final[dict[str, str]] = {
    "10": "Livres d'occasion",
    "40": "Jeux vidéo neufs",
    "50": "Accessoires gaming",
    "60": "Consoles de jeux",
    "1140": "Figurines et produits dérivés",
    "1160": "Cartes à collectionner",
    "1180": "Jeux de rôle et univers fantastiques",
    "1280": "Jouets pour enfants",
    "1281": "Jeux de société",
    "1300": "Modélisme et véhicules télécommandés",
    "1301": "Vêtements et accessoires bébé",
    "1302": "Jeux et activités de plein air",
    "1320": "Puériculture",
    "1560": "Mobilier d'intérieur",
    "1920": "Linge de maison",
    "1940": "Épicerie et alimentation",
    "2060": "Décoration d'intérieur",
    "2220": "Accessoires pour animaux",
    "2280": "Magazines et revues",
    "2403": "Lots de livres et magazines",
    "2462": "Jeux vidéo d'occasion",
    "2522": "Papeterie et fournitures",
    "2582": "Mobilier de jardin",
    "2583": "Piscine et spa",
    "2585": "Outillage et bricolage",
    "2705": "Livres neufs",
    "2905": "Jeux vidéo dématérialisés",
}

# Coarse families, used to colour charts and group the class-level tables.
CATEGORY_FAMILIES: Final[dict[str, str]] = {
    "10": "Culture",
    "40": "Gaming",
    "50": "Gaming",
    "60": "Gaming",
    "1140": "Loisirs",
    "1160": "Loisirs",
    "1180": "Loisirs",
    "1280": "Enfance",
    "1281": "Loisirs",
    "1300": "Loisirs",
    "1301": "Enfance",
    "1302": "Enfance",
    "1320": "Enfance",
    "1560": "Maison",
    "1920": "Maison",
    "1940": "Maison",
    "2060": "Maison",
    "2220": "Maison",
    "2280": "Culture",
    "2403": "Culture",
    "2462": "Gaming",
    "2522": "Maison",
    "2582": "Jardin",
    "2583": "Jardin",
    "2585": "Jardin",
    "2705": "Culture",
    "2905": "Gaming",
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
