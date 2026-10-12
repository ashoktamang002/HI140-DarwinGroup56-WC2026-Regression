"""One canonical team name per country, whichever source the name came from
(FBref fixtures, FBref team-file names, FIFA ranking)."""
import re
import unicodedata

CANONICAL = {
    # FBref / FIFA spelling            -> canonical name used in this project
    "Korea Republic": "South Korea", "Czechia": "Czech Republic",
    "Bosnia–Herz": "Bosnia and Herzegovina", "USA": "United States",
    "Türkiye": "Turkey", "Turkiye": "Turkey", "Côte d'Ivoire": "Ivory Coast",
    "Côte d’Ivoire": "Ivory Coast", "Cote dIvoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde", "IR Iran": "Iran", "Congo DR": "DR Congo",
    "Curacao": "Curaçao", "Rep. of Ireland": "Republic of Ireland",
}


def _key(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


_BY_KEY = {_key(k): v for k, v in CANONICAL.items()}


def canonical(name: str) -> str:
    name = unicodedata.normalize("NFC", str(name)).strip()
    return CANONICAL.get(name, _BY_KEY.get(_key(name), name))
