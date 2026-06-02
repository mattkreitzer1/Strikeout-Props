from __future__ import annotations

import re
import unicodedata


def normalize_name_key(name: str) -> str:
    """Lowercase ASCII key for fuzzy joins (accents stripped, punctuation removed)."""
    text = unicodedata.normalize("NFKD", str(name).strip())
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9, ]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def mlb_full_name_to_savant(full_name: str) -> str:
    """MLB `fullName` (First Last) -> Savant `Last, First`."""
    parts = str(full_name).strip().split()
    if len(parts) < 2:
        return str(full_name).strip()
    last = parts[-1]
    first = " ".join(parts[:-1])
    return f"{last}, {first}"


def match_player_name(
    name: str,
    candidates: list[str],
) -> str | None:
    """
    Return the single matching candidate name, or None if ambiguous / missing.

    Tries exact match, then normalized key, then last-name + first-initial.
    """
    name = str(name).strip()
    if not name:
        return None

    if name in candidates:
        return name

    key = normalize_name_key(name)
    by_key = {normalize_name_key(c): c for c in candidates}
    if key in by_key:
        return by_key[key]

    if "," in name:
        last = name.split(",", maxsplit=1)[0].strip()
        first_bit = name.split(",", maxsplit=1)[1].strip()[:1]
    else:
        parts = name.split()
        last = parts[-1] if parts else ""
        first_bit = parts[0][:1] if parts else ""

    last_key = normalize_name_key(last)
    matches = [
        c
        for c in candidates
        if normalize_name_key(c).startswith(last_key)
        and (not first_bit or normalize_name_key(c).endswith(first_bit))
    ]
    if len(matches) == 1:
        return matches[0]
    return None
