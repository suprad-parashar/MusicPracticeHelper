"""Normalization, transliteration-style variants, and fuzzy matching for song titles."""

from __future__ import annotations

import difflib
import re
import unicodedata


def strip_diacritics(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def slug_song_id(name: str) -> str:
    s = strip_diacritics(name)
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def normalize_for_match(text: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace and common punctuation."""
    s = strip_diacritics(text).lower()
    s = re.sub(r"[\s\-_./,;:'\"]+", "", s)
    return s


TRANSLITERATION_SWAPS: list[tuple[str, str]] = [
    ("dw", "dv"),
    ("dv", "dw"),
    ("w", "v"),
    ("v", "w"),
    ("th", "t"),
    ("dh", "d"),
    ("bh", "b"),
    ("sh", "s"),
    ("zh", "j"),
    ("ch", "c"),
    ("ee", "i"),
    ("oo", "u"),
    ("aa", "a"),
]


def transliteration_variants(name: str) -> list[str]:
    """Common spelling alternates for Carnatic song / transliterated titles."""
    variants: set[str] = set()
    ascii_form = strip_diacritics(name)
    if ascii_form != name:
        variants.add(ascii_form.strip())

    base = strip_diacritics(name).lower()
    for old, new in TRANSLITERATION_SWAPS:
        if old in base:
            v = base.replace(old, new)
            variants.add(" ".join(w.capitalize() for w in v.split()))
    # Space / hyphen toggles
    if " " in name:
        variants.add(name.replace(" ", ""))
    else:
        # optional split on camelCase not needed for these titles
        pass
    variants.discard(name)
    return [v for v in variants if v]


def search_keys_for_query(query: str) -> list[str]:
    """All strings to score against index titles (query + normalized + variants)."""
    keys: list[str] = [query.strip()]
    keys.append(strip_diacritics(query).strip())
    keys.extend(transliteration_variants(query))
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        k = k.strip()
        if not k:
            continue
        low = k.lower()
        if low not in seen:
            seen.add(low)
            out.append(k)
    return out


def tokenize_swara_notation_line(line: str) -> list[str]:
    """
    Split a swara line into tokens in order, preserving structural punctuation.

    Semicolons (;) mark ties between notes; vertical bars (|) often mark beats or
    barlines. They must appear as their own tokens when the source uses them
    (Shivkumar-style lines often look like: ``M ; G S | S ; ; S |``).
    """
    s = (line or "").strip()
    if not s:
        return []
    # Isolate tie (;), bar (|), comma (phrase separators in some scores) when glued to swaras
    s = re.sub(r"([;|,])", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [t for t in s.split() if t]


def fuzzy_best_matches(
    query: str,
    candidates: list[str],
    *,
    limit: int = 8,
    score_cutoff: float = 0.42,
) -> list[tuple[str, float]]:
    """
    Return (candidate, ratio) sorted by similarity to query.
    Uses difflib on normalized strings; tolerates minor spelling differences.
    """
    if not candidates:
        return []
    keys = search_keys_for_query(query)
    best: dict[str, float] = {}
    for cand in candidates:
        cn = normalize_for_match(cand)
        if not cn:
            continue
        score = 0.0
        for q in keys:
            qn = normalize_for_match(q)
            if not qn:
                continue
            if qn in cn or cn in qn:
                score = max(score, 0.95)
                break
            r = difflib.SequenceMatcher(None, qn, cn).ratio()
            score = max(score, r)
        if score >= score_cutoff:
            best[cand] = max(best.get(cand, 0.0), score)

    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return ranked[:limit]
