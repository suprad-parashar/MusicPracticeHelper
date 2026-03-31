"""ASCII slug for raga_id (shared by extraction, uniqueness, popular-janya resolution)."""

import re
import unicodedata


def slug_raga_id(raga_name: str) -> str:
    """Stable ASCII slug for raga_id (matches template raga_id)."""
    s = unicodedata.normalize("NFKD", raga_name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")
