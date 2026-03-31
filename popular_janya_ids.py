"""Resolve popular_janya_ragas entries to catalog raga_id strings."""

import json
import logging
import os

from raga_slug import slug_raga_id

logger = logging.getLogger(__name__)


def build_lookup_from_rows(rows: list[dict]) -> tuple[set[str], dict[str, str]]:
    """All known raga_ids plus maps from normalized name/alias keys to raga_id."""
    ids: set[str] = set()
    name_to_id: dict[str, str] = {}
    for row in rows:
        rid = (row.get("raga_id") or "").strip()
        if not rid:
            continue
        ids.add(rid)
        rn = (row.get("raga_name") or "").strip()
        if rn:
            name_to_id[rn.lower()] = rid
            name_to_id[slug_raga_id(rn)] = rid
        for alias in row.get("other_names") or []:
            if not isinstance(alias, str):
                continue
            a = alias.strip()
            if not a:
                continue
            name_to_id[a.lower()] = rid
            name_to_id[slug_raga_id(a)] = rid
    return ids, name_to_id


def load_json_rows_from_output_dir(output_dir: str) -> list[dict]:
    rows: list[dict] = []
    if not os.path.isdir(output_dir):
        return rows
    for fn in sorted(os.listdir(output_dir)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(output_dir, fn)
        try:
            with open(path, encoding="utf-8") as f:
                rows.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def _compact_slug(s: str) -> str:
    """Compare spellings that differ only by spaces/underscores (e.g. yadukula kambhoji → yadukulakamboji)."""
    return slug_raga_id(s).replace("_", "").replace("-", "")


def resolve_popular_janya_entry(entry: str, ids: set[str], name_to_id: dict[str, str]) -> str | None:
    e = (entry or "").strip()
    if not e:
        return None
    if e in ids:
        return e
    el = e.lower()
    if el in name_to_id:
        return name_to_id[el]
    slug = slug_raga_id(e)
    if slug in ids:
        return slug
    for key, rid in name_to_id.items():
        if slug_raga_id(key) == slug:
            return rid
    # "Yadukula kambhoji" vs raga_id yadukulakamboji
    c = _compact_slug(e)
    for rid in ids:
        if _compact_slug(rid) == c:
            return rid
    return None


def normalize_popular_janya_list(
    entries: list[str],
    ids: set[str],
    name_to_id: dict[str, str],
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for e in entries:
        rid = resolve_popular_janya_entry(e, ids, name_to_id)
        if rid and rid not in seen:
            seen.add(rid)
            out.append(rid)
        elif rid is None and (e or "").strip():
            logger.warning("popular_janya_ragas: could not resolve to raga_id: %r", e)
    return out
