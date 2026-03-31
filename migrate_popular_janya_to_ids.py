#!/usr/bin/env python3
"""
One-shot migration: popular_janya_ragas from display names to raga_id slugs.
Ongoing saves also normalize via save_raga_info() → popular_janya_ids.normalize_popular_janya_list.

Usage (from MusicPracticeHelper repo root):
  python3 migrate_popular_janya_to_ids.py
  python3 migrate_popular_janya_to_ids.py --web ../Web/MusicPractice/data/all_ragas.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from popular_janya_ids import build_lookup_from_rows, normalize_popular_janya_list, resolve_popular_janya_entry

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def migrate_all_ragas(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    ids, name_to_id = build_lookup_from_rows(rows)
    changed = 0
    for row in rows:
        if not row.get("is_melakarta"):
            continue
        pj = row.get("popular_janya_ragas") or []
        if not pj:
            continue
        before = list(pj)
        for b in before:
            if (b or "").strip() and resolve_popular_janya_entry(b, ids, name_to_id) is None:
                logger.warning(
                    "Unmapped popular_janya %r (melakarta %s)",
                    b,
                    row.get("raga_name"),
                )
        new_pj = normalize_popular_janya_list(pj, ids, name_to_id)
        if new_pj != before:
            row["popular_janya_ragas"] = new_pj
            changed += 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.info("Wrote %s (%d melakarta rows with popular_janya updated)", path, changed)
    return changed


def migrate_output_dir(output_dir: str, catalog_rows: list[dict]) -> int:
    ids, name_to_id = build_lookup_from_rows(catalog_rows)
    n_files = 0
    for fn in sorted(os.listdir(output_dir)):
        if not fn.endswith(".json"):
            continue
        fp = os.path.join(output_dir, fn)
        try:
            with open(fp, encoding="utf-8") as f:
                row = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not row.get("is_melakarta"):
            continue
        pj = row.get("popular_janya_ragas") or []
        if not pj:
            continue
        before = list(pj)
        new_pj = normalize_popular_janya_list(pj, ids, name_to_id)
        if new_pj == before:
            continue
        row["popular_janya_ragas"] = new_pj
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, ensure_ascii=False)
            f.write("\n")
        n_files += 1
    logger.info("Updated %d files under %s/", n_files, output_dir)
    return n_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate popular_janya_ragas to raga_id strings")
    parser.add_argument("--all-ragas", default="all_ragas.json", help="Bundled catalog JSON")
    parser.add_argument("--output-dir", default="output", help="Per-raga JSON directory")
    parser.add_argument("--web", default="", help="Optional second all_ragas.json (e.g. Web app)")
    args = parser.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    all_path = args.all_ragas if os.path.isabs(args.all_ragas) else os.path.join(root, args.all_ragas)
    migrate_all_ragas(all_path)

    with open(all_path, encoding="utf-8") as f:
        catalog = json.load(f)

    out_path = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(root, args.output_dir)
    if os.path.isdir(out_path):
        migrate_output_dir(out_path, catalog)

    if args.web:
        web_path = args.web if os.path.isabs(args.web) else os.path.join(root, args.web)
        migrate_all_ragas(web_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
