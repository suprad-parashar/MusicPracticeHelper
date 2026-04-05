"""
Web search for Carnatic songs: Google Custom Search JSON API (optional) and DuckDuckGo (ddgs).

Set GOOGLE_API_KEY + GOOGLE_CSE_ID (Programmable Search Engine cx) for Google.
Without them, searches use ddgs (often similar coverage; include site: filters for karnatik/shivkumar).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


def _google_cse_search(query: str, num: int = 10) -> list[dict[str, Any]]:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_CUSTOM_SEARCH_API_KEY")
    cx = os.environ.get("GOOGLE_CSE_ID") or os.environ.get("GOOGLE_SEARCH_ENGINE_ID")
    if not key or not cx:
        return []

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": key,
        "cx": cx,
        "q": query,
        "num": min(max(num, 1), 10),
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    items = data.get("items") or []
    out: list[dict[str, Any]] = []
    for it in items:
        out.append({
            "title": it.get("title", ""),
            "url": it.get("link", ""),
            "snippet": (it.get("snippet") or "")[:500],
        })
    return out


def _ddgs_search(query: str, max_results: int = 8) -> list[dict[str, Any]]:
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("ddgs not installed (pip install ddgs), skipping web search")
        return []

    out: list[dict[str, Any]] = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            for r in results:
                out.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": (r.get("body", "") or "")[:500],
                })
    except Exception as exc:
        logger.debug("ddgs search failed for %r: %s", query, exc)
    time.sleep(0.35)
    return out


def search_web(query: str, *, max_results: int = 8) -> list[dict[str, Any]]:
    """Prefer Google CSE when configured; otherwise DuckDuckGo."""
    g = _google_cse_search(query, num=min(max_results, 10))
    if g:
        return g[:max_results]
    return _ddgs_search(query, max_results=max_results)


def build_song_search_queries(song_query: str) -> list[str]:
    """Queries for general + site-specific discovery."""
    q = song_query.strip()
    return [
        f'{q} Carnatic kriti lyrics',
        f'site:karnatik.com {q}',
        f'site:shivkumar.org {q}',
        f'{q} Tyagaraja kriti raga tala',
        f'{q} Carnatic composition notation',
    ]


def search_song_multi(song_query: str, *, per_query: int = 5) -> tuple[str, list[dict[str, Any]]]:
    """
    Run several searches, merge unique URLs, return (combined_snippet_text, result_dicts).
    """
    seen: set[str] = set()
    combined: list[str] = []
    all_results: list[dict[str, Any]] = []

    for q in build_song_search_queries(song_query):
        hits = search_web(q, max_results=per_query)
        for h in hits:
            url = (h.get("url") or "").strip()
            key = url or h.get("title", "")
            if key in seen:
                continue
            seen.add(key)
            title = h.get("title", "")
            snip = h.get("snippet", "")
            combined.append(f"[{title}] ({url})\n{snip}")
            all_results.append({"title": title, "url": url, "snippet": snip})
        time.sleep(0.2)

    return "\n\n".join(combined), all_results
