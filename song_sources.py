"""
Fetch and parse Carnatic song metadata and notation from shivkumar.org
and lyrics pages from karnatik.com (URLs discovered via web search).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

logger = logging.getLogger(__name__)

SHIVKUMAR_BASE = "https://www.shivkumar.org/music/"
SHIVKUMAR_INDEX = "https://www.shivkumar.org/music/index.html"
KARNATIK_NETLOC = "karnatik.com"

USER_AGENT = os.environ.get(
    "HTTP_USER_AGENT",
    "MusicPracticeHelper/1.0 (Carnatic song research; Python; +https://github.com/)",
)


@dataclass
class ShivkumarSongRow:
    """One krithi row from the Shivkumar index."""

    title: str
    raga: str
    raga_url: str
    tala: str
    composer_line: str
    notation_html_url: str = ""
    notation_doc_url: str = ""
    notation_pdf_url: str = ""
    mp3_url: str = ""
    detail_mp3_url: str = ""

    def all_urls(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for label, u in [
            ("notation_html", self.notation_html_url),
            ("notation_pdf", self.notation_pdf_url),
            ("notation_doc", self.notation_doc_url),
            ("audio_mp3", self.mp3_url),
            ("lesson_mp3", self.detail_mp3_url),
            ("raga_anchor", self.raga_url),
        ]:
            if u:
                out.append((label, u))
        return out


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
    return s


def fetch_url(sess: requests.Session, url: str, timeout: float = 45.0) -> str:
    r = sess.get(url, timeout=timeout)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def html_to_plain_text(html: str, max_chars: int | None = None) -> str:
    """Strip tags and scripts; preserve some newlines for notation."""
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>|<div|<tr|<h[1-6]", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = text.strip()
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars] + "\n... [TRUNCATED]"
    return text


HEADER = re.compile(
    r"<b>\s*(?P<title>[^:<]+):\s*"
    r'<a\s+href="(?P<raga_href>[^"]+)"[^>]*>\s*(?P<raga>[^<]+?)\s*</a>\s*;\s*'
    r"(?P<tala>[^;]+);\s*(?P<rest>.*?)\s*</b>",
    re.IGNORECASE | re.DOTALL,
)


def _abs_shivkumar(href: str) -> str:
    return urljoin(SHIVKUMAR_BASE, href.strip())


def _find_href(block: str, suffix: str) -> str:
    m = re.search(rf'href="([^"]+{re.escape(suffix)})"', block, re.IGNORECASE)
    if not m:
        return ""
    return _abs_shivkumar(m.group(1))


def parse_shivkumar_index_html(html: str) -> list[ShivkumarSongRow]:
    """Parse index HTML. Many legacy pages use `<li>` without a closing `</li>`."""
    rows: list[ShivkumarSongRow] = []
    parts = re.split(r"(?i)<li[^>]*>", html)
    for body in parts:
        if "Notation" not in body and ".htm" not in body.lower():
            continue
        hm = HEADER.search(body)
        if not hm:
            continue
        title = re.sub(r"\s+", " ", hm.group("title")).strip()
        raga = re.sub(r"\s+", " ", hm.group("raga")).strip()
        raga_url = _abs_shivkumar(hm.group("raga_href"))
        tala = re.sub(r"\s+", " ", hm.group("tala")).strip()
        rest = re.sub(r"\s+", " ", hm.group("rest")).strip()

        row = ShivkumarSongRow(
            title=title,
            raga=raga,
            raga_url=raga_url,
            tala=tala,
            composer_line=rest,
            notation_html_url=_find_href(body, ".htm") or _find_href(body, ".html"),
            notation_doc_url=_find_href(body, ".doc"),
            notation_pdf_url=_find_href(body, ".pdf"),
            mp3_url="",
            detail_mp3_url="",
        )
        mp3s = re.findall(r'href="([^"]+\.mp3)"', body, re.IGNORECASE)
        if len(mp3s) >= 2:
            row.mp3_url = _abs_shivkumar(mp3s[0])
            row.detail_mp3_url = _abs_shivkumar(mp3s[1])
        elif mp3s:
            row.mp3_url = _abs_shivkumar(mp3s[0])
        rows.append(row)
    return rows


def load_or_build_shivkumar_index(
    cache_path: Path,
    *,
    force_refresh: bool = False,
    delay_s: float = 0.0,
) -> list[ShivkumarSongRow]:
    """Load parsed index from JSON cache, or fetch and parse Shivkumar krithi index."""
    cache_path = cache_path.resolve()
    if not force_refresh and cache_path.is_file():
        with open(cache_path, encoding="utf-8") as f:
            raw = json.load(f)
        return [ShivkumarSongRow(**row) for row in raw]

    sess = session()
    if delay_s > 0:
        time.sleep(delay_s)
    html = fetch_url(sess, SHIVKUMAR_INDEX)
    rows = parse_shivkumar_index_html(html)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.info("Shivkumar index: parsed %s rows → %s", len(rows), cache_path)
    return rows


def is_karnatik_url(url: str) -> bool:
    try:
        return KARNATIK_NETLOC in urlparse(url).netloc.lower()
    except OSError:
        return False


def fetch_karnatik_lyrics_text(sess: requests.Session, url: str, max_chars: int) -> str:
    html = fetch_url(sess, url)
    return html_to_plain_text(html, max_chars=max_chars)


def fetch_shivkumar_notation_text(sess: requests.Session, url: str, max_chars: int) -> str:
    html = fetch_url(sess, url)
    return html_to_plain_text(html, max_chars=max_chars)
