"""
LLM extraction for Carnatic compositions: merge Shivkumar notation, karnatik lyrics,
and web snippets into structured JSON.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError, RateLimitError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from raga_extractor import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MODEL,
    MAX_RETRIES,
    _parse_retry_after,
    _sleep_with_progress,
    _trim_context,
    get_pacer,
)
from song_utils import slug_song_id, tokenize_swara_notation_line

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)


class SourceRef(BaseModel):
    """Explicit schema for OpenAI structured outputs (plain dict[str,str] is rejected in strict mode)."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(default="", description="Short label, e.g. shivkumar_notation_html")
    url: str = Field(default="", description="Source URL")


class SongPhraseLine(BaseModel):
    """One sahitya line with swara notation (phrase-level practice). Matches song_template.json lines[]."""

    model_config = ConfigDict(extra="forbid")

    notes: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered swara tokens: syllables plus structural symbols. "
            "Semicolon (;) = tie — MUST appear as its own token when present in the source. "
            "Bar (|) likewise. Do not drop ; or | when transcribing from notation."
        ),
    )
    lyrics: str = Field(default="", description="Sahitya for this phrase")
    translation: str = Field(default="", description="English or other gloss when known")

    @field_validator("notes", mode="before")
    @classmethod
    def _coerce_notes(cls, v: object) -> list[str]:
        """Allow legacy JSON: single notation string is split with ties (;) and bars (|) preserved."""
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        if isinstance(v, str):
            s = v.strip()
            return [] if not s else tokenize_swara_notation_line(s)
        return []


class SongStanza(BaseModel):
    """A section (pallavi / anupallavi / charanam) or a single scored block."""

    model_config = ConfigDict(extra="forbid")

    section: str = Field(
        default="",
        description="Section name: pallavi, anupallavi, charanam, or stanza label",
    )
    raga_id: str = Field(default="", description="Catalog raga_id slug, e.g. hindolam, malahari")
    tala: str = Field(default="", description="Tala as in your catalog, e.g. Adi, 4-Rupaka")
    tempo: int = Field(default=0, description="Practice tempo in BPM; 0 if unknown")
    lines: list[SongPhraseLine] = Field(default_factory=list, description="Phrase-by-phrase notes + lyrics + translation")


class SongInfo(BaseModel):
    """Canonical song record; must match song_template.json (same keys and nesting)."""

    model_config = ConfigDict(extra="forbid")

    song_id: str = Field(default="", description="ASCII slug; pipeline fills if empty")
    title: str = Field(default="", description="Primary song title as commonly known")
    other_titles: list[str] = Field(default_factory=list, description="Alternate spellings / names")
    composer: str = Field(default="", description="Composer name")
    raga: str = Field(default="", description="Raga name")
    tala: str = Field(default="", description="Tala name")
    language: str = Field(default="", description="Sahitya language if known")

    lyrics_full_text: str = Field(default="", description="Full lyrics if present in sources")
    lyrics_source_url: str = Field(default="", description="Primary lyrics page URL (e.g. karnatik)")

    notation_url: str = Field(
        default="",
        description="Primary notation page URL (e.g. shivkumar .htm); PDF/DOC links may appear in sources",
    )

    youtube_url: str = Field(
        default="",
        description="Primary YouTube watch URL for a Carnatic reference performance of this composition",
    )

    stanzas: list[SongStanza] = Field(
        default_factory=list,
        description="Structured phrase lines: swara + sahitya + translation per line, grouped by section",
    )

    sources: list[SourceRef] = Field(
        default_factory=list,
        description="Every page used: label + url",
    )


_YOUTUBE_VIDEO_ID = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|music\.youtube\.com/watch\?v=)([a-zA-Z0-9_-]{6,})",
)


def enrich_youtube_fields(info: SongInfo) -> SongInfo:
    """Normalize youtube_url to https://www.youtube.com/watch?v=VIDEO_ID when a video id is present."""
    u0 = (info.youtube_url or "").strip()
    if not u0:
        return info
    m = _YOUTUBE_VIDEO_ID.search(u0)
    if not m:
        return info
    canonical = f"https://www.youtube.com/watch?v={m.group(1)}"
    return info.model_copy(update={"youtube_url": canonical}) if u0 != canonical else info


SONG_SYSTEM_PROMPT = """You are an expert in Carnatic music. You receive raw text from:
- Shivkumar.org notation pages (often Word-exported HTML with swara/sahitya layout)
- karnATik.com lyric pages
- Web search snippets (Google/DuckDuckGo) about the composition

Extract structured information about ONE Carnatic composition.

Output MUST match the project's song_template.json shape exactly:
- Top-level keys only: song_id, title, other_titles, composer, raga, tala, language, lyrics_full_text, lyrics_source_url, notation_url, youtube_url, stanzas, sources.
- stanzas[]: section, raga_id, tala, tempo, lines[] where each line has notes (JSON array of tokens), lyrics, translation.
- sources[]: label and url for every distinct page you used (e.g. karnatik_lyrics, shivkumar_notation_html).

Rules:
- Fill fields ONLY from the provided text; do not invent lyrics or notation.
- If lyrics appear in the sources, copy them faithfully into lyrics_full_text (fix obvious HTML artifacts only).
- Preserve alternate spellings in other_titles when the sources show variants.
- notation_url: set to the primary Shivkumar notation .htm/.html URL when present in the text; otherwise leave empty and put PDF/DOC links in sources with clear labels.
- lyrics_source_url: primary karnatik (or other) lyrics page URL when present.
- youtube_url: If sources include a YouTube link for this classical kriti (not unrelated film songs), set it to the watch URL.
- stanzas / notes[]: Transcribe swara EXACTLY as in the notation source. The semicolon (;) indicates a TIE — it is musically essential. Include ``;`` and ``|`` (bar/beat) as their own string tokens in order (e.g. ``["M",";","G","S","N","D","N","N","|","S",";",";","S","|","M","G","M",";"]``). Never omit ``;`` or ``|`` when they appear in the source. One array element per swara syllable OR per standalone symbol. Use [] for notes when unknown. Otherwise leave stanzas empty.
"""


# Phase 2: gap analysis + DuckDuckGo web search (same machinery as raga_extractor.identify_gaps / web_search)
SONG_GAP_SEARCH_TEMPLATES: dict[str, list[str]] = {
    "lyrics_full_text": [
        'site:karnatik.com "{title}" lyrics',
        "{title} {composer_context} Carnatic kriti lyrics full sahitya",
        "{title} kriti pallavi anupallavi charanam lyrics text",
    ],
    "metadata": [
        "{title} Carnatic kriti composer raga tala language",
        "{title} kriti ragam talam",
        "{title} {composer_context} kriti raga",
    ],
    "other_titles": [
        "{title} kriti alternate spellings transliteration names",
        "{title} Carnatic song title variants",
    ],
    "lyrics_source_url": [
        'site:karnatik.com "{title}"',
        "{title} kriti lyrics karnatik.com page",
    ],
    "notation_url": [
        'site:shivkumar.org "{title}"',
        "{title} {composer_context} kriti notation swaras",
        "{title} Carnatic notation shivkumar HTML",
    ],
    "youtube_url": [
        "{title} {composer_context} kriti Carnatic classical YouTube",
        "{title} kriti vocal concert performance YouTube",
    ],
    "stanzas": [
        'site:shivkumar.org "{title}" swara sahitya',
        "{title} kriti phrase swara notation lines",
    ],
}

SONG_GAP_MAX_QUERIES = 28

SONG_SUPPLEMENT_PROMPT = """You are an expert in Carnatic music. You have an EXISTING song JSON (song_template.json shape) for ONE composition.
Some fields are empty or incomplete. You are given ADDITIONAL web search snippets (DuckDuckGo / similar) that may contain the missing information.

Your task: return a COMPLETE, UPDATED SongInfo JSON with the same keys as before. Match song_template.json: top-level fields only; stanzas use section, raga_id, tala, tempo, lines with notes (array of swara strings), lyrics, translation; sources are objects with label and url.

PRIORITY — fill empty fields when the snippets clearly support them:
- composer, raga, tala, language: factual metadata from reputable pages (karnatik, shivkumar, university/CM sites).
- lyrics_full_text: copy full sahitya from snippets or implied page content only when quoted or clearly extractable; never invent lines.
- lyrics_source_url: karnatik (or other) lyrics page URL if it appears in the snippets.
- notation_url: primary Shivkumar .htm/.html or similar notation page URL if present.
- youtube_url: a YouTube watch URL for a classical Carnatic performance OF THIS kriti (not film, not unrelated); must match the composition.
- other_titles: alternate spellings / transliterations when sources list them.
- stanzas: only if snippets or context give clear phrase-by-phrase swara+sahitya alignment; each line's notes must be a JSON array of tokens preserving ``;`` (ties) and ``|`` (bars) as separate elements when present in snippets. Otherwise leave stanzas empty. Do not invent swara lines.
- sources: add any new page URLs you used from the snippets (short labels like karnatik_lyrics, gap_fill_web).

Rules:
- KEEP all existing data that is already correct — do not remove or weaken it.
- Do NOT fabricate lyrics, notation, or URLs. If a field cannot be filled from the snippets, leave it as-is.
- Prefer authoritative Carnatic sites (karnatik.com, shivkumar.org, reputable artists/teachers) over random blogs when choosing URLs.
"""


def _dedupe_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        q = q.strip()
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(q)
    return out


def _scan_song_id_paths(output_dir: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not os.path.isdir(output_dir):
        return out
    for fn in os.listdir(output_dir):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(output_dir, fn)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            sid = (data.get("song_id") or "").strip()
            if sid:
                out.setdefault(sid, []).append(path)
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _signature(data: dict) -> tuple:
    return (
        (data.get("title") or "").strip().lower(),
        (data.get("composer") or "").strip().lower(),
        (data.get("raga") or "").strip().lower(),
    )


def resolve_unique_song_id(info: SongInfo, output_dir: str) -> SongInfo:
    base = (info.song_id or "").strip() or slug_song_id(info.title)
    rid_map = _scan_song_id_paths(output_dir)
    our_sig = (
        info.title.strip().lower(),
        info.composer.strip().lower(),
        info.raga.strip().lower(),
    )

    def available(sid: str) -> bool:
        paths = rid_map.get(sid, [])
        if not paths:
            return True
        for p in paths:
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                if _signature(data) != our_sig:
                    return False
            except (OSError, json.JSONDecodeError):
                return False
        return True

    if available(base):
        return info.model_copy(update={"song_id": base})

    comp = slug_song_id(info.composer)[:24] if info.composer else ""
    raga = slug_song_id(info.raga)[:20] if info.raga else ""
    for cand in [f"{base}_{comp}" if comp else "", f"{base}_{raga}" if raga else ""]:
        cand = cand.strip("_") or base
        if available(cand):
            return info.model_copy(update={"song_id": cand})

    n = 2
    while n < 500:
        cand = f"{base}_{n}"
        if available(cand):
            return info.model_copy(update={"song_id": cand})
        n += 1
    return info.model_copy(update={"song_id": base})


def extract_song_info(
    user_query: str,
    context: str,
    *,
    model: str | None = None,
    extra_user_notes: str | None = None,
) -> SongInfo:
    if model is None:
        model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)

    max_context = int(os.environ.get("MAX_CONTEXT_CHARS", DEFAULT_MAX_CONTEXT_CHARS))
    context = _trim_context(context, max_context)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=0)

    def _build_messages(ctx: str) -> list[dict]:
        u = f"User asked about the Carnatic composition: {user_query!r}\n\n"
        if extra_user_notes and extra_user_notes.strip():
            u += (
                "--- USER-PROVIDED FACTS (authoritative when conflicting) ---\n"
                f"{extra_user_notes.strip()}\n--- END ---\n\n"
            )
        u += f"--- AGGREGATED SOURCES ---\n{ctx}\n--- END SOURCES ---"
        return [
            {"role": "system", "content": SONG_SYSTEM_PROMPT},
            {"role": "user", "content": u},
        ]

    messages = _build_messages(context)
    pacer = get_pacer()
    trimmed_once = False

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            pacer.wait_if_needed()
            response = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=SongInfo,
                temperature=0.1,
            )
            if hasattr(response, "_raw_response") and hasattr(response._raw_response, "headers"):
                pacer.update_from_headers(dict(response._raw_response.headers))

            info = response.choices[0].message.parsed
            if not (info.title or "").strip():
                info = info.model_copy(update={"title": user_query.strip()})
            return info

        except RateLimitError as e:
            msg = str(e)
            if "Limit" in msg and "Requested" in msg:
                m = re.search(r"Limit\s+(\d+).*?Requested\s+(\d+)", msg)
                if m and int(m.group(2)) > int(m.group(1)) and not trimmed_once:
                    new_max = max(len(context) // 2, 8000)
                    context = _trim_context(context, new_max)
                    messages = _build_messages(context)
                    trimmed_once = True
                    continue
            retry_after = _parse_retry_after(e) or min(30 * (2 ** (attempt - 1)), 300)
            if attempt == MAX_RETRIES:
                raise
            _sleep_with_progress(retry_after + 5.0, f"Rate limited — retry {attempt}/{MAX_RETRIES}")

        except (APITimeoutError, APIConnectionError) as e:
            backoff = min(10 * (2 ** (attempt - 1)), 120)
            if attempt == MAX_RETRIES:
                raise
            logger.warning("API issue: %s; retry in %ss", e, backoff)
            _sleep_with_progress(backoff, f"Connection — retry {attempt}/{MAX_RETRIES}")

        except APIError as e:
            if e.status_code and e.status_code >= 500:
                backoff = min(15 * (2 ** (attempt - 1)), 120)
                if attempt == MAX_RETRIES:
                    raise
                _sleep_with_progress(backoff, f"Server {e.status_code} — retry {attempt}/{MAX_RETRIES}")
            else:
                raise

    raise RuntimeError("extract_song_info: exhausted retries")


def identify_song_gaps(info: SongInfo) -> tuple[list[str], list[str]]:
    """Identify empty or thin fields and return (gap_names, search_queries), mirroring raga_extractor.identify_gaps."""
    title = (info.title or "").strip() or "composition"
    composer = (info.composer or "").strip()
    composer_context = composer if composer else "Carnatic"

    gaps: list[str] = []
    queries: list[str] = []

    def add(field_key: str, label: str) -> None:
        gaps.append(label)
        for tmpl in SONG_GAP_SEARCH_TEMPLATES.get(field_key, []):
            queries.append(
                tmpl.format(title=title, composer_context=composer_context)
            )

    lyrics_thin = not info.lyrics_full_text or len(info.lyrics_full_text.strip()) < 40
    metadata_incomplete = not (
        (info.composer or "").strip()
        and (info.raga or "").strip()
        and (info.tala or "").strip()
        and (info.language or "").strip()
    )

    if lyrics_thin:
        add("lyrics_full_text", "lyrics_full_text")

    if metadata_incomplete:
        add("metadata", "composer/raga/tala/language")

    if not info.other_titles and (lyrics_thin or metadata_incomplete):
        add("other_titles", "other_titles")

    if not (info.lyrics_source_url or "").strip():
        add("lyrics_source_url", "lyrics_source_url")

    if not (info.notation_url or "").strip():
        add("notation_url", "notation_url")

    if not (info.youtube_url or "").strip():
        add("youtube_url", "youtube_url")

    # Phrase-level structure is rare from snippets; only search when we already have a notation page but no stanzas.
    if not info.stanzas and (info.notation_url or "").strip():
        add("stanzas", "stanzas")

    return gaps, _dedupe_queries(queries)[:SONG_GAP_MAX_QUERIES]


def fill_song_gaps(
    info: SongInfo,
    additional_context: str,
    *,
    model: str | None = None,
    extra_user_notes: str | None = None,
) -> SongInfo:
    if model is None:
        model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)

    max_c = int(os.environ.get("MAX_CONTEXT_CHARS", DEFAULT_MAX_CONTEXT_CHARS))
    if len(additional_context) > max_c:
        additional_context = additional_context[:max_c]

    existing = json.dumps(info.model_dump(), indent=2, ensure_ascii=False)
    user = ""
    if extra_user_notes and extra_user_notes.strip():
        user += "USER FACTS (authoritative):\n" + extra_user_notes.strip() + "\n\n"
    user += (
        f"Existing JSON:\n```json\n{existing}\n```\n\n"
        f"--- MORE WEB SNIPPETS ---\n{additional_context}\n--- END ---\n"
        "Return the complete updated SongInfo JSON. Do not remove correct fields."
    )

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=0)
    pacer = get_pacer()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            pacer.wait_if_needed()
            response = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": SONG_SUPPLEMENT_PROMPT},
                    {"role": "user", "content": user},
                ],
                response_format=SongInfo,
                temperature=0.1,
            )
            if hasattr(response, "_raw_response") and hasattr(response._raw_response, "headers"):
                pacer.update_from_headers(dict(response._raw_response.headers))
            return response.choices[0].message.parsed
        except Exception as e:
            logger.warning("fill_song_gaps attempt %s: %s", attempt, e)
            if attempt == MAX_RETRIES:
                return info
            _sleep_with_progress(8.0, "Supplement retry")

    return info


def _song_template_path() -> Path:
    return Path(__file__).resolve().parent / "song_template.json"


def validate_song_dict_matches_template(data: dict) -> None:
    """
    Ensure saved output matches song_template.json: same key sets at each level,
    and line notes are JSON arrays of strings (as in the template).
    """
    p = _song_template_path()
    if not p.is_file():
        raise FileNotFoundError(f"Missing template file: {p}")

    with open(p, encoding="utf-8") as f:
        tpl = json.load(f)

    def _fail(msg: str) -> None:
        raise ValueError(f"song JSON does not conform to song_template.json: {msg}")

    if set(data.keys()) != set(tpl.keys()):
        _fail(f"top-level keys {sorted(data.keys())} != template {sorted(tpl.keys())}")

    st0 = (tpl.get("stanzas") or [])[:1]
    if not st0 or not st0[0].get("lines"):
        raise ValueError(f"{p.name} must include at least one stanza with at least one line (schema reference)")
    sr0 = (tpl.get("sources") or [])[:1]
    if not sr0:
        raise ValueError(f"{p.name} must include at least one sources[] entry (schema reference)")

    stanza_keys = set(st0[0].keys())
    line_keys = set(st0[0]["lines"][0].keys())
    source_keys = set(sr0[0].keys())

    for si, stanza in enumerate(data.get("stanzas") or []):
        if not isinstance(stanza, dict):
            _fail(f"stanzas[{si}] is not an object")
        if set(stanza.keys()) != stanza_keys:
            _fail(f"stanzas[{si}] keys {sorted(stanza.keys())} != template {sorted(stanza_keys)}")
        for li, line in enumerate(stanza.get("lines") or []):
            if not isinstance(line, dict):
                _fail(f"stanzas[{si}].lines[{li}] is not an object")
            if set(line.keys()) != line_keys:
                _fail(
                    f"stanzas[{si}].lines[{li}] keys {sorted(line.keys())} != template {sorted(line_keys)}"
                )
            notes = line.get("notes", [])
            if not isinstance(notes, list):
                _fail(f"stanzas[{si}].lines[{li}].notes must be a JSON array")
            for i, n in enumerate(notes):
                if not isinstance(n, str):
                    _fail(f"stanzas[{si}].lines[{li}].notes[{i}] must be a string")

    for ri, ref in enumerate(data.get("sources") or []):
        if not isinstance(ref, dict):
            _fail(f"sources[{ri}] is not an object")
        if set(ref.keys()) != source_keys:
            _fail(f"sources[{ri}] keys {sorted(ref.keys())} != template {sorted(source_keys)}")


def save_song_info(info: SongInfo, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    info = resolve_unique_song_id(info, output_dir)
    path = os.path.join(output_dir, f"{info.song_id}.json")
    data = info.model_dump()
    validate_song_dict_matches_template(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.info("Saved song → %s", path)
    return path


def build_aggregate_context(
    *,
    shivkumar_block: str,
    karnatik_blocks: list[str],
    web_block: str,
) -> str:
    parts: list[str] = []
    if shivkumar_block.strip():
        parts.append(
            "\n" + "=" * 60 + "\nSOURCE: shivkumar.org notation / index\n" + "=" * 60 + "\n"
            + shivkumar_block
        )
    for i, kb in enumerate(karnatik_blocks):
        if kb.strip():
            parts.append(
                "\n" + "=" * 60 + f"\nSOURCE: karnatik.com page {i + 1}\n" + "=" * 60 + "\n" + kb
            )
    if web_block.strip():
        parts.append(
            "\n" + "=" * 60 + "\nSOURCE: web search snippets\n" + "=" * 60 + "\n" + web_block
        )
    return "\n".join(parts)
