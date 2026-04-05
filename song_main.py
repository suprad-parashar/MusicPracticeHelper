#!/usr/bin/env python3
"""
Carnatic Song RAG pipeline

Primary sources:
  - https://www.shivkumar.org/music/ — krithi index + notation (HTML/PDF/DOC)
  - https://www.karnatik.com/ — lyrics (URLs from web search)

Also aggregates Google Custom Search (if GOOGLE_API_KEY + GOOGLE_CSE_ID) or DuckDuckGo.

Examples:
  python song_main.py --name "Chintayamaa"
  python song_main.py --name "chintayama" --list-matches
  python song_main.py --name "Enduku peddalu" --crawl-only
  python song_main.py --compile
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich import box

from raga_extractor import reset_pacer, web_search
from song_extractor import (
    SourceRef,
    build_aggregate_context,
    enrich_youtube_fields,
    extract_song_info,
    fill_song_gaps,
    identify_song_gaps,
    save_song_info,
    DEFAULT_MODEL,
)
from song_sources import (
    SHIVKUMAR_INDEX,
    fetch_karnatik_lyrics_text,
    fetch_shivkumar_notation_text,
    is_karnatik_url,
    load_or_build_shivkumar_index,
    session,
    ShivkumarSongRow,
)
from song_utils import fuzzy_best_matches
from song_web import search_song_multi

console = Console()

CACHE_INDEX = Path(".cache/shivkumar_krithi_index.json")
OUTPUT_SONGS_DIR = "songs"
ALL_SONGS_JSON = "all_songs.json"


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


def compile_songs_bundle(output_dir: str, dest: Path) -> None:
    out = Path(output_dir)
    if not out.is_dir():
        console.print(f"[red]Not a directory: {out.resolve()}[/red]")
        sys.exit(1)
    paths = sorted(out.glob("*.json"))
    songs: list[dict] = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                songs.append(json.load(f))
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON {path}: {e}[/red]")
            sys.exit(1)
    songs.sort(key=lambda x: (x.get("title") or "", x.get("song_id") or ""))
    dest = dest.resolve()
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(songs, f, indent=2, ensure_ascii=False)
        f.write("\n")
    console.print(f"[green]Compiled {len(songs)} songs → {dest}[/green]")


def pick_shivkumar_row(
    query: str,
    rows: list[ShivkumarSongRow],
    *,
    min_score: float = 0.55,
) -> tuple[ShivkumarSongRow | None, list[tuple[str, float]]]:
    titles = [r.title for r in rows]
    ranked = fuzzy_best_matches(query, titles, limit=12, score_cutoff=min_score)
    if not ranked:
        return None, []
    title_to_row = {r.title: r for r in rows}
    resolved: list[tuple[str, float]] = []
    for title, score in ranked:
        if title in title_to_row:
            resolved.append((title, score))
    if not resolved:
        return None, []
    best_title, best_s = resolved[0]
    return title_to_row[best_title], resolved


def gather_context(
    query: str,
    row: ShivkumarSongRow | None,
    *,
    max_notation_chars: int,
    max_karnatik_chars: int,
    max_karnatik_pages: int,
) -> tuple[str, list[dict], ShivkumarSongRow | None]:
    """Fetch Shivkumar notation text, karnatik pages from search, and web snippets."""
    sess = session()
    shiv_block = ""
    meta = row

    if row and row.notation_html_url:
        try:
            shiv_block = fetch_shivkumar_notation_text(sess, row.notation_html_url, max_notation_chars)
            shiv_block = (
                f"INDEX METADATA:\nTitle: {row.title}\nRaga: {row.raga}\nTala: {row.tala}\n"
                f"Composer line: {row.composer_line}\n"
                f"Notation URL: {row.notation_html_url}\n\nNOTATION PAGE TEXT:\n{shiv_block}"
            )
        except Exception as e:
            logging.getLogger(__name__).warning("Shivkumar notation fetch failed: %s", e)
            shiv_block = f"(Could not fetch notation HTML: {e})\n{SHIVKUMAR_INDEX}"

    web_text, web_hits = search_song_multi(query, per_query=5)

    karnatik_texts: list[str] = []
    seen_urls: set[str] = set()
    for h in web_hits:
        url = (h.get("url") or "").strip()
        if not is_karnatik_url(url):
            continue
        if "lyrics.shtml" in url or url.rstrip("/").endswith("karnatik.com"):
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        if len(karnatik_texts) >= max_karnatik_pages:
            break
        try:
            txt = fetch_karnatik_lyrics_text(sess, url, max_karnatik_chars)
            karnatik_texts.append(f"URL: {url}\n\n{txt}")
        except Exception as e:
            logging.getLogger(__name__).debug("Karnatik fetch %s: %s", url, e)

    if not shiv_block and not karnatik_texts:
        # Still have web_hits for LLM-only extraction
        pass

    ctx = build_aggregate_context(
        shivkumar_block=shiv_block,
        karnatik_blocks=karnatik_texts,
        web_block=web_text,
    )
    extra_meta = [{"title": h.get("title", ""), "url": h.get("url", ""), "snippet": h.get("snippet", "")} for h in web_hits]
    return ctx, extra_meta, meta


def run_pipeline(args: argparse.Namespace) -> None:
    setup_logging(args.verbose)
    if args.llm_delay is not None:
        reset_pacer(args.llm_delay)

    project_root = Path(__file__).resolve().parent
    if args.compile:
        compile_songs_bundle(args.output, project_root / ALL_SONGS_JSON)
        return

    index_delay = float(os.environ.get("SONG_INDEX_FETCH_DELAY", "0.35"))
    rows = load_or_build_shivkumar_index(
        CACHE_INDEX,
        force_refresh=args.refresh_index,
        delay_s=index_delay,
    )

    query = " ".join(args.name).strip()
    if not query:
        console.print("[red]Provide --name \"Song title\"[/red]")
        sys.exit(1)

    row, ranked = pick_shivkumar_row(query, rows, min_score=args.min_match_score)

    if args.list_matches:
        table = Table(title="Shivkumar index fuzzy matches", box=box.ROUNDED)
        table.add_column("Score", justify="right", style="cyan")
        table.add_column("Title", style="bold")
        table.add_column("Raga", style="green")
        for title, score in ranked[:20]:
            r = next((x for x in rows if x.title == title), None)
            table.add_row(f"{score:.2f}", title, r.raga if r else "")
        console.print(table)
        if not row:
            console.print("[yellow]No row above threshold; pipeline will rely on web search only.[/yellow]")
        return

    if row:
        console.print(Panel(
            f"[bold]{row.title}[/bold]  [dim]({row.raga}, {row.tala})[/dim]\n"
            f"Notation: {row.notation_html_url or '—'}",
            title="Matched Shivkumar index",
            border_style="green",
        ))
    else:
        console.print(Panel(
            "[yellow]No Shivkumar index match above threshold.[/yellow]\n"
            f"[dim]Trying web search for: {query!r}[/dim]",
            title="Shivkumar",
            border_style="yellow",
        ))

    ctx, web_meta, _ = gather_context(
        query,
        row,
        max_notation_chars=args.max_notation_chars,
        max_karnatik_chars=args.max_karnatik_chars,
        max_karnatik_pages=args.max_karnatik_pages,
    )

    console.print(f"[dim]Aggregated context: {len(ctx)} characters[/dim]")

    if args.crawl_only:
        preview = ctx[:6000] + ("…" if len(ctx) > 6000 else "")
        console.print(Panel(preview, title="Crawl preview", border_style="cyan"))
        console.print(f"[dim]{len(web_meta)} web hits merged[/dim]")
        return

    if not os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") == "your-api-key-here":
        console.print("[red]OPENAI_API_KEY not set. Use --crawl-only to fetch sources without LLM.[/red]")
        sys.exit(1)

    model = args.model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    extra = None
    if args.extra_info_file:
        extra = Path(args.extra_info_file).expanduser().read_text(encoding="utf-8").strip()
    elif args.extra_info:
        extra = args.extra_info.strip()

    with console.status("[cyan]LLM extraction...", spinner="dots"):
        info = extract_song_info(query, ctx, model=model, extra_user_notes=extra)

    seen_source_urls = {r.url for r in info.sources}
    for m in web_meta[:25]:
        url = (m.get("url") or "").strip()
        if not url or url in seen_source_urls:
            continue
        seen_source_urls.add(url)
        title = (m.get("title") or "").strip()[:80]
        label = "web_search"
        if "karnatik.com" in url:
            label = "karnatik_web"
        elif "shivkumar.org" in url:
            label = "shivkumar_web"
        if title:
            label = f"{label}:{title}" if len(label) + len(title) < 120 else label
        info.sources.append(SourceRef(label=label, url=url))
    if row and (row.notation_html_url or "").strip():
        nu = row.notation_html_url.strip()
        if not (info.notation_url or "").strip():
            info = info.model_copy(update={"notation_url": nu})
        if nu not in seen_source_urls:
            seen_source_urls.add(nu)
            info.sources.append(SourceRef(label="shivkumar_notation_html", url=nu))

    info = enrich_youtube_fields(info)
    if not (info.youtube_url or "").strip():
        for m in web_meta:
            url = (m.get("url") or "").strip()
            if url and ("youtube.com/watch" in url or "youtu.be/" in url):
                info = info.model_copy(update={"youtube_url": url})
                break
    info = enrich_youtube_fields(info)

    if not args.no_gap_fill:
        gap_names, gap_queries = identify_song_gaps(info)
        if gap_queries:
            console.print(f"[bold]Gap fill:[/bold] {', '.join(gap_names) or 'fields'}")
            with console.status("[cyan]Web search for gaps (DDG, same as raga pipeline)...", spinner="dots"):
                add_text, add_hits = web_search(
                    gap_queries, max_results_per_query=args.max_gap_results
                )
            if add_text.strip():
                with console.status("[cyan]LLM gap merge...", spinner="dots"):
                    info = fill_song_gaps(info, add_text, model=model, extra_user_notes=extra)
                seen_source_urls = {r.url for r in info.sources}
                for m in add_hits:
                    url = (m.get("url") or "").strip()
                    if not url or url in seen_source_urls:
                        continue
                    seen_source_urls.add(url)
                    info.sources.append(SourceRef(label="web_gap_fill", url=url))

    info = enrich_youtube_fields(info)
    out_path = save_song_info(info, output_dir=args.output)
    console.print(f"[green]Saved {out_path}[/green]")

    preview = json.dumps(info.model_dump(), indent=2, ensure_ascii=False)[:4000]
    console.print(Panel(preview + ("…" if len(preview) >= 4000 else ""), title="SongInfo", border_style="cyan"))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Carnatic song RAG: Shivkumar notation + karnatik lyrics + web search + LLM JSON",
        epilog=(
            "Web search: set GOOGLE_API_KEY and GOOGLE_CSE_ID (Programmable Search Engine cx) "
            "for Google JSON API; otherwise DuckDuckGo (ddgs) is used with site:karnatik.com / "
            "site:shivkumar.org queries."
        ),
    )
    p.add_argument(
        "-n", "--name",
        nargs="+",
        help="Song title (natural spelling; fuzzy-matched to Shivkumar index)",
    )
    p.add_argument(
        "--list-matches",
        action="store_true",
        help="List fuzzy matches against Shivkumar index and exit",
    )
    p.add_argument(
        "--min-match-score",
        type=float,
        default=0.55,
        help="Minimum fuzzy score to tie Shivkumar index (0-1, default 0.55)",
    )
    p.add_argument(
        "--refresh-index",
        action="store_true",
        help="Re-download and parse Shivkumar krithi index",
    )
    p.add_argument(
        "--crawl-only",
        action="store_true",
        help="Fetch and show context only; no LLM",
    )
    p.add_argument(
        "--no-gap-fill",
        action="store_true",
        help="Skip secondary web search + LLM for empty fields",
    )
    p.add_argument(
        "--max-gap-results",
        type=int,
        default=4,
        metavar="N",
        help="Max search hits per gap query (DuckDuckGo via raga_extractor.web_search; default 4)",
    )
    p.add_argument(
        "--max-notation-chars",
        type=int,
        default=18000,
        help="Max characters from Shivkumar notation HTML",
    )
    p.add_argument(
        "--max-karnatik-chars",
        type=int,
        default=12000,
        help="Max characters per karnatik page",
    )
    p.add_argument(
        "--max-karnatik-pages",
        type=int,
        default=3,
        help="Max karnatik.com pages to fetch from search results",
    )
    p.add_argument(
        "--model",
        default=None,
        help=f"OpenAI model (default {DEFAULT_MODEL} or OPENAI_MODEL)",
    )
    p.add_argument(
        "--output",
        "-o",
        default=OUTPUT_SONGS_DIR,
        help=f"Output directory (default {OUTPUT_SONGS_DIR})",
    )
    p.add_argument(
        "--llm-delay",
        type=float,
        default=None,
        help="Override LLM pacing delay (seconds)",
    )
    p.add_argument(
        "--extra-info",
        default=None,
        help="Extra authoritative facts for the LLM",
    )
    p.add_argument(
        "--extra-info-file",
        default=None,
        metavar="PATH",
        help="Extra facts from UTF-8 file",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help=f"Merge {OUTPUT_SONGS_DIR}/*.json into {ALL_SONGS_JSON}",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.compile:
        run_pipeline(args)
        return
    if not args.name:
        parser.print_help()
        sys.exit(1)
    run_pipeline(args)


if __name__ == "__main__":
    main()
