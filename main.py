#!/usr/bin/env python3
"""
Carnatic Raga RAG Pipeline

A CLI tool that crawls Wikipedia to gather information about Carnatic ragas
(both melakarta and janya), selects the most relevant pages, and uses an LLM
to extract structured data.

Usage:
    python main.py --number 15              # Melakarta raga by number
    python main.py --name Shankarabharanam   # Melakarta raga by name
    python main.py --name Mohanam            # Janya raga by name
    python main.py --file ragas.txt          # List of ragas from file
    python main.py --all                     # All 72 melakarta ragas
    python main.py --crawl-only --number 29  # Only crawl, skip LLM extraction
    python main.py --name X --skip-wikipedia  # DuckDuckGo web search only (no Wikipedia API)
    python main.py --name Kanada --extra-info "Janya of 22; known compositions: …"
    python main.py --compile                 # Merge output/*.json → all_ragas.json
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich import box

from melakarta_ragas import (
    get_raga_info,
    find_raga_by_name,
    resolve_raga_input,
    get_all_ragas,
    make_janya_raga_dict,
)
from wikipedia_crawler import WikipediaCrawler, WikiPage
from raga_extractor import (
    extract_raga_info,
    save_raga_info,
    reset_pacer,
    identify_gaps,
    web_search,
    fill_gaps,
    fetch_web_context_for_raga,
    RagaInfo,
    DEFAULT_MODEL,
)

console = Console()

OUTPUT_DIR = "output"
CACHE_DIR = ".cache/wiki"
ALL_RAGAS_JSON = "all_ragas.json"


def compile_ragas_to_bundle(output_dir: str, dest_path: Path) -> None:
    """Load every *.json raga file from output_dir, sort, write as one JSON array."""
    out = Path(output_dir)
    if not out.is_dir():
        console.print(f"[red]Error: Output directory not found: {out.resolve()}[/red]")
        sys.exit(1)

    paths = sorted(out.glob("*.json"))
    ragas: list[dict] = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                ragas.append(json.load(f))
        except json.JSONDecodeError as e:
            console.print(f"[red]Error: Invalid JSON in {path}: {e}[/red]")
            sys.exit(1)

    def sort_key(obj: dict):
        if obj.get("is_melakarta") and obj.get("melakarta_number") is not None:
            return (0, obj["melakarta_number"], obj.get("raga_id") or "")
        return (1, 0, obj.get("raga_id") or "")

    ragas.sort(key=sort_key)

    dest_path = dest_path.resolve()
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(ragas, f, indent=2, ensure_ascii=False)
        f.write("\n")

    console.print(
        f"[green]Compiled {len(ragas)} ragas from {out.resolve()}/ → {dest_path}[/green]"
    )


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


def resolve_ragas_from_args(args) -> list[dict]:
    """Resolve the list of ragas to process from CLI arguments."""
    ragas = []

    if args.all:
        return get_all_ragas()

    if args.number:
        for num in args.number:
            if not 1 <= num <= 72:
                console.print(f"[red]Error: Melakarta number must be 1-72, got {num}[/red]")
                sys.exit(1)
            ragas.append(get_raga_info(num))

    if args.range:
        start, end = args.range
        if not (1 <= start <= 72 and 1 <= end <= 72):
            console.print(f"[red]Error: Range must be within 1-72, got {start}-{end}[/red]")
            sys.exit(1)
        if start > end:
            start, end = end, start
        for num in range(start, end + 1):
            ragas.append(get_raga_info(num))

    if args.name:
        for name in args.name:
            raga = find_raga_by_name(name)
            if not raga:
                raga = make_janya_raga_dict(name)
            ragas.append(raga)

    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            console.print(f"[red]Error: File not found: {filepath}[/red]")
            sys.exit(1)
        for line in filepath.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                resolved = resolve_raga_input(line)
                ragas.extend(resolved)
            except ValueError as e:
                console.print(f"[yellow]Warning: {e} (skipping)[/yellow]")

    if not ragas:
        console.print("[red]Error: No ragas specified. Use --number, --name, --file, --range, or --all[/red]")
        sys.exit(1)

    seen = set()
    unique_ragas = []
    for r in ragas:
        key = r["melakarta_number"] if r.get("melakarta_number") else r["name"].lower()
        if key not in seen:
            seen.add(key)
            unique_ragas.append(r)

    return unique_ragas


def _extract_wiki_name(pages: list, user_name: str) -> str:
    """Extract the canonical raga name from the best matching Wikipedia page.

    Prefers the main page (score >= 100) but falls back to high-scoring
    supplementary pages (score >= 50).  Strips common suffixes like
    '(raga)' or '(Carnatic raga)'.
    """
    import re as _re
    best = None
    for page in pages:
        if page.relevance_score >= 50.0:
            if best is None or page.relevance_score > best.relevance_score:
                best = page
    if best:
        title = _re.sub(r"\s*\(.*?\)\s*$", "", best.title).strip()
        if title:
            return title
    return user_name


def display_raga_table(ragas: list[dict]):
    """Display a table of ragas that will be processed."""
    table = Table(title="Ragas to Process", box=box.ROUNDED, show_lines=False)
    table.add_column("#", style="cyan", justify="right", width=4)
    table.add_column("Raga Name", style="bold white")
    table.add_column("Type", style="magenta", justify="center")
    table.add_column("Arohana", style="green")

    for raga in ragas:
        num = raga.get("melakarta_number")
        table.add_row(
            str(num) if num else "-",
            raga["name"],
            "Melakarta" if num else "Janya",
            raga.get("arohana_str") or "[dim]from Wikipedia[/dim]",
        )

    console.print(table)
    console.print()


def display_crawl_results(pages: list[WikiPage], raga_name: str):
    """Display crawl results as a tree."""
    tree = Tree(f"[bold cyan]Wikipedia pages for [yellow]{raga_name}[/yellow]")
    for page in pages:
        branch = tree.add(
            f"[{'bold green' if page.relevance_score >= 10 else 'white'}]"
            f"{page.title} [dim](score: {page.relevance_score:.1f})[/dim]"
        )
        branch.add(f"[dim]{page.url}[/dim]")
        branch.add(f"[dim]{len(page.content)} chars[/dim]")
    console.print(tree)
    console.print()


def display_web_search_preview(raga_name: str, search_results: list[dict], context_chars: int):
    """Show web search hits used as primary context (DuckDuckGo, not Wikipedia API)."""
    tree = Tree(
        f"[bold cyan]Web search results for [yellow]{raga_name}[/yellow] "
        f"[dim]({context_chars} chars)[/dim]"
    )
    for sr in search_results[:15]:
        branch = tree.add(f"[white]{sr.get('title', '')}[/white]")
        branch.add(f"[dim]{sr.get('url', '')}[/dim]")
        snip = (sr.get("snippet") or "")[:200]
        if snip:
            branch.add(f"[dim italic]{snip}...[/dim italic]")
    if len(search_results) > 15:
        tree.add(f"[dim]... and {len(search_results) - 15} more[/dim]")
    console.print(tree)
    console.print()


def display_extraction_result(raga_info: RagaInfo):
    """Display extracted raga information in a rich panel."""
    sections = []

    name_line = f"[bold yellow]{raga_info.raga_name}[/bold yellow]"
    if raga_info.melakarta_number:
        name_line += f"  [dim](Melakarta #{raga_info.melakarta_number})[/dim]"
    elif raga_info.parent_raga:
        name_line += f"  [dim](Janya of Melakarta #{raga_info.parent_raga})[/dim]"
    else:
        name_line += f"  [dim](Janya raga)[/dim]"
    sections.append(name_line)
    if raga_info.other_names:
        sections.append(f"[dim]Also known as: {', '.join(raga_info.other_names)}[/dim]")

    sections.append("")
    if raga_info.raga_type:
        sections.append(f"[magenta]Type:[/magenta] {raga_info.raga_type}")
    if raga_info.chakra:
        sections.append(f"[magenta]Chakra:[/magenta] {raga_info.chakra}")

    sections.append("")
    if raga_info.arohana:
        sections.append(f"[green]Arohana:[/green]  {' '.join(raga_info.arohana)}")
    if raga_info.avrohana:
        sections.append(f"[green]Avrohana:[/green] {' '.join(raga_info.avrohana)}")

    sections.append("")
    if raga_info.rasa:
        sections.append(f"[cyan]Rasa:[/cyan] {', '.join(raga_info.rasa)}")
    if raga_info.mood:
        sections.append(f"[cyan]Mood:[/cyan] {raga_info.mood}")
    if raga_info.description:
        sections.append(f"[cyan]Description:[/cyan] {raga_info.description}")
    if raga_info.time_of_day:
        sections.append(f"[cyan]Time:[/cyan] {raga_info.time_of_day}")
    if raga_info.gamaka_usage:
        sections.append(f"[cyan]Gamakas:[/cyan] {raga_info.gamaka_usage}")

    equivs = []
    if raga_info.hindustani_equivalent:
        equivs.append(f"Hindustani: {raga_info.hindustani_equivalent}")
    if raga_info.western_equivalent:
        equivs.append(f"Western: {raga_info.western_equivalent}")
    if equivs:
        sections.append("")
        for eq in equivs:
            sections.append(f"[blue]{eq}[/blue]")

    if raga_info.notable_compositions:
        sections.append("")
        sections.append(f"[bold]Compositions ({len(raga_info.notable_compositions)}):[/bold]")
        for comp in raga_info.notable_compositions[:12]:
            line = f"  - {comp.name}"
            if comp.composer:
                line += f" [dim]({comp.composer})[/dim]"
            if comp.language:
                line += f" [dim][{comp.language}][/dim]"
            sections.append(line)
        remaining = len(raga_info.notable_compositions) - 12
        if remaining > 0:
            sections.append(f"  [dim]... and {remaining} more[/dim]")

    if raga_info.popular_janya_ragas:
        sections.append("")
        sections.append(f"[bold]Janya Ragas:[/bold] {', '.join(raga_info.popular_janya_ragas[:20])}")

    if raga_info.notable_features:
        sections.append("")
        sections.append(f"[bold]Notes:[/bold] {raga_info.notable_features}")

    console.print(Panel(
        "\n".join(sections),
        title=f"[bold]Raga: {raga_info.raga_name}[/bold]",
        border_style="cyan",
        padding=(1, 2),
    ))


def process_single_raga(
    raga: dict,
    crawler: WikipediaCrawler,
    crawl_only: bool = False,
    top_n: int = 5,
    model: str | None = None,
    output_dir: str = OUTPUT_DIR,
    skip_wikipedia: bool = False,
    extra_info: str | None = None,
) -> RagaInfo | None:
    """Process a single raga through the full pipeline."""
    name = raga["name"]
    aliases = raga.get("aliases", [])
    melakarta_num = raga.get("melakarta_number")

    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    if melakarta_num:
        console.print(f"[bold]Processing: [yellow]{name}[/yellow] (Melakarta #{melakarta_num})[/bold]")
        console.print(f"[dim]Swaras: {raga.get('arohana_str', '')}[/dim]")
    else:
        console.print(f"[bold]Processing: [yellow]{name}[/yellow] (Janya raga)[/bold]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

    if skip_wikipedia:
        console.print("[bold]Phase 1: Web search (Wikipedia skipped)[/bold]")
        with console.status(f"[cyan]Searching the web for {name}...", spinner="dots"):
            context, primary_results = fetch_web_context_for_raga(
                name, aliases=aliases, melakarta_number=melakarta_num
            )

        if not (context or "").strip():
            console.print(
                f"[red]No web search results for {name}.[/red] "
                "[dim]Install ddgs (pip install ddgs) and check connectivity.[/dim]"
            )
            return None

        wiki_name = name
        display_web_search_preview(wiki_name, primary_results, len(context))
        console.print(
            f"[dim]Primary context: {len(context)} chars from DuckDuckGo text search[/dim]\n"
        )

        if crawl_only:
            console.print("[yellow]Crawl-only mode: skipping LLM extraction[/yellow]")
            return None
    else:
        console.print("[bold]Phase 1: Wikipedia Crawling[/bold]")
        with console.status(f"[cyan]Crawling Wikipedia for {name}...", spinner="dots"):
            pages = crawler.crawl_raga(
                raga_name=name,
                aliases=aliases,
                melakarta_number=melakarta_num,
                top_n=top_n,
            )

        if not pages:
            console.print(f"[red]No Wikipedia pages found for {name}[/red]")
            return None

        wiki_name = _extract_wiki_name(pages, name)
        if wiki_name != name:
            console.print(f"[dim]Wikipedia name: {wiki_name}[/dim]")

        display_crawl_results(pages, wiki_name)

        context = crawler.build_context(pages)
        console.print(f"[dim]Total context: {len(context)} chars from {len(pages)} pages[/dim]\n")

        if crawl_only:
            console.print("[yellow]Crawl-only mode: skipping LLM extraction[/yellow]")
            return None

    console.print("[bold]Phase 2: LLM Extraction[/bold]")
    known_swaras = raga if melakarta_num else None
    source_label = "WEB SEARCH SNIPPETS" if skip_wikipedia else "WIKIPEDIA CONTENT"
    with console.status(f"[cyan]Extracting raga information via LLM...", spinner="dots"):
        raga_info = extract_raga_info(
            raga_name=wiki_name,
            context=context,
            known_swaras=known_swaras,
            melakarta_number=melakarta_num,
            model=model,
            extra_user_notes=extra_info,
            source_label=source_label,
        )

    if not skip_wikipedia:
        best_wiki_page = max(pages, key=lambda p: p.relevance_score)
        if best_wiki_page.relevance_score >= 50.0:
            raga_info.wikipedia_url = best_wiki_page.url

    gap_names, gap_queries = identify_gaps(raga_info)
    if gap_queries:
        console.print(f"\n[bold]Phase 3: Filling gaps[/bold]")
        console.print(f"[yellow]Missing fields:[/yellow] {', '.join(gap_names)}")

        with console.status("[cyan]Searching the web for missing information...", spinner="dots"):
            additional_context, search_results = web_search(gap_queries)

        if search_results:
            console.print(f"[dim]Found {len(search_results)} results ({len(additional_context)} chars):[/dim]")
            for sr in search_results[:10]:
                console.print(f"  [dim]• {sr['title']}[/dim]")
                console.print(f"    [dim italic]{sr['snippet']}...[/dim italic]")
            if len(search_results) > 10:
                console.print(f"  [dim]... and {len(search_results) - 10} more[/dim]")

            with console.status("[cyan]Supplementary LLM extraction...", spinner="dots"):
                raga_info = fill_gaps(
                    raga_info, additional_context, model=model, extra_user_notes=extra_info
                )
        else:
            console.print("[dim]No additional context found from web search[/dim]")
    else:
        console.print("[dim]All fields populated, no gap-filling needed[/dim]")

    filepath = save_raga_info(raga_info, output_dir=output_dir)
    console.print(f"\n[green]Saved to {filepath}[/green]\n")

    display_extraction_result(raga_info)
    return raga_info


def main():
    parser = argparse.ArgumentParser(
        description="Carnatic Raga RAG Pipeline - Extract raga information from Wikipedia",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --number 15                  # Process Mayamalavagowla (melakarta)
  %(prog)s --number 29 65               # Process multiple by number
  %(prog)s --range 1 10                 # Process ragas 1 through 10
  %(prog)s --name Shankarabharanam       # Melakarta raga by name
  %(prog)s --name Mohanam                # Janya raga by name
  %(prog)s --name Mohanam Bhairavi       # Multiple ragas (melakarta + janya)
  %(prog)s --file ragas.txt              # Process from file (one name/number per line)
  %(prog)s --all                         # All 72 melakarta ragas
  %(prog)s --crawl-only --name Abheri    # Only crawl Wikipedia, skip LLM
  %(prog)s --skip-wikipedia --name X    # Web search only (DuckDuckGo), no Wikipedia API
  %(prog)s --name Kanada --extra-info "Janya of 22; arohana S R2 G2 M1 P M1 D2 N2 S"
  %(prog)s --list                        # List all 72 melakarta ragas
  %(prog)s --compile                     # Merge output/*.json into all_ragas.json
        """,
    )

    input_group = parser.add_argument_group("Input (choose one or more)")
    input_group.add_argument(
        "-n", "--number",
        type=int, nargs="+",
        help="Melakarta raga number(s) (1-72)",
    )
    input_group.add_argument(
        "--name",
        type=str, nargs="+",
        help="Raga name(s) — works for both melakarta and janya ragas",
    )
    input_group.add_argument(
        "-f", "--file",
        type=str,
        help="Path to file with raga names/numbers (one per line)",
    )
    input_group.add_argument(
        "-r", "--range",
        type=int, nargs=2, metavar=("START", "END"),
        help="Range of melakarta numbers (e.g. --range 1 10 for ragas 1 through 10)",
    )
    input_group.add_argument(
        "-a", "--all",
        action="store_true",
        help="Process all 72 melakarta ragas",
    )

    options_group = parser.add_argument_group("Options")
    options_group.add_argument(
        "--crawl-only",
        action="store_true",
        help="Only fetch sources (Wikipedia or web search with --skip-wikipedia), don't run LLM extraction",
    )
    options_group.add_argument(
        "--skip-wikipedia",
        action="store_true",
        help="Skip Wikipedia; use DuckDuckGo web search as the primary context (gap-filling still uses web search)",
    )
    options_group.add_argument(
        "--extra-info",
        type=str,
        default=None,
        help="Extra facts you already know about the raga; injected into the LLM as authoritative (use quotes for multiple words)",
    )
    options_group.add_argument(
        "--extra-info-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Same as --extra-info but read from a UTF-8 file; overrides --extra-info if both are set",
    )
    options_group.add_argument(
        "--top-n",
        type=int, default=3,
        help="Number of top linked pages to fetch (default: 3)",
    )
    options_group.add_argument(
        "--model",
        type=str, default=None,
        help=f"OpenAI model for extraction (default: {DEFAULT_MODEL}, or OPENAI_MODEL env var)",
    )
    options_group.add_argument(
        "-o", "--output",
        type=str, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    options_group.add_argument(
        "--delay",
        type=float, default=None,
        help="Delay between Wikipedia API requests in seconds (default: 0.5, or WIKI_CRAWL_DELAY env var)",
    )
    options_group.add_argument(
        "--llm-delay",
        type=float, default=None,
        help="Minimum delay between LLM API calls in seconds (default: 2.0, or LLM_CALL_DELAY env var)",
    )
    options_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    options_group.add_argument(
        "--list",
        action="store_true",
        help="List all 72 melakarta ragas and exit",
    )
    options_group.add_argument(
        "--compile",
        action="store_true",
        help=f"Merge all JSON files from --output into {ALL_RAGAS_JSON} in the project root and exit",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.llm_delay is not None:
        reset_pacer(args.llm_delay)

    extra_info_text: str | None = None
    if args.extra_info_file:
        extra_path = Path(args.extra_info_file).expanduser()
        if not extra_path.is_file():
            console.print(f"[red]Error: --extra-info-file not found: {extra_path.resolve()}[/red]")
            sys.exit(1)
        extra_info_text = extra_path.read_text(encoding="utf-8").strip()
    elif args.extra_info:
        extra_info_text = args.extra_info.strip()

    project_root = Path(__file__).resolve().parent
    if args.compile:
        compile_ragas_to_bundle(args.output, project_root / ALL_RAGAS_JSON)
        return

    effective_model = args.model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    wiki_delay = args.delay if args.delay is not None else float(os.environ.get("WIKI_CRAWL_DELAY", "0.5"))

    pipeline_line = (
        "[dim]Web search → Extract → JSON[/dim]"
        if args.skip_wikipedia
        else "[dim]Wikipedia → Crawl → Rank → Extract → JSON[/dim]"
    )
    extra_line = ""
    if extra_info_text:
        preview = extra_info_text.replace("\n", " ")[:80]
        if len(extra_info_text) > 80:
            preview += "…"
        extra_line = f"\n[dim]Extra user facts: {preview}[/dim]"
    console.print(Panel(
        "[bold cyan]Carnatic Raga RAG Pipeline[/bold cyan]\n"
        f"{pipeline_line}\n"
        f"[dim]Model: {effective_model}[/dim]"
        f"{extra_line}",
        border_style="cyan",
    ))

    if args.list:
        table = Table(title="72 Melakarta Ragas", box=box.ROUNDED, show_lines=False)
        table.add_column("#", style="cyan", justify="right", width=4)
        table.add_column("Name", style="bold white", min_width=20)
        table.add_column("Arohana", style="green")
        table.add_column("Ma", style="yellow", justify="center")

        for raga in get_all_ragas():
            table.add_row(
                str(raga["melakarta_number"]),
                raga["name"],
                raga["arohana_str"],
                raga["ma"],
            )
        console.print(table)
        return

    if not any([args.number, args.range, args.name, args.file, args.all]):
        parser.print_help()
        sys.exit(1)

    ragas = resolve_ragas_from_args(args)
    display_raga_table(ragas)

    if not args.crawl_only:
        if not os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") == "your-api-key-here":
            console.print(
                "[red]Error: OPENAI_API_KEY not set (or still the placeholder).[/red]\n"
                "[dim]Set it in your .env file or with: export OPENAI_API_KEY='sk-...'[/dim]\n"
                "[dim]Or use --crawl-only to skip LLM extraction.[/dim]"
            )
            sys.exit(1)

    crawler = WikipediaCrawler(
        cache_dir=Path(CACHE_DIR),
        delay=wiki_delay,
    )

    results: list[RagaInfo] = []
    failed: list[str] = []
    start_time = time.time()

    for i, raga in enumerate(ragas, 1):
        console.print(f"\n[bold magenta]Progress: {i}/{len(ragas)}[/bold magenta]")
        try:
            result = process_single_raga(
                raga=raga,
                crawler=crawler,
                crawl_only=args.crawl_only,
                top_n=args.top_n,
                model=args.model,
                output_dir=args.output,
                skip_wikipedia=args.skip_wikipedia,
                extra_info=extra_info_text,
            )
            if result:
                results.append(result)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user.[/yellow]")
            console.print(f"[green]Completed {len(results)} ragas before exit.[/green]")
            sys.exit(130)
        except Exception as e:
            console.print(f"[red]Error processing {raga['name']}: {e}[/red]")
            failed.append(raga["name"])
            if args.verbose:
                console.print_exception()

    elapsed = time.time() - start_time

    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print("[bold]Pipeline Complete[/bold]")
    console.print(f"  Processed: {len(ragas)} ragas")
    console.print(f"  Successful extractions: {len(results)}")
    if failed:
        console.print(f"  Failed: {len(failed)} ({', '.join(failed)})")
    console.print(f"  Time: {elapsed:.1f}s")
    console.print(f"  Output: {args.output}/")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")


if __name__ == "__main__":
    main()
