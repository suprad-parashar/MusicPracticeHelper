"""
Wikipedia crawler that finds and ranks relevant pages for a given Carnatic raga.

Strategy:
1. Search Wikipedia for the raga name
2. Fetch the main article and extract all internal links
3. Score links by relevance to Carnatic music context
4. Fetch top-N linked pages
5. Return aggregated content for downstream LLM extraction
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_BASE = "https://en.wikipedia.org/wiki/"

MUSIC_KEYWORDS = {
    "raga", "raag", "ragam", "melakarta", "janya", "carnatic", "karnatik",
    "hindustani", "swara", "swaras", "arohana", "avarohana", "arohanam",
    "avarohanam", "tala", "thala", "kriti", "krithi", "varnam", "pallavi",
    "composition", "composer", "tyagaraja", "thyagaraja", "dikshitar",
    "muthuswami", "syama sastri", "shyama shastri", "purandaradasa",
    "purandara dasa", "annamacharya", "oottukkadu", "swathi thirunal",
    "papanasam sivan", "bharatiyar", "music", "musical", "scale", "mode",
    "gamaka", "gamakas", "sangeetham", "sangeeta", "sangita", "veena",
    "vina", "violin", "mridangam", "flute", "vocal", "concert",
    "bhajan", "keertana", "keerthana", "kirtana", "devotional",
    "notation", "sahitya", "charanam", "anupallavi",
}

COMPOSER_NAMES = {
    "tyagaraja", "thyagaraja", "muthuswami dikshitar", "muthuswamy dikshitar",
    "shyama shastri", "syama sastri", "purandara dasa", "purandaradasa",
    "annamacharya", "oottukkadu venkata kavi", "swathi thirunal",
    "papanasam sivan", "mysore vasudevachar", "harikesanallur muthiah bhagavathar",
    "koteeswara iyer", "patnam subramania iyer", "maha vaidyanatha sivan",
    "gopalakrishna bharati", "arunachala kavi", "marimutha pillai",
    "subbaraya sastri", "walajapet venkataramana bhagavathar",
}

NOISE_PATTERNS = re.compile(
    r"(^List of |^Category:|^Template|^Wikipedia:|^Help:|^Portal:|^File:|"
    r"^Draft:|^User:|^Talk:|^Module:|ISBN|ISSN|OCLC|^[0-9]{4} in )",
    re.IGNORECASE,
)

FILM_NOISE_PATTERNS = re.compile(
    r"(\bfilm\b|\bmovie\b|\bsoundtrack\b|\balbum\b|\btelevision\b|\btv series\b"
    r"|\bactor\b|\bactress\b|\bdirector\b|\bcinema\b|\bbollywood\b|\bkollywood\b"
    r"|\btollywood\b|\bboxoffice\b|\bsinger\b(?!.*carnatic))",
    re.IGNORECASE,
)

MIN_CONTENT_LENGTH = 200

# When Wikipedia uses a different primary title than common spellings / aliases.
# Keys: normalized with _normalize_lookup_key (diacritics stripped, lower, spaces collapsed).
WIKIPEDIA_CANONICAL_TITLES: dict[str, str] = {
    "suddha dhanyasi": "Udayaravichandrika",
    "shuddha dhanyasi": "Udayaravichandrika",
    "sudha dhanyasi": "Udayaravichandrika",
    "suddhadhanyasi": "Udayaravichandrika",
    "shuddhadhanyasi": "Udayaravichandrika",
    "udayaraga": "Udayaravichandrika",
    "udaya raga": "Udayaravichandrika",
    # Wikipedia: "Shuddha Saveri"; common spellings omit / swap "h"
    "suddha saveri": "Shuddha Saveri",
    "shuddha saveri": "Shuddha Saveri",
    "suddhasaveri": "Shuddha Saveri",
    "shuddhasaveri": "Shuddha Saveri",
    "karnataka suddha saveri": "Karnataka Shuddha Saveri",
    "karnataka shuddha saveri": "Karnataka Shuddha Saveri",
    # Wikipedia: "Kalyanavasantam" (one word; second part is vasantam, not vasantham)
    "kalyana vasantam": "Kalyanavasantam",
    "kalyana vasantham": "Kalyanavasantam",
    "kalyanavasantam": "Kalyanavasantam",
    "kalyanavasantham": "Kalyanavasantam",
    # Wikipedia: "Gambhiranata"; also known as Shuddha Nata; Tamil-style "Gambheeranattai"
    "gambheeranattai": "Gambhiranata",
    "gambhiranattai": "Gambhiranata",
    "gambheeranata": "Gambhiranata",
    "gambhiranata": "Gambhiranata",
}


def _normalize_lookup_key(name: str) -> str:
    """Normalize for WIKIPEDIA_CANONICAL_TITLES lookup."""
    return " ".join(_strip_diacritics(name).lower().split())


def _canonical_wikipedia_title(raga_name: str, aliases: list[str]) -> str | None:
    """Return the English Wikipedia article title when the raga uses a different primary page name."""
    for raw in [raga_name] + aliases:
        key = _normalize_lookup_key(raw)
        if key in WIKIPEDIA_CANONICAL_TITLES:
            return WIKIPEDIA_CANONICAL_TITLES[key]
    return None


TRANSLITERATION_SWAPS = [
    ("dw", "dv"),
    ("dv", "dw"),
    ("w", "v"),
    ("v", "w"),
    ("th", "t"),
    ("dh", "d"),
    ("bh", "b"),
    ("sh", "s"),
    ("ee", "i"),
    ("oo", "u"),
    ("aa", "a"),
]


def _strip_diacritics(text: str) -> str:
    """Strip diacritical marks: ā→a, ē→e, ś→s, ṇ→n, etc."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _transliteration_variants(name: str) -> list[str]:
    """Generate common transliteration variants for Carnatic raga names.

    Handles diacritical stripping (ā→a, ś→s) and consonant swaps
    (dw↔dv, w↔v, th→t, bh→b, etc.).
    """
    variants = set()
    name_lower = name.lower()

    ascii_form = _strip_diacritics(name)
    if ascii_form != name:
        variants.add(ascii_form)
        name_lower = ascii_form.lower()

    bases = [name.lower(), name_lower]
    for base in bases:
        for old, new in TRANSLITERATION_SWAPS:
            if old in base:
                variant = base.replace(old, new)
                capitalized = _capitalize_words(variant)
                variants.add(capitalized)

    variants.discard(name)
    # Common spelling alternates (whole-name only; avoids corrupting e.g. Reetigowla)
    _simple = _strip_diacritics(name).lower()
    if _simple == "gowla":
        variants.add("Gaula")
    elif _simple == "gaula":
        variants.add("Gowla")

    return list(variants)


def _capitalize_words(text: str) -> str:
    """Title-case each word: 'karnataka shuddha saveri' -> 'Karnataka Shuddha Saveri'."""
    return " ".join(w.capitalize() for w in text.split())


RAGA_WORD_PARTS = [
    "bhairavi", "kalyani", "varali", "saveri", "kambhoji", "dhanyasi",
    "gaula", "gowla", "kannada", "kapi", "mohana", "mohanam", "ranjani",
    "todi", "bilahari", "manohari", "kedaram", "sindhu", "nata", "hindolam",
    "sahana", "begada", "darbar", "mukhari", "ananda", "madhyamavati",
    "vasanta", "valaji", "hamsa", "sri", "shree", "subha", "gambhira",
    "suddha", "shuddha",
]


def _space_split_variants(name: str) -> list[str]:
    """Try splitting a single compound word into two words at known raga-word boundaries.

    'Sindhubhairavi' -> 'Sindhu Bhairavi', 'Anandabhairavi' -> 'Ananda Bhairavi', etc.
    """
    if " " in name:
        return []
    name_lower = _strip_diacritics(name).lower()
    variants = []
    for part in RAGA_WORD_PARTS:
        idx = name_lower.find(part)
        if idx > 1 and idx + len(part) <= len(name_lower):
            left = name[:idx]
            right = name[idx:]
            variant = f"{left} {right}"
            variants.append(_capitalize_words(_strip_diacritics(variant)))
    return variants


def _normalize_for_match(text: str) -> str:
    """Lowercase, strip diacritics, and remove spaces for fuzzy comparison."""
    return _strip_diacritics(text).lower().replace(" ", "")


def _term_matches_title(term: str, title: str) -> bool:
    """True if term matches the raga name in title as a whole word.

    Prevents false positives like 'gowla' matching inside 'Reetigowla'.
    """
    if not term:
        return False
    title_clean = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
    term_norm = _normalize_for_match(term)
    if not term_norm:
        return False
    t_norm = _normalize_for_match(title_clean)
    if term_norm == t_norm:
        return True
    for tok in re.split(r"[\s\-_/]+", title_clean):
        if _normalize_for_match(tok) == term_norm:
            return True
    if re.search(r"(?<![a-z])" + re.escape(term_norm) + r"(?![a-z])", t_norm):
        return True
    return False


def _title_matches_raga(title: str, raga_name: str, aliases: list[str]) -> bool:
    """Check if a page title is about the target raga (fuzzy on transliteration).

    Space-insensitive whole-word matching so 'Sindhubhairavi' matches
    'Sindhu Bhairavi (raga)', but 'Gowla' does not match 'Reetigowla'.
    """
    candidates = set()
    for raw in [raga_name] + aliases:
        candidates.add(_normalize_for_match(raw))
        candidates.add(_strip_diacritics(raw).lower().replace(" ", ""))
        candidates.add(raw.lower())
    for raw in [raga_name] + aliases:
        for v in _transliteration_variants(raw):
            candidates.add(_normalize_for_match(v))
    for term in candidates:
        if not term:
            continue
        if _term_matches_title(term, title):
            return True
    return False


@dataclass
class WikiPage:
    title: str
    url: str
    content: str
    summary: str = ""
    categories: list[str] = field(default_factory=list)
    relevance_score: float = 0.0


class WikipediaCrawler:
    def __init__(self, cache_dir: Path | None = None, delay: float = 0.5):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MusicPracticeHelper/1.0 (Carnatic Raga Research; Python)"
        })
        self.delay = delay
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._request_count = 0

    def _rate_limit(self):
        if self._request_count > 0:
            time.sleep(self.delay)
        self._request_count += 1

    def _api_get(self, params: dict, _retries: int = 3) -> dict:
        self._rate_limit()
        params.setdefault("format", "json")
        params.setdefault("formatversion", "2")
        for attempt in range(1, _retries + 1):
            resp = self.session.get(WIKI_API, params=params, timeout=30)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 5)) + 2
                logger.warning(
                    f"Wikipedia rate limited (attempt {attempt}/{_retries}), "
                    f"waiting {wait:.0f}s..."
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return resp.json()

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search Wikipedia and return matching page titles."""
        data = self._api_get({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
        })
        return data.get("query", {}).get("search", [])

    def fetch_page_content(self, title: str) -> str | None:
        """Fetch plain-text extract of a Wikipedia page."""
        data = self._api_get({
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "exlimit": 1,
        })
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            return None
        return pages[0].get("extract", "")

    def fetch_page_links(self, title: str) -> list[str]:
        """Fetch all internal wiki links from a page."""
        all_links = []
        params = {
            "action": "query",
            "titles": title,
            "prop": "links",
            "pllimit": "max",
        }
        while True:
            data = self._api_get(params)
            pages = data.get("query", {}).get("pages", [])
            if pages:
                links = pages[0].get("links", [])
                all_links.extend(link["title"] for link in links)
            if "continue" in data:
                params["plcontinue"] = data["continue"]["plcontinue"]
            else:
                break
        return all_links

    def fetch_page_categories(self, title: str) -> list[str]:
        """Fetch categories for a page."""
        data = self._api_get({
            "action": "query",
            "titles": title,
            "prop": "categories",
            "cllimit": "max",
        })
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return []
        cats = pages[0].get("categories", [])
        return [c["title"].replace("Category:", "") for c in cats]

    def score_link_relevance(self, link_title: str, raga_name: str, raga_aliases: list[str]) -> float:
        """Score a linked page's relevance to the raga being researched."""
        if NOISE_PATTERNS.search(link_title):
            return -1.0

        title_lower = link_title.lower()

        if FILM_NOISE_PATTERNS.search(title_lower):
            return -1.0

        score = 0.0

        if _title_matches_raga(link_title, raga_name, raga_aliases):
            score += 15.0

        music_hits = sum(1 for kw in MUSIC_KEYWORDS if kw in title_lower)
        score += music_hits * 2.0

        composer_match = any(c in title_lower for c in COMPOSER_NAMES)
        if composer_match:
            score += 4.0

        if re.search(r"\(.*raga\)", title_lower) or re.search(r"\(.*carnatic", title_lower):
            score += 4.0
        elif "raga" in title_lower or "ragam" in title_lower or "raag" in title_lower:
            score += 2.0

        if "carnatic" in title_lower or "karnatik" in title_lower:
            score += 2.0

        if "composition" in title_lower or "kriti" in title_lower:
            score += 2.0

        if "melakarta" in title_lower or "janya" in title_lower:
            score += 3.0

        return score

    def _try_direct_page(self, title_guess: str) -> tuple[str, str] | None:
        """Try to fetch a page by exact title. Returns (title, content) or None."""
        content = self.fetch_page_content(title_guess)
        if content and len(content) >= MIN_CONTENT_LENGTH:
            return title_guess, content
        return None

    def _find_main_page(
        self, raga_name: str, aliases: list[str], melakarta_number: int | None,
    ) -> tuple[str | None, str | None, bool]:
        """Find the main Wikipedia page for a raga via direct lookup + search.

        Strategy:
        1. Try direct page fetches (exact name, aliases, transliteration variants)
        2. Fall back to Wikipedia search, but ONLY accept results whose title
           matches the target raga (never pick a random 'raga' page).
        3. Last resort: first search result as general context.

        Returns (title, content, is_exact_match).
        """
        canonical = _canonical_wikipedia_title(raga_name, aliases)
        if canonical:
            seen_canon: set[str] = set()
            for suffix in ["", " (raga)", " (Carnatic raga)", " (Carnatic)"]:
                candidate = canonical + suffix
                if candidate in seen_canon:
                    continue
                seen_canon.add(candidate)
                logger.info(f"Trying canonical Wikipedia title: '{candidate}'")
                result = self._try_direct_page(candidate)
                if result:
                    logger.info(f"Found direct page: '{result[0]}'")
                    return result[0], result[1], True

        base_names = []
        seen_bases: set[str] = set()
        all_raw = [raga_name] + aliases
        for n in all_raw + _transliteration_variants(raga_name):
            key = n.lower().rstrip("āīūēō")
            if key not in seen_bases:
                seen_bases.add(key)
                base_names.append(n)
        for a in aliases:
            for v in _transliteration_variants(a):
                key = v.lower()
                if key not in seen_bases:
                    seen_bases.add(key)
                    base_names.append(v)
        for raw in all_raw:
            for sv in _space_split_variants(raw):
                key = sv.lower()
                if key not in seen_bases:
                    seen_bases.add(key)
                    base_names.append(sv)

        suffixes = ["", " (raga)", " (Carnatic raga)", " (Carnatic)"]
        seen_attempts: set[str] = set()
        for base in base_names:
            for suffix in suffixes:
                candidate = base + suffix
                if candidate in seen_attempts:
                    continue
                seen_attempts.add(candidate)
                logger.debug(f"Trying direct page: '{candidate}'")
                result = self._try_direct_page(candidate)
                if result:
                    logger.info(f"Found direct page: '{result[0]}'")
                    return result[0], result[1], True

        search_queries = [
            f"{raga_name} raga Carnatic",
            f"{raga_name} raga",
            raga_name,
        ]
        for a in aliases:
            search_queries.append(f"{a} raga")
        for v in _transliteration_variants(raga_name):
            search_queries.append(f"{v} raga")
        for sv in _space_split_variants(raga_name):
            search_queries.append(f"{sv} raga")
        if melakarta_number:
            search_queries.append(f"Melakarta raga {melakarta_number}")

        last_results = []
        for query in search_queries:
            logger.info(f"Searching Wikipedia: '{query}'")
            results = self.search(query, limit=5)
            last_results = results
            for r in results:
                if _title_matches_raga(r["title"], raga_name, aliases):
                    logger.info(f"Search matched: '{r['title']}'")
                    return r["title"], None, True

        if last_results:
            fallback = last_results[0]["title"]
            logger.warning(
                f"No raga-specific page found for '{raga_name}', "
                f"using best search result: '{fallback}'"
            )
            return fallback, None, False
        return None, None, False

    def crawl_raga(
        self,
        raga_name: str,
        aliases: list[str] | None = None,
        melakarta_number: int | None = None,
        top_n: int = 5,
    ) -> list[WikiPage]:
        """
        Full crawl pipeline for a raga:
        1. Try direct page lookup with transliteration variants
        2. Fall back to search if direct lookup fails
        3. Fetch content + links from the main page
        4. Score and rank linked pages
        5. Fetch top-N linked pages
        6. Return all pages sorted by relevance
        """
        aliases = aliases or []
        pages: list[WikiPage] = []
        seen_titles: set[str] = set()

        main_title, prefetched_content, is_exact = self._find_main_page(
            raga_name, aliases, melakarta_number,
        )

        if not main_title:
            logger.warning(f"No Wikipedia page found for raga '{raga_name}'")
            return pages

        main_score = 100.0 if is_exact else 0.5
        logger.info(f"Main page: '{main_title}' (score: {main_score})")
        content = prefetched_content or self.fetch_page_content(main_title)
        if content:
            categories = self.fetch_page_categories(main_title)
            page = WikiPage(
                title=main_title,
                url=WIKI_BASE + quote(main_title.replace(" ", "_")),
                content=content if is_exact else content[:3000],
                summary=content[:500],
                categories=categories,
                relevance_score=main_score,
            )
            pages.append(page)
            seen_titles.add(main_title)

        logger.info(f"Fetching links from '{main_title}'")
        links = self.fetch_page_links(main_title)
        logger.info(f"Found {len(links)} links")

        scored_links = []
        for link_title in links:
            if link_title in seen_titles:
                continue
            score = self.score_link_relevance(link_title, raga_name, aliases)
            if score > 0:
                scored_links.append((link_title, score))

        scored_links.sort(key=lambda x: x[1], reverse=True)
        top_links = scored_links[:top_n]

        logger.info(f"Top {len(top_links)} relevant links:")
        for title, score in top_links:
            logger.info(f"  [{score:.1f}] {title}")

        for link_title, score in top_links:
            logger.info(f"Fetching linked page: '{link_title}'")
            link_content = self.fetch_page_content(link_title)
            if link_content and len(link_content) >= MIN_CONTENT_LENGTH:
                link_page = WikiPage(
                    title=link_title,
                    url=WIKI_BASE + quote(link_title.replace(" ", "_")),
                    content=link_content,
                    summary=link_content[:500],
                    relevance_score=score,
                )
                pages.append(link_page)
                seen_titles.add(link_title)
            elif link_content:
                logger.debug(f"Skipping '{link_title}': too short ({len(link_content)} chars)")

        has_raga_page = any(
            _title_matches_raga(p.title, raga_name, aliases) for p in pages
        )

        supplementary_searches = []
        if not has_raga_page:
            supplementary_searches.append(f"{raga_name} raga")
            for a in aliases:
                supplementary_searches.append(f"{a} raga")
        supplementary_searches += [
            f"{raga_name} compositions Carnatic",
            f"Tyagaraja compositions {raga_name}",
        ]
        for query in supplementary_searches:
            if len(pages) >= top_n + 3:
                break
            logger.info(f"Supplementary search: '{query}'")
            results = self.search(query, limit=3)
            for r in results:
                title = r["title"]
                if title in seen_titles:
                    continue
                supp_score = 50.0 if _title_matches_raga(title, raga_name, aliases) else 1.0
                content = self.fetch_page_content(title)
                if content and len(content) >= MIN_CONTENT_LENGTH:
                    pages.append(WikiPage(
                        title=title,
                        url=WIKI_BASE + quote(title.replace(" ", "_")),
                        content=content[:5000],
                        summary=content[:500],
                        relevance_score=supp_score,
                    ))
                    seen_titles.add(title)

        reference_titles = ["Melakarta", "Carnatic music"]
        for title in reference_titles:
            if title not in seen_titles and len(pages) < top_n + 4:
                content = self.fetch_page_content(title)
                if content and len(content) >= MIN_CONTENT_LENGTH:
                    pages.append(WikiPage(
                        title=title,
                        url=WIKI_BASE + quote(title.replace(" ", "_")),
                        content=content[:3000],
                        summary=content[:500],
                        relevance_score=0.5,
                    ))
                    seen_titles.add(title)

        pages.sort(key=lambda p: p.relevance_score, reverse=True)
        return pages

    def build_context(self, pages: list[WikiPage], max_chars: int | None = None) -> str:
        if max_chars is None:
            max_chars = int(os.environ.get("MAX_CONTEXT_CHARS", 25000))
        """Build a combined context string from crawled pages for LLM consumption."""
        parts = []
        total = 0
        for page in pages:
            header = f"\n{'='*60}\nSOURCE: {page.title}\nURL: {page.url}\nRELEVANCE SCORE: {page.relevance_score}\n{'='*60}\n"
            content = page.content
            remaining = max_chars - total - len(header)
            if remaining <= 0:
                break
            if len(content) > remaining:
                content = content[:remaining] + "\n... [TRUNCATED]"
            parts.append(header + content)
            total += len(header) + len(content)
        return "\n".join(parts)
