"""
LLM-based extraction pipeline that takes Wikipedia context about a Carnatic raga
and produces structured JSON output with all relevant information.

Uses OpenAI's API with structured output via Pydantic models.
Includes robust rate-limit handling with proactive pacing and automatic retry.
"""

import json
import logging
import os
import re
import time
from openai import OpenAI, RateLimitError, APIError, APITimeoutError, APIConnectionError
from pydantic import BaseModel, Field
from rich.console import Console

from melakarta_ragas import get_raga_info
from popular_janya_ids import (
    build_lookup_from_rows,
    load_json_rows_from_output_dir,
    normalize_popular_janya_list,
)
from raga_slug import slug_raga_id

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("ddgs").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_LLM_DELAY = 2.0
DEFAULT_MAX_CONTEXT_CHARS = 25000
MAX_RETRIES = 5

_console = Console(stderr=True)


def _sleep_with_progress(seconds: float, desc: str = "Waiting"):
    """Sleep with a Rich status spinner that cooperates with Rich logging."""
    if seconds <= 0:
        return
    total = int(seconds)
    if total < 1:
        time.sleep(seconds)
        return
    with _console.status(f"[bold cyan]⏳ {desc} — {total}s") as status:
        for elapsed in range(total):
            remaining = total - elapsed
            status.update(f"[bold cyan]⏳ {desc} — {remaining}s")
            time.sleep(1)
        remainder = seconds - total
        if remainder > 0:
            time.sleep(remainder)


def _raga_signature_from_dict(data: dict) -> tuple:
    return (
        (data.get("raga_name") or "").strip(),
        bool(data.get("is_melakarta")),
        data.get("parent_raga"),
        data.get("melakarta_number"),
    )


def _load_signature_from_path(path: str) -> tuple | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return _raga_signature_from_dict(data)
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _scan_raga_id_to_paths(output_dir: str) -> dict[str, list[str]]:
    """Map raga_id -> paths of JSON files that currently claim that id."""
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
            rid = (data.get("raga_id") or "").strip()
            if rid:
                out.setdefault(rid, []).append(path)
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _parent_slug(parent_raga: int | None) -> str | None:
    if parent_raga is None or not (1 <= parent_raga <= 72):
        return None
    try:
        name = get_raga_info(parent_raga)["name"]
        return slug_raga_id(name)
    except (KeyError, ValueError):
        return None


def _candidate_id_available(
    candidate: str,
    rid_to_paths: dict[str, list[str]],
    our_sig: tuple,
) -> bool:
    paths = rid_to_paths.get(candidate, [])
    if not paths:
        return True
    for p in paths:
        ps = _load_signature_from_path(p)
        if ps is None:
            return False
        if ps != our_sig:
            return False
    return True


class Composition(BaseModel):
    name: str = Field(description="Name of the composition")
    composer: str = Field(default="", description="Composer name")
    language: str = Field(default="", description="Language: Telugu, Sanskrit, Tamil, Kannada, Malayalam, etc.")


class RagaInfo(BaseModel):
    raga_id: str = Field(
        default="",
        description="Stable slug id (lowercase ASCII, underscores). Same meaning as template raga_id.",
    )
    raga_name: str = Field(description="Primary Carnatic name of the raga")
    other_names: list[str] = Field(default_factory=list, description="All alternative names/spellings/transliterations")
    melakarta_number: int | None = Field(default=None, description="Melakarta number (1-72) if this is a melakarta raga")
    is_melakarta: bool = Field(default=False, description="Whether this is a melakarta (parent) raga")
    parent_raga: int | None = Field(default=None, description="Melakarta number (1-72) of the parent raga, if this is a janya raga")
    inventor: str = Field(default="", description="Who formalized/popularized this raga, if known")
    chakra: str = Field(default="", description="Chakra name, e.g. 'Indu', 'Agni', 'Bana'")
    arohana: list[str] = Field(default_factory=list, description="Ascending scale as list of swaras, e.g. ['S', 'R1', 'G1', 'M1', 'P', 'D1', 'N1', '>S']. Use >S for upper shadjam.")
    avrohana: list[str] = Field(default_factory=list, description="Descending scale as list of swaras, e.g. ['>S', 'N1', 'D1', 'P', 'M1', 'G1', 'R1', 'S']. Use >S for upper shadjam.")
    raga_type: str = Field(default="", description="Classification: sampurna, audava, shadava, audava-sampurna, etc.")
    is_vakra: bool = Field(default=False, description="Whether the raga has vakra (zig-zag) prayoga in its scale")
    is_bhashanga: bool = Field(default=False, description="Whether the raga uses anya swaras (notes outside parent scale)")
    anya_swaras: list[str] = Field(default_factory=list, description="Any anya swaras (foreign notes) used")
    uses_vivadi_swaras: bool = Field(default=False, description="Whether the raga uses vivadi (discordant) swaras")
    rasa: list[str] = Field(default_factory=list, description="Rasa terms: Shanta, Bhakti, Karuna, Shringara, Veera, Roudra, Bhayanaka, Bibhatsa, Adbhuta, etc.")
    mood: str = Field(default="", description="Description of the raga's mood/rasa and comparison to common feelings and emotions")
    description: str = Field(default="", description="Description of the raga's nature, character, history, and significance")
    time_of_day: str = Field(default="", description="Best time of day to perform the raga: morning, afternoon, evening, night, dawn, etc.")
    gamaka_usage: str = Field(default="", description="How gamakas are used in the raga. Which swaras are used for gamakas and how, where to use strong or weak gamakas, etc.")
    hindustani_equivalent: str = Field(default="unknown", description="Equivalent raga in Hindustani music. If no equivalent, use 'unknown'.")
    western_equivalent: str = Field(default="unknown", description="Equivalent scale/mode in Western music. If no equivalent, use 'unknown'.")
    popular_janya_ragas: list[str] = Field(
        default_factory=list,
        description="Popular janya ragas for this melakarta: each entry is a raga_id (ASCII slug) present in the catalog, not a display name",
    )
    notable_compositions: list[Composition] = Field(default_factory=list, description="All notable compositions. Be exhaustive.")
    notable_features: str = Field(default="", description="Any other notable facts: graha bhedam, prati madhyama equivalent, pedagogical significance, etc.")
    wikipedia_url: str = Field(default="", description="URL of the most relevant Wikipedia page for this raga")


def _raga_signature_from_info(info: RagaInfo) -> tuple:
    """Identity tuple for disambiguating duplicate names (same raga may re-save the same raga_id)."""
    return (
        (info.raga_name or "").strip(),
        bool(info.is_melakarta),
        info.parent_raga,
        info.melakarta_number,
    )


def build_raga_id_candidates(raga_info: RagaInfo, base: str) -> list[str]:
    """Ordered slug candidates when `base` is already taken by another raga."""
    out: list[str] = []

    def add(x: str) -> None:
        if x and x not in out:
            out.append(x)

    add(base)
    if raga_info.is_melakarta and raga_info.melakarta_number is not None:
        add(f"{base}_m{raga_info.melakarta_number}")
    if not raga_info.is_melakarta and raga_info.parent_raga is not None:
        pn = _parent_slug(raga_info.parent_raga)
        if pn:
            add(f"{base}_{pn}")
        add(f"{base}_parent_{raga_info.parent_raga}")
    n = 2
    while len(out) < 200:
        add(f"{base}_{n}")
        n += 1
    return out


def resolve_unique_raga_id(raga_info: RagaInfo, output_dir: str) -> RagaInfo:
    """
    Ensure raga_id is unique among JSON files in output_dir.

    If another file already uses the same id for a *different* raga (name/parent/melakarta),
    pick the next candidate: melakarta may use ``{base}_m{N}``, janya ``{base}_{parent_slug}``, etc.
    """
    base = (raga_info.raga_id or "").strip() or slug_raga_id(raga_info.raga_name)
    rid_map = _scan_raga_id_to_paths(output_dir)
    our_sig = _raga_signature_from_info(raga_info)

    for cand in build_raga_id_candidates(raga_info, base):
        if _candidate_id_available(cand, rid_map, our_sig):
            if cand != (raga_info.raga_id or "").strip():
                logger.info(
                    "raga_id '%s' unavailable or ambiguous — using unique id '%s' for %r",
                    (raga_info.raga_id or base) or base,
                    cand,
                    raga_info.raga_name,
                )
            return raga_info.model_copy(update={"raga_id": cand})

    raise RuntimeError(f"Could not assign a unique raga_id for {raga_info.raga_name!r} (base={base!r})")


SYSTEM_PROMPT = """You are an expert in Carnatic music with encyclopedic knowledge of ragas, compositions, composers, and musical theory.

Extract ALL available structured information about a Carnatic raga from the provided content.

## Output Format

### parent_raga
- For janya (derived) ragas, set parent_raga to the melakarta number (1-72) of the parent raga.
- For melakarta ragas, leave as null.
- Example: Hamsadhvani is a janya of Shankarabharanam → parent_raga: 29

### raga_id
- Prefer a lowercase ASCII slug from `raga_name` (e.g. `shankarabharanam`). If you leave it empty, the pipeline fills it.
- The save step guarantees a **unique** id across all JSON files in `output/`; if the same name exists as both a melakarta and a janya, set `parent_raga` / `is_melakarta` correctly — disambiguation uses that.

### popular_janya_ragas (melakarta only)
- List **raga_id** strings for well-known janya ragas derived from this melakarta (e.g. `hamsadhvani`, `mohanam`), matching ids used elsewhere in the catalog — **not** long prose names.

### arohana / avrohana
- Return as a JSON list of individual swaras: ["S", "R1", "G1", "M1", "P", "D1", "N1", ">S"]
- Use >S for upper shadjam, <P for lower panchamam, etc.
- Use swara numbers 1-3 (R1, R2, R3, G1, G2, G3, M1, M2, D1, D2, D3, N1, N2, N3).

### rasa — IMPORTANT, ALWAYS FILL
- JSON list of rasa terms: ["Shanta", "Bhakti", "Karuna", "Shringara", "Veera", "Roudra", "Bhayanaka", "Bibhatsa", "Adbhuta"].
- Include ALL that apply. Every raga evokes at least one rasa — infer from the raga's character, swaras, and descriptions if not explicitly stated.
- Even if the text doesn't explicitly name a rasa, deduce it from mood descriptions (e.g., "soothing" → Shanta, "devotional" → Bhakti, "pathos" → Karuna, "romantic" → Shringara).

### mood — IMPORTANT, ALWAYS FILL
- A descriptive paragraph explaining the raga's mood, emotional quality, and how it relates to common human feelings and emotions.
- Go beyond rasa labels — describe what the listener might feel, what imagery the raga evokes, what time or setting it suits.
- Even for obscure ragas, describe the general emotional character based on the swara pattern and any available clues.

### description
- Combine all relevant info about the raga's nature, character, history, origin, significance, pedagogical role, graha bhedam relationships, prati madhyama equivalent, etc. into one rich description string.

### gamaka_usage — IMPORTANT, ALWAYS FILL
- Describe in detail how gamakas are applied in this raga.
- Specify which swaras oscillate (kampita), which are held flat/steady, which use slides (jaru/jaanta).
- Note if certain swaras must be stressed or avoided, or if there are special rules about swara treatment.
- Describe the difference in gamaka treatment compared to similar ragas if relevant.
- Even if the source text is sparse, infer gamaka patterns from the swara structure (e.g., vivadi swaras often use kampita, shuddha swaras may be held steady).

### notable_features
- Any additional facts not captured by other fields: graha bhedam details, vivadi swara usage, teaching significance, etc.

### Compositions — BE EXHAUSTIVE
- Extract EVERY composition mentioned in the text.
- For each: name, composer (full name), tala if stated, language if stated.
- Include full proper names of composers (e.g., "Muthuswami Dikshitar" not "Dikshitar").

### General Rules
- Extract ONLY information present or strongly implied in the provided text.
- If information is not available, leave the field as its default (empty string, empty list, false).
- Be thorough — capture everything the text offers."""


def _is_request_too_large(error: RateLimitError) -> tuple[bool, int, int]:
    """Check if the 429 is because the request itself exceeds the TPM limit.
    Returns (is_too_large, requested_tokens, limit_tokens)."""
    msg = str(error)
    match = re.search(r"Limit\s+(\d+).*?Requested\s+(\d+)", msg)
    if match:
        limit = int(match.group(1))
        requested = int(match.group(2))
        if requested > limit:
            return True, requested, limit
    return False, 0, 0


def _parse_retry_after(error: RateLimitError) -> float | None:
    """Extract wait time from a rate limit error message or headers."""
    msg = str(error)

    match = re.search(r"try again in (\d+(?:\.\d+)?)\s*m?s", msg, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if "ms" in msg[match.start():match.end() + 5].lower():
            return val / 1000.0
        return val

    match = re.search(r"retry.after[:\s]+(\d+(?:\.\d+)?)", msg, re.IGNORECASE)
    if match:
        return float(match.group(1))

    match = re.search(r"(\d+(?:\.\d+)?)\s*seconds?", msg, re.IGNORECASE)
    if match:
        return float(match.group(1))

    if hasattr(error, "response") and error.response is not None:
        retry_after = error.response.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass

    return None


class RateLimitPacer:
    """
    Proactive rate limit pacer that tracks API response headers
    and adds delays to stay under limits without hitting 429s.
    """

    def __init__(self, min_delay: float = DEFAULT_LLM_DELAY):
        self.min_delay = min_delay
        self._last_call_time: float = 0
        self._remaining_requests: int | None = None
        self._remaining_tokens: int | None = None
        self._reset_requests_at: float | None = None
        self._reset_tokens_at: float | None = None

    def update_from_headers(self, headers: dict):
        """Update internal state from OpenAI response headers."""
        now = time.time()

        if "x-ratelimit-remaining-requests" in headers:
            self._remaining_requests = int(headers["x-ratelimit-remaining-requests"])
        if "x-ratelimit-remaining-tokens" in headers:
            self._remaining_tokens = int(headers["x-ratelimit-remaining-tokens"])

        reset_req = headers.get("x-ratelimit-reset-requests")
        if reset_req:
            self._reset_requests_at = now + self._parse_duration(reset_req)

        reset_tok = headers.get("x-ratelimit-reset-tokens")
        if reset_tok:
            self._reset_tokens_at = now + self._parse_duration(reset_tok)

        logger.debug(
            f"Rate limits: requests_remaining={self._remaining_requests}, "
            f"tokens_remaining={self._remaining_tokens}"
        )

    @staticmethod
    def _parse_duration(duration_str: str) -> float:
        """Parse duration strings like '1m30s', '500ms', '2s'."""
        total = 0.0
        for match in re.finditer(r"(\d+(?:\.\d+)?)(ms|m|s|h)", duration_str):
            val = float(match.group(1))
            unit = match.group(2)
            if unit == "h":
                total += val * 3600
            elif unit == "m":
                total += val * 60
            elif unit == "s":
                total += val
            elif unit == "ms":
                total += val / 1000
        return total if total > 0 else 1.0

    def wait_if_needed(self):
        """Block until it's safe to make the next API call."""
        now = time.time()
        elapsed = now - self._last_call_time

        delay = self.min_delay

        if self._remaining_requests is not None and self._remaining_requests <= 2:
            if self._reset_requests_at and self._reset_requests_at > now:
                wait_for_reset = self._reset_requests_at - now + 1.0
                delay = max(delay, wait_for_reset)
                logger.info(
                    f"Only {self._remaining_requests} requests remaining, "
                    f"waiting {wait_for_reset:.1f}s for reset"
                )

        if self._remaining_tokens is not None and self._remaining_tokens < 5000:
            if self._reset_tokens_at and self._reset_tokens_at > now:
                wait_for_reset = self._reset_tokens_at - now + 1.0
                delay = max(delay, wait_for_reset)
                logger.info(
                    f"Only {self._remaining_tokens} tokens remaining, "
                    f"waiting {wait_for_reset:.1f}s for reset"
                )

        remaining_delay = delay - elapsed
        if remaining_delay > 0:
            logger.debug(f"Pacing: waiting {remaining_delay:.1f}s before next LLM call")
            _sleep_with_progress(remaining_delay, "Pacing LLM calls")

        self._last_call_time = time.time()


_pacer = RateLimitPacer(
    min_delay=float(os.environ.get("LLM_CALL_DELAY", DEFAULT_LLM_DELAY))
)


def get_pacer() -> RateLimitPacer:
    return _pacer


def reset_pacer(min_delay: float | None = None):
    global _pacer
    _pacer = RateLimitPacer(
        min_delay=min_delay or float(os.environ.get("LLM_CALL_DELAY", DEFAULT_LLM_DELAY))
    )


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


def _trim_context(context: str, target_chars: int) -> str:
    """Trim context to target_chars, keeping higher-scored pages (listed first)."""
    if len(context) <= target_chars:
        return context
    sections = re.split(r"\n={60}\nSOURCE:", context)
    if len(sections) <= 1:
        return context[:target_chars] + "\n... [TRIMMED TO FIT TOKEN LIMIT]"
    trimmed_parts = []
    total = 0
    for i, section in enumerate(sections):
        part = section if i == 0 else f"\n{'='*60}\nSOURCE:{section}"
        if total + len(part) > target_chars:
            remaining = target_chars - total
            if remaining > 500:
                trimmed_parts.append(part[:remaining] + "\n... [TRIMMED TO FIT TOKEN LIMIT]")
            break
        trimmed_parts.append(part)
        total += len(part)
    return "".join(trimmed_parts)


def extract_raga_info(
    raga_name: str,
    context: str,
    known_swaras: dict | None = None,
    melakarta_number: int | None = None,
    model: str | None = None,
) -> RagaInfo:
    """
    Extract structured raga information from Wikipedia context using an LLM.
    Auto-trims context to fit token limits and retries on transient errors.
    """
    if model is None:
        model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)

    max_context = int(os.environ.get("MAX_CONTEXT_CHARS", DEFAULT_MAX_CONTEXT_CHARS))
    context = _trim_context(context, max_context)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=0)

    def _build_messages(ctx: str) -> list[dict]:
        user_content = f"Extract all available information about the Carnatic raga '{raga_name}'"
        if melakarta_number:
            user_content += f" (Melakarta #{melakarta_number})"
        if known_swaras:
            user_content += f"\n\nKnown swaras from Melakarta system:\n"
            user_content += f"Arohana: {known_swaras.get('arohana_str', 'N/A')}\n"
            user_content += f"Avrohana: {known_swaras.get('avrohana_str', 'N/A')}\n"
        user_content += f"\n\n--- WIKIPEDIA CONTENT ---\n{ctx}\n--- END CONTENT ---"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    messages = _build_messages(context)
    logger.info(f"Calling LLM ({model}) for raga extraction...")
    logger.info(f"Context size: {len(context)} chars (~{_estimate_tokens(context)} tokens)")

    pacer = get_pacer()
    trimmed_once = False

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            pacer.wait_if_needed()

            response = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=RagaInfo,
                temperature=0.1,
            )

            if hasattr(response, "_raw_response") and hasattr(response._raw_response, "headers"):
                pacer.update_from_headers(dict(response._raw_response.headers))
            elif hasattr(response, "headers"):
                pacer.update_from_headers(dict(response.headers))

            raga_info = response.choices[0].message.parsed

            if known_swaras and not raga_info.arohana:
                arohana_str = known_swaras.get("arohana_str", "")
                raga_info.arohana = arohana_str.split() if arohana_str else []
            if known_swaras and not raga_info.avrohana:
                avrohana_str = known_swaras.get("avrohana_str", "")
                raga_info.avrohana = avrohana_str.split() if avrohana_str else []
            if melakarta_number:
                raga_info.melakarta_number = melakarta_number
                raga_info.is_melakarta = True

            return raga_info

        except RateLimitError as e:
            too_large, requested, limit = _is_request_too_large(e)

            if too_large:
                if trimmed_once:
                    raise RuntimeError(
                        f"Request still too large after trimming ({requested} tokens > {limit} TPM). "
                        f"Lower MAX_CONTEXT_CHARS in .env (currently {max_context}) or use a model with higher limits."
                    ) from e

                overshoot_ratio = requested / limit
                new_max = int(len(context) / overshoot_ratio * 0.8)
                logger.warning(
                    f"Request too large ({requested} tokens > {limit} TPM). "
                    f"Auto-trimming context from {len(context)} to {new_max} chars and retrying..."
                )
                context = _trim_context(context, new_max)
                messages = _build_messages(context)
                logger.info(f"Trimmed context: {len(context)} chars (~{_estimate_tokens(context)} tokens)")
                trimmed_once = True
                continue

            retry_after = _parse_retry_after(e)
            if retry_after is None:
                retry_after = min(30 * (2 ** (attempt - 1)), 300)

            wait_time = retry_after + 5.0
            logger.warning(
                f"Rate limited (attempt {attempt}/{MAX_RETRIES}). "
                f"Retry-After: {retry_after:.1f}s, waiting {wait_time:.1f}s..."
            )

            if attempt == MAX_RETRIES:
                raise

            _sleep_with_progress(wait_time, f"Rate limited — retry {attempt}/{MAX_RETRIES}")

        except (APITimeoutError, APIConnectionError) as e:
            backoff = min(10 * (2 ** (attempt - 1)), 120)
            logger.warning(
                f"API connection issue (attempt {attempt}/{MAX_RETRIES}): {e}. "
                f"Retrying in {backoff}s..."
            )
            if attempt == MAX_RETRIES:
                raise
            _sleep_with_progress(backoff, f"Connection error — retry {attempt}/{MAX_RETRIES}")

        except APIError as e:
            if e.status_code and e.status_code >= 500:
                backoff = min(15 * (2 ** (attempt - 1)), 120)
                logger.warning(
                    f"Server error {e.status_code} (attempt {attempt}/{MAX_RETRIES}). "
                    f"Retrying in {backoff}s..."
                )
                if attempt == MAX_RETRIES:
                    raise
                _sleep_with_progress(backoff, f"Server error {e.status_code} — retry {attempt}/{MAX_RETRIES}")
            else:
                raise


def save_raga_info(raga_info: RagaInfo, output_dir: str = "output") -> str:
    """Save extracted raga info to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    if not (raga_info.raga_id or "").strip():
        raga_info = raga_info.model_copy(update={"raga_id": slug_raga_id(raga_info.raga_name)})

    raga_info = resolve_unique_raga_id(raga_info, output_dir)

    if raga_info.is_melakarta and raga_info.popular_janya_ragas:
        rows = load_json_rows_from_output_dir(output_dir)
        ids, name_to_id = build_lookup_from_rows(rows)
        new_pj = normalize_popular_janya_list(raga_info.popular_janya_ragas, ids, name_to_id)
        if new_pj != raga_info.popular_janya_ragas:
            raga_info = raga_info.model_copy(update={"popular_janya_ragas": new_pj})

    safe_name = raga_info.raga_name.lower().replace(" ", "_")
    if raga_info.is_melakarta and raga_info.melakarta_number is not None:
        filename = f"{raga_info.melakarta_number:02d}_{safe_name}.json"
    else:
        filename = f"{raga_info.raga_id}.json"

    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(raga_info.model_dump(), f, indent=2, ensure_ascii=False)

    logger.info(f"Saved raga info to {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Phase 3: Gap analysis + web search + supplementary extraction
# ---------------------------------------------------------------------------

GAP_SEARCH_TEMPLATES: dict[str, list[str]] = {
    "mood": [
        "{raga} raga mood rasa emotion Carnatic",
        "{raga} ragam bhava feeling sentiment",
        "{raga} raga shanta bhakti karuna shringara",
        "{raga} raga evokes feeling character",
    ],
    "time_of_day": [
        "{raga} raga time of day performance Carnatic",
        "{raga} ragam morning evening night dawn",
    ],
    "gamaka_description": [
        "{raga} raga gamaka oscillation swara treatment Carnatic",
        "{raga} ragam gamakas kampita jaru how to sing",
        "{raga} raga lakshana swaras flat sharp held oscillate",
        "{raga} ragam alapana technique gamaka usage",
    ],
    "hindustani_equivalent": [
        "{raga} Carnatic Hindustani equivalent raga thaat",
        "{raga} raga North Indian equivalent",
    ],
    "western_equivalent": [
        "{raga} raga Western scale mode equivalent",
        "{raga} raga Western music comparison notes",
    ],
    "composition_details": [
        "{raga} raga compositions kriti composer language",
        "{raga} ragam famous compositions list",
    ],
    "janya_ragas": [
        "{raga} melakarta janya ragas derived",
    ],
    "parent_and_scale": [
        "{raga} raga arohana avrohana scale notes Carnatic",
        "{raga} ragam parent melakarta janya derived from",
        "{raga} raga lakshana swara positions ascending descending",
    ],
}


def identify_gaps(raga_info: RagaInfo) -> tuple[list[str], list[str]]:
    """Identify empty fields and return (gap_names, search_queries)."""
    name = raga_info.raga_name
    gaps: list[str] = []
    queries: list[str] = []

    def _add(field_key: str, label: str):
        gaps.append(label)
        for tmpl in GAP_SEARCH_TEMPLATES.get(field_key, []):
            queries.append(tmpl.format(raga=name))

    if not raga_info.rasa:
        _add("mood", "rasa")
    if not raga_info.mood:
        _add("mood", "mood")
    if not raga_info.time_of_day:
        _add("time_of_day", "time_of_day")
    if not raga_info.gamaka_usage:
        _add("gamaka_description", "gamaka_usage")
    if not raga_info.hindustani_equivalent or raga_info.hindustani_equivalent == "unknown":
        _add("hindustani_equivalent", "hindustani_equivalent")
    if not raga_info.western_equivalent or raga_info.western_equivalent == "unknown":
        _add("western_equivalent", "western_equivalent")

    if not raga_info.arohana or not raga_info.avrohana:
        _add("parent_and_scale", "arohana/avrohana")
    if not raga_info.is_melakarta and not raga_info.parent_raga:
        _add("parent_and_scale", "parent_raga")

    if raga_info.is_melakarta and not raga_info.popular_janya_ragas:
        _add("janya_ragas", "popular_janya_ragas")

    has_incomplete_compositions = any(
        not c.language for c in raga_info.notable_compositions
    )
    if has_incomplete_compositions or len(raga_info.notable_compositions) < 3:
        _add("composition_details", "notable_compositions")

    return gaps, queries


def web_search(queries: list[str], max_results_per_query: int = 3) -> tuple[str, list[dict]]:
    """Run DuckDuckGo searches and return (combined_text, list_of_result_dicts)."""
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("ddgs not installed (pip install ddgs), skipping web search")
        return "", []

    seen = set()
    snippets: list[str] = []
    result_list: list[dict] = []

    with DDGS() as ddgs:
        for query in queries:
            try:
                results = ddgs.text(query, max_results=max_results_per_query)
                for r in results:
                    title = r.get("title", "")
                    body = r.get("body", "")
                    href = r.get("href", "")
                    key = href or title
                    if key in seen:
                        continue
                    seen.add(key)
                    snippets.append(f"[{title}] ({href})\n{body}")
                    result_list.append({"title": title, "url": href, "snippet": body[:150]})
                time.sleep(0.5)
            except Exception as exc:
                logger.debug(f"Search failed for '{query}': {exc}")
                continue

    return "\n\n".join(snippets), result_list


SUPPLEMENT_PROMPT = """You are an expert in Carnatic music. You have an EXISTING extraction for a raga (provided as JSON).
Some fields are empty or incomplete. You are given ADDITIONAL web search snippets that may contain the missing information.

Your task: return a COMPLETE, UPDATED version of the raga JSON with all fields filled in where possible.

PRIORITY FIELDS — fill these even if you need to infer from context:
- rasa: JSON list of rasa terms. Every raga evokes at least one rasa. Infer from mood descriptions if not explicit (e.g., "soothing" → Shanta, "devotional" → Bhakti, "pathos" → Karuna).
- mood: Descriptive paragraph about the emotional experience — what the listener feels, what imagery it evokes. Not just rasa labels.
- gamaka_usage: DETAILED description of gamaka treatment — which specific swaras oscillate, which are held steady, slides, kampita, jaru patterns, special rules. Compare to similar ragas if helpful.

Other rules:
- KEEP all existing data that is already correct — do not remove or change it.
- FILL IN empty fields using information from the additional snippets.
- parent_raga: for janya ragas, set to the melakarta NUMBER (1-72), not the name. E.g. Shankarabharanam → 29, Kalyani → 65.
- popular_janya_ragas: for melakarta ragas only — list **raga_id** slugs (e.g. `hamsadhvani`), not display names. The save step maps names to ids when possible.
- arohana/avrohana: JSON list of swaras (e.g. ["S", "R2", "G3", "M1", "P", "D2", "N3", ">S"]).
- description: rich text combining nature, character, history, significance.
- notable_features: any additional facts (graha bhedam, prati madhyama equivalent, teaching use, etc.)
- hindustani_equivalent / western_equivalent: fill if found, otherwise keep as "unknown".
- If information is genuinely unavailable even in the snippets, leave the field as-is.
- Do NOT fabricate information. Only extract what is supported by the provided text."""


def fill_gaps(
    raga_info: RagaInfo,
    additional_context: str,
    model: str | None = None,
) -> RagaInfo:
    """Supplementary LLM call to fill in missing fields using web search results."""
    if model is None:
        model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)

    existing_json = json.dumps(raga_info.model_dump(), indent=2, ensure_ascii=False)

    max_snippet_chars = int(os.environ.get("MAX_CONTEXT_CHARS", DEFAULT_MAX_CONTEXT_CHARS))
    if len(additional_context) > max_snippet_chars:
        additional_context = additional_context[:max_snippet_chars]

    user_content = (
        f"Existing extraction for raga '{raga_info.raga_name}':\n"
        f"```json\n{existing_json}\n```\n\n"
        f"--- ADDITIONAL WEB SEARCH RESULTS ---\n{additional_context}\n--- END ---\n\n"
        f"Return the complete updated JSON with all fields filled where possible."
    )

    messages = [
        {"role": "system", "content": SUPPLEMENT_PROMPT},
        {"role": "user", "content": user_content},
    ]

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=0)
    pacer = get_pacer()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            pacer.wait_if_needed()
            response = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=RagaInfo,
                temperature=0.1,
            )

            if hasattr(response, "_raw_response") and hasattr(response._raw_response, "headers"):
                pacer.update_from_headers(dict(response._raw_response.headers))

            updated = response.choices[0].message.parsed

            if raga_info.melakarta_number:
                updated.melakarta_number = raga_info.melakarta_number
                updated.is_melakarta = True

            return updated

        except RateLimitError as e:
            too_large, requested, limit = _is_request_too_large(e)
            if too_large:
                logger.warning(
                    f"Supplementary request too large ({requested} > {limit} TPM). "
                    f"Trimming context and retrying..."
                )
                additional_context = additional_context[:len(additional_context) // 2]
                user_content = (
                    f"Existing extraction for raga '{raga_info.raga_name}':\n"
                    f"```json\n{existing_json}\n```\n\n"
                    f"--- ADDITIONAL WEB SEARCH RESULTS ---\n{additional_context}\n--- END ---\n\n"
                    f"Return the complete updated JSON with all fields filled where possible."
                )
                messages[1]["content"] = user_content
                continue

            retry_after = _parse_retry_after(e)
            if retry_after is None:
                retry_after = min(30 * (2 ** (attempt - 1)), 300)
            wait_time = retry_after + 5.0
            logger.warning(f"Rate limited (supplement attempt {attempt}/{MAX_RETRIES}). Waiting {wait_time:.1f}s...")
            if attempt == MAX_RETRIES:
                logger.warning("Supplementary extraction failed after retries, returning original")
                return raga_info
            _sleep_with_progress(wait_time, f"Rate limited — supplement retry {attempt}/{MAX_RETRIES}")

        except (APITimeoutError, APIConnectionError, APIError) as e:
            logger.warning(f"Supplementary extraction error (attempt {attempt}): {e}")
            if attempt == MAX_RETRIES:
                logger.warning("Supplementary extraction failed, returning original")
                return raga_info
            _sleep_with_progress(10, f"Retrying supplement — attempt {attempt}/{MAX_RETRIES}")

    return raga_info
