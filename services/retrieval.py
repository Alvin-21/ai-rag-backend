import json
import threading
from pathlib import Path
from typing import List, Dict, Tuple
from rapidfuzz import fuzz

# Load the JSON file once when the module is imported (not on every request)
DATA_PATH = Path(__file__).parent.parent / "data" / "knowledge_base.json"

# ─── Private hot-reload state ─────────────────────────────────────────────────
#
# These are MODULE-LEVEL variables — they live for the lifetime of the process.
# Prefixed with _ to signal "internal, don't import these directly".
#
# _kb_data  : the in-memory knowledge base (list of article dicts)
# _kb_mtime : the st_mtime value at the time of the last successful load
# _kb_lock  : prevents two threads reloading simultaneously

_kb_data: List[Dict] = []
_kb_mtime: float = 0.0
_kb_lock = threading.Lock()

FUZZY_THRESHOLD = 60
MAX_KEYWORD_SCORE = 30.0
KEYWORD_WEIGHT = 0.6
FUZZY_WEIGHT   = 0.4

def _load_kb() -> None:
    """
    Read DATA_PATH from disk and update the in-memory cache.

    Called on startup and whenever the file's mtime changes.
    NOT called directly by application code — use _get_kb() instead.

    Raises:
        FileNotFoundError  – if the JSON file doesn't exist yet
        json.JSONDecodeError – if the file contains invalid JSON

    Both exceptions are caught by _get_kb() so the server never crashes.
    """
    global _kb_data, _kb_mtime

    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # Only update state after a successful parse.
    # If json.load() raises, _kb_data and _kb_mtime are unchanged.
    _kb_data  = data
    _kb_mtime = DATA_PATH.stat().st_mtime

    print(
        f"[KB] ✓ Loaded {len(_kb_data)} article(s) from '{DATA_PATH.name}'"
    )

def _get_kb() -> List[Dict]:
    """
    Return the knowledge base, reloading from disk if the file changed.

    This is the single entry point used by all retrieval functions.

    Decision tree on every call:
      1. stat() the file → get current mtime              (~1 µs)
      2. Compare to stored _kb_mtime
         a. Different → acquire lock → reload → release   (~1 ms, rare)
         b. Same      → return cached _kb_data            (~0 µs)
      3. Return _kb_data

    Error handling:
      FileNotFoundError  → warn, return current cache (may be [])
      json.JSONDecodeError → warn, keep previous valid data
    """
    global _kb_mtime

    try:
        current_mtime = DATA_PATH.stat().st_mtime
    except FileNotFoundError:
        print(f"[KB] ⚠ '{DATA_PATH.name}' not found — returning cached data")
        return _kb_data

    # Fast path: file hasn't changed, return cache immediately
    if current_mtime == _kb_mtime:
        return _kb_data

    # Slow path: file changed — reload under the lock
    with _kb_lock:
        # Re-check inside the lock.
        # Another thread may have already reloaded while we waited to acquire.
        if current_mtime != _kb_mtime:
            print(f"[KB] ↻ Change detected in '{DATA_PATH.name}' — reloading…")
            try:
                _load_kb()
            except json.JSONDecodeError as e:
                # Bad JSON (e.g. file saved mid-edit) — keep previous data
                print(f"[KB] ⚠ JSON error: {e}")
                print(f"[KB]   Fix the file and save again to reload.")
                # Bump _kb_mtime so we don't retry on every request
                # until the file is saved again (mtime will change again).
                _kb_mtime = current_mtime

    return _kb_data

# ─── Initial load at import time ──────────────────────────────────────────────
#
# When uvicorn imports this module, we load the KB immediately so the first
# request is never slower than subsequent ones.
# If the file is missing at startup, we log a warning but don't crash —
# _get_kb() will retry on every request.

try:
    _load_kb()
except FileNotFoundError:
    print(
        f"[KB] ⚠ '{DATA_PATH}' not found at startup.\n"
        f"[KB]   Create the file and the next request will load it."
    )

def tokenize(text: str) -> set:
    """
    Split text into lowercase words and remove punctuation.
    
    "What's the Refund Policy?" → {"whats", "the", "refund", "policy"}
    
    This lets us compare words regardless of capitalization or punctuation.
    """
    # Remove punctuation and lowercase everything
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    return set(cleaned.split())


def score_article(question_tokens: set, article: Dict) -> float:
    """
    Calculate a relevance score for one article.
    
    We check overlap with:
    - The article title
    - The article content  
    - Manually set tags
    
    Tags get a higher weight because they're hand-picked for this purpose.
    """
    score = 0.0

    # Check title overlap
    title_tokens = tokenize(article["title"])
    score += len(question_tokens & title_tokens) * 2.0  # title matches are worth more

    # Check content overlap
    content_tokens = tokenize(article["content"])
    score += len(question_tokens & content_tokens) * 1.0

    # Check tag overlap — tags are curated keywords, so give them extra weight
    for tag in article.get("tags", []):
        tag_tokens = tokenize(tag)
        score += len(question_tokens & tag_tokens) * 3.0  # highest weight

    return score

def fuzzy_score_article(question: str, article: Dict) -> float:
    """
    Score an article using rapidfuzz fuzzy string matching.

    WHY TWO DIFFERENT rapidfuzz FUNCTIONS?

    fuzz.token_set_ratio → used for title + tags (short text)
        Works by comparing sets of tokens from both strings.
        Handles: word order differences, extra words, partial token overlap.
        "refund policy" vs "Refund Policy" → 100
        "can I get refund?" vs "Refund Policy" → ~85 (ignores "can I get")

    fuzz.partial_ratio → used for content (long text)
        Slides the shorter string (query) across the longer string (content)
        and returns the best substring similarity found.
        "refudn" vs "...request a full refund..." → ~83 (best substring match)
        This is better than token_set_ratio for short queries vs long text.

    Title and tags get 0.6 weight — they are concise, curated, and reliable.
    Content gets 0.4 weight — longer text means more noise.

    Returns a float from 0.0 to 100.0.
    """
    # Combine title and tags into one short searchable string
    title_tag_text = article["title"] + " " + " ".join(article.get("tags", []))

    # token_set_ratio: best for short, structured text (handles word order, extra words)
    title_tag_score = fuzz.token_set_ratio(question, title_tag_text)

    # partial_ratio: best for short query vs long content (finds best substring)
    content_score = fuzz.partial_ratio(question, article["content"])

    return title_tag_score * 0.6 + content_score * 0.4

def combined_score(kw_score: float, fz_score: float) -> float:
    """
    Merge the keyword and fuzzy scores into one comparable number (0-100).

    Steps:
      1. Normalise keyword score: divide by MAX_KEYWORD_SCORE, cap at 1.0.
         This puts keyword on the same 0–1 scale as fuzzy/100.
      2. Apply weights and scale back to 0–100.

    Example A — typo query ("refudn"), Refund Policy article:
        kw_score = 0    → normalised = 0.0
        fz_score = 78
        combined = 0.6 × 0 + 0.4 × 78 = 31.2

    Example B — exact query ("refund"), Refund Policy article:
        kw_score = 9.0  → normalised = min(9/30, 1) = 0.30 → 30/100
        fz_score = 95
        combined = 0.6 × 30 + 0.4 × 95 = 18 + 38 = 56.0

    The exact-match case scores higher in combined, which is correct behaviour.
    """
    normalised_kw = min(kw_score / MAX_KEYWORD_SCORE, 1.0) * 100
    return KEYWORD_WEIGHT * normalised_kw + FUZZY_WEIGHT * fz_score

def search_knowledge_base(question: str, top_k: int = 2) -> List[Dict]:
    """
    Return the top_k most relevant knowledge base articles for a question.

    INCLUSION RULE:
        An article is a candidate if:
          • keyword_score > 0  (at least one exact word matched), OR
          • fuzzy_score ≥ FUZZY_THRESHOLD  (strings are similar enough)

        This means:
          - Exact queries always find matches (keyword path).
          - Typo queries find matches when fuzzy similarity is high enough.
          - Completely unrelated queries return nothing (AI says "I don't know").

    RANKING:
        Candidates are sorted by combined_score() descending.
        Exact matches rank above fuzzy-only matches because the keyword
        component of the combined score adds a significant boost.

    Examples:
        search_knowledge_base("refund policy")
        → ["Refund Policy"]  (kw=9, fz=95, combined=56)

        search_knowledge_base("refudn")            ← typo
        → ["Refund Policy"]  (kw=0, fz=78, combined=31.2  — fuzzy saves it)

        search_knowledge_base("warantee broken")   ← two typos
        → ["Product Warranty"]  (kw=0, fz=71, combined=28.4)

        search_knowledge_base("xyzzy gibberish")
        → []  (kw=0, fz=22 < 65 threshold — correctly returns nothing)
    """
    kb = _get_kb()

    question_tokens = tokenize(question)

    # Score every article with both methods
    scored: List[Tuple[Dict, float, float, float]] = []

    for article in kb:
        kw = score_article(question_tokens, article)
        fz = fuzzy_score_article(question, article)
        cs = combined_score(kw, fz)
        scored.append((article, kw, fz, cs))

    # Filter: keep articles that pass either the keyword or fuzzy bar
    candidates = [
        (article, cs)
        for article, kw, fz, cs in scored
        if kw > 0 or fz >= FUZZY_THRESHOLD
    ]

    # Sort by combined score, highest first
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Return just the article dicts (callers don't need the scores)
    return [article for article, _ in candidates[:top_k]]