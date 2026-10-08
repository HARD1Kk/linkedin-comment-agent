import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from comment_generator import (
    generate_comment,
    is_company_account,
    extract_author_first_name,
    normalize_comment_text,
    passes_basic_checks,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("linkedin_agent")

OUTPUT_FILE = Path("post.json")
CANDIDATES_FILE = Path("candidates.json")
PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(filename: str) -> str:
    """Helper to load prompt template from prompts/ directory."""
    path = PROMPTS_DIR / filename
    return path.read_text(encoding="utf-8")


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/154.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Loads configuration from YAML file or returns default settings."""
    default_config: dict[str, Any] = {
        "niche_description": (
            "Software engineering, backend architecture, Python, AI developer tools, and system design."
        ),
        "target_topics": [
            "FastAPI observability",
            "Python performance",
            "MCP protocol",
            "RAG architecture",
            "LLM evaluation",
            "Kubernetes operators",
            "PostgreSQL optimization",
            "AI Agents architecture",
            "Backend system design",
        ],
        "whitelist_authors": ["Aniketsingh", "Scaler", "Guido van Rossum"],
        "blocklist_authors": ["Spam Bot", "Promotional Account"],
        "blocklist_keywords": ["we are hiring", "job alert", "discount code", "buy now"],
        "trending_topics_enabled": True,
        "serper_enabled": True,
        "serper_results_per_query": 10,
        "serper_timeframe": "qdr:w",
        "max_serper_calls_per_run": 20,
        "max_age_hours": 72,
        "max_comments_per_day": 5,
        "top_n_candidates": 5,
        "scoring_weights": {
            "recency": 35,
            "author_fit": 25,
            "topic_match": 25,
            "engagement": 15,
        },
    }


    path = Path(config_path)
    if not path.exists():
        return default_config

    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}

        for key, val in user_config.items():
            if (
                isinstance(val, dict)
                and key in default_config
                and isinstance(default_config[key], dict)
            ):
                merged_dict = dict(default_config[key])
                merged_dict.update(val)
                default_config[key] = merged_dict
            else:
                default_config[key] = val
    except Exception as exc:
        logger.warning(
            f"Error loading {config_path}: {exc}. Using default configuration."
        )

    return default_config


# ---------------------------------------------------------
# Helpers & Validation
# ---------------------------------------------------------

def validate_linkedin_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    hostname = (parsed.hostname or "").lower()
    return hostname == "linkedin.com" or hostname.endswith(".linkedin.com")


def normalize_url(url: str | None) -> str | None:
    """
    Normalizes a LinkedIn URL by stripping query parameters, fragments,
    standardizing scheme to https, removing www/language subdomains, and trailing slashes.
    """
    if not url or not isinstance(url, str):
        return None
    clean = url.strip()
    if not clean:
        return None
    try:
        parsed = urlparse(clean)
        scheme = "https"
        netloc = (parsed.hostname or "").lower()
        if "linkedin.com" in netloc:
            netloc = "linkedin.com"

        path = parsed.path.rstrip("/")
        if not path:
            return f"{scheme}://{netloc}"
        return f"{scheme}://{netloc}{path}".lower()
    except Exception:
        return clean.split("?")[0].split("#")[0].rstrip("/").lower()


def build_candidate_dedup_index(
    existing_candidates: list[dict[str, Any]]
) -> tuple[set[int], set[str], set[str]]:
    """
    Builds lookup sets of numeric Snowflake post IDs, normalized URLs, and author names from candidate database.
    Returns (seen_post_ids, seen_normalized_urls, seen_authors).
    """
    seen_post_ids: set[int] = set()
    seen_normalized_urls: set[str] = set()
    seen_authors: set[str] = set()

    for item in existing_candidates:
        if not isinstance(item, dict):
            continue

        # Extract numeric Snowflake post_id
        pid = item.get("post_id")
        if pid is not None:
            try:
                seen_post_ids.add(int(pid))
            except (ValueError, TypeError):
                pass

        # Also extract post_id & normalized url from url and canonical_url fields
        for field in ("url", "canonical_url"):
            val = item.get(field)
            if val and isinstance(val, str):
                norm = normalize_url(val)
                if norm:
                    seen_normalized_urls.add(norm)
                extracted_pid = extract_linkedin_post_id(val)
                if extracted_pid:
                    seen_post_ids.add(extracted_pid)

        # Extract author name
        author = item.get("author")
        if author and isinstance(author, str):
            clean_a = author.strip().lower()
            if clean_a:
                seen_authors.add(clean_a)

    return seen_post_ids, seen_normalized_urls, seen_authors



def clean_text(text: str | None) -> str | None:
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    return text.strip() or None


def format_post_text(text: str | None) -> str | None:
    if not text:
        return None
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() or None


def get_meta(soup: BeautifulSoup, *names: str) -> str | None:
    for name in names:
        tag = soup.find("meta", attrs={"property": name})
        if tag and tag.get("content"):
            return tag["content"].strip() or None
        tag = soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            return tag["content"].strip() or None
    return None


def extract_hashtags(text: str | None) -> list[str]:
    if not text:
        return []
    hashtags = re.findall(r"#\w+", text)
    return list(dict.fromkeys(hashtags))


def clean_post_text(text: str | None) -> str | None:
    if not text:
        return None
    text = clean_text(text)
    if not text:
        return None

    ui_markers = [
        "Report this post",
        "Like",
        "Comment",
        "Share",
        "Copy",
        "LinkedIn",
        "Facebook",
        "X",
        "Reply",
        "Reaction",
        "Repost",
        "Send",
        "Follow",
    ]
    cut_positions = []
    for marker in ui_markers:
        position = text.find(marker)
        if position != -1:
            cut_positions.append(position)

    if cut_positions:
        text = text[: min(cut_positions)]
    return clean_text(text)


# ---------------------------------------------------------
# Author Resolution & Brand/Company Classification
# ---------------------------------------------------------

def is_valid_author_name(name: str | None) -> bool:
    if not name:
        return False
    clean = name.strip()
    if not clean:
        return False
    lower = clean.lower()
    if lower in {
        "linkedin",
        "linkedin member",
        "unknown",
        "none",
        "n/a",
        "report this post",
    }:
        return False
    if (
        "comment" in lower
        or "reaction" in lower
        or "follower" in lower
        or lower.startswith("#")
    ):
        return False
    return True


def extract_author_from_url(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"/posts/([a-zA-Z0-9%\._\-]+)", url)
    if not match:
        return None

    raw_slug = match.group(1)
    user_slug = raw_slug.split("_")[0] if "_" in raw_slug else raw_slug

    if user_slug.lower().startswith("its-"):
        user_slug = user_slug[4:]
    elif user_slug.lower().startswith("its"):
        user_slug = user_slug[3:]

    user_slug = re.sub(r"-[a-f0-9]{7,15}$", "", user_slug, flags=re.IGNORECASE)
    user_slug = re.sub(r"\d+$", "", user_slug)
    user_slug = re.sub(r"([a-z])([A-Z])", r"\1 \2", user_slug)

    parts = [p.capitalize() for p in re.split(r"[-._]+", user_slug) if p]
    if not parts:
        return None

    name = " ".join(parts)
    if not is_valid_author_name(name):
        return None
    return name


def extract_author(
    article_text: str | None,
    soup: BeautifulSoup | None = None,
    url: str | None = None,
) -> str | None:
    if soup:
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                if isinstance(data, dict):
                    author_obj = (
                        data.get("author")
                        or data.get("creator")
                        or data.get("publisher")
                    )
                    if isinstance(author_obj, dict) and author_obj.get("name"):
                        candidate = clean_text(author_obj["name"])
                        if is_valid_author_name(candidate):
                            return candidate
                    elif isinstance(author_obj, str):
                        candidate = clean_text(author_obj)
                        if is_valid_author_name(candidate):
                            return candidate
            except Exception:
                pass

    if article_text:
        match = re.match(
            r"^(.*?)\s+[\d,.]+[KMB]?\s+followers\b",
            article_text,
            flags=re.IGNORECASE,
        )
        if match:
            candidate = clean_text(match.group(1))
            if is_valid_author_name(candidate):
                return candidate

    if soup:
        author_meta = get_meta(soup, "author", "article:author")
        if is_valid_author_name(author_meta):
            return author_meta

        first_name = get_meta(soup, "profile:first_name")
        last_name = get_meta(soup, "profile:last_name")
        if first_name and last_name:
            candidate = f"{first_name} {last_name}".strip()
            if is_valid_author_name(candidate):
                return candidate

        og_title = get_meta(soup, "og:title", "twitter:title")
        if og_title and "|" in og_title:
            candidate = og_title.split("|")[-1].strip()
            if is_valid_author_name(candidate):
                return candidate

        title = clean_text(soup.title.get_text()) if soup.title else None
        if title and "on LinkedIn" in title:
            candidate = title.split("on LinkedIn")[0].strip()
            if is_valid_author_name(candidate):
                return candidate
        elif title and "|" in title:
            candidate = title.split("|")[-1].strip()
            if is_valid_author_name(candidate):
                return candidate

    if url:
        candidate_from_url = extract_author_from_url(url)
        if is_valid_author_name(candidate_from_url):
            return candidate_from_url

    return None




# ---------------------------------------------------------
# Freshness & Post ID Timestamp Decoding
# ---------------------------------------------------------

def extract_linkedin_post_id(url: str | None) -> int | None:
    """Extracts 19-digit LinkedIn Snowflake Post/Activity ID from URL."""
    if not url:
        return None
    match = re.search(r"\b(\d{18,20})\b", url)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def post_id_to_datetime(post_id: int | None) -> datetime | None:
    """Decodes 64-bit LinkedIn Snowflake ID (timestamp_ms = post_id >> 22) into UTC datetime."""
    if not post_id or post_id <= 0:
        return None
    try:
        timestamp_ms = post_id >> 22
        timestamp_sec = timestamp_ms / 1000.0
        # Check realistic timestamp bounds (2010 to 2038)
        if 1262304000 <= timestamp_sec <= 2147483647:
            return datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
    except Exception:
        pass
    return None


def parse_age_in_hours(age_str: str | None) -> float | None:
    if not age_str:
        return None
    match = re.search(
        r"(\d+)\s*(months?|mo|weeks?|w|days?|d|hours?|hrs?|h|minutes?|mins?|m)\b",
        age_str.lower().strip(),
    )
    if not match:
        return None
    val = float(match.group(1))
    unit = match.group(2)
    if unit in {"m", "min", "mins", "minute", "minutes"}:
        return val / 60.0
    elif unit in {"h", "hr", "hrs", "hour", "hours"}:
        return val
    elif unit in {"d", "day", "days"}:
        return val * 24.0
    elif unit in {"w", "week", "weeks"}:
        return val * 24.0 * 7.0
    elif unit in {"mo", "month", "months"}:
        return val * 24.0 * 30.0
    return None


def calculate_post_age_hours(
    url: str | None, html_age_str: str | None = None
) -> tuple[float | None, str | None, str]:
    """
    Computes primary post age in hours.
    Primary: URL Snowflake ID (post_id >> 22).
    Fallback: Parsed HTML page text.
    Returns: (hours, formatted_age_display, source_indicator)
    """
    post_id = extract_linkedin_post_id(url)
    if post_id:
        post_dt = post_id_to_datetime(post_id)
        if post_dt:
            now_dt = datetime.now(timezone.utc)
            total_seconds = (now_dt - post_dt).total_seconds()
            if total_seconds >= 0:
                hours = round(total_seconds / 3600.0, 1)
                display_str = f"{hours}h (URL ID)"
                return hours, display_str, "url_id"

    if html_age_str:
        hours = parse_age_in_hours(html_age_str)
        if hours is not None:
            hours_r = round(hours, 1)
            return hours_r, f"{hours_r}h (page text)", "page_text"

    return None, None, "unknown"


# ---------------------------------------------------------
# Extract Follower Count & Engagement
# ---------------------------------------------------------

def extract_followers(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(
        r"([\d,.]+[KMB]?)\s+followers",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    value = match.group(1).replace(",", "")
    try:
        if value.upper().endswith("K"):
            return int(float(value[:-1]) * 1_000)
        if value.upper().endswith("M"):
            return int(float(value[:-1]) * 1_000_000)
        if value.upper().endswith("B"):
            return int(float(value[:-1]) * 1_000_000_000)
        return int(float(value))
    except ValueError:
        return None


def extract_post_age(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(
        r"\b(\d+\s*(?:m|min|h|hr|d|w))\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)
    return None


def extract_engagement(text: str | None) -> dict[str, int | None]:
    if not text:
        return {"reactions": None, "comments": None, "reposts": None}
    reactions = None
    comments = None
    reposts = None

    match = re.search(
        r"(\d+)\s+(?:\d+\s+)?Comments?\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        reactions = int(match.group(1))

    match = re.search(
        r"(\d+)\s+Comments?\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        comments = int(match.group(1))

    match = re.search(
        r"(\d+)\s+(?:Repost|Reposts)\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        reposts = int(match.group(1))

    return {"reactions": reactions, "comments": comments, "reposts": reposts}


# ---------------------------------------------------------
# Main Page Extraction
# ---------------------------------------------------------

def extract_post_data(url: str, html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    description = get_meta(
        soup, "og:description", "description", "twitter:description"
    )
    image = get_meta(soup, "og:image", "twitter:image")

    canonical = None
    canonical_tag = soup.find("link", rel="canonical")
    if canonical_tag:
        canonical = canonical_tag.get("href")
    if not canonical:
        canonical = get_meta(soup, "og:url", "al:android:url")

    article = soup.find("article")
    raw_article_text = None
    if article:
        raw_article_text = clean_text(article.get_text(" ", strip=True))

    if description:
        post_text = format_post_text(description)
    elif raw_article_text:
        post_text = clean_post_text(raw_article_text)
    else:
        post_text = None

    hashtags = extract_hashtags(post_text)
    author = extract_author(raw_article_text, soup, url)
    followers = extract_followers(raw_article_text)
    html_age_str = extract_post_age(raw_article_text)

    post_id = extract_linkedin_post_id(url)
    age_hours, display_age, age_source = calculate_post_age_hours(url, html_age_str)
    engagement = extract_engagement(raw_article_text)

    return {
        "url": url,
        "canonical_url": canonical,
        "post_id": post_id,
        "author": author,
        "followers": followers,
        "post_age": display_age or html_age_str,
        "post_age_hours": age_hours,
        "post_age_source": age_source,
        "post_text": post_text,
        "hashtags": hashtags,
        "engagement": engagement,
        "image": image,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_url(url: str) -> tuple[int, str]:
    time.sleep(1.0)  # Polite delay between HTTP fetches
    with httpx.Client(
        headers=HEADERS,
        follow_redirects=True,
        timeout=20.0,
    ) as client:
        response = client.get(url)
        return response.status_code, response.text


# ---------------------------------------------------------
# Dynamic Topic Discovery
# ---------------------------------------------------------

TRENDING_TOPIC_COUNT = 15
MIN_DYNAMIC_TOPICS = 5

FALLBACK_TOPICS = [
    "FastAPI async patterns",
    "Python GIL removal",
    "MCP protocol",
    "RAG architecture",
    "LLM evaluation",
    "Kubernetes operators",
    "PostgreSQL optimization",
    "AI Agents architecture",
    "Backend system design",
    "Redis caching strategies",
    "gRPC vs REST",
    "Event-driven architecture",
    "Vector databases",
    "Prompt engineering",
    "LLM fine-tuning",
    "Observability OpenTelemetry",
    "CQRS pattern",
    "GraphQL federation",
    "WebAssembly WASM",
    "Rust for Python devs",
    "Temporal workflows",
    "Feature flags",
    "A/B testing infrastructure",
    "Rate limiting algorithms",
    "Idempotency keys",
    "Saga pattern distributed transactions",
    "LLM routing strategies",
    "Embedding models comparison",
    "Streaming data pipelines",
    "Platform engineering",
    "Service mesh Istio",
    "eBPF networking",
    "Dapr microservices",
    "Ray distributed computing",
    "Milvus vector search",
    "LangChain agents",
    "Semantic caching",
    "Structured output LLM",
    "Model distillation",
    "Spec-driven development",
    "Database sharding",
    "Change data capture",
    "Backpressure reactive streams",
    "LLM guardrails",
    "Tool use function calling",
    "Context window optimization",
    "Multi-tenant architecture",
    "Zero-downtime deployments",
    "Chaos engineering",
]

GENERIC_TOPIC_BLACKLIST = {
    "technology",
    "software",
    "ai",
    "programming",
    "development",
    "tech news",
    "computer science",
    "coding",
    "tech",
    "engineering",
}


def normalize_topics(raw_topics: list[str]) -> list[str]:
    seen = set()
    normalized = []
    for t in raw_topics:
        if not t or not isinstance(t, str):
            continue
        cleaned = re.sub(r"^\d+[\.\)\-]\s*", "", t).strip().strip('"').strip("'")
        if not cleaned:
            continue
        lower = cleaned.lower()
        if lower in GENERIC_TOPIC_BLACKLIST:
            continue
        if lower not in seen:
            seen.add(lower)
            formatted = cleaned.title() if cleaned.islower() else cleaned
            normalized.append(formatted)
    return normalized


def discover_trending_topics(niche_description: str = "") -> tuple[list[str], int, int]:
    logger.info("Gathering recent public web search signals & GitHub Trending for dynamic topic discovery...")
    print("\n🌐 Gathering recent public web search signals & GitHub Trending repos for trending tech topics...")

    # Build targeted queries dynamically incorporating the configured niche description
    niche_words = [
        w for w in re.findall(r"\b[A-Za-z0-9+#\-]{3,}\b", niche_description)
        if w.lower() not in {"and", "the", "for", "with", "software", "engineering", "tools", "developer", "system"}
    ]

    queries = []
    if niche_words:
        for i in range(0, min(len(niche_words), 4), 2):
            pair = " ".join(niche_words[i : i + 2])
            queries.append(f"trending {pair} developer tools 2026")

    base_queries = [
        "trending programming technologies developer tools 2026",
        "latest software engineering frameworks AI developer tools 2026",
        "system design patterns distributed systems 2026",
        "LLM engineering RAG evaluation agents 2026",
        "Kubernetes cloud native DevOps platform engineering 2026",
        "PostgreSQL database optimization indexing 2026",
        "MCP protocol AI tool integration 2026",
    ]
    for bq in base_queries:
        if len(queries) < 6:
            queries.append(bq)

    # GitHub Trending queries via Serper / Google search
    gh_queries = [
        "site:github.com/trending programming",
        "site:github.com/trending python",
        "site:github.com/trending javascript",
        "site:github.com/trending ai",
    ]

    all_queries = queries + gh_queries
    snippets = []

    serper_key = os.getenv("SERPER_API_KEY") or os.getenv("SEARCH_API_KEY")
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache = load_serper_cache()

    if serper_key:
        for q in all_queries:
            cache_key = f"trending::{q}::{today_str}"
            try:
                if cache_key in cache:
                    organic_items = cache[cache_key]
                else:
                    url = "https://google.serper.dev/search"
                    headers = {
                        "X-API-KEY": serper_key,
                        "Content-Type": "application/json",
                    }
                    payload = {"q": q, "num": 6}
                    resp = httpx.post(url, headers=headers, json=payload, timeout=10.0)
                    organic_items = []
                    if resp.status_code == 200:
                        organic_items = resp.json().get("organic", [])
                        cache[cache_key] = organic_items
                        save_serper_cache(cache)

                for item in organic_items:
                    if isinstance(item, dict):
                        title = item.get("title", "")
                        snippet = item.get("snippet", "")
                        if title or snippet:
                            prefix = "GitHub Repo Trending: " if "github.com" in q else ""
                            snippets.append(f"{prefix}Title: {title}\nSnippet: {snippet}")
            except Exception as exc:
                logger.warning(f"Serper web query '{q}' for trending topics failed: {exc}")

    if not snippets:
        try:
            from ddgs import DDGS

            ddgs = DDGS()
            for q in all_queries:
                try:
                    res = list(ddgs.text(q, max_results=6))
                    for r in res:
                        title = r.get("title", "")
                        body = r.get("body", "")
                        if title or body:
                            prefix = "GitHub Repo Trending: " if "github.com" in q else ""
                            snippets.append(f"{prefix}Title: {title}\nSnippet: {body}")
                except Exception as exc:
                    logger.warning(f"Public web query '{q}' failed: {exc}")
        except Exception as exc:
            logger.warning(f"DDGS initialization error: {exc}")

    dynamic_topics = []
    if snippets:
        context = "\n\n".join(snippets[:10])
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            prompt_template = load_prompt("trending_topics_prompt.txt")
            niche_text = (
                niche_description.strip()
                if niche_description and niche_description.strip()
                else "Software engineering, backend architecture, Python, AI developer tools"
            )
            try:
                prompt = prompt_template.format(
                    context=context,
                    count=TRENDING_TOPIC_COUNT,
                    niche_description=niche_text,
                )
            except KeyError as ke:
                logger.warning(f"Missing key in trending_topics_prompt.txt template: {ke}")
                prompt = (
                    prompt_template.replace("{context}", context)
                    .replace("{count}", str(TRENDING_TOPIC_COUNT))
                    .replace("{niche_description}", niche_text)
                )

            try:
                from groq import Groq

                client = Groq(api_key=api_key)
                comp = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                raw_output = comp.choices[0].message.content.strip()
                match = re.search(r"\[.*\]", raw_output, re.DOTALL)
                if match:
                    extracted_list = json.loads(match.group(0))
                    if isinstance(extracted_list, list):
                        dynamic_topics = normalize_topics(extracted_list)
            except Exception as exc:
                logger.warning(f"Dynamic topic extraction via Groq failed: {exc}")
                print(f"⚠️ Trending topic discovery failed: {exc}. Using fallback topics.")

    dynamic_count = len(dynamic_topics)
    fallback_added = 0
    final_topics = list(dynamic_topics)

    if len(final_topics) < MIN_DYNAMIC_TOPICS:
        # Rotate fallback topics by day-of-year so different topics are used each day
        import random
        day_of_year = datetime.now(timezone.utc).timetuple().tm_yday
        rotation_offset = day_of_year % len(FALLBACK_TOPICS)
        rotated_fallbacks = FALLBACK_TOPICS[rotation_offset:] + FALLBACK_TOPICS[:rotation_offset]
        # Shuffle deterministically based on the day so order varies but is reproducible
        rng = random.Random(day_of_year)
        rng.shuffle(rotated_fallbacks)

        needed = MIN_DYNAMIC_TOPICS - len(final_topics)
        for ft in rotated_fallbacks:
            if ft.lower() not in {t.lower() for t in final_topics}:
                final_topics.append(ft)
                fallback_added += 1
                if fallback_added >= needed:
                    break

    return final_topics, dynamic_count, fallback_added


# ---------------------------------------------------------
# Candidate Discovery & Topic Attribution
# ---------------------------------------------------------

SERPER_CACHE_FILE = Path("serper_cache.json")


def load_serper_cache(cache_file: Path | str | None = None) -> dict[str, list[str]]:
    """Loads Serper search cache from JSON file."""
    path = Path(cache_file) if cache_file else SERPER_CACHE_FILE
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning(f"Error reading Serper cache {path}: {exc}")
    return {}


def save_serper_cache(
    cache_data: dict[str, list[str]], cache_file: Path | str | None = None
) -> None:
    """Saves Serper search cache to JSON file."""
    path = Path(cache_file) if cache_file else SERPER_CACHE_FILE
    try:
        path.write_text(
            json.dumps(cache_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning(f"Error saving Serper cache {path}: {exc}")



def filter_linkedin_post_urls(urls: list[str]) -> list[str]:
    """Filters list of URLs, keeping only those containing 'linkedin.com/posts/'."""
    kept: list[str] = []
    for u in urls:
        if u and isinstance(u, str) and "linkedin.com/posts/" in u.lower():
            kept.append(u)
    return kept


def search_serper(
    query: str,
    num: int = 10,
    timeframe: str = "qdr:w",
    api_key: str | None = None,
) -> list[str]:
    """
    Queries google.serper.dev/search API and returns organic link URLs
    filtered to contain 'linkedin.com/posts/'.
    """
    key = api_key or os.getenv("SERPER_API_KEY")
    if not key:
        raise ValueError("SERPER_API_KEY environment variable is missing.")

    url = "https://google.serper.dev/search"
    payload = {
        "q": query,
        "num": num,
        "tbs": timeframe,
    }
    headers = {
        "X-API-KEY": key,
        "Content-Type": "application/json",
    }

    response = httpx.post(url, headers=headers, json=payload, timeout=15.0)
    if response.status_code != 200:
        raise RuntimeError(
            f"Serper API returned non-200 status code {response.status_code}: {response.text}"
        )

    data = response.json()
    organic_results = data.get("organic", [])

    links: list[str] = []
    for item in organic_results:
        if isinstance(item, dict):
            link = item.get("link")
            if link and isinstance(link, str):
                links.append(link)

    return links


def discover_linkedin_urls(
    topics: list[str],
    config: dict[str, Any] | None = None,
    existing_candidates: list[dict[str, Any]] | None = None,
) -> list[tuple[str, str]]:
    """Discovers public LinkedIn post URLs for target topics using Serper API with DDGS fallback."""
    candidates: list[tuple[str, str]] = []

    # Load candidate DB for early deduplication if not explicitly passed
    db = existing_candidates if existing_candidates is not None else load_candidates_db()
    seen_post_ids, seen_normalized_urls, _ = build_candidate_dedup_index(db)

    cfg = config or {}
    serper_enabled = cfg.get("serper_enabled", True)
    num = int(cfg.get("serper_results_per_query", 10))
    timeframe = str(cfg.get("serper_timeframe", "qdr:w"))
    max_calls = int(cfg.get("max_serper_calls_per_run", 20))

    serper_key = os.getenv("SERPER_API_KEY") or os.getenv("SEARCH_API_KEY")
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cache = load_serper_cache()
    serper_calls_count = 0
    early_duplicates_skipped = 0

    print("\n🌐 Initiating public discovery for LinkedIn posts...")

    for topic in topics:
        serper_success = False
        organic_links: list[str] = []
        kept_links: list[str] = []

        # Build Serper query: unquoted topic
        serper_query = f"site:linkedin.com/posts {topic}"
        cache_key = f"{serper_query}::{today_str}"

        if serper_key and serper_enabled:
            if cache_key in cache:
                organic_links = cache[cache_key]
                kept_links = filter_linkedin_post_urls(organic_links)
                serper_success = True
                logger.info(
                    f"[Serper Cache Hit] Topic '{topic}': {len(organic_links)} returned, {len(kept_links)} kept"
                )
                print(
                    f"  📌 [Serper Cache Hit] Topic '{topic}': {len(organic_links)} returned, {len(kept_links)} kept"
                )
            elif serper_calls_count < max_calls:
                time.sleep(0.5)
                try:
                    serper_calls_count += 1
                    organic_links = search_serper(
                        serper_query, num=num, timeframe=timeframe, api_key=serper_key
                    )
                    cache[cache_key] = organic_links
                    save_serper_cache(cache)
                    kept_links = filter_linkedin_post_urls(organic_links)
                    serper_success = True
                    logger.info(
                        f"[Serper] Topic '{topic}': {len(organic_links)} returned, {len(kept_links)} kept"
                    )
                    print(
                        f"  📌 [Serper] Topic '{topic}': {len(organic_links)} returned, {len(kept_links)} kept"
                    )
                except Exception as exc:
                    logger.warning(f"Serper API query for topic '{topic}' failed: {exc}")
                    print(
                        f"  ⚠️ Serper query failed for topic '{topic}': {exc}. Falling back to free search."
                    )
            else:
                logger.info(
                    f"Max Serper call limit ({max_calls}) reached. Skipping Serper for topic '{topic}'."
                )
                print(
                    f"  ⚠️ Max Serper call limit ({max_calls}) reached. Falling back to free search."
                )

            if serper_success:
                for link in kept_links:
                    norm = normalize_url(link)
                    pid = extract_linkedin_post_id(link)
                    if (pid and pid in seen_post_ids) or (norm and norm in seen_normalized_urls):
                        early_duplicates_skipped += 1
                        logger.info(f"Early deduplication: Skipping candidate link {link} (post_id: {pid})")
                        continue
                    if pid:
                        seen_post_ids.add(pid)
                    if norm:
                        seen_normalized_urls.add(norm)
                    candidates.append((link, topic))
                continue

        # Fallback to free DDGS search if Serper is disabled, missing key, failed, or returned zero kept URLs
        ddgs_query = f'site:linkedin.com/posts "{topic}"'
        try:
            from ddgs import DDGS

            ddgs = DDGS()
            time.sleep(0.5)
            results = list(ddgs.text(ddgs_query, max_results=8))
            ddgs_returned = len(results)
            found_count = 0
            for r in results:
                href = r.get("href", "")
                if (
                    "linkedin.com/posts/" in href
                    or "linkedin.com/feed/update/" in href
                ):
                    norm = normalize_url(href)
                    pid = extract_linkedin_post_id(href)
                    if (pid and pid in seen_post_ids) or (norm and norm in seen_normalized_urls):
                        early_duplicates_skipped += 1
                        logger.info(f"Early deduplication: Skipping candidate link {href} (post_id: {pid})")
                        continue
                    if pid:
                        seen_post_ids.add(pid)
                    if norm:
                        seen_normalized_urls.add(norm)
                    candidates.append((href, topic))
                    found_count += 1
            logger.info(
                f"[DDGS Fallback] Topic '{topic}': {ddgs_returned} returned, {found_count} kept"
            )
            print(
                f"  📌 [DDGS Fallback] Topic '{topic}': {ddgs_returned} returned, {found_count} kept"
            )
        except Exception as exc:
            logger.warning(f"Public search query failed for topic '{topic}': {exc}")
            print(f"  ⚠️ Public search query failed for topic '{topic}': {exc}")

    print(
        f"✅ Discovery completed. Found {len(candidates)} unique candidate post URLs "
        f"({early_duplicates_skipped} duplicates filtered early without requests).\n"
    )
    return candidates



# ---------------------------------------------------------
# Transparent Scoring System
# ---------------------------------------------------------

def calculate_candidate_score(
    data: dict[str, Any], topic: str, config: dict[str, Any] | None = None
) -> tuple[float, dict[str, float]]:
    """
    Computes a weighted total score (0 to 100) and per-component breakdown for candidate posts:
    - recency (35%)
    - author_fit (25%)
    - topic_match (25%)
    - engagement (15%)
    """
    config = config or load_config()
    weights = config.get("scoring_weights", {})
    w_recency = float(weights.get("recency", 35))
    w_author = float(weights.get("author_fit", 25))
    w_topic = float(weights.get("topic_match", 25))
    w_eng = float(weights.get("engagement", 15))

    text = (data.get("post_text") or "").lower()
    author = data.get("author") or ""

    if not text:
        return 0.0, {
            "recency": 0.0,
            "author_fit": 0.0,
            "topic_match": 0.0,
            "engagement": 0.0,
            "total": 0.0,
        }

    # 1. Recency Component (Smooth decay up to max_age)
    hours = data.get("post_age_hours")
    max_age = float(config.get("max_age_hours", 72))

    if hours is not None:
        if hours <= 12.0:
            recency_pts = w_recency * 1.0
        elif hours <= 24.0:
            recency_pts = w_recency * 0.9
        elif hours <= 48.0:
            recency_pts = w_recency * 0.7
        elif hours <= max_age:
            recency_pts = w_recency * 0.5
        else:
            recency_pts = 0.0
    else:
        recency_pts = w_recency * 0.35  # Safe fallback score for unknown age

    # 2. Author Fit Component
    whitelist = [a.lower() for a in config.get("whitelist_authors", [])]
    if author.lower() in whitelist:
        author_pts = w_author * 1.0
    elif is_company_account(author):
        author_pts = w_author * 0.2  # Penalize company/brand accounts
    elif author:
        author_pts = w_author * 0.75  # Individual human author
    else:
        author_pts = w_author * 0.4

    # 3. Dynamic Topic Match / Relevance Component
    # Dynamically extract niche words from topic, niche_description, and target_topics
    niche_words: set[str] = set()
    t_lower = topic.lower()
    for word in re.findall(r"\b[a-zA-Z0-9+#\-]{3,}\b", t_lower):
        niche_words.add(word)

    niche_desc = str(config.get("niche_description", "")).lower()
    for word in re.findall(r"\b[a-zA-Z0-9+#\-]{3,}\b", niche_desc):
        if word not in {"and", "the", "for", "with", "software", "engineering", "tools", "developer", "system"}:
            niche_words.add(word)

    for target_t in config.get("target_topics", []):
        for word in re.findall(r"\b[a-zA-Z0-9+#\-]{3,}\b", str(target_t).lower()):
            niche_words.add(word)

    matches = sum(1 for kw in niche_words if kw in text)

    if t_lower in text:
        topic_pts = w_topic * 1.0
    elif any(word in text for word in t_lower.split() if len(word) > 3):
        topic_pts = w_topic * 0.85
    elif matches >= 3:
        topic_pts = w_topic * 0.7
    elif matches >= 1:
        topic_pts = w_topic * 0.45
    else:
        topic_pts = w_topic * 0.2

    # 4. Engagement Component
    eng = data.get("engagement") or {}
    reactions = float(eng.get("reactions") or 0)
    comments = float(eng.get("comments") or 0)
    reposts = float(eng.get("reposts") or 0)

    if reactions > 0 or comments > 0 or reposts > 0:
        eng_scale = min(1.0, (reactions * 1.0 + comments * 2.0 + reposts * 3.0) / 40.0)
        eng_pts = w_eng * eng_scale
    else:
        eng_pts = w_eng * 0.3  # Baseline for posts where search snippet didn't include engagement numbers

    total = round(recency_pts + author_pts + topic_pts + eng_pts, 1)

    breakdown = {
        "recency": round(recency_pts, 1),
        "author_fit": round(author_pts, 1),
        "topic_match": round(topic_pts, 1),
        "engagement": round(eng_pts, 1),
        "total": total,
    }

    return total, breakdown


def is_blocklisted(data: dict[str, Any], config: dict[str, Any]) -> bool:
    """Checks if a candidate post matches configured author/keyword blocklists or low-quality/promotional patterns."""
    author = (data.get("author") or "").lower()
    text = (data.get("post_text") or "").lower()
    raw_text = (data.get("post_text") or "").strip()

    block_authors = [a.lower() for a in config.get("blocklist_authors", [])]
    if author in block_authors:
        return True

    block_kw = [k.lower() for k in config.get("blocklist_keywords", [])]
    for kw in block_kw:
        if kw in text:
            return True

    # 1. Filter out empty or ultra-short low-quality text (< 25 chars or < 4 words)
    words = raw_text.split()
    if len(raw_text) < 25 or len(words) < 4:
        logger.info(f"Filtering low-quality post: text too short ({len(raw_text)} chars, {len(words)} words)")
        return True

    # 2. Filter out promotional engagement bait & spam patterns
    promo_patterns = [
        r"\bcomment\b.*\b(below|yes|link|info)\b",
        r"\bdm\b.*\b(me|us|for|link)\b",
        r"\blink\s+in\s+(comments?|bio)\b",
        r"\b(we\s+are\s+hiring|job\s+opening|hiring\s+for|apply\s+here)\b",
        r"\b(register\s+now|webinar\s+alert|join\s+our\s+whatsapp)\b",
        r"\b(discount\s+code|buy\s+now|limited\s+time\s+offer)\b",
        r"\b(follow\s+me\s+for\s+more|repost\s+this)\b",
    ]
    for pattern in promo_patterns:
        if re.search(pattern, text):
            logger.info(f"Filtering promotional/spam post matching pattern: '{pattern}'")
            return True

    return False


# ---------------------------------------------------------
# Candidate Database & Append-Safe Persistence
# ---------------------------------------------------------

def load_candidates_db() -> list[dict[str, Any]]:
    if not CANDIDATES_FILE.exists():
        return []
    try:
        content = CANDIDATES_FILE.read_text(encoding="utf-8")
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except Exception as exc:
        logger.warning(f"Error loading candidates from {CANDIDATES_FILE}: {exc}")
    return []


def save_candidates_db(candidates: list[dict[str, Any]]) -> None:
    CANDIDATES_FILE.write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def count_comments_generated_today(existing_candidates: list[dict[str, Any]]) -> int:
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    count = 0
    for c in existing_candidates:
        if c.get("generated_comment") and not str(c.get("generated_comment")).startswith("["):
            disc_at = c.get("discovered_at", "")
            if disc_at.startswith(today_str):
                count += 1
    return count


# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------

from comment_generator import generate_comment


def display_top_candidates(candidates: list[dict[str, Any]], top_n: int = 5) -> None:
    if not candidates:
        print("⚠️ No valid candidates to display.")
        return

    print("=" * 50)
    print(f"🔥 TOP {min(top_n, len(candidates))} DISCOVERED LINKEDIN POSTS")
    print("=" * 50)

    for i, c in enumerate(candidates[:top_n], 1):
        post_preview = c.get("post_text") or "No text available"
        if len(post_preview) > 300:
            post_preview = post_preview[:300] + "..."

        suggested_comment = c.get("generated_comment") or "No comment generated."
        bd = c.get("score_breakdown") or {}

        print(f"\n#{i}")
        print(f"Topic:  {c.get('topic')}")
        print(f"Author: {c.get('author') or 'Unknown'}")
        print(f"Age:    {c.get('post_age') or 'Unknown'}")
        print(
            f"Score:  {c.get('score')} "
            f"(Recency: {bd.get('recency')}, AuthorFit: {bd.get('author_fit')}, "
            f"TopicMatch: {bd.get('topic_match')}, Eng: {bd.get('engagement')})"
        )
        print("\nPost:")
        print(post_preview)
        print("\nSuggested comment:")
        print(suggested_comment)
        print("\nLinkedIn:")
        print(c.get("url"))
        print("-" * 50)

    print(
        f"\n💾 Saved candidate database to: "
        f"{CANDIDATES_FILE.absolute()}"
    )


# ---------------------------------------------------------
# Workflow Execution
# ---------------------------------------------------------

def run_discovery_workflow(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)

    target_topics = list(config.get("target_topics", []))
    trending_enabled = bool(config.get("trending_topics_enabled", False))

    dynamic_count = 0
    fallback_count = 0

    topics = list(target_topics)

    if trending_enabled:
        niche_desc = str(config.get("niche_description", "")).strip()
        trending_topics, d_count, f_count = discover_trending_topics(niche_description=niche_desc)
        dynamic_count = d_count
        fallback_count = f_count
        for tt in trending_topics:
            if tt.lower() not in {t.lower() for t in topics}:
                topics.append(tt)

    print("=" * 50)
    print("TARGET DISCOVERY TOPICS")
    print("=" * 50)

    for idx, t in enumerate(topics, 1):
        print(f"{idx}. {t}")

    print("=" * 50)
    print("TOPIC DISCOVERY SUMMARY")
    print("=" * 50)
    print(f"Niche target topics:    {len(target_topics)}")
    print(f"Trending topics added:  {dynamic_count}")
    print(f"Fallback topics added:  {fallback_count}")
    print(f"Total topics searched:  {len(topics)}")

    existing_db = load_candidates_db()
    url_topic_pairs = discover_linkedin_urls(topics, config=config, existing_candidates=existing_db)

    if not url_topic_pairs:
        print("❌ No LinkedIn post URLs could be discovered.")
        return

    seen_post_ids, seen_normalized_urls, _ = build_candidate_dedup_index(existing_db)

    max_age_h = float(config.get("max_age_hours", 72))
    top_n = int(config.get("top_n_candidates", 5))
    max_daily_comments = int(config.get("max_comments_per_day", 5))

    new_candidates: list[dict[str, Any]] = []
    removed_stale_age = 0
    removed_unknown_age = 0
    removed_blocklisted = 0
    removed_author_dup = 0
    removed_early_dup = 0

    seen_authors_in_run: set[str] = set()
    candidates_discovered = len(url_topic_pairs)

    print(f"📥 Processing {candidates_discovered} candidate URLs...")

    for idx, (url, topic) in enumerate(url_topic_pairs, 1):
        norm_url = normalize_url(url)
        post_id = extract_linkedin_post_id(url)

        if (post_id and post_id in seen_post_ids) or (norm_url and norm_url in seen_normalized_urls):
            removed_early_dup += 1
            logger.info(f"Skipping duplicate URL before fetch: {url}")
            print(f" [{idx}/{candidates_discovered}] ⏩ Skipping duplicate (caught before request): {url}")
            continue

        author_from_slug = extract_author_from_url(url)
        if author_from_slug and author_from_slug.strip().lower() in seen_authors_in_run:
            removed_author_dup += 1
            print(f" [{idx}/{candidates_discovered}] ⏩ Skipping duplicate author '{author_from_slug}' (caught before request): {url}")
            continue

        print(f" [{idx}/{candidates_discovered}] Fetching: {url}")

        try:
            status_code, html = fetch_url(url)
            if status_code >= 400:
                print(f"   ⚠️ HTTP {status_code}, skipping.")
                continue

            data = extract_post_data(url, html)
            if not data.get("post_text") and not data.get("author"):
                print("   ⚠️ Empty extraction, skipping.")
                continue

            # Hard-filter blocklisted authors or keywords
            if is_blocklisted(data, config):
                removed_blocklisted += 1
                print("   ⚠️ Blocklisted author or keyword, skipping.")
                continue

            # Deduplicate by author (max 1 per author per run)
            author_name = (data.get("author") or "").strip().lower()
            if author_name and author_name in seen_authors_in_run:
                removed_author_dup += 1
                print(f"   ⚠️ Duplicate author '{data.get('author')}' in this run, skipping.")
                continue
            elif author_name:
                seen_authors_in_run.add(author_name)

            # Check post freshness against MAX_AGE_HOURS (default 72h)
            hours = data.get("post_age_hours")
            if hours is not None and hours > max_age_h:
                removed_stale_age += 1
                print(f"   ⚠️ Post age {hours}h (>{max_age_h}h stale limit), skipping.")
                continue
            elif hours is None:
                removed_unknown_age += 1

            score, breakdown = calculate_candidate_score(data, topic, config)

            candidate_record = {
                "url": data.get("url"),
                "canonical_url": data.get("canonical_url"),
                "post_id": data.get("post_id"),
                "author": data.get("author"),
                "is_company_account": is_company_account(data.get("author")),
                "followers": data.get("followers"),
                "post_age": data.get("post_age"),
                "post_age_hours": data.get("post_age_hours"),
                "post_age_source": data.get("post_age_source"),
                "post_text": data.get("post_text"),
                "hashtags": data.get("hashtags", []),
                "engagement": data.get("engagement", {}),
                "image": data.get("image"),
                "topic": topic,
                "score": score,
                "score_breakdown": breakdown,
                "status": "new",
                "discovered_at": datetime.now(timezone.utc).isoformat(),
            }

            if post_id:
                seen_post_ids.add(post_id)
            if norm_url:
                seen_normalized_urls.add(norm_url)
            if data.get("canonical_url"):
                canon_norm = normalize_url(data["canonical_url"])
                if canon_norm:
                    seen_normalized_urls.add(canon_norm)
            if data.get("post_id"):
                try:
                    seen_post_ids.add(int(data["post_id"]))
                except (TypeError, ValueError):
                    pass

            new_candidates.append(candidate_record)

        except Exception as exc:
            print(f"   ⚠️ Error processing {url}: {exc}")
            continue

    new_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Check daily comment limit
    comments_today = count_comments_generated_today(existing_db)
    remaining_comment_quota = max(0, max_daily_comments - comments_today)

    comments_generated = 0
    comment_failures = 0
    failure_counts: dict[str, int] = defaultdict(int)

    print("\n🤖 Generating Groq-powered comments for top candidates...")
    if remaining_comment_quota <= 0:
        print(f"  ⚠️ Daily limit of {max_daily_comments} generated comments reached for today. Skipping comment drafting.")
    else:
        for idx, c in enumerate(new_candidates[:top_n], 1):
            if comments_generated >= remaining_comment_quota:
                print(f"  ⚠️ Daily limit of {max_daily_comments} comments reached. Skipping remaining.")
                break

            if idx > 1:
                time.sleep(2.0)

            print(
                f"   Generating comment for candidate #{idx} ({c.get('topic')}, score: {c.get('score')})..."
            )
            comm = generate_comment(c, config)

            if comm and not comm.startswith("["):
                comments_generated += 1
            else:
                comment_failures += 1
                fail_reason = c.get("failure_reason") or c.get("skip_reason") or "unknown_failure"
                print(f"      ❌ Candidate #{idx} failed: {fail_reason}")

                cat_prefix = fail_reason.split(":")[0].strip() if ":" in fail_reason else fail_reason.strip()
                failure_counts[cat_prefix] += 1

            c["generated_comment"] = comm

    # Merge new candidates into existing database safely
    combined_db = existing_db + new_candidates
    save_candidates_db(combined_db)

    print("\n" + "=" * 50)
    print("CANDIDATE DISCOVERY SUMMARY")
    print("=" * 50)
    print(f"Candidates discovered:          {candidates_discovered}")
    print(f"Removed (>{max_age_h}h stale):         {removed_stale_age}")
    print(f"Unknown age (tracked):          {removed_unknown_age}")
    print(f"Removed (blocklisted):          {removed_blocklisted}")
    print(f"Removed (author dups):          {removed_author_dup}")
    print(f"Eligible for ranking:           {len(new_candidates)}")
    print(f"Top selected:                   {min(top_n, len(new_candidates))}")
    print(f"Comments generated successfully: {comments_generated}")
    print(f"Comment generation failures:     {comment_failures}")
    if failure_counts:
        print("\nFailure Reasons Breakdown:")
        for cat, count in sorted(failure_counts.items()):
            print(f"  - {cat}: {count}")
    print("=" * 50 + "\n")

    display_top_candidates(new_candidates, top_n=top_n)


def save_result(data: dict[str, Any]) -> None:
    OUTPUT_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_test_one(url: str, config_path: str = "config.yaml") -> None:
    """Runs full comment generation, check, and review flow for a single URL and prints intermediate outputs."""
    print("=" * 60)
    print(f"🧪 RUNNING TEST-ONE FLOW FOR: {url}")
    print("=" * 60)

    url_clean = url.strip()
    if not validate_linkedin_url(url_clean):
        print("❌ Invalid LinkedIn post URL format.")
        return

    config = load_config(config_path)
    print("📥 Fetching URL content...")
    try:
        status_code, html = fetch_url(url_clean)
        print(f"   HTTP status: {status_code}")
        if status_code >= 400:
            print("❌ HTTP error fetching post.")
            return
    except Exception as exc:
        print(f"❌ Failed to fetch URL: {exc}")
        return

    data = extract_post_data(url_clean, html)
    print("\n--- Extracted Candidate Post ---")
    print(f"Author:       {data.get('author')}")
    first_name = extract_author_first_name(data.get('author'))
    print(f"First Name:   {first_name}")
    print(f"Age:          {data.get('post_age')}")
    print(f"Text Snippet: {(data.get('post_text') or '')[:200]!r}")

    print("\n--- Running Comment Generator Flow ---")
    comment = generate_comment(data, config=config, verbose=True)

    print("\n" + "=" * 60)
    print("TEST-ONE FINAL RESULT")
    print("=" * 60)
    print(f"Status:         {data.get('status')}")
    print(f"Skip Reason:    {data.get('skip_reason')}")
    print(f"Failure Reason: {data.get('failure_reason')}")
    print(f"Final Comment:  {comment!r}")
    print("=" * 60)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="LinkedIn engagement assistant & public post extractor."
    )
    parser.add_argument(
        "url",
        nargs="?",
        default=None,
        help="Optional public LinkedIn post URL to scrape directly.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--test-one",
        default=None,
        metavar="POST_URL",
        help="Run full comment generation, check, and review flow for a single LinkedIn post URL.",
    )

    args = parser.parse_args()

    if args.test_one:
        run_test_one(args.test_one, config_path=args.config)
        return

    if not args.url:
        run_discovery_workflow(config_path=args.config)
        return

    url = args.url.strip()
    if not validate_linkedin_url(url):
        print("❌ Please provide a valid LinkedIn URL.")
        return

    print("🔎 Fetching LinkedIn URL...")
    print(f"   {url}\n")

    try:
        status_code, html = fetch_url(url)
    except httpx.RequestError as exc:
        print(f"❌ Request failed: {exc}")
        return

    print(f"HTTP status: {status_code}")
    if status_code >= 400:
        print("⚠️ Server returned an HTTP error.")

    data = extract_post_data(url, html)
    save_result(data)

    print("\n--- Extracted post ---")
    print(f"Author:     {data['author']}")
    print(f"Followers:  {data['followers']}")
    print(f"Age:        {data['post_age']}")
    print(f"Reactions:  {data['engagement']['reactions']}")
    print(f"Comments:   {data['engagement']['comments']}")

    print("\nHashtags:")
    for hashtag in data["hashtags"]:
        print(f"  {hashtag}")

    print("\nPost:")
    print(data["post_text"] or "Not available")
    print(f"\n💾 Saved to: {OUTPUT_FILE.absolute()}")


if __name__ == "__main__":
    main()