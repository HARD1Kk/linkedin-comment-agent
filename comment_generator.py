import logging
import os
import re
import time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file if present
load_dotenv()

logger = logging.getLogger("linkedin_agent.comment_generator")

MODEL_NAME = "openai/gpt-oss-120b"
PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(filename: str) -> str:
    """Helper to load prompt template from prompts/ directory."""
    path = PROMPTS_DIR / filename
    return path.read_text(encoding="utf-8")


SYSTEM_PROMPT = load_prompt("system_prompt.txt")
SELF_CHECK_PROMPT = load_prompt("self_check_prompt.txt")

BANNED_WORDS = {
    "guarantees",
    "eliminates",
    "always",
    "never",
    "flawless",
    "perfect",
    "delve",
    "leverage",
    "robust",
    "seamless",
    "pivotal",
    "tapestry",
}

STOCK_QUESTIONS = [
    "what do you think?",
    "thoughts?",
    "how do you handle this?",
    "have you tried this?",
]


def is_company_account(author_name: str | None) -> bool:
    """Classifies if an author appears to be a brand/company account."""
    if not author_name:
        return False
    clean = author_name.strip()
    lower = clean.lower()

    company_keywords = {
        "corp", "inc", "technologies", "solutions", "academy", "university",
        "official", "page", "company", "institute", "software", "labs",
        "hub", "scaler", "linkedin", "team", "media", "agency", "group",
        "global", "systems", "consulting", "services", "enterprise",
        "foundation", "llc", "ltd", "gmbh", "co.", "corporation", "network",
    }
    words = set(re.split(r"[\s,._\-]+", lower))
    if words.intersection(company_keywords):
        return True

    if (
        " & " in clean
        or clean.endswith(" Inc")
        or clean.endswith(" LLC")
        or clean.endswith(" Ltd")
    ):
        return True

    return False


def extract_author_first_name(author: str | None) -> str | None:
    """Extracts the first name of a candidate author. Returns None for company/brand accounts."""
    if not author or not isinstance(author, str):
        return None

    author_clean = author.strip()
    if is_company_account(author_clean):
        return None

    # Strip leading titles like Dr., Prof., Mr., Mrs., Ms.
    author_clean = re.sub(r"^(?:Dr|Prof|Mr|Mrs|Ms)\.?\s+", "", author_clean, flags=re.IGNORECASE).strip()

    # Remove non-alphanumeric / non-name characters (except spaces and hyphens)
    author_clean = re.sub(r"[^\w\s\-]", "", author_clean, flags=re.UNICODE)

    tokens = author_clean.split()
    if not tokens:
        return None

    first_name = tokens[0].strip(".,:-_")
    if not first_name or len(first_name) < 2 or is_company_account(first_name):
        return None

    if first_name.isupper() or first_name.islower():
        first_name = first_name.capitalize()

    return first_name


def normalize_comment_text(text: str) -> str:
    """Normalizes comment text: converts Unicode dashes/quotes, limits em-dashes, strips quotes."""
    if not text:
        return ""

    # 1. Replace curly quotes
    t = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # 2. Replace non-breaking hyphens, en-dashes, soft hyphens with '-'
    t = t.replace("\u2011", "-").replace("\u2013", "-").replace("\u00ad", "-")

    # 3. Handle em dashes (\u2014 or '—' or '--')
    em_dash_pattern = re.compile(r"\u2014|—|--")
    matches = list(em_dash_pattern.finditer(t))

    if len(matches) > 1:
        # Keep 1st em dash as '-', replace 2nd and subsequent with ','
        result_parts = []
        last_idx = 0
        for idx, match in enumerate(matches):
            result_parts.append(t[last_idx:match.start()])
            if idx == 0:
                result_parts.append("-")
            else:
                result_parts.append(",")
            last_idx = match.end()
        result_parts.append(t[last_idx:])
        t = "".join(result_parts)
    elif len(matches) == 1:
        t = em_dash_pattern.sub("-", t)

    # 4. Clean up spaces before punctuation (e.g. " ," -> ",")
    t = re.sub(r"\s+([,\.\?\!])", r"\1", t)

    # 5. Replace non-breaking spaces
    t = t.replace("\u00a0", " ")

    # 6. Strip surrounding quotation marks and whitespace
    t = t.strip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()

    return t


def contains_emoji(text: str) -> bool:
    """Checks if text contains emoji characters."""
    for char in text:
        cp = ord(char)
        if (
            0x1F600 <= cp <= 0x1F64F or
            0x1F300 <= cp <= 0x1F5FF or
            0x1F680 <= cp <= 0x1F6FF or
            0x1F1E0 <= cp <= 0x1F1FF or
            0x2600 <= cp <= 0x27BF or
            0x1F900 <= cp <= 0x1F9FF or
            0x1FA70 <= cp <= 0x1FAFF
        ):
            return True
    return False


def passes_basic_checks(comment: str, first_name: str) -> tuple[bool, list[str]]:
    """Runs deterministic basic checks on a comment string."""
    failed_checks: list[str] = []

    if not comment:
        return False, ["Comment is empty"]

    trimmed = comment.strip()

    # 1. Not "SKIP"
    if trimmed.upper() == "SKIP":
        return False, ["Comment is SKIP"]

    # 2. Length between 200 and 350 characters
    c_len = len(trimmed)
    if c_len < 200:
        failed_checks.append(f"Length {c_len} is under 200 character minimum")
    elif c_len > 350:
        failed_checks.append(f"Length {c_len} is over 350 character maximum")

    # 3. Starts with the author's first name
    if first_name:
        if not trimmed.lower().startswith(first_name.lower()):
            failed_checks.append(f"Does not start with author's first name '{first_name}'")

    # 4. Contains no "#" and no emoji
    if "#" in trimmed:
        failed_checks.append("Contains hashtag '#'")

    if contains_emoji(trimmed):
        failed_checks.append("Contains emoji")

    # 5. Does not contain "However" at the start of a sentence
    if re.search(r"(?:^|[.!?]\s+)However\b", trimmed):
        failed_checks.append("Sentence starts with 'However'")


    # 6. Does not contain any banned words
    for word in BANNED_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", trimmed, re.IGNORECASE):
            failed_checks.append(f"Contains banned word '{word}'")

    # 7. Does not end with a stock question
    trimmed_lower = trimmed.lower().rstrip()
    for sq in STOCK_QUESTIONS:
        if trimmed_lower.endswith(sq):
            failed_checks.append(f"Ends with stock closing question '{sq}'")
            break

    is_passing = len(failed_checks) == 0
    return is_passing, failed_checks


def _call_groq_with_retry(
    client: Groq,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
    initial_delay: float = 1.0,
) -> str:
    """Calls Groq completions API with exponential backoff retries."""
    delay = initial_delay
    last_exception: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_completion_tokens=300,
                top_p=1,
            )
            text = completion.choices[0].message.content.strip()
            return text
        except Exception as exc:
            last_exception = exc
            logger.warning(
                f"Groq API call attempt {attempt}/{max_retries} failed: {exc}"
            )
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2.0

    if last_exception:
        raise last_exception
    raise RuntimeError("Groq API call failed after retries.")


def generate_comment(post: dict[str, Any], config: dict[str, Any] | None = None) -> str:
    """
    Generate a human-sounding LinkedIn comment for a candidate post using Groq API.
    Enforces author first-name extraction, deterministic basic checks, retry with feedback,
    and a reviewer pass. Updates post dict with 'status' ('new' or 'skipped') and 'skip_reason'.
    Total LLM calls per candidate are capped at 4.
    """
    author_raw = post.get("author")
    first_name = extract_author_first_name(author_raw)
    logger.info(f"[{author_raw}] Extracting author first name: '{first_name}'")

    if not first_name:
        post["status"] = "skipped"
        post["skip_reason"] = "Author is a company/brand or first name could not be determined"
        logger.warning(f"[{author_raw}] Candidate skipped: {post['skip_reason']}")
        return ""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found in environment variables.")
        post["status"] = "skipped"
        post["skip_reason"] = "GROQ_API_KEY not found in environment"
        return ""

    config = config or {}
    niche_desc = config.get(
        "niche_description",
        "Software engineering, backend architecture, Python, AI developer tools, and system design.",
    )

    post_text = post.get("post_text") or ""
    author = author_raw or "the author"
    topic = post.get("topic") or "Technology"

    llm_calls = 0
    MAX_LLM_CALLS = 4

    client = Groq(api_key=api_key)

    # Prepare Prompts
    system_prompt = SYSTEM_PROMPT.format(
        niche_description=niche_desc,
        author_first_name=first_name,
        post_text=post_text,
        draft_comment="",
    )

    user_prompt = f"""Topic: {topic}
Author: {author}
Author First Name: {first_name}

Post Text:
{post_text}

Generate ONE high-value, natural LinkedIn comment following all instructions. Return ONLY the comment text."""

    try:
        # a. Initial Draft Generation
        logger.info(f"[{author}] Generating initial draft comment...")
        llm_calls += 1
        raw_draft = _call_groq_with_retry(client, system_prompt, user_prompt, max_retries=3)
        comment = normalize_comment_text(raw_draft)

        if comment.upper() == "SKIP":
            post["status"] = "skipped"
            post["skip_reason"] = "LLM generated SKIP on initial draft"
            logger.info(f"[{author}] Candidate skipped: {post['skip_reason']}")
            return ""

        # b. Basic Checks
        passed, failed_checks = passes_basic_checks(comment, first_name)
        if not passed:
            logger.warning(
                f"[{author}] Initial draft failed basic checks: {failed_checks}. Retrying generation once with feedback..."
            )
            if llm_calls >= MAX_LLM_CALLS:
                post["status"] = "skipped"
                post["skip_reason"] = f"Failed basic checks and reached LLM call limit: {', '.join(failed_checks)}"
                logger.info(f"[{author}] Candidate skipped: {post['skip_reason']}")
                return ""

            feedback = "\n".join([f"- {fc}" for fc in failed_checks])
            retry_user_prompt = f"{user_prompt}\n\nPrevious draft failed these checks:\n{feedback}\nPlease generate a new comment fixing all failed checks."
            llm_calls += 1
            raw_retry = _call_groq_with_retry(client, system_prompt, retry_user_prompt, max_retries=3)
            comment = normalize_comment_text(raw_retry)

            if comment.upper() == "SKIP":
                post["status"] = "skipped"
                post["skip_reason"] = "LLM generated SKIP on retry draft"
                logger.info(f"[{author}] Candidate skipped: {post['skip_reason']}")
                return ""

            passed, failed_checks = passes_basic_checks(comment, first_name)
            if not passed:
                post["status"] = "skipped"
                post["skip_reason"] = f"Failed basic checks after retry: {', '.join(failed_checks)}"
                logger.info(f"[{author}] Candidate skipped: {post['skip_reason']}")
                return ""
            else:
                logger.info(f"[{author}] Retry draft passed basic checks.")
        else:
            logger.info(f"[{author}] Initial draft passed basic checks.")

        # c. Reviewer Pass
        if llm_calls >= MAX_LLM_CALLS:
            post["status"] = "skipped"
            post["skip_reason"] = "Reached LLM call limit before reviewer pass"
            logger.info(f"[{author}] Candidate skipped: {post['skip_reason']}")
            return ""

        check_prompt = SELF_CHECK_PROMPT.format(
            post_text=post_text[:1200],
            draft_comment=comment,
            author_first_name=first_name,
            niche_description=niche_desc,
        )

        logger.info(f"[{author}] Running reviewer evaluation...")
        llm_calls += 1
        reviewer_resp = _call_groq_with_retry(
            client,
            system_prompt="You are a strict text quality reviewer.",
            user_prompt=check_prompt,
            max_retries=2,
        )
        reviewer_resp_clean = reviewer_resp.strip()

        if reviewer_resp_clean.upper() == "PASSED":
            post["status"] = "new"
            post["generated_comment"] = comment
            logger.info(f"[{author}] Reviewer returned PASSED. Final comment approved.")
            return comment

        if reviewer_resp_clean.upper() == "SKIP":
            post["status"] = "skipped"
            post["skip_reason"] = "Reviewer output SKIP"
            logger.info(f"[{author}] Candidate skipped: {post['skip_reason']}")
            return ""

        if reviewer_resp_clean.startswith("REWRITE:"):
            raw_rewrite = reviewer_resp_clean.replace("REWRITE:", "").strip()
            rewrite_text = normalize_comment_text(raw_rewrite)
            if rewrite_text.upper() == "SKIP":
                post["status"] = "skipped"
                post["skip_reason"] = "Reviewer rewrite returned SKIP"
                logger.info(f"[{author}] Candidate skipped: {post['skip_reason']}")
                return ""

            rw_passed, rw_failed = passes_basic_checks(rewrite_text, first_name)
            if rw_passed:
                post["status"] = "new"
                post["generated_comment"] = rewrite_text
                logger.info(f"[{author}] Reviewer returned REWRITE which passed basic checks. Final comment approved.")
                return rewrite_text
            else:
                post["status"] = "skipped"
                post["skip_reason"] = f"Reviewer rewrite failed basic checks: {', '.join(rw_failed)}"
                logger.info(f"[{author}] Candidate skipped: {post['skip_reason']}")
                return ""

        # Default fallback if reviewer response is generic text
        rw_passed, rw_failed = passes_basic_checks(reviewer_resp_clean, first_name)
        if rw_passed:
            post["status"] = "new"
            post["generated_comment"] = reviewer_resp_clean
            logger.info(f"[{author}] Reviewer output accepted. Final comment approved.")
            return reviewer_resp_clean
        else:
            post["status"] = "new"
            post["generated_comment"] = comment
            logger.info(f"[{author}] Reviewer evaluation complete. Final comment approved.")
            return comment

    except Exception as exc:
        logger.error(f"[{author}] Comment generation failed: {exc}", exc_info=True)
        post["status"] = "skipped"
        post["skip_reason"] = f"Error during comment generation: {exc}"
        return ""
