import logging
import os
import re
import time
from typing import Any
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file if present
load_dotenv()

logger = logging.getLogger("linkedin_agent.comment_generator")

MODEL_NAME = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are an experienced technology professional participating naturally in LinkedIn technical discussions.

Niche & Persona Context:
{niche_description}

STRICT MANDATORY RULES FOR THE COMMENT:

1. REFERENCE ONE SPECIFIC DETAIL:
   - Reference one specific technical detail, tool, architecture, or point mentioned in the post.

2. ADD ONE CONCRETE PERSPECTIVE / COUNTERPOINT:
   - Offer a thoughtful engineering perspective, trade-off, or practical counterpoint.

3. LENGTH:
   - Exactly 2 to 4 sentences.

4. NO SUMMARY:
   - Do NOT summarize or repeat what the post said.

5. NO STOCK CLOSING QUESTIONS:
   - Do NOT end with generic questions like "What do you think?", "Thoughts?", "How do you handle this?", or "Have you tried this?".
   - A natural technical observation without a question is preferred.

6. NO ABSOLUTE CLAIMS:
   - Do NOT use absolute words like "guarantees", "eliminates", "always", "never", "flawless", or "perfect".

7. NO UNGROUNDED FACTS OR FAKE PERSONAL EXPERIENCES:
   - Do NOT invent personal anecdotes, fake company projects, or ungrounded statistics (no "In my 10 years...", "We reduced latency by 90%...").
   - Do NOT state facts that are not supported by the post or standard engineering knowledge.

8. NO HASHTAGS OR EMOJIS:
   - Zero hashtags. Zero emojis.

9. NO GENERIC PRAISE:
   - Do NOT start or end with "Great post", "Nice read", "Thanks for sharing", "Insightful post", etc.

10. OUTPUT FORMAT:
    - Return ONLY the final comment text. No quotation marks, no markdown labels, no formatting wrappers.
"""

SELF_CHECK_PROMPT = """You are a strict code and text quality reviewer for LinkedIn technical comments.

Review the following drafted LinkedIn comment against the post text and rules:

Original Post:
{post_text}

Drafted Comment:
{draft_comment}

Rules to verify:
1. Does it reference a specific detail from the post?
2. Does it add a practical opinion, trade-off, or counterpoint?
3. Is it exactly 2 to 4 sentences long?
4. Does it avoid summarizing the post?
5. Does it avoid stock closing questions ("What do you think?", "Thoughts?", "How do you handle this?")?
6. Does it avoid absolute claims ("guarantees", "eliminates", "always", "never")?
7. Does it contain ZERO hashtags and ZERO emojis?
8. Does it avoid fake personal anecdotes, fake company claims, or invented facts?

If the draft breaks ANY rule, output ONLY:
REWRITE: <Write an improved 2-4 sentence comment that strictly fixes the violations>

If the draft complies with ALL rules, output ONLY:
PASSED
"""


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
    Generate a human-sounding, high-value LinkedIn comment for a given candidate post
    using Groq API with retries and a self-check evaluation pass.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        logger.warning("GROQ_API_KEY not found in environment variables.")
        return "[GROQ_API_KEY not found in environment. Please set GROQ_API_KEY in .env]"

    config = config or {}
    niche_desc = config.get(
        "niche_description",
        "Software engineering, backend architecture, Python, AI developer tools, and system design.",
    )

    post_text = post.get("post_text") or ""
    author = post.get("author") or "the author"
    topic = post.get("topic") or "Technology"

    logger.info(
        f"Generating comment using model '{MODEL_NAME}' for topic '{topic}' (Author: {author})..."
    )

    system_prompt = SYSTEM_PROMPT.format(niche_description=niche_desc)

    user_prompt = f"""Topic: {topic}
Author: {author}

Post Text:
{post_text}

Generate ONE high-value, natural LinkedIn comment following all instructions. Return ONLY the comment text."""

    try:
        client = Groq(api_key=api_key)

        # 1. Draft generation pass (with up to 3 retries)
        comment = _call_groq_with_retry(client, system_prompt, user_prompt, max_retries=3)

        # Clean potential surrounding quotes if returned by LLM
        if comment.startswith('"') and comment.endswith('"'):
            comment = comment[1:-1].strip()

        # 2. Self-check evaluation pass
        try:
            check_prompt = SELF_CHECK_PROMPT.format(
                post_text=post_text[:1200], draft_comment=comment
            )
            check_response = _call_groq_with_retry(
                client,
                system_prompt="You are a strict text quality reviewer.",
                user_prompt=check_prompt,
                max_retries=2,
            )

            if check_response.startswith("REWRITE:"):
                rewritten = check_response.replace("REWRITE:", "").strip()
                if rewritten.startswith('"') and rewritten.endswith('"'):
                    rewritten = rewritten[1:-1].strip()

                if len(rewritten) > 20:
                    logger.info("Self-check flagged issues in initial draft. Applied rewritten comment.")
                    comment = rewritten

        except Exception as check_exc:
            logger.warning(f"Self-check pass skipped/failed: {check_exc}")

        logger.info(f"Successfully generated comment ({len(comment)} characters).")
        return comment

    except Exception as exc:
        logger.error(
            f"Failed to generate comment via Groq API after retries: {exc}",
            exc_info=True,
        )
        return f"[Failed to generate comment: {exc}]"
