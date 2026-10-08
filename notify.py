import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, List, Optional


def load_top_n_setting(config_path: str = "config.yaml") -> int:
    """Extracts top_n_candidates from config.yaml using stdlib regex, defaulting to 5."""
    path = Path(config_path)
    if not path.exists():
        return 5
    try:
        content = path.read_text(encoding="utf-8")
        match = re.search(r"top_n_candidates:\s*(\d+)", content)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return 5


def format_candidate_message(candidate: dict[str, Any]) -> str:
    """Formats a candidate post into a plain text Telegram message capped at 4096 characters."""
    url = candidate.get("url") or candidate.get("canonical_url") or "N/A"
    author = candidate.get("author") or "Unknown"
    score = candidate.get("score", "N/A")
    topic = candidate.get("topic") or "N/A"
    comment = candidate.get("generated_comment") or "No comment generated"

    msg = (
        f"🎯 New LinkedIn Candidate\n\n"
        f"Author: {author}\n"
        f"Topic: {topic}\n"
        f"Score: {score}\n"
        f"URL: {url}\n\n"
        f"Drafted Comment:\n{comment}"
    )

    if len(msg) > 4096:
        msg = msg[:4093] + "..."

    return msg


def build_candidate_reply_markup(candidate: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Builds a Telegram InlineKeyboardMarkup with:
    1. '📋 Copy Comment' button (Telegram Bot API 7.3+ copy_text for 1-click clipboard copy).
    2. '🔗 Open Post' button to open the LinkedIn post directly.
    """
    comment = candidate.get("generated_comment")
    url = candidate.get("url") or candidate.get("canonical_url")

    buttons = []
    if comment and comment != "No comment generated" and not str(comment).startswith("["):
        # Telegram Bot API copy_text limit is 1024 characters
        copy_val = str(comment).strip()[:1024]
        buttons.append({
            "text": "📋 Copy Comment",
            "copy_text": {"text": copy_val},
        })

    if url and str(url).startswith("http"):
        buttons.append({
            "text": "🔗 Open Post",
            "url": str(url).strip(),
        })

    if not buttons:
        return None

    return {"inline_keyboard": [buttons]}


def send_telegram_message(
    token: str,
    chat_id: str,
    text: str,
    reply_markup: Optional[dict[str, Any]] = None,
) -> bool:
    """Sends a message to a Telegram chat via standard library urllib, with optional inline keyboard."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as exc:
        # If reply_markup caused an error (e.g. client/server copy_text incompatibility), retry once without reply_markup
        if reply_markup:
            print(f"⚠️ Telegram send with reply_markup failed ({exc.code}). Retrying without reply_markup...")
            return send_telegram_message(token, chat_id, text, reply_markup=None)
        print(f"⚠️ Telegram notification HTTP error: {exc.code} {exc.reason}")
        return False
    except Exception as exc:
        print(f"⚠️ Telegram notification warning: {exc}")
        return False


def notify_new_candidates(
    candidates_path: str = "candidates.json",
    config_path: str = "config.yaml",
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> bool:
    """Reads candidates.json, picks top new candidates, and sends Telegram notifications."""
    if token is None:
        token = os.getenv("TELEGRAM_TOKEN")
    if chat_id is None:
        chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("⚠️ TELEGRAM_TOKEN or TELEGRAM_CHAT_ID environment variable not set. Skipping notifications.")
        return False

    top_n = load_top_n_setting(config_path)

    candidates: List[dict[str, Any]] = []
    c_path = Path(candidates_path)
    if c_path.exists():
        try:
            candidates = json.loads(c_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"⚠️ Failed to parse {candidates_path}: {exc}")

    # Pick candidates with status 'new'
    new_candidates = [c for c in candidates if isinstance(c, dict) and c.get("status") == "new"]

    if not new_candidates:
        msg = "No new posts today."
        print(f"Sending Telegram notification: '{msg}'")
        return send_telegram_message(token, chat_id, msg)

    # Rank by score descending
    new_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    selected_candidates = new_candidates[:top_n]

    print(f"Sending Telegram notifications for {len(selected_candidates)} top candidate(s)...")
    success_all = True
    notified_urls: set[str] = set()
    for idx, cand in enumerate(selected_candidates, 1):
        msg = format_candidate_message(cand)
        reply_markup = build_candidate_reply_markup(cand)
        ok = send_telegram_message(token, chat_id, msg, reply_markup=reply_markup)
        if ok:
            print(f"   Sent notification {idx}/{len(selected_candidates)}.")
            notified_urls.add(cand.get("url") or cand.get("canonical_url") or "")
        else:
            print(f"   Failed sending notification {idx}/{len(selected_candidates)}.")
            success_all = False

    # Mark notified candidates so they aren't re-sent on the next run
    if notified_urls:
        for cand in candidates:
            if isinstance(cand, dict):
                cand_url = cand.get("url") or cand.get("canonical_url") or ""
                if cand_url in notified_urls:
                    cand["status"] = "notified"
        try:
            c_path.write_text(json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"   Marked {len(notified_urls)} candidate(s) as notified.")
        except Exception as exc:
            print(f"   Failed to save updated candidates: {exc}")

    return success_all


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    notify_new_candidates()
