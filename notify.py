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


def send_telegram_message(token: str, chat_id: str, text: str) -> bool:
    """Sends a plain text message to a Telegram chat via standard library urllib."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
    }
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
    for idx, cand in enumerate(selected_candidates, 1):
        msg = format_candidate_message(cand)
        ok = send_telegram_message(token, chat_id, msg)
        if ok:
            print(f"   Sent notification {idx}/{len(selected_candidates)}.")
        else:
            print(f"   Failed sending notification {idx}/{len(selected_candidates)}.")
            success_all = False

    return success_all


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    notify_new_candidates()
