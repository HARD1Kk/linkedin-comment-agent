import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional


REPO_OWNER = "HARD1Kk"
REPO_NAME = "linkedin-comment-agent"


def send_telegram_message(token: str, chat_id: str, text: str) -> bool:
    """Sends a plain text message to Telegram via urllib."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:
        print(f"⚠️ Telegram bot error sending message: {exc}")
        return False


def trigger_github_workflow(
    pat: str, owner: str = REPO_OWNER, repo: str = REPO_NAME
) -> tuple[bool, str]:
    """Triggers GitHub Actions workflow via repository_dispatch API."""
    url = f"https://api.github.com/repos/{owner}/{repo}/dispatches"
    payload = {"event_type": "telegram_draft"}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {pat}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "LinkedInAgentBot",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status in (200, 204):
                return True, "Workflow dispatched successfully!"
            return False, f"GitHub API status code: {resp.status}"
    except urllib.error.HTTPError as exc:
        return False, f"GitHub API error: {exc.code} {exc.reason}"
    except Exception as exc:
        return False, f"Failed to reach GitHub API: {exc}"


def poll_telegram_bot(
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
    pat: Optional[str] = None,
) -> None:
    """Long-polling daemon for Telegram bot commands."""
    token = token or os.getenv("TELEGRAM_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    pat = pat or os.getenv("GH_PAT") or os.getenv("GITHUB_TOKEN")

    if not token or not chat_id:
        print("❌ TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not configured.")
        return

    print("🤖 Telegram Bot command listener started.")
    print("   Send '/draft' or '/run' in your Telegram chat to trigger the pipeline.")

    offset = 0
    while True:
        try:
            url = f"https://api.telegram.org/bot{token}/getUpdates?offset={offset}&timeout=30"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=35) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            if not data.get("ok"):
                time.sleep(5)
                continue

            for update in data.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message", {})
                text = (msg.get("text") or "").strip()
                sender_chat_id = str(msg.get("chat", {}).get("id"))

                # Restrict to configured chat_id for security
                if sender_chat_id != str(chat_id):
                    continue

                if text.startswith("/draft") or text.startswith("/run"):
                    if not pat:
                        send_telegram_message(
                            token,
                            chat_id,
                            "⚠️ GH_PAT (GitHub Personal Access Token) environment variable not set. Cannot dispatch GitHub Actions.",
                        )
                        continue

                    send_telegram_message(
                        token,
                        chat_id,
                        "⏳ Command received! Dispatching daily discovery workflow on GitHub Actions...",
                    )

                    ok, err_msg = trigger_github_workflow(pat)
                    if ok:
                        send_telegram_message(
                            token,
                            chat_id,
                            "🚀 GitHub Actions workflow started successfully! Candidate comments will be delivered here when ready.",
                        )
                    else:
                        send_telegram_message(
                            token,
                            chat_id,
                            f"❌ Failed to trigger GitHub Actions: {err_msg}",
                        )

                elif text.startswith("/start") or text.startswith("/help"):
                    send_telegram_message(
                        token,
                        chat_id,
                        "🤖 LinkedIn Engagement Assistant Bot\n\nAvailable commands:\n/draft - Trigger discovery & comment generation on GitHub Actions",
                    )

        except KeyboardInterrupt:
            print("\n👋 Bot listener stopped.")
            break
        except Exception as exc:
            print(f"⚠️ Listener error: {exc}. Retrying in 5s...")
            time.sleep(5)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    poll_telegram_bot()
