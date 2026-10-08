import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from notify import (
    format_candidate_message,
    load_top_n_setting,
    notify_new_candidates,
    send_telegram_message,
)


class TestNotify(unittest.TestCase):

    def test_format_candidate_message_standard(self):
        candidate = {
            "url": "https://www.linkedin.com/posts/test-author-12345",
            "author": "Alice Engineer",
            "score": 95,
            "topic": "FastAPI observability",
            "generated_comment": "This is a thoughtful technical comment on FastAPI tracing.",
        }
        msg = format_candidate_message(candidate)
        self.assertIn("Author: Alice Engineer", msg)
        self.assertIn("Topic: FastAPI observability", msg)
        self.assertIn("Score: 95", msg)
        self.assertIn("URL: https://www.linkedin.com/posts/test-author-12345", msg)
        self.assertIn("Drafted Comment:\nThis is a thoughtful technical comment on FastAPI tracing.", msg)
        self.assertLessEqual(len(msg), 4096)

    def test_format_candidate_message_truncation(self):
        long_comment = "A" * 5000
        candidate = {
            "url": "https://www.linkedin.com/posts/long-post",
            "author": "Bob Developer",
            "score": 88,
            "topic": "Python performance",
            "generated_comment": long_comment,
        }
        msg = format_candidate_message(candidate)
        self.assertEqual(len(msg), 4096)
        self.assertTrue(msg.endswith("..."))

    def test_format_candidate_message_missing_fields(self):
        candidate = {}
        msg = format_candidate_message(candidate)
        self.assertIn("Author: Unknown", msg)
        self.assertIn("Topic: N/A", msg)
        self.assertIn("Score: N/A", msg)
        self.assertIn("URL: N/A", msg)
        self.assertIn("Drafted Comment:\nNo comment generated", msg)
        self.assertLessEqual(len(msg), 4096)

    def test_load_top_n_setting(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tmp:
            tmp.write("top_n_candidates: 3\n")
            tmp_path = tmp.name

        try:
            self.assertEqual(load_top_n_setting(tmp_path), 3)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        self.assertEqual(load_top_n_setting("non_existent_config.yaml"), 5)

    @patch("notify.send_telegram_message")
    def test_notify_no_new_candidates(self, mock_send):
        mock_send.return_value = True

        with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tmp_cand, \
             tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tmp_cfg:
            
            json.dump([{"url": "old", "status": "reviewed"}], tmp_cand)
            tmp_cand.flush()
            tmp_cfg.write("top_n_candidates: 5\n")
            tmp_cfg.flush()

            tmp_cand_path = tmp_cand.name
            tmp_cfg_path = tmp_cfg.name

        try:
            result = notify_new_candidates(
                candidates_path=tmp_cand_path,
                config_path=tmp_cfg_path,
                token="dummy_token",
                chat_id="123456",
            )
            self.assertTrue(result)
            mock_send.assert_called_once_with("dummy_token", "123456", "No new posts today.")
        finally:
            for p in (tmp_cand_path, tmp_cfg_path):
                if os.path.exists(p):
                    os.remove(p)

    @patch("notify.send_telegram_message")
    def test_notify_new_candidates_sorting_and_top_n(self, mock_send):
        mock_send.return_value = True

        candidates = [
            {"url": "http://link1", "author": "User 1", "score": 70, "topic": "Topic A", "status": "new", "generated_comment": "Comm 1"},
            {"url": "http://link2", "author": "User 2", "score": 95, "topic": "Topic B", "status": "new", "generated_comment": "Comm 2"},
            {"url": "http://link3", "author": "User 3", "score": 85, "topic": "Topic C", "status": "new", "generated_comment": "Comm 3"},
        ]

        with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tmp_cand, \
             tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tmp_cfg:
            
            json.dump(candidates, tmp_cand)
            tmp_cand.flush()
            tmp_cfg.write("top_n_candidates: 2\n")
            tmp_cfg.flush()

            tmp_cand_path = tmp_cand.name
            tmp_cfg_path = tmp_cfg.name

        try:
            result = notify_new_candidates(
                candidates_path=tmp_cand_path,
                config_path=tmp_cfg_path,
                token="dummy_token",
                chat_id="123456",
            )
            self.assertTrue(result)
            self.assertEqual(mock_send.call_count, 2)
            # Check first sent message is highest score (95 -> User 2)
            first_msg = mock_send.call_args_list[0][0][2]
            self.assertIn("Author: User 2", first_msg)
            self.assertIn("Score: 95", first_msg)

            # Check second sent message is second highest score (85 -> User 3)
            second_msg = mock_send.call_args_list[1][0][2]
            self.assertIn("Author: User 3", second_msg)
            self.assertIn("Score: 85", second_msg)

            # Check that reply_markup was passed with 1-click copy button
            markup1 = mock_send.call_args_list[0][1].get("reply_markup")
            self.assertIsNotNone(markup1)
            buttons1 = markup1["inline_keyboard"][0]
            self.assertEqual(buttons1[0]["text"], "📋 Copy Comment")
            self.assertEqual(buttons1[0]["copy_text"]["text"], "Comm 2")
            self.assertEqual(buttons1[1]["text"], "🔗 Open Post")
            self.assertEqual(buttons1[1]["url"], "http://link2")
        finally:
            for p in (tmp_cand_path, tmp_cfg_path):
                if os.path.exists(p):
                    os.remove(p)

    def test_build_candidate_reply_markup(self):
        from notify import build_candidate_reply_markup

        cand_valid = {
            "url": "https://www.linkedin.com/posts/test-post",
            "generated_comment": "Great insight on system design and scaling!",
        }
        markup = build_candidate_reply_markup(cand_valid)
        self.assertIsNotNone(markup)
        buttons = markup["inline_keyboard"][0]
        self.assertEqual(len(buttons), 2)
        self.assertEqual(buttons[0]["text"], "📋 Copy Comment")
        self.assertEqual(buttons[0]["copy_text"]["text"], "Great insight on system design and scaling!")
        self.assertEqual(buttons[1]["text"], "🔗 Open Post")
        self.assertEqual(buttons[1]["url"], "https://www.linkedin.com/posts/test-post")

        # Candidate with no comment and no url
        cand_empty = {}
        self.assertIsNone(build_candidate_reply_markup(cand_empty))

        # Candidate with placeholder comment
        cand_placeholder = {"generated_comment": "No comment generated", "url": "N/A"}
        self.assertIsNone(build_candidate_reply_markup(cand_placeholder))


if __name__ == "__main__":
    unittest.main()
