import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

from bs4 import BeautifulSoup


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from main import (
    validate_linkedin_url,
    clean_text,
    format_post_text,
    get_meta,
    extract_hashtags,
    clean_post_text,
    extract_author,
    extract_followers,
    extract_post_age,
    extract_engagement,
    extract_post_data,
    extract_linkedin_post_id,
    post_id_to_datetime,
    calculate_post_age_hours,
    is_company_account,
    calculate_candidate_score,
    is_blocklisted,
    load_config,
    normalize_topics,
    parse_age_in_hours,
    extract_author_from_url,
    discover_trending_topics,
    normalize_url,
    build_candidate_dedup_index,
)


class TestLinkedInAgent(unittest.TestCase):

    def test_validate_linkedin_url(self):
        valid_urls = [
            "https://www.linkedin.com/posts/example_post-12345",
            "http://linkedin.com/posts/abc",
            "https://in.linkedin.com/posts/xyz",
        ]
        for url in valid_urls:
            self.assertTrue(validate_linkedin_url(url), f"Should be valid: {url}")

        invalid_urls = [
            "https://example.com/post",
            "ftp://linkedin.com/post",
            "invalid_url",
            "https://notlinkedin.com",
        ]
        for url in invalid_urls:
            self.assertFalse(validate_linkedin_url(url), f"Should be invalid: {url}")

    def test_clean_text(self):
        self.assertIsNone(clean_text(None))
        self.assertIsNone(clean_text("   "))
        self.assertEqual(clean_text("  Hello   World  "), "Hello World")
        self.assertEqual(clean_text("Line1\nLine2"), "Line1 Line2")

    def test_format_post_text(self):
        self.assertIsNone(format_post_text(None))
        self.assertIsNone(format_post_text("   "))

        raw_multiline = "  Paragraph 1   \r\n\r\n  Paragraph 2 \n\n\n Paragraph 3  "
        expected = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        self.assertEqual(format_post_text(raw_multiline), expected)

    def test_get_meta(self):
        html = """
        <html>
            <head>
                <meta property="og:description" content="Open Graph Description" />
                <meta name="description" content="Meta Description" />
                <meta property="og:image" content="https://example.com/image.jpg" />
            </head>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        self.assertEqual(
            get_meta(soup, "og:description", "description"),
            "Open Graph Description",
        )
        self.assertEqual(
            get_meta(soup, "og:image"),
            "https://example.com/image.jpg",
        )
        self.assertIsNone(get_meta(soup, "nonexistent"))

    def test_extract_hashtags(self):
        self.assertEqual(extract_hashtags(None), [])
        self.assertEqual(extract_hashtags("No hashtags here"), [])

        text = "Check out #Python #Coding and #Python for more!"
        self.assertEqual(extract_hashtags(text), ["#Python", "#Coding"])

    def test_clean_post_text(self):
        self.assertIsNone(clean_post_text(None))
        text_with_ui = "Great post body content here. Report this post Like Comment Share"
        self.assertEqual(clean_post_text(text_with_ui), "Great post body content here.")

    def test_extract_author(self):
        article_text = "Scaler 314,817 followers 11h Report this post"
        self.assertEqual(extract_author(article_text), "Scaler")

        html_og = '<html><head><meta property="og:title" content="#tag1 #tag2 | Tech Corp" /></head></html>'
        soup_og = BeautifulSoup(html_og, "html.parser")
        self.assertEqual(extract_author(None, soup_og), "Tech Corp")

        html_title = "<html><head><title>Jane Doe on LinkedIn: Sample post</title></head></html>"
        soup_title = BeautifulSoup(html_title, "html.parser")
        self.assertEqual(extract_author(None, soup_title), "Jane Doe")

    def test_extract_followers(self):
        self.assertIsNone(extract_followers(None))
        self.assertEqual(extract_followers("100 followers"), 100)
        self.assertEqual(extract_followers("314,817 followers"), 314817)
        self.assertEqual(extract_followers("1.5K followers"), 1500)
        self.assertEqual(extract_followers("2.5M followers"), 2500000)

    def test_extract_post_age(self):
        self.assertIsNone(extract_post_age(None))
        self.assertEqual(extract_post_age("Scaler 314,817 followers 11h"), "11h")
        self.assertEqual(extract_post_age("Posted 45m ago"), "45m")
        self.assertEqual(extract_post_age("Published 3d ago"), "3d")

    def test_extract_engagement(self):
        self.assertEqual(
            extract_engagement("16 1 Comment 2 Reposts"),
            {"reactions": 16, "comments": 1, "reposts": 2},
        )

    # ---------------------------------------------------------
    # NEW TESTS: Snowflake URL Post ID & Timestamp Decoding
    # ---------------------------------------------------------

    def test_extract_linkedin_post_id(self):
        url1 = "https://www.linkedin.com/posts/share-7511003917338628096-xxxx"
        url2 = "https://www.linkedin.com/feed/update/urn:li:activity:7511003917338628096/"
        url3 = "https://www.linkedin.com/posts/user_topic-7511003917338628096"

        self.assertEqual(extract_linkedin_post_id(url1), 7511003917338628096)
        self.assertEqual(extract_linkedin_post_id(url2), 7511003917338628096)
        self.assertEqual(extract_linkedin_post_id(url3), 7511003917338628096)
        self.assertIsNone(extract_linkedin_post_id("https://linkedin.com/posts/no-id-here"))

    def test_post_id_to_datetime(self):
        post_id = 7511003917338628096
        dt = post_id_to_datetime(post_id)
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.month, 9)
        self.assertEqual(dt.day, 30)

    def test_calculate_post_age_hours(self):
        url = "https://www.linkedin.com/posts/share-7511003917338628096-xxxx"
        hours, display_str, source = calculate_post_age_hours(url, "5h")
        self.assertIsNotNone(hours)
        self.assertEqual(source, "url_id")
        self.assertIn("URL ID", display_str)

        # Fallback to page text when URL ID is missing
        hours_fb, display_fb, source_fb = calculate_post_age_hours("https://linkedin.com/posts/abc", "5h")
        self.assertEqual(hours_fb, 5.0)
        self.assertEqual(source_fb, "page_text")
        self.assertIn("page text", display_fb)

    # ---------------------------------------------------------
    # NEW TESTS: Company Account Classification & Scoring
    # ---------------------------------------------------------

    def test_is_company_account(self):
        self.assertTrue(is_company_account("Scaler"))
        self.assertTrue(is_company_account("Acme Technologies"))
        self.assertTrue(is_company_account("Python Software Foundation"))
        self.assertFalse(is_company_account("Aniketsingh"))
        self.assertFalse(is_company_account("Jane Doe"))

    def test_calculate_candidate_score_breakdown(self):
        config = {
            "max_age_hours": 72,
            "whitelist_authors": ["Aniketsingh"],
            "scoring_weights": {
                "recency": 35,
                "author_fit": 25,
                "topic_match": 25,
                "engagement": 15,
            },
        }

        sample_post = {
            "post_text": "Building AI Agents with Python and FastAPI for backend development.",
            "author": "Aniketsingh",
            "post_age_hours": 12.0,
            "engagement": {"reactions": 30, "comments": 10, "reposts": 5},
        }

        score, breakdown = calculate_candidate_score(sample_post, "AI Agents", config)
        self.assertGreaterEqual(score, 80.0)
        self.assertEqual(breakdown["recency"], 35.0)
        self.assertEqual(breakdown["author_fit"], 25.0)
        self.assertIn("total", breakdown)

    def test_is_blocklisted(self):
        config = {
            "blocklist_authors": ["Spam Bot"],
            "blocklist_keywords": ["hiring now", "job alert"],
        }
        valid_post = {"author": "Jane Doe", "post_text": "Great article on FastAPI performance."}
        blocked_author = {"author": "Spam Bot", "post_text": "Hello"}
        blocked_keyword = {"author": "Jane Doe", "post_text": "We are hiring now for backend."}

        self.assertFalse(is_blocklisted(valid_post, config))
        self.assertTrue(is_blocklisted(blocked_author, config))
        self.assertTrue(is_blocklisted(blocked_keyword, config))

    def test_load_config(self):
        cfg = load_config("config.yaml")
        self.assertIn("target_topics", cfg)
        self.assertIn("max_age_hours", cfg)
        self.assertIn("scoring_weights", cfg)

    def test_extract_author_url_fallback(self):
        url1 = "https://www.linkedin.com/posts/its-aniketsingh_system-design-ugcPost-7510529409955827713-JyCX"
        url2 = "https://pe.linkedin.com/posts/dibika-dahal-a720642b0_some-topic-1234"
        self.assertEqual(extract_author_from_url(url1), "Aniketsingh")
        self.assertEqual(extract_author_from_url(url2), "Dibika Dahal")

        author = extract_author("78 comments", None, url2)
        self.assertEqual(author, "Dibika Dahal")

    def test_parse_age_in_hours(self):
        self.assertEqual(parse_age_in_hours("45m"), 0.75)
        self.assertEqual(parse_age_in_hours("5h"), 5.0)
        self.assertEqual(parse_age_in_hours("1d"), 24.0)
        self.assertEqual(parse_age_in_hours("2w"), 336.0)
        self.assertIsNone(parse_age_in_hours(None))

    def test_normalize_topics(self):
        raw = ["AI Agents", "ai agents", "AI AGENTS", "technology", "1. Claude Code", "MCP"]
        norm = normalize_topics(raw)
        self.assertIn("AI Agents", norm)
        self.assertIn("Claude Code", norm)
        self.assertIn("MCP", norm)
        self.assertNotIn("technology", norm)
        self.assertEqual(len(norm), 3)


from unittest.mock import patch
from comment_generator import (
    extract_author_first_name,
    normalize_comment_text,
    passes_basic_checks,
    generate_comment,
)


class TestCommentGenerator(unittest.TestCase):

    def test_extract_author_first_name(self):
        self.assertEqual(extract_author_first_name("Ilker Akkaya"), "Ilker")
        self.assertEqual(extract_author_first_name("NATALIA VALENZUELA"), "Natalia")
        self.assertEqual(extract_author_first_name("Dr. Jane Doe"), "Jane")
        self.assertEqual(extract_author_first_name("Guido van Rossum"), "Guido")
        self.assertIsNone(extract_author_first_name("Scaler"))
        self.assertIsNone(extract_author_first_name("Acme Technologies Inc"))
        self.assertIsNone(extract_author_first_name(""))
        self.assertIsNone(extract_author_first_name(None))

    def test_normalize_comment_text(self):
        # Curly quotes and non-breaking hyphens
        input1 = "“Ilker, this is a test\u2011case description.”"
        self.assertEqual(normalize_comment_text(input1), "Ilker, this is a test-case description.")

        # Multiple em dashes (1st becomes -, 2nd becomes ,)
        input2 = "Ilker, first point \u2014 second point \u2014 third point."
        normalized2 = normalize_comment_text(input2)
        self.assertIn("first point - second point, third point.", normalized2)

        # Surrounding quotes
        input3 = '"Ilker, simple comment text."'
        self.assertEqual(normalize_comment_text(input3), "Ilker, simple comment text.")

    def test_passes_basic_checks(self):
        valid_comment = (
            "Ilker, referencing the OpenTelemetry middleware detail from your post, native tracing "
            "in FastAPI simplifies observability pipelines significantly. Trace propagation overhead "
            "can still impact edge latency under high load. Balancing span sampling rates is essential."
        )
        passed, failed = passes_basic_checks(valid_comment, "Ilker")
        self.assertTrue(passed, f"Should pass basic checks, but failed: {failed}")

        # 1. SKIP
        passed, failed = passes_basic_checks("SKIP", "Ilker")
        self.assertFalse(passed)
        self.assertIn("Comment is SKIP", failed)

        # 2. Too short (<200)
        passed, failed = passes_basic_checks("Ilker, too short comment.", "Ilker")
        self.assertFalse(passed)
        self.assertTrue(any("under 200" in f for f in failed))

        # 3. Too long (>350)
        long_comment = "Ilker, " + "a" * 350
        passed, failed = passes_basic_checks(long_comment, "Ilker")
        self.assertFalse(passed)
        self.assertTrue(any("over 350" in f for f in failed))

        # 4. Wrong start
        wrong_start = valid_comment.replace("Ilker,", "Alice,")
        passed, failed = passes_basic_checks(wrong_start, "Ilker")
        self.assertFalse(passed)
        self.assertTrue(any("Does not start with author's first name" in f for f in failed))

        # 5. Banned word
        banned_word_comment = valid_comment.replace("simplifies", "delves into robust")
        passed, failed = passes_basic_checks(banned_word_comment, "Ilker")
        self.assertFalse(passed)
        self.assertTrue(any("banned word" in f for f in failed))

        # 6. Sentence starting with "However"
        however_comment = valid_comment.replace("Trace propagation", "However, trace propagation")
        passed, failed = passes_basic_checks(however_comment, "Ilker")
        self.assertFalse(passed)
        self.assertTrue(any("sentence starts with 'however'" in f.lower() for f in failed))


        # 7. Stock closing question
        stock_q_comment = valid_comment[:230] + " What do you think?"
        passed, failed = passes_basic_checks(stock_q_comment, "Ilker")
        self.assertFalse(passed)
        self.assertTrue(any("stock closing question" in f.lower() for f in failed))

    @patch("comment_generator.os.getenv", return_value="dummy_key")
    @patch("comment_generator._call_groq_with_retry")
    def test_generate_comment_flow_branch1_passed(self, mock_groq, mock_env):
        # Branch 1: Draft passes basic checks + reviewer PASSED
        valid_comment = (
            "Ilker, referencing the OpenTelemetry middleware detail from your post, native tracing "
            "in FastAPI simplifies observability pipelines significantly. Trace propagation overhead "
            "can still impact edge latency under high load. Balancing span sampling rates is essential."
        )
        mock_groq.side_effect = [valid_comment, "PASSED"]

        candidate = {"author": "Ilker Akkaya", "post_text": "Sample FastAPI post text...", "topic": "FastAPI"}
        res = generate_comment(candidate, {})

        self.assertEqual(res, valid_comment)
        self.assertEqual(candidate["status"], "new")
        self.assertEqual(candidate["generated_comment"], valid_comment)

    @patch("comment_generator.os.getenv", return_value="dummy_key")
    @patch("comment_generator._call_groq_with_retry")
    def test_generate_comment_flow_branch2_skip_initial(self, mock_groq, mock_env):
        # Branch 2: Draft returns SKIP
        mock_groq.side_effect = ["SKIP"]

        candidate = {"author": "Ilker Akkaya", "post_text": "No details post", "topic": "General"}
        res = generate_comment(candidate, {})

        self.assertEqual(res, "")
        self.assertEqual(candidate["status"], "skipped")
        self.assertIn("SKIP", candidate["skip_reason"])

    @patch("comment_generator.os.getenv", return_value="dummy_key")
    @patch("comment_generator._call_groq_with_retry")
    def test_generate_comment_flow_branch3_retry_success(self, mock_groq, mock_env):
        # Branch 3: Initial draft fails basic checks (too short), retry draft passes, reviewer PASSED
        valid_comment = (
            "Ilker, referencing the OpenTelemetry middleware detail from your post, native tracing "
            "in FastAPI simplifies observability pipelines significantly. Trace propagation overhead "
            "can still impact edge latency under high load. Balancing span sampling rates is essential."
        )
        too_short = "Ilker, too short draft."
        mock_groq.side_effect = [too_short, valid_comment, "PASSED"]

        candidate = {"author": "Ilker Akkaya", "post_text": "Sample FastAPI post text...", "topic": "FastAPI"}
        res = generate_comment(candidate, {})

        self.assertEqual(res, valid_comment)
        self.assertEqual(candidate["status"], "new")

    @patch("comment_generator.os.getenv", return_value="dummy_key")
    @patch("comment_generator._call_groq_with_retry")
    def test_generate_comment_flow_branch4_reviewer_rewrite_success(self, mock_groq, mock_env):
        # Branch 5: Initial draft passes, reviewer returns REWRITE <valid comment>
        valid_comment1 = (
            "Ilker, referencing the OpenTelemetry middleware detail from your post, native tracing "
            "in FastAPI simplifies observability pipelines significantly. Trace propagation overhead "
            "can still impact edge latency under high load. Balancing span sampling rates is essential."
        )
        valid_rewrite = (
            "Ilker, referencing the OpenTelemetry middleware detail from your post, native tracing "
            "in FastAPI simplifies observability pipelines nicely. Span sampling rates are key to "
            "preventing overhead in production systems under high load."
        )
        mock_groq.side_effect = [valid_comment1, f"REWRITE: {valid_rewrite}"]

        candidate = {"author": "Ilker Akkaya", "post_text": "Sample FastAPI post text...", "topic": "FastAPI"}
        res = generate_comment(candidate, {})

        self.assertEqual(res, valid_rewrite)
        self.assertEqual(candidate["status"], "new")

    @patch("comment_generator.os.getenv", return_value="dummy_key")
    @patch("comment_generator._call_groq_with_retry")
    def test_generate_comment_flow_branch5_reviewer_skip(self, mock_groq, mock_env):
        # Branch 7: Initial draft passes, reviewer returns SKIP
        valid_comment = (
            "Ilker, referencing the OpenTelemetry middleware detail from your post, native tracing "
            "in FastAPI simplifies observability pipelines significantly. Trace propagation overhead "
            "can still impact edge latency under high load. Balancing span sampling rates is essential."
        )
        mock_groq.side_effect = [valid_comment, "SKIP"]

        candidate = {"author": "Ilker Akkaya", "post_text": "Sample FastAPI post text...", "topic": "FastAPI"}
        res = generate_comment(candidate, {})

        self.assertEqual(res, "")
        self.assertEqual(candidate["status"], "skipped")
        self.assertEqual(candidate["skip_reason"], "Reviewer output SKIP")


from main import (
    search_serper,
    filter_linkedin_post_urls,
    load_serper_cache,
    save_serper_cache,
    discover_linkedin_urls,
)


class TestSerperSearch(unittest.TestCase):

    @patch("httpx.post")
    def test_search_serper_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "organic": [
                {"link": "https://www.linkedin.com/posts/user1-topic-123"},
                {"link": "https://nl.linkedin.com/posts/user2-topic-456"},
                {"link": "https://github.com/some/repo"},
            ]
        }
        mock_post.return_value = mock_resp

        links = search_serper("site:linkedin.com/posts FastAPI", num=10, api_key="dummy_serper_key")
        self.assertEqual(len(links), 3)
        self.assertIn("https://www.linkedin.com/posts/user1-topic-123", links)

    def test_filter_linkedin_post_urls(self):
        urls = [
            "https://www.linkedin.com/posts/user1-topic-123",
            "https://nl.linkedin.com/posts/user2-topic-456",
            "https://www.linkedin.com/in/john-doe",
            "https://github.com/some/repo",
            "https://example.com/article",
        ]
        filtered = filter_linkedin_post_urls(urls)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0], "https://www.linkedin.com/posts/user1-topic-123")
        self.assertEqual(filtered[1], "https://nl.linkedin.com/posts/user2-topic-456")

    @patch("httpx.post")
    def test_search_serper_non_200_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_post.return_value = mock_resp

        with self.assertRaises(RuntimeError):
            search_serper("query", api_key="dummy_key")

    @patch("main.os.getenv", return_value=None)
    @patch("ddgs.DDGS")

    def test_missing_api_key_fallback(self, mock_ddgs, mock_env):
        mock_ddgs_inst = MagicMock()
        mock_ddgs_inst.text.return_value = [
            {"href": "https://www.linkedin.com/posts/fallback-post-123"}
        ]
        mock_ddgs.return_value = mock_ddgs_inst

        config = {"serper_enabled": True}
        candidates = discover_linkedin_urls(["FastAPI"], config=config)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0][0], "https://www.linkedin.com/posts/fallback-post-123")

    @patch("httpx.post")
    def test_cache_hit_prevents_api_call(self, mock_post):
        with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tmp_cache:
            tmp_cache_path = tmp_cache.name

        try:
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            query = "site:linkedin.com/posts FastAPI"
            cache_key = f"{query}::{today_str}"
            cached_data = {
                cache_key: [
                    "https://www.linkedin.com/posts/cached-post-123"
                ]
            }
            save_serper_cache(cached_data, cache_file=tmp_cache_path)

            with patch("main.SERPER_CACHE_FILE", Path(tmp_cache_path)), \
                 patch("main.os.getenv", return_value="dummy_key"):
                
                config = {"serper_enabled": True}
                candidates = discover_linkedin_urls(["FastAPI"], config=config)

                self.assertEqual(len(candidates), 1)
                self.assertEqual(candidates[0][0], "https://www.linkedin.com/posts/cached-post-123")
                mock_post.assert_not_called()
        finally:
            if os.path.exists(tmp_cache_path):
                os.remove(tmp_cache_path)

    @patch("httpx.post")
    def test_max_call_limit_reached(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "organic": [{"link": "https://www.linkedin.com/posts/serper-post-1"}]
        }
        mock_post.return_value = mock_resp

        with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tmp_cache:
            tmp_cache_path = tmp_cache.name

        try:
            with patch("main.SERPER_CACHE_FILE", Path(tmp_cache_path)), \
                 patch("main.os.getenv", return_value="dummy_key"), \
                 patch("ddgs.DDGS") as mock_ddgs:


                mock_ddgs_inst = MagicMock()
                mock_ddgs_inst.text.return_value = [
                    {"href": "https://www.linkedin.com/posts/ddgs-fallback-post-2"}
                ]
                mock_ddgs.return_value = mock_ddgs_inst

                # Limit max_serper_calls_per_run to 1, but provide 2 topics
                config = {"serper_enabled": True, "max_serper_calls_per_run": 1}
                candidates = discover_linkedin_urls(["Topic1", "Topic2"], config=config)

                # Should call Serper once for Topic1, then fall back for Topic2
                self.assertEqual(mock_post.call_count, 1)
                self.assertEqual(len(candidates), 2)
        finally:
            if os.path.exists(tmp_cache_path):
                os.remove(tmp_cache_path)

    @patch("httpx.post")
    @patch("groq.Groq")
    def test_discover_trending_topics_with_niche(self, mock_groq, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "organic": [
                {"title": "Latest Python Trends", "snippet": "FastAPI and PyDantic 2026 updates"}
            ]
        }
        mock_post.return_value = mock_resp

        mock_client = MagicMock()
        mock_comp = MagicMock()
        mock_comp.choices = [MagicMock(message=MagicMock(content='["FastAPI", "PyDantic"]'))]
        mock_client.chat.completions.create.return_value = mock_comp
        mock_groq.return_value = mock_client

        with patch("main.os.getenv") as mock_env:
            mock_env.side_effect = lambda k: "dummy_key" if k in ("SERPER_API_KEY", "GROQ_API_KEY") else None
            topics, d_count, f_count = discover_trending_topics(niche_description="Python AI dev")

            self.assertIn("FastAPI", topics)
            self.assertIn("PyDantic", topics)
            self.assertEqual(d_count, 2)

    def test_prompts_formatting(self):
        """Verify that both system and reviewer prompts format without KeyError/ValueError for all 4 placeholders."""
        from comment_generator import SYSTEM_PROMPT, SELF_CHECK_PROMPT

        sample_kwargs = {
            "niche_description": "Software Engineering and Python",
            "author_first_name": "Ilker",
            "post_text": "Evaluating LLM pipelines with Promptfoo and deterministic tests.",
            "draft_comment": "Ilker, deterministic evaluation is great for factuality, but stochastic runs reveal edge cases.",
        }

        formatted_sys = SYSTEM_PROMPT.format(**sample_kwargs)
        self.assertIn("Ilker", formatted_sys)
        self.assertIn("Software Engineering", formatted_sys)

        formatted_rev = SELF_CHECK_PROMPT.format(**sample_kwargs)
        self.assertIn("Ilker", formatted_rev)
        self.assertIn("Promptfoo", formatted_rev)

    @patch("comment_generator.Groq")
    def test_empty_response_handling(self, mock_groq):
        """Verify _call_groq_with_retry treats empty content as retryable error."""
        from comment_generator import _call_groq_with_retry

        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = ""
        mock_choice.finish_reason = "length"
        mock_comp = MagicMock()
        mock_comp.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_comp

        with self.assertRaises(ValueError) as ctx:
            _call_groq_with_retry(mock_client, "sys", "user", max_retries=1)
        self.assertIn("empty message content", str(ctx.exception))

    def test_candidate_failure_reason_recording(self):
        """Verify generate_comment records failure_reason in post dict when skipping."""
        from comment_generator import generate_comment

        # 1. Company account -> generator_skip
        company_post = {"author": "Tech Solutions Inc", "post_text": "Sample text"}
        generate_comment(company_post)
        self.assertEqual(company_post.get("status"), "skipped")
        self.assertTrue(company_post.get("failure_reason", "").startswith("generator_skip:"))


    def test_normalize_url(self):
        self.assertIsNone(normalize_url(None))
        self.assertIsNone(normalize_url("  "))
        
        url1 = "https://www.linkedin.com/posts/user_post-activity-7123456789012345678-abcd?utm_source=share"
        url2 = "http://es.linkedin.com/posts/user_post-activity-7123456789012345678-abcd/"
        expected = "https://linkedin.com/posts/user_post-activity-7123456789012345678-abcd"
        
        self.assertEqual(normalize_url(url1), expected)
        self.assertEqual(normalize_url(url2), expected)

    def test_build_candidate_dedup_index(self):
        db = [
            {
                "url": "https://es.linkedin.com/posts/user1-activity-7123456789012345678-abcd",
                "canonical_url": "https://www.linkedin.com/posts/user1-activity-7123456789012345678-abcd",
                "author": "User One",
                "post_id": 7123456789012345678,
            },
            {
                "url": "https://linkedin.com/posts/user2-slug",
                "author": "User Two",
            }
        ]
        seen_pids, seen_urls, seen_authors = build_candidate_dedup_index(db)
        self.assertIn(7123456789012345678, seen_pids)
        self.assertIn("https://linkedin.com/posts/user1-activity-7123456789012345678-abcd", seen_urls)
        self.assertIn("https://linkedin.com/posts/user2-slug", seen_urls)
        self.assertIn("user one", seen_authors)

    @patch("httpx.post")
    def test_early_url_deduplication_in_discovery(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "organic": [
                {"link": "https://www.linkedin.com/posts/user1-activity-7123456789012345678-abcd?utm=1"},
                {"link": "https://es.linkedin.com/posts/user1-activity-7123456789012345678-abcd/"},
                {"link": "https://www.linkedin.com/posts/user2-activity-8888888888888888888-xyz"},
            ]
        }
        mock_post.return_value = mock_resp

        existing_db = [
            {
                "url": "https://linkedin.com/posts/user1-activity-7123456789012345678-abcd",
                "post_id": 7123456789012345678,
            }
        ]

        with patch("main.os.getenv", return_value="dummy_key"):
            config = {"serper_enabled": True}
            candidates = discover_linkedin_urls(["FastAPI"], config=config, existing_candidates=existing_db)

            # user1 post is in existing_db (both link #1 and link #2 are filtered early)
            # Only user2 post should be returned!
            self.assertEqual(len(candidates), 1)
            self.assertIn("8888888888888888888", candidates[0][0])


if __name__ == "__main__":
    unittest.main()



