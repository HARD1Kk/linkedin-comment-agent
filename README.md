# LinkedIn Engagement Assistant & Comment Suggestion Pipeline

A lightweight, automated Python tool that discovers public LinkedIn posts matching your target technical niche, scores candidate posts transparently, and drafts high-value, natural LinkedIn comments powered by Groq LLM (`openai/gpt-oss-120b`).

> **Note**: This tool does NOT auto-post, perform login automation, or scrape behind authentication. It strictly outputs post suggestions and drafted comments for manual review.

---

## 🌟 Key Features

1. **Niche-Anchored Discovery**:
   - Primary discovery queries specific target topics (e.g. `FastAPI observability`, `MCP protocol`, `RAG architecture`).
   - Dynamic trending topic discovery is optional and configurable via `trending_topics_enabled`.
   - Every candidate post tracks the exact search query topic that discovered it.

2. **LinkedIn URL Snowflake Timestamp Freshness**:
   - LinkedIn post URLs contain 19-digit IDs (e.g., `activity-7511003917338628096`).
   - Post timestamp is computed via `timestamp_ms = post_id >> 22` for accurate post age calculation.
   - Posts older than `max_age_hours` (default: 72h) are automatically filtered out.

3. **Transparent Weighted Scoring**:
   - Evaluates posts using a weighted scoring model:
     - **Recency**: Fresh posts (<24h, <48h) get top score allocation.
     - **Author Fit**: Prefers individual human creators; awards whitelist bonus (+pts) and penalizes brand/company accounts (-pts).
     - **Topic Match**: Evaluates overlap with niche keywords and target topics.
     - **Engagement**: Scales based on reactions, comments, and reposts.
   - Per-component score breakdowns are stored in `candidates.json` and printed in the terminal CLI.

4. **Self-Checking Groq Comment Generation**:
   - Drafts 2-4 sentence technical comments grounded in the post text.
   - Enforces strict rules: no fake anecdotes, no stock closing questions, no absolute claims ("guarantees", "eliminates"), no hashtags or emojis, and no generic praise.
   - Includes a self-check evaluation pass to automatically rewrite draft comments if any rule is violated.
   - Automatic exponential backoff retries (up to 3 attempts) for Groq API calls.

5. **Hygiene & Append-Safe Persistence**:
   - Deduplicates candidates by post ID and limits to max 1 post per author per run.
   - Merges candidate entries safely into `candidates.json` with a tracking `status` field (`new`, `reviewed`, `commented`, `skipped`).
   - Enforces a configurable daily limit (`max_comments_per_day`) to prevent API overuse.

---

## ⚙️ Configuration (`config.yaml`)

Customize the pipeline behavior using `config.yaml`:

```yaml
# Niche & Target Persona
niche_description: >
  Software engineering, backend architecture, Python, AI developer tools,
  system design, and LLM engineering.

# Target topics (Niche-anchored discovery)
target_topics:
  - "FastAPI observability"
  - "Python performance"
  - "MCP protocol"
  - "RAG architecture"
  - "LLM evaluation"
  - "Kubernetes operators"
  - "PostgreSQL optimization"
  - "AI Agents architecture"

# Author Whitelist (+bonus score) & Blocklist (filtered immediately)
whitelist_authors:
  - "Aniketsingh"
  - "Scaler"
blocklist_authors:
  - "Spam Bot"
blocklist_keywords:
  - "we are hiring"
  - "job alert"

# Dynamic Trending Topics Toggle
trending_topics_enabled: false

# Freshness & Output Limits
max_age_hours: 72
max_comments_per_day: 5
top_n_candidates: 5

# Scoring Weights (Target total: 100 points)
scoring_weights:
  recency: 35
  author_fit: 25
  topic_match: 25
  engagement: 15
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
uv sync
```

### 2. Configure API Key
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
# Optional: SERPER_API_KEY=your_serper_key_here
```

### 3. Run Candidate Discovery & Comment Generation
```bash
uv run main.py
```

### 4. Run Unit Tests
```bash
uv run python -m unittest test_main.py
```
