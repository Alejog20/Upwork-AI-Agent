# Ulysses — Multi-Agent Ecosystem Architecture
*Stack: Python 3.14 · LangGraph · Telegram Bot API · Step 3.7 (LLM) · IMAP/RSS*

---

## System Overview

Ulysses is an orchestrated pipeline of 5 specialized agents that work together: one watches for gigs, one scores them, one notifies you, one writes proposals, and one builds demo prototypes. You interact entirely through Telegram inline buttons or the Python CLI.

```
┌─────────────────────────────────────────────────────────────────┐
│                        ULYSSES ECOSYSTEM                         │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────────┐  │
│  │  SCOUT   │───▶│  SCORER  │───▶│       NOTIFIER           │  │
│  │  Agent   │    │  Agent   │    │   (Telegram Bot)         │  │
│  └──────────┘    └──────────┘    └────────────┬─────────────┘  │
│       │                                        │                │
│  Watches email/                         Inline buttons:         │
│  RSS for new gigs                       [📝 Draft] [🛠 Build]   │
│                                         [⏭ Skip] [📁 Archive]  │
│                                                │                │
│                               ┌────────────────┴──────────┐    │
│                               │                           │    │
│                          ┌────▼─────┐            ┌────────▼──┐ │
│                          │ PROPOSAL │            │ PROTOTYPE │ │
│                          │  Agent   │            │   Agent   │ │
│                          └────┬─────┘            └────┬──────┘ │
│                               │                       │        │
│                          Draft sent              Demo script +  │
│                          to Telegram             README sent    │
│                          for edit/copy           as Telegram    │
│                                                  document       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agent Definitions

### 1. Scout Agent
**Responsibility:** Monitor and ingest new Upwork job postings.

**How it works:**
- Connects to your email via IMAP (Gmail/iCloud) and watches for Upwork notification emails
- Parses job title, description, budget, skills, client metadata, and post timestamp
- Alternatively polls the Upwork RSS feed if you have access
- Deduplicates jobs (stores seen job IDs in a local SQLite DB)
- Pushes new jobs into the LangGraph state for Scorer Agent processing

**Key data extracted:**
```python
JobPost(
    id: str,
    title: str,
    description: str,
    budget: BudgetRange,
    skills_required: list[str],
    client_hires: int,
    client_rating: float,
    payment_verified: bool,
    proposals_count: int,    # if visible
    posted_at: datetime,
    url: str,
)
```

---

### 2. Scorer Agent
**Responsibility:** Score each job on a 0–100 scale using your profile and the research-backed weighting model.

**Scoring formula:**
```python
score = (
    freshness(posted_at)         * 0.30  # <15 min = 30, <1h = 20, >4h = 5
  + low_proposal_count(count)    * 0.25  # <5 = 25, 5-15 = 15, >15 = 5
  + new_client(client_hires)     * 0.20  # 0 hires = 20, 1-3 = 12, >3 = 5
  + skill_match(skills, resume)  * 0.15  # % overlap with your tag list
  + budget_match(budget)         * 0.10  # $50-$500 range = 10, else scaled
)
```

**Also outputs:**
- `matched_repos`: which of your GitHub projects best match this job
- `gig_category`: Tier 1 / Tier 2 / Tier 3 (from the strategy doc)
- `red_flags`: list of detected warning patterns (see below)
- `recommendation`: APPLY_NOW / REVIEW / SKIP

**Red flag detector (NLP keyword patterns):**
```python
RED_FLAGS = [
    "will pay after results",
    "prove yourself",
    "simple task",
    "previous freelancer disappeared",
    "long-term if this works out",  # vague scope
    "per hour once hired",           # bait-and-switch budget
]
```

---

### 3. Notifier Agent (Telegram Bot)
**Responsibility:** Format and send scored jobs to you via Telegram with inline action buttons.

**Message format:**
```
🎯 Score: 84/100 | Tier 1 | APPLY NOW

📌 Python scraper for real estate listings
💰 $150 fixed | ⏱ Posted 8 min ago
👤 Client: 0 hires | ✅ Payment verified
📊 Proposals: ~3

🔗 Skills matched: web scraping, BeautifulSoup, CSV output
📁 Best repo match: Multiple_source_scraper

⚠️ Red flags: none

[📝 Draft Proposal] [🛠 Build Demo] [⏭ Skip] [📁 Archive]
```

**Inline button actions:**
| Button | Triggers |
|--------|----------|
| 📝 Draft Proposal | → Proposal Agent |
| 🛠 Build Demo | → Prototype Agent |
| ⏭ Skip | → Marks job as skipped, no alert again |
| 📁 Archive | → Saves to local DB for reference |
| Both (Draft + Demo) | → Both agents run in parallel |

**Threshold-based alerting:**
- Score ≥ 75: immediate push notification
- Score 50–74: batched every 30 min
- Score < 50: silent archive only

---

### 4. Proposal Agent
**Responsibility:** Draft a complete, Spartan-style Upwork cover letter tailored to the specific job.

**Input:** Job description + Scorer output (matched repos, category, red flags)

**Output (sent to Telegram as copyable text):**
```
--- PROPOSAL DRAFT ---

[HOOK — 2 sentences, job-specific, pattern interrupt]

[PROOF — links to matched GitHub repo + 1 sentence on what it does]

[PLAN — 3-bullet solution outline specific to their problem]

[CLOSE — 1 confident sentence + CTA]

Est. timeline: X days | Suggested bid: $XXX
--- END DRAFT ---
```

**Proposal generation rules (baked into the system prompt):**
- No greeting ("Hello", "Dear hiring manager")
- No "I am interested in your project"
- Tone: polite, warm, professional, and human — confident without being arrogant, brief without
  being curt. Must not read as AI-generated (no "In today's...", "Leveraging my expertise...",
  "Furthermore/Moreover" transitions, or similar tells)
- Always reference the specific pain point from the job description
- Always include at least one GitHub link
- Address the "new to Upwork, not new to the field" objection naturally
- Hard cap: 800 characters (enforced by truncating the LLM-generated hook/bullets before the
  template is filled, so the timeline/bid line is never what gets cut)
- At most 1-2 emoji, only if genuinely fitting and professional
- LLM calls are token-aware: the job description is truncated before it's sent, and the
  completion itself is capped via `max_tokens`, to keep quality high without wasting tokens

**Example output for a scraping job:**
```
Your listings are being updated manually — that's hours of work that should take seconds. 🔍

Proof: Multiple_source_scraper (github.com/Alejog20/Multiple_source_scraper) — validation,
dedup, and clean output already built in.

Plan:
• Targeted scraper for the listing site with Playwright/BeautifulSoup
• Validation and dedup with Pandas, saved to CSV/Sheets
• Scheduled via cron or a lightweight server

Timeline: 3 days | Bid: $120
```
(Under the 800-character cap.)

---

### 5. Prototype Agent
**Responsibility:** Generate a demo Python script + README that proves you can do the job, before you're hired.

**Input:** Job description + Scorer output

**Output (sent as Telegram document zip):**
```
ulysses_demo_[job_id]/
├── demo.py          # Working script demonstrating the core concept
├── requirements.txt # All dependencies pinned
├── README.md        # What it does, how to run it, how to extend it
└── config.example.env  # Any env vars needed (no real keys)
```

**What the demo script does:**
- Solves the CORE ask (not everything, but the hardest/most impressive part)
- Is fully runnable with `pip install -r requirements.txt && python demo.py`
- Has inline comments explaining choices
- Takes 0–1 arguments (keep it dead simple to run)

**README structure:**
```markdown
# Demo: [Job Title] — Built by Alejandro García

## What this demonstrates
[1 sentence on what the script does]

## How to run
pip install -r requirements.txt
python demo.py

## What a full implementation would add
- [Feature 1]
- [Feature 2]
- [Production hardening]

## About me
[2 sentences + GitHub link]
```

**Agent behavior:** It uses the LLM to generate the script based on job type, selects a template from a library (see below), then injects job-specific logic.

---

## LangGraph State Machine

```python
class UlyssesState(TypedDict):
    job: JobPost
    score: JobScore
    user_action: Literal["draft", "build", "both", "skip", "archive"] | None
    proposal_draft: str | None
    prototype_files: dict[str, str] | None  # filename → content
    notification_sent: bool
    completed: bool

# Graph nodes
graph = StateGraph(UlyssesState)
graph.add_node("scout", scout_agent)
graph.add_node("scorer", scorer_agent)
graph.add_node("notifier", notifier_agent)
graph.add_node("proposal", proposal_agent)
graph.add_node("prototype", prototype_agent)
graph.add_node("done", done_node)

# Edges
graph.add_edge("scout", "scorer")
graph.add_edge("scorer", "notifier")
graph.add_conditional_edges(
    "notifier",
    route_user_action,
    {
        "draft": "proposal",
        "build": "prototype",
        "both": ["proposal", "prototype"],  # parallel
        "skip": "done",
        "archive": "done",
    }
)
graph.add_edge("proposal", "done")
graph.add_edge("prototype", "done")

# Human-in-the-loop: notifier pauses graph until Telegram button is pressed
graph.add_interrupt("notifier")  # LangGraph interrupt point
```

---

## Python CLI Interface

```bash
# Start the full monitoring loop (scout -> scorer -> Telegram notifier)
ulysses start

# Show job counts and pipeline status
ulysses status

# Draft a proposal for a job already seen via email (looked up by its Upwork URL)
ulysses draft https://www.upwork.com/jobs/~01234567890

# Build a demo prototype for a job already seen, saved to ./output/<job_id>/
ulysses build https://www.upwork.com/jobs/~01234567890

# Do both draft + build in one call
ulysses go https://www.upwork.com/jobs/~01234567890

# Interactive chat: paste a job listing straight from the Upwork website (no
# email needed) and run it through the whole pipeline, right in the terminal
ulysses chat

# List known jobs, optionally filtered by minimum score and/or category
ulysses queue --min-score 70 --category tier1

# Mark a job as archived
ulysses archive <job_id>

# View your current profile.yaml
ulysses config show

# Update a single profile.yaml field by its dotted key
ulysses config set freelancer.rate_usd_hr 25
ulysses config set skills.primary "python,fastapi,scraping,automation"

# Install/remove a macOS LaunchAgent so `ulysses start` runs automatically on login
ulysses install
ulysses uninstall
```

See `README.md` for the full setup walkthrough and usage examples for each command.

---

## Project File Structure

```
ulysses/
├── agents/
│   ├── __init__.py
│   ├── scout.py          # Email/RSS watcher + parser
│   ├── scorer.py         # Job scoring engine
│   ├── notifier.py       # Telegram bot sender
│   ├── proposal.py       # Proposal drafter (LLM-powered)
│   └── prototype.py      # Demo script generator (LLM-powered)
│
├── graph/
│   ├── __init__.py
│   ├── state.py          # UlyssesState TypedDict
│   ├── nodes.py          # Node wrappers for each agent
│   ├── edges.py          # Routing logic
│   └── graph.py          # LangGraph StateGraph assembly
│
├── tools/
│   ├── email_reader.py   # IMAP connection + email parsing
│   ├── job_parser.py     # Extracts structured JobPost from raw email
│   ├── db.py             # SQLite — seen jobs, archives, scores
│   ├── github_mapper.py  # Maps job skills → your repos
│   └── red_flag.py       # NLP pattern detector
│
├── templates/
│   ├── proposals/
│   │   ├── scraping.txt
│   │   ├── automation.txt
│   │   ├── api_dev.txt
│   │   ├── data_pipeline.txt
│   │   └── ai_integration.txt
│   └── prototypes/
│       ├── scraper_base.py
│       ├── file_automation_base.py
│       ├── api_base.py
│       ├── email_bot_base.py
│       └── ai_pipeline_base.py
│
├── cli/
│   ├── __init__.py
│   └── main.py           # Typer-based CLI entry point
│
├── config/
│   ├── settings.py       # Pydantic Settings (loads from .env)
│   └── profile.yaml      # Your skills, repos, rate, preferences
│
├── tests/
│   ├── test_scorer.py
│   ├── test_parser.py
│   └── test_proposal.py
│
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Configuration (profile.yaml)

```yaml
# Your Upwork profile configuration — Ulysses uses this for scoring and proposals
freelancer:
  name: "Alejandro García"
  title: "I automate the boring stuff — Python scripts, scrapers, and AI pipelines"
  github: "https://github.com/Alejog20"
  rate_usd_hr: 25

skills:
  primary:
    - python
    - fastapi
    - web scraping
    - automation
    - data pipelines
    - pandas
    - beautifulsoup
    - playwright
  secondary:
    - langchain
    - openai api
    - sqlite
    - postgresql
    - react
    - typescript

repos:
  - name: Multiple_source_scraper
    url: https://github.com/Alejog20/Multiple_source_scraper
    tags: [scraping, python, data extraction, multi-platform]
  - name: download-autoprocessor
    url: https://github.com/Alejog20/download-autoprocessor
    tags: [automation, file processing, watchdog, python]
  - name: Invoice_generator_from_xlsx
    url: https://github.com/Alejog20/Invoice_generator_from_xlsx
    tags: [automation, excel, pdf, pandas, reporting]
  - name: Extract_Reddit_comments_bot
    url: https://github.com/Alejog20/Extract_Reddit_comments_bot
    tags: [api, reddit, data extraction, python bot]

scoring:
  target_budget_min: 50
  target_budget_max: 800
  min_score_to_notify: 50
  instant_alert_threshold: 75
  skip_if_proposals_above: 25
  skip_if_posted_hours_ago: 6

alerts:
  telegram_chat_id: "YOUR_CHAT_ID"
  batch_interval_minutes: 30
```

---

## Key Dependencies

```
# requirements.txt
langchain>=0.3.0
langgraph>=0.2.0
langchain-openai>=0.2.0   # or langchain-anthropic / step integration
python-telegram-bot>=21.0
typer>=0.12.0
pydantic>=2.0
pydantic-settings>=2.0
imaplib2>=3.6             # email monitoring
beautifulsoup4>=4.12      # email HTML parsing
sqlmodel>=0.0.18          # SQLite ORM
httpx>=0.27.0             # async HTTP
watchdog>=4.0.0           # file system events (for file-based triggers)
python-dotenv>=1.0.0
pyyaml>=6.0
rich>=13.0                # CLI output formatting
```

---

## Privacy & Data Integrity Principles

- **No job data leaves your machine** unless you explicitly trigger the LLM call
- SQLite DB stores all jobs locally — no cloud sync
- `.env` holds all credentials — never committed to git
- LLM is called only when you press Draft or Build
- Proposal drafts are never auto-sent — always shown to you first in Telegram
- Email credentials use app-specific passwords, never your main password

---

## Implementation Phases

### Phase 1 — Core Pipeline (Week 1–2)
Scout → Scorer → Telegram notifier with inline buttons. Basic skip/archive working.

### Phase 2 — Proposal Agent (Week 2–3)
LLM-powered proposal drafts sent to Telegram. Template library per job category.

### Phase 3 — Prototype Agent (Week 3–4)
Demo script generator from job description. Zip sent as Telegram document.

### Phase 4 — CLI Polish (Week 4)
Full Typer CLI. `ulysses score`, `ulysses draft`, `ulysses build`, `ulysses status`.
Native macOS menu bar app with LaunchAgent auto-start.

### Phase 5 — Built-in Chat REPL
`ulysses chat` opens an interactive terminal session: paste a job listing
copied straight from the Upwork website (no email required), and it's
extracted into a structured job via LLM (since real notification emails have
a stable HTML structure to key off of, but a manual paste doesn't), scored,
persisted to the same database scout-ingested jobs use, then drafted and
prototyped exactly like `ulysses go`. Results print to the terminal and save
to `./output/<job_id>/`; the loop continues for the next paste until you quit.
This bypasses the LangGraph pipeline's Telegram-oriented interrupt/resume
step entirely — the same way the real production Telegram flow already does
after a button press — rather than building on that mechanism's unused,
untested resume path.

### Phase 6 — Intelligence Upgrades (Ongoing)
- Fine-tune scoring weights based on your actual win rate
- Add "what worked" feedback loop: when you get a contract, log which job score/category it was
- A/B test proposal hooks over time
