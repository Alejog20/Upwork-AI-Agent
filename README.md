# Ulysses

Ulysses is a multi-agent AI ecosystem that monitors your email for new Upwork
job postings, scores them against your profile, and pushes the good ones to
Telegram with one-tap actions to draft a proposal or build a demo prototype.

See `ULYSSES-ARQUITECHTURE.md` for the full system design and `CLAUDE.md` for
project development standards.

Current status: **Phase 3** — Scout, Scorer, Telegram notifier, the Proposal
Agent, and the Prototype Agent are all live (skip/archive/draft/regenerate/
copy/build all working). The macOS menu bar app and auto-start land in
Phase 4.

## Requirements

- Python 3.14
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- A Gmail or iCloud mailbox that receives Upwork job notification emails
- A Telegram bot (see below)

## Setup

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Create your `.env` file from the template and fill in your credentials:

   ```bash
   cp .env.example .env
   ```

3. Set up email access:

   - **Gmail**: enable 2-Step Verification, then create an
     [App Password](https://myaccount.google.com/apppasswords) and use it as
     `ULYSSES_IMAP_APP_PASSWORD`.
   - **iCloud**: enable two-factor authentication, then create an
     [app-specific password](https://support.apple.com/en-us/102654) at
     appleid.apple.com and use it as `ULYSSES_IMAP_APP_PASSWORD`. Set
     `ULYSSES_IMAP_PROVIDER=icloud`.

   Never use your real account password — Ulysses only ever needs an
   app-specific one.

4. Create a Telegram bot:

   - Message [@BotFather](https://t.me/BotFather) on Telegram, send `/newbot`,
     and follow the prompts. Copy the token it gives you into
     `ULYSSES_TELEGRAM_BOT_TOKEN`.
   - Message [@userinfobot](https://t.me/userinfobot) to get your numeric
     chat ID, and put it in `ULYSSES_TELEGRAM_CHAT_ID`. Ulysses only ever sends
     to and accepts button presses from this chat.

5. Review `ulysses/config/profile.yaml` and adjust your skills, GitHub repos,
   rate, and scoring thresholds to match your own profile.

6. Set `ULYSSES_LLM_API_KEY` (and `ULYSSES_LLM_MODEL`/`ULYSSES_LLM_BASE_URL`
   if you're not using OpenAI directly) — needed for the Proposal Agent to
   draft proposals.

## Running

```bash
uv run ulysses start
```

This starts the monitoring loop: Ulysses polls your mailbox every
`ULYSSES_EMAIL_POLL_INTERVAL_SECONDS` (default 180s) for new Upwork
notification emails, scores each one, and sends scored jobs to your Telegram
chat with inline buttons:

- **Draft Proposal** — the Proposal Agent drafts a cover letter and sends it
  back with **Copy to Clipboard** / **Regenerate** buttons.
- **Build Demo** — the Prototype Agent generates a runnable demo script +
  README and sends it back as a zip file, with the README as a plain-text
  preview message.
- **Skip** — marks the job as skipped; you won't be alerted about it again.
- **Archive** — saves it for later reference in the local database.

Check on things anytime with:

```bash
uv run ulysses status
```

For a job you've already seen (looked up by its Upwork URL), you can also
work from the terminal instead of Telegram:

```bash
# Draft a proposal, printed to the terminal
uv run ulysses draft https://www.upwork.com/jobs/~0112345678901234

# Build a demo prototype, written to ./output/<job_id>/
uv run ulysses build https://www.upwork.com/jobs/~0112345678901234

# Both at once, written to ./output/<job_id>/ (includes proposal.txt)
uv run ulysses go https://www.upwork.com/jobs/~0112345678901234
```

## Development

```bash
# Lint and format
uv run ruff check .
uv run ruff format .

# Tests with coverage
uv run pytest --cov=ulysses --cov-report=term-missing -v
```

## Privacy

- All job data stays on your machine in a local SQLite database
  (`~/.ulysses/ulysses.db`) — nothing is synced to the cloud.
- Job data is only sent to an external service (your configured LLM provider)
  when you explicitly trigger an LLM call — pressing Draft/Regenerate/Build,
  or running `ulysses draft`/`build`/`go`.
- Credentials live only in `.env`, which is git-ignored.
