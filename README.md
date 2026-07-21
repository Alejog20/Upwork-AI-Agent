# Claudio

Claudio is a multi agent AI ecosystem that monitors your email for new Upwork
job postings, scores them against your profile, and pushes the good ones to
Telegram with one-tap actions to draft a proposal or build a demo prototype.

See `HERMES-ARQUITECHTURE.md` for the full system design and `CLAUDE.md` for
project development standards.

Current status: **Phase 1** — Scout, Scorer, and the Telegram notifier with
inline buttons (skip/archive working; draft/build are wired into the graph
but implemented in Phases 2-3).

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
     `HERMES_IMAP_APP_PASSWORD`.
   - **iCloud**: enable two-factor authentication, then create an
     [app-specific password](https://support.apple.com/en-us/102654) at
     appleid.apple.com and use it as `HERMES_IMAP_APP_PASSWORD`. Set
     `HERMES_IMAP_PROVIDER=icloud`.

   Never use your real account password — Hermes only ever needs an
   app-specific one.

4. Create a Telegram bot:

   - Message [@BotFather](https://t.me/BotFather) on Telegram, send `/newbot`,
     and follow the prompts. Copy the token it gives you into
     `HERMES_TELEGRAM_BOT_TOKEN`.
   - Message [@userinfobot](https://t.me/userinfobot) to get your numeric
     chat ID, and put it in `HERMES_TELEGRAM_CHAT_ID`. Hermes only ever sends
     to and accepts button presses from this chat.

5. Review `hermes/config/profile.yaml` and adjust your skills, GitHub repos,
   rate, and scoring thresholds to match your own profile.

## Running

```bash
uv run hermes start
```

This starts the monitoring loop: Hermes polls your mailbox every
`HERMES_EMAIL_POLL_INTERVAL_SECONDS` (default 180s) for new Upwork
notification emails, scores each one, and sends scored jobs to your Telegram
chat with inline buttons:

- **Draft Proposal** / **Build Demo** — reserved for the Proposal and
  Prototype Agents (Phases 2-3).
- **Skip** — marks the job as skipped; you won't be alerted about it again.
- **Archive** — saves it for later reference in the local database.

Check on things anytime with:

```bash
uv run hermes status
```

## Development

```bash
# Lint and format
uv run ruff check .
uv run ruff format .

# Tests with coverage
uv run pytest --cov=hermes --cov-report=term-missing -v
```

## Privacy

- All job data stays on your machine in a local SQLite database
  (`~/.hermes/hermes.db`) — nothing is synced to the cloud.
- Job data is only sent to an external service when you explicitly trigger an
  LLM call (drafting a proposal or building a prototype, in later phases).
- Credentials live only in `.env`, which is git-ignored.
