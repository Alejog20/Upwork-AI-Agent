# Ulysses — Testing Framework

This document describes how Ulysses is tested: the tools, the conventions, the
coverage bar, and — just as importantly — what's deliberately *not* unit
tested and why. See `CLAUDE.md` for the project-wide standards this framework
implements (`CLAUDE.md` is git-ignored/local-only; this file is the public,
committed reference).

## Stack

- **Runner:** [`pytest`](https://docs.pytest.org/), with `pytest-asyncio`
  (`asyncio_mode = "auto"` in `pyproject.toml` — async test functions need no
  `@pytest.mark.asyncio` decorator).
- **Coverage:** `pytest-cov`.
- **Mocking:** `pytest-mock` (the `mocker` fixture) and `unittest.mock`
  (`AsyncMock`/`MagicMock`) interchangeably — use whichever reads more
  naturally for the case at hand.
- **Snapshot testing:** `syrupy`, for proposal/prototype text output where a
  full-output diff is more useful than field-by-field assertions.
- **CLI testing:** Typer's `CliRunner`, invoking real commands end-to-end
  against a throwaway SQLite DB (see "Isolation" below).

Run the full suite:

```bash
uv run pytest --cov=ulysses --cov-report=term-missing -v
```

Run one file or one test:

```bash
uv run pytest tests/test_proposal.py -v
uv run pytest tests/test_proposal.py -k test_generate_includes_close_field_in_full_text -v
```

## Organization

- `tests/` mirrors `ulysses/`: `ulysses/agents/proposal.py` → `tests/test_proposal.py`,
  `ulysses/tools/analytics.py` → `tests/test_analytics.py`, etc. One test file
  per source module, no exceptions — if a module doesn't have a test file
  yet, that's the gap to close, not a precedent to extend.
- Inside a test file, group related tests into a class per function/concept
  under test: `class TestScoreJob`, `class TestFindBestMatchingExample`, etc.
  Test method names are full sentences describing the behavior, not the
  input: `test_returns_the_example_with_the_highest_cosine_similarity`, not
  `test_case_1`.
- Shared fixtures (`profile`, `fresh_job`, `now`) live in `tests/conftest.py`
  and are available by parameter name in any test file, no import needed.
  Fixtures specific to one file's concerns (e.g. `mock_llm` in
  `test_proposal.py`, `_isolated_settings` in `test_cli.py`) stay local to
  that file.

## Mocking rules

**Every external call is mocked in unit tests — no exceptions.** This means:

- LLM calls (`get_llm()`, `ainvoke_with_retry`, embeddings via `aembed_texts`)
- IMAP (`EmailReader`)
- Telegram Bot API (`NotifierAgent`, `Application`)
- The filesystem, beyond `tmp_path`-scoped writes a test itself makes

A unit test that reaches a real network endpoint is a bug in the test, not a
feature. If you notice a test taking multiple seconds or depending on
credentials, it's almost certainly making a real call it shouldn't — find the
un-mocked construction site (usually a bare `SomeAgent()` instantiated inside
the code under test with no dependency-injection override) and patch it.

**Real end-to-end verification is a separate, deliberate practice, not a
substitute for mocked unit tests.** Before a nontrivial change is considered
done, drive the actual CLI (`uv run ulysses chat`, etc.) against a live,
configured LLM in an isolated scratch `ULYSSES_HOME`/working directory (never
the real `~/.ulysses` or the real `profile.yaml`) to confirm it behaves
correctly for real, not just under mocks. This is how several real bugs in
this codebase were actually caught — mocked tests only prove the code does
what the mock says it does, not that the real output reads correctly. See the
`verify` skill.

## Coverage bar

- **≥ 80%** per module, mechanically enforced in CI (see `.github/workflows/ci.yml`).
- **≥ 90%** for scoring and parsing logic specifically (`agents/scorer.py`,
  `tools/job_parser.py`, `tools/manual_job.py`, `tools/red_flag.py`, and any
  module doing deterministic extraction/classification) — enforced by code
  review, not a separate CI gate, since `pytest-cov`'s `--cov-fail-under`
  only supports a single global threshold, not per-file ones. Check the
  `term-missing` column for these files specifically before calling a change
  to them done.
- **Deliberately excluded from the coverage bar** — long-running daemon
  orchestration and thin infra-wiring code, where unit testing would mean
  mocking nearly everything the function touches for very little confidence
  gained:
  - `ulysses/cli/main.py::run_forever` — the actual scout → score → notify
    polling loop shared by `ulysses start` and the menu bar app. Its pieces
    (`ScoutAgent.run_forever`, `NotifierAgent.handle_scored_job`,
    `_start_telegram_with_retry`, `_shutdown_telegram`) are each tested on
    their own; the full wiring is validated by actually running `ulysses
    start`, not by simulating `asyncio.gather` over mocked Telegram/IMAP
    clients.
  - `ulysses/cli/main.py::_build_dependencies`,
    `_build_telegram_application` — plain constructor wiring with no branches
    worth asserting on.
  - The `rumps`-based menu bar app's native GUI event loop
    (`ulysses/app/menubar.py`) — `rumps` itself isn't mockable in a way that
    proves anything; its testable logic (stats formatting, click handlers)
    is covered, the native rendering loop isn't.

  If you're about to write a test that mocks four or five collaborators just
  to assert one `.assert_called_once()`, stop and ask whether this is one of
  these cases — a real end-to-end run is probably worth more than the test.

## Isolation

Every CLI-level test (`tests/test_cli.py`) uses an autouse `_isolated_settings`
fixture that points `ULYSSES_ULYSSES_HOME` at a `tmp_path`-scoped directory
and clears `get_settings()`'s cache before and after — no test ever touches
the real `~/.ulysses/ulysses.db`. Tests that write prototype/proposal files to
disk additionally `monkeypatch.chdir(tmp_path)` first, since those paths are
relative to the CWD (`./output/<job_id>/`). Never construct a `UlyssesDB`
against a real path in a test.

## Writing tests for a new or changed feature

This is the actual workflow, not just a checklist:

1. **New pure/deterministic function** (no LLM, no I/O) — e.g. a new scoring
   component, a new classifier, a new formatting helper: write direct unit
   tests with plain inputs/outputs, no mocking needed. These are the
   cheapest, most valuable tests in the suite — aim for every branch covered.
2. **New LLM-backed agent method** — mock the chat model the same way
   `test_proposal.py`'s `mock_llm` fixture does: a `MagicMock` whose
   `.bind()` returns itself and whose `.with_structured_output()` returns an
   `AsyncMock` with a canned `.ainvoke` return value. Assert on the *prompt
   sent* (via `structured_llm.ainvoke.await_args.args[0]`) for anything where
   prompt construction is the actual logic being tested, not just the output
   parsing.
3. **New CLI command** — add a `CliRunner`-based test class mirroring the
   existing per-command classes in `test_cli.py` (`TestDraftCommand`,
   `TestWonLostCommands`, etc.): one test for the happy path, one for each
   documented error path (unknown ID, missing data, etc.), all collaborators
   (`ProposalAgent`, `PrototypeAgent`, `NarratorAgent`, etc.) patched via
   `mocker.patch("ulysses.cli.main.<Name>", ...)`.
4. **New DB table/method** — add tests to `test_db.py` against the real
   `UlyssesDB` class (SQLite via `aiosqlite` is fast enough that there's no
   need to mock the DB layer itself — only what's *above* it).
5. **Behavioral change to existing code** — update the existing test(s) that
   cover the old behavior first (they should fail against your change before
   you touch them), then add any new test the change specifically calls for.
   A change that doesn't break any existing test is a signal the area was
   under-tested before your change, not that your change is risk-free — add
   the missing coverage as part of the same change.

A change isn't done until `uv run ruff check .`, `uv run ruff format --check
.`, and `uv run pytest --cov=ulysses --cov-report=term-missing -v` are all
clean — see `CLAUDE.md`'s Definition of Done.
