# CLAUDE.md — Hermes Project Standards

## Project Identity
Hermes is a production-grade multi-agent AI ecosystem. This is not a prototype. Every decision must prioritize correctness, privacy, maintainability, and developer experience. Reference `Hermes_Architecture.md` for system design.

## Language & Runtime
- Python 3.14 (use new syntax: `type` aliases, `tomllib`, improved typing features)
- Package manager: `uv` exclusively — no pip, no conda, no poetry
- `pyproject.toml` is the single source of truth for dependencies and tooling config

## Code Style
- PEP 8 strictly enforced via `ruff`
- Formatter: `ruff format` (replaces black)
- Linter: `ruff check` with rules: E, F, I, N, UP, ANN, ASYNC, RUF
- Max line length: 100
- All public functions, classes, and modules must have docstrings (Google style)
- Type hints on every function signature — no `Any` unless truly unavoidable
- Use `type` keyword for type aliases (Python 3.12+ syntax)
- Prefer `pathlib.Path` over `os.path` everywhere
- f-strings only — no `.format()` or `%` formatting
- will always avoid the use of emojis in any document the agent touches

## Architecture Patterns
- **Clean Architecture**: agents never import from CLI or app layers; tools never import from agents
- **Dependency injection**: pass dependencies as constructor arguments, never import globals
- **Pydantic v2** for all data models — use `model_validator`, `field_validator`, `computed_field`
- **Async-first**: all I/O operations must be async. Use `asyncio.gather` for parallelism
- No mutable global state. Use `contextvars` if context propagation is needed
- Prefer composition over inheritance

## Agent Framework
- **LangGraph** for orchestration — StateGraph with typed state, explicit edges, interrupt points
- **LangChain** for LLM chains — LCEL (pipe syntax `|`) preferred over legacy chains
- LLM client is accessed only through `tools/llm.py::get_llm()` — never instantiate models directly
- All LLM calls must have retry logic (use `tenacity`) and timeout (30s max)
- Log every LLM call: model, token usage, latency — use `loguru`

## Testing
- Framework: `pytest` with `pytest-asyncio` (mode = `auto`)
- Coverage: `pytest-cov` — minimum 80% per module, 90% for scoring and parsing logic
- Mocking: `pytest-mock` and `unittest.mock` — all external calls (LLM, IMAP, Telegram API) must be mocked in tests
- Test file naming: `tests/test_<module>.py`
- Use fixtures in `conftest.py` for shared objects (DB session, mock LLM, sample job fixtures)
- Every agent must have: unit tests for pure logic, integration tests for the LangGraph node
- Use `pytest.mark.asyncio` for async tests; use `anyio` backend where needed
- Snapshot testing for proposal and prototype text output: use `syrupy`
- Run tests: `uv run pytest --cov=hermes --cov-report=term-missing -v`

## Logging
- Use `loguru` exclusively — never `print()`, never `logging` stdlib directly
- Log levels: DEBUG for agent internals, INFO for state transitions, WARNING for red flags, ERROR for failures
- Structured logging: always include `job_id` and `agent` name in log context using `logger.bind()`
- Log file: `~/.hermes/logs/hermes.log` with rotation (10 MB, 7 days retention)

## Database
- SQLModel + aiosqlite for async SQLite
- DB file: `~/.hermes/hermes.db`
- All DB operations go through `tools/db.py` — no raw SQL outside that module
- Use Alembic for migrations (even for SQLite — keeps schema changes traceable)

## Security & Privacy
- No credentials in code or config files — `.env` only, loaded via `pydantic-settings`
- `.env` and `~/.hermes/` are in `.gitignore`
- App-specific passwords for email (never main account password)
- No job data, proposals, or personal data sent to any external service except the configured LLM API
- All Telegram messages go to your personal chat ID only — validate in `notifier.py`

## Error Handling
- Never use bare `except:` — always catch specific exceptions
- All agent failures must be caught at the graph level and surfaced as Telegram error messages
- Use `Result` pattern for functions that can fail: return `tuple[T, None] | tuple[None, Exception]` or use a `Result` type via `returns` library
- IMAP connection failures must retry with exponential backoff before alerting

## File & Directory Conventions
- Source code: `hermes/` package
- Tests: `tests/` (mirrors `hermes/` structure)
- Templates: `hermes/templates/` (proposals and prototypes)
- Config: `hermes/config/` (settings.py, profile.yaml)
- Output artifacts: `~/.hermes/output/<job_id>/`
- Logs: `~/.hermes/logs/`

## CLI Standards
- Framework: `Typer` with `rich` integration for output
- All commands must have `--help` text
- Use `typer.echo()` never `print()`
- Long-running commands show a `rich.Progress` spinner
- All destructive commands require `--confirm` flag or prompt for confirmation

## macOS App Standards
- Menu bar app: `rumps` library
- Never block the main thread — all agent work runs in a background `asyncio` event loop via `threading.Thread`
- macOS notifications via `rumps.notification()` — title ≤ 50 chars, message ≤ 100 chars
- LaunchAgent plist stored at `~/Library/LaunchAgents/com.hermes.agent.plist`

## Git Conventions
- Commit format: `type(scope): description` — types: feat, fix, refactor, test, docs, chore
- One logical change per commit
- Never commit: `.env`, `*.db`, `__pycache__`, `.ruff_cache`, `~/.hermes/`
- Branch naming: `phase-N/feature-name`

## Definition of Done (per phase)
A phase is complete when:
1. All tasks listed in the Claude Code prompt are implemented
2. `ruff check .` returns zero errors
3. `ruff format --check .` returns zero diffs
4. `pytest --cov=hermes --cov-report=term-missing` passes with ≥ 80% coverage
5. The feature works end-to-end in a manual test
6. All new public APIs have docstrings and type hints