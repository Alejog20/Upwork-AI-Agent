"""Typer CLI entry point for Ulysses."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path

import typer
import yaml
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from telegram.error import InvalidToken, NetworkError, RetryAfter
from telegram.ext import Application

from ulysses.agents.notifier import InstantAlertHook, NotifierAgent
from ulysses.agents.proposal import ProposalAgent
from ulysses.agents.prototype import PrototypeAgent, build_prototype_zip
from ulysses.agents.scorer import score_job
from ulysses.agents.scout import ScoutAgent
from ulysses.config.profile import (
    Profile,
    ProfileKeyError,
    load_profile,
    save_profile,
    set_profile_value,
)
from ulysses.config.settings import Settings, get_settings
from ulysses.graph.graph import build_graph
from ulysses.models import GeneratedPrototype, JobPost, JobScore
from ulysses.tools.db import Job, JobStatus, UlyssesDB
from ulysses.tools.email_reader import EmailReader
from ulysses.tools.launch_agent import install_launch_agent, uninstall_launch_agent
from ulysses.tools.manual_job import ManualJobParseError, extract_job_from_text

__all__ = ["app", "run_forever"]

app = typer.Typer(help="Ulysses — monitors, scores, and helps you respond to Upwork jobs.")
config_app = typer.Typer(help="View or update your profile.yaml settings.")
app.add_typer(config_app, name="config")
console = Console()

_TELEGRAM_HTTP_TIMEOUT_SECONDS = 20
_TELEGRAM_STARTUP_MAX_BACKOFF_SECONDS = 60


def _configure_logging(settings: Settings) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        settings.log_path,
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        enqueue=True,
    )


def _build_dependencies(
    settings: Settings, profile: Profile
) -> tuple[UlyssesDB, ScoutAgent, NotifierAgent]:
    db = UlyssesDB(settings.db_path)
    email_reader = EmailReader(
        host=settings.imap_host,
        port=settings.imap_port,
        user=settings.imap_user,
        app_password=settings.imap_app_password,
        mailbox=settings.imap_mailbox,
    )
    scout = ScoutAgent(email_reader=email_reader, db=db, profile=profile)
    notifier = NotifierAgent(
        bot_token=settings.telegram_bot_token, chat_id=settings.telegram_chat_id, db=db
    )
    return db, scout, notifier


def _make_draft_handler(
    db: UlyssesDB, proposal_agent: ProposalAgent, notifier: NotifierAgent, profile: Profile
) -> Callable[[str], Awaitable[None]]:
    async def on_draft_requested(job_id: str) -> None:
        full = await db.get_full_job(job_id)
        if full is None:
            logger.warning("Draft requested for unknown or pre-Phase-2 job_id={}", job_id)
            await notifier.send_error_message(
                "Can't draft this job — it was seen before detailed data was stored."
            )
            return
        job, score = full
        draft = await proposal_agent.generate(job, score, profile)
        await db.add_proposal_draft(job_id, draft.full_text)
        await notifier.send_proposal_draft(job_id, draft.full_text)

    return on_draft_requested


def _make_build_handler(
    db: UlyssesDB, prototype_agent: PrototypeAgent, notifier: NotifierAgent, profile: Profile
) -> Callable[[str], Awaitable[None]]:
    async def on_build_requested(job_id: str) -> None:
        full = await db.get_full_job(job_id)
        if full is None:
            logger.warning("Build requested for unknown or pre-Phase-2 job_id={}", job_id)
            await notifier.send_error_message(
                "Can't build a demo for this job — it was seen before detailed data was stored."
            )
            return
        job, score = full
        prototype = await prototype_agent.generate(job, score, profile)
        await _persist_prototype_files(db, job_id, prototype)
        zip_bytes = build_prototype_zip(prototype)
        await notifier.send_prototype_zip(job_id, prototype, zip_bytes)

    return on_build_requested


async def _persist_prototype_files(
    db: UlyssesDB, job_id: str, prototype: GeneratedPrototype
) -> None:
    for filename, content in (
        ("demo.py", prototype.demo_script),
        ("requirements.txt", prototype.requirements_txt),
        ("README.md", prototype.readme_md),
        ("config.example.env", prototype.config_example_env),
    ):
        await db.add_prototype_file(job_id, filename, content)


@app.command()
def start() -> None:
    """Start the Ulysses monitoring loop: scout, score, and notify via Telegram."""
    settings = get_settings()
    profile = load_profile(settings.profile_path)
    _configure_logging(settings)

    console.print("[bold green]Ulysses is starting...[/bold green]")
    console.print(
        f"  Watching [cyan]{settings.imap_user}[/cyan] on [cyan]{settings.imap_host}[/cyan]"
    )
    console.print(f"  Poll interval: [cyan]{settings.email_poll_interval_seconds}s[/cyan]")
    console.print("  Press Ctrl+C to stop.\n")

    try:
        asyncio.run(run_forever(settings, profile))
    except KeyboardInterrupt:
        console.print("\n[yellow]Ulysses stopped.[/yellow]")
    except InvalidToken:
        console.print(
            "\n[red]Telegram rejected the bot token — check ULYSSES_TELEGRAM_BOT_TOKEN "
            "in .env.[/red]"
        )
        raise typer.Exit(code=1) from None
    except Exception:
        logger.exception("Ulysses crashed")
        console.print(f"\n[red]Ulysses crashed — see {get_settings().log_path} for details.[/red]")
        raise typer.Exit(code=1) from None


async def run_forever(
    settings: Settings,
    profile: Profile,
    *,
    stop_event: threading.Event | None = None,
    paused_event: threading.Event | None = None,
    on_instant_alert: InstantAlertHook | None = None,
) -> None:
    """Run the full agent loop (scout -> scorer -> notifier) until stopped.

    This is the shared entry point for both `ulysses start` and the macOS
    menu bar app (`ulysses.app.menubar`) -- they differ only in what they
    pass for `stop_event`/`paused_event`/`on_instant_alert`. Plain CLI usage
    leaves all three `None`, which preserves the original run-forever,
    never-paused, Telegram-only behavior.
    """
    db, scout, notifier = _build_dependencies(settings, profile)
    await db.init()
    if on_instant_alert is not None:
        notifier.set_instant_alert_hook(on_instant_alert)

    proposal_agent = ProposalAgent()
    prototype_agent = PrototypeAgent()
    graph = build_graph(profile, notifier, proposal_agent, prototype_agent, db)
    notifier.set_draft_handler(_make_draft_handler(db, proposal_agent, notifier, profile))
    notifier.set_build_handler(_make_build_handler(db, prototype_agent, notifier, profile))

    async def on_scored_job(job: JobPost, score: JobScore) -> None:
        config = {"configurable": {"thread_id": job.id}}
        try:
            await graph.ainvoke(
                {
                    "job": job,
                    "score": None,
                    "user_action": None,
                    "proposal_draft": None,
                    "prototype_files": None,
                    "notification_sent": False,
                    "completed": False,
                },
                config=config,
            )
        except Exception:
            logger.exception("Pipeline failed for job_id={}", job.id)
            try:
                await notifier.send_error_message(f"Failed to process job: {job.title}")
            except Exception:
                logger.exception("Also failed to notify about the pipeline failure")

    telegram_app = _build_telegram_application(settings)
    telegram_app.add_handler(notifier.callback_handler)

    try:
        await _start_telegram_with_retry(telegram_app)
        await asyncio.gather(
            scout.run_forever(
                settings.email_poll_interval_seconds,
                on_scored_job,
                stop_event=stop_event,
                paused_event=paused_event,
            ),
            notifier.run_batch_loop(profile.alerts.batch_interval_minutes, stop_event=stop_event),
        )
    finally:
        await _shutdown_telegram(telegram_app)
        await db.dispose()


def _build_telegram_application(settings: Settings) -> Application:
    return (
        Application.builder()
        .token(settings.telegram_bot_token)
        .connect_timeout(_TELEGRAM_HTTP_TIMEOUT_SECONDS)
        .read_timeout(_TELEGRAM_HTTP_TIMEOUT_SECONDS)
        .get_updates_read_timeout(_TELEGRAM_HTTP_TIMEOUT_SECONDS)
        .build()
    )


async def _start_telegram_with_retry(telegram_app: Application) -> None:
    """Start polling, retrying transient network errors with capped backoff.

    Permanent errors (e.g. `InvalidToken`) are not caught here — they
    propagate immediately so the operator can fix the configuration instead
    of retrying forever against a bot token that will never work.
    """
    attempt = 0
    while True:
        try:
            await telegram_app.initialize()
            await telegram_app.start()
            await telegram_app.updater.start_polling()
            return
        except (NetworkError, RetryAfter) as exc:
            attempt += 1
            wait_seconds = min(_TELEGRAM_STARTUP_MAX_BACKOFF_SECONDS, 2**attempt)
            logger.warning(
                "Telegram startup failed (attempt {}): {} -- retrying in {}s",
                attempt,
                exc,
                wait_seconds,
            )
            await asyncio.sleep(wait_seconds)


async def _shutdown_telegram(telegram_app: Application) -> None:
    try:
        if telegram_app.updater is not None and telegram_app.updater.running:
            await telegram_app.updater.stop()
        if telegram_app.running:
            await telegram_app.stop()
        await telegram_app.shutdown()
    except Exception:
        logger.exception("Error while shutting down Telegram (ignoring, exiting anyway)")


@app.command()
def status() -> None:
    """Show job counts and pipeline status."""
    settings = get_settings()
    asyncio.run(_print_status(settings))


async def _print_status(settings: Settings) -> None:
    db = UlyssesDB(settings.db_path)
    await db.init()
    try:
        counts = await db.stats()
    finally:
        await db.dispose()

    table = Table(title="Ulysses Status")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    for status_name, count in counts.items():
        if status_name == "total":
            continue
        table.add_row(status_name, str(count))
    table.add_row("[bold]total[/bold]", f"[bold]{counts['total']}[/bold]")

    console.print(table)


@app.command()
def draft(url: str) -> None:
    """Draft a proposal for a job Ulysses has already seen (looked up by its Upwork URL)."""
    settings = get_settings()
    profile = load_profile(settings.profile_path)
    asyncio.run(_draft_async(settings, profile, url))


async def _draft_async(settings: Settings, profile: Profile, url: str) -> None:
    db = UlyssesDB(settings.db_path)
    await db.init()
    try:
        job, score = await _lookup_full_job(db, url)
        proposal_agent = ProposalAgent()
        with console.status("[bold cyan]Drafting proposal...[/bold cyan]"):
            generated = await proposal_agent.generate(job, score, profile)
        await db.add_proposal_draft(job.id, generated.full_text)

        console.print(Panel(generated.full_text, title=f"Proposal — {job.title}"))
        console.print(
            f"[dim]Category: {generated.category} | Timeline: {generated.timeline} | "
            f"Bid: ${generated.bid_usd:.0f}[/dim]"
        )
    finally:
        await db.dispose()


async def _lookup_full_job(db: UlyssesDB, url: str) -> tuple[JobPost, JobScore]:
    job_row = await db.get_job_by_url(url)
    if job_row is None:
        console.print(f"[red]No job found for URL:[/red] {url}")
        console.print("Ulysses only knows about jobs it has already seen via email.")
        raise typer.Exit(code=1)

    full = await db.get_full_job(job_row.id)
    if full is None:
        console.print("[red]This job predates detailed storage — re-run scout to refresh it.[/red]")
        raise typer.Exit(code=1)
    return full


def _write_prototype_to_disk(prototype: GeneratedPrototype, job_id: str) -> Path:
    output_dir = Path("./output") / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "demo.py").write_text(prototype.demo_script, encoding="utf-8")
    (output_dir / "requirements.txt").write_text(prototype.requirements_txt, encoding="utf-8")
    (output_dir / "README.md").write_text(prototype.readme_md, encoding="utf-8")
    (output_dir / "config.example.env").write_text(prototype.config_example_env, encoding="utf-8")
    return output_dir


@app.command()
def build(url: str) -> None:
    """Build a demo prototype for a job Ulysses has already seen, saved to ./output/<job_id>/."""
    settings = get_settings()
    profile = load_profile(settings.profile_path)
    asyncio.run(_build_async(settings, profile, url))


async def _build_async(settings: Settings, profile: Profile, url: str) -> None:
    db = UlyssesDB(settings.db_path)
    await db.init()
    try:
        job, score = await _lookup_full_job(db, url)
        prototype_agent = PrototypeAgent()
        with console.status("[bold cyan]Building prototype...[/bold cyan]"):
            generated = await prototype_agent.generate(job, score, profile)
        await _persist_prototype_files(db, job.id, generated)

        output_dir = _write_prototype_to_disk(generated, job.id)
        console.print(f"[green]Prototype written to {output_dir}[/green]")
        console.print(Panel(generated.readme_md, title="README.md"))
    finally:
        await db.dispose()


@app.command()
def go(url: str) -> None:
    """Draft a proposal AND build a demo for a job, saved to ./output/<job_id>/."""
    settings = get_settings()
    profile = load_profile(settings.profile_path)
    asyncio.run(_go_async(settings, profile, url))


async def _go_async(settings: Settings, profile: Profile, url: str) -> None:
    db = UlyssesDB(settings.db_path)
    await db.init()
    try:
        job, score = await _lookup_full_job(db, url)

        proposal_agent = ProposalAgent()
        prototype_agent = PrototypeAgent()
        with console.status("[bold cyan]Drafting proposal and building prototype...[/bold cyan]"):
            proposal, prototype = await asyncio.gather(
                proposal_agent.generate(job, score, profile),
                prototype_agent.generate(job, score, profile),
            )
        await db.add_proposal_draft(job.id, proposal.full_text)
        await _persist_prototype_files(db, job.id, prototype)

        output_dir = _write_prototype_to_disk(prototype, job.id)
        (output_dir / "proposal.txt").write_text(proposal.full_text, encoding="utf-8")

        console.print(f"[green]Output written to {output_dir}[/green]")
        console.print(Panel(proposal.full_text, title="Proposal"))
        console.print(Panel(prototype.readme_md, title="README.md"))
    finally:
        await db.dispose()


_CHAT_QUIT_COMMANDS = {"quit", "exit"}


@app.command()
def chat() -> None:
    """Interactive chat: paste job listings and get scored proposals + prototypes, one session."""
    settings = get_settings()
    profile = load_profile(settings.profile_path)
    asyncio.run(_chat_async(settings, profile))


async def _chat_async(settings: Settings, profile: Profile) -> None:
    console.print("[bold green]Ulysses chat[/bold green] — paste a job listing below.")
    console.print(
        "Paste the listing, then press [cyan]Ctrl+D[/cyan] to submit it. Press "
        "[cyan]Ctrl+D[/cyan] again with nothing typed (or type [cyan]quit[/cyan]/"
        "[cyan]exit[/cyan]) to leave.\n"
    )

    db = UlyssesDB(settings.db_path)
    await db.init()
    try:
        while True:
            raw_text = _read_pasted_job_listing()
            if raw_text is None:
                console.print("[yellow]Goodbye.[/yellow]")
                return
            if not raw_text.strip():
                console.print(
                    "[yellow]Nothing pasted — try again, or type quit to leave.[/yellow]\n"
                )
                continue

            try:
                await _process_pasted_job(db, profile, raw_text)
            except ManualJobParseError as exc:
                console.print(f"[red]Couldn't read that listing:[/red] {exc}\n")
            except Exception:
                logger.exception("Failed to process a pasted job listing")
                console.print(
                    "[red]Something went wrong processing that listing — see the log for "
                    "details. You can paste another one.[/red]\n"
                )
    finally:
        await db.dispose()


async def _process_pasted_job(db: UlyssesDB, profile: Profile, raw_text: str) -> None:
    """Extract, score, draft, and prototype one pasted job listing.

    `ProposalAgent`/`PrototypeAgent` are constructed here, not once for the
    whole chat session, so quitting (or an extraction failure) never
    requires LLM credentials to be configured at all -- constructing them
    only wraps the already process-wide-cached `get_llm()` client, so there's
    no real cost to doing it per job instead of once per session.
    """
    with console.status("[bold cyan]Extracting job details...[/bold cyan]"):
        job = await extract_job_from_text(raw_text)

    score = score_job(job, profile)
    await db.upsert_job(
        Job(
            id=job.id,
            title=job.title,
            description=job.description,
            url=job.url,
            score=score.total_score,
            category=score.gig_category.value,
            status=JobStatus.NEW,
            posted_at=job.posted_at,
            job_json=job.model_dump_json(),
            score_json=score.model_dump_json(),
        )
    )
    _print_score_summary(job, score)

    proposal_agent = ProposalAgent()
    prototype_agent = PrototypeAgent()
    with console.status("[bold cyan]Drafting proposal and building prototype...[/bold cyan]"):
        proposal, prototype = await asyncio.gather(
            proposal_agent.generate(job, score, profile),
            prototype_agent.generate(job, score, profile),
        )
    await db.add_proposal_draft(job.id, proposal.full_text)
    await _persist_prototype_files(db, job.id, prototype)

    output_dir = _write_prototype_to_disk(prototype, job.id)
    (output_dir / "proposal.txt").write_text(proposal.full_text, encoding="utf-8")

    console.print(f"[green]Output written to {output_dir}[/green]")
    console.print(Panel(proposal.full_text, title="Proposal"))
    console.print(Panel(prototype.readme_md, title="README.md"))
    console.print("\n[dim]Paste the next job listing, or type quit to leave.[/dim]\n")


def _print_score_summary(job: JobPost, score: JobScore) -> None:
    table = Table(title=f"Score — {job.title}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Total", f"{score.total_score:.0f}/100")
    table.add_row("Category", score.gig_category.value)
    table.add_row("Recommendation", score.recommendation.value)
    if score.red_flags:
        table.add_row("Red flags", ", ".join(score.red_flags))
    console.print(table)


def _read_pasted_job_listing() -> str | None:
    """Read one multi-line pasted job listing from stdin, submitted with Ctrl+D (EOF).

    EOF is a terminal-level signal, not typed text, so it can never collide
    with the pasted content the way a typed sentinel word could -- pasted
    text commonly has no trailing newline, so a sentinel typed right after
    it would silently concatenate onto the same line instead of becoming its
    own line, and the submission would never fire. Typing "quit"/"exit"
    (case-insensitive) as the very first line, or pressing Ctrl+D again with
    nothing typed yet, returns `None` instead -- the caller treats that as
    "leave the chat".
    """
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            return "\n".join(lines) if lines else None
        stripped = line.strip()
        if not lines and stripped.lower() in _CHAT_QUIT_COMMANDS:
            return None
        lines.append(line)


@app.command()
def queue(
    min_score: float = typer.Option(
        0.0, "--min-score", help="Only show jobs at or above this score."
    ),
    category: str | None = typer.Option(
        None, "--category", help="Only show jobs in this category (e.g. tier1)."
    ),
) -> None:
    """List known jobs, optionally filtered by minimum score and/or category."""
    settings = get_settings()
    asyncio.run(_queue_async(settings, min_score, category))


async def _queue_async(settings: Settings, min_score: float, category: str | None) -> None:
    db = UlyssesDB(settings.db_path)
    await db.init()
    try:
        jobs = await db.list_jobs(min_score=min_score, category=category)
    finally:
        await db.dispose()

    if not jobs:
        console.print("[yellow]No jobs match those filters.[/yellow]")
        return

    table = Table(title="Ulysses Job Queue")
    table.add_column("Score", justify="right", style="magenta")
    table.add_column("Category", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Title")
    table.add_column("URL", overflow="fold")
    for job in jobs:
        table.add_row(f"{job.score:.0f}", job.category, job.status.value, job.title, job.url)

    console.print(table)


@app.command()
def archive(job_id: str) -> None:
    """Mark a job as archived."""
    settings = get_settings()
    asyncio.run(_archive_async(settings, job_id))


async def _archive_async(settings: Settings, job_id: str) -> None:
    db = UlyssesDB(settings.db_path)
    await db.init()
    try:
        job = await db.get_job(job_id)
        if job is None:
            console.print(f"[red]No job found with id:[/red] {job_id}")
            raise typer.Exit(code=1)
        await db.update_status(job_id, JobStatus.ARCHIVED)
        console.print(f"[green]Archived:[/green] {job.title}")
    finally:
        await db.dispose()


@config_app.command("show")
def config_show() -> None:
    """Print the current profile.yaml contents."""
    settings = get_settings()
    profile = load_profile(settings.profile_path)
    dumped = yaml.safe_dump(profile.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    console.print(Panel(dumped, title=str(settings.profile_path)))


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    """Set a single profile.yaml field, e.g. `ulysses config set freelancer.rate_usd_hr 30`."""
    settings = get_settings()
    profile = load_profile(settings.profile_path)
    try:
        updated = set_profile_value(profile, key, value)
    except ProfileKeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    save_profile(updated, settings.profile_path)
    console.print(f"[green]Set[/green] {key} = {value}")


@app.command()
def install() -> None:
    """Install a macOS LaunchAgent so `ulysses start` runs automatically on login."""
    settings = get_settings()
    try:
        path = install_launch_agent(Path.cwd(), settings.log_dir)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        console.print(f"[red]Failed to install LaunchAgent:[/red] {exc}")
        raise typer.Exit(code=1) from None
    console.print(f"[green]Installed and started LaunchAgent:[/green] {path}")
    console.print("Ulysses will now start automatically on login.")


@app.command()
def uninstall() -> None:
    """Remove the macOS LaunchAgent installed by `ulysses install`."""
    removed = uninstall_launch_agent()
    if removed:
        console.print("[green]LaunchAgent removed.[/green] Ulysses will no longer auto-start.")
    else:
        console.print("[yellow]No LaunchAgent was installed.[/yellow]")
