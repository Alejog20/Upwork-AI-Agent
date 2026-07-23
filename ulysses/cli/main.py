"""Typer CLI entry point for Ulysses."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from telegram.error import InvalidToken, NetworkError, RetryAfter
from telegram.ext import Application

from ulysses.agents.notifier import NotifierAgent
from ulysses.agents.proposal import ProposalAgent
from ulysses.agents.prototype import PrototypeAgent, build_prototype_zip
from ulysses.agents.scout import ScoutAgent
from ulysses.config.profile import Profile, load_profile
from ulysses.config.settings import Settings, get_settings
from ulysses.graph.graph import build_graph
from ulysses.models import GeneratedPrototype, JobPost, JobScore
from ulysses.tools.db import UlyssesDB
from ulysses.tools.email_reader import EmailReader

__all__ = ["app"]

app = typer.Typer(help="Ulysses — monitors, scores, and helps you respond to Upwork jobs.")
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
        asyncio.run(_run_forever(settings, profile))
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


async def _run_forever(settings: Settings, profile: Profile) -> None:
    db, scout, notifier = _build_dependencies(settings, profile)
    await db.init()

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
            scout.run_forever(settings.email_poll_interval_seconds, on_scored_job),
            notifier.run_batch_loop(profile.alerts.batch_interval_minutes),
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
