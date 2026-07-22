"""Typer CLI entry point for Ulysses."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from telegram.ext import Application

from ulysses.agents.notifier import NotifierAgent
from ulysses.agents.proposal import ProposalAgent
from ulysses.agents.scout import ScoutAgent
from ulysses.config.profile import Profile, load_profile
from ulysses.config.settings import Settings, get_settings
from ulysses.graph.graph import build_graph
from ulysses.models import JobPost, JobScore
from ulysses.tools.db import UlyssesDB
from ulysses.tools.email_reader import EmailReader

__all__ = ["app"]

app = typer.Typer(help="Ulysses — monitors, scores, and helps you respond to Upwork jobs.")
console = Console()


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


async def _run_forever(settings: Settings, profile: Profile) -> None:
    db, scout, notifier = _build_dependencies(settings, profile)
    await db.init()

    proposal_agent = ProposalAgent()
    graph = build_graph(profile, notifier, proposal_agent, db)
    notifier.set_draft_handler(_make_draft_handler(db, proposal_agent, notifier, profile))

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
            await notifier.send_error_message(f"Failed to process job: {job.title}")

    telegram_app = Application.builder().token(settings.telegram_bot_token).build()
    telegram_app.add_handler(notifier.callback_handler)

    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.updater.start_polling()

    try:
        await asyncio.gather(
            scout.run_forever(settings.email_poll_interval_seconds, on_scored_job),
            notifier.run_batch_loop(profile.alerts.batch_interval_minutes),
        )
    finally:
        await telegram_app.updater.stop()
        await telegram_app.stop()
        await telegram_app.shutdown()
        await db.dispose()


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
        job_row = await db.get_job_by_url(url)
        if job_row is None:
            console.print(f"[red]No job found for URL:[/red] {url}")
            console.print("Ulysses only knows about jobs it has already seen via email.")
            raise typer.Exit(code=1)

        full = await db.get_full_job(job_row.id)
        if full is None:
            console.print(
                "[red]This job predates detailed storage — re-run scout to refresh it.[/red]"
            )
            raise typer.Exit(code=1)

        job, score = full
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
