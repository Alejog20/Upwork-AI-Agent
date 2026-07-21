"""Typer CLI entry point for Hermes."""

from __future__ import annotations

import asyncio
import sys

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table
from telegram.ext import Application

from hermes.agents.notifier import NotifierAgent
from hermes.agents.scout import ScoutAgent
from hermes.config.profile import Profile, load_profile
from hermes.config.settings import Settings, get_settings
from hermes.graph.graph import build_graph
from hermes.models import JobPost, JobScore
from hermes.tools.db import HermesDB
from hermes.tools.email_reader import EmailReader

__all__ = ["app"]

app = typer.Typer(help="Hermes — monitors, scores, and helps you respond to Upwork jobs.")
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
) -> tuple[HermesDB, ScoutAgent, NotifierAgent]:
    db = HermesDB(settings.db_path)
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


@app.command()
def start() -> None:
    """Start the Hermes monitoring loop: scout, score, and notify via Telegram."""
    settings = get_settings()
    profile = load_profile(settings.profile_path)
    _configure_logging(settings)

    console.print("[bold green]Hermes is starting...[/bold green]")
    console.print(
        f"  Watching [cyan]{settings.imap_user}[/cyan] on [cyan]{settings.imap_host}[/cyan]"
    )
    console.print(f"  Poll interval: [cyan]{settings.email_poll_interval_seconds}s[/cyan]")
    console.print("  Press Ctrl+C to stop.\n")

    try:
        asyncio.run(_run_forever(settings, profile))
    except KeyboardInterrupt:
        console.print("\n[yellow]Hermes stopped.[/yellow]")


async def _run_forever(settings: Settings, profile: Profile) -> None:
    db, scout, notifier = _build_dependencies(settings, profile)
    await db.init()

    graph = build_graph(profile, notifier)

    async def on_scored_job(job: JobPost, score: JobScore) -> None:
        config = {"configurable": {"thread_id": job.id}}
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
    db = HermesDB(settings.db_path)
    await db.init()
    try:
        counts = await db.stats()
    finally:
        await db.dispose()

    table = Table(title="Hermes Status")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    for status_name, count in counts.items():
        if status_name == "total":
            continue
        table.add_row(status_name, str(count))
    table.add_row("[bold]total[/bold]", f"[bold]{counts['total']}[/bold]")

    console.print(table)
