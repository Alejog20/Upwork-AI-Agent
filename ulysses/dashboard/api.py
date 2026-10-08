"""FastAPI app for the live dashboard: a read/action API over the same SQLite DB
and agents the CLI/Telegram surfaces already use, plus a WebSocket for live updates.

Mounted and served by a `uvicorn.Server` task inside `ulysses.cli.main.run_forever`
(see `_build_dashboard_server`) -- this module only builds the `FastAPI` app, it
never runs its own server or owns its own event loop.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from ulysses.actions import archive_job, build_job, draft_job, record_job_outcome, skip_job
from ulysses.agents.chat import (
    CHAT_SYSTEM_PROMPT,
    ChatAgent,
    build_advisor_profile_summary,
    build_job_context_message,
    build_queue_digest,
)
from ulysses.agents.proposal import ProposalAgent
from ulysses.agents.prototype import PrototypeAgent, build_prototype_zip
from ulysses.config.profile import Profile
from ulysses.models import GeneratedPrototype, JobNotFoundError, JobPost, JobScore
from ulysses.tools.analytics import (
    average_connects_spent_per_win,
    average_score_won_vs_lost,
    scoring_weight_suggestions,
    win_rate_by_category,
    win_rate_by_red_flags,
    win_rate_by_score_bucket,
)
from ulysses.tools.db import Job, JobStatus, UlyssesDB
from ulysses.tools.events import DashboardEventBus

__all__ = ["build_dashboard_app"]

_FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"
_GENERAL_CHAT_THREAD_ID = "__general__"
_MAX_CHAT_HISTORY_MESSAGES = 12  # ~6 exchanges -- a plain token-cost cap, not a context limit

_PLACEHOLDER_HTML = """
<!doctype html>
<html>
<head><title>Ulysses Dashboard</title></head>
<body style="font-family: system-ui; background: #0b0f14; color: #e6edf3;
             display: flex; align-items: center; justify-content: center;
             height: 100vh; margin: 0;">
  <div style="text-align: center; max-width: 32rem;">
    <h1>Dashboard frontend not built yet</h1>
    <p>Run <code>cd ulysses/dashboard/frontend &amp;&amp; npm install &amp;&amp;
       npm run build</code> and reload this page. The agent loop itself is
       running fine either way.</p>
  </div>
</body>
</html>
"""


class OutcomeRequest(BaseModel):
    """Body for `POST /api/jobs/{job_id}/outcome`."""

    won: bool
    contract_value_usd: float | None = None
    connects_spent: int | None = None
    note: str | None = None


class DraftEditRequest(BaseModel):
    """Body for `PUT /api/jobs/{job_id}/draft`."""

    content: str


def build_dashboard_app(
    db: UlyssesDB,
    profile: Profile,
    proposal_agent: ProposalAgent,
    prototype_agent: PrototypeAgent,
    events: DashboardEventBus,
    chat_agent: ChatAgent,
) -> FastAPI:
    """Build the dashboard's FastAPI app, bound to one agent-loop process's dependencies."""
    app = FastAPI(title="Ulysses Dashboard")

    @app.get("/api/jobs")
    async def list_jobs(
        min_score: float = 0.0,
        category: str | None = None,
        status: JobStatus | None = None,
        source: str | None = None,
        max_proposals: int | None = None,
        has_red_flags: bool | None = None,
    ) -> list[dict[str, Any]]:
        jobs = await db.list_jobs(min_score=min_score, category=category, status=status)
        summaries = [_job_summary(job) for job in jobs]
        if source is not None:
            summaries = [s for s in summaries if s["source"] == source]
        if max_proposals is not None:
            summaries = [
                s
                for s in summaries
                if s.get("proposals_count") is not None and s["proposals_count"] <= max_proposals
            ]
        if has_red_flags is not None:
            summaries = [s for s in summaries if bool(s.get("red_flags")) == has_red_flags]
        return summaries

    @app.get("/api/jobs/stale")
    async def list_stale_jobs(days: int = 5) -> list[dict[str, Any]]:
        """List applied-but-unresolved jobs (notified/drafted/built) seen over `days` ago.

        Directly answers the most-repeated freelancer pain point found during
        feature research: proposals vanish into a black hole once sent, with
        no systematic "this has gone quiet" signal. `seen_at` round-trips out
        of SQLite as a naive datetime even though it's stored via
        `datetime.now(UTC)` -- compare against a naive-UTC "now" rather than
        an aware one to avoid a naive/aware comparison error.
        """
        threshold = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
        stale: list[dict[str, Any]] = []
        for status in (JobStatus.NOTIFIED, JobStatus.DRAFTED, JobStatus.BUILT):
            for job in await db.list_jobs(status=status):
                if job.seen_at >= threshold:
                    continue
                summary = _job_summary(job)
                summary["days_since_seen"] = (
                    datetime.now(UTC).replace(tzinfo=None) - job.seen_at
                ).days
                stale.append(summary)
        stale.sort(key=lambda s: s["days_since_seen"], reverse=True)
        return stale

    @app.get("/api/jobs/{job_id}")
    async def job_detail(job_id: str) -> dict[str, Any]:
        full = await db.get_full_job(job_id)
        if full is None:
            raise HTTPException(status_code=404, detail="Job not found")
        job, score = full
        drafts = await db.get_proposal_drafts(job_id)
        files = await db.get_prototype_files(job_id)
        return {
            "job": job.model_dump(mode="json"),
            "score": score.model_dump(mode="json"),
            "proposal_drafts": [draft.content for draft in drafts],
            "prototype_files": [f.filename for f in files],
        }

    @app.get("/api/stats")
    async def stats() -> dict[str, int]:
        return await db.stats()

    @app.get("/api/analytics")
    async def analytics() -> dict[str, Any]:
        pairs = await db.list_jobs_with_outcomes()
        return {
            "win_rate_by_category": win_rate_by_category(pairs),
            "win_rate_by_score_bucket": win_rate_by_score_bucket(pairs),
            "win_rate_by_red_flags": win_rate_by_red_flags(pairs),
            "average_score_won_vs_lost": average_score_won_vs_lost(pairs),
            "average_connects_spent_per_win": average_connects_spent_per_win(pairs),
            "scoring_weight_suggestions": scoring_weight_suggestions(pairs),
        }

    @app.post("/api/jobs/{job_id}/draft")
    async def request_draft(job_id: str) -> dict[str, Any]:
        try:
            draft = await draft_job(db, proposal_agent, profile, job_id, events=events)
        except JobNotFoundError:
            raise HTTPException(status_code=404, detail="Job not found") from None
        return {"content": draft.full_text}

    @app.put("/api/jobs/{job_id}/draft")
    async def save_draft(job_id: str, body: DraftEditRequest) -> dict[str, Any]:
        """Persist a hand-edited draft as the latest version, no LLM call involved.

        Appends via the same `add_proposal_draft` call `draft_job` already
        uses -- draft history is append-only by design, so an edit is just
        another row with the edited text as its content.
        """
        job = await db.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        await db.add_proposal_draft(job_id, body.content)
        return {"content": body.content}

    @app.post("/api/jobs/{job_id}/build")
    async def request_build(job_id: str) -> dict[str, Any]:
        try:
            prototype, _ = await build_job(db, prototype_agent, profile, job_id, events=events)
        except JobNotFoundError:
            raise HTTPException(status_code=404, detail="Job not found") from None
        return {"zip_filename": prototype.zip_filename, "readme_md": prototype.readme_md}

    @app.get("/api/jobs/{job_id}/prototype.zip")
    async def download_prototype_zip(job_id: str) -> StreamingResponse:
        full = await db.get_full_job(job_id)
        files = await db.get_prototype_files(job_id)
        if full is None or not files:
            raise HTTPException(status_code=404, detail="No prototype built for this job yet")
        _, score = full
        content_by_filename = {f.filename: f.content for f in files}
        prototype = GeneratedPrototype(
            job_id=job_id,
            category=score.gig_category.value,
            demo_script=content_by_filename.get("demo.py", ""),
            requirements_txt=content_by_filename.get("requirements.txt", ""),
            readme_md=content_by_filename.get("README.md", ""),
            config_example_env=content_by_filename.get("config.example.env", ""),
            zip_filename=f"ulysses_demo_{job_id}.zip",
        )
        zip_bytes = build_prototype_zip(prototype)
        return StreamingResponse(
            io.BytesIO(zip_bytes),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={prototype.zip_filename}"},
        )

    @app.post("/api/jobs/{job_id}/skip")
    async def request_skip(job_id: str) -> dict[str, Any]:
        try:
            job = await skip_job(db, job_id, events=events)
        except JobNotFoundError:
            raise HTTPException(status_code=404, detail="Job not found") from None
        return _job_summary(job)

    @app.post("/api/jobs/{job_id}/archive")
    async def request_archive(job_id: str) -> dict[str, Any]:
        try:
            job = await archive_job(db, job_id, events=events)
        except JobNotFoundError:
            raise HTTPException(status_code=404, detail="Job not found") from None
        return _job_summary(job)

    @app.post("/api/jobs/{job_id}/outcome")
    async def request_outcome(job_id: str, body: OutcomeRequest) -> dict[str, Any]:
        try:
            outcome = await record_job_outcome(
                db,
                job_id,
                won=body.won,
                value=body.contract_value_usd,
                connects_spent=body.connects_spent,
                note=body.note,
                events=events,
            )
        except JobNotFoundError:
            raise HTTPException(status_code=404, detail="Job not found") from None
        return {
            "won": outcome.won,
            "contract_value_usd": outcome.contract_value_usd,
            "connects_spent": outcome.connects_spent,
            "note": outcome.note,
        }

    @app.get("/api/chat/{thread_id}/messages")
    async def get_chat_messages(thread_id: str) -> list[dict[str, Any]]:
        messages = await db.get_chat_messages(thread_id)
        return [
            {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
            for m in messages
        ]

    @app.delete("/api/chat/{thread_id}/messages")
    async def delete_chat_messages(thread_id: str) -> dict[str, str]:
        await db.clear_chat_thread(thread_id)
        return {"status": "cleared"}

    @app.websocket("/ws/chat/{thread_id}")
    async def ws_chat(websocket: WebSocket, thread_id: str) -> None:
        """Stream a chat reply token-by-token, persisting both sides of every turn.

        `thread_id` is either `_GENERAL_CHAT_THREAD_ID` (queue-wide copilot,
        same context `ulysses advise` builds) or a real job id (a focused
        conversation about that one job, grounded in its full detail). The
        system prompt is built once per connection, not per message -- it
        doesn't change mid-conversation.
        """
        await websocket.accept()

        if thread_id == _GENERAL_CHAT_THREAD_ID:
            system_prompt = "\n\n".join(
                [
                    CHAT_SYSTEM_PROMPT,
                    build_advisor_profile_summary(profile),
                    build_queue_digest(await db.list_jobs()),
                ]
            )
        else:
            full = await db.get_full_job(thread_id)
            if full is None:
                await websocket.send_json({"type": "error", "detail": "Job not found"})
                await websocket.close()
                return
            job, score = full
            system_prompt = "\n\n".join(
                [CHAT_SYSTEM_PROMPT, build_job_context_message(job, score, None, None)]
            )

        try:
            while True:
                payload = await websocket.receive_json()
                user_message = str(payload.get("message", "")).strip()
                if not user_message:
                    continue

                await db.add_chat_message(thread_id, "user", user_message)
                prior_messages = await db.get_chat_messages(thread_id)
                history = [{"role": m.role, "content": m.content} for m in prior_messages[:-1]][
                    -_MAX_CHAT_HISTORY_MESSAGES:
                ]

                reply_chunks: list[str] = []
                try:
                    async for chunk in chat_agent.stream(
                        history, user_message, system_prompt=system_prompt
                    ):
                        reply_chunks.append(chunk)
                        await websocket.send_json({"type": "delta", "text": chunk})
                except Exception:
                    logger.exception("Chat stream failed for thread_id={}", thread_id)
                    await websocket.send_json(
                        {"type": "error", "detail": "Something went wrong generating a reply."}
                    )
                    continue

                full_reply = "".join(reply_chunks).strip()
                if full_reply:
                    await db.add_chat_message(thread_id, "assistant", full_reply)
                await websocket.send_json({"type": "done"})
        except WebSocketDisconnect:
            pass

    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            async for event in events.subscribe():
                await websocket.send_json(event)
        except WebSocketDisconnect:
            pass

    if _FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
    else:

        @app.get("/", response_class=HTMLResponse)
        async def placeholder() -> str:
            return _PLACEHOLDER_HTML

    return app


def _job_summary(job: Job) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "id": job.id,
        "title": job.title,
        "url": job.url,
        "score": job.score,
        "category": job.category,
        "status": job.status.value,
        "source": "manual" if job.url.startswith("manual://") else "email",
        "posted_at": job.posted_at.isoformat(),
        "seen_at": job.seen_at.isoformat(),
    }
    if job.job_json is not None and job.score_json is not None:
        post = JobPost.model_validate_json(job.job_json)
        score = JobScore.model_validate_json(job.score_json)
        summary.update(
            budget=str(post.budget),
            skills_required=post.skills_required,
            client_hires=post.client_hires,
            client_rating=post.client_rating,
            payment_verified=post.payment_verified,
            proposals_count=post.proposals_count,
            red_flags=score.red_flags,
            recommendation=score.recommendation.value,
            best_repo_match=score.matched_repos[0].repo_name if score.matched_repos else None,
            freshness_score=score.freshness_score,
            proposal_score=score.proposal_score,
            client_score=score.client_score,
            skill_score=score.skill_score,
            budget_score=score.budget_score,
        )
    return summary
