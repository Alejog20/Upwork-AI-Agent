"""Tests for the dashboard's FastAPI app in `ulysses.dashboard.api`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from ulysses.agents.scorer import score_job
from ulysses.config.profile import Profile
from ulysses.dashboard.api import build_dashboard_app
from ulysses.models import GeneratedProposal, GeneratedPrototype, JobPost, JobScore
from ulysses.tools.db import Job, JobStatus, UlyssesDB
from ulysses.tools.events import DashboardEventBus


@pytest.fixture
async def db(tmp_path: Path) -> UlyssesDB:
    database = UlyssesDB(tmp_path / "dashboard-test.db")
    await database.init()
    yield database
    await database.dispose()


async def _seed_scored_job(
    db: UlyssesDB,
    job: JobPost,
    score: JobScore,
    *,
    status: JobStatus = JobStatus.NEW,
    seen_at: datetime | None = None,
) -> None:
    await db.upsert_job(
        Job(
            id=job.id,
            title=job.title,
            description=job.description,
            url=job.url,
            score=score.total_score,
            category=score.gig_category.value,
            status=status,
            posted_at=job.posted_at,
            seen_at=seen_at or datetime.now(UTC).replace(tzinfo=None),
            job_json=job.model_dump_json(),
            score_json=score.model_dump_json(),
        )
    )


def _copy_with_id(job: JobPost, job_id: str, **extra: object) -> JobPost:
    """Copy `job` with a new id and a matching unique url (`Job.url` has a unique index)."""
    return job.model_copy(
        update={"id": job_id, "url": f"https://www.upwork.com/jobs/~{job_id}", **extra}
    )


def _mock_proposal_agent() -> AsyncMock:
    agent = AsyncMock()
    agent.generate = AsyncMock(
        return_value=GeneratedProposal(
            job_id="job-1",
            category="tier1",
            hook="A hook",
            plan_bullets=["Step one"],
            close="A close",
            proof_repo="demo-repo",
            proof_repo_url="https://github.com/example/demo-repo",
            timeline="1 week",
            bid_usd=100.0,
            full_text="Generated proposal text.",
        )
    )
    return agent


def _mock_prototype_agent() -> AsyncMock:
    agent = AsyncMock()
    agent.generate = AsyncMock(
        return_value=GeneratedPrototype(
            job_id="job-1",
            category="tier1",
            demo_script="print('hi')",
            requirements_txt="requests==2.32.3\n",
            readme_md="# Demo README",
            config_example_env="# none needed\n",
            zip_filename="ulysses_demo_job-1.zip",
        )
    )
    return agent


def _mock_chat_agent(chunks: list[str] | None = None, *, raises: bool = False) -> MagicMock:
    agent = MagicMock()
    agent.stream_calls: list[tuple[list[dict[str, str]], str, str]] = []

    async def _stream(history: list[dict[str, str]], user_message: str, *, system_prompt: str):
        agent.stream_calls.append((history, user_message, system_prompt))
        if raises:
            raise RuntimeError("LLM boom")
        for chunk in chunks if chunks is not None else ["Mocked ", "reply."]:
            yield chunk

    agent.stream = _stream
    return agent


@pytest.fixture
def events() -> DashboardEventBus:
    return DashboardEventBus()


@pytest.fixture
def client(db: UlyssesDB, profile: Profile, events: DashboardEventBus) -> TestClient:
    app = build_dashboard_app(
        db,
        profile,
        _mock_proposal_agent(),
        _mock_prototype_agent(),
        events,
        _mock_chat_agent(),
    )
    return TestClient(app)


class TestListJobs:
    async def test_returns_enriched_summaries_for_scored_jobs(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        await _seed_scored_job(db, fresh_job, score)

        response = client.get("/api/jobs")

        assert response.status_code == 200
        [summary] = response.json()
        assert summary["id"] == fresh_job.id
        assert summary["url"] == fresh_job.url
        assert summary["skills_required"] == fresh_job.skills_required
        assert summary["recommendation"] == score.recommendation.value
        assert summary["source"] == "email"
        assert summary["freshness_score"] == score.freshness_score
        assert summary["proposal_score"] == score.proposal_score
        assert summary["client_score"] == score.client_score
        assert summary["skill_score"] == score.skill_score
        assert summary["budget_score"] == score.budget_score

    async def test_filters_by_min_score(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        await _seed_scored_job(db, fresh_job, score)

        response = client.get("/api/jobs", params={"min_score": 9999})

        assert response.status_code == 200
        assert response.json() == []

    async def test_returns_empty_list_when_no_jobs(self, client: TestClient) -> None:
        response = client.get("/api/jobs")

        assert response.status_code == 200
        assert response.json() == []

    async def test_filters_by_source(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        await _seed_scored_job(db, fresh_job, score)
        manual_job = fresh_job.model_copy(
            update={"id": "manual-1", "url": "manual://11111111-1111-1111-1111-111111111111"}
        )
        await _seed_scored_job(db, manual_job, score_job(manual_job, profile))

        email_only = client.get("/api/jobs", params={"source": "email"})
        manual_only = client.get("/api/jobs", params={"source": "manual"})

        assert [j["id"] for j in email_only.json()] == [fresh_job.id]
        assert [j["id"] for j in manual_only.json()] == ["manual-1"]

    async def test_filters_by_max_proposals(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        low_competition = _copy_with_id(fresh_job, "low-1", proposals_count=5)
        high_competition = _copy_with_id(fresh_job, "high-1", proposals_count=50)
        await _seed_scored_job(db, low_competition, score_job(low_competition, profile))
        await _seed_scored_job(db, high_competition, score_job(high_competition, profile))

        response = client.get("/api/jobs", params={"max_proposals": 5})

        assert [j["id"] for j in response.json()] == ["low-1"]

    async def test_filters_by_has_red_flags(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        clean_job = _copy_with_id(fresh_job, "clean-1")
        flagged_job = _copy_with_id(fresh_job, "flagged-1", client_hires=0)
        clean_score = score_job(clean_job, profile)
        flagged_score = score_job(flagged_job, profile).model_copy(
            update={"red_flags": ["No client payment history"]}
        )
        await _seed_scored_job(db, clean_job, clean_score)
        await _seed_scored_job(db, flagged_job, flagged_score)

        flagged_only = client.get("/api/jobs", params={"has_red_flags": True})
        clean_only = client.get("/api/jobs", params={"has_red_flags": False})

        assert [j["id"] for j in flagged_only.json()] == ["flagged-1"]
        assert [j["id"] for j in clean_only.json()] == ["clean-1"]


class TestListStaleJobs:
    async def test_includes_notified_drafted_and_built_jobs_older_than_the_threshold(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        old = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=10)
        for status, suffix in (
            (JobStatus.NOTIFIED, "notified"),
            (JobStatus.DRAFTED, "drafted"),
            (JobStatus.BUILT, "built"),
        ):
            job = _copy_with_id(fresh_job, f"stale-{suffix}")
            await _seed_scored_job(db, job, score_job(job, profile), status=status, seen_at=old)

        response = client.get("/api/jobs/stale")

        assert response.status_code == 200
        ids = {j["id"] for j in response.json()}
        assert ids == {"stale-notified", "stale-drafted", "stale-built"}

    async def test_excludes_jobs_seen_within_the_threshold(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        job = _copy_with_id(fresh_job, "fresh-notified")
        await _seed_scored_job(
            db,
            job,
            score_job(job, profile),
            status=JobStatus.NOTIFIED,
            seen_at=datetime.now(UTC).replace(tzinfo=None),
        )

        response = client.get("/api/jobs/stale")

        assert response.json() == []

    async def test_excludes_resolved_and_untouched_statuses(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        old = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=10)
        for status, suffix in (
            (JobStatus.NEW, "new"),
            (JobStatus.SKIPPED, "skipped"),
            (JobStatus.ARCHIVED, "archived"),
            (JobStatus.WON, "won"),
            (JobStatus.LOST, "lost"),
        ):
            job = _copy_with_id(fresh_job, f"old-{suffix}")
            await _seed_scored_job(db, job, score_job(job, profile), status=status, seen_at=old)

        response = client.get("/api/jobs/stale")

        assert response.json() == []

    async def test_respects_a_custom_days_threshold(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        two_days_old = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=2)
        job = _copy_with_id(fresh_job, "recent-notified")
        await _seed_scored_job(
            db, job, score_job(job, profile), status=JobStatus.NOTIFIED, seen_at=two_days_old
        )

        default_window = client.get("/api/jobs/stale")
        short_window = client.get("/api/jobs/stale", params={"days": 1})

        assert default_window.json() == []
        assert [j["id"] for j in short_window.json()] == ["recent-notified"]

    async def test_includes_days_since_seen(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        old = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=10)
        job = _copy_with_id(fresh_job, "stale-1")
        await _seed_scored_job(
            db, job, score_job(job, profile), status=JobStatus.NOTIFIED, seen_at=old
        )

        response = client.get("/api/jobs/stale")

        assert response.json()[0]["days_since_seen"] == 10


class TestJobDetail:
    async def test_returns_full_detail_for_a_known_job(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        await _seed_scored_job(db, fresh_job, score)

        response = client.get(f"/api/jobs/{fresh_job.id}")

        assert response.status_code == 200
        body = response.json()
        assert body["job"]["id"] == fresh_job.id
        assert body["score"]["total_score"] == score.total_score
        assert body["proposal_drafts"] == []
        assert body["prototype_files"] == []

    async def test_returns_404_for_unknown_job(self, client: TestClient) -> None:
        response = client.get("/api/jobs/unknown-id")

        assert response.status_code == 404


class TestStats:
    async def test_returns_db_stats(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        await _seed_scored_job(db, fresh_job, score)

        response = client.get("/api/stats")

        assert response.status_code == 200
        assert response.json()["total"] == 1


class TestAnalytics:
    async def test_returns_all_analytics_keys(self, client: TestClient) -> None:
        response = client.get("/api/analytics")

        assert response.status_code == 200
        body = response.json()
        assert set(body.keys()) == {
            "win_rate_by_category",
            "win_rate_by_score_bucket",
            "win_rate_by_red_flags",
            "average_score_won_vs_lost",
            "average_connects_spent_per_win",
            "scoring_weight_suggestions",
        }


class TestDraftRoute:
    async def test_drafts_for_a_known_job(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        await _seed_scored_job(db, fresh_job, score)

        response = client.post(f"/api/jobs/{fresh_job.id}/draft")

        assert response.status_code == 200
        assert response.json() == {"content": "Generated proposal text."}

    async def test_returns_404_for_unknown_job(self, client: TestClient) -> None:
        response = client.post("/api/jobs/unknown-id/draft")

        assert response.status_code == 404


class TestSaveDraftRoute:
    async def test_saves_an_edited_draft_for_a_known_job(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        await _seed_scored_job(db, fresh_job, score)

        response = client.put(
            f"/api/jobs/{fresh_job.id}/draft", json={"content": "A hand-edited draft."}
        )

        assert response.status_code == 200
        assert response.json() == {"content": "A hand-edited draft."}
        drafts = await db.get_proposal_drafts(fresh_job.id)
        assert drafts[-1].content == "A hand-edited draft."

    async def test_returns_404_for_unknown_job(self, client: TestClient) -> None:
        response = client.put("/api/jobs/unknown-id/draft", json={"content": "x"})

        assert response.status_code == 404


class TestBuildRoute:
    async def test_builds_for_a_known_job(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        await _seed_scored_job(db, fresh_job, score)

        response = client.post(f"/api/jobs/{fresh_job.id}/build")

        assert response.status_code == 200
        assert response.json()["zip_filename"] == "ulysses_demo_job-1.zip"

    async def test_returns_404_for_unknown_job(self, client: TestClient) -> None:
        response = client.post("/api/jobs/unknown-id/build")

        assert response.status_code == 404


class TestPrototypeZipDownload:
    async def test_downloads_zip_after_building(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        await _seed_scored_job(db, fresh_job, score)
        client.post(f"/api/jobs/{fresh_job.id}/build")

        response = client.get(f"/api/jobs/{fresh_job.id}/prototype.zip")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

    async def test_returns_404_when_nothing_built_yet(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        await _seed_scored_job(db, fresh_job, score)

        response = client.get(f"/api/jobs/{fresh_job.id}/prototype.zip")

        assert response.status_code == 404


class TestSkipAndArchiveRoutes:
    async def test_skip_marks_job_skipped(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        await _seed_scored_job(db, fresh_job, score)

        response = client.post(f"/api/jobs/{fresh_job.id}/skip")

        assert response.status_code == 200
        assert response.json()["status"] == JobStatus.SKIPPED.value

    async def test_archive_marks_job_archived(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        await _seed_scored_job(db, fresh_job, score)

        response = client.post(f"/api/jobs/{fresh_job.id}/archive")

        assert response.status_code == 200
        assert response.json()["status"] == JobStatus.ARCHIVED.value

    async def test_skip_returns_404_for_unknown_job(self, client: TestClient) -> None:
        response = client.post("/api/jobs/unknown-id/skip")

        assert response.status_code == 404

    async def test_archive_returns_404_for_unknown_job(self, client: TestClient) -> None:
        response = client.post("/api/jobs/unknown-id/archive")

        assert response.status_code == 404


class TestOutcomeRoute:
    async def test_records_a_won_outcome(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        await _seed_scored_job(db, fresh_job, score)

        response = client.post(
            f"/api/jobs/{fresh_job.id}/outcome",
            json={
                "won": True,
                "contract_value_usd": 500.0,
                "connects_spent": 6,
                "note": "great client",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["won"] is True
        assert body["contract_value_usd"] == 500.0
        assert body["connects_spent"] == 6

    async def test_returns_404_for_unknown_job(self, client: TestClient) -> None:
        response = client.post("/api/jobs/unknown-id/outcome", json={"won": True})

        assert response.status_code == 404

    async def test_connects_spent_surfaces_in_analytics(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        await _seed_scored_job(db, fresh_job, score)
        client.post(f"/api/jobs/{fresh_job.id}/outcome", json={"won": True, "connects_spent": 4})

        response = client.get("/api/analytics")

        assert response.json()["average_connects_spent_per_win"] == 4.0


class TestChatMessagesRoutes:
    async def test_get_messages_on_a_new_thread_returns_empty_list(
        self, client: TestClient
    ) -> None:
        response = client.get("/api/chat/__general__/messages")

        assert response.status_code == 200
        assert response.json() == []

    async def test_delete_clears_a_thread(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/chat/__general__") as websocket:
            websocket.send_json({"message": "hello"})
            while websocket.receive_json()["type"] != "done":
                pass

        before = client.get("/api/chat/__general__/messages").json()
        assert len(before) == 2

        response = client.delete("/api/chat/__general__/messages")

        assert response.status_code == 200
        assert client.get("/api/chat/__general__/messages").json() == []


class TestWebSocketChat:
    async def test_general_thread_streams_deltas_then_done(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/chat/__general__") as websocket:
            websocket.send_json({"message": "hi"})
            frames = []
            while True:
                frame = websocket.receive_json()
                frames.append(frame)
                if frame["type"] == "done":
                    break

        assert frames[0] == {"type": "delta", "text": "Mocked "}
        assert frames[1] == {"type": "delta", "text": "reply."}
        assert frames[2] == {"type": "done"}

    async def test_general_thread_persists_both_sides_of_the_turn(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/chat/__general__") as websocket:
            websocket.send_json({"message": "hi there"})
            while websocket.receive_json()["type"] != "done":
                pass

        messages = client.get("/api/chat/__general__/messages").json()
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[0]["content"] == "hi there"
        assert messages[1]["content"] == "Mocked reply."

    async def test_per_job_thread_builds_a_job_grounded_system_prompt(
        self, db: UlyssesDB, profile: Profile, events: DashboardEventBus, fresh_job: JobPost
    ) -> None:
        score = score_job(fresh_job, profile)
        await _seed_scored_job(db, fresh_job, score)
        chat_agent = _mock_chat_agent()
        app = build_dashboard_app(
            db, profile, _mock_proposal_agent(), _mock_prototype_agent(), events, chat_agent
        )
        client = TestClient(app)

        with client.websocket_connect(f"/ws/chat/{fresh_job.id}") as websocket:
            websocket.send_json({"message": "what's the budget?"})
            while websocket.receive_json()["type"] != "done":
                pass

        _, _, system_prompt = chat_agent.stream_calls[0]
        assert fresh_job.title in system_prompt
        assert str(fresh_job.budget) in system_prompt

    async def test_unknown_job_thread_sends_an_error_and_closes(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/chat/does-not-exist") as websocket:
            frame = websocket.receive_json()

        assert frame == {"type": "error", "detail": "Job not found"}

    async def test_stream_failure_sends_an_error_frame_without_persisting_a_reply(
        self, db: UlyssesDB, profile: Profile, events: DashboardEventBus
    ) -> None:
        chat_agent = _mock_chat_agent(raises=True)
        app = build_dashboard_app(
            db, profile, _mock_proposal_agent(), _mock_prototype_agent(), events, chat_agent
        )
        client = TestClient(app)

        with client.websocket_connect("/ws/chat/__general__") as websocket:
            websocket.send_json({"message": "hi"})
            frame = websocket.receive_json()

        assert frame["type"] == "error"
        messages = client.get("/api/chat/__general__/messages").json()
        assert [m["role"] for m in messages] == ["user"]


class TestWebSocketEvents:
    async def test_receives_a_job_updated_event_triggered_by_an_action(
        self, client: TestClient, db: UlyssesDB, fresh_job: JobPost, profile: Profile
    ) -> None:
        score = score_job(fresh_job, profile)
        await _seed_scored_job(db, fresh_job, score)

        with client.websocket_connect("/ws/events") as websocket:
            client.post(f"/api/jobs/{fresh_job.id}/skip")
            event = websocket.receive_json()

        assert event == {"type": "job_updated", "job_id": fresh_job.id}


class TestPlaceholderPage:
    async def test_serves_a_placeholder_when_frontend_is_not_built(
        self,
        db: UlyssesDB,
        profile: Profile,
        events: DashboardEventBus,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force the "frontend never built" branch regardless of whether this
        # checkout happens to have a real `dist/` on disk.
        monkeypatch.setattr("ulysses.dashboard.api._FRONTEND_DIST", tmp_path / "does-not-exist")
        app = build_dashboard_app(
            db, profile, _mock_proposal_agent(), _mock_prototype_agent(), events, _mock_chat_agent()
        )
        client = TestClient(app)

        response = client.get("/")

        assert response.status_code == 200
        assert "not built yet" in response.text

    async def test_serves_the_built_spa_when_dist_exists(
        self,
        db: UlyssesDB,
        profile: Profile,
        events: DashboardEventBus,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<html><body>the real SPA</body></html>")
        monkeypatch.setattr("ulysses.dashboard.api._FRONTEND_DIST", dist)
        app = build_dashboard_app(
            db, profile, _mock_proposal_agent(), _mock_prototype_agent(), events, _mock_chat_agent()
        )
        client = TestClient(app)

        response = client.get("/")

        assert response.status_code == 200
        assert "the real SPA" in response.text
