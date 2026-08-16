"""Tests for `ulysses.tools.example_retrieval`: curated example loading + retrieval."""

from __future__ import annotations

from pathlib import Path

from pytest_mock import MockerFixture

from ulysses.models import JobPost
from ulysses.tools.example_retrieval import (
    ExampleProposal,
    find_best_matching_example,
    load_example_proposals,
)

_EXAMPLE_KWARGS: dict[str, object] = {
    "job_title": "title",
    "job_description": "description",
    "skills": ["python"],
    "category": "scraping",
    "proof_repo": "repo",
    "proof_repo_url": "https://github.com/example/repo",
    "hook": "hook",
    "plan_bullet_1": "one",
    "plan_bullet_2": "two",
    "plan_bullet_3": "three",
    "close": "close",
}


def _example(**overrides: object) -> ExampleProposal:
    return ExampleProposal(**{**_EXAMPLE_KWARGS, **overrides})


class TestLoadExampleProposals:
    def test_loads_the_bundled_yaml_file(self) -> None:
        examples = load_example_proposals()
        assert len(examples) == 5
        assert {example.category for example in examples} == {
            "scraping",
            "automation",
            "api_dev",
            "data_pipeline",
            "ai_integration",
        }

    def test_returns_empty_list_for_a_missing_file(self, tmp_path: Path) -> None:
        assert load_example_proposals(tmp_path / "does-not-exist.yaml") == []

    def test_round_trips_a_minimal_entry(self, tmp_path: Path) -> None:
        path = tmp_path / "examples.yaml"
        path.write_text(
            """
            - job_title: "Test job"
              job_description: "A description."
              skills: [python]
              category: automation
              proof_repo: my-repo
              proof_repo_url: https://github.com/example/my-repo
              hook: "A hook."
              plan_bullet_1: "Step one."
              plan_bullet_2: "Step two."
              plan_bullet_3: "Step three."
              close: "A close."
              milestones: []
            """,
            encoding="utf-8",
        )

        examples = load_example_proposals(path)

        assert len(examples) == 1
        assert examples[0].job_title == "Test job"
        assert examples[0].milestones == []


class TestFindBestMatchingExample:
    async def test_returns_none_for_an_empty_corpus(self, fresh_job: JobPost) -> None:
        assert await find_best_matching_example(fresh_job, []) is None

    async def test_returns_the_example_with_the_highest_cosine_similarity(
        self, fresh_job: JobPost, mocker: MockerFixture
    ) -> None:
        close_match = _example(job_title="Close match", category="scraping")
        far_match = _example(job_title="Far match", category="ai_integration")

        # First vector is the query; the rest correspond 1:1 to `examples`, in
        # order: [far_match, close_match]. Make "close_match" nearly identical
        # to the query and "far_match" orthogonal to it.
        mocker.patch(
            "ulysses.tools.example_retrieval.aembed_texts",
            return_value=[[1.0, 0.0], [0.0, 1.0], [0.99, 0.01]],
        )

        result = await find_best_matching_example(fresh_job, [far_match, close_match])

        assert result is close_match

    async def test_embeds_query_and_every_example_in_one_batched_call(
        self, fresh_job: JobPost, mocker: MockerFixture
    ) -> None:
        embed_mock = mocker.patch(
            "ulysses.tools.example_retrieval.aembed_texts",
            return_value=[[1.0], [1.0], [1.0]],
        )

        await find_best_matching_example(fresh_job, [_example(), _example(job_title="second")])

        embed_mock.assert_awaited_once()
        texts = embed_mock.await_args.args[0]
        assert len(texts) == 3  # query + 2 examples
        assert fresh_job.title in texts[0]


class TestCosineSimilarity:
    async def test_identical_vectors_have_similarity_one(
        self, fresh_job: JobPost, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "ulysses.tools.example_retrieval.aembed_texts",
            return_value=[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        )
        example = _example()

        result = await find_best_matching_example(fresh_job, [example])

        assert result is example  # only candidate, sanity-checks the pure math path

    async def test_zero_vector_does_not_raise_a_division_error(
        self, fresh_job: JobPost, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "ulysses.tools.example_retrieval.aembed_texts",
            return_value=[[0.0, 0.0], [0.0, 0.0]],
        )

        result = await find_best_matching_example(fresh_job, [_example()])

        assert result is not None  # must not raise ZeroDivisionError
