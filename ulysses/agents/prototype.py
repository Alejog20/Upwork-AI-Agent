"""Prototype Agent — generates a runnable demo script + README proving Alejandro can do the job.

Template category selection, requirements.txt, and the env template are all
deterministic (no LLM), so they stay fast, free, and fully unit-testable.
Only the demo script's TODO-filling and the README's contextual sentences go
through the LLM, via structured output.

The LLM is instructed to only use libraries already imported in the base
template — this keeps `requirements.txt` deterministic and correct instead of
trusting the model to invent a valid, installable dependency list.
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger
from pydantic import BaseModel, Field

from ulysses.config.profile import Profile
from ulysses.config.settings import get_settings
from ulysses.models import GeneratedPrototype, JobPost, JobScore
from ulysses.tools.llm import ainvoke_with_retry, get_llm

__all__ = [
    "PrototypeAgent",
    "build_prototype_zip",
    "classify_prototype_category",
]

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates" / "prototypes"

_PROTOTYPE_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "scraper": (
        "scrape",
        "scraper",
        "scraping",
        "crawl",
        "beautifulsoup",
        "playwright",
        "selenium",
        "web scraping",
    ),
    "email_bot": (
        "email",
        "imap",
        "smtp",
        "gmail",
        "inbox",
        "mailbox",
        "notification email",
    ),
    "api": (
        "api",
        "rest api",
        "endpoint",
        "webhook",
        "fastapi",
        "flask",
        "third-party api",
        "integration",
    ),
    "ai_pipeline": (
        "openai",
        "llm",
        "gpt",
        "langchain",
        "ai agent",
        "chatbot",
        "machine learning",
        "nlp",
    ),
}
_DEFAULT_PROTOTYPE_CATEGORY = "file_automation"

_CATEGORY_REQUIREMENTS: dict[str, list[str]] = {
    "scraper": ["requests==2.32.3", "beautifulsoup4==4.12.3"],
    "file_automation": [],
    "api": ["httpx==0.27.2"],
    "email_bot": [],
    "ai_pipeline": ["openai==1.51.0"],
}

_CATEGORY_ENV_TEMPLATES: dict[str, str] = {
    "scraper": "# No environment variables needed for this demo.\n",
    "file_automation": "# No environment variables needed for this demo.\n",
    "api": "# API_KEY=your-api-key-here\n# API_BASE_URL=https://api.example.com\n",
    "email_bot": "EMAIL_USER=you@gmail.com\nEMAIL_APP_PASSWORD=your-app-specific-password\n",
    "ai_pipeline": "OPENAI_API_KEY=sk-your-key-here\n",
}

_MAX_OUTPUT_TOKENS = 1500
_DESCRIPTION_INPUT_CHAR_LIMIT = 500

_SYSTEM_PROMPT = """You write demo Python scripts for Alejandro Garcia, a Python freelancer, to \
prove he can do a specific Upwork job before being hired.

You will be given a base template with TODO markers. Rules:
- Return the COMPLETE script with every TODO replaced by real, working logic specific to the \
job. Do not leave any TODO comments in the output.
- Only use imports already present in the template. Do not add new third-party imports -- the \
demo must run with exactly the pinned requirements already listed, nothing else.
- Solve the CORE ask only: the single most important or impressive part of what the client \
needs, not the entire project.
- The script must be syntactically valid Python 3.14, fully runnable via \
`pip install -r requirements.txt && python demo.py`, with zero further modification.
- Include brief inline comments explaining non-obvious choices, but don't over-comment.
- Keep it a demo, not a production system: no unnecessary abstractions, classes, or config files.
- Never include real credentials, API keys, or secrets -- use `os.environ.get(...)` placeholders \
exactly as the template does.

You also write a short, concrete README summary: one sentence on what the demo shows, and three \
bullets on what a full implementation would add (the third should be a production-hardening \
concern like error handling, retries, logging, or monitoring).
"""

_USER_PROMPT_TEMPLATE = """Job title: {title}
Job description: {description}
Base template ({category}):
```python
{template}
```
"""


class _PrototypeLLMOutput(BaseModel):
    """Structured output the LLM must produce: the script and README content."""

    demo_script: str = Field(
        description="The complete demo.py script with every TODO replaced by real logic."
    )
    what_it_demonstrates: str = Field(description="One sentence describing what the demo shows.")
    extension_bullet_1: str = Field(description="First thing a full implementation would add.")
    extension_bullet_2: str = Field(description="Second thing a full implementation would add.")
    extension_bullet_3: str = Field(
        description="Third thing a full implementation would add -- a production-hardening concern."
    )


def classify_prototype_category(job: JobPost) -> str:
    """Classify a job into one of the prototype template categories.

    Deterministic keyword matching over the job title, description, and
    required skills. Falls back to "file_automation" when nothing matches
    clearly, mirroring `agents.proposal.classify_category`'s pattern but with
    the prototype library's own category names.
    """
    haystack = " ".join([job.title, job.description, *job.skills_required]).lower()
    best_category = _DEFAULT_PROTOTYPE_CATEGORY
    best_hits = 0
    for category, keywords in _PROTOTYPE_CATEGORY_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in haystack)
        if hits > best_hits:
            best_hits = hits
            best_category = category
    return best_category


class PrototypeAgent:
    """Generates a runnable demo script + README for a scored job."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        """Create a Prototype Agent.

        Args:
            llm: Chat model to use. Defaults to the shared client from `get_llm()`.
        """
        self._llm = llm or get_llm()

    async def generate(self, job: JobPost, score: JobScore, profile: Profile) -> GeneratedPrototype:
        """Generate a complete demo prototype for a scored job."""
        category = classify_prototype_category(job)
        template_text = _load_template(category)

        structured_llm = self._llm.bind(max_tokens=_MAX_OUTPUT_TOKENS).with_structured_output(
            _PrototypeLLMOutput
        )
        prompt = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_PROMPT_TEMPLATE.format(
                    title=job.title,
                    description=_truncate(job.description, _DESCRIPTION_INPUT_CHAR_LIMIT),
                    category=category,
                    template=template_text,
                ),
            },
        ]

        start = time.monotonic()
        llm_output: _PrototypeLLMOutput = await ainvoke_with_retry(structured_llm, prompt)
        elapsed = time.monotonic() - start
        logger.bind(job_id=job.id, agent="prototype").info(
            "LLM call complete: model={} latency={:.2f}s", get_settings().llm_model, elapsed
        )

        demo_script = _validate_script(llm_output.demo_script.strip(), job.id)
        requirements_txt = _build_requirements_txt(category)
        readme_md = _build_readme(job, profile, llm_output)
        config_example_env = _CATEGORY_ENV_TEMPLATES[category]

        return GeneratedPrototype(
            job_id=job.id,
            category=category,
            demo_script=demo_script,
            requirements_txt=requirements_txt,
            readme_md=readme_md,
            config_example_env=config_example_env,
            zip_filename=f"ulysses_demo_{job.id}.zip",
        )


def _load_template(category: str) -> str:
    return (_TEMPLATES_DIR / f"{category}_base.py").read_text(encoding="utf-8")


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    return (truncated[:last_space] if last_space > 0 else truncated).rstrip()


def _validate_script(script: str, job_id: str) -> str:
    """Check the generated script compiles; log (don't block) if it doesn't.

    `compile()` only parses the source -- it never executes the generated
    code, so this is safe to run on untrusted LLM output.
    """
    try:
        compile(script, "demo.py", "exec")
    except SyntaxError as exc:
        logger.bind(job_id=job_id, agent="prototype").warning(
            "Generated demo script has a syntax error: {}", exc
        )
    return script


def _build_requirements_txt(category: str) -> str:
    packages = _CATEGORY_REQUIREMENTS[category]
    if not packages:
        return "# No third-party dependencies needed for this demo.\n"
    return "\n".join(packages) + "\n"


def _build_readme(job: JobPost, profile: Profile, llm_output: _PrototypeLLMOutput) -> str:
    return (
        f"# Demo: {job.title} — Built by {profile.freelancer.name}\n\n"
        f"## What this demonstrates\n{llm_output.what_it_demonstrates.strip()}\n\n"
        f"## How to run\n```bash\npip install -r requirements.txt\npython demo.py\n```\n\n"
        f"## What a full implementation would add\n"
        f"- {llm_output.extension_bullet_1.strip()}\n"
        f"- {llm_output.extension_bullet_2.strip()}\n"
        f"- {llm_output.extension_bullet_3.strip()}\n\n"
        f"## About me\n{profile.freelancer.title}.\n"
        f"See more of my work: {profile.freelancer.github}\n"
    )


def build_prototype_zip(prototype: GeneratedPrototype) -> bytes:
    """Package a generated prototype into an in-memory zip file."""
    buffer = io.BytesIO()
    folder = f"ulysses_demo_{prototype.job_id}"
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{folder}/demo.py", prototype.demo_script)
        zf.writestr(f"{folder}/requirements.txt", prototype.requirements_txt)
        zf.writestr(f"{folder}/README.md", prototype.readme_md)
        zf.writestr(f"{folder}/config.example.env", prototype.config_example_env)
    return buffer.getvalue()
