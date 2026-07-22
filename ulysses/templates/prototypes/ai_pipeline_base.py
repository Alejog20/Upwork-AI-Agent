"""Demo AI pipeline skeleton -- TODO markers show where job-specific logic goes.

Run with: python demo.py
"""

from __future__ import annotations

import os

from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

# TODO: tune this system prompt for the job's real task.
SYSTEM_PROMPT = "You are a helpful assistant."


def run_pipeline(user_input: str) -> str:
    """Run a single LLM call and return the response text.

    TODO: replace with the real pipeline for this job (retrieval, multi-step
    chaining, structured output, etc). This skeleton is a single plain call.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
    )
    return response.choices[0].message.content or ""


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY (see config.example.env) and re-run.")
        return
    # TODO: replace this sample input with the job's real use case.
    result = run_pipeline("Say hello in one sentence.")
    print(result)


if __name__ == "__main__":
    main()
