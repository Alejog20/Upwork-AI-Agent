"""Demo API integration skeleton -- TODO markers show where job-specific logic goes.

Run with: python demo.py
"""

from __future__ import annotations

import httpx

# TODO: point this at the real API endpoint for this job.
API_BASE_URL = "https://api.example.com"


def call_api(endpoint: str, **params: object) -> dict:
    """Call the target API and return its parsed JSON response.

    TODO: add whatever real authentication this API needs (bearer token,
    API key header, etc) -- this skeleton assumes no auth.
    """
    with httpx.Client(timeout=15) as client:
        response = client.get(f"{API_BASE_URL}{endpoint}", params=params)
        response.raise_for_status()
        return response.json()


def main() -> None:
    # TODO: replace with the real endpoint/params for this job's integration.
    data = call_api("/example")
    print(data)


if __name__ == "__main__":
    main()
