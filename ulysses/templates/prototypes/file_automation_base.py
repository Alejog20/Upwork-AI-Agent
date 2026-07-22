"""Demo file-automation skeleton -- TODO markers show where job-specific logic goes.

Run with: python demo.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

# TODO: point these at the real folders this job needs to watch/process.
SOURCE_DIR = Path("./input")
DEST_DIR = Path("./processed")


def process_file(path: Path) -> None:
    """Process a single file.

    TODO: replace this with the real per-file logic for this job (rename,
    convert, validate, extract data, etc). This skeleton just moves the file.
    """
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    destination = DEST_DIR / path.name
    shutil.move(str(path), str(destination))
    print(f"Processed {path.name} -> {destination}")


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in SOURCE_DIR.glob("*") if p.is_file())
    if not files:
        print(f"No files found in {SOURCE_DIR}. Add some and re-run.")
        return
    for path in files:
        process_file(path)


if __name__ == "__main__":
    main()
