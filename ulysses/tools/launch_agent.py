"""Generates and manages the macOS LaunchAgent that auto-starts `ulysses start` on login."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from loguru import logger

__all__ = [
    "LABEL",
    "build_plist",
    "install_launch_agent",
    "plist_path",
    "uninstall_launch_agent",
]

LABEL = "com.ulysses.agent"


def plist_path() -> Path:
    """Path where the LaunchAgent plist is (or would be) installed."""
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def build_plist(project_dir: Path, log_dir: Path, uv_path: str) -> str:
    """Build the LaunchAgent plist XML content.

    Args:
        project_dir: Directory to run `uv run ulysses start` from.
        log_dir: Directory for launchd's own stdout/stderr capture (separate
            from Ulysses' rotating log file, which is configured independently).
        uv_path: Absolute path to the `uv` executable.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{uv_path}</string>
        <string>run</string>
        <string>--directory</string>
        <string>{project_dir}</string>
        <string>ulysses</string>
        <string>start</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir / "launchd.out.log"}</string>
    <key>StandardErrorPath</key>
    <string>{log_dir / "launchd.err.log"}</string>
</dict>
</plist>
"""


def install_launch_agent(project_dir: Path, log_dir: Path) -> Path:
    """Write the plist and load it via `launchctl`.

    Args:
        project_dir: Directory to run `ulysses start` from (passed to `uv run --directory`).
        log_dir: Directory for launchd's stdout/stderr log files.

    Returns:
        The path the plist was written to.

    Raises:
        RuntimeError: If `uv` isn't found on `PATH`.
        subprocess.CalledProcessError: If `launchctl load` fails.
    """
    uv_path = shutil.which("uv")
    if uv_path is None:
        raise RuntimeError("Couldn't find `uv` on PATH — install it before running this.")

    log_dir.mkdir(parents=True, exist_ok=True)
    path = plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_plist(project_dir, log_dir, uv_path), encoding="utf-8")

    subprocess.run(["launchctl", "load", "-w", str(path)], check=True, capture_output=True)
    logger.info("Installed and loaded LaunchAgent at {}", path)
    return path


def uninstall_launch_agent() -> bool:
    """Unload and remove the LaunchAgent plist, if present.

    Returns:
        Whether a plist was actually found and removed.
    """
    path = plist_path()
    if not path.exists():
        return False

    subprocess.run(["launchctl", "unload", "-w", str(path)], check=False, capture_output=True)
    path.unlink()
    logger.info("Unloaded and removed LaunchAgent at {}", path)
    return True
