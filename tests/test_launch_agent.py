"""Tests for `ulysses.tools.launch_agent`: the LaunchAgent plist install/uninstall."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from ulysses.tools.launch_agent import (
    LABEL,
    build_plist,
    install_launch_agent,
    plist_path,
    uninstall_launch_agent,
)


class TestBuildPlist:
    def test_contains_label_and_program_arguments(self, tmp_path: Path) -> None:
        xml = build_plist(tmp_path / "project", tmp_path / "logs", "/opt/homebrew/bin/uv")

        assert f"<string>{LABEL}</string>" in xml
        assert "<string>/opt/homebrew/bin/uv</string>" in xml
        assert "<string>run</string>" in xml
        assert "<string>--directory</string>" in xml
        assert str(tmp_path / "project") in xml
        assert "<string>ulysses</string>" in xml
        assert "<string>start</string>" in xml

    def test_configures_run_at_load_and_keep_alive(self, tmp_path: Path) -> None:
        xml = build_plist(tmp_path / "project", tmp_path / "logs", "/opt/homebrew/bin/uv")
        assert "<key>RunAtLoad</key>" in xml
        assert "<key>KeepAlive</key>" in xml

    def test_points_log_paths_at_the_given_log_dir(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        xml = build_plist(tmp_path / "project", log_dir, "/opt/homebrew/bin/uv")
        assert str(log_dir / "launchd.out.log") in xml
        assert str(log_dir / "launchd.err.log") in xml


class TestInstallLaunchAgent:
    def test_writes_plist_and_calls_launchctl_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        mocker.patch("shutil.which", return_value="/opt/homebrew/bin/uv")
        run_mock = mocker.patch("subprocess.run")

        path = install_launch_agent(tmp_path / "project", tmp_path / "logs")

        assert path == tmp_path / "Library" / "LaunchAgents" / f"{LABEL}.plist"
        assert path.exists()
        assert LABEL in path.read_text(encoding="utf-8")
        run_mock.assert_called_once_with(
            ["launchctl", "load", "-w", str(path)], check=True, capture_output=True
        )

    def test_raises_if_uv_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        mocker.patch("shutil.which", return_value=None)

        with pytest.raises(RuntimeError, match="uv"):
            install_launch_agent(tmp_path / "project", tmp_path / "logs")

    def test_propagates_launchctl_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        mocker.patch("shutil.which", return_value="/opt/homebrew/bin/uv")
        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["launchctl", "load"]),
        )

        with pytest.raises(subprocess.CalledProcessError):
            install_launch_agent(tmp_path / "project", tmp_path / "logs")


class TestUninstallLaunchAgent:
    def test_removes_existing_plist_and_calls_launchctl_unload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        path = plist_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<plist/>", encoding="utf-8")
        run_mock = mocker.patch("subprocess.run")

        removed = uninstall_launch_agent()

        assert removed is True
        assert not path.exists()
        run_mock.assert_called_once_with(
            ["launchctl", "unload", "-w", str(path)], check=False, capture_output=True
        )

    def test_returns_false_when_nothing_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        run_mock = mocker.patch("subprocess.run")

        removed = uninstall_launch_agent()

        assert removed is False
        run_mock.assert_not_called()
