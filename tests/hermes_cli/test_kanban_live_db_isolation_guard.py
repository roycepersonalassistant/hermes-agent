"""Behavioral regression tests for Kanban live-board test isolation.

Forensic background (Aug 2026): an orphan/PID safety review probe changed only
``HERMES_HOME`` but inherited ``HERMES_KANBAN_DB`` and workspace pins from its
Kanban worker.  The direct DB pin has higher precedence, so two synthetic
``zombie-owner`` tasks landed on the live portfolio board even though the probe
created a temporary Hermes home.  These tests exercise that exact import-then-
redirect shape and require the DB choke point to fail before any production
path is created or opened.
"""

from __future__ import annotations

import os
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db


class TestProductionKanbanPathRefused:
    def test_explicit_default_board_path_raises_before_creation(
        self, tmp_path, monkeypatch
    ):
        operator_home = tmp_path / "operator-home"
        production_root = operator_home / ".hermes"
        production_db = production_root / "kanban.db"
        monkeypatch.setenv("HOME", str(operator_home))
        monkeypatch.setattr(
            kanban_db, "_KANBAN_DB_GUARD_EXTRA_DENY_ROOTS", (production_root,)
        )

        with pytest.raises(RuntimeError, match="live-system guard"):
            kanban_db.connect(db_path=production_db)

        assert not production_root.exists()

    def test_explicit_named_board_path_raises_before_creation(
        self, tmp_path, monkeypatch
    ):
        operator_home = tmp_path / "operator-home"
        production_root = operator_home / ".hermes"
        production_db = (
            production_root
            / "kanban"
            / "boards"
            / "portfolio-operating-system"
            / "kanban.db"
        )
        monkeypatch.setenv("HOME", str(operator_home))
        monkeypatch.setattr(
            kanban_db, "_KANBAN_DB_GUARD_EXTRA_DENY_ROOTS", (production_root,)
        )

        with pytest.raises(RuntimeError, match="live-system guard"):
            kanban_db.connect(db_path=production_db)

        assert not production_root.exists()

    def test_runtime_home_redirect_does_not_override_inherited_live_db_pin(
        self, tmp_path, monkeypatch
    ):
        """Reproduce the review-probe leak without touching a real board."""
        operator_home = tmp_path / "operator-home"
        production_root = operator_home / ".hermes"
        production_db = (
            production_root
            / "kanban"
            / "boards"
            / "portfolio-operating-system"
            / "kanban.db"
        )
        monkeypatch.setenv("HOME", str(operator_home))
        monkeypatch.setattr(
            kanban_db, "_KANBAN_DB_GUARD_EXTRA_DENY_ROOTS", (production_root,)
        )
        monkeypatch.setenv("HERMES_KANBAN_DB", str(production_db))
        monkeypatch.setenv(
            "HERMES_KANBAN_WORKSPACES_ROOT",
            str(production_db.parent / "workspaces"),
        )
        # This is what the probe did after importing kanban_db.  The explicit
        # live DB pin still wins, so the guard must reject it.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "probe-home"))

        with pytest.raises(RuntimeError, match="live-system guard"):
            kanban_db.connect()

        assert not production_root.exists()


    def test_arbitrary_inherited_db_pin_is_an_exact_denied_target(
        self, tmp_path, monkeypatch
    ):
        inherited_db = tmp_path / "portable-live-board" / "custom.sqlite3"
        monkeypatch.setenv("_HERMES_TEST_KANBAN_DENY_DBS", str(inherited_db))

        with pytest.raises(RuntimeError, match="live-system guard"):
            kanban_db._ensure_test_isolation(inherited_db)

        assert not inherited_db.parent.exists()

    def test_lexical_production_path_is_blocked_when_symlink_resolves_outward(
        self, tmp_path, monkeypatch
    ):
        production_root = tmp_path / "operator-home" / ".hermes"
        outside = tmp_path / "outside"
        (outside / "kanban" / "boards" / "portfolio").mkdir(parents=True)
        production_root.mkdir(parents=True)
        (production_root / "kanban").symlink_to(
            outside / "kanban", target_is_directory=True
        )
        candidate = production_root / "kanban" / "boards" / "portfolio" / "kanban.db"
        monkeypatch.setattr(
            kanban_db, "_KANBAN_DB_GUARD_EXTRA_DENY_ROOTS", (production_root,)
        )

        with pytest.raises(RuntimeError, match="live-system guard"):
            kanban_db._ensure_test_isolation(candidate)

        assert not candidate.exists()

    def test_internal_pytest_bridge_survives_scrubbed_pytest_variables(
        self, tmp_path, monkeypatch
    ):
        production_root = tmp_path / "immutable-production"
        production_db = production_root / "kanban.db"
        monkeypatch.setattr(
            kanban_db, "_KANBAN_DB_GUARD_EXTRA_DENY_ROOTS", (production_root,)
        )
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("PYTEST_VERSION", raising=False)
        monkeypatch.setenv("_HERMES_TEST_KANBAN_GUARD_ACTIVE", "1")

        with pytest.raises(RuntimeError, match="live-system guard"):
            kanban_db._ensure_test_isolation(production_db)

    def test_mutating_home_to_temp_does_not_create_a_false_production_root(
        self, tmp_path, monkeypatch
    ):
        immutable_root = tmp_path / "actual-production"
        monkeypatch.setattr(
            kanban_db, "_KANBAN_DB_GUARD_EXTRA_DENY_ROOTS", (immutable_root,)
        )
        temp_home = tmp_path / "test-home"
        monkeypatch.setenv("HOME", str(temp_home))
        temp_db = temp_home / ".hermes" / "kanban.db"

        kanban_db._ensure_test_isolation(temp_db)
        assert not temp_db.exists()

    def test_lexical_root_stays_denied_after_parent_symlink_is_repointed(
        self, tmp_path, monkeypatch
    ):
        first = tmp_path / "first-target"
        second = tmp_path / "second-target"
        first.mkdir()
        second.mkdir()
        production_link = tmp_path / "production-link"
        production_link.symlink_to(first, target_is_directory=True)
        payload = json.dumps(
            [
                {
                    "lexical": str(production_link),
                    "resolved": str(production_link.resolve()),
                }
            ]
        )
        monkeypatch.setenv("_HERMES_TEST_KANBAN_DENY_ROOTS", payload)
        monkeypatch.setattr(kanban_db, "_KANBAN_DB_GUARD_EXTRA_DENY_ROOTS", ())

        production_link.unlink()
        production_link.symlink_to(second, target_is_directory=True)
        candidate = production_link / "arbitrary" / "cache.sqlite3"

        with pytest.raises(RuntimeError, match="live-system guard"):
            kanban_db._ensure_test_isolation(candidate)

    def test_non_board_database_filename_under_denied_root_is_blocked(
        self, tmp_path, monkeypatch
    ):
        production_root = tmp_path / "production"
        candidate = production_root / "unrelated" / "state.sqlite3"
        monkeypatch.setattr(
            kanban_db, "_KANBAN_DB_GUARD_EXTRA_DENY_ROOTS", (production_root,)
        )

        with pytest.raises(RuntimeError, match="live-system guard"):
            kanban_db._ensure_test_isolation(candidate)

    @pytest.mark.skipif(os.pathsep == ";", reason="POSIX path-separator case")
    def test_json_encoded_exact_pin_preserves_path_separator_characters(
        self, tmp_path, monkeypatch
    ):
        inherited_db = tmp_path / "pin:with:separators" / "custom.sqlite3"
        payload = json.dumps(
            [
                {
                    "lexical": str(inherited_db),
                    "resolved": str(inherited_db.resolve(strict=False)),
                }
            ]
        )
        monkeypatch.setenv("_HERMES_TEST_KANBAN_DENY_DBS", payload)

        with pytest.raises(RuntimeError, match="live-system guard"):
            kanban_db._ensure_test_isolation(inherited_db)


class TestHermeticKanbanPathAllowed:
    def test_tmp_db_can_create_synthetic_orphan_probe_task(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "probe-home" / "kanban.db"
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "probe-home"))
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "probe-home"))
        monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
        monkeypatch.setenv(
            "HERMES_KANBAN_WORKSPACES_ROOT", str(tmp_path / "probe-workspaces")
        )

        with kanban_db.connect() as conn:
            task_id = kanban_db.create_task(
                conn,
                title="zombie-owner",
                workspace_kind="worktree",
                workspace_path=str(tmp_path / "probe-workspaces" / "zombie-owner"),
                branch_name="probe/zombie-owner",
            )
            row = conn.execute(
                "SELECT title, workspace_path FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()

        assert row is not None
        assert row["title"] == "zombie-owner"
        assert Path(row["workspace_path"]).is_relative_to(tmp_path)
        assert db_path.exists()


class TestBypassMarker:
    @pytest.mark.kanban_live_db_guard_bypass
    def test_dedicated_bypass_marker_disables_only_kanban_guard(
        self, tmp_path, monkeypatch
    ):
        operator_home = tmp_path / "operator-home"
        production_db = operator_home / ".hermes" / "kanban.db"
        monkeypatch.setenv("HOME", str(operator_home))

        # Drive the guard directly; never open the production-shaped DB.
        kanban_db._ensure_test_isolation(production_db)
        assert not production_db.exists()

    @pytest.mark.live_system_guard_bypass
    def test_broad_live_system_bypass_does_not_disable_kanban_guard(
        self, tmp_path, monkeypatch
    ):
        production_root = tmp_path / "operator-home" / ".hermes"
        production_db = production_root / "kanban.db"
        monkeypatch.setattr(
            kanban_db, "_KANBAN_DB_GUARD_EXTRA_DENY_ROOTS", (production_root,)
        )

        with pytest.raises(RuntimeError, match="live-system guard"):
            kanban_db._ensure_test_isolation(production_db)
        assert not production_db.exists()


class TestSubprocessOrphanProbeCovered:
    @staticmethod
    def _base_child_env() -> dict[str, str]:
        return {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "HERMES_HOME",
                "HERMES_KANBAN_HOME",
                "HERMES_KANBAN_DB",
                "HERMES_KANBAN_BOARD",
                "HERMES_KANBAN_WORKSPACES_ROOT",
                "HERMES_KANBAN_TASK",
                "HERMES_KANBAN_WORKSPACE",
                "HERMES_KANBAN_BRANCH",
                "PYTEST_PLUGINS",
                "PYTHONPATH",
            }
        }

    def test_child_with_inherited_live_pin_is_refused_and_writes_nothing(
        self, tmp_path
    ):
        """A pytest child cannot leak a zombie-owner task through a stale pin."""
        operator_home = tmp_path / "operator-home"
        production_root = operator_home / ".hermes"
        production_db = (
            production_root
            / "kanban"
            / "boards"
            / "portfolio-operating-system"
            / "kanban.db"
        )
        probe_home = tmp_path / "probe-home"
        env = self._base_child_env()
        env.update(
            {
                "HOME": str(operator_home),
                "HERMES_HOME": str(probe_home),
                "HERMES_KANBAN_DB": str(production_db),
                "HERMES_KANBAN_BOARD": "portfolio-operating-system",
                "HERMES_KANBAN_WORKSPACES_ROOT": str(
                    production_db.parent / "workspaces"
                ),
                "PYTEST_CURRENT_TEST": (
                    "tests/fake.py::test_orphan_pid_pgid_safety (call)"
                ),
                "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            }
        )
        code = (
            "from hermes_cli import kanban_db as kb\n"
            "with kb.connect() as conn:\n"
            "    kb.create_task(conn, title='zombie-owner')\n"
        )

        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )

        assert proc.returncode != 0
        assert "live-system guard" in proc.stderr
        assert not production_root.exists()
        assert not probe_home.exists()

    def test_child_with_all_kanban_paths_redirected_writes_only_tmp_db(
        self, tmp_path
    ):
        operator_home = tmp_path / "operator-home"
        production_root = operator_home / ".hermes"
        probe_home = tmp_path / "probe-home"
        probe_db = probe_home / "kanban.db"
        probe_workspaces = probe_home / "kanban" / "workspaces"
        env = self._base_child_env()
        env.update(
            {
                "HOME": str(operator_home),
                "HERMES_HOME": str(probe_home),
                "HERMES_KANBAN_HOME": str(probe_home),
                "HERMES_KANBAN_DB": str(probe_db),
                "HERMES_KANBAN_BOARD": "default",
                "HERMES_KANBAN_WORKSPACES_ROOT": str(probe_workspaces),
                "PYTEST_CURRENT_TEST": (
                    "tests/fake.py::test_orphan_pid_pgid_safety (call)"
                ),
                "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            }
        )
        code = (
            "from hermes_cli import kanban_db as kb\n"
            "with kb.connect() as conn:\n"
            "    task_id = kb.create_task(\n"
            "        conn, title='zombie-owner', workspace_kind='worktree',\n"
            "        workspace_path=str(kb.workspaces_root() / 'zombie-owner'),\n"
            "        branch_name='probe/zombie-owner')\n"
            "    print(task_id)\n"
        )

        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )

        assert proc.returncode == 0, proc.stderr
        assert probe_db.exists()
        assert not production_root.exists()
        with sqlite3.connect(probe_db) as conn:
            rows = conn.execute("SELECT title FROM tasks").fetchall()
        assert rows == [("zombie-owner",)]
