"""E2E coverage for the canonical Kanban worktree lifecycle registry."""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_workspaces as kw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(cwd),
            "-c",
            "user.name=Kanban Test",
            "-c",
            "user.email=kanban@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    return repo


def test_resolved_worktree_is_registered_before_creation_and_reused(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "owned"

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="implement",
            assignee="developer",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/owned",
        )
        task = kb.get_task(conn, task_id)
        first = kb.resolve_workspace(task)
        kb.set_workspace_path(conn, task_id, first)
        second = kb.resolve_workspace(kb.get_task(conn, task_id))

    assert first == second == target
    records = kw.list_workspace_records(task_id=task_id)
    assert len(records) == 1
    record = records[0]
    assert record.workspace_id.startswith("w_")
    assert record.repo_path == str(repo.resolve())
    assert record.workspace_path == str(target.resolve())
    assert record.task_id == task_id
    assert record.board_id == "default"
    assert record.purpose == "implementation"
    assert record.branch == "feat/owned"
    assert record.head_sha == _git(target, "rev-parse", "HEAD")
    assert record.status == "active"
    assert record.cleanup_policy == "on_task_terminal"
    assert record.retention_condition == "task_terminal"
    assert record.estimated_bytes >= 0
    assert record.last_verified_at is not None


def test_preexisting_matching_worktree_is_adopted_only_for_its_task(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "preexisting"
    _git(repo, "worktree", "add", "-b", "feat/preexisting", str(target), "HEAD")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="claim preexisting",
            assignee="developer",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/preexisting",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert kb.resolve_workspace(task) == target.resolve()

    records = kw.list_workspace_records(task_id=task_id)
    assert len(records) == 1
    assert records[0].workspace_path == str(target.resolve())
    assert records[0].head_sha == _git(target, "rev-parse", "HEAD")
    assert records[0].status == "active"


def test_preexisting_matching_external_linked_worktree_is_reused_via_public_resolver(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    target = tmp_path / "external-linked-checkout"
    _git(repo, "worktree", "add", "-b", "feat/external", str(target), "HEAD")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="reuse external linked checkout",
            assignee="developer",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/external",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert kb.resolve_workspace(task) == target.resolve()

    record = kw.list_workspace_records(task_id=task_id)[0]
    assert record.repo_path == str(repo.resolve())
    assert record.workspace_path == str(target.resolve())
    assert record.status == "active"


def test_existing_target_on_wrong_branch_fails_closed_without_mutating_git(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="wrong branch collision",
            assignee="developer",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name="feat/wanted",
        )
        target = repo / ".worktrees" / task_id
        _git(repo, "worktree", "add", "-b", "feat/occupied", str(target), "HEAD")
        before = _git(repo, "worktree", "list", "--porcelain")
        task = kb.get_task(conn, task_id)
        assert task is not None
        with pytest.raises(RuntimeError, match="branch mismatch"):
            kb.resolve_workspace(task)
        after = _git(repo, "worktree", "list", "--porcelain")

    assert after == before
    records = kw.list_workspace_records(task_id=task_id)
    assert len(records) == 1
    assert records[0].status == "creation_failed"
    assert _git(target, "branch", "--show-current") == "feat/occupied"
    report = kw.reconcile_inventory(
        repo_paths=[repo], approved_roots=[repo / ".worktrees"]
    )
    assert report["unowned_paths"] == [str(target.resolve())]


def test_drifted_existing_registration_stays_owned_when_branch_check_fails(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "drifted"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="detect branch drift",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/expected",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)
        _git(target, "switch", "-c", "feat/drifted")

        current_task = kb.get_task(conn, task_id)
        assert current_task is not None
        with pytest.raises(RuntimeError, match="branch mismatch"):
            kb.resolve_workspace(current_task)

    record = kw.list_workspace_records(task_id=task_id)[0]
    assert record.status == "active"
    assert record.workspace_path == str(target.resolve())


def test_registered_branch_drift_reaches_materialization_policy_seam(
    kanban_home, tmp_path, monkeypatch
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "drifted-policy-seam"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="allow branch policy integration",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/expected",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)
        _git(target, "switch", "-c", "feat/drifted")

        calls: list[tuple[Path, Path, str]] = []

        def realign_at_materialization_seam(
            repo_root: Path, requested: Path, branch_name: str
        ) -> None:
            calls.append((repo_root, requested, branch_name))
            _git(requested, "switch", branch_name)

        monkeypatch.setattr(kb, "_ensure_git_worktree", realign_at_materialization_seam)
        current_task = kb.get_task(conn, task_id)
        assert current_task is not None
        assert kb.resolve_workspace(current_task) == target.resolve()

    assert calls == [(repo.resolve(), target.resolve(), "feat/expected")]
    assert _git(target, "branch", "--show-current") == "feat/expected"
    assert kw.list_workspace_records(task_id=task_id)[0].status == "active"


def test_duplicate_task_repo_requires_recorded_replacement_reason(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    first_path = repo / ".worktrees" / "first"
    replacement_path = repo / ".worktrees" / "replacement"

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="replace worktree",
            assignee="developer",
            workspace_kind="worktree",
            workspace_path=str(first_path),
            branch_name="feat/first",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)

        conn.execute(
            "UPDATE tasks SET workspace_path=?, branch_name=? WHERE id=?",
            (str(replacement_path), "feat/replacement", task_id),
        )
        conn.commit()
        with pytest.raises(RuntimeError, match="replacement reason"):
            kb.resolve_workspace(kb.get_task(conn, task_id))

        conn.execute(
            "UPDATE tasks SET workspace_replacement_reason=? WHERE id=?",
            ("first checkout was created from the wrong anchor", task_id),
        )
        conn.commit()
        replaced = kb.resolve_workspace(kb.get_task(conn, task_id))

    assert replaced == replacement_path
    records = kw.list_workspace_records(task_id=task_id)
    assert len(records) == 2
    assert records[1].replacement_reason == "first checkout was created from the wrong anchor"


def test_complete_and_archive_require_explicit_workspace_disposition(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    dirty_path = repo / ".worktrees" / "dirty"
    clean_path = repo / ".worktrees" / "clean"

    with kb.connect() as conn:
        dirty_task_id = kb.create_task(
            conn,
            title="dirty terminal",
            workspace_kind="worktree",
            workspace_path=str(dirty_path),
            branch_name="feat/dirty",
        )
        dirty_task = kb.get_task(conn, dirty_task_id)
        assert dirty_task is not None
        kb.resolve_workspace(dirty_task)
        recovery_artifact = tmp_path / "dirty-recovery.patch"
        recovery_artifact.write_text("durable recovery evidence\n", encoding="utf-8")
        (dirty_path / "evidence.txt").write_text("keep\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="workspace_disposition"):
            kb.complete_task(conn, dirty_task_id, metadata={"tests_run": 1})
        assert kb.get_task(conn, dirty_task_id).status == "ready"

        with pytest.raises(RuntimeError, match="dirty worktree.*retired"):
            kb.complete_task(
                conn,
                dirty_task_id,
                metadata={"workspace_disposition": "retired"},
            )

        with pytest.raises(RuntimeError, match="requires preserved_dirty"):
            kb.complete_task(
                conn,
                dirty_task_id,
                metadata={
                    "workspace_disposition": "retained_until:review-required"
                },
            )

        assert kb.complete_task(
            conn,
            dirty_task_id,
            metadata={
                "workspace_disposition": (
                    f"preserved_dirty:artifact={recovery_artifact};owner=reviewer"
                )
            },
        )
        dirty_record = kw.list_workspace_records(task_id=dirty_task_id)[0]
        assert dirty_record.disposition == (
            f"preserved_dirty:artifact={recovery_artifact};owner=reviewer"
        )
        assert dirty_record.status == "terminal_dirty"
        assert kb.archive_task(conn, dirty_task_id)
        assert kb.get_task(conn, dirty_task_id).status == "archived"

        clean_task_id = kb.create_task(
            conn,
            title="archive clean terminal",
            workspace_kind="worktree",
            workspace_path=str(clean_path),
            branch_name="feat/clean",
        )
        clean_task = kb.get_task(conn, clean_task_id)
        assert clean_task is not None
        kb.resolve_workspace(clean_task)
        with pytest.raises(RuntimeError, match="workspace_disposition"):
            kb.archive_task(conn, clean_task_id)
        assert kb.archive_task(
            conn,
            clean_task_id,
            workspace_disposition="retained_until:janitor",
        )
        clean_record = kw.list_workspace_records(task_id=clean_task_id)[0]
        assert clean_record.retention_condition == "janitor"
        assert clean_record.status == "retirement_queued"


@pytest.mark.parametrize("failed_fact", ["status", "head", "branch"])
def test_terminal_disposition_fails_closed_when_git_fact_is_unverifiable(
    kanban_home, tmp_path, monkeypatch, failed_fact
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / f"unverifiable-{failed_fact}"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title=f"unverifiable {failed_fact}",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name=f"feat/unverifiable-{failed_fact}",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)

        real_git = kw._git

        def fail_selected_fact(path, *args):
            if failed_fact == "status" and args and args[0] == "status":
                return None
            if failed_fact == "head" and args == ("rev-parse", "HEAD"):
                return None
            if failed_fact == "branch" and args in {
                ("branch", "--show-current"),
                ("rev-parse", "--abbrev-ref", "HEAD"),
            }:
                return None
            return real_git(path, *args)

        monkeypatch.setattr(kw, "_git", fail_selected_fact)
        with pytest.raises(RuntimeError, match=rf"verify Git {failed_fact}"):
            kb.complete_task(
                conn,
                task_id,
                metadata={"workspace_disposition": "retained_until:review"},
            )
        assert kb.get_task(conn, task_id).status == "ready"


def test_retired_disposition_rejects_stale_git_worktree_registration(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "stale-registration"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="stale registration",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/stale-registration",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)

        # Simulate an unsafe filesystem-only removal. Git still owns an
        # authoritative worktree registration for the now-absent path.
        shutil.rmtree(target)
        assert str(target.resolve()) in _git(repo, "worktree", "list", "--porcelain")

        with pytest.raises(RuntimeError, match="still registered by Git"):
            kb.complete_task(
                conn,
                task_id,
                metadata={"workspace_disposition": "retired"},
            )
        assert kb.get_task(conn, task_id).status == "ready"


def test_retired_disposition_rejects_dangling_workspace_symlink(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "dangling-workspace"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="dangling workspace",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/dangling-workspace",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)
        _git(repo, "worktree", "remove", "--force", str(target))
        target.symlink_to(tmp_path / "missing-target", target_is_directory=True)

        with pytest.raises(
            RuntimeError, match="could not verify Git status|still exists"
        ):
            kb.complete_task(
                conn,
                task_id,
                metadata={"workspace_disposition": "retired"},
            )
        assert kb.get_task(conn, task_id).status == "ready"


@pytest.mark.parametrize("artifact_kind", ["inside-worktree", "directory", "symlink"])
def test_preserved_dirty_requires_independently_durable_regular_artifact(
    kanban_home, tmp_path, artifact_kind
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / f"artifact-{artifact_kind}"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title=f"artifact {artifact_kind}",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name=f"feat/artifact-{artifact_kind}",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)
        (target / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")

        if artifact_kind == "inside-worktree":
            artifact = target / "recovery.patch"
            artifact.write_text("patch\n", encoding="utf-8")
        elif artifact_kind == "directory":
            artifact = tmp_path / "recovery-directory"
            artifact.mkdir()
        else:
            durable = tmp_path / "durable.patch"
            durable.write_text("patch\n", encoding="utf-8")
            artifact = tmp_path / "recovery-link.patch"
            artifact.symlink_to(durable)

        with pytest.raises(RuntimeError, match="durable regular file outside"):
            kb.complete_task(
                conn,
                task_id,
                metadata={
                    "workspace_disposition": (
                        f"preserved_dirty:artifact={artifact};owner=reviewer"
                    )
                },
            )
        assert kb.get_task(conn, task_id).status == "ready"


def test_terminal_disposition_recovers_after_process_dies_between_commits(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "crash-recovery"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="crash during terminalization",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/crash-recovery",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)

    source_root = Path(__file__).parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(source_root), env.get("PYTHONPATH", "")])
    )
    script = f"""
import os
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_workspaces as kw

def die_after_task_commit(operation_id):
    os._exit(91)

kw.finalize_terminal_disposition_intent = die_after_task_commit
with kb.connect() as conn:
    kb.complete_task(
        conn,
        {task_id!r},
        metadata={{"workspace_disposition": "retained_until:crash-review"}},
    )
raise SystemExit(92)
"""
    interrupted = subprocess.run(
        [sys.executable, "-c", script],
        cwd=source_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert interrupted.returncode == 91, interrupted.stderr

    with kb.connect() as conn:
        recovered_task = kb.get_task(conn, task_id)
        assert recovered_task is not None
        assert recovered_task.status == "done"
    with sqlite3.connect(kw.registry_path()) as registry:
        workspace = registry.execute(
            "SELECT status, disposition FROM workspaces WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        pending = registry.execute(
            "SELECT COUNT(*) FROM terminal_disposition_intents WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0]
    assert workspace == ("terminalizing", None)
    assert pending == 1

    report = kw.recover_terminal_disposition_intents(
        board_id="default", task_id=task_id
    )
    assert len(report["finalized"]) == 1
    assert report["aborted"] == []
    assert report["pending"] == []
    record = kw.list_workspace_records(task_id=task_id)[0]
    assert record.status == "terminal_clean"
    assert record.disposition == "retained_until:crash-review"
    with sqlite3.connect(kw.registry_path()) as registry:
        assert registry.execute(
            "SELECT COUNT(*) FROM terminal_disposition_intents WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0] == 0


def test_terminal_disposition_aborts_after_process_dies_before_task_commit(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "crash-before-task-commit"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="crash before task commit",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/crash-before-task-commit",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)

    source_root = Path(__file__).parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(source_root), env.get("PYTHONPATH", "")])
    )
    script = f"""
import os
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_workspaces as kw

original_stage = kw.stage_terminal_disposition_intent

def die_after_intent_commit(*args, **kwargs):
    operation_id = original_stage(*args, **kwargs)
    assert operation_id is not None
    os._exit(93)

kw.stage_terminal_disposition_intent = die_after_intent_commit
with kb.connect() as conn:
    kb.complete_task(
        conn,
        {task_id!r},
        metadata={{"workspace_disposition": "retained_until:crash-review"}},
    )
raise SystemExit(94)
"""
    interrupted = subprocess.run(
        [sys.executable, "-c", script],
        cwd=source_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert interrupted.returncode == 93, interrupted.stderr

    with kb.connect() as conn:
        interrupted_task = conn.execute(
            "SELECT status, terminal_disposition_operation_id FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert tuple(interrupted_task) == ("ready", None)
    with sqlite3.connect(kw.registry_path()) as registry:
        workspace = registry.execute(
            "SELECT status, disposition FROM workspaces WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        pending = registry.execute(
            "SELECT COUNT(*) FROM terminal_disposition_intents WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0]
    assert workspace == ("terminalizing", None)
    assert pending == 1

    report = kw.recover_terminal_disposition_intents(
        board_id="default", task_id=task_id
    )
    assert report["finalized"] == []
    assert len(report["aborted"]) == 1
    assert report["pending"] == []
    record = kw.list_workspace_records(task_id=task_id)[0]
    assert record.status == "active"
    assert record.disposition is None
    with sqlite3.connect(kw.registry_path()) as registry:
        assert registry.execute(
            "SELECT COUNT(*) FROM terminal_disposition_intents WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0] == 0


def test_live_intent_is_not_aborted_when_lease_timestamp_expires(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "live-terminal-intent"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="live terminal intent",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/live-terminal-intent",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)

    plans = kw.prepare_terminal_disposition(
        task_id=task_id,
        board_id="default",
        disposition="retained_until:review",
    )
    operation_id = kw.stage_terminal_disposition_intent(
        plans,
        terminal_status="done",
        task_db_path=kb.kanban_db_path(),
    )
    assert operation_id is not None
    with sqlite3.connect(kw.registry_path()) as registry:
        registry.execute(
            "UPDATE terminal_disposition_intents SET expires_at = 1 "
            "WHERE operation_id = ?",
            (operation_id,),
        )

    report = kw.recover_terminal_disposition_intents(
        board_id="default", task_id=task_id
    )
    assert report["pending"] == [operation_id]
    assert kw.list_workspace_records(task_id=task_id)[0].status == "terminalizing"
    assert kw.abort_terminal_disposition_intent(operation_id) == 1


def test_recovery_rejects_reused_owner_pid(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "reused-owner-pid"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="reused owner pid",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/reused-owner-pid",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)

    plans = kw.prepare_terminal_disposition(
        task_id=task_id,
        board_id="default",
        disposition="retained_until:review",
    )
    operation_id = kw.stage_terminal_disposition_intent(
        plans,
        terminal_status="done",
        task_db_path=kb.kanban_db_path(),
    )
    assert operation_id is not None
    with sqlite3.connect(kw.registry_path()) as registry:
        registry.execute(
            "UPDATE terminal_disposition_intents SET owner_started_at = 0 "
            "WHERE operation_id = ?",
            (operation_id,),
        )

    report = kw.recover_terminal_disposition_intents(
        board_id="default", task_id=task_id
    )
    assert report["finalized"] == []
    assert report["aborted"] == [operation_id]
    assert report["pending"] == []
    record = kw.list_workspace_records(task_id=task_id)[0]
    assert record.status == "active"
    assert record.disposition is None


def test_inventory_cas_does_not_overwrite_new_terminalizing_marker(
    kanban_home, tmp_path, monkeypatch
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "inventory-terminal-race"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="inventory terminal race",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/inventory-terminal-race",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)

    plans = kw.prepare_terminal_disposition(
        task_id=task_id,
        board_id="default",
        disposition="retained_until:review",
    )
    original_connect_registry = kw.connect_registry
    calls = 0
    staged_operation_id: str | None = None

    def racing_connect_registry():
        nonlocal calls, staged_operation_id
        calls += 1
        if calls == 3:
            staged_operation_id = kw.stage_terminal_disposition_intent(
                plans,
                terminal_status="done",
                task_db_path=kb.kanban_db_path(),
            )
        return original_connect_registry()

    monkeypatch.setattr(kw, "connect_registry", racing_connect_registry)
    kw.reconcile_inventory(
        repo_paths=[repo],
        approved_roots=[repo / ".worktrees"],
    )

    assert staged_operation_id is not None
    record = kw.list_workspace_records(task_id=task_id)[0]
    assert record.status == "terminalizing"
    with sqlite3.connect(kw.registry_path()) as registry:
        assert registry.execute(
            "SELECT COUNT(*) FROM terminal_disposition_intents "
            "WHERE operation_id = ?",
            (staged_operation_id,),
        ).fetchone()[0] == 1
    assert kw.abort_terminal_disposition_intent(staged_operation_id) == 1


def test_recovery_aborts_intent_not_correlated_to_task_commit(
    kanban_home, tmp_path, monkeypatch
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "mismatched-terminal-intent"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="mismatched terminal intent",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/mismatched-terminal-intent",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)
        assert kb.complete_task(
            conn,
            task_id,
            metadata={"workspace_disposition": "retained_until:done-review"},
        )

    plans = kw.prepare_terminal_disposition(
        task_id=task_id,
        board_id="default",
        disposition="retained_until:archive-review",
    )
    operation_id = kw.stage_terminal_disposition_intent(
        plans,
        terminal_status="archived",
        task_db_path=kb.kanban_db_path(),
    )
    assert operation_id is not None
    monkeypatch.setattr(kw, "_intent_owner_is_alive", lambda _row: False)

    report = kw.recover_terminal_disposition_intents(
        board_id="default", task_id=task_id
    )
    assert report["finalized"] == []
    assert report["aborted"] == [operation_id]
    assert report["pending"] == []
    record = kw.list_workspace_records(task_id=task_id)[0]
    assert record.status == "terminal_clean"
    assert record.disposition == "retained_until:done-review"


def test_concurrent_terminal_recovery_is_idempotent(
    kanban_home, tmp_path, monkeypatch
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "concurrent-recovery"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="concurrent recovery",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/concurrent-recovery",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)

    plans = kw.prepare_terminal_disposition(
        task_id=task_id,
        board_id="default",
        disposition="retained_until:review",
    )
    operation_id = kw.stage_terminal_disposition_intent(
        plans,
        terminal_status="done",
        task_db_path=kb.kanban_db_path(),
    )
    assert operation_id is not None
    with kb.connect() as conn, kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'done', "
            "terminal_disposition_operation_id = ? WHERE id = ?",
            (operation_id, task_id),
        )

    original_connect_registry = kw.connect_registry
    calls = 0

    def racing_connect_registry():
        nonlocal calls
        calls += 1
        if calls == 2:
            monkeypatch.setattr(kw, "connect_registry", original_connect_registry)
            try:
                assert kw.finalize_terminal_disposition_intent(operation_id) == "finalized"
            finally:
                monkeypatch.setattr(kw, "connect_registry", racing_connect_registry)
        return original_connect_registry()

    monkeypatch.setattr(kw, "connect_registry", racing_connect_registry)
    assert kw.finalize_terminal_disposition_intent(operation_id) == "absent"
    record = kw.list_workspace_records(task_id=task_id)[0]
    assert record.status == "terminal_clean"
    assert record.disposition == "retained_until:review"


def test_terminal_disposition_intent_is_aborted_when_task_transaction_rolls_back(
    kanban_home, tmp_path, monkeypatch
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "exception-rollback"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="rollback terminalization",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/exception-rollback",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)

        def fail_closing_run(*args, **kwargs):
            raise RuntimeError("injected task transaction failure")

        monkeypatch.setattr(kb, "_end_run", fail_closing_run)
        with pytest.raises(RuntimeError, match="injected task transaction failure"):
            kb.complete_task(
                conn,
                task_id,
                metadata={"workspace_disposition": "retained_until:review"},
            )
        rolled_back_task = kb.get_task(conn, task_id)
        assert rolled_back_task is not None
        assert rolled_back_task.status == "ready"

    record = kw.list_workspace_records(task_id=task_id)[0]
    assert record.status == "active"
    assert record.disposition is None
    with sqlite3.connect(kw.registry_path()) as registry:
        assert registry.execute(
            "SELECT COUNT(*) FROM terminal_disposition_intents WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0] == 0


def test_terminal_disposition_rolls_back_base_exception(
    kanban_home, tmp_path, monkeypatch
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "base-exception-rollback"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="base exception rollback",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/base-exception-rollback",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)

        def interrupt_closing_run(*args, **kwargs):
            raise KeyboardInterrupt("injected task transaction interrupt")

        monkeypatch.setattr(kb, "_end_run", interrupt_closing_run)
        with pytest.raises(KeyboardInterrupt, match="injected task transaction interrupt"):
            kb.complete_task(
                conn,
                task_id,
                metadata={"workspace_disposition": "retained_until:review"},
            )
        assert not conn.in_transaction
        rolled_back_task = kb.get_task(conn, task_id)
        assert rolled_back_task is not None
        assert rolled_back_task.status == "ready"

    record = kw.list_workspace_records(task_id=task_id)[0]
    assert record.status == "active"
    assert record.disposition is None
    with sqlite3.connect(kw.registry_path()) as registry:
        assert registry.execute(
            "SELECT COUNT(*) FROM terminal_disposition_intents WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0] == 0


def test_terminal_disposition_uses_the_connection_board_not_ambient_board(
    kanban_home, tmp_path, monkeypatch
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "beta-task"
    kb.create_board("beta")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")

    with kb.connect(board="beta") as conn:
        task_id = kb.create_task(
            conn,
            title="beta worktree",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/beta-task",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task, board="beta")
        assert kb.complete_task(
            conn,
            task_id,
            metadata={"workspace_disposition": "retained_until:beta-review"},
        )

    record = kw.list_workspace_records(task_id=task_id)[0]
    assert record.board_id == "beta"
    assert record.disposition == "retained_until:beta-review"
    assert record.status == "terminal_clean"


def test_terminal_recovery_uses_persisted_db_path_not_ambient_override(
    kanban_home, tmp_path, monkeypatch
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "beta-recovery"
    kb.create_board("beta")
    beta_db_path = kb.board_dir("beta") / "kanban.db"
    original_finalize = kw.finalize_terminal_disposition_intent

    def leave_committed_intent(_operation_id, **_kwargs):
        raise RuntimeError("injected post-commit recovery interruption")

    monkeypatch.setattr(
        kw, "finalize_terminal_disposition_intent", leave_committed_intent
    )
    with kb.connect(board="beta") as conn:
        task_id = kb.create_task(
            conn,
            title="beta recovery",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/beta-recovery",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task, board="beta")
        with pytest.raises(
            RuntimeError, match="injected post-commit recovery interruption"
        ):
            kb.complete_task(
                conn,
                task_id,
                metadata={"workspace_disposition": "retained_until:beta-review"},
            )
        committed_task = conn.execute(
            "SELECT status, terminal_disposition_operation_id FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert committed_task[0] == "done"
        assert committed_task[1] is not None

    monkeypatch.setattr(
        kw, "finalize_terminal_disposition_intent", original_finalize
    )
    with sqlite3.connect(kw.registry_path()) as registry:
        intent_db_path = registry.execute(
            "SELECT task_db_path FROM terminal_disposition_intents WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0]
    assert Path(intent_db_path) == beta_db_path.resolve()

    monkeypatch.setenv("HERMES_KANBAN_DB", str(kanban_home / "kanban.db"))
    report = kw.recover_terminal_disposition_intents(
        board_id="beta", task_id=task_id
    )
    assert len(report["finalized"]) == 1
    assert report["aborted"] == []
    assert report["pending"] == []
    record = kw.list_workspace_records(board_id="beta", task_id=task_id)[0]
    assert record.status == "terminal_clean"
    assert record.disposition == "retained_until:beta-review"
    with kb.connect(db_path=beta_db_path, board="beta") as conn:
        recovered_task = kb.get_task(conn, task_id)
        assert recovered_task is not None
        assert recovered_task.status == "done"


def test_delete_archived_task_recovers_committed_terminal_intent(
    kanban_home, tmp_path, monkeypatch
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "delete-archived-recovery"
    original_finalize = kw.finalize_terminal_disposition_intent

    def leave_committed_intent(_operation_id, **_kwargs):
        raise RuntimeError("injected post-archive recovery interruption")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="delete archived recovery",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/delete-archived-recovery",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)
        monkeypatch.setattr(
            kw, "finalize_terminal_disposition_intent", leave_committed_intent
        )
        with pytest.raises(
            RuntimeError, match="injected post-archive recovery interruption"
        ):
            kb.archive_task(
                conn,
                task_id,
                workspace_disposition="retained_until:review",
            )
        archived_task = kb.get_task(conn, task_id)
        assert archived_task is not None
        assert archived_task.status == "archived"

        monkeypatch.setattr(
            kw, "finalize_terminal_disposition_intent", original_finalize
        )
        assert not kb.delete_archived_task(conn, task_id)
        assert kb.get_task(conn, task_id) is not None

    record = kw.list_workspace_records(task_id=task_id)[0]
    assert record.status == "terminal_clean"
    assert record.disposition == "retained_until:review"
    assert not kw.has_terminal_disposition_intents(
        board_id="default", task_id=task_id
    )


def test_delete_task_refuses_live_intent_then_recovers_dead_owner(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "delete-live-intent"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="delete live intent",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/delete-live-intent",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)

    plans = kw.prepare_terminal_disposition(
        task_id=task_id,
        board_id="default",
        disposition="retained_until:review",
    )
    operation_id = kw.stage_terminal_disposition_intent(
        plans,
        terminal_status="done",
        task_db_path=kb.kanban_db_path(),
    )
    assert operation_id is not None
    with kb.connect() as conn:
        assert not kb.delete_task(conn, task_id)
        assert kb.get_task(conn, task_id) is not None
    assert kw.list_workspace_records(task_id=task_id)[0].status == "terminalizing"

    with sqlite3.connect(kw.registry_path()) as registry:
        registry.execute(
            "UPDATE terminal_disposition_intents SET owner_started_at = 0 "
            "WHERE operation_id = ?",
            (operation_id,),
        )
    with kb.connect() as conn:
        assert not kb.delete_task(conn, task_id)
        assert kb.get_task(conn, task_id) is not None
    record = kw.list_workspace_records(task_id=task_id)[0]
    assert record.status == "active"
    assert record.disposition is None
    assert not kw.has_terminal_disposition_intents(
        board_id="default", task_id=task_id
    )


@pytest.mark.parametrize(
    "lifecycle_status", ["active", "retained", "dirty", "retirement_queued"]
)
@pytest.mark.parametrize("delete_api", ["delete_task", "delete_archived_task"])
def test_hard_delete_preserves_task_evidence_required_by_workspace_registry(
    kanban_home, tmp_path, lifecycle_status, delete_api
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / f"delete-{delete_api}-{lifecycle_status}"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title=f"delete {delete_api} {lifecycle_status}",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name=f"feat/delete-{delete_api}-{lifecycle_status}",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)

        if lifecycle_status == "retained":
            assert kb.complete_task(
                conn,
                task_id,
                metadata={"workspace_disposition": "retained_until:review"},
            )
        elif lifecycle_status == "dirty":
            (target / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")
            artifact = tmp_path / f"{task_id}-recovery.patch"
            artifact.write_text("durable patch\n", encoding="utf-8")
            assert kb.complete_task(
                conn,
                task_id,
                metadata={
                    "workspace_disposition": (
                        f"preserved_dirty:artifact={artifact};owner=reviewer"
                    )
                },
            )
        elif lifecycle_status == "retirement_queued":
            assert kb.complete_task(
                conn,
                task_id,
                metadata={"workspace_disposition": "retained_until:janitor"},
            )

        if delete_api == "delete_archived_task":
            conn.execute("UPDATE tasks SET status = 'archived' WHERE id = ?", (task_id,))
            conn.commit()

        assert not getattr(kb, delete_api)(conn, task_id)
        assert kb.get_task(conn, task_id) is not None

    records = kw.list_workspace_records(task_id=task_id)
    assert len(records) == 1
    expected = {
        "active": "active",
        "retained": "terminal_clean",
        "dirty": "terminal_dirty",
        "retirement_queued": "retirement_queued",
    }[lifecycle_status]
    assert records[0].status == expected


@pytest.mark.parametrize("terminal_api", ["complete_task", "archive_task"])
def test_terminal_transition_and_workspace_registration_are_serialized(
    kanban_home, tmp_path, monkeypatch, terminal_api
):
    repo = _make_repo(tmp_path)
    workspace = repo / ".worktrees" / f"race-{terminal_api}"
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title=f"race {terminal_api}")
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
        conn.commit()

    registry_locked = threading.Event()
    release_registry = threading.Event()
    original_connect_registry = kw.connect_registry

    class PausingRegistry:
        def __init__(self, inner):
            self._inner = inner
            self._paused = False

        def execute(self, sql, parameters=()):
            if "INSERT INTO workspaces" in sql and not self._paused:
                self._paused = True
                registry_locked.set()
                assert release_registry.wait(timeout=10)
            return self._inner.execute(sql, parameters)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def pausing_connect_registry():
        return PausingRegistry(original_connect_registry())

    monkeypatch.setattr(kw, "connect_registry", pausing_connect_registry)

    def register_workspace():
        return kw.reserve_workspace(
            repo_path=repo,
            workspace_path=workspace,
            task_id=task_id,
            board_id="default",
            owner_profile="developer",
            task_db_path=kb.kanban_db_path(),
        )

    def transition_task():
        with kb.connect() as conn:
            if terminal_api == "complete_task":
                return kb.complete_task(conn, task_id)
            return kb.archive_task(conn, task_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        registration = pool.submit(register_workspace)
        assert registry_locked.wait(timeout=10)
        transition = pool.submit(transition_task)
        release_registry.set()
        assert registration.result(timeout=10).task_id == task_id
        with pytest.raises(RuntimeError, match="gained workspace registrations"):
            transition.result(timeout=10)

    with kb.connect() as conn:
        assert kb.get_task(conn, task_id).status == "ready"
    assert len(kw.list_workspace_records(task_id=task_id)) == 1


@pytest.mark.parametrize("delete_api", ["delete_task", "delete_archived_task"])
def test_deletion_fence_blocks_concurrent_workspace_registration(
    kanban_home, tmp_path, monkeypatch, delete_api
):
    repo = _make_repo(tmp_path)
    workspace = repo / ".worktrees" / f"race-{delete_api}"
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title=f"race {delete_api}")
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
        conn.commit()
        if delete_api == "delete_archived_task":
            assert kb.archive_task(conn, task_id)

    fence_installed = threading.Event()
    release_deletion = threading.Event()
    original_insert_fence = kw._insert_task_lifecycle_fence

    def pausing_insert_fence(registry, **kwargs):
        original_insert_fence(registry, **kwargs)
        if kwargs["fence_kind"] == "deletion":
            fence_installed.set()
            assert release_deletion.wait(timeout=10)

    monkeypatch.setattr(kw, "_insert_task_lifecycle_fence", pausing_insert_fence)

    def delete_task():
        with kb.connect() as conn:
            return getattr(kb, delete_api)(conn, task_id)

    def register_workspace():
        try:
            return kw.reserve_workspace(
                repo_path=repo,
                workspace_path=workspace,
                task_id=task_id,
                board_id="default",
                owner_profile="developer",
                task_db_path=kb.kanban_db_path(),
            )
        except RuntimeError as exc:
            return exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        deletion = pool.submit(delete_task)
        assert fence_installed.wait(timeout=10)
        registration = pool.submit(register_workspace)
        release_deletion.set()
        assert deletion.result(timeout=10)
        error = registration.result(timeout=10)
        assert isinstance(error, RuntimeError)
        assert "fenced for deletion" in str(error)

    with kb.connect() as conn:
        assert kb.get_task(conn, task_id) is None
    assert kw.list_workspace_records(task_id=task_id) == []


def test_archive_cli_records_direct_disposition_and_reuses_completed_one(
    kanban_home, tmp_path
):
    from hermes_cli import kanban as kc

    repo = _make_repo(tmp_path)
    direct_path = repo / ".worktrees" / "direct-archive"
    completed_path = repo / ".worktrees" / "completed-archive"
    with kb.connect() as conn:
        direct_id = kb.create_task(
            conn,
            title="direct archive",
            workspace_kind="worktree",
            workspace_path=str(direct_path),
            branch_name="feat/direct-archive",
        )
        kb.resolve_workspace(kb.get_task(conn, direct_id))
        completed_id = kb.create_task(
            conn,
            title="completed archive",
            workspace_kind="worktree",
            workspace_path=str(completed_path),
            branch_name="feat/completed-archive",
        )
        kb.resolve_workspace(kb.get_task(conn, completed_id))
        assert kb.complete_task(
            conn,
            completed_id,
            metadata={"workspace_disposition": "retained_until:review-closeout"},
        )

    assert kc.run_slash(
        f"archive --workspace-disposition retained_until:review-closeout {direct_id}"
    ) == "Archived " + direct_id
    assert kc.run_slash(f"archive {completed_id}") == "Archived " + completed_id
    with kb.connect() as conn:
        assert kb.get_task(conn, direct_id).status == "archived"
        assert kb.get_task(conn, completed_id).status == "archived"


def test_inventory_reconciles_git_counts_and_reports_deterministic_findings(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    approved_root = repo / ".worktrees"
    owned = approved_root / "owned"
    duplicate = approved_root / "duplicate"
    outside = tmp_path / "outside-worktree"

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="terminal implementation",
            assignee="developer",
            workspace_kind="worktree",
            workspace_path=str(owned),
            branch_name="feat/owned",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)

        _git(repo, "worktree", "add", "-b", "feat/duplicate", str(duplicate), "HEAD")
        replacement = kw.reserve_workspace(
            repo_path=repo,
            workspace_path=duplicate,
            task_id=task_id,
            board_id="default",
            owner_profile="developer",
            branch="feat/duplicate",
            replacement_reason="clean_replacement_after_dirty_collision",
        )
        kw.verify_workspace(
            replacement.workspace_id,
            duplicate,
            reservation_token=replacement.reservation_token,
        )

        _git(repo, "worktree", "add", "--detach", str(outside), "HEAD")
        (owned / "dirty.txt").write_text("preserve me\n", encoding="utf-8")
        conn.execute(
            "UPDATE tasks SET status='done', completed_at=1 WHERE id = ?",
            (task_id,),
        )
        conn.commit()

    first = kw.reconcile_inventory(
        repo_paths=[repo], approved_roots=[approved_root]
    )
    second = kw.reconcile_inventory(
        repo_paths=[repo], approved_roots=[approved_root]
    )

    assert first == second
    assert first["totals"]["git_worktrees"] == 4
    assert first["per_repo"] == [
        {
            "repo_id": first["per_repo"][0]["repo_id"],
            "repo_path": str(repo.resolve()),
            "worktree_count": 4,
            "estimated_bytes": first["per_repo"][0]["estimated_bytes"],
        }
    ]
    assert first["unowned_paths"] == [str(outside.resolve())]
    assert first["duplicate_task_repo"] == [
        {
            "board_id": "default",
            "task_id": task_id,
            "repo_id": first["per_repo"][0]["repo_id"],
            "workspace_ids": sorted(
                record.workspace_id
                for record in kw.list_workspace_records(task_id=task_id)
            ),
            "paths": sorted([str(owned.resolve()), str(duplicate.resolve())]),
        }
    ]
    assert len(first["duplicate_heads"]) == 1
    assert first["duplicate_heads"][0]["paths"] == sorted(
        [str(repo.resolve()), str(owned.resolve()), str(duplicate.resolve()), str(outside.resolve())]
    )
    assert first["terminal_without_disposition"] == sorted(
        [str(owned.resolve()), str(duplicate.resolve())]
    )
    assert first["dirty_terminal"] == [str(owned.resolve())]
    assert first["outside_approved_root"] == [str(outside.resolve())]

    from hermes_cli import kanban as kc

    cli_payload = json.loads(
        kc.run_slash(
            f"worktrees inventory --repo {repo} "
            f"--approved-root {approved_root} --json"
        )
    )
    assert cli_payload == first

    terminal_report = kc.run_slash(
        f"worktrees inventory --repo {repo} --approved-root {approved_root}"
    )
    assert f"{repo.resolve()}: 4 worktree(s)" in terminal_report
    assert "unowned_paths (1)" in terminal_report
    assert str(outside.resolve()) in terminal_report
    assert "duplicate_task_repo (1)" in terminal_report
    assert "terminal_without_disposition (2)" in terminal_report
    assert "dirty_terminal (1)" in terminal_report
    assert "outside_approved_root (1)" in terminal_report


def test_janitor_is_dry_run_only_and_excludes_unsafe_worktrees(
    kanban_home, tmp_path, monkeypatch
):
    repo = _make_repo(tmp_path)
    approved = repo / ".worktrees"

    def make_terminal(name: str, disposition: str) -> tuple[str, Path]:
        path = approved / name
        with kb.connect() as conn:
            task_id = kb.create_task(
                conn,
                title=name,
                workspace_kind="worktree",
                workspace_path=str(path),
                branch_name=f"feat/{name}",
            )
            task = kb.get_task(conn, task_id)
            assert task is not None
            kb.resolve_workspace(task)
            assert kb.archive_task(
                conn, task_id, workspace_disposition=disposition
            )
        return task_id, path

    candidate_id, candidate = make_terminal("candidate", "retained_until:janitor")
    pr_id, pr_path = make_terminal("open-pr", "retained_until:janitor")
    protected_path = approved / "deepsec"
    _git(repo, "worktree", "add", "-b", "ops/deepsec", str(protected_path), "HEAD")
    exception_config = tmp_path / "exceptions.yaml"
    exception_config.write_text(
        "version: 1\nexceptions:\n"
        "  - policy_id: deepsec-operational\n"
        f"    repo_path: {repo}\n"
        f"    workspace_path: {protected_path}\n",
        encoding="utf-8",
    )
    imported = kw.import_protected_exceptions(exception_config)
    assert imported["imported"] == [
        {
            "policy_id": "deepsec-operational",
            "workspace_path": str(protected_path.resolve()),
        }
    ]
    dependency_id, dependency_path = make_terminal(
        "dependency", "retained_until:successor_terminal"
    )
    dirty_id, dirty_path = make_terminal("dirty", "retained_until:janitor")
    (dirty_path / "late-change.txt").write_text("changed after terminal\n", encoding="utf-8")
    active_path = approved / "active"
    with kb.connect() as conn:
        active_id = kb.create_task(
            conn,
            title="active",
            workspace_kind="worktree",
            workspace_path=str(active_path),
            branch_name="feat/active",
        )
        kb.resolve_workspace(kb.get_task(conn, active_id))

    with kw.connect_registry() as registry:
        registry.execute(
            "UPDATE workspaces SET pr_numbers='71978' WHERE task_id=?", (pr_id,)
        )
        registry.execute(
            "UPDATE workspaces SET disposition='retained_until:janitor', "
            "status='retirement_queued' WHERE task_id=?",
            (active_id,),
        )

    real_subprocess_run = kw.subprocess.run
    removal_attempts: list[list[str]] = []

    def reject_removal(command, *args, **kwargs):
        rendered = [str(part) for part in command]
        if "worktree" in rendered and "remove" in rendered:
            removal_attempts.append(rendered)
            raise AssertionError("dry-run janitor attempted worktree removal")
        return real_subprocess_run(command, *args, **kwargs)

    monkeypatch.setattr(kw.subprocess, "run", reject_removal)
    plan = kw.plan_janitor(repo_paths=[repo], approved_roots=[approved])
    assert plan["dry_run"] is True
    assert plan["removed"] == []
    assert [item["workspace_path"] for item in plan["candidates"]] == [
        str(candidate.resolve())
    ]
    assert plan["receipts"] == [
        {
            "workspace_id": kw.list_workspace_records(task_id=candidate_id)[0].workspace_id,
            "workspace_path": str(candidate.resolve()),
            "repo_path": str(repo.resolve()),
            "head_sha": _git(candidate, "rev-parse", "HEAD"),
            "dirty_manifest_hash": kw.dirty_manifest_hash(candidate),
            "planned_action": "retire_worktree",
            "execution": "disabled",
            "requires_live_reverification": True,
        }
    ]
    assert removal_attempts == []
    reasons = {
        item["workspace_path"]: item["reasons"] for item in plan["excluded"]
    }
    assert "open_or_unverifiable_pr" in reasons[str(pr_path.resolve())]
    assert "protected_exception" in reasons[str(protected_path.resolve())]
    assert "retention_condition_not_met" in reasons[str(dependency_path.resolve())]
    assert "dirty_or_changed" in reasons[str(dirty_path.resolve())]
    assert "task_not_terminal" in reasons[str(active_path.resolve())]
    assert dependency_id and dirty_id

    inventory = kw.reconcile_inventory(
        repo_paths=[repo], approved_roots=[approved]
    )
    assert inventory["missing_expired_protected_exceptions"] == []
    protected_record = next(
        record
        for record in kw.list_workspace_records(repo_path=repo)
        if record.workspace_path == str(protected_path.resolve())
    )
    assert protected_record.status == "protected"

    from hermes_cli import kanban as kc

    cli_plan = json.loads(
        kc.run_slash(
            f"worktrees janitor --repo {repo} --approved-root {approved} --json"
        )
    )
    assert cli_plan == plan


def test_janitor_excludes_workspace_with_live_terminal_intent(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    approved = repo / ".worktrees"
    target = approved / "terminalizing-janitor"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="terminalizing janitor safety",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/terminalizing-janitor",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)
        assert kb.complete_task(
            conn,
            task_id,
            metadata={"workspace_disposition": "retained_until:janitor"},
        )

    plans = kw.prepare_terminal_disposition(
        task_id=task_id,
        board_id="default",
        disposition=None,
    )
    operation_id = kw.stage_terminal_disposition_intent(
        plans,
        terminal_status="archived",
        task_db_path=kb.kanban_db_path(),
    )
    assert operation_id is not None

    plan = kw.plan_janitor(repo_paths=[repo], approved_roots=[approved])
    assert plan["candidates"] == []
    excluded = plan["excluded"]
    assert isinstance(excluded, list)
    exclusion = next(
        item
        for item in excluded
        if item["workspace_path"] == str(target.resolve())
    )
    assert "terminal_disposition_pending" in exclusion["reasons"]
    record = kw.list_workspace_records(task_id=task_id)[0]
    assert record.status == "terminalizing"
    assert kw.abort_terminal_disposition_intent(operation_id) == 1


def test_janitor_never_candidates_primary_or_locked_worktrees(
    kanban_home, tmp_path
):
    repo = _make_repo(tmp_path)
    locked = repo / ".worktrees" / "locked"

    with kb.connect() as conn:
        primary_task = kb.create_task(
            conn,
            title="primary safety",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name="main",
        )
        primary_reservation = kw.reserve_workspace(
            repo_path=repo,
            workspace_path=repo,
            task_id=primary_task,
            board_id="default",
            owner_profile="developer",
            branch="main",
        )
        kw.verify_workspace(
            primary_reservation.workspace_id,
            repo,
            reservation_token=primary_reservation.reservation_token,
        )
        assert kb.archive_task(
            conn,
            primary_task,
            workspace_disposition="retained_until:janitor",
        )

        locked_task = kb.create_task(
            conn,
            title="locked safety",
            workspace_kind="worktree",
            workspace_path=str(locked),
            branch_name="feat/locked",
        )
        locked_record = kb.get_task(conn, locked_task)
        assert locked_record is not None
        kb.resolve_workspace(locked_record)
        assert kb.archive_task(
            conn,
            locked_task,
            workspace_disposition="retained_until:janitor",
        )
    _git(repo, "worktree", "lock", str(locked))

    plan = kw.plan_janitor(repo_paths=[repo], approved_roots=[tmp_path])
    assert plan["candidates"] == []
    excluded = plan["excluded"]
    assert isinstance(excluded, list)
    reasons = {
        item["workspace_path"]: item["reasons"] for item in excluded
    }
    assert "primary_worktree" in reasons[str(repo.resolve())]
    assert "locked_worktree" in reasons[str(locked.resolve())]


def test_terminal_snapshot_advances_once_and_inventory_does_not_bless_drift(
    tmp_path, kanban_home
):
    repo = _make_repo(tmp_path)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="snapshot",
            assignee="developer",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name="feat/snapshot",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        workspace = kb.resolve_workspace(task)

        (workspace / "before-terminal.txt").write_text("before\n", encoding="utf-8")
        _git(workspace, "add", "before-terminal.txt")
        _git(workspace, "commit", "-m", "before terminal")
        terminal_head = _git(workspace, "rev-parse", "HEAD")

        assert kb.archive_task(
            conn,
            task_id,
            workspace_disposition="retained_until:janitor",
        )

    record = kw.list_workspace_records(task_id=task_id)[0]
    assert record.head_sha == terminal_head

    (workspace / "after-terminal.txt").write_text("after\n", encoding="utf-8")
    _git(workspace, "add", "after-terminal.txt")
    _git(workspace, "commit", "-m", "after terminal")
    drifted_head = _git(workspace, "rev-parse", "HEAD")
    assert drifted_head != terminal_head

    kw.reconcile_inventory(
        repo_paths=[repo],
        approved_roots=[repo / ".worktrees"],
    )
    refreshed = kw.list_workspace_records(task_id=task_id)[0]
    assert refreshed.head_sha == terminal_head

    plan = kw.plan_janitor(
        repo_paths=[repo],
        approved_roots=[repo / ".worktrees"],
    )
    excluded = plan["excluded"]
    assert isinstance(excluded, list)
    reasons = {
        str(item["workspace_path"]): item["reasons"]
        for item in excluded
        if isinstance(item, dict)
    }
    assert "dirty_or_changed" in reasons[str(workspace.resolve())]
    assert plan["candidates"] == []


def test_creation_pressure_thresholds_are_config_backed_and_report_only_by_default():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    lifecycle = DEFAULT_CONFIG["kanban"]["worktree_lifecycle"]
    assert lifecycle["enforcement"] == "report_only"
    assert lifecycle["warning_worktree_count_per_repo"] == 20
    assert lifecycle["warning_estimated_bytes_per_repo"] == 100 * 1024 * 1024 * 1024
    assert lifecycle["warning_free_space_floor_bytes"] == 50 * 1024 * 1024 * 1024



def test_protected_exception_import_is_config_driven_and_reported(tmp_path, kanban_home):
    import yaml

    from hermes_cli import kanban as kc

    repo = _make_repo(tmp_path)
    deepsec = repo / ".worktrees" / "deepsec-operational"
    missing = repo / ".worktrees" / "missing-protection"
    expired = repo / ".worktrees" / "expired-protection"
    _git(repo, "worktree", "add", "-b", "ops/deepsec", str(deepsec), "HEAD")
    _git(repo, "worktree", "add", "-b", "ops/missing", str(missing), "HEAD")
    _git(repo, "worktree", "add", "-b", "ops/expired", str(expired), "HEAD")
    policy = tmp_path / "exceptions.yaml"
    policy.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "exceptions": [
                    {
                        "policy_id": "deepsec-operational",
                        "repo_path": str(repo),
                        "workspace_path": str(deepsec),
                        "purpose": "operational",
                        "expires_at": None,
                    },
                    {
                        "policy_id": "expired-operational",
                        "repo_path": str(repo),
                        "workspace_path": str(expired),
                        "purpose": "operational",
                        "expires_at": 1,
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = json.loads(
        kc.run_slash(f"worktrees exceptions import --config {policy} --json")
    )
    assert result == {
        "imported": [
            {
                "policy_id": "deepsec-operational",
                "workspace_path": str(deepsec.resolve()),
            }
        ],
        "skipped": [
            {
                "policy_id": "expired-operational",
                "workspace_path": str(expired.resolve()),
                "reason": "expired_policy",
            }
        ],
    }
    record = next(
        item for item in kw.list_workspace_records()
        if item.workspace_path == str(deepsec.resolve())
    )
    assert record.task_id is None
    assert record.status == "protected"
    assert record.cleanup_policy == "never_automatic"
    assert record.disposition == "operational_exception:deepsec-operational"
    assert record.exception_policy_id == "deepsec-operational"
    assert record.exception_expires_at is None

    missing_policy = tmp_path / "missing-exceptions.yaml"
    missing_policy.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "exceptions": [
                    {
                        "policy_id": "deepsec-operational",
                        "repo_path": str(repo),
                        "workspace_path": str(deepsec),
                        "purpose": "operational",
                        "expires_at": None,
                    },
                    {
                        "policy_id": "missing-operational",
                        "repo_path": str(repo),
                        "workspace_path": str(missing),
                        "purpose": "operational",
                        "expires_at": None,
                    },
                    {
                        "policy_id": "expired-operational",
                        "repo_path": str(repo),
                        "workspace_path": str(expired),
                        "purpose": "operational",
                        "expires_at": 1,
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report = json.loads(
        kc.run_slash(
            f"worktrees inventory --repo {repo} "
            f"--approved-root {repo / '.worktrees'} "
            f"--exception-config {missing_policy} --json"
        )
    )
    assert report["missing_expired_protected_exceptions"] == [
        {
            "policy_id": "expired-operational",
            "workspace_path": str(expired.resolve()),
            "reason": "expired_policy",
        },
        {
            "policy_id": "missing-operational",
            "workspace_path": str(missing.resolve()),
            "reason": "missing_registry_record",
        },
    ]

    checked_in = (
        Path(__file__).parents[2]
        / "docs" / "examples" / "kanban-worktree-exceptions.yaml"
    )
    example = checked_in.read_text(encoding="utf-8")
    assert "deepsec-operational" in example
    assert "/Users/royce" not in example


def test_inventory_reports_missing_and_expired_protected_exceptions(
    tmp_path, kanban_home
):
    import yaml

    repo = _make_repo(tmp_path)
    approved = repo / ".worktrees"
    current = approved / "deepsec-current"
    expired = approved / "deepsec-expired"
    _git(repo, "worktree", "add", "-b", "ops/deepsec-current", str(current), "HEAD")
    _git(repo, "worktree", "add", "-b", "ops/deepsec-expired", str(expired), "HEAD")

    policy = tmp_path / "exceptions.yaml"
    policy.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "exceptions": [
                    {
                        "policy_id": "deepsec-current",
                        "repo_path": str(repo),
                        "workspace_path": str(current),
                        "purpose": "operational",
                    },
                    {
                        "policy_id": "deepsec-expired",
                        "repo_path": str(repo),
                        "workspace_path": str(expired),
                        "purpose": "operational",
                        "expires_at": 1,
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    missing = kw.reconcile_inventory(
        repo_paths=[repo],
        approved_roots=[approved],
        exception_config=policy,
    )
    assert missing["missing_expired_protected_exceptions"] == [
        {
            "policy_id": "deepsec-current",
            "workspace_path": str(current.resolve()),
            "reason": "missing_registry_record",
        },
        {
            "policy_id": "deepsec-expired",
            "workspace_path": str(expired.resolve()),
            "reason": "expired_policy",
        },
    ]

    imported = kw.import_protected_exceptions(policy)
    assert imported["imported"] == [
        {
            "policy_id": "deepsec-current",
            "workspace_path": str(current.resolve()),
        }
    ]
    assert imported["skipped"] == [
        {
            "policy_id": "deepsec-expired",
            "workspace_path": str(expired.resolve()),
            "reason": "expired_policy",
        }
    ]
    reconciled = kw.reconcile_inventory(
        repo_paths=[repo],
        approved_roots=[approved],
        exception_config=policy,
    )
    assert reconciled["missing_expired_protected_exceptions"] == [
        {
            "policy_id": "deepsec-expired",
            "workspace_path": str(expired.resolve()),
            "reason": "expired_policy",
        }
    ]

    empty_policy = tmp_path / "empty-exceptions.yaml"
    empty_policy.write_text("version: 1\nexceptions: []\n", encoding="utf-8")
    missing_config = kw.reconcile_inventory(
        repo_paths=[repo],
        approved_roots=[approved],
        exception_config=empty_policy,
    )
    assert missing_config["missing_expired_protected_exceptions"] == [
        {
            "policy_id": "deepsec-current",
            "workspace_path": str(current.resolve()),
            "reason": "missing_policy_config",
        }
    ]


def test_exception_config_rejects_duplicate_policy_ids_and_workspace_paths(
    tmp_path, kanban_home
):
    import yaml

    repo = _make_repo(tmp_path)
    first = repo / ".worktrees" / "first"
    second = repo / ".worktrees" / "second"
    policy = tmp_path / "duplicates.yaml"

    policy.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "exceptions": [
                    {
                        "policy_id": "duplicate",
                        "repo_path": str(repo),
                        "workspace_path": str(first),
                    },
                    {
                        "policy_id": "duplicate",
                        "repo_path": str(repo),
                        "workspace_path": str(second),
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="duplicate.*policy_id"):
        kw.load_exception_config(policy)

    policy.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "exceptions": [
                    {
                        "policy_id": "first",
                        "repo_path": str(repo),
                        "workspace_path": str(first),
                    },
                    {
                        "policy_id": "second",
                        "repo_path": str(repo),
                        "workspace_path": str(first),
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="duplicate.*workspace_path"):
        kw.load_exception_config(policy)


def test_board_identity_stays_exact_when_database_path_is_overridden(
    tmp_path, kanban_home, monkeypatch
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "beta-overridden"
    kb.create_board("beta")
    beta_db = kb.kanban_db_path("beta").resolve()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(beta_db))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")

    with kb.connect(board="beta") as conn:
        task_id = kb.create_task(
            conn,
            title="beta override",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/beta-override",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task, board="beta")
        with pytest.raises(RuntimeError, match="workspace_disposition"):
            kb.complete_task(conn, task_id)
        assert kb.get_task(conn, task_id).status == "ready"

    record = kw.list_workspace_records(task_id=task_id)[0]
    assert record.board_id == "beta"
    assert record.disposition is None


def test_stale_completion_cannot_commit_workspace_disposition(
    tmp_path, kanban_home
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "stale-run"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="stale completion",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/stale-run",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)
        claimed = kb.claim_task(conn, task_id, claimer="test-worker")
        assert claimed is not None and claimed.current_run_id is not None
        assert not kb.complete_task(
            conn,
            task_id,
            expected_run_id=claimed.current_run_id + 1,
            metadata={"workspace_disposition": "retained_until:review"},
        )
        assert kb.get_task(conn, task_id).status == "running"

    record = kw.list_workspace_records(task_id=task_id)[0]
    assert record.status == "active"
    assert record.disposition is None


def test_registry_intent_stage_failure_leaves_task_nonterminal(
    tmp_path, kanban_home, monkeypatch
):
    import importlib

    live_kw = importlib.import_module("hermes_cli.kanban_workspaces")
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "registry-failure"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="registry failure",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/registry-failure",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)

        def fail_registry_stage(*_args, **_kwargs):
            raise RuntimeError("injected registry failure")

        monkeypatch.setattr(
            live_kw,
            "stage_terminal_disposition_intent",
            fail_registry_stage,
        )
        with pytest.raises(RuntimeError, match="injected registry failure"):
            kb.complete_task(
                conn,
                task_id,
                metadata={"workspace_disposition": "retained_until:review"},
            )
        unchanged_task = kb.get_task(conn, task_id)
        assert unchanged_task is not None
        assert unchanged_task.status == "ready"

    record = kw.list_workspace_records(task_id=task_id)[0]
    assert record.status == "active"
    assert record.disposition is None


def test_protected_import_never_reclassifies_task_owned_workspace(
    tmp_path, kanban_home
):
    import yaml

    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "task-owned"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="task owned",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/task-owned",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.resolve_workspace(task)

    policy = tmp_path / "task-owned-exception.yaml"
    policy.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "exceptions": [
                    {
                        "policy_id": "cannot-adopt-task-owned",
                        "repo_path": str(repo),
                        "workspace_path": str(target),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = kw.import_protected_exceptions(policy)
    assert result["imported"] == []
    assert result["skipped"] == [
        {
            "policy_id": "cannot-adopt-task-owned",
            "workspace_path": str(target.resolve()),
            "reason": "task_owned_registry_record",
        }
    ]
    record = kw.list_workspace_records(task_id=task_id)[0]
    assert record.status == "active"
    assert record.disposition is None
    assert record.exception_policy_id is None


def test_repository_identity_is_shared_by_primary_and_linked_checkouts(
    tmp_path, kanban_home
):
    repo = _make_repo(tmp_path)
    linked = repo / ".worktrees" / "linked-anchor"
    target = repo / ".worktrees" / "owned"
    _git(repo, "worktree", "add", "-b", "feat/linked-anchor", str(linked), "HEAD")

    first = kw.reserve_workspace(
        repo_path=repo,
        workspace_path=target,
        task_id="t_repo_identity",
        board_id="default",
        owner_profile="developer",
        branch="feat/owned",
    )
    second = kw.reserve_workspace(
        repo_path=linked,
        workspace_path=target,
        task_id="t_repo_identity",
        board_id="default",
        owner_profile="developer",
        branch="feat/owned",
    )

    assert second.workspace_id == first.workspace_id
    assert second.repo_id == first.repo_id
    assert second.repo_path == str(repo.resolve())
    assert len(kw.list_workspace_records(task_id="t_repo_identity")) == 1


def test_dispositions_require_recovery_owner_and_current_exception_policy(
    tmp_path, kanban_home, monkeypatch
):
    import importlib
    import yaml

    live_kw = importlib.import_module("hermes_cli.kanban_workspaces")
    repo = _make_repo(tmp_path)
    dirty = repo / ".worktrees" / "dirty-policy"
    protected = repo / ".worktrees" / "protected-policy"
    with kb.connect() as conn:
        dirty_id = kb.create_task(
            conn,
            title="dirty policy",
            workspace_kind="worktree",
            workspace_path=str(dirty),
            branch_name="feat/dirty-policy",
        )
        kb.resolve_workspace(kb.get_task(conn, dirty_id))
        (dirty / "recovery.patch").write_text("recover\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="artifact.*owner"):
            kb.complete_task(
                conn,
                dirty_id,
                metadata={"workspace_disposition": "preserved_dirty:review later"},
            )

        protected_id = kb.create_task(
            conn,
            title="protected policy",
            workspace_kind="worktree",
            workspace_path=str(protected),
            branch_name="feat/protected-policy",
        )
        kb.resolve_workspace(kb.get_task(conn, protected_id))
        with pytest.raises(RuntimeError, match="current operational exception policy"):
            kb.archive_task(
                conn,
                protected_id,
                workspace_disposition="operational_exception:current-policy",
            )

        policy = tmp_path / "current-policy.yaml"
        policy.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "exceptions": [
                        {
                            "policy_id": "current-policy",
                            "repo_path": str(repo),
                            "workspace_path": str(protected),
                            "expires_at": None,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            live_kw, "_configured_exception_path", lambda: policy
        )
        assert kb.archive_task(
            conn,
            protected_id,
            workspace_disposition="operational_exception:current-policy",
        )


def test_inventory_reports_expired_registered_exception_without_config(
    tmp_path, kanban_home
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "expired-registered"
    _git(repo, "worktree", "add", "-b", "ops/expired-registered", str(target), "HEAD")
    now = 1
    with kw.connect_registry() as registry:
        registry.execute(
            "INSERT INTO workspaces (workspace_id, repo_id, repo_path, workspace_path, "
            "owner_profile, purpose, branch, status, cleanup_policy, disposition, "
            "exception_policy_id, exception_expires_at, created_at, last_used_at) "
            "VALUES ('w_expired', ?, ?, ?, 'operator', 'operational', "
            "'ops/expired-registered', 'protected', 'never_automatic', "
            "'operational_exception:expired-registered', 'expired-registered', 1, ?, ?)",
            (kw._repo_id(str(repo.resolve())), str(repo.resolve()), str(target.resolve()), now, now),
        )

    report = kw.reconcile_inventory(repo_paths=[repo], approved_roots=[repo / ".worktrees"])
    assert report["missing_expired_protected_exceptions"] == [
        {
            "policy_id": "expired-registered",
            "workspace_path": str(target.resolve()),
            "reason": "expired_policy",
        }
    ]


def test_detached_branchless_reservation_verifies_as_active(tmp_path, kanban_home):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "detached"
    _git(repo, "worktree", "add", "--detach", str(target), "HEAD")

    reservation = kw.reserve_workspace(
        repo_path=repo,
        workspace_path=target,
        task_id="t_detached",
        board_id="default",
        owner_profile="developer",
        branch=None,
    )
    verified = kw.verify_workspace(
        reservation.workspace_id,
        target,
        reservation_token=reservation.reservation_token,
    )

    assert verified.status == "active"
    assert verified.branch is None
    assert verified.head_sha == _git(target, "rev-parse", "HEAD")


def test_reservation_reuse_is_exact_and_creation_failure_cannot_downgrade_active(
    tmp_path, kanban_home
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "concurrent"
    reservation = kw.reserve_workspace(
        repo_path=repo,
        workspace_path=target,
        task_id="t_concurrent",
        board_id="default",
        owner_profile="developer",
        purpose="implementation",
        branch="feat/concurrent",
        cleanup_policy="on_task_terminal",
        retention_condition="task_terminal",
    )
    _git(repo, "worktree", "add", "-b", "feat/concurrent", str(target), "HEAD")
    kw.verify_workspace(
        reservation.workspace_id,
        target,
        reservation_token=reservation.reservation_token,
    )
    kw.mark_creation_failed(reservation.workspace_id)
    assert kw.list_workspace_records(task_id="t_concurrent")[0].status == "active"

    with pytest.raises(RuntimeError, match="ambiguous worktree ownership"):
        kw.reserve_workspace(
            repo_path=repo,
            workspace_path=target,
            task_id="t_concurrent",
            board_id="default",
            owner_profile="reviewer",
            purpose="review",
            branch="feat/concurrent",
            cleanup_policy="never_automatic",
            retention_condition="manual",
        )


def test_concurrent_public_resolvers_share_one_materialization_without_poisoning(
    tmp_path, kanban_home, monkeypatch
):
    repo = _make_repo(tmp_path)
    target = repo / ".worktrees" / "shared-reservation"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="concurrent shared reservation",
            assignee="developer",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name="feat/shared-reservation",
        )
        task = kb.get_task(conn, task_id)
    assert task is not None

    real_ensure = kb._ensure_git_worktree
    real_verify = kw.verify_workspace
    materialized = threading.Event()
    non_owner_waiting = threading.Event()
    release_owner = threading.Event()
    call_lock = threading.Lock()
    materialization_calls = 0

    def controlled_materialization(
        repo_root: Path, requested: Path, branch_name: str
    ) -> None:
        nonlocal materialization_calls
        with call_lock:
            materialization_calls += 1
            call_number = materialization_calls
        if call_number != 1:
            raise RuntimeError("compatible resolver must not materialize shared reservation")
        real_ensure(repo_root, requested, branch_name)
        materialized.set()
        assert release_owner.wait(timeout=10), "test did not release reservation owner"

    def observed_verification(
        workspace_id: str,
        workspace_path: Path | str,
        *,
        reservation_token: str | None = None,
    ) -> kw.WorkspaceRecord:
        if reservation_token is None:
            non_owner_waiting.set()
        return real_verify(
            workspace_id,
            workspace_path,
            reservation_token=reservation_token,
        )

    monkeypatch.setattr(kb, "_ensure_git_worktree", controlled_materialization)
    monkeypatch.setattr(kw, "verify_workspace", observed_verification)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(kb.resolve_workspace, task)
        assert materialized.wait(timeout=10), "owner did not materialize worktree"
        compatible = pool.submit(kb.resolve_workspace, task)
        assert non_owner_waiting.wait(timeout=10), "losing resolver did not observe claim"
        with pytest.raises(concurrent.futures.TimeoutError):
            compatible.result(timeout=0.1)
        release_owner.set()
        owner_result = owner.result(timeout=10)
        compatible_result = compatible.result(timeout=10)

    assert owner_result == compatible_result == target.resolve()
    assert materialization_calls == 1
    records = kw.list_workspace_records(task_id=task_id)
    assert len(records) == 1
    assert records[0].status == "active"


def test_janitor_uses_production_pr_and_task_dependency_evidence(
    tmp_path, kanban_home
):
    repo = _make_repo(tmp_path)
    approved = repo / ".worktrees"
    pr_path = approved / "explicit-pr"
    dependency_path = approved / "explicit-dependency"
    with kb.connect() as conn:
        pr_id = kb.create_task(
            conn,
            title="explicit PR",
            workspace_kind="worktree",
            workspace_path=str(pr_path),
            branch_name="feat/explicit-pr",
        )
        kb.resolve_workspace(kb.get_task(conn, pr_id))
        assert kb.complete_task(
            conn,
            pr_id,
            metadata={
                "workspace_disposition": "retained_until:janitor",
                "workspace_pr_numbers": [71978],
            },
        )

        dependency_id = kb.create_task(
            conn,
            title="explicit dependency",
            workspace_kind="worktree",
            workspace_path=str(dependency_path),
            branch_name="feat/explicit-dependency",
        )
        kb.resolve_workspace(kb.get_task(conn, dependency_id))
        assert kb.complete_task(
            conn,
            dependency_id,
            metadata={"workspace_disposition": "retained_until:janitor"},
        )
        successor_id = kb.create_task(conn, title="active successor", parents=[dependency_id])
        assert kb.get_task(conn, successor_id).status == "ready"

    pr_record = kw.list_workspace_records(task_id=pr_id)[0]
    assert pr_record.pr_numbers == (71978,)
    plan = kw.plan_janitor(repo_paths=[repo], approved_roots=[approved])
    reasons = {
        item["workspace_path"]: item["reasons"] for item in plan["excluded"]
    }
    assert "open_or_unverifiable_pr" in reasons[str(pr_path.resolve())]
    assert "dependency_retained" in reasons[str(dependency_path.resolve())]
