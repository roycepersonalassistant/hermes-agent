"""Canonical, profile-aware registry for Kanban-managed Git worktrees.

The registry is shared across profiles and boards at
``<hermes-root>/kanban/workspace-registry.db``.  Git remains the authority for
which linked worktrees physically exist; this database is the authority for
ownership, lifecycle policy, and cleanup disposition.  Unknown Git paths are
reported by reconciliation and are never adopted or deleted implicitly.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import socket
import sqlite3
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import psutil
import yaml


_log = logging.getLogger(__name__)


VALID_DISPOSITION_PREFIXES = {
    "retired",
    "retained_until",
    "preserved_dirty",
    "operational_exception",
}
ACTIVE_REGISTRY_STATUSES = {
    "reserved",
    "active",
    "blocked",
    "terminalizing",
    "terminal_clean",
    "terminal_dirty",
    "superseded",
    "retirement_queued",
    "protected",
}
RESERVATION_LEASE_SECONDS = 180
TERMINAL_INTENT_LEASE_SECONDS = 300


@dataclass(frozen=True)
class WorkspaceRecord:
    workspace_id: str
    repo_id: str
    repo_path: str
    workspace_path: str
    task_id: Optional[str]
    board_id: Optional[str]
    owner_profile: Optional[str]
    purpose: str
    branch: Optional[str]
    head_sha: Optional[str]
    pr_numbers: tuple[int, ...]
    status: str
    cleanup_policy: str
    retention_condition: Optional[str]
    disposition: Optional[str]
    dirty_manifest_hash: Optional[str]
    estimated_bytes: int
    exception_policy_id: Optional[str]
    exception_expires_at: Optional[int]
    replacement_reason: Optional[str]
    created_at: int
    last_used_at: int
    last_verified_at: Optional[int]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "WorkspaceRecord":
        raw_prs = str(row["pr_numbers"] or "")
        prs = tuple(int(value) for value in raw_prs.split(",") if value.strip())
        return cls(
            workspace_id=row["workspace_id"],
            repo_id=row["repo_id"],
            repo_path=row["repo_path"],
            workspace_path=row["workspace_path"],
            task_id=row["task_id"],
            board_id=row["board_id"],
            owner_profile=row["owner_profile"],
            purpose=row["purpose"],
            branch=row["branch"],
            head_sha=row["head_sha"],
            pr_numbers=prs,
            status=row["status"],
            cleanup_policy=row["cleanup_policy"],
            retention_condition=row["retention_condition"],
            disposition=row["disposition"],
            dirty_manifest_hash=row["dirty_manifest_hash"],
            estimated_bytes=int(row["estimated_bytes"] or 0),
            exception_policy_id=row["exception_policy_id"],
            exception_expires_at=row["exception_expires_at"],
            replacement_reason=row["replacement_reason"],
            created_at=int(row["created_at"]),
            last_used_at=int(row["last_used_at"]),
            last_verified_at=row["last_verified_at"],
        )


@dataclass(frozen=True)
class WorkspaceReservation(WorkspaceRecord):
    """A registry record plus this caller's materialization claim.

    Compatible concurrent callers can observe the same registry row, but only
    the caller holding ``reservation_token`` may execute ``git worktree add``,
    publish the row as active, or mark creation failed. Non-owners wait for that
    explicit active publication before using the checkout.
    """

    reservation_token: Optional[str] = None
    owns_materialization: bool = False

    @classmethod
    def from_record(
        cls,
        record: WorkspaceRecord,
        *,
        reservation_token: Optional[str],
        owns_materialization: bool,
    ) -> "WorkspaceReservation":
        return cls(
            **record.__dict__,
            reservation_token=reservation_token,
            owns_materialization=owns_materialization,
        )


class WorkspaceReservationPending(RuntimeError):
    """The reservation owner has not exposed a verifiable Git checkout yet."""


@dataclass(frozen=True)
class _TerminalDispositionPlan:
    record: WorkspaceRecord
    disposition: str
    status: str
    dirty_manifest_hash: Optional[str]
    exception_policy_id: Optional[str]
    retention_condition: Optional[str]
    head_sha: Optional[str]
    branch: Optional[str]
    estimated_bytes: int
    pr_numbers: tuple[int, ...]
    exception_expires_at: Optional[int]
    verified_at: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id          TEXT PRIMARY KEY,
    repo_id               TEXT NOT NULL,
    repo_path             TEXT NOT NULL,
    workspace_path        TEXT NOT NULL UNIQUE,
    task_id               TEXT,
    board_id              TEXT,
    owner_profile         TEXT,
    purpose               TEXT NOT NULL,
    branch                TEXT,
    head_sha              TEXT,
    pr_numbers            TEXT NOT NULL DEFAULT '',
    status                TEXT NOT NULL,
    cleanup_policy        TEXT NOT NULL,
    retention_condition   TEXT,
    disposition           TEXT,
    dirty_manifest_hash   TEXT,
    estimated_bytes       INTEGER NOT NULL DEFAULT 0,
    exception_policy_id   TEXT,
    exception_expires_at  INTEGER,
    replacement_reason    TEXT,
    reservation_token     TEXT,
    reservation_expires_at INTEGER,
    created_at            INTEGER NOT NULL,
    last_used_at          INTEGER NOT NULL,
    last_verified_at      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_workspaces_task_repo
    ON workspaces(board_id, task_id, repo_id);
CREATE INDEX IF NOT EXISTS idx_workspaces_repo_head
    ON workspaces(repo_id, head_sha);
CREATE INDEX IF NOT EXISTS idx_workspaces_status
    ON workspaces(status);
CREATE TABLE IF NOT EXISTS terminal_disposition_intents (
    operation_id            TEXT NOT NULL,
    workspace_id            TEXT NOT NULL UNIQUE,
    board_id                TEXT NOT NULL,
    task_id                 TEXT NOT NULL,
    terminal_status         TEXT NOT NULL,
    task_db_path            TEXT NOT NULL,
    expected_repo_id        TEXT NOT NULL,
    expected_workspace_path TEXT NOT NULL,
    expected_status         TEXT NOT NULL,
    disposition             TEXT NOT NULL,
    target_status           TEXT NOT NULL,
    dirty_manifest_hash     TEXT,
    exception_policy_id     TEXT,
    exception_expires_at    INTEGER,
    retention_condition     TEXT,
    head_sha                TEXT,
    branch                  TEXT,
    pr_numbers              TEXT NOT NULL DEFAULT '',
    estimated_bytes         INTEGER NOT NULL DEFAULT 0,
    verified_at             INTEGER NOT NULL,
    owner_host              TEXT NOT NULL,
    owner_pid               INTEGER NOT NULL,
    owner_started_at        REAL,
    expires_at              INTEGER NOT NULL,
    created_at              INTEGER NOT NULL,
    PRIMARY KEY (operation_id, workspace_id)
);
CREATE INDEX IF NOT EXISTS idx_terminal_disposition_intents_task
    ON terminal_disposition_intents(board_id, task_id, operation_id);
"""


def registry_path() -> Path:
    from hermes_cli import kanban_db

    return kanban_db.kanban_home() / "kanban" / "workspace-registry.db"


def _add_registry_column(
    conn: sqlite3.Connection, columns: set[str], name: str, ddl: str
) -> None:
    if name in columns:
        return
    try:
        conn.execute(ddl)
    except sqlite3.OperationalError as exc:
        # Two first-use processes can both observe the pre-migration schema.
        # ALTER TABLE is not IF-NOT-EXISTS-capable on supported SQLite builds;
        # the loser may safely accept the winner's identical column.
        if "duplicate column name" not in str(exc).lower():
            raise


def connect_registry() -> sqlite3.Connection:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(_SCHEMA)
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(workspaces)").fetchall()
    }
    _add_registry_column(
        conn,
        columns,
        "exception_expires_at",
        "ALTER TABLE workspaces ADD COLUMN exception_expires_at INTEGER",
    )
    _add_registry_column(
        conn,
        columns,
        "reservation_token",
        "ALTER TABLE workspaces ADD COLUMN reservation_token TEXT",
    )
    _add_registry_column(
        conn,
        columns,
        "reservation_expires_at",
        "ALTER TABLE workspaces ADD COLUMN reservation_expires_at INTEGER",
    )
    intent_columns = {
        str(row[1])
        for row in conn.execute(
            "PRAGMA table_info(terminal_disposition_intents)"
        ).fetchall()
    }
    _add_registry_column(
        conn,
        intent_columns,
        "task_db_path",
        "ALTER TABLE terminal_disposition_intents ADD COLUMN task_db_path TEXT",
    )
    _add_registry_column(
        conn,
        intent_columns,
        "owner_started_at",
        "ALTER TABLE terminal_disposition_intents ADD COLUMN owner_started_at REAL",
    )
    return conn


def _canonical(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _canonical_repo(path: Path | str) -> str:
    """Return one identity path for a repository from any of its checkouts."""
    candidate = _canonical(path)
    raw = _git(candidate, "worktree", "list", "--porcelain")
    if raw:
        for line in raw.splitlines():
            if line.startswith("worktree "):
                return _canonical(line.removeprefix("worktree "))
    return candidate


def _repo_id(repo_path: str) -> str:
    canonical = _canonical_repo(repo_path)
    return "r_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _git(path: Path | str, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip()


def dirty_manifest_hash(path: Path | str) -> Optional[str]:
    raw = _git(path, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if raw is None:
        return None
    return hashlib.sha256(raw.encode("utf-8", errors="surrogateescape")).hexdigest()


def estimated_bytes(path: Path | str) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    try:
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d != ".git")
            for name in sorted(files):
                candidate = Path(current) / name
                try:
                    if not candidate.is_symlink():
                        total += candidate.stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


def list_workspace_records(
    *,
    task_id: Optional[str] = None,
    board_id: Optional[str] = None,
    repo_path: Optional[Path | str] = None,
) -> list[WorkspaceRecord]:
    # Any registry read is also a recovery boundary. This keeps inventory and
    # the fail-closed janitor from silently observing a terminal task without
    # first reconciling its durable disposition intent.
    recover_terminal_disposition_intents(board_id=board_id, task_id=task_id)
    clauses: list[str] = []
    params: list[object] = []
    if task_id is not None:
        clauses.append("task_id = ?")
        params.append(task_id)
    if board_id is not None:
        clauses.append("board_id = ?")
        params.append(board_id)
    if repo_path is not None:
        clauses.append("repo_id = ?")
        params.append(_repo_id(_canonical_repo(repo_path)))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with contextlib.closing(connect_registry()) as conn:
        rows = conn.execute(
            "SELECT * FROM workspaces" + where
            + " ORDER BY repo_path, workspace_path, workspace_id",
            tuple(params),
        ).fetchall()
    return [WorkspaceRecord.from_row(row) for row in rows]


def creation_pressure_warnings(repo_path: Path | str) -> list[str]:
    """Evaluate config-backed host/repository pressure thresholds."""
    try:
        from hermes_cli.config import load_config

        config = load_config()
        lifecycle = config.get("kanban", {}).get("worktree_lifecycle", {})
    except Exception:
        lifecycle = {}

    def _positive(name: str, fallback: int) -> int:
        try:
            value = int(lifecycle.get(name, fallback))
        except (TypeError, ValueError):
            return fallback
        return value if value > 0 else fallback

    count_limit = _positive("warning_worktree_count_per_repo", 20)
    byte_limit = _positive(
        "warning_estimated_bytes_per_repo", 100 * 1024 * 1024 * 1024
    )
    free_floor = _positive(
        "warning_free_space_floor_bytes", 50 * 1024 * 1024 * 1024
    )
    repo_path = _canonical_repo(repo_path)
    entries = _git_worktrees(repo_path)
    # Creation must not recursively walk every checkout just to emit a
    # report-only warning. Daily inventory refreshes these cached estimates;
    # creation reads the latest verified values in constant registry time.
    total_bytes = sum(
        record.estimated_bytes
        for record in list_workspace_records(repo_path=repo_path)
        if record.status != "creation_failed"
    )
    warnings: list[str] = []
    if len(entries) >= count_limit:
        warnings.append(
            f"repository {_canonical(repo_path)} has {len(entries)} worktrees "
            f"(warning threshold {count_limit})"
        )
    if total_bytes >= byte_limit:
        warnings.append(
            f"repository {_canonical(repo_path)} worktrees use approximately "
            f"{total_bytes} bytes (warning threshold {byte_limit})"
        )
    try:
        free = shutil.disk_usage(_canonical(repo_path)).free
    except OSError:
        free = None
    if free is not None and free <= free_floor:
        warnings.append(
            f"filesystem free space is {free} bytes (warning floor {free_floor})"
        )
    # The shipped contract is report-only. A future owner-approved change can
    # add execution enforcement without changing these deterministic findings.
    return warnings


def reserve_workspace(
    *,
    repo_path: Path | str,
    workspace_path: Path | str,
    task_id: str,
    board_id: str,
    owner_profile: Optional[str],
    purpose: str = "implementation",
    branch: Optional[str] = None,
    cleanup_policy: str = "on_task_terminal",
    retention_condition: Optional[str] = "task_terminal",
    replacement_reason: Optional[str] = None,
) -> WorkspaceReservation:
    """Reserve ownership before ``git worktree add`` can mutate the repo.

    The registry row remains the durable ownership identity. A short-lived,
    compare-and-swap reservation token separately identifies the one caller
    allowed to materialize it. Compatible concurrent callers observe the same
    row without receiving that token and can only verify a live checkout.
    """
    repo = _canonical_repo(repo_path)
    target = _canonical(workspace_path)
    rid = _repo_id(repo)
    for warning in creation_pressure_warnings(repo):
        _log.warning("kanban worktree lifecycle (report-only): %s", warning)
    now = int(time.time())
    claim_token = uuid.uuid4().hex
    claim_expires = now + RESERVATION_LEASE_SECONDS

    def is_compatible(row: sqlite3.Row) -> bool:
        return (
            row["workspace_path"] == target
            and row["task_id"] == task_id
            and row["board_id"] == board_id
            and row["repo_id"] == rid
            and row["owner_profile"] == owner_profile
            and row["purpose"] == purpose
            and row["branch"] == branch
            and row["cleanup_policy"] == cleanup_policy
            and row["retention_condition"] == retention_condition
        )

    def result_for(
        conn: sqlite3.Connection,
        workspace_id: str,
        *,
        token: Optional[str],
        owns: bool,
    ) -> WorkspaceReservation:
        row = conn.execute(
            "SELECT * FROM workspaces WHERE workspace_id = ?", (workspace_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"unknown workspace reservation: {workspace_id}")
        return WorkspaceReservation.from_record(
            WorkspaceRecord.from_row(row),
            reservation_token=token,
            owns_materialization=owns,
        )

    with contextlib.closing(connect_registry()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            rows = conn.execute(
                "SELECT * FROM workspaces WHERE board_id = ? AND task_id = ? "
                "AND repo_id = ? AND status IN ('reserved', 'active') "
                "ORDER BY workspace_path, workspace_id",
                (board_id, task_id, rid),
            ).fetchall()
            compatible = [row for row in rows if is_compatible(row)]
            if len(rows) == 1 and len(compatible) == 1:
                row = compatible[0]
                owns = False
                token: Optional[str] = None
                if row["status"] == "reserved" and (
                    row["reservation_token"] is None
                    or int(row["reservation_expires_at"] or 0) <= now
                ):
                    cur = conn.execute(
                        "UPDATE workspaces SET reservation_token = ?, "
                        "reservation_expires_at = ?, last_used_at = ? "
                        "WHERE workspace_id = ? AND status = 'reserved' AND ("
                        "reservation_token IS NULL OR reservation_expires_at <= ?)",
                        (claim_token, claim_expires, now, row["workspace_id"], now),
                    )
                    owns = cur.rowcount == 1
                    token = claim_token if owns else None
                else:
                    conn.execute(
                        "UPDATE workspaces SET last_used_at = ? WHERE workspace_id = ?",
                        (now, row["workspace_id"]),
                    )
                conn.execute("COMMIT")
                return result_for(
                    conn, row["workspace_id"], token=token, owns=owns
                )
            if rows and not replacement_reason:
                raise RuntimeError(
                    f"ambiguous worktree ownership for task {task_id} in repo {repo}: "
                    f"{len(rows)} active registry row(s); record an explicit "
                    "workspace replacement reason before creating another"
                )

            existing = conn.execute(
                "SELECT * FROM workspaces WHERE workspace_path = ?", (target,)
            ).fetchone()
            if existing is not None:
                if is_compatible(existing) and existing["status"] == "creation_failed":
                    cur = conn.execute(
                        "UPDATE workspaces SET status = 'reserved', reservation_token = ?, "
                        "reservation_expires_at = ?, last_used_at = ? "
                        "WHERE workspace_id = ? AND status = 'creation_failed'",
                        (
                            claim_token,
                            claim_expires,
                            now,
                            existing["workspace_id"],
                        ),
                    )
                    if cur.rowcount != 1:
                        raise RuntimeError(
                            f"workspace reservation changed while retrying: "
                            f"{existing['workspace_id']}"
                        )
                    conn.execute("COMMIT")
                    return result_for(
                        conn,
                        existing["workspace_id"],
                        token=claim_token,
                        owns=True,
                    )
                exact_compatible_owner = (
                    is_compatible(existing)
                    and existing["status"] in {"reserved", "active"}
                )
                if not exact_compatible_owner and not replacement_reason:
                    raise RuntimeError(
                        f"worktree path {target} is already owned by "
                        f"{existing['board_id']}:{existing['task_id']}; "
                        "record an explicit workspace replacement reason"
                    )
                if exact_compatible_owner:
                    conn.execute("COMMIT")
                    return result_for(
                        conn, existing["workspace_id"], token=None, owns=False
                    )
                raise RuntimeError(
                    f"replacement reason does not authorize stealing an existing "
                    f"workspace path: {target}; choose a new path"
                )

            workspace_id = "w_" + uuid.uuid4().hex[:16]
            conn.execute(
                "INSERT INTO workspaces ("
                "workspace_id, repo_id, repo_path, workspace_path, task_id, board_id, "
                "owner_profile, purpose, branch, status, cleanup_policy, "
                "retention_condition, replacement_reason, reservation_token, "
                "reservation_expires_at, created_at, last_used_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?, ?, ?, ?, ?, ?, ?)",
                (
                    workspace_id,
                    rid,
                    repo,
                    target,
                    task_id,
                    board_id,
                    owner_profile,
                    purpose,
                    branch,
                    cleanup_policy,
                    retention_condition,
                    replacement_reason,
                    claim_token,
                    claim_expires,
                    now,
                    now,
                ),
            )
            conn.execute("COMMIT")
            return result_for(
                conn, workspace_id, token=claim_token, owns=True
            )
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise


def get_workspace_record(workspace_id: str) -> WorkspaceRecord:
    with contextlib.closing(connect_registry()) as conn:
        row = conn.execute(
            "SELECT * FROM workspaces WHERE workspace_id = ?", (workspace_id,)
        ).fetchone()
    if row is None:
        raise RuntimeError(f"unknown workspace reservation: {workspace_id}")
    return WorkspaceRecord.from_row(row)


def renew_reservation(
    workspace_id: str, reservation_token: Optional[str]
) -> None:
    """Renew a materializer's lease immediately before Git mutation."""
    if reservation_token is None:
        raise RuntimeError(f"workspace reservation {workspace_id} has no owner token")
    now = int(time.time())
    with contextlib.closing(connect_registry()) as conn:
        cur = conn.execute(
            "UPDATE workspaces SET reservation_expires_at = ?, last_used_at = ? "
            "WHERE workspace_id = ? AND status = 'reserved' "
            "AND reservation_token = ?",
            (
                now + RESERVATION_LEASE_SECONDS,
                now,
                workspace_id,
                reservation_token,
            ),
        )
    if cur.rowcount != 1:
        raise RuntimeError(
            f"workspace reservation {workspace_id} is no longer owned by this resolver"
        )


def verify_workspace(
    workspace_id: str,
    workspace_path: Path | str,
    *,
    reservation_token: Optional[str] = None,
) -> WorkspaceRecord:
    """Verify Git facts and activate a reservation with compare-and-swap safety.

    The reservation owner supplies its token and is the only caller allowed to
    promote ``reserved`` to ``active``. Non-owners may reverify an already-active
    checkout, but cannot infer materialization completion from Git metadata that
    becomes visible before ``git worktree add`` exits.
    """
    path = _canonical(workspace_path)
    now = int(time.time())
    with contextlib.closing(connect_registry()) as conn:
        row = conn.execute(
            "SELECT * FROM workspaces WHERE workspace_id = ?", (workspace_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"unknown workspace reservation: {workspace_id}")
        if row["status"] not in {"reserved", "active"}:
            raise RuntimeError(
                f"workspace reservation {workspace_id} is not reservable or active"
            )
        if row["workspace_path"] != path:
            raise RuntimeError(
                f"workspace reservation {workspace_id} path mismatch: expected "
                f"{row['workspace_path']}, found {path}"
            )
        if row["status"] == "reserved":
            if reservation_token is None:
                raise WorkspaceReservationPending(
                    f"workspace reservation {workspace_id} is still materializing"
                )
            if row["reservation_token"] != reservation_token:
                raise RuntimeError(
                    f"workspace reservation {workspace_id} is owned by another resolver"
                )

        head = _git(path, "rev-parse", "HEAD")
        actual_branch = _git(path, "branch", "--show-current") or None
        missing_expected_branch = row["branch"] is not None and actual_branch is None
        if head is None or missing_expected_branch:
            if row["status"] == "reserved":
                raise WorkspaceReservationPending(
                    f"workspace reservation {workspace_id} is not materialized yet"
                )
            raise RuntimeError(
                f"active workspace reservation {workspace_id} is not a usable checkout"
            )
        actual_repo = _canonical_repo(path)
        if actual_repo != row["repo_path"]:
            raise RuntimeError(
                f"workspace reservation {workspace_id} repository mismatch: expected "
                f"{row['repo_path']}, found {actual_repo}"
            )
        if row["branch"] is not None and actual_branch != row["branch"]:
            raise RuntimeError(
                f"workspace reservation {workspace_id} branch mismatch: expected "
                f"{row['branch']!r}, found {actual_branch!r}"
            )
        dirty_hash = dirty_manifest_hash(path)
        size = estimated_bytes(path)

        conn.execute("BEGIN IMMEDIATE")
        try:
            current = conn.execute(
                "SELECT * FROM workspaces WHERE workspace_id = ?", (workspace_id,)
            ).fetchone()
            if current is None or current["status"] not in {"reserved", "active"}:
                raise RuntimeError(
                    f"workspace reservation {workspace_id} is not reservable or active"
                )
            if (
                current["status"] == "reserved"
                and reservation_token is not None
                and current["reservation_token"] != reservation_token
            ):
                raise RuntimeError(
                    f"workspace reservation {workspace_id} is owned by another resolver"
                )
            cur = conn.execute(
                "UPDATE workspaces SET head_sha = ?, status = 'active', "
                "dirty_manifest_hash = ?, estimated_bytes = ?, last_used_at = ?, "
                "last_verified_at = ?, reservation_token = NULL, "
                "reservation_expires_at = NULL "
                "WHERE workspace_id = ? AND status IN ('reserved', 'active')",
                (head, dirty_hash, size, now, now, workspace_id),
            )
            if cur.rowcount != 1:
                raise RuntimeError(
                    f"workspace reservation {workspace_id} changed during verification"
                )
            verified = conn.execute(
                "SELECT * FROM workspaces WHERE workspace_id = ?", (workspace_id,)
            ).fetchone()
            conn.execute("COMMIT")
            return WorkspaceRecord.from_row(verified)
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise


def mark_creation_failed(
    workspace_id: str, reservation_token: Optional[str] = None
) -> None:
    """Fail only the still-reserved row owned by this materializer."""
    with contextlib.closing(connect_registry()) as conn:
        if reservation_token is None:
            conn.execute(
                "UPDATE workspaces SET status = 'creation_failed', "
                "last_verified_at = ?, reservation_expires_at = NULL "
                "WHERE workspace_id = ? AND status = 'reserved' "
                "AND reservation_token IS NULL",
                (int(time.time()), workspace_id),
            )
        else:
            conn.execute(
                "UPDATE workspaces SET status = 'creation_failed', "
                "last_verified_at = ?, reservation_token = NULL, "
                "reservation_expires_at = NULL WHERE workspace_id = ? "
                "AND status = 'reserved' AND reservation_token = ?",
                (int(time.time()), workspace_id, reservation_token),
            )


def _parse_disposition(value: object) -> tuple[str, Optional[str]]:
    text = str(value or "").strip()
    if text == "retired":
        return "retired", None
    prefix, separator, detail = text.partition(":")
    if prefix not in VALID_DISPOSITION_PREFIXES or prefix == "retired":
        raise RuntimeError(
            "workspace disposition must be one of: retired, "
            "retained_until:<condition>, preserved_dirty:<reason>, or "
            "operational_exception:<policy-id>"
        )
    if not separator or not detail.strip():
        raise RuntimeError(f"workspace disposition {prefix} requires a non-empty detail")
    return prefix, detail.strip()


def _configured_exception_path() -> Optional[Path]:
    try:
        from hermes_cli.config import load_config

        raw = (
            load_config()
            .get("kanban", {})
            .get("worktree_lifecycle", {})
            .get("exception_config", "")
        )
    except Exception:
        return None
    text = os.path.expandvars(str(raw or "").strip())
    return Path(text).expanduser() if text else None


def _validate_recovery_detail(record: WorkspaceRecord, detail: str) -> None:
    fields: dict[str, str] = {}
    for item in detail.split(";"):
        key, separator, value = item.partition("=")
        if separator and key.strip() and value.strip():
            fields[key.strip()] = value.strip()
    artifact = fields.get("artifact", "")
    owner = fields.get("owner", "")
    if not artifact or not owner:
        raise RuntimeError(
            "preserved_dirty requires artifact=<recovery artifact>;owner=<responsible owner>"
        )
    artifact_path = Path(artifact).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = Path(record.workspace_path) / artifact_path
    workspace_path = Path(record.workspace_path).expanduser()
    try:
        artifact_lexical = Path(os.path.abspath(artifact_path))
        workspace_lexical = Path(os.path.abspath(workspace_path))
        artifact_resolved = artifact_path.resolve(strict=True)
        workspace_resolved = workspace_path.resolve(strict=False)
        inside_workspace = (
            artifact_lexical.is_relative_to(workspace_lexical)
            or artifact_resolved.is_relative_to(workspace_resolved)
        )
    except (OSError, RuntimeError):
        inside_workspace = True
    if (
        inside_workspace
        or artifact_path.is_symlink()
        or not artifact_path.is_file()
    ):
        raise RuntimeError(
            "preserved_dirty recovery artifact must be an independently durable "
            f"regular file outside disposable worktree {record.workspace_path}: "
            f"{artifact_path}"
        )


def _validate_operational_exception(
    record: WorkspaceRecord,
    policy_id: str,
    *,
    now: int,
) -> Optional[int]:
    config_path = _configured_exception_path()
    if config_path is None:
        raise RuntimeError(
            f"operational_exception:{policy_id} requires a current operational "
            "exception policy configured at kanban.worktree_lifecycle.exception_config"
        )
    policies = {
        str(policy["policy_id"]): policy for policy in load_exception_config(config_path)
    }
    policy = policies.get(policy_id)
    if policy is None or _exception_expired(policy, now=now):
        raise RuntimeError(
            f"operational_exception:{policy_id} does not name a current operational "
            "exception policy"
        )
    if (
        str(policy["repo_path"]) != _canonical_repo(record.repo_path)
        or str(policy["workspace_path"]) != record.workspace_path
    ):
        raise RuntimeError(
            f"operational_exception:{policy_id} policy does not match workspace "
            f"{record.workspace_path}"
        )
    return _exception_expires_at(policy)


def _normalize_pr_numbers(value: object) -> Optional[tuple[int, ...]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple, set)):
        raise RuntimeError("workspace_pr_numbers must be a list of positive integers")
    numbers: set[int] = set()
    for raw in value:
        try:
            number = int(str(raw))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "workspace_pr_numbers must be a list of positive integers"
            ) from exc
        if number <= 0:
            raise RuntimeError("workspace_pr_numbers must contain positive integers")
        numbers.add(number)
    return tuple(sorted(numbers))


def prepare_terminal_disposition(
    *,
    task_id: str,
    board_id: str,
    disposition: object,
    pr_numbers: object = None,
) -> list[_TerminalDispositionPlan]:
    """Validate terminal facts without mutating either lifecycle database."""
    owned = [
        record for record in list_workspace_records(task_id=task_id)
        if record.board_id == board_id and record.status != "creation_failed"
    ]
    if not owned:
        return []
    normalized_prs = _normalize_pr_numbers(pr_numbers)
    by_id: dict[str, object]
    if disposition is None or disposition == "":
        if any(not record.disposition for record in owned):
            raise RuntimeError(
                f"task {task_id} owns {len(owned)} worktree(s); metadata must include "
                "workspace_disposition (retired, retained_until:<condition>, "
                "preserved_dirty:artifact=<path>;owner=<owner>, or "
                "operational_exception:<policy-id>)"
            )
        by_id = {
            record.workspace_id: str(record.disposition) for record in owned
        }
    elif isinstance(disposition, dict):
        by_id = {str(key): value for key, value in disposition.items()}
        missing = sorted(
            record.workspace_id for record in owned
            if record.workspace_id not in by_id
        )
        extra = sorted(set(by_id) - {record.workspace_id for record in owned})
        if missing or extra:
            raise RuntimeError(
                "workspace_disposition mapping must cover exactly the task's owned "
                f"workspace ids; missing={missing}, unknown={extra}"
            )
    else:
        by_id = {record.workspace_id: disposition for record in owned}

    plans: list[_TerminalDispositionPlan] = []
    now = int(time.time())
    for record in owned:
        raw_text = str(by_id[record.workspace_id]).strip()
        kind, detail = _parse_disposition(raw_text)
        exists = Path(record.workspace_path).exists()
        if kind == "retired":
            if exists:
                status_raw = _git(
                    record.workspace_path,
                    "status",
                    "--porcelain=v1",
                    "-z",
                    "--untracked-files=all",
                )
                if status_raw is None:
                    raise RuntimeError(
                        f"could not verify Git status for {record.workspace_path}; "
                        "refusing terminal disposition"
                    )
                if bool(status_raw):
                    raise RuntimeError(
                        f"dirty worktree {record.workspace_path} cannot be marked retired; "
                        "use preserved_dirty with a recovery artifact and owner, or remove "
                        "it safely first"
                    )
                raise RuntimeError(
                    f"worktree {record.workspace_path} still exists and cannot be marked "
                    "retired; use retained_until:<condition> or remove it safely first"
                )
            worktree_list = _git(record.repo_path, "worktree", "list", "--porcelain")
            if worktree_list is None:
                raise RuntimeError(
                    f"could not verify Git worktree registration for {record.repo_path}; "
                    f"refusing to retire {record.workspace_path}"
                )
            registered_paths = {
                _canonical(line.removeprefix("worktree "))
                for line in worktree_list.splitlines()
                if line.startswith("worktree ")
            }
            if _canonical(record.workspace_path) in registered_paths:
                raise RuntimeError(
                    f"worktree {record.workspace_path} is absent on disk but still "
                    "registered by Git; remove the registration safely before marking retired"
                )
            status_raw = ""
            head_sha = record.head_sha
            branch = record.branch
            workspace_bytes = record.estimated_bytes
        else:
            if not exists:
                raise RuntimeError(
                    f"could not verify Git status for missing workspace "
                    f"{record.workspace_path}; only a Git-confirmed retired disposition is valid"
                )
            status_raw = _git(
                record.workspace_path,
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
            )
            if status_raw is None:
                raise RuntimeError(
                    f"could not verify Git status for {record.workspace_path}; "
                    "refusing terminal disposition"
                )
            head_sha = _git(record.workspace_path, "rev-parse", "HEAD")
            if head_sha is None:
                raise RuntimeError(
                    f"could not verify Git head for {record.workspace_path}; "
                    "refusing terminal disposition"
                )
            branch_raw = _git(
                record.workspace_path, "rev-parse", "--abbrev-ref", "HEAD"
            )
            if branch_raw is None:
                raise RuntimeError(
                    f"could not verify Git branch for {record.workspace_path}; "
                    "refusing terminal disposition"
                )
            branch = None if branch_raw == "HEAD" else branch_raw
            workspace_bytes = estimated_bytes(record.workspace_path)

        is_dirty = bool(status_raw)
        if kind == "preserved_dirty" and not is_dirty:
            raise RuntimeError(
                f"workspace {record.workspace_path} is clean; preserved_dirty would be "
                "an inaccurate disposition"
            )
        if kind == "preserved_dirty":
            _validate_recovery_detail(record, detail or "")
        if kind == "retained_until" and is_dirty:
            raise RuntimeError(
                f"dirty worktree {record.workspace_path} requires "
                "preserved_dirty:artifact=<path>;owner=<owner>; retained_until would "
                "hide recovery-owned work"
            )

        exception_expires_at = record.exception_expires_at
        if kind == "operational_exception":
            exception_expires_at = _validate_operational_exception(
                record, detail or "", now=now
            )
        if kind == "retired":
            lifecycle_status = "retired"
            retention_condition = None
        elif kind == "retained_until":
            lifecycle_status = (
                "retirement_queued" if detail == "janitor"
                else ("terminal_dirty" if is_dirty else "terminal_clean")
            )
            retention_condition = detail
        elif kind == "preserved_dirty":
            lifecycle_status = "terminal_dirty"
            retention_condition = "recovery_owned"
        else:
            lifecycle_status = "protected"
            retention_condition = "operational_exception"
        manifest = (
            hashlib.sha256((status_raw or "").encode("utf-8")).hexdigest()
            if status_raw is not None else None
        )
        plans.append(
            _TerminalDispositionPlan(
                record=record,
                disposition=raw_text,
                status=lifecycle_status,
                dirty_manifest_hash=manifest,
                exception_policy_id=(
                    detail if kind == "operational_exception"
                    else record.exception_policy_id
                ),
                retention_condition=retention_condition,
                head_sha=head_sha,
                branch=branch,
                estimated_bytes=workspace_bytes,
                pr_numbers=(
                    normalized_prs if normalized_prs is not None else record.pr_numbers
                ),
                exception_expires_at=exception_expires_at,
                verified_at=now,
            )
        )
    return plans


def _intent_rows(
    conn: sqlite3.Connection,
    *,
    operation_id: Optional[str] = None,
    board_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> list[sqlite3.Row]:
    clauses: list[str] = []
    params: list[object] = []
    if operation_id is not None:
        clauses.append("operation_id = ?")
        params.append(operation_id)
    if board_id is not None:
        clauses.append("board_id = ?")
        params.append(board_id)
    if task_id is not None:
        clauses.append("task_id = ?")
        params.append(task_id)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    return conn.execute(
        "SELECT * FROM terminal_disposition_intents"
        + where
        + " ORDER BY operation_id, workspace_id",
        tuple(params),
    ).fetchall()


def abort_terminal_disposition_intent(operation_id: str) -> int:
    """Restore staged rows after the task transaction definitely rolled back."""
    with contextlib.closing(connect_registry()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            rows = _intent_rows(conn, operation_id=operation_id)
            for row in rows:
                restored = conn.execute(
                    "UPDATE workspaces SET status = ? WHERE workspace_id = ? "
                    "AND board_id = ? AND task_id = ? AND repo_id = ? "
                    "AND workspace_path = ? AND status = 'terminalizing'",
                    (
                        row["expected_status"],
                        row["workspace_id"],
                        row["board_id"],
                        row["task_id"],
                        row["expected_repo_id"],
                        row["expected_workspace_path"],
                    ),
                )
                if restored.rowcount != 1:
                    raise RuntimeError(
                        f"workspace {row['workspace_id']} changed while aborting "
                        "terminal disposition"
                    )
            cur = conn.execute(
                "DELETE FROM terminal_disposition_intents WHERE operation_id = ?",
                (operation_id,),
            )
            if cur.rowcount != len(rows):
                raise RuntimeError(
                    f"terminal intent {operation_id} changed while aborting"
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return cur.rowcount


def stage_terminal_disposition_intent(
    plans: list[_TerminalDispositionPlan],
    *,
    terminal_status: str,
    task_db_path: Path | str,
) -> Optional[str]:
    """Durably classify workspaces before the WAL task commit can begin.

    The intent lives entirely in the shared registry database, so its commit is
    crash-atomic without relying on SQLite multi-database ATTACH semantics.
    Recovery later decides whether to finalize or abort it from the task DB's
    durable status.
    """
    if not plans:
        return None
    if terminal_status not in {"done", "archived"}:
        raise ValueError(f"unsupported terminal intent status: {terminal_status}")
    ownership = {
        (plan.record.board_id, plan.record.task_id) for plan in plans
    }
    if len(ownership) != 1:
        raise RuntimeError("terminal disposition plans must share one board and task")
    board_id, task_id = next(iter(ownership))
    if not board_id or not task_id:
        raise RuntimeError("terminal disposition plans require board and task ownership")
    canonical_task_db_path = _canonical(task_db_path)

    # Resolve any prior crash before reserving these workspace ids. A live
    # concurrent operation remains pending under its PID/lease and therefore
    # keeps the UNIQUE(workspace_id) guard fail-closed.
    recover_terminal_disposition_intents(board_id=board_id, task_id=task_id)

    operation_id = "td_" + uuid.uuid4().hex
    now = int(time.time())
    owner_host = socket.gethostname() or "unknown"
    owner_pid = os.getpid()
    try:
        owner_started_at: Optional[float] = float(
            psutil.Process(owner_pid).create_time()
        )
    except Exception:
        # Missing liveness evidence must fail closed during recovery. psutil is
        # a core dependency, but preserving an intent is safer than guessing.
        owner_started_at = None
    expires_at = now + TERMINAL_INTENT_LEASE_SECONDS
    with contextlib.closing(connect_registry()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            for plan in plans:
                record = plan.record
                current = conn.execute(
                    "SELECT repo_id, workspace_path, board_id, task_id, status "
                    "FROM workspaces WHERE workspace_id = ?",
                    (record.workspace_id,),
                ).fetchone()
                expected = (
                    record.repo_id,
                    record.workspace_path,
                    record.board_id,
                    record.task_id,
                    record.status,
                )
                if current is None or tuple(current) != expected:
                    raise RuntimeError(
                        f"workspace {record.workspace_id} changed while staging "
                        "terminal disposition"
                    )
                staged = conn.execute(
                    "UPDATE workspaces SET status = 'terminalizing' "
                    "WHERE workspace_id = ? AND board_id = ? AND task_id = ? "
                    "AND repo_id = ? AND workspace_path = ? AND status = ?",
                    (
                        record.workspace_id,
                        record.board_id,
                        record.task_id,
                        record.repo_id,
                        record.workspace_path,
                        record.status,
                    ),
                )
                if staged.rowcount != 1:
                    raise RuntimeError(
                        f"workspace {record.workspace_id} changed while staging "
                        "terminal disposition"
                    )
                conn.execute(
                    "INSERT INTO terminal_disposition_intents ("
                    "operation_id, workspace_id, board_id, task_id, terminal_status, "
                    "task_db_path, "
                    "expected_repo_id, expected_workspace_path, expected_status, "
                    "disposition, target_status, dirty_manifest_hash, "
                    "exception_policy_id, exception_expires_at, retention_condition, "
                    "head_sha, branch, pr_numbers, estimated_bytes, verified_at, "
                    "owner_host, owner_pid, owner_started_at, expires_at, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "?, ?, ?, ?, ?, ?, ?)",
                    (
                        operation_id,
                        record.workspace_id,
                        board_id,
                        task_id,
                        terminal_status,
                        canonical_task_db_path,
                        record.repo_id,
                        record.workspace_path,
                        record.status,
                        plan.disposition,
                        plan.status,
                        plan.dirty_manifest_hash,
                        plan.exception_policy_id,
                        plan.exception_expires_at,
                        plan.retention_condition,
                        plan.head_sha,
                        plan.branch,
                        ",".join(str(number) for number in plan.pr_numbers),
                        plan.estimated_bytes,
                        plan.verified_at,
                        owner_host,
                        owner_pid,
                        owner_started_at,
                        expires_at,
                        now,
                    ),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return operation_id


def _intent_owner_is_alive(row: sqlite3.Row) -> Optional[bool]:
    if str(row["owner_host"]) != (socket.gethostname() or "unknown"):
        return None
    try:
        process = psutil.Process(int(row["owner_pid"]))
        if process.status() == psutil.STATUS_ZOMBIE:
            return False
        expected_started_at = row["owner_started_at"]
        if expected_started_at is None:
            return None
        return abs(process.create_time() - float(expected_started_at)) <= 0.01
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except Exception:
        # Unknown liveness remains pending and fail-closed. A lease timestamp is
        # audit evidence, not permission to abort a possibly live transition.
        return None


def finalize_terminal_disposition_intent(
    operation_id: str,
    *,
    abort_nonterminal: bool = True,
) -> str:
    """Resolve one intent to ``finalized``, ``aborted``, or ``pending``."""
    with contextlib.closing(connect_registry()) as registry:
        rows = _intent_rows(registry, operation_id=operation_id)
    if not rows:
        return "absent"
    identities = {
        (
            str(row["board_id"]),
            str(row["task_id"]),
            str(row["terminal_status"]),
            str(row["task_db_path"] or ""),
        )
        for row in rows
    }
    if len(identities) != 1:
        raise RuntimeError(f"terminal intent {operation_id} has mixed ownership")
    board_id, task_id, _terminal_status, task_db_path = next(iter(identities))
    if not task_db_path or not Path(task_db_path).is_file():
        # Never let recovery create or guess a board DB. The exact path is part
        # of the durable intent because ambient HERMES_KANBAN_DB can override
        # even an explicit board selector.
        return "pending"

    from hermes_cli import kanban_db

    try:
        with kanban_db.connect_closing(
            db_path=Path(task_db_path), board=board_id
        ) as task_conn:
            task_row = task_conn.execute(
                "SELECT status, terminal_disposition_operation_id "
                "FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
    except Exception as exc:
        raise RuntimeError(
            f"cannot reconcile terminal intent {operation_id} against board {board_id}"
        ) from exc
    if task_row is None:
        # The exact task DB is part of the intent, so a missing row means the
        # terminal transition did not survive (or an explicit delete won the
        # race). Abort only when this caller knows the task txn failed or the
        # original process is dead; a live owner remains pending fail-closed.
        owner_alive = _intent_owner_is_alive(rows[0])
        if abort_nonterminal or owner_alive is False:
            abort_terminal_disposition_intent(operation_id)
            return "aborted"
        return "pending"

    task_committed_this_intent = (
        str(task_row["status"]) == _terminal_status
        and task_row["terminal_disposition_operation_id"] == operation_id
    )
    if not task_committed_this_intent:
        owner_alive = _intent_owner_is_alive(rows[0])
        if abort_nonterminal or owner_alive is False:
            abort_terminal_disposition_intent(operation_id)
            return "aborted"
        return "pending"

    # The task commit is durable. Finalize every registry row plus intent
    # deletion in one single-database transaction. A crash during this COMMIT
    # therefore leaves either the complete intent or the complete disposition.
    with contextlib.closing(connect_registry()) as registry:
        registry.execute("BEGIN IMMEDIATE")
        try:
            current_rows = _intent_rows(registry, operation_id=operation_id)
            if not current_rows:
                # Another recovery boundary resolved the complete operation
                # while this caller was reading task evidence. Intent deletion
                # is atomic with disposition finalization/abort, so an empty
                # re-read is an idempotent already-resolved outcome.
                registry.execute("COMMIT")
                return "absent"
            if len(current_rows) != len(rows):
                raise RuntimeError(
                    f"terminal intent {operation_id} changed during recovery"
                )
            for row in current_rows:
                cur = registry.execute(
                    "UPDATE workspaces SET disposition = ?, status = ?, "
                    "dirty_manifest_hash = ?, exception_policy_id = ?, "
                    "exception_expires_at = ?, retention_condition = ?, head_sha = ?, "
                    "branch = ?, pr_numbers = ?, estimated_bytes = ?, last_used_at = ?, "
                    "last_verified_at = ? WHERE workspace_id = ? AND board_id = ? "
                    "AND task_id = ? AND repo_id = ? AND workspace_path = ? "
                    "AND status = 'terminalizing'",
                    (
                        row["disposition"],
                        row["target_status"],
                        row["dirty_manifest_hash"],
                        row["exception_policy_id"],
                        row["exception_expires_at"],
                        row["retention_condition"],
                        row["head_sha"],
                        row["branch"],
                        row["pr_numbers"],
                        row["estimated_bytes"],
                        row["verified_at"],
                        row["verified_at"],
                        row["workspace_id"],
                        row["board_id"],
                        row["task_id"],
                        row["expected_repo_id"],
                        row["expected_workspace_path"],
                    ),
                )
                if cur.rowcount != 1:
                    raise RuntimeError(
                        f"workspace {row['workspace_id']} changed during terminal recovery"
                    )
            deleted = registry.execute(
                "DELETE FROM terminal_disposition_intents WHERE operation_id = ?",
                (operation_id,),
            )
            if deleted.rowcount != len(current_rows):
                raise RuntimeError(
                    f"terminal intent {operation_id} changed before deletion"
                )
            registry.execute("COMMIT")
        except Exception:
            registry.execute("ROLLBACK")
            raise
    return "finalized"


def recover_terminal_disposition_intents(
    *,
    board_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> dict[str, list[str]]:
    """Deterministically reconcile crash-left terminal disposition intents."""
    with contextlib.closing(connect_registry()) as registry:
        rows = _intent_rows(registry, board_id=board_id, task_id=task_id)
    operation_ids = sorted({str(row["operation_id"]) for row in rows})
    report: dict[str, list[str]] = {
        "finalized": [],
        "aborted": [],
        "pending": [],
    }
    for operation_id in operation_ids:
        outcome = finalize_terminal_disposition_intent(
            operation_id, abort_nonterminal=False
        )
        if outcome in report:
            report[outcome].append(operation_id)
    return report


def has_terminal_disposition_intents(*, board_id: str, task_id: str) -> bool:
    """Return whether deletion must preserve task evidence for recovery."""
    with contextlib.closing(connect_registry()) as registry:
        row = registry.execute(
            "SELECT 1 FROM terminal_disposition_intents "
            "WHERE board_id = ? AND task_id = ? LIMIT 1",
            (board_id, task_id),
        ).fetchone()
    return row is not None


@contextlib.contextmanager
def task_evidence_deletion_guard(*, board_id: str, task_id: str):
    """Serialize hard deletion against lifecycle registration and intents.

    The registry write lock is held across the caller's task-DB transaction, so
    a new workspace row or terminal intent cannot appear between validation and
    task deletion. ``True`` means durable task evidence is still required.
    """
    with contextlib.closing(connect_registry()) as registry:
        registry.execute("BEGIN IMMEDIATE")
        try:
            intent = registry.execute(
                "SELECT 1 FROM terminal_disposition_intents "
                "WHERE board_id = ? AND task_id = ? LIMIT 1",
                (board_id, task_id),
            ).fetchone()
            workspace = registry.execute(
                "SELECT 1 FROM workspaces WHERE board_id = ? AND task_id = ? "
                "AND status NOT IN ('retired', 'creation_failed') LIMIT 1",
                (board_id, task_id),
            ).fetchone()
            yield intent is not None or workspace is not None
            registry.execute("COMMIT")
        except BaseException:
            if registry.in_transaction:
                registry.execute("ROLLBACK")
            raise


@contextlib.contextmanager
def terminal_task_transaction(
    conn: sqlite3.Connection,
    plans: list[_TerminalDispositionPlan],
    *,
    terminal_status: str,
):
    """Wrap a task write transaction in the durable intent protocol."""
    from hermes_cli import kanban_db

    task_db_path = ""
    for row in conn.execute("PRAGMA database_list").fetchall():
        if str(row[1]) == "main":
            task_db_path = str(row[2] or "")
            break
    if plans and not task_db_path:
        raise RuntimeError("terminal disposition requires a file-backed task database")
    operation_id = stage_terminal_disposition_intent(
        plans,
        terminal_status=terminal_status,
        task_db_path=task_db_path,
    )
    try:
        with kanban_db.write_txn(conn):
            yield operation_id
    except BaseException:
        # write_txn handles ordinary Exception, but BaseException subclasses
        # such as KeyboardInterrupt/SystemExit bypass its rollback handler.
        # Never reconcile registry evidence while uncommitted task changes are
        # still visible on this connection.
        if conn.in_transaction:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                _log.exception(
                    "task transaction for terminal intent %s could not roll back",
                    operation_id,
                )
        if operation_id is not None and not conn.in_transaction:
            try:
                # Inspect durable task state rather than assuming COMMIT failed;
                # write_txn can surface a post-commit invariant error.
                finalize_terminal_disposition_intent(operation_id)
            except Exception:
                _log.exception(
                    "terminal intent %s remains for deterministic recovery",
                    operation_id,
                )
        elif operation_id is not None:
            _log.error(
                "terminal intent %s remains pending because task transaction "
                "state is uncertain",
                operation_id,
            )
        raise
    else:
        if operation_id is not None:
            outcome = finalize_terminal_disposition_intent(operation_id)
            if outcome == "pending":
                raise RuntimeError(
                    f"terminal intent {operation_id} could not be reconciled"
                )


def _apply_terminal_plans(
    conn: sqlite3.Connection,
    plans: list[_TerminalDispositionPlan],
    *,
    table: str,
) -> None:
    for plan in plans:
        cur = conn.execute(
            f"UPDATE {table} SET disposition = ?, status = ?, "
            "dirty_manifest_hash = ?, exception_policy_id = ?, "
            "exception_expires_at = ?, retention_condition = ?, head_sha = ?, "
            "branch = ?, pr_numbers = ?, estimated_bytes = ?, last_used_at = ?, "
            "last_verified_at = ? WHERE workspace_id = ? AND board_id = ? "
            "AND task_id = ? AND repo_id = ? AND workspace_path = ? AND status = ?",
            (
                plan.disposition,
                plan.status,
                plan.dirty_manifest_hash,
                plan.exception_policy_id,
                plan.exception_expires_at,
                plan.retention_condition,
                plan.head_sha,
                plan.branch,
                ",".join(str(number) for number in plan.pr_numbers),
                plan.estimated_bytes,
                plan.verified_at,
                plan.verified_at,
                plan.record.workspace_id,
                plan.record.board_id,
                plan.record.task_id,
                plan.record.repo_id,
                plan.record.workspace_path,
                plan.record.status,
            ),
        )
        if cur.rowcount != 1:
            raise RuntimeError(
                f"workspace {plan.record.workspace_id} changed during terminal transition"
            )


def apply_terminal_disposition(
    *,
    task_id: str,
    board_id: str,
    disposition: object,
    pr_numbers: object = None,
) -> list[WorkspaceRecord]:
    """Compatibility wrapper for registry-only callers."""
    plans = prepare_terminal_disposition(
        task_id=task_id,
        board_id=board_id,
        disposition=disposition,
        pr_numbers=pr_numbers,
    )
    with contextlib.closing(connect_registry()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            _apply_terminal_plans(conn, plans, table="workspaces")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return [
        record for record in list_workspace_records(task_id=task_id)
        if record.board_id == board_id and record.status != "creation_failed"
    ]


def _git_worktrees(repo_path: Path | str) -> list[dict[str, object]]:
    raw = _git(repo_path, "worktree", "list", "--porcelain")
    if raw is None:
        return []
    entries: list[dict[str, object]] = []
    current: dict[str, object] = {}
    for line in raw.splitlines() + [""]:
        if not line:
            if current:
                current["is_primary"] = not entries
                current["workspace_path"] = _canonical(str(current["workspace_path"]))
                entries.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            current["workspace_path"] = value
        elif key == "HEAD":
            current["head_sha"] = value
        elif key == "branch":
            current["branch"] = value.removeprefix("refs/heads/")
        elif key in {"detached", "bare", "locked", "prunable"}:
            current[key] = value or True
    return entries


def _is_within(path: str, roots: Iterable[str]) -> bool:
    candidate = Path(path)
    for root in roots:
        try:
            candidate.relative_to(Path(root))
            return True
        except ValueError:
            continue
    return False


def _task_inventory() -> tuple[
    dict[tuple[str, str], object], set[str], set[tuple[str, str]]
]:
    """Read task lifecycle and dependency evidence from every board."""
    from hermes_cli import kanban_db

    tasks: dict[tuple[str, str], object] = {}
    worktree_paths: set[str] = set()
    retained_dependencies: set[tuple[str, str]] = set()
    for board in kanban_db.list_boards():
        slug = str(board["slug"])
        try:
            with kanban_db.connect_closing(board=slug) as conn:
                for task in kanban_db.list_tasks(
                    conn, include_archived=True, limit=100000
                ):
                    tasks[(slug, task.id)] = task
                    if task.workspace_kind == "worktree" and task.workspace_path:
                        worktree_paths.add(_canonical(task.workspace_path))
                for row in conn.execute(
                    "SELECT DISTINCT l.parent_id FROM task_links l "
                    "JOIN tasks child ON child.id = l.child_id "
                    "WHERE child.status NOT IN ('done', 'cancelled', 'archived')"
                ).fetchall():
                    retained_dependencies.add((slug, str(row["parent_id"])))
        except Exception:
            # A board that cannot be opened is itself an operational problem,
            # but inventory must stay useful for all other repositories.  The
            # CLI surfaces board-open failures separately.
            continue
    return tasks, worktree_paths, retained_dependencies


def load_exception_config(path: Path | str) -> list[dict[str, object]]:
    source = Path(path).expanduser()
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or data.get("version") != 1:
        raise RuntimeError("worktree exception config must be a version: 1 mapping")
    raw = data.get("exceptions", [])
    if not isinstance(raw, list):
        raise RuntimeError("worktree exception config 'exceptions' must be a list")
    policies: list[dict[str, object]] = []
    seen_policy_ids: set[str] = set()
    seen_workspace_paths: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise RuntimeError(f"exception #{index + 1} must be a mapping")
        policy_id = str(item.get("policy_id") or "").strip()
        repo_path = os.path.expandvars(str(item.get("repo_path") or "").strip())
        workspace_path = os.path.expandvars(
            str(item.get("workspace_path") or "").strip()
        )
        if not policy_id or not repo_path or not workspace_path:
            raise RuntimeError(
                f"exception #{index + 1} requires policy_id, repo_path, and workspace_path"
            )
        policy = dict(item)
        policy["policy_id"] = policy_id
        policy["repo_path"] = _canonical(repo_path)
        policy["workspace_path"] = _canonical(workspace_path)
        if policy_id in seen_policy_ids:
            raise RuntimeError(f"duplicate worktree exception policy_id: {policy_id}")
        canonical_workspace = str(policy["workspace_path"])
        if canonical_workspace in seen_workspace_paths:
            raise RuntimeError(
                f"duplicate worktree exception workspace_path: {canonical_workspace}"
            )
        seen_policy_ids.add(policy_id)
        seen_workspace_paths.add(canonical_workspace)
        policies.append(policy)
    return sorted(policies, key=lambda item: str(item["policy_id"]))


def _exception_expired(policy: dict[str, object], now: Optional[int] = None) -> bool:
    expires = _exception_expires_at(policy)
    return expires is not None and expires <= (
        int(time.time()) if now is None else now
    )


def _exception_expires_at(policy: dict[str, object]) -> Optional[int]:
    raw = policy.get("expires_at")
    if raw in (None, "", 0, "0"):
        return None
    try:
        expires = int(str(raw))
    except (TypeError, ValueError):
        try:
            from datetime import datetime

            expires = int(datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp())
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"invalid expires_at for exception {policy.get('policy_id')}: {raw!r}"
            ) from exc
    return expires


def import_protected_exceptions(config_path: Path | str) -> dict[str, object]:
    """Safely register existing physical worktrees from an explicit config."""
    imported: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for policy in load_exception_config(config_path):
        policy_id = str(policy["policy_id"])
        repo = _canonical_repo(str(policy["repo_path"]))
        workspace = str(policy["workspace_path"])
        if _exception_expired(policy):
            skipped.append({
                "policy_id": policy_id,
                "workspace_path": workspace,
                "reason": "expired_policy",
            })
            continue
        physical = {
            str(entry["workspace_path"]) for entry in _git_worktrees(repo)
        }
        if workspace not in physical:
            skipped.append({
                "policy_id": policy_id,
                "workspace_path": workspace,
                "reason": "not_a_git_worktree",
            })
            continue
        now = int(time.time())
        expires_at = _exception_expires_at(policy)
        head = _git(workspace, "rev-parse", "HEAD")
        branch = _git(workspace, "branch", "--show-current") or None
        manifest = dirty_manifest_hash(workspace)
        size = estimated_bytes(workspace)
        with contextlib.closing(connect_registry()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    "SELECT * FROM workspaces WHERE workspace_path = ?",
                    (workspace,),
                ).fetchone()
                if existing and existing["task_id"]:
                    conn.execute("ROLLBACK")
                    skipped.append({
                        "policy_id": policy_id,
                        "workspace_path": workspace,
                        "reason": "task_owned_registry_record",
                    })
                    continue
                if existing:
                    conn.execute(
                        "UPDATE workspaces SET repo_id=?, repo_path=?, purpose=?, branch=?, "
                        "head_sha=?, status='protected', cleanup_policy='never_automatic', "
                        "retention_condition='operational_exception', disposition=?, "
                        "exception_policy_id=?, exception_expires_at=?, "
                        "dirty_manifest_hash=?, estimated_bytes=?, last_used_at=?, "
                        "last_verified_at=? WHERE workspace_id=?",
                        (
                            _repo_id(repo), repo,
                            str(policy.get("purpose") or "operational_exception"),
                            branch, head, f"operational_exception:{policy_id}",
                            policy_id, expires_at, manifest, size, now, now,
                            existing["workspace_id"],
                        ),
                    )
                else:
                    conn.execute(
                        "INSERT INTO workspaces (workspace_id, repo_id, repo_path, "
                        "workspace_path, owner_profile, purpose, branch, head_sha, status, "
                        "cleanup_policy, retention_condition, disposition, "
                        "exception_policy_id, exception_expires_at, dirty_manifest_hash, estimated_bytes, "
                        "created_at, last_used_at, last_verified_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'protected', 'never_automatic', "
                        "'operational_exception', ?, ?, ?, ?, ?, ?, ?, ?)\n",
                        (
                            "w_" + uuid.uuid4().hex[:16], _repo_id(repo), repo, workspace,
                            str(policy.get("owner_profile") or "operator"),
                            str(policy.get("purpose") or "operational_exception"),
                            branch, head, f"operational_exception:{policy_id}", policy_id,
                            expires_at, manifest, size, now, now, now,
                        ),
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        imported.append({"policy_id": policy_id, "workspace_path": workspace})
    return {"imported": imported, "skipped": skipped}


def reconcile_inventory(
    *,
    repo_paths: Iterable[Path | str],
    approved_roots: Iterable[Path | str] = (),
    exception_config: Optional[Path | str] = None,
) -> dict[str, object]:
    """Return deterministic Git↔registry↔task reconciliation findings.

    The pass is report-only with respect to Git: it updates verification facts
    on known registry rows but never adopts, removes, prunes, or repairs a
    worktree. Unknown paths remain findings until an operator explicitly
    registers or retires them.
    """
    repos = sorted({_canonical_repo(path) for path in repo_paths})
    approved = sorted({_canonical(path) for path in approved_roots})
    repo_ids = {_repo_id(repo) for repo in repos}
    exception_policies = (
        {
            str(item["policy_id"]): item
            for item in load_exception_config(exception_config)
            if str(item["repo_path"]) in repos
        }
        if exception_config else {}
    )
    tasks, _task_paths, _retained_dependencies = _task_inventory()
    records = [
        record for record in list_workspace_records()
        if record.repo_id in repo_ids
    ]
    records_by_path = {
        record.workspace_path: record
        for record in records
        if record.status in ACTIVE_REGISTRY_STATUSES
    }
    physical_paths: set[str] = set()
    per_repo: list[dict[str, object]] = []
    git_entries: list[dict[str, object]] = []
    unowned: list[str] = []
    outside: list[str] = []

    for repo in repos:
        entries = _git_worktrees(repo)
        rid = _repo_id(repo)
        repo_bytes = 0
        for entry in entries:
            path = str(entry["workspace_path"])
            physical_paths.add(path)
            size = estimated_bytes(path)
            repo_bytes += size
            entry["repo_id"] = rid
            entry["repo_path"] = repo
            entry["estimated_bytes"] = size
            git_entries.append(entry)
            if not bool(entry.get("is_primary")):
                if path not in records_by_path:
                    unowned.append(path)
                if approved and not _is_within(path, approved):
                    outside.append(path)
        per_repo.append(
            {
                "repo_id": rid,
                "repo_path": repo,
                "worktree_count": len(entries),
                "estimated_bytes": repo_bytes,
            }
        )

    terminal_without: list[str] = []
    dirty_terminal: list[str] = []
    now = int(time.time())
    terminal_statuses = {"done", "cancelled", "archived"}
    with contextlib.closing(connect_registry()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            for record in records:
                if record.workspace_path not in physical_paths:
                    continue
                if record.status == "terminalizing":
                    # The durable intent owns this row until task-state recovery
                    # atomically finalizes or restores it. Inventory must not
                    # race that compare-and-swap snapshot.
                    continue
                task = tasks.get((record.board_id or "default", record.task_id or ""))
                task_status = getattr(task, "status", None)
                status_raw = _git(
                    record.workspace_path,
                    "status",
                    "--porcelain=v1",
                    "-z",
                    "--untracked-files=all",
                )
                is_dirty = bool(status_raw)
                head = _git(record.workspace_path, "rev-parse", "HEAD")
                branch = _git(record.workspace_path, "branch", "--show-current") or None
                size = estimated_bytes(record.workspace_path)
                lifecycle_status = record.status
                is_protected = (
                    record.status == "protected"
                    or (record.disposition or "").startswith(
                        "operational_exception:"
                    )
                    or bool(record.exception_policy_id)
                )
                if is_protected:
                    lifecycle_status = "protected"
                elif record.status == "retirement_queued":
                    # This status is durable disposition state, not a fresh
                    # cleanliness observation. Janitor independently rechecks
                    # task terminality, dirt, PRs, and dependencies, so keep
                    # the queue classification while those gates fail closed.
                    lifecycle_status = "retirement_queued"
                elif task_status in terminal_statuses:
                    lifecycle_status = "terminal_dirty" if is_dirty else "terminal_clean"
                    if not record.disposition:
                        terminal_without.append(record.workspace_path)
                    if is_dirty:
                        dirty_terminal.append(record.workspace_path)
                elif record.status not in {"protected", "blocked", "superseded"}:
                    lifecycle_status = "active"
                manifest = (
                    hashlib.sha256((status_raw or "").encode("utf-8")).hexdigest()
                    if status_raw is not None else None
                )
                # A terminal disposition is the cleanup drift baseline.
                # Inventory may observe later changes but must never bless them
                # by overwriting the recorded branch/head/manifest snapshot.
                preserve_terminal_snapshot = bool(record.disposition) and (
                    task_status in terminal_statuses or is_protected
                )
                conn.execute(
                    "UPDATE workspaces SET branch = COALESCE(?, branch), head_sha = ?, "
                    "status = ?, dirty_manifest_hash = ?, estimated_bytes = ?, "
                    "last_verified_at = ? WHERE workspace_id = ? AND status = ?",
                    (
                        record.branch if preserve_terminal_snapshot else branch,
                        record.head_sha if preserve_terminal_snapshot else head,
                        lifecycle_status,
                        (
                            record.dirty_manifest_hash
                            if preserve_terminal_snapshot
                            else manifest
                        ),
                        size,
                        now,
                        record.workspace_id,
                        record.status,
                    ),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    task_groups: dict[tuple[str, str, str], list[WorkspaceRecord]] = {}
    for record in records:
        if record.workspace_path not in physical_paths:
            continue
        if not record.task_id or not record.board_id:
            continue
        task_groups.setdefault(
            (record.board_id, record.task_id, record.repo_id), []
        ).append(record)
    duplicate_task_repo = []
    for (board_id, task_id, repo_id), group in sorted(task_groups.items()):
        if len(group) < 2:
            continue
        duplicate_task_repo.append(
            {
                "board_id": board_id,
                "task_id": task_id,
                "repo_id": repo_id,
                "workspace_ids": sorted(item.workspace_id for item in group),
                "paths": sorted(item.workspace_path for item in group),
            }
        )

    head_groups: dict[tuple[str, str], list[str]] = {}
    for entry in git_entries:
        head = str(entry.get("head_sha") or "")
        if head:
            head_groups.setdefault((str(entry["repo_id"]), head), []).append(
                str(entry["workspace_path"])
            )
    duplicate_heads = [
        {"repo_id": repo_id, "head_sha": head, "paths": sorted(paths)}
        for (repo_id, head), paths in sorted(head_groups.items())
        if len(paths) > 1
    ]

    protected_findings: list[dict[str, str]] = []
    for policy_id, policy in sorted(exception_policies.items()):
        workspace_path = str(policy["workspace_path"])
        if _exception_expired(policy, now=now):
            reason = "expired_policy"
        elif workspace_path not in physical_paths:
            reason = "missing_physical_worktree"
        else:
            record = records_by_path.get(workspace_path)
            if record is None:
                reason = "missing_registry_record"
            elif (
                record.status != "protected"
                or record.exception_policy_id != policy_id
                or record.disposition != f"operational_exception:{policy_id}"
            ):
                reason = "registry_policy_mismatch"
            else:
                continue
        protected_findings.append(
            {
                "policy_id": policy_id,
                "workspace_path": workspace_path,
                "reason": reason,
            }
        )

    for record in records:
        policy_id = record.exception_policy_id
        is_protected = (
            record.status == "protected"
            or (record.disposition or "").startswith("operational_exception:")
            or bool(policy_id)
        )
        if not is_protected or not policy_id:
            continue
        if (
            record.exception_expires_at is not None
            and record.exception_expires_at <= now
        ):
            reason = "expired_policy"
        elif exception_config is not None and policy_id not in exception_policies:
            reason = "missing_policy_config"
        else:
            continue
        finding = {
            "policy_id": policy_id,
            "workspace_path": record.workspace_path,
            "reason": reason,
        }
        if finding not in protected_findings:
            protected_findings.append(finding)
    protected_findings.sort(
        key=lambda item: (
            item["policy_id"], item["workspace_path"], item["reason"]
        )
    )

    return {
        "totals": {
            "repositories": len(per_repo),
            "git_worktrees": len(git_entries),
            "registry_records": len(records),
            "estimated_bytes": sum(
                int(str(repo["estimated_bytes"])) for repo in per_repo
            ),
        },
        "per_repo": per_repo,
        "unowned_paths": sorted(set(unowned)),
        "duplicate_task_repo": duplicate_task_repo,
        "duplicate_heads": duplicate_heads,
        "terminal_without_disposition": sorted(set(terminal_without)),
        "dirty_terminal": sorted(set(dirty_terminal)),
        "outside_approved_root": sorted(set(outside)),
        "missing_expired_protected_exceptions": protected_findings,
    }


def plan_janitor(
    *,
    repo_paths: Iterable[Path | str],
    approved_roots: Iterable[Path | str],
) -> dict[str, object]:
    """Build an auditable deletion plan without removing anything.

    There is intentionally no execute mode. Enabling deletion requires a
    separate owner-approved change after this dry-run has proven conservative
    against real inventories.
    """
    repos = sorted({_canonical_repo(path) for path in repo_paths})
    repo_ids = {_repo_id(path) for path in repos}
    approved = sorted({_canonical(path) for path in approved_roots})
    records = [
        record for record in list_workspace_records()
        if record.repo_id in repo_ids and record.status != "creation_failed"
    ]
    tasks, _task_paths, retained_dependencies = _task_inventory()
    terminal_task_statuses = {"done", "cancelled", "archived"}
    physical_entries = {
        str(entry["workspace_path"]): entry
        for repo in repos
        for entry in _git_worktrees(repo)
    }
    physical = set(physical_entries)
    owner_counts: dict[tuple[Optional[str], Optional[str], str], int] = {}
    for record in records:
        key = (record.board_id, record.task_id, record.repo_id)
        owner_counts[key] = owner_counts.get(key, 0) + 1

    candidates: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []
    for record in sorted(records, key=lambda item: (item.workspace_path, item.workspace_id)):
        reasons: list[str] = []
        task = tasks.get((record.board_id or "default", record.task_id or ""))
        if getattr(task, "status", None) not in terminal_task_statuses:
            reasons.append("task_not_terminal")
        if record.status == "terminalizing":
            # A retained-until-janitor disposition can be an old snapshot while
            # a newer terminal transition owns the row. Never treat that stale
            # disposition as deletion authority while recovery is pending.
            reasons.append("terminal_disposition_pending")
        if record.workspace_path not in physical:
            reasons.append("missing_from_git_inventory")
        else:
            physical_entry = physical_entries[record.workspace_path]
            if bool(physical_entry.get("is_primary")):
                reasons.append("primary_worktree")
            if physical_entry.get("locked"):
                reasons.append("locked_worktree")
        if approved and not _is_within(record.workspace_path, approved):
            reasons.append("outside_approved_root")
        if (
            record.status == "protected"
            or (record.disposition or "").startswith("operational_exception:")
            or record.exception_policy_id
        ):
            reasons.append("protected_exception")
        if record.pr_numbers:
            # No network/API lookup occurs in janitor. Any associated PR is
            # therefore open-or-unverifiable and must be preserved.
            reasons.append("open_or_unverifiable_pr")
        if record.disposition != "retained_until:janitor":
            reasons.append("retention_condition_not_met")
        if (
            record.cleanup_policy != "on_task_terminal"
            or (record.board_id or "default", record.task_id or "")
            in retained_dependencies
        ):
            reasons.append("dependency_retained")
        if owner_counts.get((record.board_id, record.task_id, record.repo_id), 0) > 1:
            reasons.append("ambiguous_duplicate_owner")

        current_status = _git(
            record.workspace_path,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
        current_hash = (
            hashlib.sha256((current_status or "").encode("utf-8")).hexdigest()
            if current_status is not None else None
        )
        current_head = _git(record.workspace_path, "rev-parse", "HEAD")
        if (
            current_status is None
            or bool(current_status)
            or current_hash != record.dirty_manifest_hash
            or current_head != record.head_sha
        ):
            reasons.append("dirty_or_changed")

        if reasons:
            excluded.append(
                {
                    "workspace_id": record.workspace_id,
                    "workspace_path": record.workspace_path,
                    "reasons": sorted(set(reasons)),
                }
            )
            continue
        candidates.append(
            {
                "workspace_id": record.workspace_id,
                "workspace_path": record.workspace_path,
                "repo_path": record.repo_path,
                "task_id": record.task_id,
                "board_id": record.board_id,
                "estimated_bytes": record.estimated_bytes,
            }
        )
        receipts.append(
            {
                "workspace_id": record.workspace_id,
                "workspace_path": record.workspace_path,
                "repo_path": record.repo_path,
                "head_sha": current_head,
                "dirty_manifest_hash": current_hash,
                "planned_action": "retire_worktree",
                "execution": "disabled",
                "requires_live_reverification": True,
            }
        )

    return {
        "dry_run": True,
        "candidates": candidates,
        "excluded": excluded,
        "receipts": receipts,
        "removed": [],
    }
