"""
AETHER — Task Manager Tool (DEPRECATION / SUBORDINATION NOTICE)
================================================================
NOTICE (Prompt 6 — Authoritative Tools & Verified Actions):
The primary, authoritative store for all tasks, projects, notes, and memory
is PostgreSQL via AETHER_BAC (src/modules/ai/tools/task-tools.ts and TasksRepository).
This SQLite-backed TaskManager is preserved for local inference tests and
subordinate offline model execution only. It is NOT the authoritative source of truth.
================================================================
"""
from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger("aether.tools.task_manager")

_VALID_PRIORITIES = {"low", "medium", "high"}
_VALID_STATUSES = {"pending", "in_progress", "done", "cancelled"}


class TaskManager:
    """
    SQLite-backed task store.

    Thread-safe: SQLite WAL mode + connection-per-call pattern.
    """

    def __init__(self, db_path: str):
        """
        Args:
            db_path: Path to the SQLite file (created if not exists).
        """
        self.db_path = db_path
        self._init_db()
        logger.info(f"TaskManager initialized | db='{db_path}'")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id          TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL DEFAULT 'default',
                    title       TEXT NOT NULL,
                    priority    TEXT NOT NULL DEFAULT 'medium',
                    status      TEXT NOT NULL DEFAULT 'pending',
                    notes       TEXT,
                    created_at  INTEGER NOT NULL,
                    updated_at  INTEGER NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_user ON tasks(user_id)")

    # ------------------------------------------------------------------
    # CRUD operations (also exposed as agent tools)
    # ------------------------------------------------------------------

    def add_task(
        self,
        title: str,
        priority: str = "medium",
        notes: Optional[str] = None,
        user_id: str = "default",
    ) -> dict:
        """
        Add a new task.

        Returns:
            Dict with the created task's id, title, priority, status.
        Raises:
            ValueError: On invalid arguments.
        """
        if not title or not title.strip():
            raise ValueError("Task title cannot be empty")
        priority = priority.lower()
        if priority not in _VALID_PRIORITIES:
            raise ValueError(f"priority must be one of {_VALID_PRIORITIES}, got '{priority}'")

        task_id = str(uuid.uuid4())
        now = int(time.time())
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO tasks (id, user_id, title, priority, status, notes, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)""",
                (task_id, user_id, title.strip(), priority, notes, now, now),
            )
        result = {"id": task_id, "title": title.strip(), "priority": priority, "status": "pending"}
        logger.info(f"Task added | id='{task_id}' title='{title[:60]}' priority='{priority}'")
        return result

    def list_tasks(
        self,
        user_id: str = "default",
        status: Optional[str] = None,
        priority: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        List tasks for a user, with optional filters.

        Returns:
            List of task dicts ordered by priority (high→low) then creation time.
        """
        priority_order = "CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END"
        params: list = [user_id]
        filters = ["user_id = ?"]

        if status:
            if status not in _VALID_STATUSES:
                raise ValueError(f"status must be one of {_VALID_STATUSES}")
            filters.append("status = ?")
            params.append(status)
        if priority:
            priority = priority.lower()
            if priority not in _VALID_PRIORITIES:
                raise ValueError(f"priority must be one of {_VALID_PRIORITIES}")
            filters.append("priority = ?")
            params.append(priority)

        where = " AND ".join(filters)
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM tasks WHERE {where} ORDER BY {priority_order}, created_at ASC LIMIT ?",
                params,
            ).fetchall()

        return [dict(r) for r in rows]

    def update_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> dict:
        """
        Update a task's status, priority, or notes.

        Returns:
            Updated task dict.
        Raises:
            ValueError: If task_id not found or invalid field values.
        """
        if status and status not in _VALID_STATUSES:
            raise ValueError(f"status must be one of {_VALID_STATUSES}")
        if priority and priority.lower() not in _VALID_PRIORITIES:
            raise ValueError(f"priority must be one of {_VALID_PRIORITIES}")

        sets, params = [], []
        if status:
            sets.append("status = ?"); params.append(status)
        if priority:
            sets.append("priority = ?"); params.append(priority.lower())
        if notes is not None:
            sets.append("notes = ?"); params.append(notes)
        if not sets:
            raise ValueError("No fields to update were provided")

        sets.append("updated_at = ?"); params.append(int(time.time()))
        params.append(task_id)

        with self._conn() as conn:
            cursor = conn.execute(
                f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", params
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Task id='{task_id}' not found")
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()

        result = dict(row)
        logger.info(f"Task updated | id='{task_id}' changes={sets[:-2]}")
        return result

    def delete_task(self, task_id: str) -> dict:
        """Delete a task by ID. Returns confirmation dict."""
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            if cursor.rowcount == 0:
                raise ValueError(f"Task id='{task_id}' not found")
        logger.info(f"Task deleted | id='{task_id}'")
        return {"deleted": True, "task_id": task_id}

    # ------------------------------------------------------------------
    # Tool callables for AgentLoop (thin wrappers with user_id bound)
    # ------------------------------------------------------------------

    def get_tools(self, user_id: str = "default") -> dict:
        """
        Returns a dict of {tool_name: callable} ready to pass to AgentLoop.
        Each callable has user_id pre-bound so the model doesn't need to know it.
        """
        return {
            "add_task": lambda title, priority="medium", notes=None: self.add_task(
                title=title, priority=priority, notes=notes, user_id=user_id
            ),
            "list_tasks": lambda status=None, priority=None, limit=10: self.list_tasks(
                user_id=user_id, status=status, priority=priority, limit=limit
            ),
            "update_task": lambda task_id, status=None, priority=None, notes=None: self.update_task(
                task_id=task_id, status=status, priority=priority, notes=notes
            ),
        }
