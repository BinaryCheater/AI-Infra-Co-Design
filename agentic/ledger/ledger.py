"""Append-only ledger: every event the agents produce, and every artifact by content id.

One SQLite file. Agents never mutate; they append. The three stores (population, archive, proof
library) and the UI are views rebuilt from the event sequence (see views.py). Checkpoints are
ordinary events whose payload holds a complete loop state, so replay and rewind need nothing else.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from ..schema import canonical, content_id, strip_timing, validate

Listener = Callable[[dict], None]


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._listeners: list[Listener] = []
        with self._lock:
            self._db.executescript(
                "CREATE TABLE IF NOT EXISTS events ("
                " seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, run TEXT NOT NULL,"
                " kind TEXT NOT NULL, iteration INTEGER NOT NULL, artifact TEXT, payload TEXT NOT NULL);"
                "CREATE TABLE IF NOT EXISTS artifacts ("
                " id TEXT PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL);"
                "CREATE INDEX IF NOT EXISTS events_run ON events(run, seq);")
            self._db.commit()

    # ------------------------------------------------------------------ artifacts
    def put(self, kind: str, body: dict, prefix: str | None = None) -> str:
        """Store an artifact by content id (idempotent) after schema validation; return its id."""
        body = dict(body)
        body["id"] = content_id(kind, body, prefix)
        validate(kind, body)
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO artifacts(id, kind, body) VALUES (?, ?, ?)",
                             (body["id"], kind, canonical(body)))
            self._db.commit()
        return body["id"]

    def get(self, artifact_id: str) -> dict:
        with self._lock:
            row = self._db.execute("SELECT kind, body FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
        if row is None:
            raise KeyError(artifact_id)
        return json.loads(row["body"])

    def kind_of(self, artifact_id: str) -> str:
        with self._lock:
            row = self._db.execute("SELECT kind FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
        if row is None:
            raise KeyError(artifact_id)
        return row["kind"]

    def artifacts(self, kind: str | None = None) -> list[dict]:
        q = "SELECT body FROM artifacts" + (" WHERE kind = ?" if kind else "")
        with self._lock:
            rows = self._db.execute(q, (kind,) if kind else ()).fetchall()
        return [json.loads(r["body"]) for r in rows]

    # --------------------------------------------------------------------- events
    def append(self, run: str, kind: str, iteration: int, payload: dict | None = None,
               artifact: str | None = None) -> dict:
        ev = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "run": run, "kind": kind,
              "iteration": iteration, "artifact": artifact, "payload": payload or {}}
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO events(ts, run, kind, iteration, artifact, payload) VALUES (?, ?, ?, ?, ?, ?)",
                (ev["ts"], run, kind, iteration, artifact, canonical(ev["payload"])))
            self._db.commit()
            ev["seq"] = cur.lastrowid
        validate("event", ev)
        for fn in list(self._listeners):
            fn(ev)
        return ev

    def events(self, run: str | None = None, since: int = 0, kinds: Iterable[str] | None = None) -> list[dict]:
        q, args = "SELECT * FROM events WHERE seq > ?", [since]
        if run:
            q += " AND run = ?"; args.append(run)
        if kinds:
            ks = list(kinds); q += f" AND kind IN ({','.join('?' * len(ks))})"; args += ks
        q += " ORDER BY seq"
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
        return [self._row(r) for r in rows]

    def last_seq(self, run: str | None = None) -> int:
        with self._lock:
            row = self._db.execute("SELECT MAX(seq) AS m FROM events" + (" WHERE run = ?" if run else ""),
                                   (run,) if run else ()).fetchone()
        return int(row["m"] or 0)

    def runs(self) -> list[str]:
        with self._lock:
            rows = self._db.execute("SELECT run, MIN(seq) AS s FROM events GROUP BY run ORDER BY s").fetchall()
        return [r["run"] for r in rows]

    def checkpoints(self, run: str) -> list[dict]:
        return self.events(run, kinds=["checkpoint"])

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        return {"seq": r["seq"], "ts": r["ts"], "run": r["run"], "kind": r["kind"], "iteration": r["iteration"],
                "artifact": r["artifact"], "payload": json.loads(r["payload"])}

    # ------------------------------------------------------------------ streaming
    def subscribe(self, fn: Listener) -> Callable[[], None]:
        self._listeners.append(fn)
        return lambda: self._listeners.remove(fn)

    def close(self):
        with self._lock:
            self._db.close()


def replay_signature(events: list[dict]) -> list[tuple]:
    """What must be identical when a run is replayed: everything except seq and wall-clock fields."""
    return [(e["kind"], e["iteration"], e["artifact"], canonical(strip_timing(e["payload"]))) for e in events]
