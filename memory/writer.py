"""Fail-soft write interface to the shared memory database.

MemoryStore never raises: if sqlite-vec can't be loaded (missing package,
extension-loading unsupported on this Python build, etc.) it silently
becomes a no-op rather than taking down the voice loop, matching the
try/except-and-warn fallback style used elsewhere in this codebase (e.g.
dashboard_server.py's optional overlay_detections call).

A MemoryStore wraps one sqlite3 connection, so it must be created and used
from a single thread -- open a separate instance per thread (see
VLMBackgroundWorker, which creates its own inside the thread it runs in,
rather than sharing the main conversation loop's instance).
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any


class MemoryStore:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.conn = None
        try:
            from memory import db as memory_db

            self.conn = memory_db.connect()
            self.conn.execute(
                "INSERT OR IGNORE INTO sessions (id, started_at) VALUES (?, ?)",
                (session_id, time.time()),
            )
            self.conn.commit()
        except Exception as e:
            print(f"[Memory] Disabled -- failed to open memory database: {e}", file=sys.stderr)
            self.conn = None

    @property
    def enabled(self) -> bool:
        return self.conn is not None

    def record(
        self,
        event_type: str,
        text: str,
        *,
        chunk_ref: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.conn is None:
            return
        try:
            cur = self.conn.execute(
                "INSERT INTO events (session_id, chunk_ref, ts_wall, type, text, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self.session_id,
                    chunk_ref,
                    time.time(),
                    event_type,
                    text,
                    json.dumps(metadata) if metadata is not None else None,
                ),
            )
            self.conn.commit()
        except Exception as e:
            print(f"[Memory] Failed to record {event_type} event: {e}", file=sys.stderr)
            return

        # Embedding is a separate, independently-fail-soft step: the raw
        # event above must land even if the embedder is unavailable (no
        # internet for the first-run model download, missing onnxruntime,
        # etc.) or a given piece of text fails to embed.
        try:
            from memory.embedder import embed

            vector = embed(text)
            self.conn.execute(
                "INSERT INTO event_embeddings (event_id, embedding) VALUES (?, ?)",
                (cur.lastrowid, vector),
            )
            self.conn.commit()
        except Exception as e:
            print(f"[Memory] Failed to embed {event_type} event: {e}", file=sys.stderr)

    def close(self) -> None:
        if self.conn is None:
            return
        try:
            self.conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ?",
                (time.time(), self.session_id),
            )
            self.conn.commit()
            self.conn.close()
        except Exception as e:
            print(f"[Memory] Failed to close cleanly: {e}", file=sys.stderr)
        finally:
            self.conn = None
