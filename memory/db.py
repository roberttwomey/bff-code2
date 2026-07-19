#!/usr/bin/env python3
"""Connection and schema bootstrap for the shared robot memory database.

The database lives at ``<BFF_LOG_ROOT>/memory.sqlite3`` -- the same root
chat-manager.py uses for session directories -- so it persists across runs
and recall queries can span the robot's full history.

Requirements:
    - sqlite-vec (pip install sqlite-vec)
    - a Python whose stdlib sqlite3 module supports loadable extensions
      (Connection.enable_load_extension); if that's missing, connect()
      raises RuntimeError with a pointer to the fix rather than silently
      running without vector search.

Manual smoke test (verifies the extension loads and the schema applies,
e.g. after `pip install sqlite-vec` on a new device):
    python3 -m memory.db
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import dotenv
import sqlite_vec

# Load .env the same way chat-manager.py does, so BFF_LOG_ROOT (and anything
# else) resolves the same regardless of which script is the entry point --
# chat-manager.py loads it itself, but memory.inspect/other standalone
# entry points otherwise wouldn't see it and would silently fall back to
# the default path instead of the real, shared database.
dotenv.load_dotenv()

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output size

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

LOG_ROOT = Path(
    os.environ.get("BFF_LOG_ROOT", Path(__file__).resolve().parent.parent / "captures")
).expanduser()


def default_db_path() -> Path:
    return LOG_ROOT / "memory.sqlite3"


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (creating and initializing if needed) the shared memory database."""
    path = db_path or default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    try:
        conn.enable_load_extension(True)
    except AttributeError as exc:
        raise RuntimeError(
            "This Python's sqlite3 module was built without loadable-extension "
            "support, so sqlite-vec can't be loaded. Rebuild Python with "
            "--enable-loadable-sqlite-extensions, or switch to apsw."
        ) from exc

    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    schema = SCHEMA_PATH.read_text().replace("{EMBEDDING_DIM}", str(EMBEDDING_DIM))
    conn.executescript(schema)

    return conn


if __name__ == "__main__":
    conn = connect()
    vec_version = conn.execute("select vec_version()").fetchone()[0]
    tables = [
        row[0]
        for row in conn.execute(
            "select name from sqlite_master where type in ('table', 'table')"
            " and name not like 'sqlite_%' order by name"
        ).fetchall()
    ]
    print(f"sqlite-vec {vec_version} loaded OK")
    print(f"database: {default_db_path()}")
    print(f"tables: {', '.join(tables)}")
    conn.close()
