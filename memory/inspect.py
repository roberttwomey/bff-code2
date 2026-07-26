#!/usr/bin/env python3
"""Quick CLI to inspect the shared memory database.

Usage:
    python3 -m memory.inspect                    # summary + most recent session's events
    python3 -m memory.inspect --session <id>      # detail for one session
    python3 -m memory.inspect --limit 20           # show more/fewer recent events
"""

from __future__ import annotations

import argparse
from datetime import datetime

from memory import db


def _fmt_ts(ts: float | None) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "-"


def print_sessions(conn, limit: int = 5) -> None:
    rows = conn.execute(
        """
        SELECT s.id, s.started_at, s.ended_at, COUNT(e.id) AS event_count
        FROM sessions s
        LEFT JOIN events e ON e.session_id = s.id
        GROUP BY s.id
        ORDER BY s.started_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    print(f"Sessions (most recent {len(rows)}):")
    for r in rows:
        print(
            f"  {r['id']:<20} started {_fmt_ts(r['started_at']):<20} "
            f"ended {_fmt_ts(r['ended_at']):<20} events={r['event_count']}"
        )
    print()


def print_session_detail(conn, session_id: str, limit: int) -> None:
    type_counts = conn.execute(
        """
        SELECT e.type,
               COUNT(*) AS total,
               COUNT(ee.event_id) AS embedded
        FROM events e
        LEFT JOIN event_embeddings ee ON ee.event_id = e.id
        WHERE e.session_id = ?
        GROUP BY e.type
        ORDER BY e.type
        """,
        (session_id,),
    ).fetchall()

    if not type_counts:
        print(f"No events found for session '{session_id}'.")
        return

    print(f"Event counts for session '{session_id}':")
    total, total_embedded = 0, 0
    for r in type_counts:
        flag = "" if r["total"] == r["embedded"] else "  <-- missing embeddings"
        print(f"  {r['type']:<22} {r['embedded']}/{r['total']} embedded{flag}")
        total += r["total"]
        total_embedded += r["embedded"]
    print(f"  {'TOTAL':<22} {total_embedded}/{total} embedded\n")

    rows = conn.execute(
        """
        SELECT ts_wall, type, text
        FROM events
        WHERE session_id = ?
        ORDER BY ts_wall DESC
        LIMIT ?
        """,
        (session_id, limit),
    ).fetchall()

    print(f"Most recent {len(rows)} events:")
    for r in reversed(rows):
        text = r["text"].replace("\n", " ")
        if len(text) > 90:
            text = text[:87] + "..."
        print(f"  {_fmt_ts(r['ts_wall'])}  [{r['type']:<20}]  {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", help="Session id to inspect (default: most recent)")
    parser.add_argument("--limit", type=int, default=10, help="Number of recent events to show")
    args = parser.parse_args()

    conn = db.connect()

    session_id = args.session
    if session_id is None:
        row = conn.execute("SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1").fetchone()
        session_id = row["id"] if row else None

    print_sessions(conn)

    if session_id is None:
        print("No sessions recorded yet.")
        return

    print_session_detail(conn, session_id, args.limit)
    conn.close()


if __name__ == "__main__":
    main()
