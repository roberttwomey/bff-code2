-- Shared robot episodic memory schema.
--
-- One database (captures/memory.sqlite3) spans all sessions, so recall and
-- consolidation can reach across the robot's full history rather than a
-- single run. Raw, high-frequency sensor streams (lowstate.jsonl,
-- lidar.jsonl, video) stay exactly where capture_go2_data.py already puts
-- them -- this schema only covers the lower-frequency, queryable memory
-- layer: conversation turns, VLM captions, detections, telemetry
-- snapshots, and their consolidated summaries.
--
-- Requires the sqlite-vec extension to be loaded on the connection before
-- this script runs (the vec0 virtual table module below is registered by
-- that extension). See db.py.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- One row per robot run (matches a captures/session-<id> directory).
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    ended_at   REAL,
    notes      TEXT
);

-- LLM-consolidated rollups of a contiguous run of raw events.
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    ts_start    REAL NOT NULL,
    ts_end      REAL NOT NULL,
    summary     TEXT NOT NULL,
    event_count INTEGER NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS episode_embeddings USING vec0(
    episode_id INTEGER PRIMARY KEY,
    embedding  FLOAT[{EMBEDDING_DIM}]
);

-- Single append-only stream for everything the robot experiences.
CREATE TABLE IF NOT EXISTS events (
    id                INTEGER PRIMARY KEY,
    session_id        TEXT NOT NULL REFERENCES sessions(id),
    chunk_ref         TEXT,      -- e.g. "chunk_12", links back to raw video/audio/lidar on disk
    ts_wall           REAL NOT NULL,
    type              TEXT NOT NULL CHECK (type IN (
                          'conversation_user',
                          'conversation_assistant',
                          'vlm_caption',
                          'detection',
                          'telemetry',
                          'episode_summary'
                      )),
    text              TEXT NOT NULL,  -- transcript / caption / "saw: cat (0.82)" / body-state string
    metadata          TEXT,           -- JSON blob, shape varies by type
    consolidated_into INTEGER REFERENCES episodes(id)  -- NULL until rolled into an episode
);

CREATE INDEX IF NOT EXISTS idx_events_session_ts ON events(session_id, ts_wall);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_unconsolidated ON events(consolidated_into) WHERE consolidated_into IS NULL;

CREATE VIRTUAL TABLE IF NOT EXISTS event_embeddings USING vec0(
    event_id  INTEGER PRIMARY KEY,
    embedding FLOAT[{EMBEDDING_DIM}]
);
