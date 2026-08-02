#!/usr/bin/env python3
"""Timeline replay of a recorded BFF capture session.

Unlike ``dashboard_server.py --simulate`` (which streams the newest session
forward, chunk by chunk, with no audio and no way to move around), this is a
dedicated *replay* tool built around a single global timeline and a scrubber:

  1. audio   - plays the recorded per-turn speech: the user's input utterance
               (``turn-NNN-input.wav``) and the robot's synthesised reply
               (``turn-NNN-response.wav``), plus wake/reset cues, each placed
               at the moment it was actually heard. Optionally mixes in the
               robot-mic ambient (``chunk_*/audio.wav``) with --mic-audio.
  2. lidar   - the recorded voxel scans, accumulated up to the playhead.
  3. body    - the recorded lowstate telemetry sample at the playhead.
  4. name    - the header shows the *recorded* device/robot (e.g. snapper.local),
               read from session.jsonl's session_start, not this host.
  5. scrub   - a timeline slider drives one server-authoritative playhead;
               play/pause and seeking move every stream together.

Sessions are large - a full performance is tens of minutes and its
``lidar.jsonl`` runs to several gigabytes - so nothing bulk is held in memory.
Each chunk's telemetry/lidar/detection log is reduced to a compact
``(timestamp -> byte offset)`` index (cached under the session so later runs are
instant), and individual records are read from disk lazily at the playhead.

It reuses the live dashboard's front end (templates/dashboard.html) unchanged
except for a scrubber block that only renders when replay_mode is set, and it
speaks the same SocketIO/MJPEG interface the live dashboard already consumes.

Usage:
    ./replay.py                         # newest session with video
    ./replay.py --session session-20260723-092744
    ./replay.py --mic-audio             # also mix in robot-mic ambient
    ./replay.py --no-audio --loop
"""

from __future__ import annotations

import os
import re
import sys
import time
import json
import base64
import signal
import bisect
import argparse
import datetime as dt
import threading
import wave
from pathlib import Path

import numpy as np
import cv2
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
import dotenv

dotenv.load_dotenv()

# Piper writes its speech wavs with the voice-config rate in the header (22050)
# while the samples are really the 48 kHz playback stream - same fact
# fix_recordings.py corrects. We read them at the true rate.
SPEECH_TRUE_RATE = 48000


_DERIVED_CACHE: dict[str, Path | None] = {}


def derived_session_dir(session_dir: Path) -> Path | None:
    """The processed/ bundle for a session, wherever the session itself came from.

    A session can be replayed from the archive (logs-all/<group>/<id>), from the
    gathered by-phase tree, or straight out of the live capture dir (~/bff/logs),
    and only the first of those sits next to its machine group. Locating the
    bundle by session id instead means the repaired videos and the sample-rate
    verdicts are found in all three cases - otherwise replaying the live copy
    silently loses both.
    """
    sid = session_dir.name
    if sid in _DERIVED_CACHE:
        return _DERIVED_CACHE[sid]
    root = Path(os.getenv("BFF_DERIVED_ROOT", "/Volumes/Cohab2024/BFF/processed"))
    found = None
    if root.is_dir():
        for g in sorted(os.listdir(root)):
            if g in ("retranscribed", "complete-logs"):
                continue
            p = root / g / sid
            if p.is_dir():
                found = p
                break
    _DERIVED_CACHE[sid] = found
    return found


_RATE_CACHE: dict[str, set[str]] = {}


def _corrected_wavs(session_dir: Path) -> set[str]:
    """Names of the response wavs whose declared rate is a lie, per the archive.

    Which Piper wavs mislabel their rate is NOT a function of the session date.
    Most pre-2026-07-21 wavs are genuinely 22050; most later ones hold 48000 under
    a 22050 header; and the 1969-clock sessions (unset RTC) are mixed - 41 are
    really 48000 and 367 are not. No date rule can express that, and guessing
    wrong plays the audio at 2.18x the correct speed in either direction.

    The re-transcription pass settled it per file by transcribing both ways and
    scoring against the logged text. Its verdict is recorded as `rate_corrected`
    in retranscription.json, which is what this reads.
    """
    key = str(session_dir)
    if key in _RATE_CACHE:
        return _RATE_CACHE[key]
    names: set[str] = set()
    cands = [session_dir / "transcript" / "retranscription.json",
             session_dir / "retranscription.json"]
    root = Path(os.getenv("BFF_DERIVED_ROOT", "/Volumes/Cohab2024/BFF/processed")) / "retranscribed"
    if root.is_dir():
        for g in sorted(os.listdir(root)):
            cands.append(root / g / session_dir.name / "retranscription.json")
    for p in cands:
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as ex:
            print(f"[Replay] could not read {p}: {ex}")
            continue
        for w in data.get("wavs", []):
            if w.get("rate_corrected") and w.get("file"):
                names.add(os.path.basename(w["file"]))
        break
    _RATE_CACHE[key] = names
    return names


def get_speech_forced_rate(wav_path: Path, etype: str | None = None,
                           session_dir: Path | None = None) -> int | None:
    """The true sample rate for a speech wav, or None to trust its header.

    Only wavs the archive has positively identified as mislabelled are forced;
    everything else plays at its declared rate. Falling back to the header when
    no re-transcription bundle is present is the safe default - it is what the
    file itself claims, rather than a guess.
    """
    if not (etype == "assistant" or wav_path.name.endswith("-response.wav")):
        return None
    if session_dir is None:
        return None
    if wav_path.name not in _corrected_wavs(session_dir):
        return None
    try:
        with wave.open(str(wav_path), "rb") as w:
            if w.getframerate() != SPEECH_TRUE_RATE:
                return SPEECH_TRUE_RATE
    except Exception:
        pass
    return None


def resolve_chunk_video(cdir: Path) -> Path | None:
    """The playable video for a chunk, preferring a repaired copy.

    122 chunk videos in the archive have no moov atom - OpenCV only writes the
    index on release(), so a hard shutdown truncates whatever chunk was open.
    It is always the session's last chunk, i.e. the end of a take.
    fix_recordings.py rebuilds them alongside the derived work, at
    processed/<group>/<session>/repaired/<chunk>/video.mp4, and deliberately
    leaves the broken original in place. Reading the original regardless drops
    the ending of 119 sessions on the floor.
    """
    own = cdir / "video.mp4"
    session = cdir.parent
    cands = [
        # by-phase layout gathers the session's files under one directory
        session / "repaired" / cdir.name / "video.mp4",
        session.parent / "repaired" / cdir.name / "video.mp4",
    ]
    derived = os.getenv("BFF_DERIVED_ROOT", "/Volumes/Cohab2024/BFF/processed")
    if derived and session.parent.name:
        cands.append(Path(derived) / session.parent.name / session.name /
                     "repaired" / cdir.name / "video.mp4")
    dsd = derived_session_dir(session)
    if dsd is not None:
        cands.append(dsd / "repaired" / cdir.name / "video.mp4")
    for c in cands:
        try:
            if c.exists() and c.resolve() != own.resolve():
                return c
        except OSError:
            continue
    return own if own.exists() else None


# The single rate the mixed replay audio is delivered at. The browser's
# AudioContext resamples per chunk to the output device anyway; 24 kHz keeps the
# in-memory master mix small without audibly hurting speech.
DEFAULT_TARGET_RATE = 24000

# Downsampling budgets so a huge session can't blow up a lidar payload. The
# client caps again on the GPU side.
SINGLE_SCAN_MAX = 1500       # points sent for the scan at the playhead
ACCUM_STRIDE = 5             # per-scan stride when accumulating (matches client)
ACCUM_SCAN_BUDGET = 120      # scans sampled when rebuilding accumulation on a seek
MAX_ACCUM_POINTS = 60000     # hard cap on an accumulated payload

# The writers put "timestamp" first on every line, so we can read a record's
# time without parsing its (possibly enormous) points array.
_TS_RE = re.compile(rb'"timestamp":\s*([0-9eE.+-]+)')

INDEX_FILENAME = ".replay-index.json"


# --------------------------------------------------------------------------- #
# Shared live state (same globals the dashboard's video/overlay path expects)
# --------------------------------------------------------------------------- #
latest_frame = None
frame_lock = threading.Lock()

latest_detections = []
latest_detection_time = 0.0
detections_lock = threading.Lock()
yolo_enabled = True

latest_lowstate = None
lowstate_lock = threading.Lock()


def draw_detections(frame):
    """Draw the current detection boxes onto a copy of frame (dashboard parity)."""
    global latest_detections, latest_detection_time
    with detections_lock:
        if time.time() - latest_detection_time > 1.0:
            dets = []
        else:
            dets = list(latest_detections)
    if not dets or not yolo_enabled:
        return frame
    frame = frame.copy()
    for det in dets:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1), (x1 + tw + 4, y1 + th + 6), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def generate_mjpeg():
    """MJPEG generator over latest_frame, identical in spirit to the dashboard."""
    global latest_frame
    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is not None:
            frame = draw_detections(frame)
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            time.sleep(0.05)
        time.sleep(0.04)


# --------------------------------------------------------------------------- #
# Session discovery / low-level file helpers
# --------------------------------------------------------------------------- #
def get_session_root() -> Path:
    """Same BFF_LOG_ROOT chat-manager.py and dashboard_server.py use."""
    default_root = Path(__file__).resolve().parent / "captures"
    return Path(os.environ.get("BFF_LOG_ROOT", default_root)).expanduser()


def _session_has_video(session_dir: Path) -> bool:
    return bool(any(session_dir.glob("chunk_*/video.mp4"))) or (session_dir / "video.mp4").exists()


def resolve_session(name: str | None) -> Path | None:
    """Absolute path to the session to replay: an explicit --session, else the
    newest session directory that actually has video to show."""
    root = get_session_root()
    if name:
        cand = Path(name)
        if not cand.is_absolute():
            cand = root / name
        return cand if cand.exists() else None
    if not root.exists():
        return None
    sessions = sorted([d for d in root.glob("session-*") if d.is_dir()],
                      key=lambda p: p.name, reverse=True)
    for s in sessions:
        if _session_has_video(s):
            return s
    return sessions[0] if sessions else None


def load_curated_playlist() -> list[dict]:
    """Load the 41 curated sessions from docs/replay/bff-replay-index.json."""
    index_path = Path(__file__).resolve().parent / "docs" / "replay" / "bff-replay-index.json"
    playlist: list[dict] = []
    if not index_path.exists():
        return playlist
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return playlist

    exchanges = data.get("exchanges", [])
    captures_root = get_session_root()
    archive_root_env = os.getenv("BFF_ARCHIVE_ROOT", "/Volumes/Cohab2024/BFF/logs-all")
    archive_root = Path(archive_root_env).expanduser() if archive_root_env else None
    by_phase_root = archive_root.parent / "by-phase" if archive_root and archive_root.parent.exists() else None

    for idx, e in enumerate(exchanges):
        sid = e.get("session_id", "")
        phase = e.get("phase", "")
        label = e.get("label", "")
        machine = e.get("machine", "")
        cand_path = None

        # 1. Search in captures_root
        if captures_root.exists():
            p = captures_root / sid
            if p.is_dir():
                cand_path = p
        # 2. Search in archive_root across the machine groups. Take the richest
        #    copy, not the first: _conflicts holds partial duplicates and sorts
        #    ahead of every machine name, so picking alphabetically handed back a
        #    copy with no chunk_* at all - session-19691231-160201 lost all 10 of
        #    its video chunks that way.
        if not cand_path and archive_root and archive_root.exists():
            try:
                cands = [archive_root / g / sid for g in sorted(os.listdir(archive_root))
                         if (archive_root / g / sid).is_dir()]
                if cands:
                    cand_path = max(cands, key=lambda p: (
                        len(list(p.glob("chunk_*"))),
                        p.parent.name != "_conflicts",
                        sum(1 for _ in p.iterdir()),
                    ))
            except Exception:
                pass
        # 3. Search in by_phase_root
        if not cand_path and by_phase_root and by_phase_root.exists():
            try:
                for ph_dir in by_phase_root.glob("*"):
                    if ph_dir.is_dir():
                        for match in ph_dir.glob(f"*{sid}*"):
                            if match.is_dir():
                                cand_path = match
                                break
            except Exception:
                pass

        playlist.append({
            "index": idx,
            "session_id": sid,
            "phase": phase,
            "label": label,
            "machine": machine,
            "path": cand_path,
            "turns_count": len(e.get("turns", [])),
        })

    return playlist



def parse_event_time(value) -> float | None:
    """session.jsonl carries ISO-8601 stamps (and, historically, compact
    YYYYMMDD-HHMMSS). Return a wall-clock epoch, or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    try:
        return dt.datetime.fromisoformat(text).timestamp()
    except ValueError:
        pass
    for fmt in ("%Y%m%d-%H%M%S.%f", "%Y%m%d-%H%M%S", "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    return None


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def first_timestamp(path: Path) -> float | None:
    """First record's timestamp, read from a single line (cheap)."""
    if not path.exists():
        return None
    with path.open("rb") as f:
        for raw in f:
            m = _TS_RE.search(raw[:120])
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    return None
    return None


def last_timestamp(path: Path) -> float | None:
    """Last record's timestamp, read from the tail without scanning the file."""
    if not path.exists():
        return None
    size = path.stat().st_size
    if size == 0:
        return None
    with path.open("rb") as f:
        block = min(size, 131072)
        f.seek(size - block)
        tail = f.read()
    for raw in reversed(tail.split(b"\n")):
        m = _TS_RE.search(raw[:120])
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def build_offset_index(path: Path) -> list[list]:
    """Stream a jsonl file once, returning [[timestamp, byte_offset], ...] for
    every record whose timestamp we can read from the line prefix. This reads
    the whole file (the only unavoidable pass) but keeps nothing bulky."""
    entries: list[list] = []
    if not path.exists():
        return entries
    with path.open("rb") as f:
        off = 0
        for raw in f:
            m = _TS_RE.search(raw[:120])
            if m:
                try:
                    entries.append([float(m.group(1)), off])
                except ValueError:
                    pass
            off += len(raw)
    return entries


def read_json_at(path: Path, offset: int) -> dict | None:
    try:
        with path.open("rb") as f:
            f.seek(offset)
            line = f.readline()
        return json.loads(line)
    except Exception:
        return None


def load_wav_mono(path: Path, forced_rate: int | None, target_rate: int) -> np.ndarray:
    """Read a 16-bit PCM wav as mono float32, resampled to target_rate.

    forced_rate overrides the header (used for the mislabelled speech wavs)."""
    try:
        with wave.open(str(path), "rb") as w:
            nch, sw, fr, n = (w.getnchannels(), w.getsampwidth(),
                              w.getframerate(), w.getnframes())
            raw = w.readframes(n)
    except Exception:
        return np.zeros(0, dtype=np.float32)
    if sw != 2 or n == 0:
        return np.zeros(0, dtype=np.float32)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    rate = forced_rate or fr
    if rate != target_rate and len(data) > 1:
        n_out = max(1, int(round(len(data) * target_rate / rate)))
        idx = np.linspace(0.0, len(data) - 1, n_out)
        data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
    return data.astype(np.float32)


def wav_duration(path: Path, forced_rate: int | None) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            n, fr = w.getnframes(), w.getframerate()
    except Exception:
        return 0.0
    rate = forced_rate or fr
    return n / rate if rate else 0.0


# --------------------------------------------------------------------------- #
# Index cache (so the one-pass indexing cost is paid once per session)
# --------------------------------------------------------------------------- #
def _file_sig(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


class IndexCache:
    def __init__(self, session_dir: Path):
        self.path = session_dir / INDEX_FILENAME
        self._data = {}
        self._dirty = False
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def get(self, key: str, path: Path) -> list[list]:
        sig = _file_sig(path)
        cached = self._data.get(key)
        if cached and cached.get("sig") == sig:
            return cached["entries"]
        entries = build_offset_index(path)
        self._data[key] = {"sig": sig, "entries": entries}
        self._dirty = True
        return entries

    def save(self):
        if not self._dirty:
            return
        try:
            self.path.write_text(json.dumps(self._data), encoding="utf-8")
            self._dirty = False
        except Exception as e:
            print(f"[Replay] Could not write index cache ({e}); "
                  f"indices will rebuild next run.")


# --------------------------------------------------------------------------- #
# Timeline model
# --------------------------------------------------------------------------- #
class Chunk:
    def __init__(self, name, cdir, video_path, start_epoch, video_duration, fps):
        self.name = name
        self.dir = cdir
        self.video_path = video_path        # str or None
        self.start_epoch = start_epoch       # wall epoch of this chunk's first sample
        self.video_duration = video_duration
        self.fps = fps
        self.offset = 0.0                    # seconds from t0, filled after t0 known


class SessionTimeline:
    """Everything replay needs, resolved to one t0-relative timeline. Telemetry
    and lidar are held only as (t -> byte offset) indices and read lazily."""

    def __init__(self, session_dir: Path, target_rate: int,
                 include_mic: bool, want_audio: bool):
        self.session_dir = session_dir
        self.target_rate = target_rate
        self.chunks: list[Chunk] = []
        # Global sorted indices: parallel arrays for bisect. Each entry knows its
        # chunk (file to read from) and byte offset within it.
        self.low_t: list[float] = []
        self.low_ref: list[tuple[Path, int]] = []
        self.lid_t: list[float] = []
        self.lid_ref: list[tuple[Path, int]] = []
        self.detections: list[dict] = []     # small; held in full: {t, detections}
        self.events: list[dict] = []         # user/assistant lines: {t, record}
        self.master_audio = np.zeros(0, dtype=np.int16)
        self.duration = 0.0
        self.robot_name = "SNAPPER"
        self.device_name = "snapper.local"
        # Cache of strided scans (accumulation stride), so re-seeking or replaying
        # an already-visited stretch doesn't re-parse multi-megabyte scans. Keyed
        # by lidar index; there are only a few thousand scans, but cap anyway.
        self._scan_cache: dict[int, list] = {}
        self._build(include_mic, want_audio)

    # -- session_start: recorded robot/device name (requirement 4) ----------- #
    def _read_identity(self, events: list[dict]):
        start = next((e for e in events if e.get("type") == "session_start"), None)
        speaker = None
        device = None
        if start:
            cfg = start.get("config") or {}
            speaker = cfg.get("speaker")
            env = start.get("env_overrides") or {}
            speaker = env.get("BFF_SPEAKER", speaker)
            device = env.get("BFF_DEVICE_NAME")
        self.robot_name = (speaker or "SNAPPER").upper()
        # If the recording didn't pin a device name it defaulted to the
        # recording host's hostname, which isn't stored - but the robot it ran
        # on is the speaker, so snapper -> snapper.local is the honest label.
        self.device_name = device or f"{self.robot_name.lower()}.local"

    def _get_retranscribed_map(self, session_dir: Path) -> dict:
        """Load the distil-large-v3 text keyed by wav name and (turn, role).

        Not used for display - see the note in the event loop. Kept because it is
        the natural hook for an optional "show what was audible" toggle, which is
        a different question from "show what the model saw".
        """
        m = {}
        paths = []
        if session_dir.is_dir():
            paths.extend([
                session_dir / "transcript" / "retranscription.json",
                session_dir / "retranscription.json"
            ])
        for p in paths:
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    for w in data.get("wavs", []):
                        txt = w.get("text")
                        if txt:
                            if w.get("file"):
                                m[w["file"]] = txt
                            if w.get("turn") and w.get("kind"):
                                role = "user" if w["kind"] in ("user", "input") else "assistant"
                                m[(w["turn"], role)] = txt
                except Exception as ex:
                    print(f"[Replay] Error reading retranscription.json: {ex}")

        idx_path = Path(__file__).resolve().parent / "docs" / "replay" / "bff-replay-index.json"
        if idx_path.exists():
            try:
                idx_data = json.loads(idx_path.read_text(encoding="utf-8"))
                ex_match = next((x for x in idx_data.get("exchanges", [])
                                if x["session_id"] == session_dir.name or x["session_id"] in str(session_dir)), None)
                if ex_match and ex_match.get("turns"):
                    for t in ex_match["turns"]:
                        txt = t.get("text")
                        kind = t.get("kind", "user")
                        role = "user" if kind in ("user", "input") else "assistant"
                        if txt:
                            if t.get("turn"):
                                m[(t["turn"], role)] = txt
                            if t.get("file"):
                                m[t["file"]] = txt
            except Exception as ex:
                print(f"[Replay] Error reading bff-replay-index.json: {ex}")

        return m

    def _build(self, include_mic: bool, want_audio: bool):
        session = self.session_dir
        events = _read_jsonl(session / "session.jsonl")
        if not events:
            events = _read_jsonl(session / "transcript" / "session.jsonl")
        if not events:
            events = _read_jsonl(session / "transcript" / "session-complete.jsonl")
        if not events and session.is_file() and session.name.endswith(".jsonl"):
            events = _read_jsonl(session)
        if not events and (session / f"{session.name}.jsonl").exists():
            events = _read_jsonl(session / f"{session.name}.jsonl")

        # Check chat-sessions-v1 directory for legacy v1 logs
        if not events:
            v1_path = Path(__file__).resolve().parent / "docs" / "replay" / "chat-sessions-v1" / f"{session.name}.jsonl"
            if v1_path.exists():
                events = _read_jsonl(v1_path)

        # Parse v1 chat_stream_request / chat_stream_response records if present
        parsed_v1_events = []
        seen_ms = 0
        n_turn = 0
        for r in events:
            t_type = r.get("type")
            if t_type == "chat_stream_request":
                ms = r.get("messages") or []
                for m in ms[seen_ms:]:
                    if m.get("role") != "user": continue
                    txt = (m.get("content") or "").strip()
                    if not txt: continue
                    if parsed_v1_events and parsed_v1_events[-1]["type"] == "user" and parsed_v1_events[-1]["text"] == txt:
                        continue
                    n_turn += 1
                    parsed_v1_events.append({"type": "user", "turn": n_turn, "timestamp": r.get("timestamp"), "text": txt})
                seen_ms = len(ms)
            elif t_type == "chat_stream_response":
                txt = (r.get("reply") or "").strip()
                if not txt: continue
                n_turn += 1
                parsed_v1_events.append({"type": "assistant", "turn": n_turn, "timestamp": r.get("timestamp"), "text": txt, "model": r.get("model")})
        if parsed_v1_events:
            events.extend(parsed_v1_events)

        self._read_identity(events)

        chunk_dirs = sorted([d for d in session.glob("chunk_*") if d.is_dir()],
                            key=lambda d: int(d.name.split('_')[1])) if session.is_dir() else []
        if not chunk_dirs and session.is_dir() and _session_has_video(session):
            chunk_dirs = [session]  # legacy single-directory session

        cache = IndexCache(session if session.is_dir() else session.parent)

        # First pass: per-chunk anchor (first data timestamp, read cheaply) and
        # video metadata. A chunk's video/audio and telemetry all started
        # together, so its first data timestamp anchors them.
        raw = []
        for cdir in chunk_dirs:
            low_p = cdir / "lowstate.jsonl"
            lid_p = cdir / "lidar.jsonl"
            det_p = cdir / "detections.jsonl"
            starts = [first_timestamp(p) for p in (low_p, lid_p, det_p)]
            starts = [s for s in starts if s is not None]
            start_epoch = min(starts) if starts else None

            resolved = resolve_chunk_video(cdir)
            video_path = str(resolved) if resolved else None
            vdur, fps = 0.0, 30.0
            if video_path:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                    vdur = (nfr / fps) if fps else 0.0
                    if resolved.name == "video.mp4" and resolved.parent.parent.name == "repaired":
                        print(f"[Replay] {cdir.name}: using repaired video "
                              f"({vdur:.0f}s recovered).")
                else:
                    print(f"[Replay] {cdir.name}/video.mp4 is unreadable and has no repaired "
                          f"copy; skipping its video. Run fix_recordings.py --steps repair.")
                    video_path = None
                cap.release()
            raw.append({"dir": cdir, "low_p": low_p, "lid_p": lid_p, "det_p": det_p,
                        "start_epoch": start_epoch, "video_path": video_path,
                        "vdur": vdur, "fps": fps})

        # Fill a missing anchor from the previous chunk's end.
        for i, rc in enumerate(raw):
            if rc["start_epoch"] is None and i > 0 and raw[i - 1]["start_epoch"] is not None:
                rc["start_epoch"] = raw[i - 1]["start_epoch"] + raw[i - 1]["vdur"]

        dialogue = self._collect_dialogue(session, events) if (want_audio and session.is_dir()) else []

        anchors = [rc["start_epoch"] for rc in raw if rc["start_epoch"]]
        anchors += [c["start_epoch"] for c in dialogue]
        for e in events:
            if e.get("type") in ("user", "assistant", "assistant_chunk"):
                et = parse_event_time(e.get("timestamp"))
                if et:
                    anchors.append(et)

        # Fallback to bff-replay-index.json curated turns if no timestamps were found
        if not anchors:
            idx_file = Path(__file__).resolve().parent / "docs" / "replay" / "bff-replay-index.json"
            if idx_file.exists():
                try:
                    idx_data = json.loads(idx_file.read_text(encoding="utf-8"))
                    ex_match = next((x for x in idx_data.get("exchanges", []) if x["session_id"] == session.name or x["session_id"] in str(session)), None)
                    if ex_match and ex_match.get("turns"):
                        for t in ex_match["turns"]:
                            kind = t.get("kind", "user")
                            role = "user" if kind == "user" else "assistant"
                            events.append({
                                "type": role,
                                "timestamp": t.get("iso"),
                                "text": t.get("text"),
                                "model": t.get("model")
                            })
                            et = parse_event_time(t.get("iso")) or t.get("epoch")
                            if et:
                                anchors.append(float(et))
                except Exception as ex:
                    print(f"[Replay] Error reading index fallback: {ex}")

        if not anchors:
            raise RuntimeError("session has no usable timestamps to build a timeline")
        t0 = min(anchors)

        # Second pass: build the lazy indices and last-timestamp for duration.
        end = 0.0
        for rc in raw:
            if rc["start_epoch"] is None:
                continue
            off = rc["start_epoch"] - t0
            ch = Chunk(rc["dir"].name, rc["dir"], rc["video_path"],
                       rc["start_epoch"], rc["vdur"], rc["fps"])
            ch.offset = off
            self.chunks.append(ch)
            end = max(end, off + rc["vdur"])

            for tstamp, boff in cache.get(f"{rc['dir'].name}/lowstate", rc["low_p"]):
                self.low_t.append(tstamp - t0)
                self.low_ref.append((rc["low_p"], boff))
            for tstamp, boff in cache.get(f"{rc['dir'].name}/lidar", rc["lid_p"]):
                self.lid_t.append(tstamp - t0)
                self.lid_ref.append((rc["lid_p"], boff))
            for r in _read_jsonl(rc["det_p"]):
                tv = r.get("timestamp")
                if isinstance(tv, (int, float)):
                    self.detections.append({"t": tv - t0,
                                            "detections": r.get("detections", [])})
            for p, lt in ((rc["low_p"], None), (rc["lid_p"], None)):
                lts = last_timestamp(p)
                if lts is not None:
                    end = max(end, lts - t0)
        cache.save()

        # The per-chunk indices are already time-ordered and chunks are sorted,
        # so the concatenated arrays are globally sorted.
        self.detections.sort(key=lambda x: x["t"])
        dialogue = self._collect_dialogue(session, events) if (want_audio and session.is_dir()) else []

        # Map event object id -> start_epoch (beginning of utterance)
        event_start_times = {}
        for d in dialogue:
            ev = d.get("event")
            if ev and d.get("start_epoch") is not None:
                event_start_times[id(ev)] = d["start_epoch"]

        for e in events:
            if e.get("type") not in ("user", "assistant", "assistant_chunk"):
                continue
            et = parse_event_time(e.get("timestamp"))
            if et is None:
                continue

            # Deliberately NOT overridden with the re-transcription. The logged
            # text is what the model actually had in context at that moment, so
            # it is what a replay of the moment should show. The re-transcription
            # is a better record of what was *audible* - it recovers speech the
            # live system dropped - which is why the key-phrase analysis uses it,
            # but replaying it here inserts turns the dog never saw: noise-
            # triggered fragments like "thank you" that it was not responding to.

            # Display transcript text at the beginning of the utterance rather than the end
            if id(e) in event_start_times:
                et = event_start_times[id(e)]
            elif e.get("type") == "assistant" and e.get("audio_path"):
                base_name = os.path.basename(e["audio_path"])
                wav = session / base_name
                if not wav.exists():
                    wav = session / "speech" / base_name
                if wav.exists():
                    frate = get_speech_forced_rate(wav, "assistant", session)
                    dur = wav_duration(wav, frate)
                    et = et - dur

            self.events.append({"t": et - t0, "record": e})
            end = max(end, et - t0)
        self.events.sort(key=lambda x: x["t"])

        if want_audio:
            sources = list(dialogue)
            if include_mic:
                for rc in raw:
                    if rc["start_epoch"] is None:
                        continue
                    mic = rc["dir"] / "audio.wav"
                    if mic.exists():
                        sources.append({"path": mic, "forced_rate": None,
                                        "start_epoch": rc["start_epoch"]})
            self._mix_audio(sources, t0)
            end = max(end, len(self.master_audio) / self.target_rate)

        self.duration = max(end, 0.001)

        # Collect phrase/turn timestamps (human, assistant, audio start times)
        raw_turn_ts = []
        for ev_item in self.events:
            raw_turn_ts.append(ev_item["t"])
        for d in dialogue:
            if d.get("start_epoch") is not None:
                raw_turn_ts.append(d["start_epoch"] - t0)
        raw_turn_ts.sort()

        self.turn_timestamps: list[float] = []
        for t_val in raw_turn_ts:
            if t_val < 0:
                continue
            if not self.turn_timestamps or (t_val - self.turn_timestamps[-1]) >= 0.25:
                self.turn_timestamps.append(t_val)


    def _collect_dialogue(self, session: Path, events: list[dict]) -> list[dict]:
        """The spoken clips, each with an absolute start_epoch. Mirrors the
        timing rules in fix_recordings.py:
          - assistant reply ends exactly at its `assistant` event;
          - user input wav ends at its file mtime (closed before the event);
          - reset/wake cues start when the wav was produced (mtime / event).
        """
        clips: list[dict] = []
        seen: set[str] = set()

        def add(path: Path, start_epoch, forced_rate, event=None):
            if path.name in seen or not path.exists() or start_epoch is None:
                return
            seen.add(path.name)
            clips.append({"path": path, "start_epoch": start_epoch,
                          "forced_rate": forced_rate, "event": event})

        for e in events:
            etype = e.get("type")
            apath = e.get("audio_path")
            if etype in ("assistant", "user", "reset") and apath:
                base_name = os.path.basename(apath)
                wav = session / base_name
                if not wav.exists():
                    wav = session / "speech" / base_name
                if not wav.exists():
                    continue
                frate = get_speech_forced_rate(wav, etype, session)
                dur = wav_duration(wav, frate)
                if etype == "assistant":
                    end_epoch = parse_event_time(e.get("timestamp"))
                    start = (end_epoch - dur) if end_epoch is not None else None
                else:  # user speech (incl. the utterance that triggered a reset)
                    start = wav.stat().st_mtime - dur
                add(wav, start, frate, event=e)

        for e in events:
            if e.get("type") != "reset":
                continue
            turn = e.get("turn")
            if turn is None:
                continue
            base_name = f"turn-{int(turn):03d}-reset.wav"
            wav = session / base_name
            if not wav.exists():
                wav = session / "speech" / base_name
            if not wav.exists():
                continue
            frate = get_speech_forced_rate(wav, None, session)
            add(wav, parse_event_time(e.get("timestamp")), frate)

        wake_wavs = list(session.glob("*-wake.wav"))
        if (session / "speech").exists():
            wake_wavs.extend((session / "speech").glob("*-wake.wav"))
        for wav in sorted(wake_wavs):
            frate = get_speech_forced_rate(wav, None, session)
            add(wav, wav.stat().st_mtime, frate)

        for startup in (session / "startup.wav", session / "speech" / "startup.wav"):
            if startup.exists():
                frate = get_speech_forced_rate(startup, None, session)
                add(startup, startup.stat().st_mtime, frate)
                break

        return clips

    def _mix_audio(self, sources: list[dict], t0: float):
        if not sources:
            return
        rate = self.target_rate
        loaded = []
        total = 0
        for s in sources:
            samples = load_wav_mono(s["path"], s.get("forced_rate"), rate)
            if len(samples) == 0:
                continue
            start = max(0, int(round((s["start_epoch"] - t0) * rate)))
            loaded.append((start, samples))
            total = max(total, start + len(samples))
        if total == 0:
            return
        mix = np.zeros(total, dtype=np.float32)
        for start, samples in loaded:
            mix[start:start + len(samples)] += samples
        peak = float(np.max(np.abs(mix))) if total else 0.0
        if peak > 1.0:                     # only pull down when summing clipped
            mix /= peak
        self.master_audio = np.clip(mix * 32767.0, -32768, 32767).astype(np.int16)

    # -- lazy reads --------------------------------------------------------- #
    def read_lowstate(self, i: int) -> dict | None:
        path, off = self.low_ref[i]
        return read_json_at(path, off)

    def _read_lidar_points(self, i: int, stride: int) -> list:
        # The accumulation stride is the hot path (many scans per seek), so cache
        # those; single-scan reads (stride 1) are one file read and not cached.
        if stride == ACCUM_STRIDE and i in self._scan_cache:
            return self._scan_cache[i]
        path, off = self.lid_ref[i]
        rec = read_json_at(path, off)
        if not rec:
            return []
        pts = rec.get("points") or []
        if stride > 1:
            pts = pts[::stride]
        if stride == ACCUM_STRIDE:
            if len(self._scan_cache) > 4000:
                self._scan_cache.clear()
            self._scan_cache[i] = pts
        return pts

    # -- lookups used by the playback engine -------------------------------- #
    def chunk_at(self, t: float) -> Chunk | None:
        hit = None
        for c in self.chunks:
            if c.video_path and c.offset <= t < c.offset + c.video_duration:
                return c
            if c.video_path and c.offset <= t:
                hit = c
        return hit

    def lowstate_index_at(self, t: float) -> int:
        j = bisect.bisect_right(self.low_t, t) - 1
        return j  # -1 if none

    def detections_at(self, t: float, window: float = 1.0) -> list:
        ts = [d["t"] for d in self.detections]
        j = bisect.bisect_right(ts, t) - 1
        if j >= 0 and (t - self.detections[j]["t"]) <= window:
            return self.detections[j]["detections"]
        return []

    def lidar_single(self, t: float) -> list:
        j = bisect.bisect_right(self.lid_t, t) - 1
        if j < 0:
            return []
        stride = 1
        # cap the single scan
        pts = self._read_lidar_points(j, 1)
        if len(pts) > SINGLE_SCAN_MAX:
            step = len(pts) // SINGLE_SCAN_MAX + 1
            pts = pts[::step]
        return pts

    def lidar_accum(self, t: float) -> list:
        """Accumulated cloud up to t, bounded: sample at most ACCUM_SCAN_BUDGET
        scans evenly across the range, each strided, then hard-cap the total."""
        upto = bisect.bisect_right(self.lid_t, t)
        if upto <= 0:
            return []
        idxs = list(range(upto))
        if len(idxs) > ACCUM_SCAN_BUDGET:
            step = len(idxs) / ACCUM_SCAN_BUDGET
            idxs = [idxs[int(k * step)] for k in range(ACCUM_SCAN_BUDGET)]
        pts: list = []
        for i in idxs:
            pts.extend(self._read_lidar_points(i, ACCUM_STRIDE))
        if len(pts) > MAX_ACCUM_POINTS:
            step = len(pts) // MAX_ACCUM_POINTS + 1
            pts = pts[::step]
        return pts


# --------------------------------------------------------------------------- #
# Flask / SocketIO app
# --------------------------------------------------------------------------- #
app = Flask(__name__, template_folder='templates')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

engine = None  # PlaybackEngine, set in main()


@app.route('/')
def index():
    lidar_settings = None
    settings_file = Path(__file__).resolve().parent / 'lidar_settings.json'
    if settings_file.exists():
        try:
            lidar_settings = json.loads(settings_file.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"[Replay] Error loading lidar_settings.json: {e}")
    r_name = engine.timeline.robot_name if (engine and engine.timeline) else "SNAPPER"
    d_name = engine.timeline.device_name if (engine and engine.timeline) else "snapper.local"
    return render_template(
        'dashboard.html',
        server_lidar_settings=lidar_settings,
        robot_name=r_name,
        device_name=d_name,
        replay_mode=True,
    )


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/snapshot')
def snapshot():
    with frame_lock:
        if latest_frame is None:
            return "No frame available", 503
        ret, jpeg = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ret:
            return "Encoding error", 500
        return Response(jpeg.tobytes(), mimetype='image/jpeg')


@app.route('/lowstate')
def lowstate():
    with lowstate_lock:
        if latest_lowstate is None:
            return jsonify({}), 503
        return jsonify(latest_lowstate)


@app.route('/history')
def history():
    # Chat in replay is driven over the socket relative to the playhead, so the
    # backfill fetch the front end runs on load should start empty.
    return jsonify({"entries": []})


@app.route('/toggle_yolo', methods=['POST'])
def toggle_yolo():
    global yolo_enabled
    data = request.get_json(silent=True) or {}
    yolo_enabled = bool(data['enabled']) if 'enabled' in data else not yolo_enabled
    return jsonify({'yolo_enabled': yolo_enabled})


@app.route('/get_record_status')
def get_record_status():
    return jsonify({'is_recording': False})


@app.route('/toggle_record', methods=['POST'])
def toggle_record():
    return jsonify({'is_recording': False})


@socketio.on('ping_latency')
def handle_ping():
    socketio.emit('pong_latency')


@socketio.on('connect')
def handle_connect():
    if engine:
        engine.resync()


@socketio.on('replay_play')
def handle_play():
    if engine:
        engine.play()


@socketio.on('replay_pause')
def handle_pause():
    if engine:
        engine.pause()


@socketio.on('replay_seek')
def handle_seek(payload):
    if engine:
        data = payload or {}
        if 'accumulate' in data:
            engine.accumulate = bool(data['accumulate'])
        engine.seek(float(data.get('t', 0.0)))


@socketio.on('replay_set_accumulate')
def handle_set_accumulate(payload):
    if engine:
        engine.accumulate = bool((payload or {}).get('accumulate', False))


@socketio.on('replay_next_turn')
def handle_next_turn():
    if engine:
        engine.next_turn()


@socketio.on('replay_prev_turn')
def handle_prev_turn():
    if engine:
        engine.prev_turn()


@socketio.on('replay_next_session')
def handle_next_session():
    if engine:
        engine.next_session()


@socketio.on('replay_prev_session')
def handle_prev_session():
    if engine:
        engine.prev_session()


@socketio.on('replay_select_session')
def handle_select_session(payload):
    if engine:
        idx = (payload or {}).get('index')
        if idx is not None:
            engine.select_session(int(idx))


# --------------------------------------------------------------------------- #
# Playback engine - authoritative playhead & playlist manager
# --------------------------------------------------------------------------- #
class PlaybackEngine:
    def __init__(self, playlist: list[dict], initial_index: int, target_rate: int,
                 include_mic: bool, want_audio: bool, loop: bool):
        self.playlist = playlist
        self.playlist_index = initial_index if (playlist and 0 <= initial_index < len(playlist)) else 0
        self.target_rate = target_rate
        self.include_mic = include_mic
        self.want_audio_setting = want_audio
        self.loop = loop

        self.timeline: SessionTimeline | None = None
        self.want_audio = False

        self.playhead = 0.0
        self.playing = False
        self.accumulate = False

        self._lock = threading.Lock()
        self._seek_to = None
        self._stop = threading.Event()

        self._low_i = 0        # cursor into timeline.low_t
        self._lidar_i = 0      # cursor into timeline.lid_t
        self._event_i = 0      # cursor into timeline.events
        self._audio_cursor = 0

        self._cap = None
        self._cap_chunk = None
        self._last_state_emit = 0.0

        if self.playlist and 0 <= self.playlist_index < len(self.playlist):
            self._load_timeline_for_index(self.playlist_index)

    def _load_timeline_for_index(self, index: int) -> bool:
        if not self.playlist or index < 0 or index >= len(self.playlist):
            return False
        item = self.playlist[index]
        s_path = item.get("path")
        if not s_path or not Path(s_path).exists():
            s_path = resolve_session(item.get("session_id"))
        if not s_path or not Path(s_path).exists():
            print(f"[Replay] Warning: Session directory for '{item.get('session_id')}' not found on disk.")
            return False

        print(f"\n[Replay] Switched to Session [{index+1}/{len(self.playlist)}]: "
              f"{item.get('session_id')} ({item.get('phase')})")
        t0 = time.time()
        try:
            tl = SessionTimeline(Path(s_path), self.target_rate,
                                 include_mic=self.include_mic,
                                 want_audio=self.want_audio_setting)
        except Exception as e:
            print(f"[Replay] Could not load timeline for {item.get('session_id')}: {e}")
            return False

        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._cap_chunk = None

        self.timeline = tl
        self.playlist_index = index
        self.want_audio = self.want_audio_setting and len(tl.master_audio) > 0
        self.playhead = 0.0
        self._low_i = 0
        self._lidar_i = 0
        self._event_i = 0
        self._audio_cursor = 0

        socketio.emit('chat_reset')
        socketio.emit('lidar_reset')
        socketio.emit('session_changed', {
            'playlist_index': index,
            'session_id': tl.session_dir.name,
            'phase': item.get('phase', ''),
            'label': item.get('label', ''),
        })
        self._resync_locked(0.0)
        self._emit_state(force=True)
        print(f"[Replay] Built in {time.time() - t0:.1f}s | duration={tl.duration:.1f}s "
              f"turns={len(tl.turn_timestamps)} audio={'yes' if self.want_audio else 'none'}")
        return True

    # -- external controls -------------------------------------------------- #
    def play(self):
        with self._lock:
            if not self.timeline:
                return
            if self.playhead >= self.timeline.duration - 1e-3:
                self._request_seek(0.0)
            self.playing = True
            self._audio_start()

    def pause(self):
        with self._lock:
            self.playing = False
            socketio.emit('audio_interrupt')
        self._emit_state(force=True)

    def seek(self, t: float):
        with self._lock:
            self._request_seek(t)

    def _request_seek(self, t: float):
        if self.timeline:
            self._seek_to = max(0.0, min(t, self.timeline.duration))

    def resync(self):
        with self._lock:
            if self.timeline:
                self._resync_locked(self.playhead)
        self._emit_state(force=True)

    def next_turn(self):
        with self._lock:
            if not self.timeline or not self.timeline.turn_timestamps:
                self._next_session_locked()
                return
            ts = self.timeline.turn_timestamps
            target = next((t for t in ts if t > self.playhead + 0.1), None)
            if target is not None:
                self._request_seek(target)
            else:
                self._next_session_locked()

    def prev_turn(self):
        with self._lock:
            if not self.timeline or not self.timeline.turn_timestamps:
                self._prev_session_locked()
                return
            ts = self.timeline.turn_timestamps
            prevs = [t for t in ts if t < self.playhead - 0.8]
            if prevs:
                self._request_seek(prevs[-1])
            elif self.playhead > 0.8 and ts:
                self._request_seek(ts[0])
            else:
                self._prev_session_locked()

    def next_session(self):
        with self._lock:
            self._next_session_locked()

    def _next_session_locked(self):
        if not self.playlist:
            return
        nxt = (self.playlist_index + 1) % len(self.playlist)
        was_playing = self.playing
        if self._load_timeline_for_index(nxt):
            if was_playing:
                self.playing = True
                self._audio_start()

    def prev_session(self):
        with self._lock:
            self._prev_session_locked()

    def _prev_session_locked(self):
        if not self.playlist:
            return
        prev_idx = (self.playlist_index - 1 + len(self.playlist)) % len(self.playlist)
        was_playing = self.playing
        if self._load_timeline_for_index(prev_idx):
            if was_playing:
                self.playing = True
                self._audio_start()

    def select_session(self, index: int):
        with self._lock:
            was_playing = self.playing
            if self._load_timeline_for_index(index):
                if was_playing:
                    self.playing = True
                    self._audio_start()

    # -- worker ------------------------------------------------------------- #
    def start(self):
        threading.Thread(target=self._run, daemon=True, name="Playback").start()

    def stop(self):
        self._stop.set()

    def _run(self):
        last = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            dt_wall = now - last
            last = now
            seeked = False
            with self._lock:
                if self._seek_to is not None:
                    target = self._seek_to
                    self._seek_to = None
                    self._apply_seek(target)
                    seeked = True
                elif self.playing:
                    self._advance(dt_wall)
                self._update_video()
            if seeked:
                last = time.monotonic()
            self._emit_state()
            time.sleep(0.02)

    # -- forward playback --------------------------------------------------- #
    def _advance(self, dt_wall: float):
        tl = self.timeline
        if not tl:
            return
        new_head = min(self.playhead + dt_wall, tl.duration)
        self._dispatch_forward(new_head)
        self.playhead = new_head
        if self.playhead >= tl.duration - 1e-3:
            if self.playlist and self.playlist_index < len(self.playlist) - 1 and not self.loop:
                print(f"[Replay] Session {tl.session_dir.name} finished. Auto-advancing to next session...")
                self._next_session_locked()
            elif self.loop:
                self._apply_seek(0.0)
                self.playing = True
            else:
                self.playing = False
                socketio.emit('audio_interrupt')

    def _dispatch_forward(self, t_to: float):
        tl = self.timeline
        if not tl:
            return
        # telemetry - emit the newest sample crossed
        newest = -1
        while self._low_i < len(tl.low_t) and tl.low_t[self._low_i] <= t_to:
            newest = self._low_i
            self._low_i += 1
        if newest >= 0:
            payload = tl.read_lowstate(newest)
            if payload:
                self._emit_lowstate(payload)
        # lidar - emit scans crossed
        crossed = []
        while self._lidar_i < len(tl.lid_t) and tl.lid_t[self._lidar_i] <= t_to:
            crossed.append(self._lidar_i)
            self._lidar_i += 1
        if crossed:
            if self.accumulate:
                for i in crossed[-10:]:
                    pts = tl._read_lidar_points(i, ACCUM_STRIDE)
                    socketio.emit('lidar_data', {'points': pts, 'point_count': len(pts)})
            else:
                pts = tl._read_lidar_points(crossed[-1], 1)
                if len(pts) > SINGLE_SCAN_MAX:
                    pts = pts[::len(pts) // SINGLE_SCAN_MAX + 1]
                socketio.emit('lidar_data', {'points': pts, 'point_count': len(pts)})
        # chat - emit each dialogue line crossed
        while self._event_i < len(tl.events) and tl.events[self._event_i]["t"] <= t_to:
            socketio.emit('log_data', tl.events[self._event_i]["record"])
            self._event_i += 1
        # audio - stream the elapsed slice of the master mix in real time
        if self.want_audio and self.playing:
            self._emit_audio(t_to)

    # -- seeking / resync --------------------------------------------------- #
    def _apply_seek(self, target: float):
        tl = self.timeline
        if not tl:
            return
        self.playhead = target
        self._low_i = bisect.bisect_right(tl.low_t, target)
        self._lidar_i = bisect.bisect_right(tl.lid_t, target)
        self._event_i = bisect.bisect_right([e["t"] for e in tl.events], target)
        self._audio_cursor = int(target * tl.target_rate)
        self._cap_chunk = None            # force video re-seek
        socketio.emit('audio_interrupt')
        self._resync_locked(target)
        if self.playing and self.want_audio:
            self._audio_start()

    def _resync_locked(self, t: float):
        """Rebuild the client's view (telemetry, lidar, chat) at time t."""
        tl = self.timeline
        if not tl:
            return
        j = tl.lowstate_index_at(t)
        if j >= 0:
            payload = tl.read_lowstate(j)
            if payload:
                self._emit_lowstate(payload)
        socketio.emit('lidar_reset')
        pts = tl.lidar_accum(t) if self.accumulate else tl.lidar_single(t)
        if pts:
            socketio.emit('lidar_data', {'points': pts, 'point_count': len(pts)})
        socketio.emit('chat_reset')
        for ev in tl.events:
            if ev["t"] <= t:
                socketio.emit('log_data', ev["record"])
            else:
                break

    # -- video -------------------------------------------------------------- #
    def _update_video(self):
        global latest_frame, latest_detections, latest_detection_time
        tl = self.timeline
        if not tl:
            return
        t = self.playhead
        chunk = tl.chunk_at(t)
        if chunk is not None and chunk.video_path:
            if self._cap_chunk is not chunk:
                if self._cap is not None:
                    self._cap.release()
                self._cap = cv2.VideoCapture(chunk.video_path)
                self._cap_chunk = chunk
                self._cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, (t - chunk.offset) * 1000.0))
                ret, frame = self._cap.read()
                if ret:
                    with frame_lock:
                        latest_frame = frame
            else:
                target_ms = max(0.0, (t - chunk.offset) * 1000.0)
                frame = None
                guard = 0
                while self._cap and self._cap.get(cv2.CAP_PROP_POS_MSEC) <= target_ms and guard < 240:
                    ret, f = self._cap.read()
                    guard += 1
                    if not ret:
                        break
                    frame = f
                if frame is not None:
                    with frame_lock:
                        latest_frame = frame
        dets = tl.detections_at(t)
        with detections_lock:
            latest_detections = dets
            latest_detection_time = time.time()

    # -- emit helpers ------------------------------------------------------- #
    def _emit_lowstate(self, payload: dict):
        global latest_lowstate
        with lowstate_lock:
            latest_lowstate = payload
        socketio.emit('telemetry_data', payload)

    def _audio_start(self):
        if self.want_audio and self.timeline:
            self._audio_cursor = int(self.playhead * self.timeline.target_rate)
            socketio.emit('audio_start', {'sample_rate': self.timeline.target_rate})

    def _emit_audio(self, t_to: float):
        if not self.timeline:
            return
        rate = self.timeline.target_rate
        target_sample = int(t_to * rate)
        master = self.timeline.master_audio
        end = min(target_sample, len(master))
        if end <= self._audio_cursor:
            self._audio_cursor = max(self._audio_cursor, target_sample)
            return
        chunk = master[self._audio_cursor:end]
        self._audio_cursor = target_sample
        if len(chunk):
            socketio.emit('audio_chunk', {
                'data': base64.b64encode(chunk.tobytes()).decode('ascii'),
                'sample_rate': rate,
            })

    def _emit_state(self, force: bool = False):
        now = time.monotonic()
        if not force and now - self._last_state_emit < 0.1:
            return
        self._last_state_emit = now
        tl = self.timeline
        curr = self.playlist[self.playlist_index] if (self.playlist and 0 <= self.playlist_index < len(self.playlist)) else {}
        turn_idx = 0
        if tl and tl.turn_timestamps:
            turn_idx = bisect.bisect_right(tl.turn_timestamps, self.playhead)

        socketio.emit('replay_state', {
            'playhead': self.playhead if tl else 0.0,
            'duration': tl.duration if tl else 0.0,
            'playing': self.playing,
            'playlist_index': self.playlist_index,
            'playlist_total': len(self.playlist),
            'session_id': tl.session_dir.name if tl else curr.get('session_id', ''),
            'phase': curr.get('phase', ''),
            'label': curr.get('label', ''),
            'turn_index': turn_idx,
            'turns_count': len(tl.turn_timestamps) if tl else 0,
            'playlist': [{
                'index': i['index'],
                'session_id': i['session_id'],
                'phase': i['phase'],
                'label': i['label'],
                'available': i['path'] is not None
            } for i in self.playlist] if self.playlist else [],
        })


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    global engine
    parser = argparse.ArgumentParser(description="BFF recorded-session timeline replay")
    parser.add_argument("--session", type=str, default=None,
                        help="Session dir name, index (1-41), or path (default: 41 curated sessions)")
    parser.add_argument("--all", action="store_true", help="Play all 41 curated sessions sequentially from session 1")
    parser.add_argument("--port", type=int, default=None, help="Dashboard port")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio playback")
    parser.add_argument("--mic-audio", action="store_true",
                        help="Also mix in the robot-mic ambient (chunk_*/audio.wav)")
    parser.add_argument("--target-rate", type=int, default=DEFAULT_TARGET_RATE,
                        help="Mixed-audio delivery rate (default: %(default)s)")
    parser.add_argument("--loop", action="store_true", help="Loop current session at the end")
    parser.add_argument("--autoplay", action="store_true", help="Start playing immediately")
    args = parser.parse_args()

    if args.port is None:
        args.port = int(os.getenv("BFF_DASHBOARD_PORT", "8080"))

    # Load 41 curated playlist
    playlist = load_curated_playlist()
    initial_index = 0

    if args.session:
        target_str = str(args.session).strip()
        # Check if integer index 1..N
        if target_str.isdigit():
            val = int(target_str) - 1
            if 0 <= val < len(playlist):
                initial_index = val
        else:
            # Check matching session_id substring in playlist
            match_idx = next((i for i, item in enumerate(playlist) if target_str.lower() in item["session_id"].lower()), None)
            if match_idx is not None:
                initial_index = match_idx
            else:
                # Custom session directory path
                cand = resolve_session(args.session)
                if cand:
                    custom_item = {
                        "index": len(playlist),
                        "session_id": cand.name,
                        "phase": "custom",
                        "label": cand.name,
                        "machine": "",
                        "path": cand,
                        "turns_count": 0,
                    }
                    playlist.append(custom_item)
                    initial_index = len(playlist) - 1
    elif not playlist:
        # Fallback if index json not found
        default_cand = resolve_session(None)
        if default_cand:
            playlist = [{
                "index": 0,
                "session_id": default_cand.name,
                "phase": "custom",
                "label": default_cand.name,
                "machine": "",
                "path": default_cand,
                "turns_count": 0,
            }]

    if not playlist:
        print("[Replay] Error: no capture session or playlist found. Pass --session or set BFF_LOG_ROOT.")
        sys.exit(1)

    want_audio = not args.no_audio
    engine = PlaybackEngine(playlist, initial_index=initial_index, target_rate=args.target_rate,
                            include_mic=args.mic_audio, want_audio=want_audio, loop=args.loop)
    engine.start()
    if args.autoplay:
        engine.play()

    curr_item = playlist[engine.playlist_index]
    print("\n=======================================================")
    print(f"BFF Replay serving at: http://localhost:{args.port}")
    print(f"Playlist: {len(playlist)} curated sessions")
    print(f"Initial Session [{engine.playlist_index+1}/{len(playlist)}]: {curr_item['session_id']} ({curr_item.get('phase')})")
    print("=======================================================\n")

    def _stop_on_signal(signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop_on_signal)
    try:
        socketio.run(app, host='0.0.0.0', port=args.port, debug=False,
                     use_reloader=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n[Replay] Shutting down.")
    finally:
        engine.stop()


if __name__ == "__main__":
    main()

