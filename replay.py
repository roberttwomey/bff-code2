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

    def _build(self, include_mic: bool, want_audio: bool):
        session = self.session_dir
        events = _read_jsonl(session / "session.jsonl")
        self._read_identity(events)

        chunk_dirs = sorted([d for d in session.glob("chunk_*") if d.is_dir()],
                            key=lambda d: int(d.name.split('_')[1]))
        if not chunk_dirs and _session_has_video(session):
            chunk_dirs = [session]  # legacy single-directory session

        cache = IndexCache(session)

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

            video_path = str(cdir / "video.mp4") if (cdir / "video.mp4").exists() else None
            vdur, fps = 0.0, 30.0
            if video_path:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                    vdur = (nfr / fps) if fps else 0.0
                else:
                    # A chunk truncated by a hard shutdown (no moov atom) can't be
                    # decoded; drop it as a video source so we don't retry-open it
                    # on every playhead tick. Its telemetry/lidar still index fine.
                    print(f"[Replay] {cdir.name}/video.mp4 is unreadable "
                          f"(truncated?); skipping its video.")
                    video_path = None
                cap.release()
            raw.append({"dir": cdir, "low_p": low_p, "lid_p": lid_p, "det_p": det_p,
                        "start_epoch": start_epoch, "video_path": video_path,
                        "vdur": vdur, "fps": fps})

        # Fill a missing anchor from the previous chunk's end.
        for i, rc in enumerate(raw):
            if rc["start_epoch"] is None and i > 0 and raw[i - 1]["start_epoch"] is not None:
                rc["start_epoch"] = raw[i - 1]["start_epoch"] + raw[i - 1]["vdur"]

        dialogue = self._collect_dialogue(session, events) if want_audio else []

        anchors = [rc["start_epoch"] for rc in raw if rc["start_epoch"]]
        anchors += [c["start_epoch"] for c in dialogue]
        for e in events:
            if e.get("type") in ("user", "assistant", "assistant_chunk"):
                et = parse_event_time(e.get("timestamp"))
                if et:
                    anchors.append(et)
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

        for e in events:
            if e.get("type") not in ("user", "assistant", "assistant_chunk"):
                continue
            et = parse_event_time(e.get("timestamp"))
            if et is None:
                continue
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

    def _collect_dialogue(self, session: Path, events: list[dict]) -> list[dict]:
        """The spoken clips, each with an absolute start_epoch. Mirrors the
        timing rules in fix_recordings.py:
          - assistant reply ends exactly at its `assistant` event;
          - user input wav ends at its file mtime (closed before the event);
          - reset/wake cues start when the wav was produced (mtime / event).
        """
        clips: list[dict] = []
        seen: set[str] = set()

        def add(path: Path, start_epoch, forced_rate):
            if path.name in seen or not path.exists() or start_epoch is None:
                return
            seen.add(path.name)
            clips.append({"path": path, "start_epoch": start_epoch,
                          "forced_rate": forced_rate})

        for e in events:
            etype = e.get("type")
            apath = e.get("audio_path")
            if etype in ("assistant", "user", "reset") and apath:
                wav = session / os.path.basename(apath)
                if not wav.exists():
                    continue
                dur = wav_duration(wav, SPEECH_TRUE_RATE)
                if etype == "assistant":
                    end_epoch = parse_event_time(e.get("timestamp"))
                    start = (end_epoch - dur) if end_epoch is not None else None
                else:  # user speech (incl. the utterance that triggered a reset)
                    start = wav.stat().st_mtime - dur
                add(wav, start, SPEECH_TRUE_RATE)

        for e in events:
            if e.get("type") != "reset":
                continue
            turn = e.get("turn")
            if turn is None:
                continue
            wav = session / f"turn-{int(turn):03d}-reset.wav"
            add(wav, parse_event_time(e.get("timestamp")), SPEECH_TRUE_RATE)

        for wav in sorted(session.glob("*-wake.wav")):
            add(wav, wav.stat().st_mtime, SPEECH_TRUE_RATE)
        startup = session / "startup.wav"
        if startup.exists():
            add(startup, startup.stat().st_mtime, SPEECH_TRUE_RATE)

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
    return render_template(
        'dashboard.html',
        server_lidar_settings=lidar_settings,
        robot_name=engine.timeline.robot_name,
        device_name=engine.timeline.device_name,
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


# --------------------------------------------------------------------------- #
# Playback engine - the one authoritative playhead
# --------------------------------------------------------------------------- #
class PlaybackEngine:
    def __init__(self, timeline: SessionTimeline, loop: bool, want_audio: bool):
        self.timeline = timeline
        self.loop = loop
        self.want_audio = want_audio and len(timeline.master_audio) > 0

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

    # -- external controls -------------------------------------------------- #
    def play(self):
        with self._lock:
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
        self._seek_to = max(0.0, min(t, self.timeline.duration))

    def resync(self):
        with self._lock:
            self._resync_locked(self.playhead)
        self._emit_state(force=True)

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
            # A seek's resync (rebuilding an accumulated cloud can read hundreds
            # of scans) may take seconds while the lock is held. Rebase the clock
            # so that stall isn't charged to the next tick as elapsed playback -
            # otherwise a seek-while-playing lurches the playhead forward.
            if seeked:
                last = time.monotonic()
            self._emit_state()
            time.sleep(0.02)

    # -- forward playback --------------------------------------------------- #
    def _advance(self, dt_wall: float):
        tl = self.timeline
        new_head = min(self.playhead + dt_wall, tl.duration)
        self._dispatch_forward(new_head)
        self.playhead = new_head
        if self.playhead >= tl.duration - 1e-3:
            if self.loop:
                self._apply_seek(0.0)
                self.playing = True
            else:
                self.playing = False
                socketio.emit('audio_interrupt')

    def _dispatch_forward(self, t_to: float):
        tl = self.timeline
        # telemetry - emit the newest sample crossed
        newest = -1
        while self._low_i < len(tl.low_t) and tl.low_t[self._low_i] <= t_to:
            newest = self._low_i
            self._low_i += 1
        if newest >= 0:
            payload = tl.read_lowstate(newest)
            if payload:
                self._emit_lowstate(payload)
        # lidar - emit scans crossed (cap the burst after a lag/fast-forward)
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
        t = self.playhead
        chunk = tl.chunk_at(t)
        if chunk is not None:
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
                while self._cap.get(cv2.CAP_PROP_POS_MSEC) <= target_ms and guard < 240:
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
        if self.want_audio:
            self._audio_cursor = int(self.playhead * self.timeline.target_rate)
            socketio.emit('audio_start', {'sample_rate': self.timeline.target_rate})

    def _emit_audio(self, t_to: float):
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
        socketio.emit('replay_state', {
            'playhead': self.playhead,
            'duration': self.timeline.duration,
            'playing': self.playing,
        })


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    global engine
    parser = argparse.ArgumentParser(description="BFF recorded-session timeline replay")
    parser.add_argument("--session", type=str, default=None,
                        help="Session dir name or path (default: newest with video)")
    parser.add_argument("--port", type=int, default=None, help="Dashboard port")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio playback")
    parser.add_argument("--mic-audio", action="store_true",
                        help="Also mix in the robot-mic ambient (chunk_*/audio.wav)")
    parser.add_argument("--target-rate", type=int, default=DEFAULT_TARGET_RATE,
                        help="Mixed-audio delivery rate (default: %(default)s)")
    parser.add_argument("--loop", action="store_true", help="Loop at the end")
    parser.add_argument("--autoplay", action="store_true", help="Start playing immediately")
    args = parser.parse_args()

    if args.port is None:
        args.port = int(os.getenv("BFF_DASHBOARD_PORT", "8080"))

    session_dir = resolve_session(args.session)
    if not session_dir:
        print("[Replay] Error: no capture session found. "
              "Pass --session or set BFF_LOG_ROOT.")
        sys.exit(1)

    print(f"[Replay] Building timeline for: {session_dir}")
    print("[Replay] (first run indexes the logs; later runs use the cache)")
    want_audio = not args.no_audio
    t0 = time.time()
    try:
        timeline = SessionTimeline(session_dir, args.target_rate,
                                   include_mic=args.mic_audio, want_audio=want_audio)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Replay] Error: could not build timeline: {e}")
        sys.exit(1)

    print(f"[Replay] built in {time.time() - t0:.1f}s  "
          f"device={timeline.device_name} robot={timeline.robot_name}")
    print(f"[Replay] duration={timeline.duration:.1f}s  chunks={len(timeline.chunks)}  "
          f"lowstate={len(timeline.low_t)}  lidar={len(timeline.lid_t)}  "
          f"detections={len(timeline.detections)}  events={len(timeline.events)}  "
          f"audio={'%.1fs' % (len(timeline.master_audio) / args.target_rate) if len(timeline.master_audio) else 'none'}")

    engine = PlaybackEngine(timeline, loop=args.loop, want_audio=want_audio)
    engine.start()
    if args.autoplay:
        engine.play()

    print("\n=======================================================")
    print(f"BFF Replay serving at: http://localhost:{args.port}")
    print(f"Replaying: {session_dir.name}  ({timeline.duration:.1f}s)")
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
