#!/usr/bin/env python3
"""Repair and post-process BFF session recordings.

Walks recent session folders under ~/bff/logs and produces a clean set of
derived outputs under ~/bff/logs-processed, without ever modifying the
originals:

  1. repair   - rebuild the index ("moov" atom) of any truncated chunk video.
                capture_go2_data.py writes video.mp4 with OpenCV, which only
                flushes the moov atom on writer.release(); a hard shutdown
                leaves the last chunk as ftyp+mdat with no index.
  2. plate    - concatenate every chunk into one clean plate, stream-copied
                (no re-encode).
  3. overlay  - render the YOLO boxes over the clean plate using
                post_processing/overlay_detections.py.
  4. srt      - subtitles for the VLM scene descriptions (and the spoken
                dialogue), synchronised to the clean plate's timeline.
  5. speech   - relabel the synthesised speech wavs, whose headers claim
                22050 Hz while the samples are really 48000 Hz.

Usage:
    ./fix_recordings.py                    # most recent session, all steps
    ./fix_recordings.py --last 3           # three most recent sessions
    ./fix_recordings.py --session session-20260723-092744
    ./fix_recordings.py --overlay-seconds 20   # quick end-to-end check
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_LOGS = Path("~/bff/logs").expanduser()
DEFAULT_OUT = Path("~/bff/logs-processed").expanduser()

# Piper reports its voice config rate (22050) in the wav header, but the
# samples that actually get written are the resampled 48 kHz playback stream.
SPEECH_WRONG_RATE = 22050
SPEECH_TRUE_RATE = 48000

ALL_STEPS = ("repair", "plate", "overlay", "srt", "speech", "audio",
             "fulldialogue")

# Each track is one .srt; '+' stacks several sources into the same file.
DEFAULT_TRACKS = "speech,yolo+vlm,bodystate,speech+yolo+vlm+bodystate"

# ffprobe/ffmpeg fall back to this when a chunk's frame rate can't be read.
FALLBACK_FPS = 30.0


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def run(cmd, capture_stdout_bytes=False, **kw):
    """Run a command, capturing output. Never raises on non-zero exit.

    With capture_stdout_bytes, returns raw stdout (or None on failure)
    instead of the CompletedProcess — for piping binary out of ffmpeg.
    """
    argv = [str(c) for c in cmd]
    if capture_stdout_bytes:
        proc = subprocess.run(argv, capture_output=True, **kw)
        return proc.stdout if proc.returncode == 0 else None
    return subprocess.run(argv, capture_output=True, text=True, **kw)


def natural_key(name: str):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def human_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_rate(value: str) -> float | None:
    """Parse an ffprobe rational such as '30/1'."""
    if not value or value in ("0/0", "N/A"):
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            num, den = float(num), float(den)
        except ValueError:
            return None
        return num / den if den else None
    try:
        return float(value)
    except ValueError:
        return None


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------

def probe_video(path: Path) -> dict:
    """Return video stream facts. 'ok' is False when the file has no usable index."""
    info = {
        "ok": False, "path": str(path), "width": None, "height": None,
        "fps": None, "frames": 0, "timescale": None, "codec": None,
        "duration": None, "error": None,
    }
    if not path.exists() or path.stat().st_size == 0:
        info["error"] = "missing or empty"
        return info

    res = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,time_base",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1", path,
    ])
    if res.returncode != 0:
        info["error"] = (res.stderr or "ffprobe failed").strip().splitlines()[-1]
        return info

    fields = {}
    for line in res.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k] = v.strip()

    info["codec"] = fields.get("codec_name")
    for key in ("width", "height"):
        try:
            info[key] = int(fields.get(key, ""))
        except ValueError:
            pass
    info["fps"] = parse_rate(fields.get("avg_frame_rate", "")) or \
        parse_rate(fields.get("r_frame_rate", ""))
    try:
        info["duration"] = float(fields.get("duration", ""))
    except ValueError:
        pass

    tb = fields.get("time_base", "")
    if "/" in tb:
        try:
            info["timescale"] = int(tb.split("/", 1)[1])
        except ValueError:
            pass

    try:
        info["frames"] = int(fields.get("nb_frames", ""))
    except ValueError:
        info["frames"] = 0

    if info["width"] and info["height"]:
        info["ok"] = True
    else:
        info["error"] = info["error"] or "no decodable video stream"
    return info


def count_frames(path: Path) -> int:
    """Exact frame count by demuxing. Slower than reading nb_frames."""
    res = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path,
    ])
    try:
        return int(res.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0


def get_extradata(path: Path) -> bytes:
    """Pull the MPEG-4 decoder config (VOS/VOL header) out of a healthy mp4.

    OpenCV's mp4v writer stores this only in the moov's esds box, never
    in-band, so a truncated file has no way to declare its own dimensions.
    """
    res = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=extradata", "-show_data",
        "-of", "default=noprint_wrappers=1", path,
    ])
    out = []
    for line in res.stdout.splitlines():
        m = re.match(r"^([0-9a-f]{8}): (.*)$", line)
        if m:
            # ffprobe's hexdump is a fixed 40-column hex field, then ASCII.
            out += [int(h, 16) for h in re.findall(r"[0-9a-f]{2}", m.group(2)[:40])]
    return bytes(out)


def mdat_payload_offset(path: Path) -> int | None:
    """Byte offset of the raw elementary stream inside an unindexed mp4."""
    size = path.stat().st_size
    with path.open("rb") as f:
        off = 0
        while off < size:
            f.seek(off)
            hdr = f.read(8)
            if len(hdr) < 8:
                return None
            box_size = struct.unpack(">I", hdr[:4])[0]
            box_type = hdr[4:8]
            if box_type == b"mdat":
                if box_size == 1:      # 64-bit extended size
                    return off + 16
                return off + 8         # size 0 means "to end of file"
            if box_size == 0:
                return None
            off += box_size
    return None


# --------------------------------------------------------------------------
# step 1 - repair truncated chunk videos
# --------------------------------------------------------------------------

def repair_video(bad: Path, donor: Path, out_path: Path, fps: float,
                 timescale: int | None, verbose: bool = True) -> dict:
    """Rebuild a playable mp4 from a chunk that lost its moov atom.

    The mdat payload is a raw MPEG-4 Part 2 stream; prefixing it with the
    donor's decoder config makes it a valid .m4v that ffmpeg can remux.
    """
    result = {"repaired": False, "output": None, "frames": 0, "error": None}

    payload_at = mdat_payload_offset(bad)
    if payload_at is None:
        result["error"] = "no mdat box found; file is not recoverable"
        return result

    extradata = get_extradata(donor)
    if not extradata:
        result["error"] = f"could not read decoder config from donor {donor}"
        return result

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".m4v")
    with bad.open("rb") as src, tmp.open("wb") as dst:
        src.seek(payload_at)
        dst.write(extradata)
        shutil.copyfileobj(src, dst, 1024 * 1024)

    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "m4v",
           "-framerate", f"{fps:g}", "-i", tmp, "-c", "copy"]
    # Match the sibling chunks' timescale, otherwise the concat demuxer
    # rescales this segment's timestamps and inflates the plate's duration.
    if timescale:
        cmd += ["-video_track_timescale", str(timescale)]
    cmd.append(out_path)
    res = run(cmd)
    tmp.unlink(missing_ok=True)

    if res.returncode != 0:
        result["error"] = (res.stderr or "ffmpeg remux failed").strip().splitlines()[-1]
        return result

    info = probe_video(out_path)
    if not info["ok"]:
        result["error"] = f"repaired file still unreadable: {info['error']}"
        return result

    frames = info["frames"] or count_frames(out_path)
    result.update(repaired=True, output=str(out_path), frames=frames)
    if verbose:
        print(f"      recovered {frames} frames ({frames / fps:.1f}s) -> {out_path.name}")
    return result


def check_wav(path: Path) -> dict:
    """Report a wav whose declared data size is short of the bytes on disk."""
    try:
        with path.open("rb") as f:
            head = f.read(12)
            if len(head) < 12 or head[:4] != b"RIFF":
                return {"ok": False, "error": "not a RIFF file"}
            size = path.stat().st_size
            off = 12
            while off + 8 <= size:
                f.seek(off)
                ck = f.read(8)
                if len(ck) < 8:
                    break
                cid = ck[:4]
                clen = struct.unpack("<I", ck[4:8])[0]
                if cid == b"data":
                    slack = size - (off + 8 + clen)
                    return {"ok": True, "data_bytes": clen, "trailing_bytes": slack}
                off += 8 + clen + (clen & 1)
    except OSError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": False, "error": "no data chunk"}


# --------------------------------------------------------------------------
# chunk discovery
# --------------------------------------------------------------------------

def discover_chunks(session_dir: Path) -> list[dict]:
    dirs = sorted(
        (d for d in session_dir.iterdir()
         if d.is_dir() and d.name.startswith("chunk_")),
        key=lambda d: natural_key(d.name),
    )
    chunks = []
    for d in dirs:
        chunks.append({
            "name": d.name,
            "dir": d,
            "video": d / "video.mp4",
            "detections": d / "detections.jsonl",
            "audio": d / "audio.wav",
        })
    return chunks


def find_external_donor(logs_dir: Path, exclude: Path) -> dict | None:
    """Look through other sessions for a healthy chunk to borrow config from.

    A session that died before its first chunk rotated has no intact video of
    its own, but every recording comes off the same camera and encoder, so a
    sibling session's decoder config applies.
    """
    try:
        sessions = sorted(
            (d for d in logs_dir.iterdir()
             if d.is_dir() and d.name.startswith("session-")
             and d.resolve() != exclude.resolve()),
            key=lambda d: d.name, reverse=True,
        )
    except OSError:
        return None

    for session in sessions:
        for chunk in sorted((d for d in session.iterdir()
                             if d.is_dir() and d.name.startswith("chunk_")),
                            key=lambda d: natural_key(d.name)):
            video = chunk / "video.mp4"
            if not video.exists():
                continue
            info = probe_video(video)
            if info["ok"] and info["frames"]:
                return {"video": video, "info": info}
    return None


def prepare_chunks(chunks: list[dict], out_dir: Path, do_repair: bool,
                   report: dict, logs_dir: Path, session_dir: Path
                   ) -> tuple[list[dict], float]:
    """Probe every chunk, repair the broken ones, and resolve frame counts.

    Returns the chunks that can go into the plate, plus the session fps.
    """
    print("  probing chunks...")
    for c in chunks:
        c["info"] = probe_video(c["video"])

    healthy = [c for c in chunks if c["info"]["ok"]]
    broken = [c for c in chunks if not c["info"]["ok"]]

    if healthy:
        donor = healthy[0]
    else:
        donor = find_external_donor(logs_dir, session_dir)
        if donor is None:
            print("    no readable chunk video here or in any other session; "
                  "cannot rebuild an index")
            return [], FALLBACK_FPS
        note = (f"no intact video in this session; borrowing decoder config "
                f"from {Path(donor['video']).parent.parent.name}/"
                f"{Path(donor['video']).parent.name} "
                f"({donor['info']['width']}x{donor['info']['height']})")
        print(f"    {note}")
        report["warnings"].append(note)

    fps = donor["info"]["fps"] or FALLBACK_FPS
    timescale = donor["info"]["timescale"]
    print(f"    {len(healthy)} readable, {len(broken)} damaged "
          f"({donor['info']['width']}x{donor['info']['height']} @ {fps:g} fps)")

    for c in broken:
        report["repairs"].append({"chunk": c["name"], "reason": c["info"]["error"]})
        print(f"    [damaged] {c['name']}: {c['info']['error']}")
        fixed = out_dir / "repaired" / c["name"] / "video.mp4"

        # Reuse an earlier repair so that running a single step (e.g. only
        # 'overlay') still sees the same chunk list, and therefore the same
        # frame offsets, as a full run.
        if fixed.exists():
            prior = probe_video(fixed)
            if prior["ok"]:
                c["video"], c["info"] = fixed, prior
                report["repairs"][-1].update(repaired=True, reused=True,
                                             output=str(fixed))
                print(f"      reusing previous repair ({prior['frames']} frames)")
                continue

        if not do_repair:
            print("      not repaired ('repair' step not selected)")
            continue
        res = repair_video(c["video"], donor["video"], fixed, fps, timescale)
        report["repairs"][-1].update(res)
        if res["repaired"]:
            c["video"] = fixed
            c["info"] = probe_video(fixed)
        else:
            print(f"      repair failed: {res['error']}")

    usable = []
    for c in chunks:
        if not c["info"]["ok"]:
            print(f"    [skipped] {c['name']} is unusable and will be left "
                  f"out of the plate")
            continue
        frames = c["info"]["frames"] or count_frames(c["video"])
        if frames <= 0:
            print(f"    [skipped] {c['name']} reports no frames")
            continue
        c["frames"] = frames
        usable.append(c)

    # Offsets: 'global' walks every chunk that exists (this is the counter
    # capture_go2_data.py uses for frame_index); 'plate' walks only the chunks
    # that make it into the concatenated output.
    g = p = 0
    for c in chunks:
        c["global_offset"] = g
        g += c.get("frames", 0)
    for c in usable:
        c["plate_offset"] = p
        p += c["frames"]

    report["plate_frames"] = p
    report["fps"] = fps
    return usable, fps


# --------------------------------------------------------------------------
# step 2 - clean plate
# --------------------------------------------------------------------------

def build_plate(chunks: list[dict], out_path: Path, work_dir: Path) -> bool:
    """Concatenate the chunk videos, stream-copied. Video only — the audio
    step lays its tracks on afterwards at absolute offsets."""
    listing = work_dir / "concat_video.txt"
    listing.parent.mkdir(parents=True, exist_ok=True)
    with listing.open("w") as f:
        for c in chunks:
            f.write(f"file '{Path(c['video']).resolve()}'\n")

    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
           "-i", listing, "-c", "copy", out_path]

    res = run(cmd)
    if res.returncode != 0:
        print(f"    concat failed: {(res.stderr or '').strip().splitlines()[-1:]}")
        return False
    return True


# --------------------------------------------------------------------------
# step 3 - merged detections + overlay
# --------------------------------------------------------------------------

def estimate_session_base(chunks: list[dict]) -> int:
    """Global frame index that the first surviving chunk starts at.

    In circular mode capture_go2_data.py prunes old chunks but never resets
    frame_index, so a session's first surviving chunk can start at 96955.
    Each chunk's detections live in [base + offset, base + offset + frames),
    so every chunk gives an upper bound on base; the tightest one wins.
    """
    bounds = []
    for c in chunks:
        path = c["detections"]
        if not path.exists():
            continue
        lo = None
        with path.open(encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= 5:        # written in frame order; the head is enough
                    break
                try:
                    idx = json.loads(line).get("frame_index")
                except json.JSONDecodeError:
                    continue
                if isinstance(idx, int) and (lo is None or idx < lo):
                    lo = idx
        if lo is not None:
            bounds.append(lo - c["global_offset"])
    return min(bounds) if bounds else 0


def merge_detections(chunks: list[dict], fps: float, out_path: Path,
                     report: dict) -> dict:
    """Rewrite every chunk's detections into clean-plate frame numbers.

    frame_index is a single counter shared across chunks in current
    recordings, but older sessions restart it per chunk; both are handled.
    Chunk wall-clock start times are estimated here too, for the subtitles.
    """
    total_in = total_out = 0
    anchors = []
    session_base = estimate_session_base(chunks)
    if session_base:
        print(f"    global frame counter starts at {session_base} "
              f"(earlier chunks were pruned)")

    with out_path.open("w", encoding="utf-8") as out:
        for c in chunks:
            det_path = c["detections"]
            if not det_path.exists():
                continue

            records = []
            for line in det_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a torn final line on an interrupted session
            if not records:
                continue
            total_in += len(records)

            indices = [r.get("frame_index", 0) for r in records]
            lo, n = min(indices), c["frames"]

            # Try the session-wide global counter first, then a per-chunk
            # counter (older recordings), then this chunk's own first index.
            # Whichever places the most records inside the chunk wins, so a
            # pruned or renumbered session can't silently drop everything.
            candidates = [
                ("global", session_base + c["global_offset"]),
                ("per-chunk", 0),
                ("rebased", lo),
            ]
            best_mode, base, best_fit = None, 0, -1
            for mode, cand in candidates:
                fit = sum(1 for i in indices if 0 <= i - cand < n)
                if fit > best_fit:
                    best_mode, base, best_fit = mode, cand, fit
            if best_fit <= 0:
                print(f"    [skipped] {c['name']}: no detection index scheme "
                      f"fits {n} frames (indices {lo}..{max(indices)})")
                continue
            c["index_mode"] = best_mode

            offsets = []
            for r in records:
                rel = r.get("frame_index", 0) - base
                if not (0 <= rel < n):
                    continue
                r["frame_index"] = c["plate_offset"] + rel
                r["chunk"] = c["name"]
                ts = r.get("timestamp")
                if isinstance(ts, (int, float)):
                    # Logged after inference, so this always overshoots the
                    # true capture time; the low percentile is the best guess.
                    offsets.append(ts - rel / fps)
                out.write(json.dumps(r) + "\n")
                total_out += 1

            if offsets:
                offsets.sort()
                start_epoch = offsets[max(0, int(len(offsets) * 0.05) - 1)]
                anchors.append({
                    "chunk": c["name"],
                    "start_epoch": start_epoch,
                    "end_epoch": start_epoch + n / fps,
                    "plate_start": c["plate_offset"] / fps,
                    "plate_end": (c["plate_offset"] + n) / fps,
                })

    report["detections"] = {"read": total_in, "written": total_out,
                            "anchors": len(anchors)}
    print(f"    {total_out} detection records mapped onto the plate")
    return {"anchors": anchors}


def run_overlay(plate: Path, detections: Path, out_path: Path, work_dir: Path,
                limit_seconds: float | None) -> bool:
    """Drive post_processing/overlay_detections.py over the clean plate."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from post_processing.overlay_detections import overlay_detections
    except ImportError as e:
        print(f"    cannot import post_processing.overlay_detections: {e}")
        return False

    stage = work_dir / "overlay"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    source = plate
    if limit_seconds:
        clipped = stage / "clipped.mp4"
        res = run(["ffmpeg", "-y", "-v", "error", "-t", f"{limit_seconds:g}",
                   "-i", plate, "-c", "copy", clipped])
        if res.returncode == 0:
            source = clipped
            print(f"    (overlay limited to the first {limit_seconds:g}s)")

    # overlay_detections() expects video.mp4 + detections.jsonl side by side
    # and writes video_annotated.mp4 next to them.
    link = stage / "video.mp4"
    try:
        os.link(source, link)
    except OSError:
        shutil.copy2(source, link)
    shutil.copy2(detections, stage / "detections.jsonl")

    ok = overlay_detections(str(stage))
    produced = stage / "video_annotated.mp4"
    if not ok or not produced.exists():
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), str(out_path))
    shutil.rmtree(stage, ignore_errors=True)
    return True


# --------------------------------------------------------------------------
# step 4 - subtitles
# --------------------------------------------------------------------------

def parse_event_time(value) -> float | None:
    """session.jsonl carries both '20260723-092755' and ISO-8601 stamps."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    for fmt in ("%Y%m%d-%H%M%S", "%Y%m%d-%H%M%S.%f"):
        try:
            return dt.datetime.strptime(value, fmt).timestamp()
        except ValueError:
            pass
    try:
        return dt.datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def make_time_mapper(anchors: list[dict], fps: float):
    """Map a wall-clock epoch onto a position in the clean plate."""
    if not anchors:
        return None
    ordered = sorted(anchors, key=lambda a: a["start_epoch"])

    def to_plate(epoch: float) -> float | None:
        for a in ordered:
            if a["start_epoch"] <= epoch < a["end_epoch"]:
                return a["plate_start"] + (epoch - a["start_epoch"])
        # Outside every chunk: clamp into a gap or reject if far outside.
        first, last = ordered[0], ordered[-1]
        if epoch < first["start_epoch"]:
            return 0.0 if epoch > first["start_epoch"] - 60 else None
        if epoch >= last["end_epoch"]:
            return last["plate_end"] if epoch < last["end_epoch"] + 60 else None
        for a in ordered:
            if epoch < a["start_epoch"]:
                return a["plate_start"]
        return None

    return to_plate


def collect_yolo_spans(detections_path: Path, fps: float, interval: float,
                       min_conf: float, max_classes: int = 6) -> list[dict]:
    """Summarise the YOLO detections into one line per interval.

    Counts are the median across the frames in the window, which drops the
    single-frame flickers that the raw per-frame counts are full of.
    """
    if not detections_path.exists():
        return []

    buckets: dict[int, list[dict]] = {}
    with detections_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get("frame_index", 0) / fps
            counts: dict[str, int] = {}
            for det in rec.get("detections", []):
                if det.get("confidence", 0.0) < min_conf:
                    continue
                name = det.get("class", "object")
                counts[name] = counts.get(name, 0) + 1
            buckets.setdefault(int(t // interval), []).append(counts)

    spans = []
    for slot in sorted(buckets):
        frames = buckets[slot]
        names = {n for c in frames for n in c}
        summary = []
        for name in names:
            series = sorted(c.get(name, 0) for c in frames)
            median = series[len(series) // 2]
            if median >= 1:
                summary.append((median, name))
        if not summary:
            continue
        summary.sort(key=lambda x: (-x[0], x[1]))
        parts = [f"{n} {name}" if n > 1 else name
                 for n, name in summary[:max_classes]]
        spans.append({
            "start": slot * interval,
            "end": (slot + 1) * interval,
            "text": "[YOLO] " + ", ".join(parts),
        })
    return spans


def collect_body_spans(chunks: list[dict], to_plate, interval: float,
                       plate_end: float) -> list[dict]:
    """One telemetry readout per interval, sampled from lowstate.jsonl.

    LowState_ carries no velocity on this hardware, but the capture embeds
    the sport_state payload alongside it, which does.
    """
    ts_re = re.compile(rb'"timestamp":\s*([0-9.]+)')
    spans, seen = [], set()

    for c in chunks:
        path = c["dir"] / "lowstate.jsonl"
        if not path.exists() or path.stat().st_size == 0:
            continue
        with path.open("rb") as f:
            for raw in f:
                m = ts_re.search(raw, 0, 80)
                if not m:
                    continue
                plate_t = to_plate(float(m.group(1)))
                if plate_t is None:
                    continue
                slot = int(plate_t // interval)
                if slot in seen:
                    continue          # one sample per window is enough
                try:
                    data = json.loads(raw).get("data", {})
                except json.JSONDecodeError:
                    continue
                seen.add(slot)

                sport = data.get("sport_state") or {}
                vel = sport.get("velocity") or [0.0, 0.0, 0.0]
                speed = (vel[0] ** 2 + vel[1] ** 2) ** 0.5
                height = sport.get("body_height")
                yaw_speed = sport.get("yaw_speed", 0.0)
                rpy = (data.get("imu_state") or {}).get("rpy") or [0, 0, 0]
                bms = data.get("bms_state") or {}
                soc = bms.get("soc")
                watts = abs(bms.get("current", 0)) / 1000.0 * data.get("power_v", 0.0)

                if height is None:
                    posture = "?"
                elif height > 0.25:
                    posture = "standing"
                elif height > 0.15:
                    posture = "crouched"
                else:
                    posture = "sitting"

                bits = [f"[BODY] {posture}"]
                if height is not None:
                    bits.append(f"h={height:.2f}m")
                bits.append(f"v={speed:.2f}m/s")
                bits.append(f"w={yaw_speed:+.2f}rad/s")
                bits.append(f"yaw={rpy[2] * 57.2958:+.0f}deg")
                if soc is not None:
                    bits.append(f"bat={soc}%")
                if watts:
                    bits.append(f"{watts:.0f}W")

                spans.append({
                    "start": slot * interval,
                    "end": min((slot + 1) * interval, plate_end),
                    "text": " ".join(bits),
                })

    spans.sort(key=lambda s: s["start"])
    return spans


def merge_tracks(sources: list[str], spans_by_source: dict[str, list[dict]],
                 ) -> list[dict]:
    """Interleave several sources into one subtitle track.

    Every span boundary becomes a breakpoint, so each emitted cue shows all
    of the lines that are active over that whole interval, stacked in the
    order the sources were requested.
    """
    import bisect

    active = [(s, spans_by_source.get(s, [])) for s in sources]
    active = [(s, sp) for s, sp in active if sp]
    if not active:
        return []

    points = sorted({p for _, spans in active for sp in spans
                     for p in (sp["start"], sp["end"])})
    starts = {s: [sp["start"] for sp in spans] for s, spans in active}

    cues: list[dict] = []
    for a, b in zip(points, points[1:]):
        # Boundaries from different sources land close together (a VLM cue
        # starting mid-way through a YOLO window); rather than flash a sliver,
        # hand the time to the cue that is already on screen.
        if b - a < 0.25:
            if cues:
                cues[-1]["end"] = b
            continue
        mid = (a + b) / 2
        lines = []
        for source, spans in active:
            i = bisect.bisect_right(starts[source], mid) - 1
            if i >= 0 and spans[i]["end"] > mid:
                lines.append(spans[i]["text"])
        if not lines:
            continue
        text = "\n".join(lines)
        if cues and cues[-1]["text"] == text and abs(cues[-1]["end"] - a) < 0.01:
            cues[-1]["end"] = b       # same content, just extend
        else:
            cues.append({"start": a, "end": b, "text": text})
    return cues


def wrap_text(text: str, width: int = 42) -> str:
    """Wrap to `width`, keeping each source's line on its own row."""
    out = []
    for source_line in text.split("\n"):
        lines, cur = [], ""
        for w in source_line.split():
            if cur and len(cur) + 1 + len(w) > width:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        if cur:
            lines.append(cur)
        out.extend(lines or [""])
    return "\n".join(out)


def write_srt(cues: list[dict], out_path: Path) -> int:
    """cues: [{start, end, text}] in plate seconds."""
    if not cues:
        return 0
    cues.sort(key=lambda c: c["start"])
    with out_path.open("w", encoding="utf-8") as f:
        for i, cue in enumerate(cues, 1):
            f.write(f"{i}\n{srt_time(cue['start'])} --> {srt_time(cue['end'])}\n")
            f.write(wrap_text(cue["text"]) + "\n\n")
    return len(cues)


def decode_pcm(path: Path, rate: int, channels: int):
    """Decode any wav to raw int16 samples at the given rate/layout."""
    import numpy as np
    res = run(["ffmpeg", "-v", "error", "-i", path, "-ar", str(rate),
               "-ac", str(channels), "-f", "s16le", "-"], capture_stdout_bytes=True)
    if res is None:
        return np.empty(0, dtype=np.int16)
    return np.frombuffer(res, dtype=np.int16)


def voiced_rms(pcm) -> float:
    """Average level of the loud parts of a clip.

    Plain RMS would be dragged down by the leading/trailing silence that
    every turn carries, making quiet clips look even quieter than they are.
    """
    import numpy as np

    x = pcm.astype(np.float32)
    hop = max(1, len(x) // 200)
    n = (len(x) // hop) * hop
    if n == 0:
        return float(np.sqrt((x ** 2).mean())) if x.size else 0.0
    energy = np.sqrt((x[:n].reshape(-1, hop) ** 2).mean(axis=1))
    peak = energy.max()
    if peak <= 0:
        return 0.0
    voiced = energy[energy >= peak * 0.1]
    return float(voiced.mean()) if voiced.size else float(energy.mean())


def normalize_gain(pcm, target_db: float, max_gain_db: float = 30.0,
                   ceiling: float = 0.94) -> float:
    """Gain bringing a clip to the target level without clipping it.

    The user's lapel mic lands around -38 dBFS while Piper's output is near
    -16 dBFS; levelling each clip separately puts them on equal footing.
    """
    import numpy as np

    if pcm.size == 0:
        return 1.0
    level = voiced_rms(pcm)
    peak = float(np.abs(pcm.astype(np.float32)).max())
    if level <= 0 or peak <= 0:
        return 1.0
    gain = (32768.0 * (10.0 ** (target_db / 20.0))) / level
    gain = min(gain, 10.0 ** (max_gain_db / 20.0))
    return min(gain, (32767.0 * ceiling) / peak)   # never clip


def compose_audio_track(segments: list[dict], out_path: Path,
                        total_seconds: float, rate: int, channels: int,
                        work_dir: Path, target_db: float | None = None) -> int:
    """Lay each clip onto a silent timeline at its own start time.

    Placing every clip at an absolute offset (rather than concatenating)
    keeps the track locked to the video: a chunk whose audio runs a few
    hundred ms short can't push everything after it out of sync.
    """
    import numpy as np

    total = int(round(total_seconds * rate)) * channels
    raw = work_dir / (out_path.stem + ".raw")
    raw.parent.mkdir(parents=True, exist_ok=True)
    buf = np.memmap(raw, dtype=np.int16, mode="w+", shape=(total,))

    placed = 0
    try:
        for seg in segments:
            pcm = decode_pcm(seg["path"], rate, channels)
            if pcm.size == 0:
                continue
            if target_db is not None:
                gain = normalize_gain(pcm, target_db)
                if abs(gain - 1.0) > 0.01:
                    pcm = np.clip(pcm.astype(np.float32) * gain,
                                  -32768, 32767).astype(np.int16)
                    seg["gain_db"] = 20.0 * np.log10(gain)
            off = int(round(seg["start"] * rate)) * channels
            if off < 0:                     # started before the video did
                pcm = pcm[-off:]
                off = 0
            end = min(off + pcm.size, total)
            if end <= off:
                continue
            n = end - off
            # Sum rather than overwrite, so an overlap mixes instead of
            # truncating, and saturate instead of wrapping.
            mixed = buf[off:end].astype(np.int32) + pcm[:n].astype(np.int32)
            buf[off:end] = np.clip(mixed, -32768, 32767).astype(np.int16)
            placed += 1

        buf.flush()
        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(2)
            w.setframerate(rate)
            block = rate * channels * 30
            for i in range(0, total, block):
                w.writeframes(buf[i:i + block].tobytes())
    finally:
        del buf
        raw.unlink(missing_ok=True)
    return placed


def collect_mic_segments(chunks: list[dict], fps: float) -> list[dict]:
    """Each chunk's onboard mic recording, at that chunk's place in the plate."""
    segs = []
    for c in chunks:
        wav = c["dir"] / "audio.wav"
        if wav.exists() and wav.stat().st_size > 44:
            segs.append({"start": c["plate_offset"] / fps, "path": wav})
    return segs


def corrected_wav(wav: Path, speech_dir: Path, from_rate: int, to_rate: int,
                  work_dir: Path):
    """Resolve a wav to its rate-corrected copy. Returns (path, rate, frames)."""
    try:
        with wave.open(str(wav), "rb") as w:
            frames, rate = w.getnframes(), w.getframerate()
    except (wave.Error, EOFError):
        return None, None, None
    if rate != from_rate:            # already honest about its rate
        return wav, rate, frames
    fixed = speech_dir / wav.name
    if not fixed.exists():
        fixed = work_dir / wav.name
        if not fixed.exists():
            relabel_wav(wav, fixed, to_rate)
    return fixed, to_rate, frames


def dialogue_clip(session_dir: Path, speech_dir: Path, ev: dict,
                  from_rate: int, to_rate: int, work_dir: Path) -> dict | None:
    """Locate a turn's wav and work out when it was actually heard.

    Timing rules measured against this corpus:
      - a response finishes exactly as its `assistant` event is logged, so it
        starts one (corrected 48 kHz) duration earlier (n=64, stdev 0.13s);
      - an input wav is closed when recording stops, ~1.2s before its event is
        logged after transcription, so mtime marks its end. `reset` events
        carry the user's utterance in audio_path, so they follow this rule.
    Returned as wall-clock epochs for the caller to map.
    """
    etype = ev.get("type")
    if etype not in ("assistant", "user", "reset") or not ev.get("audio_path"):
        return None

    # audio_path is the robot's own path; only the name is meaningful here
    wav = session_dir / os.path.basename(ev["audio_path"])
    if not wav.exists():
        return None

    wav, rate, frames = corrected_wav(wav, speech_dir, from_rate, to_rate,
                                      work_dir)
    if not rate:
        return None

    if etype == "assistant":
        end_epoch = parse_event_time(ev.get("timestamp"))
    else:
        # user speech, including the utterance that triggered a reset
        end_epoch = wav.stat().st_mtime
    if end_epoch is None:
        return None

    duration = frames / rate
    return {"path": wav, "duration": duration,
            "start_epoch": end_epoch - duration}


def cue_clips(session_dir: Path, speech_dir: Path, from_rate: int,
              to_rate: int, work_dir: Path) -> list[dict]:
    """The robot's short cue sounds: wake acknowledgements and resets.

    Neither is an `assistant` turn, so neither ends at an event. A reset wav
    is written within ~0.4s of its `reset` event, and a wake wav has no event
    at all, so both are anchored to when they were produced and play forward.
    """
    clips = []

    for ev in iter_session_events(session_dir):
        if ev.get("type") != "reset":
            continue
        turn = ev.get("turn")
        if turn is None:
            continue
        wav = session_dir / f"turn-{int(turn):03d}-reset.wav"
        if not wav.exists():
            continue
        when = parse_event_time(ev.get("timestamp"))
        if when is None:
            continue
        fixed, rate, frames = corrected_wav(wav, speech_dir, from_rate,
                                            to_rate, work_dir)
        if rate:
            clips.append({"path": fixed, "start_epoch": when,
                          "duration": frames / rate})

    for wav in sorted(session_dir.glob("*-wake.wav")):
        fixed, rate, frames = corrected_wav(wav, speech_dir, from_rate,
                                            to_rate, work_dir)
        if rate:
            clips.append({"path": fixed, "start_epoch": wav.stat().st_mtime,
                          "duration": frames / rate})

    return clips


def iter_session_events(session_dir: Path):
    log = session_dir / "session.jsonl"
    if not log.exists():
        return
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def collect_dialogue_segments(session_dir: Path, speech_dir: Path, to_plate,
                              from_rate: int, to_rate: int,
                              work_dir: Path) -> list[dict]:
    """Every spoken clip - startup, input, response, wake, reset - in place."""
    clips = [c for c in (dialogue_clip(session_dir, speech_dir, ev, from_rate,
                                       to_rate, work_dir)
                         for ev in iter_session_events(session_dir))
             if c is not None]
    clips += cue_clips(session_dir, speech_dir, from_rate, to_rate, work_dir)

    segs, seen = [], set()
    for clip in clips:
        if clip["path"] in seen:      # a wav referenced by two events
            continue
        start = to_plate(clip["start_epoch"])
        if start is None:
            continue
        seen.add(clip["path"])
        segs.append({"start": start, "path": clip["path"]})
    segs.sort(key=lambda s: s["start"])
    return segs


def build_full_dialogue(session_dir: Path, out_dir: Path, from_rate: int,
                        to_rate: int, args, report: dict) -> None:
    """Whole-conversation dialogue WAV + SRT on the session's own clock.

    The video-synced dialogue track only spans the retained plate, but a
    long circular-buffer session prunes most of its video while session.jsonl
    keeps every turn. This lays all turns onto a timeline anchored at
    session start, so the full conversation is listenable even where the
    matching video is long gone. Needs no video, detections, or anchors.
    """
    events = list(iter_session_events(session_dir))
    epochs = [t for t in (parse_event_time(e.get("timestamp")) for e in events)
              if t is not None]
    if len(epochs) < 2:
        print("    full dialogue skipped: session.jsonl has no usable clock")
        return

    t0 = min(epochs)

    def to_session(epoch):
        return max(0.0, epoch - t0)     # session start is time zero

    speech_dir = out_dir / "speech"
    work_dir = out_dir / ".work"

    # --- audio: every spoken clip, deduped ---
    clips = [c for c in (dialogue_clip(session_dir, speech_dir, ev, from_rate,
                                       to_rate, work_dir) for ev in events)
             if c is not None]
    clips += cue_clips(session_dir, speech_dir, from_rate, to_rate, work_dir)

    segs, seen, end_max = [], set(), 0.0
    for c in clips:
        if c["path"] in seen:
            continue
        seen.add(c["path"])
        start = to_session(c["start_epoch"])
        segs.append({"start": start, "path": c["path"]})
        end_max = max(end_max, start + c["duration"])
    segs.sort(key=lambda s: s["start"])

    dlg = out_dir / "audio_dialogue_full.wav"
    n_clips = compose_audio_track(
        segs, dlg, end_max + 1.0, args.audio_rate, 1, work_dir,
        target_db=None if args.no_normalize else args.dialogue_db)

    # --- SRT: every text turn, capped so cues don't overlap the next ---
    text_cues = []
    for ev in events:
        etype = ev.get("type")
        if etype not in ("user", "assistant", "reset"):
            continue
        text = (ev.get("text") or "").strip()
        if not text:
            continue
        clip = dialogue_clip(session_dir, speech_dir, ev, from_rate, to_rate,
                             work_dir)
        if clip is not None:
            start, dur = to_session(clip["start_epoch"]), clip["duration"]
        else:
            when = parse_event_time(ev.get("timestamp"))
            if when is None:
                continue
            start, dur = to_session(when), args.max_cue
        speaker = ev.get("speaker") or ("USER" if etype == "reset"
                                        else etype.upper())
        text_cues.append({"start": start, "dur": dur,
                          "text": f"[{speaker}] {text}"})

    text_cues.sort(key=lambda c: c["start"])
    cues = []
    for i, c in enumerate(text_cues):
        end = c["start"] + min(c["dur"], args.max_cue)
        if i + 1 < len(text_cues):
            end = min(end, text_cues[i + 1]["start"])
        end = max(end, c["start"] + args.min_cue)
        cues.append({"start": c["start"], "end": end, "text": c["text"]})
    n_cues = write_srt(cues, out_dir / "dialogue_full.srt")

    print(f"    audio_dialogue_full.wav: {n_clips} clips over "
          f"{(end_max + 1) / 60:.1f} min | dialogue_full.srt: {n_cues} cues")
    report["full_dialogue"] = {"clips": n_clips, "cues": n_cues,
                               "span_min": round((end_max + 1) / 60, 1)}


def mux_audio_tracks(video: Path, mic: Path, dialogue: Path) -> bool:
    """Attach both tracks to a video in place, leaving the video untouched."""
    tmp = video.with_name(video.stem + ".muxing.mp4")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", video]
    maps = ["-map", "0:v:0"]
    idx = 1
    titles = []
    for path, title in ((mic, "Onboard mic"), (dialogue, "Dialogue (TTS + user)")):
        if path and path.exists():
            cmd += ["-i", path]
            maps += ["-map", f"{idx}:a:0"]
            titles += [f"-metadata:s:a:{idx - 1}", f"title={title}"]
            idx += 1
    if idx == 1:
        return False
    cmd += maps + ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"] + titles
    cmd += ["-movflags", "+faststart", tmp]

    res = run(cmd)
    if res.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        print(f"    mux failed for {video.name}: "
              f"{(res.stderr or '').strip().splitlines()[-1:]}")
        return False
    os.replace(tmp, video)
    return True


SRT_SOURCES = ("speech", "vlm", "yolo", "bodystate")


def build_subtitles(session_dir: Path, out_dir: Path, anchors: list[dict],
                    fps: float, chunks: list[dict], detections_path: Path,
                    tracks: list[list[str]], args, report: dict) -> None:
    """Build each requested subtitle track out of the four event sources."""
    to_plate = make_time_mapper(anchors, fps)
    if to_plate is None:
        print("    no timing anchors (detections carry no timestamps); "
              "skipping subtitles")
        return

    plate_end = max(a["plate_end"] for a in anchors)
    wanted = {s for track in tracks for s in track}
    spans: dict[str, list[dict]] = {s: [] for s in SRT_SOURCES}

    # --- speech + VLM come out of session.jsonl -------------------------
    session_log = session_dir / "session.jsonl"
    if session_log.exists() and ({"speech", "vlm"} & wanted):
        vlm_events, speech_events = [], []
        for line in session_log.read_text(encoding="utf-8",
                                          errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            when = parse_event_time(ev.get("timestamp"))
            if when is None:
                continue
            etype = ev.get("type")
            if etype == "vlm_query":
                text = (ev.get("description") or "").strip()
                if text:
                    vlm_events.append({"epoch": when, "text": f"[VLM] {text}"})
            elif etype in ("user", "assistant", "reset"):
                text = (ev.get("text") or "").strip()
                if text:
                    # a reset's text is what the user said to trigger it
                    speaker = ev.get("speaker") or (
                        "USER" if etype == "reset" else etype.upper())
                    # The event is logged when the clip *finishes*; start the
                    # caption where the audio starts so the two agree.
                    clip = dialogue_clip(session_dir, out_dir / "speech", ev,
                                         args.speech_from_rate,
                                         args.speech_rate, out_dir / ".work")
                    speech_events.append({
                        "epoch": clip["start_epoch"] if clip else when,
                        "text": f"[{speaker}] {text}",
                        "duration": clip["duration"] if clip else None,
                    })

        def to_spans(events: list[dict]) -> list[dict]:
            events.sort(key=lambda e: e["epoch"])
            out = []
            for i, ev in enumerate(events):
                start = to_plate(ev["epoch"])
                if start is None:
                    continue
                # Hold a caption for as long as its clip actually plays.
                end = start + min(ev.get("duration") or args.max_cue,
                                  args.max_cue)
                if i + 1 < len(events):
                    nxt = to_plate(events[i + 1]["epoch"])
                    if nxt is not None:
                        end = min(end, nxt)
                end = min(max(end, start + args.min_cue), plate_end)
                if end > start:
                    out.append({"start": start, "end": end, "text": ev["text"]})
            return out

        spans["vlm"] = to_spans(vlm_events)
        spans["speech"] = to_spans(speech_events)
    elif not session_log.exists():
        print("    no session.jsonl; speech and VLM tracks will be empty")

    # --- YOLO reads the plate-mapped detections -------------------------
    if "yolo" in wanted:
        spans["yolo"] = collect_yolo_spans(detections_path, fps,
                                           args.yolo_interval,
                                           args.yolo_min_conf)

    # --- body telemetry -------------------------------------------------
    if "bodystate" in wanted:
        spans["bodystate"] = collect_body_spans(chunks, to_plate,
                                                args.body_interval, plate_end)

    counts = {s: len(spans[s]) for s in SRT_SOURCES if spans[s]}
    print(f"    sources: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    written = {}
    for track in tracks:
        name = "+".join(track) + ".srt"
        cues = (spans.get(track[0], []) if len(track) == 1
                else merge_tracks(track, spans))
        n = write_srt([dict(c) for c in cues], out_dir / name)
        written[name] = n
        print(f"    {name}: {n} cues")

    report["subtitles"] = written


# --------------------------------------------------------------------------
# step 5 - relabel synthesised speech
# --------------------------------------------------------------------------

def relabel_wav(src: Path, dst: Path, new_rate: int) -> dict:
    """Copy a wav and patch its fmt chunk to the true sample rate.

    Sample data is untouched; only the declared rate (and the byte rate
    derived from it) change, which is what Audacity's "change rate" does.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    size = dst.stat().st_size
    with dst.open("r+b") as f:
        if f.read(4) != b"RIFF":
            return {"ok": False, "error": "not a RIFF file"}
        f.seek(8)
        if f.read(4) != b"WAVE":
            return {"ok": False, "error": "not a WAVE file"}
        off = 12
        while off + 8 <= size:
            f.seek(off)
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            cid = hdr[:4]
            clen = struct.unpack("<I", hdr[4:8])[0]
            if cid == b"fmt ":
                body = f.read(min(clen, 16))
                if len(body) < 16:
                    return {"ok": False, "error": "short fmt chunk"}
                channels = struct.unpack("<H", body[2:4])[0]
                bits = struct.unpack("<H", body[14:16])[0]
                block_align = channels * bits // 8
                f.seek(off + 8 + 4)
                f.write(struct.pack("<I", new_rate))            # sample rate
                f.write(struct.pack("<I", new_rate * block_align))  # byte rate
                return {"ok": True, "channels": channels, "bits": bits}
            off += 8 + clen + (clen & 1)
    return {"ok": False, "error": "no fmt chunk"}


def relabel_speech(session_dir: Path, out_dir: Path, from_rate: int,
                   to_rate: int, report: dict) -> None:
    """Gather every turn's audio into speech/, correcting the ones that lie.

    Files already carrying the right rate (the 16 kHz mic recordings) are
    copied verbatim rather than skipped, so the folder is a complete set of
    the session's audio and not just the subset that needed fixing.
    """
    speech_out = out_dir / "speech"
    fixed, copied = 0, 0

    for wav_path in sorted(session_dir.glob("*.wav")):
        try:
            with wave.open(str(wav_path), "rb") as w:
                rate = w.getframerate()
                frames = w.getnframes()
        except (wave.Error, EOFError):
            continue

        # Only the synthesised voice is mislabelled; recorded mic input is
        # genuinely 16 kHz and its header is already correct.
        if rate != from_rate:
            speech_out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wav_path, speech_out / wav_path.name)
            copied += 1
            continue

        res = relabel_wav(wav_path, speech_out / wav_path.name, to_rate)
        if res["ok"]:
            fixed += 1
            report["speech"].append({
                "file": wav_path.name,
                "from_rate": rate, "to_rate": to_rate,
                "old_duration": round(frames / rate, 2),
                "new_duration": round(frames / to_rate, 2),
            })

    print(f"    speech/: {fixed} relabelled {from_rate} -> {to_rate} Hz, "
          f"{copied} copied unchanged ({fixed + copied} total)")


# --------------------------------------------------------------------------
# per-session driver
# --------------------------------------------------------------------------

def write_report(path: Path, report: dict) -> None:
    """Fold this run into any existing report so that running one step at a
    time doesn't erase what the earlier steps recorded."""
    prior = {}
    if path.exists():
        try:
            prior = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            prior = {}

    merged = dict(prior)
    for key, value in report.items():
        if key == "outputs" and isinstance(prior.get(key), dict):
            merged[key] = {**prior[key], **value}
        elif isinstance(value, list) and not value and isinstance(prior.get(key), list):
            merged[key] = prior[key]
        else:
            merged[key] = value
    path.write_text(json.dumps(merged, indent=2))


def process_session(session_dir: Path, out_root: Path, steps: set[str],
                    args) -> dict:
    print(f"\n=== {session_dir.name} ===")
    out_dir = out_root / session_dir.name
    work_dir = out_dir / ".work"
    report = {
        "session": session_dir.name,
        "source": str(session_dir),
        "output": str(out_dir),
        "processed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "repairs": [], "speech": [], "outputs": {}, "warnings": [],
    }

    chunks = discover_chunks(session_dir)
    if not chunks:
        # No video at all (e.g. an audio-only or fully-pruned session). The
        # full-dialogue step still works from session.jsonl alone.
        if "fulldialogue" in steps and not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            work_dir.mkdir(parents=True, exist_ok=True)
            print("  no chunk_* folders; building full-session dialogue only")
            build_full_dialogue(session_dir, out_dir, args.speech_from_rate,
                                args.speech_rate, args, report)
            if not args.keep_work:
                shutil.rmtree(work_dir, ignore_errors=True)
            write_report(out_dir / "report.json", report)
            return report
        print("  no chunk_* folders; skipping")
        report["warnings"].append("no chunks found")
        return report

    if args.dry_run:
        for c in chunks:
            info = probe_video(c["video"])
            state = "ok" if info["ok"] else f"DAMAGED ({info['error']})"
            print(f"  {c['name']}: {state}")
        return report

    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    usable, fps = prepare_chunks(chunks, out_dir, "repair" in steps, report,
                                 args.logs_dir.expanduser(), session_dir)
    if not usable:
        print("  no usable chunk videos; skipping the video steps")
        report["warnings"].append("no usable chunk videos")
    else:
        total = sum(c["frames"] for c in usable)
        print(f"  plate: {len(usable)} chunks, {total} frames "
              f"({human_time(total / fps)})")

    # truncated audio.wav headers are worth knowing about, but the loss is
    # only the few bytes written after the last header flush.
    for c in chunks:
        if c["audio"].exists():
            wav = check_wav(c["audio"])
            if wav.get("ok") and wav.get("trailing_bytes", 0) > 0:
                report["warnings"].append(
                    f"{c['name']}/audio.wav has {wav['trailing_bytes']} bytes "
                    f"past its declared data size (unflushed tail)")

    plate = out_dir / "video_clean.mp4"
    if usable and "plate" in steps:
        print("  building clean plate (stream copy)...")
        if build_plate(usable, plate, work_dir):
            info = probe_video(plate)
            report["outputs"]["clean_plate"] = {
                "path": str(plate), "frames": info["frames"],
                "duration": info["duration"],
            }
            print(f"    {plate.name}: {info['frames']} frames, "
                  f"{human_time(info['duration'] or 0)}")
        else:
            report["warnings"].append("clean plate concat failed")

    merged = out_dir / "detections_combined.jsonl"
    anchors: list[dict] = []
    if usable and ({"overlay", "srt", "audio"} & steps):
        print("  merging detections...")
        anchors = merge_detections(usable, fps, merged, report)["anchors"]
        report["outputs"]["detections"] = str(merged)

    if "overlay" in steps:
        if plate.exists() and merged.exists():
            frames = report["outputs"].get("clean_plate", {}).get("frames") or 0
            print(f"  rendering YOLO overlay ({frames} frames; this is the "
                  f"slow step)...")
            overlay = out_dir / "video_overlay.mp4"
            if run_overlay(plate, merged, overlay, work_dir,
                           args.overlay_seconds):
                report["outputs"]["overlay"] = str(overlay)
                print(f"    wrote {overlay.name}")
            else:
                report["warnings"].append("overlay render failed")
        else:
            print("  skipping overlay (needs the clean plate and detections)")

    if "srt" in steps:
        print("  building subtitles...")
        build_subtitles(session_dir, out_dir, anchors, fps, chunks, merged,
                        parse_tracks(args.srt_tracks), args, report)
        for name in report.get("subtitles", {}):
            report["outputs"][name] = str(out_dir / name)

    if "speech" in steps:
        print("  relabelling synthesised speech...")
        relabel_speech(session_dir, out_dir, args.speech_from_rate,
                       args.speech_rate, report)

    if "audio" in steps and usable:
        print("  building audio tracks...")
        to_plate = make_time_mapper(anchors, fps)
        plate_seconds = report["plate_frames"] / fps

        mic_path = out_dir / "audio_mic.wav"
        mic_segs = collect_mic_segments(usable, fps)
        n_mic = compose_audio_track(mic_segs, mic_path, plate_seconds,
                                    args.audio_rate, 2, work_dir) if mic_segs else 0
        print(f"    track 1 (onboard mic): {n_mic} chunk recording(s)")

        dlg_path = out_dir / "audio_dialogue.wav"
        n_dlg = 0
        if to_plate is None:
            print("    track 2 skipped: no timing anchors to place clips with")
        else:
            dlg_segs = collect_dialogue_segments(
                session_dir, out_dir / "speech", to_plate,
                args.speech_from_rate, args.speech_rate, work_dir)
            if dlg_segs:
                n_dlg = compose_audio_track(
                    dlg_segs, dlg_path, plate_seconds, args.audio_rate, 1,
                    work_dir,
                    target_db=None if args.no_normalize else args.dialogue_db)
            gains = [s["gain_db"] for s in dlg_segs if "gain_db" in s]
            note = ""
            if gains:
                note = (f", levelled to {args.dialogue_db:g} dBFS "
                        f"(gain {min(gains):+.0f}..{max(gains):+.0f} dB)")
            print(f"    track 2 (dialogue): {n_dlg} clip(s) placed{note}")

        mic_arg = mic_path if n_mic else None
        dlg_arg = dlg_path if n_dlg else None
        for name in ("video_clean.mp4", "video_overlay.mp4"):
            video = out_dir / name
            if video.exists() and (mic_arg or dlg_arg):
                if mux_audio_tracks(video, mic_arg, dlg_arg):
                    print(f"    muxed {'+'.join(t for t, k in (('mic', n_mic), ('dialogue', n_dlg)) if k)} into {name}")
        report["audio"] = {"mic_segments": n_mic, "dialogue_clips": n_dlg}

    # Whole-conversation dialogue, independent of video — the only step that
    # runs without any usable chunks, since it reads session.jsonl alone.
    if "fulldialogue" in steps:
        out_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        print("  building full-session dialogue (session clock, no video)...")
        build_full_dialogue(session_dir, out_dir, args.speech_from_rate,
                            args.speech_rate, args, report)

    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)

    write_report(out_dir / "report.json", report)
    for w in report["warnings"]:
        print(f"  [warning] {w}")
    return report


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def parse_tracks(spec: str) -> list[list[str]]:
    """'speech,yolo+vlm' -> [['speech'], ['yolo', 'vlm']]"""
    tracks = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        sources = [s.strip() for s in part.split("+") if s.strip()]
        unknown = [s for s in sources if s not in SRT_SOURCES]
        if unknown:
            raise ValueError(
                f"unknown subtitle source(s): {', '.join(unknown)} "
                f"(available: {', '.join(SRT_SOURCES)})")
        if sources:
            tracks.append(sources)
    return tracks


def select_sessions(logs_dir: Path, args) -> list[Path]:
    if args.session:
        picked = []
        for name in args.session:
            d = Path(name)
            if not d.is_absolute():
                d = logs_dir / name
            if d.is_dir():
                picked.append(d)
            else:
                print(f"no such session: {d}", file=sys.stderr)
        return picked

    sessions = sorted(
        (d for d in logs_dir.iterdir()
         if d.is_dir() and d.name.startswith("session-")),
        key=lambda d: d.name,
    )
    if not sessions:
        return []
    return sessions if args.all else sessions[-args.last:]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Repair and post-process BFF session recordings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[-1],
    )
    p.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS,
                   help=f"session root (default: {DEFAULT_LOGS})")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                   help=f"output root (default: {DEFAULT_OUT})")
    p.add_argument("--session", action="append",
                   help="process this session by name or path (repeatable)")
    p.add_argument("--last", type=int, default=1, metavar="N",
                   help="process the N most recent sessions (default: 1)")
    p.add_argument("--all", action="store_true", help="process every session")
    p.add_argument("--steps", default=",".join(ALL_STEPS),
                   help=f"comma-separated subset of: {','.join(ALL_STEPS)}")
    p.add_argument("--audio-rate", type=int, default=48000,
                   help="sample rate for the composed audio tracks "
                        "(default: 48000)")
    p.add_argument("--dialogue-db", type=float, default=-20.0,
                   help="level each dialogue clip to this dBFS so the quiet "
                        "mic and loud TTS match (default: -20)")
    p.add_argument("--no-normalize", action="store_true",
                   help="place dialogue clips at their recorded levels")
    p.add_argument("--overlay-seconds", type=float, default=None, metavar="SEC",
                   help="only render the first SEC seconds of overlay "
                        "(quick check)")
    p.add_argument("--srt-tracks", default=DEFAULT_TRACKS,
                   help="comma-separated subtitle tracks; combine sources "
                        f"with '+' (sources: {', '.join(SRT_SOURCES)}). "
                        f"default: {DEFAULT_TRACKS}")
    p.add_argument("--min-cue", type=float, default=1.5,
                   help="minimum subtitle duration (default: 1.5s)")
    p.add_argument("--max-cue", type=float, default=12.0,
                   help="maximum subtitle duration (default: 12s)")
    p.add_argument("--yolo-interval", type=float, default=2.0,
                   help="seconds per YOLO subtitle line (default: 2)")
    p.add_argument("--yolo-min-conf", type=float, default=0.3,
                   help="ignore detections below this confidence (default: 0.3)")
    p.add_argument("--body-interval", type=float, default=2.0,
                   help="seconds per body-state readout (default: 2)")
    p.add_argument("--speech-rate", type=int, default=SPEECH_TRUE_RATE,
                   help=f"true rate of the speech wavs (default: {SPEECH_TRUE_RATE})")
    p.add_argument("--speech-from-rate", type=int, default=SPEECH_WRONG_RATE,
                   help=f"header rate to correct (default: {SPEECH_WRONG_RATE})")
    p.add_argument("--dry-run", action="store_true",
                   help="report chunk health and exit without writing anything")
    p.add_argument("--keep-work", action="store_true",
                   help="keep intermediate files")
    args = p.parse_args()

    logs_dir = args.logs_dir.expanduser()
    out_root = args.out_dir.expanduser()
    if not logs_dir.is_dir():
        print(f"logs directory not found: {logs_dir}", file=sys.stderr)
        return 1

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("ffmpeg and ffprobe are required (brew install ffmpeg)",
              file=sys.stderr)
        return 1

    steps = {s.strip() for s in args.steps.split(",") if s.strip()}
    unknown = steps - set(ALL_STEPS)
    if unknown:
        print(f"unknown step(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 1

    try:
        tracks = parse_tracks(args.srt_tracks)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    sessions = select_sessions(logs_dir, args)
    if not sessions:
        print(f"no sessions found under {logs_dir}", file=sys.stderr)
        return 1

    print(f"source: {logs_dir}")
    print(f"output: {out_root}   (originals are never modified)")
    print(f"steps:  {', '.join(s for s in ALL_STEPS if s in steps)}")

    for session in sessions:
        try:
            process_session(session, out_root, steps, args)
        except KeyboardInterrupt:
            print("\ninterrupted")
            return 130
        except Exception as e:  # keep going through a batch
            print(f"  failed: {type(e).__name__}: {e}")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
