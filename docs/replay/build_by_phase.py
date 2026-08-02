#!/usr/bin/env python3
"""Gather everything for every session into one tree, sorted by project phase.

logs-all holds the raw sessions, processed/ holds the derived work
(re-transcriptions, complete logs, concatenated video, subtitles, rate-corrected
speech). Browsing a single session means visiting both. This builds a third view
that pulls all of it together, one directory per session, grouped by phase.

Files are HARDLINKED, not copied: both trees are on the same filesystem, so this
costs no additional space and each link behaves exactly like the real file for
browsing and playback. Editing through a link would edit the original, so treat
this tree as read-only.

    python build_by_phase.py [--out DIR] [--clean]
"""
import os, re, json, glob, shutil, argparse, datetime, collections

ROOT    = os.environ.get("BFF_ARCHIVE_ROOT", "/Volumes/Cohab2024/BFF/logs-all")
DERIVED = os.environ.get("BFF_DERIVED_ROOT", "/Volumes/Cohab2024/BFF/processed")
HERE    = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(os.path.dirname(ROOT), "by-phase")

PHASES = [
    ("1-CMC-2025-11",       None,       "20251130", "CMC WIN keynote. Transcript only - that logger never wrote audio."),
    ("2-NeurIPS-2025-12",   "20251130", "20260101", "NeurIPS. Turn-segmented voice, both sides. No camera or telemetry."),
    ("3-IDEAS-2026-01",     "20260101", "20260301", "IDEAS performance. Voice at full performance scale."),
    ("4-SIGGRAPH-2026-07",  "20260301", None,       "SIGGRAPH Spatial Storytelling. Video, lidar, telemetry, VLM stills."),
]
UNKNOWN = "_unknown-clock"

def phase_of(sid):
    m = re.search(r"(\d{8})", sid)
    if not m or m.group(1).startswith("1969"): return UNKNOWN
    d = m.group(1)
    for name, lo, hi in [(p[0], p[1], p[2]) for p in PHASES]:
        if (lo is None or d >= lo) and (hi is None or d < hi): return name
    return UNKNOWN

def link(src, dst, skip_empty=True):
    """Hardlink src -> dst, falling back to copy across devices.

    Zero-byte files are skipped: the only ones in the archive are lidar.jsonl
    from sessions where the dog never stood (the L1 only spins when standing),
    and linking them makes an empty capture look like broken data when browsing.
    """
    if skip_empty:
        try:
            if os.path.getsize(src) == 0: return 0
        except OSError:
            return 0
    if os.path.exists(dst): return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try: os.link(src, dst)
    except OSError:
        try: shutil.copy2(src, dst)
        except Exception: return 0
    return 1

def find_group(base, sid):
    for g in sorted(os.listdir(base)):
        if os.path.isdir(os.path.join(base, g, sid)): return g
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--clean", action="store_true", help="remove the tree first")
    a = ap.parse_args()
    OUT = a.out
    if a.clean and os.path.isdir(OUT): shutil.rmtree(OUT)

    idx = json.load(open(os.path.join(HERE, "bff-replay-index.json")))
    curated = {e["session_id"]: e.get("label") for e in idx["exchanges"]}
    cat_path = os.path.join(HERE, "bff-sessions.json")
    stood = {}
    if os.path.exists(cat_path):
        for srec in json.load(open(cat_path)).get("sessions", []):
            p = srec.get("posture")
            if p: stood[srec["session_id"]] = p.get("stood_up")

    stats = collections.Counter(); rows = collections.defaultdict(list)

    # every session directory in the archive
    for group in sorted(os.listdir(ROOT)):
        gp = os.path.join(ROOT, group)
        if not os.path.isdir(gp) or group == "chat-sessions-v1": continue
        for sid in sorted(os.listdir(gp)):
            sdir = os.path.join(gp, sid)
            if not os.path.isdir(sdir) or not sid.startswith("session-"): continue
            ph = phase_of(sid)
            key = sid in curated
            name = ("KEY__" if key else "") + f"{sid}__{group}"
            dst = os.path.join(OUT, ph, name)
            n = collections.Counter()

            for r, _, fs in os.walk(sdir):
                rel = os.path.relpath(r, sdir)
                for f in fs:
                    if f == ".DS_Store": continue
                    s = os.path.join(r, f)
                    if f.endswith(".wav"):
                        sub = "speech"
                    elif f in ("video.mp4",) or f.endswith(".mp4"):
                        sub = os.path.join("video", rel) if rel != "." else "video"
                    elif f in ("lidar.jsonl","lowstate.jsonl","detections.jsonl","camera_path.jsonl"):
                        sub = os.path.join("telemetry", rel) if rel != "." else "telemetry"
                    elif f.startswith(("snapshot_","description_")):
                        sub = "vlm"
                    elif f == "session.jsonl" or f.endswith(".log"):
                        sub = "transcript"
                    else:
                        sub = os.path.join("other", rel) if rel != "." else "other"
                    n[sub.split(os.sep)[0]] += link(s, os.path.join(dst, sub, f))

            # derived: complete log, re-transcription, rate-corrected speech, processed media
            cg = find_group(os.path.join(DERIVED, "complete-logs"), sid)
            if cg:
                for f in ("session-complete.jsonl",):
                    p = os.path.join(DERIVED, "complete-logs", cg, sid, f)
                    if os.path.exists(p): n["transcript"] += link(p, os.path.join(dst, "transcript", f))
            rg = find_group(os.path.join(DERIVED, "retranscribed"), sid)
            if rg:
                for f in ("retranscription.json", "dialogue.md", "dialogue.srt"):
                    p = os.path.join(DERIVED, "retranscribed", rg, sid, f)
                    if os.path.exists(p): n["transcript"] += link(p, os.path.join(dst, "transcript", f))
            # Every group, not just the two that had -processed bundles originally:
            # the repair pass wrote into processed/mac/ as well, and hardcoding
            # snapper+helper silently dropped 19 rebuilt videos.
            for g2 in sorted(os.listdir(DERIVED)):
                pb = os.path.join(DERIVED, g2, sid)
                if g2 in ("retranscribed", "complete-logs") or not os.path.isdir(pb): continue
                for r, _, fs in os.walk(pb):
                    rel = os.path.relpath(r, pb)
                    for f in fs:
                        if f == ".DS_Store": continue
                        s = os.path.join(r, f)
                        if rel.startswith("repaired") and f == "video.mp4":
                            # A rebuilt chunk video. It must REPLACE the broken original
                            # already linked from the raw session, not sit beside it -
                            # otherwise the gathered tree still hands out the unplayable
                            # file and the end of the take is lost.
                            chunk = os.path.basename(rel)
                            dstv = os.path.join(dst, "video", chunk, f)
                            if os.path.exists(dstv): os.unlink(dstv)
                            n["video"] += link(s, dstv)
                            n["repaired"] += 1
                            continue
                        if rel.startswith("speech"):      sub = "speech-48k"
                        elif f.endswith(".srt"):          sub = "subtitles"
                        elif f.endswith((".mp4", ".webm")): sub = "video"
                        elif f.endswith(".wav"):          sub = "audio-mixed"
                        else:                             sub = "other"
                        n[sub] += link(s, os.path.join(dst, sub, f))

            if sum(n.values()) == 0:
                stats["empty sessions skipped"] += 1
                if os.path.isdir(dst) and not os.listdir(dst): os.rmdir(dst)
                continue
            stats["sessions"] += 1; stats["files"] += sum(n.values())
            rows[ph].append((name, sid, group, key, curated.get(sid), dict(n)))

    # chat-sessions-v1 -> phase 1
    csv1 = os.path.join(ROOT, "chat-sessions-v1")
    if os.path.isdir(csv1):
        for f in sorted(os.listdir(csv1)):
            if not f.endswith(".jsonl"): continue
            sid = f[:-6]
            key = sid in curated
            name = ("KEY__" if key else "") + sid
            dst = os.path.join(OUT, PHASES[0][0], name, "transcript", f)
            if link(os.path.join(csv1, f), dst):
                stats["sessions"] += 1; stats["files"] += 1
                rows[PHASES[0][0]].append((name, sid, "chat-sessions-v1", key,
                                           curated.get(sid), {"transcript": 1}))

    # per-phase index
    order = [p[0] for p in PHASES] + [UNKNOWN]
    blurb = {p[0]: p[3] for p in PHASES}
    blurb[UNKNOWN] = "Jetson RTC was unset; these sessions carry 1969 dates and cannot be placed."
    for ph in order:
        rs = sorted(rows.get(ph, []), key=lambda r: (not r[3], r[1]))
        if not rs: continue
        L = [f"# {ph}", "", blurb[ph], "",
             f"{len(rs)} sessions. `KEY__` marks the curated exchanges.", "",
             "| session | machine | key | speech | video | telemetry | vlm | subtitles | posture |",
             "|---|---|---|---|---|---|---|---|---|"]
        for name, sid, group, key, label, n in rs:
            L.append(f"| `{name}` | {group} | {'**'+label+'**' if label else ''} | "
                     f"{n.get('speech',0) or ''} | {n.get('video',0) or ''} | "
                     f"{n.get('telemetry',0) or ''} | {n.get('vlm',0) or ''} | "
                     f"{n.get('subtitles',0) or ''} | "
                     f"{'' if stood.get(sid) is None else ('stood' if stood[sid] else 'never stood - no lidar')} |")
        open(os.path.join(OUT, ph, "INDEX.md"), "w").write("\n".join(L) + "\n")

    open(os.path.join(OUT, "README.md"), "w").write(f"""# BFF by phase

Every file for every session, gathered from `logs-all/` (raw) and `processed/`
(derived), in one place and sorted by project phase. Built {datetime.datetime.now():%Y-%m-%d}
by `build_by_phase.py`.

**These are hardlinks, not copies.** They cost no extra space and behave exactly
like the real files, but editing one edits the original. Treat this tree as
read-only; rebuild it any time with `--clean`.

| Phase | What exists |
|---|---|
""" + "\n".join(f"| `{p[0]}` | {p[3]} |" for p in PHASES) + f"""
| `{UNKNOWN}` | {blurb[UNKNOWN]} |

Each session directory is `[KEY__]<session-id>__<machine>` and contains only the
subdirectories it actually has:

| Subdir | Contents |
|---|---|
| `transcript/` | `session.jsonl` (raw), `session-complete.jsonl` (merged + re-transcribed), `dialogue.md`, `retranscription.json`, server logs |
| `speech/` | per-turn wavs as recorded |
| `speech-48k/` | rate-corrected speech, where `fix_recordings.py` produced it |
| `video/` | per-chunk `video.mp4`, plus `video_clean.mp4` / `video_overlay.mp4` where they exist |
| `telemetry/` | `lidar.jsonl`, `lowstate.jsonl`, `detections.jsonl`, `camera_path.jsonl` |
| `vlm/` | scene snapshots and their captions |
| `subtitles/` | `.srt` tracks |
| `audio-mixed/` | concatenated dialogue / mic stems |

`KEY__` marks a session curated in `bff-replay-index.json`. Each phase has an
`INDEX.md` listing its sessions and what each holds.

Not included: the 164 `go2_capture_*` directories, which hold no dialogue - see
`../logs-all/README.md`.
""")
    print(f"wrote {OUT}")
    for k, v in stats.most_common(): print(f"  {k}: {v}")
    for ph in order:
        if rows.get(ph): print(f"    {ph:22s} {len(rows[ph]):4d} sessions")

main()
