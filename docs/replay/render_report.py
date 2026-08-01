#!/usr/bin/env python3
"""Render the dialogue+media report from bff-replay-index.json.

Usage:  python render_report.py
Regenerate the index first if the archive changed. Do not hand-edit the
markdown - it is overwritten.
"""
import json, os, glob, collections, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
BY_PHASE = os.environ.get("BFF_BY_PHASE",
                          os.path.join(os.path.dirname(
                              os.environ.get("BFF_ARCHIVE_ROOT",
                                             "/Volumes/Cohab2024/BFF/logs-all")), "by-phase"))

_by_phase_dirs = {}
if os.path.isdir(BY_PHASE):
    for p in glob.glob(os.path.join(BY_PHASE, "*", "KEY__*")):
        # KEY__<session-id>[__<group>]
        stem = os.path.basename(p)[len("KEY__"):]
        sid = stem.split("__")[0]
        _by_phase_dirs[sid] = p

def sess_link(sid, text=None):
    """Markdown link to the gathered files for a session, if the tree is built."""
    label = text or f"`{sid}`"
    p = _by_phase_dirs.get(sid)
    if not p: return label
    return f"[{label}](file://{urllib.parse.quote(p)})"
d = json.load(open(os.path.join(HERE, "bff-replay-index.json")))
cat = json.load(open(os.path.join(HERE, "bff-sessions.json")))
ex, ss = d["exchanges"], cat["sessions"]

def mb(n): return f"{n/1e9:.1f}GB" if n and n >= 1e9 else (f"{n/1e6:.0f}MB" if n and n >= 1e6 else "–")
def dur(s): return "–" if not s else (f"{s:.0f}s" if s < 60 else f"{s/60:.1f}m")
def dash(v): return str(v) if v else "–"

L = []; w = L.append
w("# BFF — Dialogue & Media Report")
w("")
w("Curated exchanges from the BFF development and performance archive, with an")
w("inventory of every recorded stream that survives for each one.")
w("")
w(f"- **Generated:** {d['generated']}")
w(f"- **Archive:** `{d['archive_root']}`")
w(f"- **Derived:** `{d['derived_root']}`")
w("- **Companion index:** [`bff-replay-index.json`](bff-replay-index.json) — per-turn timestamps,")
w("  wav paths, video frame offsets. [`bff-sessions.json`](bff-sessions.json) catalogs all")
w(f"  {sum(1 for s in ss if not s.get('is_variant'))} sessions at summary level. This report is generated from them.")
w("")
if _by_phase_dirs:
    w(f"Session ids link to that session's gathered files under `{BY_PHASE}` —")
    w("every transcript, wav, video, telemetry and subtitle track in one directory.")
    w("")
w("## Read this first")
w("")
w(d["transcript_note"])
w("")
tiers = collections.Counter(e.get("replay_tier") for e in ex)
w(f"Of {len(ex)} curated exchanges: **{tiers['A']}** fully instrumented (tier A), "
  f"**{tiers['B']}** voice-only (tier B), **{tiers['C']}** text-only (tier C).")
w("")
w("**Video, lidar and body telemetry exist only from July 2026 onward.** Everything from the")
w("CMC keynote through IDEAS is audio-only, and CMC is transcript-only — that log format never")
w("wrote audio.")
w("")
w("## Machine, persona and voice are three different things")
w("")
w("By July 2026 the **helper machine was running the SNAPPER persona prompt**, so the name in the")
w("system prompt does not tell the dogs apart. The index carries `group` (machine), `persona`")
w("(prompt) and `voice` (Piper model) as separate fields.")
w("")
w("**`group` is the only proven one** — it comes from absolute paths the logger wrote. Voice is")
w("suggestive but conditional: `aru`=snapper / `alan`=helper holds **100% (33/33)** for scripted")
w("sessions, those carrying `scene_switch` cues, and 95% across the two performance days — but")
w("only **68%** overall, because both dogs ran both voices during technical development. Trust it")
w("for rehearsal and performance material; ignore it for dev sessions.")
w("")
w("## Sessions run long and overlap")
w("")
w("A session id is a *start* timestamp. Sessions routinely run for hours, sit idle, and overlap")
w("each other. Use the `blocks` field, which gives the actually-active spans separated by silence.")
w("The clearest case is the Snapper IDEAS performance below: a 2.7-hour session whose show is an")
w("8-minute block at the end.")
w("")

PHASES = [("1-CMC", "Phase 1 — CMC WIN keynote (Nov 2025)"),
          ("2-NeurIPS", "Phase 2 — NeurIPS (Dec 2025)"),
          ("3-IDEAS", "Phase 3 — IDEAS performance (Jan 29 2026)"),
          ("4-SIGGRAPH", "Phase 4 — SIGGRAPH Spatial Storytelling (Jul 23 2026)")]
for pid, title in PHASES:
    rows = [e for e in ex if e.get("phase") == pid]
    if not rows: continue
    w(f"## {title}")
    w("")
    if pid == "1-CMC":
        w("| Exchange | Session | Persona | Turns | User | Assistant |")
        w("|---|---|---|---|---|---|")
        for e in rows:
            c = e.get("counts", {})
            w(f"| {e['label']} | {sess_link(e['session_id'])} | {e.get('persona') or '–'} | "
              f"{len(e.get('turns',[]))} | {c.get('user_turns',0)} | {c.get('assistant_turns',0)} |")
        w("")
        w("Transcript only — no audio was ever written for these, so there is nothing to")
        w("re-transcribe. Turns are reconstructed from the v1 `chat_session` format, whose")
        w("timestamps are request/response times rather than utterance times: several user")
        w("turns can share one timestamp. Good for ordering, not for tight sync.")
    else:
        w("| Exchange | Session | Machine | Voice | Tier | In | Out | Empty | **Recovered** | Bleed | Audio | Video |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for e in rows:
            c = e.get("counts", {}); m = e.get("media", {})
            vb = sum(x["video"]["bytes"] for x in (m.get("chunks") or [])
                     if isinstance(x, dict) and x.get("video")) if isinstance(m.get("chunks"), list) else 0
            voice = (e.get("voice") or "–").replace("en_GB-", "").replace("-medium", "")
            w(f"| {e['label']} | {sess_link(e['session_id'])} | {e.get('group','–')} | {voice} | {e['replay_tier']} | "
              f"{c.get('input_wavs',0)} | {c.get('response_wavs',0)} | {dash(c.get('empty_wavs'))} | "
              f"**{c.get('recovered_speech',0)}** | {dash(c.get('speaker_bleed'))} | "
              f"{dur(c.get('audio_s'))} | {mb(vb)} |")
    w("")

w("## The two IDEAS performances")
w("")
w("Both dogs performed, simultaneously.")
w("")
for sid in ["session-20260129-144803", "session-20260129-164334"]:
    e = next((x for x in ex if x["session_id"] == sid), None)
    if not e: continue
    c = e.get("counts", {})
    w(f"**{sess_link(sid)}** — machine `{e.get('group')}`, voice `{e.get('voice','?')}`, persona {e.get('persona')} · {e['label']}")
    w("")
    w(f"- Wall span `{e.get('start_iso','')[11:19]}` → `{e.get('end_iso','')[11:19]}` "
      f"({e.get('wall_duration_s',0)/3600:.2f} h)")
    for b in e.get("blocks", []):
        w(f"- Block {b['block']}: `{b['start_iso'][11:19]}` → `{b['end_iso'][11:19]}` — {b['wavs']} wavs")
    w(f"- {c.get('input_wavs',0)} mic / {c.get('response_wavs',0)} synth wavs, "
      f"{c.get('recovered_speech',0)} recovered, {c.get('speaker_bleed',0)} speaker bleed")
    w("")

pb = [e for e in ex if e.get("media", {}).get("processed_bundle")]
if pb:
    w("## Exchanges with a processed media bundle")
    w("")
    w("Session-level concatenations under `processed/` — far easier to drive a presentation from")
    w("than the raw per-chunk files.")
    w("")
    w("| Exchange | Session | Clean video | Overlay | Subtitles |")
    w("|---|---|---|---|---|")
    for e in pb:
        b = e["media"]["processed_bundle"]
        subs = ", ".join(f"`{k}`" for k in b if k.endswith(".srt")) or "–"
        w(f"| {e['label']} | {sess_link(e['session_id'])} | {mb((b.get('video_clean.mp4') or {}).get('bytes'))} | "
          f"{mb((b.get('video_overlay.mp4') or {}).get('bytes'))} | {subs} |")
    w("")

w("## Replay tiers")
w("")
w("| Tier | What you can build | Count |")
w("|---|---|---|")
for t in "ABC":
    w(f"| **{t}** | {d['replay_tiers'][t]} | {tiers[t]} |")
w("")

w("## Archive at a glance")
w("")
w("| Group | Sessions | SNAPPER | HELPER | Unattributed |")
w("|---|---|---|---|---|")
real = [s for s in ss if not s.get("is_variant")]
for g in sorted({s["group"] for s in real}):
    r = [s for s in real if s["group"] == g]
    pc = collections.Counter(s.get("persona") for s in r)
    w(f"| `{g}` | {len(r)} | {pc.get('SNAPPER',0)} | {pc.get('HELPER',0)} | {pc.get(None,0)} |")
w(f"| **total** | **{len(real)}** | | | |")
w("")

w("## Syncing notes")
w("")
for k, v in d["sync"].items():
    w(f"- **{k.replace('_',' ')}** — {v}")
w("- `lidar.jsonl` is the storage hog — up to 187MB per chunk. Decimate before shipping to a browser.")
w("")
w("## Caveats")
w("")
w("1. **`_conflicts` holds partial duplicate copies**, flagged `is_variant` and excluded from counts.")
w("   Their canonical copy lives in a machine folder.")
w("2. **Clock.** `session-19691231-160201` (Ace Hotel) ran with the Jetson RTC unset. Internally")
w("   consistent, but its epoch will not align with anything else from that day.")
w("3. **Highlight ranges are keyword matches, not verified** — see `confidence` on each. Listen before cutting.")
w("4. **`logged_text` is live `tiny.en`** and is often wrong or missing entirely; `text` is the")
w("   re-transcription. Roughly 1–2% of assistant records point at the wrong wav.")
w("")

open(os.path.join(HERE, "BFF-dialogue-media-report.md"), "w").write("\n".join(L) + "\n")
print("wrote BFF-dialogue-media-report.md", len("\n".join(L)), "chars")
