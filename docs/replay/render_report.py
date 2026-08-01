#!/usr/bin/env python3
"""Render the dialogue+media report from bff-replay-index.json.

Usage:  python render_report.py
Regenerate the index first if the archive changed. Do not hand-edit the
markdown - it is overwritten.
"""
import json, datetime, os

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "bff-replay-index.json")))
ex = d["exchanges"]
def mb(n): return f"{n/1e9:.1f}GB" if n>=1e9 else (f"{n/1e6:.0f}MB" if n>=1e6 else "")
def mins(s):
    if not s: return "–"
    return f"{s:.0f}s" if s < 60 else f"{s/60:.1f}m"
def dash(v): return str(v) if v else "–"
L=[]
w=L.append

w("# BFF — Dialogue & Media Report")
w("")
w("Forty flagged exchanges from the BFF development and performance archive, with an")
w("inventory of every recorded stream that survives for each one.")
w("")
w(f"- **Generated:** {d['generated']}")
w(f"- **Archive root:** `{d['archive_root']}`")
w("- **Companion index:** [`bff-replay-index.json`](bff-replay-index.json) — same data, machine-readable, with per-turn timestamps, wav paths and video frame offsets. This report is generated from it; edit the index builder, not this file.")
w("")
w("Selection is by user dialogue — moments where what the human said, or what the")
w("machine's answer revealed, is the interesting part. Sessions are grouped by the")
w("four development phases.")
w("")
w("## Headline finding for replay")
w("")
w("**Video, lidar and body telemetry exist only from July 2026 onward.** Everything")
w("from the CMC keynote through the IDEAS performance is audio-only, and the CMC")
w("material is transcript-only with no audio ever written to disk. That constrains")
w("which exchanges can be replayed as multimedia and which have to be reconstructed.")
w("")
tiers = {t: sum(1 for e in ex if e.get("replay_tier")==t) for t in "ABC"}
w(f"Of {len(ex)} exchanges: **{tiers['A']}** are fully instrumented (tier A), "
  f"**{tiers['B']}** are voice-only (tier B), **{tiers['C']}** are text-only (tier C).")
w("")

w("## Stream legend")
w("")
w("| Stream | File | What it is |")
w("|---|---|---|")
for r in [
 ("Human voice","`turn-NNN-input.wav`","The VAD segment Whisper transcribed — the actual human utterance"),
 ("Synth voice","`turn-NNN-response.wav`","Piper TTS as it played from the dog's speaker"),
 ("UI cue","`turn-NNN-{reset,wake,start/stop-listening,…}.wav`","System sounds, audible in the room and part of the performance"),
 ("Startup","`startup.wav`","Boot greeting"),
 ("VLM still","`vlm_captures/snapshot_*.jpg`","The frame the scene captioner actually looked at"),
 ("VLM caption","`vlm_captures/description_*.txt`","Its text description — what got injected into the prompt"),
 ("Video","`chunk_N/video.mp4`","Onboard fisheye camera, continuous, ~30fps"),
 ("Chunk audio","`chunk_N/audio.wav`","Continuous room audio, not turn-segmented"),
 ("Lidar","`chunk_N/lidar.jsonl`","Voxel point clouds, ~5cm resolution, `odom` frame"),
 ("Lowstate","`chunk_N/lowstate.jsonl`","IMU rpy + 20 motor joints (angle, temperature)"),
 ("Detections","`chunk_N/detections.jsonl`","YOLO boxes, classes, confidence, frame-indexed"),
 ("Camera path","`camera_path.jsonl`","Virtual camera position/target for the 3D dashboard view"),
]: w(f"| {r[0]} | {r[1]} | {r[2]} |")
w("")

PHASES=[("1-CMC","Phase 1 — CMC WIN keynote (Nov 2025)",
         "The `chat_session_*.jsonl` format logged only the accumulating message array. "
         "No audio was written. **Nothing from this period is replayable as sound.**"),
        ("2-NeurIPS","Phase 2 — NeurIPS (Dec 2025)",
         "Turn-segmented audio for both voices. No camera, no lidar, no body telemetry."),
        ("3-IDEAS","Phase 3 — IDEAS performance (Jan 29 2026)",
         "Still audio-only, but at full performance scale."),
        ("4-SIGGRAPH","Phase 4 — SIGGRAPH Spatial Storytelling (Jul 23 2026)",
         "Fully instrumented: video, lidar, joint telemetry, YOLO detections and VLM stills "
         "alongside both voices, all on one wall clock.")]

for pid,title,blurb in PHASES:
    rows=[e for e in ex if e["phase"]==pid]
    w(f"## {title}")
    w("")
    w(blurb)
    w("")
    if pid=="1-CMC":
        w("| Exchange | Session | Hosts | Stored |")
        w("|---|---|---|---|")
        for e in rows:
            w(f"| {e['label']} | `{e['session_id']}` | {', '.join(h.replace('bff-logs-','') for h in e['hosts'])} | transcript only |")
    elif pid=="4-SIGGRAPH":
        w("| Exchange | Session | Tier | Human | Synth | Cue | VLM stills | Chunks | Video | Session size |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for e in rows:
            c=e.get("counts",{})
            w(f"| {e['label']} | `{e['session_id']}` | {e['replay_tier']} | {c.get('input_wavs_playable',0)} | "
              f"{c.get('response_wavs_playable',0)} | {dash(c.get('cue_wavs'))} | {dash(c.get('vlm_snapshots'))} | "
              f"{dash(c.get('chunks'))} | {mb(c.get('video_bytes',0)) or '–'} | {mb(c.get('session_bytes',0)) or '–'} |")
    else:
        w("| Exchange | Session | Human wav | Synth wav | Empty | Cue | Human audio | Synth audio | Host |")
        w("|---|---|---|---|---|---|---|---|---|")
        for e in rows:
            c=e.get("counts",{})
            hw=f"{c.get('input_wavs_playable',0)}"
            if c.get('input_wavs',0)!=c.get('input_wavs_playable',0): hw+=f" / {c['input_wavs']}"
            sw=f"{c.get('response_wavs_playable',0)}"
            if c.get('response_wavs',0)!=c.get('response_wavs_playable',0): sw+=f" / {c['response_wavs']}"
            w(f"| {e['label']} | `{e['session_id']}` | {hw} | {sw} | {c.get('empty_wavs',0) or '–'} | "
              f"{dash(c.get('cue_wavs'))} | {mins(c.get('human_audio_s',0))} | {mins(c.get('tts_audio_s',0))} | "
              f"{e.get('primary_host','').replace('bff-logs-','')} |")
    w("")
    # phase notes
    if pid=="2-NeurIPS":
        sil=[e for e in rows if e.get("counts",{}).get("response_wavs",0)==0 and e.get("counts",{}).get("input_wavs",0)>0]
        if sil:
            w("**Note.** " + " and ".join(f"`{e['session_id']}`" for e in sil) +
              " have human audio but **zero synthesized responses** — the dog was listening and not")
            w("answering. Those are recordings of a household with a silent machine in the room.")
            w("")
    if pid=="3-IDEAS":
        big=max(rows,key=lambda e:e.get("counts",{}).get("human_audio_s",0))
        c=big["counts"]
        w(f"**The Jan 29 performance is the largest audio asset in the archive** — `{big['session_id']}`,")
        w(f"{c['input_wavs']} human wav segments ({c['input_wavs_playable']} playable) totalling "
          f"{mins(c['human_audio_s'])} of human speech,")
        w("including the full post-show audience Q&A. It exists only as sound: no video, no lidar,")
        w("no body telemetry was recorded for IDEAS.")
        w("")
    if pid=="4-SIGGRAPH":
        nov=[e for e in rows if e["replay_tier"]=="A" and not e.get("counts",{}).get("vlm_snapshots")]
        if nov:
            w("**Note.** " + ", ".join(f"`{e['session_id']}`" for e in nov) +
              " has video, lidar and body telemetry but **zero VLM stills** — the scene captioner")
            w("was off for that run. Relevant if you meant to show what it was seeing.")
            w("")

w("## Replay tiers")
w("")
w("| Tier | What you can build | Exchanges |")
w("|---|---|---|")
for t in "ABC":
    ids=[f"`{e['session_id']}`" for e in ex if e.get("replay_tier")==t]
    w(f"| **{t}** | {d['replay_tiers'][t]} | {len(ids)} |")
w("")
for t in "ABC":
    rows=[e for e in ex if e.get("replay_tier")==t]
    w(f"**Tier {t}** — " + ", ".join(e["label"] for e in rows) + ".")
    w("")

w("## Unlogged audio")
w("")
w("Input wavs that exist on disk with no corresponding record in `session.jsonl`.")
w("These are utterances the machine heard and dropped — mostly the last thing said")
w("before a session ended.")
w("")
w("| Session | Turn | Duration | File |")
w("|---|---|---|---|")
n=0
for e in ex:
    for o in e.get("unlogged_audio",[]):
        n+=1
        w(f"| `{e['session_id']}` | {o['turn']} | {o['duration_s']}s | `{o['file'].split('/')[-1]}` |")
w("")
w(f"{n} in total across the flagged sessions. Text was never written for these; the audio is the only record.")
w("")

tot_empty=sum(e.get("counts",{}).get("empty_wavs",0) for e in ex)
w("## Empty and truncated audio")
w("")
w(f"**{tot_empty} wav files across the flagged sessions are present but contain no audio** —")
w("almost all are 44-byte WAV headers with zero frames, written when Piper produced no output.")
w("A handful of input wavs are truncated mid-chunk and fail to open.")
w("")
w("This matters: a raw file count overstates what is actually playable. The tables above show")
w("`playable / total` wherever the two differ, and every affected turn in the index carries")
w("`audio_playable: false`. The worst case is `session-20260126-221545`, where most response")
w("wavs are empty headers.")
w("")
w("| Session | Empty wavs | Playable synth | Total synth |")
w("|---|---|---|---|")
for e in sorted(ex, key=lambda x:-x.get("counts",{}).get("empty_wavs",0))[:10]:
    c=e.get("counts",{})
    if not c.get("empty_wavs"): continue
    w(f"| `{e['session_id']}` | {c['empty_wavs']} | {c.get('response_wavs_playable',0)} | {c.get('response_wavs',0)} |")
w("")

w("## Syncing notes")
w("")
for k,v in d["sync"].items():
    w(f"- **{k.replace('_',' ')}** — {v}")
w("- Chunk boundaries are ~5 min; a single exchange can straddle chunks, so concatenate before seeking.")
w("- `lidar.jsonl` is the storage hog — up to 187MB for one chunk. Decimate before shipping anything to a browser.")
w("")

w("## Caveats")
w("")
w("1. **Duplicates.** `bff-logs-siggraph-2026-dev/` and `bff-logs-organized/` re-copy many sessions")
w("   from MAC/HELPER/SNAPPER. The index records every host under `hosts` and picks one as")
w("   `primary_host`; all counts and paths refer to that copy only. Never sum across hosts.")
w("2. **Clock.** `session-19691231-160201` (Ace Hotel) ran with the Jetson RTC unset. Its wavs and")
w("   telemetry are internally consistent but its epoch timestamps will not align with anything")
w("   else from that day. Hand-offset it if you replay it alongside another stream.")
w("3. **Highlight ranges are keyword matches, not verified.** Each exchange carries a `highlight`")
w("   block in the index giving the turn range where its defining phrase appears, with")
w("   `confidence: keyword-match, unverified`. Three sessions produced no match and are marked")
w("   as such. Listen before you cut.")
w("4. **Transcripts are `tiny.en`.** Every `text` field in the index is Whisper tiny output,")
w("   recorded live. Short utterances are unreliable. Re-transcribe with a larger model before")
w("   quoting anything as text.")
w("")

open(os.path.join(HERE, "BFF-dialogue-media-report.md"),"w").write("\n".join(L)+"\n")
print("wrote BFF-dialogue-media-report.md", len("\n".join(L)), "chars")
