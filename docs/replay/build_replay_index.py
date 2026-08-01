#!/usr/bin/env python3
"""Build the BFF replay index from the consolidated archive.

Writes two files next to this script:
  bff-replay-index.json  curated exchanges, full per-turn detail
  bff-sessions.json      every session, summary level, for navigation

Sources:
  BFF_ARCHIVE_ROOT   consolidated raw sessions (default /Volumes/Cohab2024/BFF/logs-all)
  BFF_DERIVED_ROOT   re-transcriptions + processed media (default .../BFF/processed)
"""
import os, re, json, glob, wave, contextlib, datetime

ROOT    = os.environ.get("BFF_ARCHIVE_ROOT", "/Volumes/Cohab2024/BFF/logs-all")
DERIVED = os.environ.get("BFF_DERIVED_ROOT", "/Volumes/Cohab2024/BFF/processed")
HERE    = os.path.dirname(os.path.abspath(__file__))
GROUPS  = ["snapper", "helper", "mac", "_review-unattributed-robot", "_review-unknown", "_conflicts"]

# phase, event window, label, session id, keywords locating the moment
CURATED = [
 ("1-CMC","2025-11-06","apple / cinnamon / sensory hallucination","chat_session_19691231_191711",["cinnamon"]),
 ("1-CMC","2025-11-06","Claremont McKenna keynote / carbon cost","chat_session_20251105_232211",["carbon cost"]),
 ("1-CMC","2025-11-06","first 'concrete floor'","chat_session_20251106_194243",["concrete floor"]),
 ("1-CMC","2025-11-06","'I love you'","chat_session_20251106_145039",["i love you"]),
 ("1-CMC","2025-11-06","specific apple / Honeycrisp","chat_session_20251106_194552",["honeycrisp"]),
 ("1-CMC","2025-11-06","earliest memory","chat_session_20251101_184352",["earliest memory"]),
 ("1-CMC","2025-11-06","'better if you had a memory'","chat_session_20251106_100247",["had a memory"]),
 ("2-NeurIPS","2025-12","'rehearsal for life alongside intelligent machines'","session-20251203-114233",["rehearsal"]),
 ("2-NeurIPS","2025-12","1b poet / 'a bright blue crayon'","session-20251207-154426",["crayon"]),
 ("2-NeurIPS","2025-12","programming paradox / 'resonance'","session-20251210-133949",["resonance"]),
 ("2-NeurIPS","2025-12","'Yes, be.'","session-20251211-140606",["yes be","yes, be"]),
 ("2-NeurIPS","2025-12","crisp apple / 'my friend Jesse'","session-20251211-104816",["jesse"]),
 ("2-NeurIPS","2025-12","domestic capture - fox-like dog names","session-20251207-081946",["fox"]),
 ("2-NeurIPS","2025-12","domestic capture - breakfast / nature show","session-20251207-072459",["breakfast"]),
 ("2-NeurIPS","2025-12","quantum system 'Q' prompt","session-20251219-100935",["quantum"]),
 ("3-IDEAS","2026-01-29","'named you Helper' / distributed self","session-20260126-221545",["named you helper","distributed"]),
 ("3-IDEAS","2026-01-29","'you clanker'","session-20260126-221552",["clanker"]),
 ("3-IDEAS","2026-01-29","shutdown loop","session-20260126-224024",["shutdown","shut down"]),
 ("3-IDEAS","2026-01-29","'Actually, I'm a human'","session-20260127-193910",["a human"]),
 ("3-IDEAS","2026-01-29","'I will only ask you to be a machine' / burp","session-20260128-100651",["be a machine"]),
 ("3-IDEAS","2026-01-29","Helper scripted self-intro","session-20260128-101724",["my name is helper","i am helper"]),
 ("3-IDEAS","2026-01-29","'I don't want you to compliment me'","session-20260128-105613",["compliment"]),
 ("3-IDEAS","2026-01-29","Colors scene / 'would it be pleasurable'","session-20260129-115828",["pleasurable"]),
 ("3-IDEAS","2026-01-29","confabulated hand-on-head memory","session-20260129-120117",["hand on my head","your hand"]),
 ("3-IDEAS","2026-01-29","'why do you keep mentioning my family'","session-20260129-140838",["my family"]),
 ("3-IDEAS","2026-01-29","black box in a black box","session-20260129-142720",["black box"]),
 ("3-IDEAS","2026-01-29","*** SNAPPER IDEAS performance (block 2 is the show)","session-20260129-144803",["concrete floor","first memory","canyon"]),
 ("3-IDEAS","2026-01-29","*** HELPER IDEAS performance + post-show Q&A","session-20260129-164334",[]),
 ("4-SIGGRAPH","2026-07-23","Francis & Jasper meet Snapper","session-20260201-190334",["francis","jasper"]),
 ("4-SIGGRAPH","2026-07-23","'what was the feeling of being activated'","session-20260701-125318",["activated"]),
 ("4-SIGGRAPH","2026-07-23","'have you seen my red ball'","session-20260718-195946",["red ball"]),
 ("4-SIGGRAPH","2026-07-23","embodied hallucination / Ace Hotel","session-19691231-160201",["ace hotel","hotel"]),
 ("4-SIGGRAPH","2026-07-23","naming denied (Jasper)","session-20260720-194648",["jasper"]),
 ("4-SIGGRAPH","2026-07-23","'I feel dead inside'","session-20260720-195543",["dead inside"]),
 ("4-SIGGRAPH","2026-07-23","apple as mirror","session-20260720-204648",["apple"]),
 ("4-SIGGRAPH","2026-07-23","Pinocchio / 'bad dog'","session-20260721-121342",["pinocchio","bad dog"]),
 ("4-SIGGRAPH","2026-07-23","Companion scene (Jesse)","session-20260721-124325",["companion","jesse"]),
 ("4-SIGGRAPH","2026-07-23","McCarthy / Lovelace","session-20260722-114251",["lovelace","mccarthy"]),
 ("4-SIGGRAPH","2026-07-23","Mirror / 'space between us' / interiority","session-20260722-123623",["space between"]),
 ("4-SIGGRAPH","2026-07-23","two dogs simultaneous","session-20260722-123630",[]),
 ("4-SIGGRAPH","2026-07-23","*** SIGGRAPH stage / magic","session-20260723-092744",["magic"]),
]
CURATED_BY_ID = {c[3]: c for c in CURATED}

def persona_of(prompt):
    p = (prompt or "").lower()
    if "you are snapper" in p: return "SNAPPER"
    if "you are helper" in p or "you are, helper" in p: return "HELPER"
    if "helper" in p: return "HELPER"
    if "snapper" in p: return "SNAPPER"
    return None

def iso_epoch(s):
    try: return datetime.datetime.fromisoformat(s).timestamp()
    except Exception: return None

def first_last(path, key="timestamp"):
    first = last = None
    try:
        for line in open(path, errors="replace"):
            if not line.strip(): continue
            try: r = json.loads(line)
            except Exception: continue
            v = r.get(key)
            if v is None: continue
            if first is None: first = v
            last = v
    except Exception: pass
    return first, last

def chunk_info(cdir, root):
    info = {"name": os.path.basename(cdir)}
    lw = os.path.join(cdir, "lowstate.jsonl")
    src = lw if os.path.exists(lw) else os.path.join(cdir, "detections.jsonl")
    t0, t1 = first_last(src)
    info["start_epoch"], info["end_epoch"] = t0, t1
    if t0 and t1: info["duration_s"] = round(t1 - t0, 3)
    det = os.path.join(cdir, "detections.jsonl")
    if os.path.exists(det):
        f0, f1 = first_last(det, "frame_index")
        info["first_frame_index"], info["last_frame_index"] = f0, f1
        if None not in (f0, f1, t0, t1) and t1 > t0:
            info["derived_fps"] = round((f1 - f0) / (t1 - t0), 3)
    for f, k in [("video.mp4","video"),("audio.wav","audio"),("lidar.jsonl","lidar"),
                 ("lowstate.jsonl","lowstate"),("detections.jsonl","detections")]:
        p = os.path.join(cdir, f)
        if os.path.exists(p):
            info[k] = {"path": os.path.relpath(p, root), "bytes": os.path.getsize(p)}
    return info

def load_retrans(group, sid):
    p = os.path.join(DERIVED, "retranscribed", group, sid, "retranscription.json")
    if not os.path.exists(p): return None
    try: return json.load(open(p))
    except Exception: return None

def processed_media(group, sid):
    p = os.path.join(DERIVED, group, sid)
    if not os.path.isdir(p): return None
    out = {"dir": os.path.relpath(p, DERIVED)}
    for f in ["video_clean.mp4","video_overlay.mp4","audio_dialogue.wav","audio_mic.wav",
              "speech.srt","yolo+vlm.srt","bodystate.srt","speech+yolo+vlm+bodystate.srt",
              "detections_combined.jsonl","report.json"]:
        fp = os.path.join(p, f)
        if os.path.exists(fp): out[f] = {"path": os.path.relpath(fp, DERIVED), "bytes": os.path.getsize(fp)}
    return out

def build(group, sid, full):
    sdir = os.path.join(ROOT, group, sid)
    ent = {"session_id": sid, "group": group, "session_dir": os.path.relpath(sdir, ROOT)}
    # _conflicts holds partial duplicate copies whose canonical lives in a machine
    # folder. They are kept for audit, never curated, and never counted as sessions.
    ent["is_variant"] = group == "_conflicts"
    cur = None if ent["is_variant"] else CURATED_BY_ID.get(sid)
    ent["curated"] = bool(cur)
    if cur:
        ent["phase"], ent["event_window"], ent["label"] = cur[0], cur[1], cur[2]

    recs = []
    lf = os.path.join(sdir, "session.jsonl")
    if os.path.exists(lf):
        ent["transcript"] = os.path.relpath(lf, ROOT)
        for line in open(lf, errors="replace"):
            if not line.strip(): continue
            try: recs.append(json.loads(line))
            except Exception: pass
    cfg = (recs[0].get("config") or {}) if recs and recs[0].get("type") == "session_start" else {}
    ent["persona"] = persona_of(cfg.get("system_prompt"))
    # By July 2026 the helper machine also ran the SNAPPER prompt, so persona does
    # not distinguish the two dogs. Voice is worth surfacing but is NOT proof of
    # machine: aru=snapper / alan=helper holds 100% for scripted sessions (those
    # with scene_switch cues) and 95% on performance days, but only 68% overall -
    # both dogs ran both voices during technical development. Use `group` for
    # machine identity; `group` is proven from absolute paths.
    if cfg.get("piper_voice"):
        ent["voice"] = os.path.basename(str(cfg["piper_voice"])).replace(".onnx", "")
    if cfg:
        ent["config"] = {k: cfg.get(k) for k in
                         ("ollama_model","vlm_model","whisper_model","piper_voice","system_prompt",
                          "require_wakeword","no_vlm","no_body") if k in cfg}
    ts = [r["timestamp"] for r in recs if r.get("timestamp")]
    if ts:
        ent["start_iso"], ent["end_iso"] = ts[0], ts[-1]
        a, b = iso_epoch(ts[0]), iso_epoch(ts[-1])
        ent["start_epoch"], ent["end_epoch"] = a, b
        if a and b: ent["wall_duration_s"] = round(b - a, 1)

    rt = load_retrans(group, sid)
    if rt:
        ent["retranscription"] = {"model": rt["model"],
                                  "path": os.path.relpath(
                                      os.path.join(DERIVED,"retranscribed",group,sid,"retranscription.json"), DERIVED)}
        ent["blocks"] = rt["blocks"]

    # counts
    c = {"input_wavs":0,"response_wavs":0,"cue_wavs":0,"empty_wavs":0,
         "recovered_speech":0,"speaker_bleed":0,"audio_s":0.0}
    if rt:
        for w in rt["wavs"]:
            k = w["kind"]
            if k == "input": c["input_wavs"] += 1
            elif k == "response": c["response_wavs"] += 1
            elif k.startswith("cue"): c["cue_wavs"] += 1
            if w.get("duration_s") is None: c["empty_wavs"] += 1
            else: c["audio_s"] += w["duration_s"]
            if w.get("in_transcript") is False and w.get("text"): c["recovered_speech"] += 1
            if w.get("likely_speaker_bleed"): c["speaker_bleed"] += 1
        c["audio_s"] = round(c["audio_s"], 1)
    ent["counts"] = c

    # media
    media = {}
    if os.path.isdir(sdir):
        vd = os.path.join(sdir, "vlm_captures")
        if os.path.isdir(vd):
            caps = []
            for f in sorted(os.listdir(vd)):
                m = re.match(r"description_(\d{8}-\d{6})\.txt$", f)
                if not m: continue
                stem = m.group(1); snap = os.path.join(vd, f"snapshot_{stem}.jpg")
                try: txt = open(os.path.join(vd, f), errors="replace").read().strip()
                except Exception: txt = None
                try: ep = datetime.datetime.strptime(stem, "%Y%m%d-%H%M%S").timestamp()
                except Exception: ep = None
                caps.append({"stamp": stem, "epoch": ep, "caption": txt,
                             "description": os.path.relpath(os.path.join(vd,f), ROOT),
                             "snapshot": os.path.relpath(snap, ROOT) if os.path.exists(snap) else None})
            if caps: media["vlm_captures"] = caps if full else len(caps)
        chunks = sorted(glob.glob(os.path.join(sdir, "chunk_*")),
                        key=lambda p: int(re.sub(r"\D","",os.path.basename(p)) or 0))
        if chunks:
            ci = [chunk_info(x, ROOT) for x in chunks]
            media["chunks"] = ci if full else len(ci)
            eps = [x["start_epoch"] for x in ci if x.get("start_epoch")]
            eph = [x["end_epoch"] for x in ci if x.get("end_epoch")]
            if eps and eph: media["telemetry_span_epoch"] = [min(eps), max(eph)]
            media["has_video"] = any(x.get("video") for x in ci)
        cp = os.path.join(sdir, "camera_path.jsonl")
        if os.path.exists(cp): media["camera_path"] = os.path.relpath(cp, ROOT)
    pm = processed_media(group, sid)
    if pm: media["processed_bundle"] = pm
    ent["media"] = media

    ent["replay_tier"] = "A" if media.get("has_video") else ("B" if c["audio_s"] else "C")

    if full and rt:
        turns = []
        for w in rt["wavs"]:
            if w["kind"] == "room": continue
            t = {"file": w["file"], "kind": w["kind"], "turn": w.get("turn"),
                 "block": w.get("block"), "iso": w.get("mtime_iso"), "epoch": w.get("mtime_epoch"),
                 "duration_s": w.get("duration_s"),
                 "text": w.get("text"), "logged_text": w.get("logged_text"),
                 "in_transcript": w.get("in_transcript", True)}
            if w.get("likely_speaker_bleed"):
                t["likely_speaker_bleed"] = True; t["bleed_similarity"] = w.get("bleed_similarity")
            turns.append(t)
        ent["turns"] = turns
        if cur and cur[4]:
            hits = [t for t in turns if t.get("text") and any(k in t["text"].lower() for k in cur[4])]
            ns = [h["turn"] for h in hits if h.get("turn") is not None]
            ent["highlight"] = ({"start_turn": min(ns), "end_turn": max(ns), "match_count": len(hits),
                                 "matched_keywords": cur[4], "confidence": "keyword-match, unverified"}
                                if ns else {"match_count": 0, "matched_keywords": cur[4],
                                            "confidence": "no keyword match; whole session"})
    return ent

def chat_session_entry(cur):
    phase, when, label, sid, kws = cur
    p = os.path.join(ROOT, "chat-sessions-v1", sid + ".jsonl")
    ent = {"session_id": sid, "curated": True, "phase": phase, "event_window": when,
           "label": label, "format": "chat_session_v1", "replay_tier": "C",
           "media": {"note": "transcript only; this log format never wrote audio"},
           "counts": {}, "turns": []}
    if os.path.exists(p):
        ent["group"] = "chat-sessions-v1"
        ent["transcript"] = os.path.relpath(p, ROOT)
        # persona lives in the system message of the accumulating array, not a config block
        try:
            for line in open(p, errors="replace"):
                if '"system"' not in line: continue
                for m in (json.loads(line).get("messages") or []):
                    if m.get("role") == "system":
                        ent["persona"] = persona_of(m.get("content")); break
                if "persona" in ent: break
        except Exception:
            pass
    else:
        ent["status"] = "NOT FOUND in consolidated tree"
    return ent

def main():
    sessions = []
    for g in GROUPS:
        base = os.path.join(ROOT, g)
        if not os.path.isdir(base): continue
        for sid in sorted(os.listdir(base)):
            if os.path.isdir(os.path.join(base, sid)) and sid.startswith("session-"):
                sessions.append((g, sid))
    print(f"{len(sessions)} sessions in tree")

    catalog = [build(g, s, full=False) for g, s in sessions]
    print(f"catalog built ({sum(1 for c in catalog if c['curated'])} curated)")

    exchanges = []
    for cur in CURATED:
        sid = cur[3]
        if sid.startswith("chat_session"):
            exchanges.append(chat_session_entry(cur)); continue
        hit = next(((g, s) for g, s in sessions if s == sid), None)
        if hit is None:
            exchanges.append({"session_id": sid, "curated": True, "phase": cur[0],
                              "label": cur[2], "status": "NOT FOUND"})
            print(f"  MISSING: {sid}")
            continue
        exchanges.append(build(hit[0], hit[1], full=True))

    meta = {
      "schema": "bff.replay-index/2",
      "generated": datetime.datetime.now().isoformat(timespec="seconds"),
      "archive_root": ROOT,
      "derived_root": DERIVED,
      "path_convention": "session/media paths relative to archive_root; processed/retranscription paths relative to derived_root",
      "transcript_note": ("`text` is re-transcribed with distil-large-v3 and attributed by file kind "
                          "(response.wav = the dog, definitively). `logged_text` is the original live "
                          "tiny.en text from session.jsonl, kept for comparison. Where they disagree, "
                          "prefer `text`: ~1-2% of assistant records point at the wrong wav."),
      "sync": {
        "telemetry_clock": "unix epoch float, field `timestamp`, in lowstate/lidar/detections/camera_path",
        "turn_clock": "wav mtime, given as `epoch` and `iso` per turn",
        "warning_lidar_stamp": "lidar.jsonl also carries `stamp` on the robot DDS clock - do NOT use for sync",
        "video_frames": "detections.jsonl pairs frame_index with epoch; derived_fps per chunk lets you interpolate",
        "blocks": "sessions often run for hours with long idle gaps; `blocks` gives the actually-active spans"
      },
      "replay_tiers": {"A": "video + lidar + lowstate + detections + VLM stills + both voices",
                       "B": "turn-segmented human and synthesized voice only",
                       "C": "transcript text only, no audio ever written"},
    }
    json.dump(dict(meta, exchanges=exchanges), open(os.path.join(HERE,"bff-replay-index.json"),"w"), indent=1)
    json.dump(dict(meta, schema="bff.sessions/2", sessions=catalog),
              open(os.path.join(HERE,"bff-sessions.json"),"w"), indent=1)
    for f in ["bff-replay-index.json","bff-sessions.json"]:
        print(f"wrote {f}  {os.path.getsize(os.path.join(HERE,f))/1e6:.1f} MB")

main()
