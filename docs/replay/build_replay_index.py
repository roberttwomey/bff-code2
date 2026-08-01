#!/usr/bin/env python3
"""Build a machine-readable replay index for the flagged BFF exchanges.

Usage:  python build_replay_index.py
Set BFF_ARCHIVE_ROOT to point at the logs-all archive if it is not at the
default path. Writes bff-replay-index.json next to this script.
"""
import os, re, json, glob, wave, contextlib, datetime, hashlib

ROOT = os.environ.get("BFF_ARCHIVE_ROOT",
                      "/Volumes/Cohab2024/BFF/SIGGRAPH 2026 BFF/logs-all")
HERE = os.path.dirname(os.path.abspath(__file__))
HOST_PREF = ["bff-logs-SNAPPER", "bff-logs-HELPER", "bff-logs-MAC", "bff-logs-siggraph-2026-dev"]

# phase, label, session id, keywords locating the highlighted moment in the transcript
FLAGGED = [
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
 ("2-NeurIPS","2025-12","domestic capture — fox-like dog names","session-20251207-081946",["fox"]),
 ("2-NeurIPS","2025-12","domestic capture — breakfast / nature show","session-20251207-072459",["breakfast"]),
 ("2-NeurIPS","2025-12","quantum system 'Q' prompt","session-20251219-100935",["quantum"]),
 ("3-IDEAS","2026-01-29","'named you Helper' / distributed self","session-20260126-221545",["named you helper","distributed"]),
 ("3-IDEAS","2026-01-29","'you clanker'","session-20260126-221552",["clanker"]),
 ("3-IDEAS","2026-01-29","shutdown loop","session-20260126-224024",["shutdown","shut down"]),
 ("3-IDEAS","2026-01-29","'Actually, I'm a human'","session-20260127-193910",["i'm a human","im a human","a human"]),
 ("3-IDEAS","2026-01-29","'I will only ask you to be a machine' / burp","session-20260128-100651",["be a machine"]),
 ("3-IDEAS","2026-01-29","Helper scripted self-intro","session-20260128-101724",["my name is helper","i am helper"]),
 ("3-IDEAS","2026-01-29","'I don't want you to compliment me'","session-20260128-105613",["compliment"]),
 ("3-IDEAS","2026-01-29","Colors scene / 'would it be pleasurable'","session-20260129-115828",["pleasurable"]),
 ("3-IDEAS","2026-01-29","confabulated hand-on-head memory","session-20260129-120117",["hand on my head","your hand"]),
 ("3-IDEAS","2026-01-29","'why do you keep mentioning my family'","session-20260129-140838",["my family"]),
 ("3-IDEAS","2026-01-29","black box in a black box","session-20260129-142720",["black box"]),
 ("3-IDEAS","2026-01-29","performance + post-show Q&A","session-20260129-164334",[]),
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
 ("4-SIGGRAPH","2026-07-23","SIGGRAPH stage / magic","session-20260723-092744",["magic"]),
]

def wav_dur(p):
    try:
        with contextlib.closing(wave.open(p)) as w:
            n = w.getnframes()
            return round(n/w.getframerate(), 3) if n else None
    except Exception:
        return None

def iso_to_epoch(s):
    try: return datetime.datetime.fromisoformat(s).timestamp()
    except Exception: return None

def first_last(path, key="timestamp"):
    """First and last value of `key` in a jsonl, without loading the whole file."""
    first = last = None
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                if not line.strip(): continue
                try: r = json.loads(line)
                except Exception: continue
                v = r.get(key)
                if v is None: continue
                if first is None: first = v
                last = v
    except Exception:
        pass
    return first, last

def chunk_info(cdir):
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
            info[k] = {"path": os.path.relpath(p, ROOT), "bytes": os.path.getsize(p)}
    return info

def build_session(phase, when, label, sid, kws):
    ent = {"phase": phase, "event_window": when, "label": label, "session_id": sid}
    # locate copies
    copies = [d for d in glob.glob(os.path.join(ROOT, "*", sid)) if "organized" not in d]
    if sid.startswith("chat_session"):
        copies = [p for p in glob.glob(os.path.join(ROOT, "*", sid + ".jsonl")) if "organized" not in p]
        ent["format"] = "chat_session_v1"
        ent["replay_tier"] = "C"
        ent["hosts"] = sorted({os.path.basename(os.path.dirname(p)) for p in copies})
        if not copies:
            ent["status"] = "MISSING"; return ent
        best = sorted(copies, key=lambda p: HOST_PREF.index(os.path.basename(os.path.dirname(p)))
                      if os.path.basename(os.path.dirname(p)) in HOST_PREF else 99)[0]
        ent["transcript"] = os.path.relpath(best, ROOT)
        ent["media"] = {"note": "transcript only; this log format never wrote audio"}
        ent["turns"] = []
        return ent

    ent["format"] = "session_v2"
    ent["hosts"] = sorted({os.path.basename(os.path.dirname(d)) for d in copies})
    if not copies:
        ent["status"] = "MISSING"; return ent
    best = sorted(copies, key=lambda d: HOST_PREF.index(os.path.basename(os.path.dirname(d)))
                  if os.path.basename(os.path.dirname(d)) in HOST_PREF else 99)[0]
    ent["primary_host"] = os.path.basename(os.path.dirname(best))
    ent["session_dir"] = os.path.relpath(best, ROOT)
    ent["transcript"] = os.path.relpath(os.path.join(best, "session.jsonl"), ROOT)

    recs = []
    lf = os.path.join(best, "session.jsonl")
    if os.path.exists(lf):
        for line in open(lf, errors="replace"):
            if not line.strip(): continue
            try: recs.append(json.loads(line))
            except Exception: pass
    if recs and recs[0].get("type") == "session_start":
        ent["session_start_iso"] = recs[0].get("timestamp")
        ent["session_start_epoch"] = iso_to_epoch(recs[0].get("timestamp") or "")
        cfg = recs[0].get("config") or {}
        ent["config"] = {k: cfg.get(k) for k in
                         ("ollama_model","vlm_model","whisper_model","piper_voice","system_prompt",
                          "require_wakeword","wake_phrases","no_vlm","no_body") if k in cfg}
    if recs and recs[-1].get("type") == "session_end":
        ent["session_end_iso"] = recs[-1].get("timestamp")
        ent["session_end_epoch"] = iso_to_epoch(recs[-1].get("timestamp") or "")

    # turn-level records that carry audio
    turns = []
    for r in recs:
        if r.get("type") not in ("user","assistant","reset","scene_switch","stop_conversation",
                                 "special_command","vad_segment","user_no_wake"): continue
        t = {"turn": r.get("turn"), "type": r.get("type"), "iso": r.get("timestamp"),
             "epoch": iso_to_epoch(r.get("timestamp") or ""), "text": r.get("text")}
        if r.get("speaker"): t["speaker"] = r["speaker"]
        if r.get("disposition"): t["disposition"] = r["disposition"]
        ap = r.get("audio_path")
        if ap:
            local = os.path.join(best, os.path.basename(ap))
            if os.path.exists(local):
                t["audio"] = os.path.relpath(local, ROOT)
                dur = wav_dur(local)
                t["audio_duration_s"] = dur
                if dur is None:
                    t["audio_playable"] = False
                    t["audio_note"] = "file present but empty or truncated"

            else:
                t["audio_missing"] = os.path.basename(ap)
        turns.append(t)
    ent["turns"] = turns

    # wavs on disk with no record referencing them
    referenced = {os.path.basename(r["audio_path"]) for r in recs if r.get("audio_path")}
    orphans = []
    for f in sorted(os.listdir(best)):
        if re.match(r"turn-\d+-input\.wav$", f) and f not in referenced:
            orphans.append({"file": os.path.relpath(os.path.join(best, f), ROOT),
                            "turn": int(f[5:8]), "duration_s": wav_dur(os.path.join(best, f)),
                            "note": "audio on disk, no session.jsonl record"})
    if orphans: ent["unlogged_audio"] = orphans

    # media
    media = {}
    cues = [f for f in sorted(os.listdir(best))
            if re.match(r"turn-\d+-(reset|start-listening|stop-listening|reprompt|didnt-catch|wake|goodbye|special|stop-conversation|vlm-ack)\.wav$", f)]
    if cues: media["cue_wavs"] = [os.path.relpath(os.path.join(best,f), ROOT) for f in cues]
    sp = os.path.join(best, "startup.wav")
    if os.path.exists(sp): media["startup_wav"] = os.path.relpath(sp, ROOT)

    vd = os.path.join(best, "vlm_captures")
    if os.path.isdir(vd):
        caps = []
        for f in sorted(os.listdir(vd)):
            m = re.match(r"description_(\d{8}-\d{6})\.txt$", f)
            if not m: continue
            stem = m.group(1)
            snap = os.path.join(vd, f"snapshot_{stem}.jpg")
            try: txt = open(os.path.join(vd,f), errors="replace").read().strip()
            except Exception: txt = None
            try:
                dt = datetime.datetime.strptime(stem, "%Y%m%d-%H%M%S")
                ep = dt.timestamp()
            except Exception:
                ep = None
            caps.append({"stamp": stem, "epoch": ep, "caption": txt,
                         "description": os.path.relpath(os.path.join(vd,f), ROOT),
                         "snapshot": os.path.relpath(snap, ROOT) if os.path.exists(snap) else None})
        if caps: media["vlm_captures"] = caps

    chunks = sorted(glob.glob(os.path.join(best, "chunk_*")),
                    key=lambda p: int(re.sub(r"\D","",os.path.basename(p)) or 0))
    if chunks:
        media["chunks"] = [chunk_info(c) for c in chunks]
        eps = [c["start_epoch"] for c in media["chunks"] if c.get("start_epoch")]
        eph = [c["end_epoch"] for c in media["chunks"] if c.get("end_epoch")]
        if eps and eph:
            media["telemetry_span_epoch"] = [min(eps), max(eph)]
    cp = os.path.join(best, "camera_path.jsonl")
    if os.path.exists(cp): media["camera_path"] = os.path.relpath(cp, ROOT)
    ent["media"] = media

    # counts, measured on disk (authoritative for the report)
    cnt = {"input_wavs":0,"input_wavs_playable":0,"response_wavs":0,"response_wavs_playable":0,
           "cue_wavs":0,"vlm_snapshots":0,"vlm_descriptions":0,"chunks":len(chunks),
           "human_audio_s":0.0,"tts_audio_s":0.0,"empty_wavs":0,"session_bytes":0}
    for r_,_d,fs in os.walk(best):
        for f in fs:
            p=os.path.join(r_,f); cnt["session_bytes"]+=os.path.getsize(p)
            if re.match(r"turn-\d+-input\.wav$",f):
                cnt["input_wavs"]+=1; dur=wav_dur(p)
                if dur is None: cnt["empty_wavs"]+=1
                else: cnt["input_wavs_playable"]+=1; cnt["human_audio_s"]+=dur
            elif re.match(r"turn-\d+-response\.wav$",f):
                cnt["response_wavs"]+=1; dur=wav_dur(p)
                if dur is None: cnt["empty_wavs"]+=1
                else: cnt["response_wavs_playable"]+=1; cnt["tts_audio_s"]+=dur
            elif re.match(r"turn-\d+-.*\.wav$",f): cnt["cue_wavs"]+=1
            elif f.startswith("snapshot_"): cnt["vlm_snapshots"]+=1
            elif f.startswith("description_"): cnt["vlm_descriptions"]+=1
    cnt["human_audio_s"]=round(cnt["human_audio_s"],1)
    cnt["tts_audio_s"]=round(cnt["tts_audio_s"],1)
    cnt["video_bytes"]=sum(c["video"]["bytes"] for c in media.get("chunks",[]) if c.get("video"))
    ent["counts"]=cnt

    # replay tier
    has_video = any(c.get("video") for c in media.get("chunks", []))
    has_audio = any(t.get("audio") for t in turns)
    ent["replay_tier"] = "A" if has_video else ("B" if has_audio else "C")

    # highlight turn range
    if kws:
        hits = [t for t in turns if t.get("text") and
                any(k in t["text"].lower() for k in kws)]
        if hits:
            ns = [h["turn"] for h in hits if h.get("turn") is not None]
            if ns:
                ent["highlight"] = {"start_turn": min(ns), "end_turn": max(ns),
                                    "matched_keywords": kws, "match_count": len(hits),
                                    "confidence": "keyword-match, unverified"}
        else:
            ent["highlight"] = {"matched_keywords": kws, "match_count": 0,
                                "confidence": "no keyword match; whole session"}
    return ent

def main():
    out = {
      "schema": "bff.replay-index/1",
      "generated": datetime.datetime.now().isoformat(timespec="seconds"),
      "archive_root": ROOT,
      "path_convention": "all paths relative to archive_root",
      "sync": {
        "telemetry_clock": "unix epoch float, field `timestamp`, in lowstate/lidar/detections/camera_path",
        "transcript_clock": "ISO 8601 local, field `timestamp`; `epoch` added here for convenience",
        "warning_lidar_stamp": "lidar.jsonl also carries `stamp` on the robot DDS clock - do NOT use for sync",
        "video_frames": "detections.jsonl pairs frame_index with epoch; derived_fps per chunk lets you interpolate",
        "vlm_capture_clock": "filename stamp is host local time, converted to epoch here"
      },
      "replay_tiers": {
        "A": "video + lidar + lowstate + detections + VLM stills + both voices",
        "B": "turn-segmented human and synthesized voice only",
        "C": "transcript text only, no audio ever written"
      },
      "exchanges": []
    }
    for phase, when, label, sid, kws in FLAGGED:
        e = build_session(phase, when, label, sid, kws)
        out["exchanges"].append(e)
        print(f"  {sid:30s} tier={e.get('replay_tier','?')} turns={len(e.get('turns',[]))}")
    dst = os.path.join(HERE, "bff-replay-index.json")
    json.dump(out, open(dst,"w"), indent=1)
    print("\nwrote", dst, os.path.getsize(dst), "bytes")

main()
