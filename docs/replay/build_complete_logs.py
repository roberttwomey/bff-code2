#!/usr/bin/env python3
"""Merge the re-transcriptions back into session.jsonl-shaped logs.

The raw logs under logs-all/ are never modified. This writes, per session:

    processed/complete-logs/<group>/<session>/session-complete.jsonl

Same record shape as the original so existing tooling reads it unchanged, but:
  - `text` is the re-transcription; `logged_text` keeps the original tiny.en line
  - utterances absent from the original log are inserted, marked source=recovered
  - speaker comes from FILE KIND, not record type, so the dog's mic bleed stops
    being recorded as a human turn
  - turns are renumbered monotonically by wav mtime; the original (unreliable,
    frequently colliding) number is kept as `original_turn`
  - non-audio events from the original log (scene_switch, vlm_query, session_end)
    are interleaved by timestamp so the log is genuinely complete
"""
import os, json, datetime, collections

ROOT    = os.environ.get("BFF_ARCHIVE_ROOT", "/Volumes/Cohab2024/BFF/logs-all")
DERIVED = os.environ.get("BFF_DERIVED_ROOT", "/Volumes/Cohab2024/BFF/processed")
RETRANS = os.path.join(DERIVED, "retranscribed")
OUT     = os.path.join(DERIVED, "complete-logs")
SKIP_GROUPS = {"_model-comparison"}
# streaming partials are superseded by the response wav; audio-bearing records are
# already merged into their wav record via logged_text
DROP_TYPES = {"assistant_chunk", "session_start"}

def iso_epoch(s):
    try: return datetime.datetime.fromisoformat(s).timestamp()
    except Exception: return None

def dog_name(cfg, group):
    p = (cfg.get("system_prompt") or "").lower()
    if "you are snapper" in p: return "SNAPPER"
    if "you are helper" in p or "you are, helper" in p: return "HELPER"
    if "helper" in p: return "HELPER"
    if "snapper" in p: return "SNAPPER"
    return group.upper() if group in ("snapper", "helper") else "DOG"

def raw_group(group, sid):
    """Where the raw session lives NOW. 40 sessions were re-attributed after the
    transcription ran, so the bundle's own group can be stale."""
    if os.path.isdir(os.path.join(ROOT, group, sid)): return group
    for g in os.listdir(ROOT):
        if os.path.isdir(os.path.join(ROOT, g, sid)): return g
    return group

def build(group, sid):
    rpath = os.path.join(RETRANS, group, sid, "retranscription.json")
    if not os.path.exists(rpath): return None
    rt = json.load(open(rpath))
    rgroup = raw_group(group, sid)
    sdir = os.path.join(ROOT, rgroup, sid)

    orig, cfg, ended = [], {}, None
    lf = os.path.join(sdir, "session.jsonl")
    if os.path.exists(lf):
        for line in open(lf, errors="replace"):
            if not line.strip(): continue
            try: r = json.loads(line)
            except Exception: continue
            if r.get("type") == "session_start": cfg = r.get("config") or {}
            if r.get("type") == "session_end": ended = r.get("timestamp")
            orig.append(r)
    dog = dog_name(cfg, rgroup)

    events = []
    for w in rt["wavs"]:
        if w["kind"] == "room": continue          # continuous audio, not a turn
        txt = w.get("text")
        if not txt and w.get("duration_s") is None:
            continue                               # empty wav, nothing to say
        kind = w["kind"]
        typ = {"input": "user", "response": "assistant", "startup": "startup"}.get(
            kind, "cue" if kind.startswith("cue") else kind)
        if typ == "user":
            speaker = "HUMAN"
        elif typ == "assistant":
            speaker = dog
        else:
            speaker = "SYSTEM"
        rec = {
          "epoch": w.get("mtime_epoch"),
          "timestamp": w.get("mtime_iso"),
          "type": typ,
          "speaker": speaker,
          "text": txt,
          "logged_text": w.get("logged_text"),
          "source": "logged" if w.get("in_transcript", True) else "recovered",
          "audio_path": os.path.join(rgroup, sid, w["file"]),
          "duration_s": w.get("duration_s"),
          "block": w.get("block"),
          "original_turn": w.get("turn"),
        }
        if kind.startswith("cue"): rec["cue"] = kind.split(":", 1)[1]
        if w.get("likely_speaker_bleed"):
            rec["speaker_bleed"] = True
            rec["bleed_similarity"] = w.get("bleed_similarity")
            rec["probable_speaker"] = dog
        events.append(rec)

    for r in orig:
        if r.get("type") in DROP_TYPES: continue
        if r.get("audio_path"): continue            # already merged into its wav record
        ep = iso_epoch(r.get("timestamp") or "")
        if ep is None: continue
        extra = {k: v for k, v in r.items()
                 if k not in ("timestamp", "type", "text", "config", "turn")}
        if "turn" in r: extra["original_turn"] = r["turn"]
        events.append({"epoch": ep, "timestamp": r.get("timestamp"), "type": r.get("type"),
                       "source": "logged", "text": r.get("text"), **extra})

    events.sort(key=lambda e: (e.get("epoch") or 0))
    n = 0
    for e in events:
        if e["type"] in ("user", "assistant"):
            n += 1; e["turn"] = n
    for e in events: e.pop("epoch", None)

    counts = collections.Counter(e["type"] for e in events)
    header = {
      "type": "session_start", "session_id": sid, "group": rgroup,
      "timestamp": (orig[0].get("timestamp") if orig else None),
      "config": cfg,
      "derived": {
        "schema": "bff.session-complete/1",
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "transcription_model": rt.get("model"),
        "speaker_attribution": "by file kind: input=mic, response=piper output",
        "turn_numbering": "renumbered monotonically by wav mtime; original_turn keeps the source value",
        "note": ("`text` is the re-transcription, `logged_text` the original live tiny.en line. "
                 "source=recovered marks utterances absent from the original session.jsonl."),
        "blocks": rt.get("blocks"),
        "counts": {"turns": n, "recovered": sum(1 for e in events if e.get("source") == "recovered"),
                   "speaker_bleed": sum(1 for e in events if e.get("speaker_bleed")),
                   "by_type": dict(counts)},
      },
    }
    odir = os.path.join(OUT, rgroup, sid)
    os.makedirs(odir, exist_ok=True)
    with open(os.path.join(odir, "session-complete.jsonl"), "w") as fh:
        fh.write(json.dumps(header) + "\n")
        for e in events: fh.write(json.dumps(e) + "\n")
        if ended: fh.write(json.dumps({"type": "session_end", "timestamp": ended}) + "\n")
    return header["derived"]["counts"]

def main():
    tot = collections.Counter(); n = 0
    for group in sorted(os.listdir(RETRANS)):
        gp = os.path.join(RETRANS, group)
        if not os.path.isdir(gp) or group in SKIP_GROUPS: continue
        for sid in sorted(os.listdir(gp)):
            if not os.path.isdir(os.path.join(gp, sid)): continue
            c = build(group, sid)
            if c is None: continue
            n += 1
            tot["turns"] += c["turns"]; tot["recovered"] += c["recovered"]
            tot["bleed"] += c["speaker_bleed"]
            if n % 150 == 0: print(f"  {n} sessions...", flush=True)
    print(f"\nwrote {n} complete logs -> {OUT}")
    print(f"  turns={tot['turns']}  recovered={tot['recovered']}  speaker_bleed={tot['bleed']}")

main()
