#!/usr/bin/env python3
"""Assemble a chronological playlist across the four project phases.

The arc runs CMC -> NeurIPS -> IDEAS -> SIGGRAPH, and what exists changes as it
goes: CMC is text with no audio ever recorded, NeurIPS and IDEAS are voice, and
SIGGRAPH adds video and telemetry. The playlist reflects that rather than
pretending the media is uniform.

CMC items come from the pass-1 tier C exchanges (the only place that era's
dialogue exists). Everything after comes from the pass-2 candidates - speech
absent from the original logs.

    python build_playlist.py --top 30 --out playlist.json
"""
import os, json, argparse, datetime, collections

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.environ.get("BFF_ARCHIVE_ROOT", "/Volumes/Cohab2024/BFF/logs-all")
PHASES = ["1-CMC", "2-NeurIPS", "3-IDEAS", "4-SIGGRAPH"]
WPS    = 2.6   # words/sec, for pacing text that has no audio

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=25, help="max items per phase")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--include-unknown-clock", action="store_true",
                    help="append the 4 sessions whose date cannot be recovered")
    ap.add_argument("--out", default=os.path.join(HERE, "playlist.json"))
    a = ap.parse_args()

    p1 = json.load(open(os.path.join(HERE, "bff-replay-index.json")))
    p2 = json.load(open(os.path.join(HERE, "bff-replay-index-pass2.json")))
    items = []

    # Phase 1: text only. Prefer each exchange's highlight range, else its opening.
    for e in [x for x in p1["exchanges"] if x.get("phase") == "1-CMC"]:
        h = e.get("highlight") or {}
        lo, hi = h.get("start_turn"), h.get("end_turn")
        turns = e.get("turns", [])
        sel = [t for t in turns if lo and hi and lo <= t["turn"] <= hi] or turns[:6]
        for t in sel:
            if not t.get("text"): continue
            words = len(t["text"].split())
            items.append({
              "phase": "1-CMC", "media": "text",
              "session_id": e["session_id"], "label": e.get("label"),
              "iso": t.get("iso"), "speaker": "HUMAN" if t["kind"] == "user" else "DOG",
              "text": t["text"],
              "duration_s": round(max(2.0, words / WPS), 1),
              "note": "no audio was ever recorded for this era",
            })

    # Phases 2-4: recovered speech, highest scoring first, then chronological
    by = collections.defaultdict(list)
    for c in p2["candidates"]:
        if c["score"] >= a.min_score: by[c["phase"]].append(c)
    for ph in PHASES[1:] + (["unknown-clock"] if a.include_unknown_clock else []):
        picked = sorted(by.get(ph, []), key=lambda c: -c["score"])[:a.top]
        picked.sort(key=lambda c: c.get("iso") or "")
        for c in picked:
            items.append({
              "phase": ph, "media": "audio",
              "session_id": c["session_id"], "label": None,
              "iso": c.get("iso"),
              "speaker": "DOG" if c["kind"] == "response" else "HUMAN",
              "text": c["text"],
              "audio": c["audio"],
              "audio_abs": os.path.join(ROOT, c["audio"]),
              "duration_s": c.get("duration_s"),
              "score": c["score"], "motifs": c["motifs"],
              "preceding_logged_human": c.get("preceding_logged_human"),
            })

    order = {p: i for i, p in enumerate(PHASES + ["unknown-clock"])}
    items.sort(key=lambda x: (order.get(x["phase"], 99), x.get("iso") or ""))

    missing = [x["audio_abs"] for x in items if x["media"] == "audio"
               and not os.path.exists(x["audio_abs"])]
    total = sum(x.get("duration_s") or 0 for x in items)
    out = {
      "schema": "bff.playlist/1",
      "generated": datetime.datetime.now().isoformat(timespec="seconds"),
      "archive_root": ROOT,
      "arc": ("CMC is text only - that logger never wrote audio. NeurIPS and IDEAS are "
              "voice. SIGGRAPH adds video and telemetry, though this playlist plays the "
              "speech. Items after CMC are speech absent from the original logs."),
      "counts": {"items": len(items), "total_s": round(total, 1),
                 "by_phase": dict(collections.Counter(x["phase"] for x in items)),
                 "by_media": dict(collections.Counter(x["media"] for x in items)),
                 "missing_audio": len(missing)},
      "items": items,
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")
    print(f"  {len(items)} items, {total/60:.1f} min")
    for ph in PHASES + ["unknown-clock"]:
        n = out["counts"]["by_phase"].get(ph)
        if not n: continue
        d = sum(x.get("duration_s") or 0 for x in items if x["phase"] == ph)
        med = {x["media"] for x in items if x["phase"] == ph}
        print(f"    {ph:14s} {n:4d} items  {d/60:5.1f} min  ({'/'.join(sorted(med))})")
    if missing: print(f"  WARNING: {len(missing)} audio files missing")

main()
