#!/usr/bin/env python3
"""Re-attribute sessions to a machine group, moving derived artifacts with them.

    python label_sessions.py --to snapper --basis "2026-07-20 21:35 shoot" \
        session-20260720-213420 [...]

Moves the raw session AND its re-transcription bundle and complete-log, rewrites
the group recorded inside each, and records the change in MANIFEST.json with
`evidence: operator judgement` so it never reads as proven from paths.

Nothing is guessed: the caller supplies the label. Run verify_derived.py after.
"""
import os, sys, json, shutil, argparse, datetime

ROOT    = os.environ.get("BFF_ARCHIVE_ROOT", "/Volumes/Cohab2024/BFF/logs-all")
DERIVED = os.environ.get("BFF_DERIVED_ROOT", "/Volumes/Cohab2024/BFF/processed")
GROUPS  = ["snapper", "helper", "mac"]

def locate(base, sid):
    for g in sorted(os.listdir(base)):
        if os.path.isdir(os.path.join(base, g, sid)): return g
    return None

def move_raw(sid, dst):
    src = locate(ROOT, sid)
    if src is None: return None, "not found in archive"
    if src == dst:  return src, "already there"
    s, d = os.path.join(ROOT, src, sid), os.path.join(ROOT, dst, sid)
    if os.path.exists(d): return src, f"COLLISION: already exists in {dst}"
    shutil.move(s, d)
    return src, "moved"

def move_derived(sid, src, dst):
    notes = []
    for tree in ["retranscribed", "complete-logs"]:
        base = os.path.join(DERIVED, tree)
        if not os.path.isdir(base): continue
        g = locate(base, sid)
        if g is None or g == dst: continue
        s, d = os.path.join(base, g, sid), os.path.join(base, dst, sid)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if os.path.exists(d): notes.append(f"{tree}: COLLISION"); continue
        shutil.move(s, d)
        if tree == "retranscribed":
            f = os.path.join(d, "retranscription.json")
            j = json.load(open(f)); j["group"] = dst
            j["regrouped"] = {"from": g, "on": datetime.date.today().isoformat(),
                              "reason": "session re-attributed after transcription ran"}
            json.dump(j, open(f, "w"), indent=1)
        else:
            f = os.path.join(d, "session-complete.jsonl")
            lines = open(f, errors="replace").read().split("\n")
            h = json.loads(lines[0]); h["group"] = dst; lines[0] = json.dumps(h)
            for i, ln in enumerate(lines[1:], 1):
                if not ln.strip(): continue
                try: rec = json.loads(ln)
                except Exception: continue
                ap = rec.get("audio_path", "")
                if ap.startswith(g + "/"):
                    rec["audio_path"] = ap.replace(g + "/", dst + "/", 1)
                    lines[i] = json.dumps(rec)
            open(f, "w").write("\n".join(lines))
        notes.append(f"{tree}: {g} -> {dst}")
        if not os.listdir(os.path.join(base, g)): os.rmdir(os.path.join(base, g))
    return notes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--to", required=True, choices=GROUPS)
    ap.add_argument("--basis", required=True, help="why - recorded in the manifest")
    a = ap.parse_args()

    mp = os.path.join(ROOT, "MANIFEST.json"); m = json.load(open(mp))
    idx = {r["session_id"]: r for r in m["sessions"]}
    changed = 0
    for sid in a.sessions:
        src, status = move_raw(sid, a.to)
        print(f"{sid}: {status}" + (f" ({src} -> {a.to})" if status == "moved" else ""))
        if status != "moved": continue
        for n in move_derived(sid, src, a.to): print(f"    {n}")
        r = idx.get(sid)
        if r:
            r["dest"] = f"{a.to}/{sid}"; r["recorded_on"] = a.to
            r["evidence"] = "operator judgement (not proven from paths)"
            r["reattributed"] = {"from": src, "on": datetime.date.today().isoformat(),
                                 "basis": a.basis}
            changed += 1
    json.dump(m, open(mp, "w"), indent=1)
    oj = sum(1 for r in m["sessions"] if str(r.get("evidence", "")).startswith("operator judgement"))
    print(f"\nmanifest updated for {changed}; operator-labelled total now {oj}")
    for g in GROUPS + ["_review-unattributed-robot", "_review-unknown"]:
        p = os.path.join(ROOT, g)
        if os.path.isdir(p):
            print(f"  {g:30s} {len([x for x in os.listdir(p) if os.path.isdir(os.path.join(p,x))])}")
    print("\nnow run: python verify_derived.py")

main()
