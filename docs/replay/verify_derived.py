#!/usr/bin/env python3
"""Cross-check the derived trees against the raw archive.

Every derived artifact is keyed by the machine group a session lives in, so
re-attributing a session silently invalidates anything built earlier. That has
already happened twice. This checks the invariants that catch it.

    python verify_derived.py            # full check
    python verify_derived.py --quiet    # only failures

Exit status 0 if every check passes, 1 otherwise, so it can gate a commit.
Env: BFF_ARCHIVE_ROOT, BFF_DERIVED_ROOT.
"""
import os, re, sys, json, argparse

ROOT    = os.environ.get("BFF_ARCHIVE_ROOT", "/Volumes/Cohab2024/BFF/logs-all")
DERIVED = os.environ.get("BFF_DERIVED_ROOT", "/Volumes/Cohab2024/BFF/processed")
HERE    = os.path.dirname(os.path.abspath(__file__))
RETRANS = os.path.join(DERIVED, "retranscribed")
COMPLETE= os.path.join(DERIVED, "complete-logs")
SKIP    = {"_model-comparison"}

# Piper wavs written from this date declare 22050 Hz but hold 48000 Hz samples,
# so they must be transcribed at 48 kHz and carry a `rate_corrected` block.
# startup.wav is written by a different path and is genuinely 22050 - converting
# it makes transcription markedly worse, so it must NOT be marked. Both
# directions are checked: the first pass at this fix got the second one wrong.
RATE_CUTOFF = "20260721"

class Report:
    def __init__(self, quiet): self.quiet=quiet; self.failed=0; self.checks=0
    def check(self, name, failures, detail=None, total=None):
        self.checks += 1
        n = len(failures)
        if n: self.failed += 1
        if n or not self.quiet:
            tail = f"  ({total} checked)" if total is not None else ""
            print(f"{'FAIL' if n else 'ok  '}  {name}: {n} problem(s){tail}")
        for f in failures[:8]:
            print(f"          {f}")
        if n > 8: print(f"          ... and {n-8} more")

def raw_sessions():
    out={}
    for g in sorted(os.listdir(ROOT)):
        gp=os.path.join(ROOT,g)
        if not os.path.isdir(gp): continue
        for sid in sorted(os.listdir(gp)):
            if os.path.isdir(os.path.join(gp,sid)) and sid.startswith("session-"):
                out.setdefault(sid, []).append(g)
    return out

def wav_count(p):
    return sum(1 for r,_,fs in os.walk(p) for f in fs if f.endswith(".wav"))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--quiet",action="store_true")
    args=ap.parse_args()
    R=Report(args.quiet)
    raw=raw_sessions()
    canonical={sid: gs[0] for sid, gs in raw.items() if gs[0] != "_conflicts"}
    print(f"archive: {len(raw)} session ids under {ROOT}")

    # 1. re-transcription bundles: directory group, in-file group, orphans
    misdir=[]; misfield=[]; orphan=[]; bundles={}
    for g in sorted(os.listdir(RETRANS)):
        gp=os.path.join(RETRANS,g)
        if not os.path.isdir(gp) or g in SKIP: continue
        for sid in sorted(os.listdir(gp)):
            f=os.path.join(gp,sid,"retranscription.json")
            if not os.path.exists(f): continue
            bundles[sid]=g
            if sid not in raw: orphan.append(f"{g}/{sid} has no raw session"); continue
            if g not in raw[sid]: misdir.append(f"{g}/{sid} -> raw lives in {raw[sid]}")
            try: d=json.load(open(f))
            except Exception: misfield.append(f"{g}/{sid} unreadable"); continue
            if d.get("group")!=g: misfield.append(f"{g}/{sid} in-file group={d.get('group')!r}")
    R.check("retranscription bundle is in the group its session lives in", misdir, total=len(bundles))
    R.check("retranscription in-file `group` matches its directory", misfield, total=len(bundles))
    R.check("no orphaned retranscription bundles", orphan, total=len(bundles))

    # 1b. sample-rate correction, both directions
    unmarked, wrongly_marked = [], []
    nresp = nstartup = 0
    for g in sorted(os.listdir(RETRANS)):
        gp = os.path.join(RETRANS, g)
        if not os.path.isdir(gp) or g in SKIP: continue
        for sid in sorted(os.listdir(gp)):
            f = os.path.join(gp, sid, "retranscription.json")
            if not os.path.exists(f): continue
            m = re.search(r"session-(\d{8})", sid)
            if not m: continue
            try: d = json.load(open(f))
            except Exception: continue
            for w in d.get("wavs", []):
                base = os.path.basename(w.get("file", ""))
                if base == "startup.wav":
                    nstartup += 1
                    if w.get("rate_corrected"):
                        wrongly_marked.append(f"{g}/{sid}/{base} — startup.wav is genuinely 22050")
                elif base.endswith("-response.wav") and w.get("duration_s") is not None:
                    if m.group(1) < RATE_CUTOFF: continue
                    nresp += 1
                    if not w.get("rate_corrected"):
                        unmarked.append(f"{g}/{sid}/{base}")
    R.check(f"response wavs dated >= {RATE_CUTOFF} are rate-corrected to 48 kHz",
            unmarked, total=nresp)
    R.check("startup.wav is never rate-corrected", wrongly_marked, total=nstartup)

    # 2. every session with audio has a bundle
    nobundle=[f"{canonical[sid]}/{sid} ({wav_count(os.path.join(ROOT,canonical[sid],sid))} wavs)"
              for sid in canonical
              if sid not in bundles and wav_count(os.path.join(ROOT,canonical[sid],sid))]
    R.check("every session with wavs has a re-transcription bundle", nobundle, total=len(canonical))

    # 3. complete-logs: group placement, turn monotonicity, audio_path resolution
    badgrp=[]; badturn=[]; badaudio=[]; nlogs=0; naudio=0
    if os.path.isdir(COMPLETE):
        for g in sorted(os.listdir(COMPLETE)):
            gp=os.path.join(COMPLETE,g)
            if not os.path.isdir(gp): continue
            for sid in sorted(os.listdir(gp)):
                f=os.path.join(gp,sid,"session-complete.jsonl")
                if not os.path.exists(f): continue
                nlogs+=1
                if sid in raw and g not in raw[sid]:
                    badgrp.append(f"{g}/{sid} -> raw lives in {raw[sid]}")
                turns=[]
                for line in open(f, errors="replace"):
                    try: r=json.loads(line)
                    except Exception: badturn.append(f"{g}/{sid} unparseable line"); break
                    if "turn" in r: turns.append(r["turn"])
                    ap_=r.get("audio_path")
                    if ap_:
                        naudio+=1
                        if not os.path.exists(os.path.join(ROOT,ap_)):
                            badaudio.append(f"{g}/{sid}: {ap_}")
                if turns and turns != list(range(1,len(turns)+1)):
                    badturn.append(f"{g}/{sid} turns not 1..n")
    R.check("complete-log is in the group its session lives in", badgrp, total=nlogs)
    R.check("complete-log turns are monotonic 1..n", badturn, total=nlogs)
    R.check("complete-log audio_path resolves", badaudio, total=naudio)

    # 4. indexes: every referenced path resolves, and `group` agrees with the archive
    idxbad=[]; idxgrp=[]; nref=0
    for name, key, roots in [("bff-replay-index.json","exchanges",None),
                             ("bff-sessions.json","sessions",None),
                             ("bff-replay-index-pass2.json","candidates",None)]:
        p=os.path.join(HERE,name)
        if not os.path.exists(p): continue
        d=json.load(open(p))
        for e in d.get(key,[]):
            sid=e.get("session_id")
            g=e.get("group")
            if g and sid in raw and g not in raw[sid] and not e.get("is_variant"):
                idxgrp.append(f"{name}: {sid} says group={g}, raw lives in {raw[sid]}")
            for path, base in [(e.get("transcript"), ROOT),
                               ((e.get("retranscription") or {}).get("path"), DERIVED),
                               (e.get("audio"), ROOT)]:
                if not path: continue
                nref+=1
                if not os.path.exists(os.path.join(base,path)):
                    idxbad.append(f"{name}: {path}")
    R.check("index paths resolve", idxbad, total=nref)
    R.check("index `group` agrees with the archive", idxgrp)

    print()
    if R.failed:
        print(f"{R.failed} of {R.checks} checks FAILED")
        sys.exit(1)
    print(f"all {R.checks} checks passed")

main()
