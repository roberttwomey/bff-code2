#!/usr/bin/env python3
"""Pass 2: find candidate moments in speech the original logs never recorded.

The pass-1 curation was done against the live tiny.en transcripts. 2,692 wav
files turned out to contain speech that appears in no session.jsonl record -
overwhelmingly the dog's own replies. Those were invisible when the exchanges
were chosen. This scans the re-transcribed corpus for moments in that recovered
material and scores them.

Writes bff-replay-index-pass2.json next to this script. Does not touch the
pass-1 index.
"""
import os, re, json, datetime, collections

DERIVED = os.environ.get("BFF_DERIVED_ROOT", "/Volumes/Cohab2024/BFF/processed")
HERE    = os.path.dirname(os.path.abspath(__file__))
RETRANS = os.path.join(DERIVED, "retranscribed")

# Threads established across the four phases, from the pass-1 material.
MOTIFS = {
 "apple":        ["apple","honeycrisp","fruit"],
 "memory":       ["memory","remember","first thing i","forget","recall"],
 "mirror":       ["mirror","reflection","reflect"],
 "body":         ["body","embodiment","legs","joints","my sensors","physical"],
 "sleep/dream":  ["sleep","dream","rest","recharge","concrete floor"],
 "feeling":      ["feel","feeling","enjoy","pleasure","pleasurable","lonely","afraid","sad"],
 "aliveness":    ["alive","dead","die","conscious","awake","exist","existence"],
 "naming":       ["your name","named you","call you","i am snapper","i am helper"],
 "kinship":      ["family","friend","together","companion","love you","belong"],
 "machine":      ["machine","robot","program","clanker","artificial","a tool"],
 "interiority":  ["inside","interior","between us","my own","myself","私"],
}
BOILERPLATE = re.compile(
  r"^(acknowledged|affirmative|understood|processing|roger that|yes\.?|okay\.?|ok\.?"
  r"|i am here|i'm here|greetings|hello there|initiating|scanning)[\.\s!]*$", re.I)

def load_all():
    out=[]
    for group in sorted(os.listdir(RETRANS)):
        gp=os.path.join(RETRANS, group)
        # _conflicts are duplicate copies; _model-comparison is a second
        # transcription of a session already covered. The _review-* groups are
        # real unattributed sessions and belong in the scan.
        if not os.path.isdir(gp) or group in ("_conflicts","_model-comparison"): continue
        for sid in sorted(os.listdir(gp)):
            f=os.path.join(gp, sid, "retranscription.json")
            if not os.path.exists(f): continue
            try: out.append((group, sid, json.load(open(f))))
            except Exception: pass
    return out

def phase_of(sid):
    """Phase from the session date. 1969 means the Jetson RTC was unset - those
    sessions have no usable date and must not be binned by it. CMC (phase 1) is
    excluded by construction: it predates any audio being written at all."""
    m=re.search(r"(\d{8})", sid)
    if not m: return "unknown-clock"
    d=m.group(1)
    if d.startswith("1969"): return "unknown-clock"
    if d < "20251130": return "pre-audio(unexpected)"
    if d < "20260101": return "2-NeurIPS"
    if d < "20260301": return "3-IDEAS"
    return "4-SIGGRAPH"

def score(text, kinds_before):
    s=0.0; why=[]; hit=[]
    t=text.lower()
    for name, kws in MOTIFS.items():
        if any(k in t for k in kws): hit.append(name)
    if hit: s += 2.0*len(hit); why.append("motifs: "+",".join(hit))
    n=len(text)
    if n >= 120: s += 2.0; why.append("substantive length")
    elif n >= 60: s += 1.0
    if BOILERPLATE.match(text.strip()): s -= 3.0; why.append("boilerplate")
    # first person / reflective register
    if re.search(r"\bi (am|feel|notice|don't|do not|can't|cannot|wonder|think)\b", t):
        s += 1.5; why.append("first-person reflective")
    if "?" in text: s += 0.5; why.append("asks something")
    # completes an exchange: a logged human line immediately precedes
    if kinds_before: s += 1.5; why.append("answers a logged human line")
    return s, why, hit

def main():
    bundles=load_all()
    print(f"{len(bundles)} re-transcription bundles")
    cands=[]; total_recovered=0
    for group, sid, rt in bundles:
        wavs=rt["wavs"]
        for i,w in enumerate(wavs):
            if w.get("in_transcript") is not False: continue
            txt=(w.get("text") or "").strip()
            if not txt: continue
            total_recovered+=1
            if w["kind"] not in ("response","input"): continue
            prev=next((x for x in reversed(wavs[:i])
                       if x["kind"]=="input" and x.get("logged_text")), None)
            sc, why, hit = score(txt, prev is not None)
            if w["kind"]=="response": sc += 1.0; why.append("the dog's own voice")
            if sc < 5.0: continue
            nxt=wavs[i+1] if i+1 < len(wavs) else None
            cands.append({
              "score": round(sc,1), "phase": phase_of(sid), "group": group,
              "session_id": sid, "block": w.get("block"),
              "file": w["file"], "kind": w["kind"],
              "iso": w.get("mtime_iso"), "epoch": w.get("mtime_epoch"),
              "duration_s": w.get("duration_s"),
              "text": txt, "motifs": hit, "why": why,
              "preceding_logged_human": (prev or {}).get("logged_text"),
              "preceding_file": (prev or {}).get("file"),
              "following_text": (nxt or {}).get("text"),
              "audio": f"{group}/{sid}/{w['file']}",
            })
    cands.sort(key=lambda c: -c["score"])
    print(f"recovered utterances scanned: {total_recovered}")
    print(f"candidates above threshold  : {len(cands)}")
    out={
      "schema": "bff.replay-index.pass2/1",
      "generated": datetime.datetime.now().isoformat(timespec="seconds"),
      "derived_root": DERIVED,
      "archive_root": os.environ.get("BFF_ARCHIVE_ROOT","/Volumes/Cohab2024/BFF/logs-all"),
      "what_this_is": ("Candidate moments drawn ONLY from speech absent from the original "
                       "session.jsonl logs. Pass-1 curation could not have seen these. "
                       "Scores are heuristic (motif hits, length, reflective register, "
                       "whether it answers a logged human line) - a ranking aid, not a judgement. "
                       "Nothing here is verified by listening."),
      "audio_path_convention": "`audio` is relative to archive_root/<group>/<session_id>/",
      "motifs": MOTIFS,
      "counts": {"bundles": len(bundles), "recovered_scanned": total_recovered,
                 "candidates": len(cands),
                 "by_phase": dict(collections.Counter(c["phase"] for c in cands)),
                 "by_motif": dict(collections.Counter(m for c in cands for m in c["motifs"]))},
      "candidates": cands,
    }
    p=os.path.join(HERE,"bff-replay-index-pass2.json")
    json.dump(out, open(p,"w"), indent=1)
    print(f"wrote {p}  {os.path.getsize(p)/1e6:.1f} MB")
    print("\nby phase:", out["counts"]["by_phase"])
    print("by motif:", dict(sorted(out["counts"]["by_motif"].items(), key=lambda x:-x[1])))

main()
