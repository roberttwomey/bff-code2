#!/usr/bin/env python3
"""Play a playlist straight through, in real time.

No scrubber, no seeking: it walks the items in order and plays each at its
recorded duration, the way --simulate streams a session forward. CMC items have
no audio, so they are held on screen for their reading time - the silence is the
point, not a gap to skip.

    python play_playlist.py                     # whole arc
    python play_playlist.py --phase 3-IDEAS     # one phase
    python play_playlist.py --dry-run           # print the running order
"""
import os, sys, json, time, argparse, subprocess, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
C = {"1-CMC": "\033[38;5;245m", "2-NeurIPS": "\033[38;5;110m",
     "3-IDEAS": "\033[38;5;179m", "4-SIGGRAPH": "\033[38;5;114m",
     "unknown-clock": "\033[38;5;240m"}
DIM, BOLD, OFF = "\033[2m", "\033[1m", "\033[0m"

def player():
    for c in (["afplay"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"], ["aplay", "-q"]):
        if shutil.which(c[0]): return c
    return None

def wrap(s, w=76, indent="      "):
    out, line = [], ""
    for word in s.split():
        if len(line) + len(word) + 1 > w: out.append(line); line = word
        else: line = (line + " " + word).strip()
    if line: out.append(line)
    return ("\n" + indent).join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--playlist", default=os.path.join(HERE, "playlist.json"))
    ap.add_argument("--phase", action="append", help="limit to phase(s)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gap", type=float, default=0.6, help="seconds between items")
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args()

    pl = json.load(open(a.playlist))
    items = [i for i in pl["items"] if not a.phase or i["phase"] in a.phase]
    play = None if (a.dry_run or a.no_audio) else player()
    if not a.dry_run and not a.no_audio and play is None:
        print("no audio player found (afplay/ffplay/aplay) - running silent", file=sys.stderr)

    total = sum(i.get("duration_s") or 0 for i in items)
    print(f"\n{BOLD}BFF — {len(items)} items, {total/60:.1f} min{OFF}")
    print(f"{DIM}{wrap(pl['arc'], 76, '')}{OFF}\n")

    phase = None
    for n, it in enumerate(items, 1):
        if it["phase"] != phase:
            phase = it["phase"]
            k = sum(1 for x in items if x["phase"] == phase)
            d = sum(x.get("duration_s") or 0 for x in items if x["phase"] == phase)
            print(f"\n{C.get(phase,'')}{BOLD}{'='*72}\n  {phase}   {k} items, {d/60:.1f} min"
                  f"   [{'text only' if it['media']=='text' else 'audio'}]\n{'='*72}{OFF}\n")
        dur = it.get("duration_s") or 2.0
        stamp = (it.get("iso") or "")[11:19]
        who = it["speaker"]
        col = C.get(it["phase"], "")
        print(f"{DIM}{n:3d}/{len(items)}  {stamp}  {it['session_id']}  {dur:5.1f}s{OFF}")
        if it.get("preceding_logged_human"):
            print(f"      {DIM}human (logged): {wrap(it['preceding_logged_human'][:200])}{OFF}")
        print(f"      {col}{BOLD}{who}{OFF}  {col}{wrap(it['text'])}{OFF}")
        if it["media"] == "text":
            print(f"      {DIM}— no recording exists —{OFF}")
        print()
        if a.dry_run: continue
        t0 = time.time()
        if it["media"] == "audio" and play and os.path.exists(it.get("audio_abs", "")):
            try: subprocess.run(play + [it["audio_abs"]], check=False)
            except KeyboardInterrupt: print("\ninterrupted"); return
        else:
            try: time.sleep(dur)
            except KeyboardInterrupt: print("\ninterrupted"); return
        left = a.gap - max(0.0, time.time() - t0 - dur)
        if left > 0: time.sleep(min(left, a.gap))
    print(f"\n{BOLD}end{OFF}\n")

main()
