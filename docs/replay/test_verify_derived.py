#!/usr/bin/env python3
"""Self-test for verify_derived.py.

Builds a throwaway archive containing one instance of each fault the verifier
exists to catch, runs it, and asserts it fails on exactly those checks. A
verifier that has only ever been run against healthy data is untested.

    python test_verify_derived.py
"""
import os, sys, json, wave, struct, tempfile, subprocess, shutil

HERE = os.path.dirname(os.path.abspath(__file__))

def wav(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    w = wave.open(p, "w"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(struct.pack("<h", 0) * 1600); w.close()

def bundle(D, group, sid, infile_group=None):
    p = os.path.join(D, "retranscribed", group, sid); os.makedirs(p, exist_ok=True)
    json.dump({"session_id": sid, "group": infile_group or group, "model": "x",
               "blocks": [], "wavs": []}, open(os.path.join(p, "retranscription.json"), "w"))

def clog(D, group, sid, turns, audio):
    p = os.path.join(D, "complete-logs", group, sid); os.makedirs(p, exist_ok=True)
    with open(os.path.join(p, "session-complete.jsonl"), "w") as fh:
        fh.write(json.dumps({"type": "session_start", "session_id": sid, "group": group}) + "\n")
        for i, t in enumerate(turns):
            fh.write(json.dumps({"type": "user", "turn": t, "audio_path": audio[i]}) + "\n")

def build(T):
    R, D = os.path.join(T, "raw"), os.path.join(T, "derived")
    wav(f"{R}/snapper/session-20260101-000001/turn-001-input.wav")
    wav(f"{R}/mac/session-20260101-000002/turn-001-input.wav")
    wav(f"{R}/snapper/session-20260101-000003/turn-001-input.wav")   # will have no bundle
    bundle(D, "snapper", "session-20260101-000001", infile_group="mac")  # in-file mismatch
    bundle(D, "helper",  "session-20260101-000002")                     # wrong directory
    bundle(D, "snapper", "session-20260101-000004")                     # orphan
    clog(D, "snapper", "session-20260101-000001", [1, 2],
         ["snapper/session-20260101-000001/turn-001-input.wav",
          "snapper/session-20260101-000001/NOPE.wav"])                  # unresolvable audio
    clog(D, "snapper", "session-20260101-000002", [1, 3],
         ["mac/session-20260101-000002/turn-001-input.wav"] * 2)        # wrong group + turn gap
    return R, D

EXPECT_FAIL = [
    "retranscription bundle is in the group its session lives in",
    "retranscription in-file `group` matches its directory",
    "no orphaned retranscription bundles",
    "every session with wavs has a re-transcription bundle",
    "complete-log is in the group its session lives in",
    "complete-log turns are monotonic 1..n",
    "complete-log audio_path resolves",
]

def main():
    T = tempfile.mkdtemp(prefix="bff-verify-selftest-")
    try:
        R, D = build(T)
        # copy the verifier in so its index checks look at an empty dir, not the real ones
        shutil.copy(os.path.join(HERE, "verify_derived.py"), T)
        env = dict(os.environ, BFF_ARCHIVE_ROOT=R, BFF_DERIVED_ROOT=D)
        r = subprocess.run([sys.executable, os.path.join(T, "verify_derived.py")],
                           capture_output=True, text=True, env=env)
        out = r.stdout
        missed = [c for c in EXPECT_FAIL if f"FAIL  {c}" not in out]
        if r.returncode == 0:
            print("FAILED: verifier exited 0 on a broken tree"); print(out); sys.exit(1)
        if missed:
            print("FAILED: verifier did not flag:"); [print("   ", m) for m in missed]
            print(out); sys.exit(1)
        print(f"ok — verifier caught all {len(EXPECT_FAIL)} seeded faults and exited {r.returncode}")
    finally:
        shutil.rmtree(T, ignore_errors=True)

main()
