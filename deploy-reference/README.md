# Deployment reference configs

Snapshots of the `.env` and `performance-script.json` that each machine is
actually running. Both of those filenames are gitignored at the repo root -
deliberately, since they are per-machine - which also means a fresh clone
starts with neither, and a careless `git checkout` on a machine can leave you
reconstructing settings from memory. These copies are the record.

Nothing here is read at runtime. They are reference material: copy a file to
the repo root as `.env` or `performance-script.json` to restore a machine.

| file | machine |
|---|---|
| `snapper.env` | cohab@snapper.local - the Jetson on the robot |
| `snapper-performance-script.json` | snapper's scenes |
| `helper.env` | jesse@helper.local - the second Jetson |
| `helper-performance-script.json` | helper's scenes (Default/Companion/Pinocchio/Mirror) |

Captured 2026-07-22. They drift the moment someone edits a live `.env`, so
re-snapshot after changing one.

## What differs between the two, and why

- `BFF_SPEAKER` - `SNAPPER` vs `HELPER`. Names the robot in the dashboard
  title, the device line and the chat labels.
- `BFF_WAKE_PHRASES` - each machine answers to its own name. Snapper's list
  carries Whisper misrecognitions collected from real transcripts ("snap or",
  "snabber"); helper's has none yet.
- `BFF_OLLAMA_MODEL` - both `gemma4:e2b`. Do not copy the `-mlx` variant from
  a Mac; MLX is Apple Silicon only and does not exist on the Jetsons.
- `BFF_PIPER_VOICE` - different voices, and snapper's is an absolute path
  under `/home/cohab`.
- `UNITREE_ROBOT_IP` - `192.168.123.161` on both, the robot's own SOC over the
  direct ethernet link. Reaching a robot over wifi instead costs 3.5ms RTT
  against 0.21ms, roughly 100x the jitter, and stops working entirely once the
  dog is out of range of the house AP.

Speech detection (`BFF_ACTIVATION_THRESHOLD`, `BFF_SILENCE_*`,
`BFF_MIN_PHRASE_SECONDS`, `BFF_BLOCK_DURATION`) is deliberately identical on
both machines.
