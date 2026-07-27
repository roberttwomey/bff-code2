# BFF Development Roadmap

Where the project is headed, and why. This is a planning document — nothing
here is implemented on `main` unless it says so. See [README.md](README.md) for
what the system does today.

Each entry records the *reasoning*, not just the intent, so a future reader can
tell whether a direction is still worth taking or whether its premises have
expired.

---

## In flight

Work that exists on a branch and needs finishing or merging.

| Branch | State | Next step |
| --- | --- | --- |
| `add-replay-tool` | `replay.py` timeline scrubber; even with `main` | Merge |
| `feature/aec` | Speex AEC + text-confirmed barge-in; supersedes the old `feature/barge-in` | Consolidate with `feature/self-signal-rejection`, then merge |
| `feature/context-refactor` | Perception block logging, VLM scene-look | Merge |
| `feature/memory-rebased` | Episodic SQLite memory, ported onto current `main` | Review and merge |
| `feature/dashboard-uslam` | Live USLAM map/path/odom in the Three.js dashboard | Port (see below) |
| `feature/wakeword` | Untested stub | Revive or retire |

### Port `feature/dashboard-uslam` rather than merging it

That branch is 139 commits behind and local-only. Its substantive commit
(`42039fe`) subscribes to `rt/uslam/frontend/odom`,
`rt/uslam/localization/odom`, `rt/uslam/frontend/cloud_world_ds`, and
`rt/uslam/navigation/global_path`, and adds a **SLAM MODE** toggle that switches
the 3D view from body-frame to world-frame.

`main` has no USLAM code at all, so this is not superseded work — but note that
`main` has since grown its own pose path on `rt/utlidar/robot_pose` →
`slam_pose`, consumed by `chat-manager.py` as a velocity fallback. That is
*LiDAR SLAM pose*, a different source from USLAM frontend odom. A port should
decide deliberately which one the dashboard tracks rather than quietly running
both.

The second commit (`fcee0e1`, "quota exceeded") is a half-finished session:
its `rt/uslam/server_log` → dashboard log path works, but `send_uslam_command()`
and the `uslam_control_command` socket handler are **dead code — nothing in the
UI ever emits that event**. Port the first commit; either finish or drop the
second, but don't carry the dead half forward silently.

Despite ~2,000 lines of drift in the three affected files, a dry-run merge
produces only 8 conflict hunks, because the structural anchors (the
`add_listener` dispatch chain, the `pub_sub.subscribe` block, the socket.io
emit pattern) all survive on `main`.

**This branch has never been pushed.** Push it for safety regardless of whether
the port happens.

---

## Future direction: autonomous navigation

Assessed 2026-07-26 against the [dimos](https://github.com/dimensionalOS/dimos)
project (Dimensional Inc., Apache 2.0), whose navigation stack targets the
Unitree Go2 directly. Assessment is from their published docs
([overview](https://github.com/dimensionalOS/dimos/blob/main/docs/capabilities/navigation/index.md),
[deep dive](https://github.com/dimensionalOS/dimos/blob/main/docs/capabilities/navigation/deep_dive.md))
and dependency spec — **not** from reading their source.

### Why it fits

Their pipeline consumes the Go2's **preprocessed 5 cm voxel grid over WebRTC,
not raw point clouds** — which is exactly what `capture_go2_data.py` already
subscribes to (`rt/utlidar/voxel_map_compressed`) and decodes into
`points` / `resolution` / `origin`. We already hold the input their mapper
expects.

Their documented flow:

```
voxel grid → VoxelGridMapper → CostMapper → ReplanningAStarPlanner → MovementManager
             (column-carving)  (height gradient)  (A* + continuous replan)   (Twist)
```

Licensing and versions are compatible: Apache 2.0 (permissive, attribution),
Python 3.10–3.12 (snapper is 3.10), `numpy>=1.26.4` (satisfied by our
`numpy<2` pin at 1.26.4).

### Why it is not simply a dependency

1. **BFF cannot command motion.** `SportClient` appears in
   `behavioral-state-machine.py` and `shutdown.py`, but only for posture —
   `StandUp`, `StandDown`, `BalanceStand`. There is no velocity command
   anywhere in the repo. Navigation means introducing `Move(vx, vy, vyaw)` on a
   live quadruped, contending with the state machine's own MCF-mode handling
   (`ensure_mcf_mode` / `release_mcf_mode`). That is a step-change in risk, and
   it needs its own safety story (E-stop path, velocity clamp, deadman).
2. **Open3D on Jetson has no CUDA.** `VoxelGridMapper` is built on Open3D's
   `VoxelBlockGrid` and defaults to `device: CUDA:0` with 2,000,000 blocks. Per
   [Open3D's ARM docs](https://www.open3d.org/docs/release/arm.html),
   pip-installed Open3D on ARM64 contains no CUDA module — it must be compiled.
   dimos ships an aarch64 wheel (`open3d-unofficial-arm`), so ARM is
   considered, but on our Jetsons that path falls back to CPU at a size tuned
   for GPU.
3. **dimos's own Jetson extras are disabled** in their `pyproject.toml`
   (noted as 404ing). Jetson is not a supported path there today.
4. **Memory contention.** Moondream has already been pushed to CPU
   (~190 s/caption) once Whisper and TensorRT claimed unified memory. A
   2 M-block voxel grid is a serious new claimant on the same pool.

### What to take

**The algorithms and the tuned quadruped constants — not the dependency.**

`HeightCostConfig` encodes domain knowledge that is expensive to derive
independently: `can_pass_under 0.6 m`, `can_climb 0.15 m`,
`ignore_noise 0.05 m`, `smoothing 1.0`, with a costmap encoding of `0` flat /
`50` moderate slope / `100` impassable / `-1` unknown. Those are Go2-shaped
numbers. **Column-carving** — each new LiDAR frame wholly replaces its column
region, so the map cannot accumulate stale voxels — is likewise a good idea
that is roughly 30 lines against a voxel stream we already decode.

### Staged plan

1. **Height-gradient costmap, offline.** Reduce recorded `lidar.jsonl` to a 2D
   traversability grid. Pure numpy; no Open3D, no new heavy deps, no robot.
   Verifiable against archived sessions.
2. **Costmap in the dashboard.** Render it live alongside the existing voxel
   map. Still read-only — no actuation. Composes with the USLAM port above, and
   gives a "where could the robot go" overlay.
3. **Trust the map before driving from it.** Only once the costmap is
   believable does planning (A*, `WavefrontFrontierExplorer` wavefront-BFS
   frontier selection, `PatrollingModule` coverage/random/frontier strategies)
   become worth pulling.
4. **Motion command + safety.** A separate and larger decision than
   "incorporate some navigation." Do not begin here.

Relocalization against a pre-built loop-closed map (their
`unitree-go2-relocalization` blueprint, offline pose-graph optimization via
`gtsam-extended`) is a further stage, relevant only if we need repeatable runs
in a known space — e.g. a fixed installation or performance venue.
