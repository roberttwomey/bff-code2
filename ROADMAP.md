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

## Long-term arc: memory, persona, agency

The through-line is moving the dog from a reactive assistant to something with
continuity between walks, lasting preferences, and the capacity to speak first.
Three phases, each usable on its own — none requires the next to be worth
building.

```
Phase 1  Consolidation & rumination   ("sleep and dream")
   ↓
Phase 2  Preferences & affective state ("internal life")
   ↓
Phase 3  Proactive agency              ("curiosity")
```

Where we are: live interaction and CV with the DJI mic; multimodal chat
(image + body state in-context); and episodic SQLite memory with semantic
indexing on `feature/memory-rebased`.

### Two constraints that shape every phase

**Unified memory is the scarce resource, not disk or CPU.** Ollama, Whisper,
and TensorRT already contend on the Jetsons — Moondream has been pushed to CPU
at ~190 s/caption when it lost. Every background cognitive process here runs a
*local LLM*, which is far heavier than the VLM captions we already schedule
around. `chat-manager.py` gates the VLM worker on the `is_dialogue_active`
global for exactly this reason; consolidation and rumination must use the same
gate and take a stricter one (idle *and* not mid-turn). A dog that thinks
beautifully but answers slowly is a regression.

**Unprompted speech re-opens the hardest solved problem in the project.**
Self-echo, barge-in gating, and AEC (`feature/aec`) were expensive to get
right, and they assume the robot speaks in response to a turn. Phase 3 speech
originates while the mic is live and no turn is in flight. Budget for that
explicitly rather than discovering it in the field.

### Phase 1 — Consolidation & rumination

**Do not design a new schema.** `memory/schema.sql` already anticipates this
work: an `episodes` table (`ts_start`, `ts_end`, `summary`, `event_count`), an
`episode_embeddings` vec0 table, `events.consolidated_into` (NULL until rolled
up), a partial index `idx_events_unconsolidated` for finding exactly the rows
that need processing, and `episode_summary` in the `events.type` CHECK
constraint. Phase 1 is a *writer against a schema that already exists* — the
one thing not to do is add a parallel `long_term_memories` table beside it.

**Do not write a cron.** `behavioral-state-machine.py` already has
`RobotState.DREAMING_MODE`, entered by `check_idle_state()` after
`BFF_IDLE_TIMEOUT` (default 45 s) and left by `check_wake_from_dreaming()` on
controller or UWB activity — with a cyan VUI colour already assigned to it. The
sleep-and-dream loop has a home; consolidation should be what DREAMING *does*,
and must yield immediately when the dog wakes.

> **Prompt 1.1 — consolidation pass**
>
> "Add `memory/consolidate.py`. It should select unconsolidated events via the
> existing `idx_events_unconsolidated` partial index, batch them by session and
> time window, and summarize each batch with `gemma4:e2b` through Ollama into
> the **existing** `episodes` table — then set `events.consolidated_into` and
> embed the summary into `episode_embeddings` using `memory/embedder.py`.
> Read `memory/schema.sql` first and use the tables that are already there; do
> not add new ones. Follow the fail-soft pattern in `memory/writer.py`: never
> raise into the caller. Make it interruptible — it runs during idle and must
> abandon the current batch promptly when the dog wakes. Include a dry-run flag
> that prints what it would consolidate without writing, and show me its output
> on a real archived session before we run it live."

> **Prompt 1.2 — rumination**
>
> "Add `memory/ruminate.py`. When invoked, sample 2–3 semantically *distant*
> episodes (use `episode_embeddings` — pick low cosine similarity deliberately,
> the point is unlikely juxtaposition), and prompt `gemma4:e2b` to write a short
> speculative reflection bridging them. Record it as an `episode_summary` event
> so it becomes recallable like any other memory, and also write a dated
> Markdown file under `ruminations/`. Keep the model call under a configurable
> token ceiling. Do not let this run while `is_dialogue_active` is true."

> **Prompt 1.3 — wire into DREAMING**
>
> "Read `behavioral-state-machine.py`, then have `DREAMING_MODE` drive
> consolidation and rumination. Respect the existing transitions — the state
> machine controls physical posture and must keep doing so — and make sure
> `check_wake_from_dreaming()` still wakes promptly with cognitive work in
> flight. Explain the concurrency story before you write code: which process
> owns the model, and what happens if a turn starts mid-consolidation."

### Phase 2 — Preferences & affective state

A low-dimensional state vector (curiosity, fatigue, attachment, topic
affinities) that decays and updates from experience, injected into the system
prompt each turn.

The honest risk here is **feedback collapse**: a dog that prefers what it has
already seen retrieves more of it, reinforcing the preference until it becomes
monotonous. Build the decay and a novelty floor at the same time as the
preference, not after.

> **Prompt 2.1 — persona state**
>
> "Add `persona/state.py` maintaining a small affective state vector persisted
> to `dog_state.json`: `curiosity`, `fatigue`, `attachment`, plus a topic-affinity
> map. Update it from session signals — telemetry already gives us battery,
> motion, and posture via the body-state summary in `chat-manager.py`, and
> conversation gives us topics. Include **time-based decay** so no value latches,
> and a `get_system_prompt_extension()` returning short natural-language
> directives. Keep the output well under the history truncation limit
> (`BFF_HISTORY_TRUNCATION_LIMIT`, default 11) — this competes with real
> conversation for context, so show me the token cost."

> **Prompt 2.2 — preference-weighted recall**
>
> "Add hybrid retrieval to the memory module: vector distance from
> `event_embeddings`/`episode_embeddings` combined with keyword matching,
> re-weighted by the topic affinities in `dog_state.json`. Include an explicit
> novelty term so high-affinity topics cannot crowd the context window —
> I want to see the weighting exposed as tunable constants, and a small
> evaluation over an archived session showing what changes versus pure vector
> search."

### Phase 3 — Proactive agency

The dog initiates: an unprompted observation after silence, a question about
something visually novel, or a move toward something interesting.

> **Prompt 3.1 — initiation loop**
>
> "Add `proactive.py`, an async monitor that proposes unprompted utterances
> when silence duration, visual-novelty score, and the `curiosity` state value
> cross thresholds. Reuse the VLM worker's existing frame-difference change
> detection (`BFF_VLM_CHANGE_THRESHOLD`) for novelty rather than adding a second
> comparator. Generate a short observation (under 4 sentences) drawn from recent
> ruminations or the current scene.
>
> Before writing the audio path, read how `feature/aec` handles barge-in and
> self-echo and tell me what breaks when TTS starts with no user turn in
> flight — unprompted speech while the mic is live is the risk here, not the
> generation. Put it behind an env flag, default off."

> **Prompt 3.2 — physical gestures as tools**
>
> "Expose a small tool surface to the model in `robot_tools.py`:
> `orient_towards_human()`, `curious_pause()`, `speak_unprompted(text)`. Bind
> them to the **posture-level** `SportClient` calls already used in
> `behavioral-state-machine.py` (`StandUp`, `StandDown`, `BalanceStand`) — do
> **not** introduce velocity commands in this step. Every tool call must be
> logged to `session.jsonl` and recorded to memory, and must refuse while the
> behavioral state machine holds a conflicting state. Explain how this
> coordinates with the state machine's MCF-mode handling before implementing."

**Exploration is gated on navigation.** "Move toward something interesting"
needs a velocity command and a costmap the dog can trust — see the navigation
section below, which is the physical substrate for this part of Phase 3 and
carries its own safety prerequisites. Phases 1 and 2, and the speech half of
Phase 3, need none of that and can proceed independently.

*This arc was drafted with Google Gemini and adapted here against the actual
codebase — the schema, state machine, and contention notes above are the main
departures from that draft.*

---

## Future direction: autonomous navigation

Also the physical substrate for the exploration half of Phase 3 above: the dog
cannot move toward what interests it without a velocity command and a costmap
it can trust.

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
