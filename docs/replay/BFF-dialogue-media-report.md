# BFF — Dialogue & Media Report

Forty flagged exchanges from the BFF development and performance archive, with an
inventory of every recorded stream that survives for each one.

- **Generated:** 2026-08-01T06:46:47
- **Archive root:** `/Volumes/Cohab2024/BFF/SIGGRAPH 2026 BFF/logs-all`
- **Companion index:** [`bff-replay-index.json`](bff-replay-index.json) — same data, machine-readable, with per-turn timestamps, wav paths and video frame offsets. This report is generated from it; edit the index builder, not this file.

Selection is by user dialogue — moments where what the human said, or what the
machine's answer revealed, is the interesting part. Sessions are grouped by the
four development phases.

## Headline finding for replay

**Video, lidar and body telemetry exist only from July 2026 onward.** Everything
from the CMC keynote through the IDEAS performance is audio-only, and the CMC
material is transcript-only with no audio ever written to disk. That constrains
which exchanges can be replayed as multimedia and which have to be reconstructed.

Of 40 exchanges: **10** are fully instrumented (tier A), **23** are voice-only (tier B), **7** are text-only (tier C).

## Stream legend

| Stream | File | What it is |
|---|---|---|
| Human voice | `turn-NNN-input.wav` | The VAD segment Whisper transcribed — the actual human utterance |
| Synth voice | `turn-NNN-response.wav` | Piper TTS as it played from the dog's speaker |
| UI cue | `turn-NNN-{reset,wake,start/stop-listening,…}.wav` | System sounds, audible in the room and part of the performance |
| Startup | `startup.wav` | Boot greeting |
| VLM still | `vlm_captures/snapshot_*.jpg` | The frame the scene captioner actually looked at |
| VLM caption | `vlm_captures/description_*.txt` | Its text description — what got injected into the prompt |
| Video | `chunk_N/video.mp4` | Onboard fisheye camera, continuous, ~30fps |
| Chunk audio | `chunk_N/audio.wav` | Continuous room audio, not turn-segmented |
| Lidar | `chunk_N/lidar.jsonl` | Voxel point clouds, ~5cm resolution, `odom` frame |
| Lowstate | `chunk_N/lowstate.jsonl` | IMU rpy + 20 motor joints (angle, temperature) |
| Detections | `chunk_N/detections.jsonl` | YOLO boxes, classes, confidence, frame-indexed |
| Camera path | `camera_path.jsonl` | Virtual camera position/target for the 3D dashboard view |

## Phase 1 — CMC WIN keynote (Nov 2025)

The `chat_session_*.jsonl` format logged only the accumulating message array. No audio was written. **Nothing from this period is replayable as sound.**

| Exchange | Session | Hosts | Stored |
|---|---|---|---|
| apple / cinnamon / sensory hallucination | `chat_session_19691231_191711` | HELPER | transcript only |
| Claremont McKenna keynote / carbon cost | `chat_session_20251105_232211` | MAC, SNAPPER | transcript only |
| first 'concrete floor' | `chat_session_20251106_194243` | MAC, SNAPPER | transcript only |
| 'I love you' | `chat_session_20251106_145039` | MAC, SNAPPER | transcript only |
| specific apple / Honeycrisp | `chat_session_20251106_194552` | MAC, SNAPPER | transcript only |
| earliest memory | `chat_session_20251101_184352` | HELPER | transcript only |
| 'better if you had a memory' | `chat_session_20251106_100247` | MAC, SNAPPER | transcript only |

## Phase 2 — NeurIPS (Dec 2025)

Turn-segmented audio for both voices. No camera, no lidar, no body telemetry.

| Exchange | Session | Human wav | Synth wav | Empty | Cue | Human audio | Synth audio | Host |
|---|---|---|---|---|---|---|---|---|
| 'rehearsal for life alongside intelligent machines' | `session-20251203-114233` | 28 | 9 | – | – | 1.3m | 2.1m | HELPER |
| 1b poet / 'a bright blue crayon' | `session-20251207-154426` | 13 | 13 | – | – | 36s | 44s | MAC |
| programming paradox / 'resonance' | `session-20251210-133949` | 10 | 10 | – | – | 40s | 1.4m | MAC |
| 'Yes, be.' | `session-20251211-140606` | 7 | 7 | – | – | 18s | 36s | MAC |
| crisp apple / 'my friend Jesse' | `session-20251211-104816` | 45 | 37 / 43 | 6 | 2 | 1.5m | 3.2m | MAC |
| domestic capture — fox-like dog names | `session-20251207-081946` | 63 | 0 | – | 2 | 2.4m | – | MAC |
| domestic capture — breakfast / nature show | `session-20251207-072459` | 6 | 0 | – | – | 17s | – | MAC |
| quantum system 'Q' prompt | `session-20251219-100935` | 1 | 0 / 1 | 1 | – | 2s | – | MAC |

**Note.** `session-20251207-081946` and `session-20251207-072459` have human audio but **zero synthesized responses** — the dog was listening and not
answering. Those are recordings of a household with a silent machine in the room.

## Phase 3 — IDEAS performance (Jan 29 2026)

Still audio-only, but at full performance scale.

| Exchange | Session | Human wav | Synth wav | Empty | Cue | Human audio | Synth audio | Host |
|---|---|---|---|---|---|---|---|---|
| 'named you Helper' / distributed self | `session-20260126-221545` | 124 | 41 / 123 | 82 | 3 | 6.4m | 5.9m | MAC |
| 'you clanker' | `session-20260126-221552` | 60 | 44 / 60 | 16 | 7 | 2.5m | 1.3m | MAC |
| shutdown loop | `session-20260126-224024` | 12 | 9 / 11 | 2 | 2 | 24s | 21s | MAC |
| 'Actually, I'm a human' | `session-20260127-193910` | 69 | 52 / 69 | 17 | – | 3.1m | 3.3m | HELPER |
| 'I will only ask you to be a machine' / burp | `session-20260128-100651` | 155 | 51 / 52 | 1 | 11 | 8.5m | 17.0m | HELPER |
| Helper scripted self-intro | `session-20260128-101724` | 20 | 3 / 20 | 17 | 2 | 1.0m | 2.3m | HELPER |
| 'I don't want you to compliment me' | `session-20260128-105613` | 151 | 21 | – | 14 | 6.9m | 2.5m | HELPER |
| Colors scene / 'would it be pleasurable' | `session-20260129-115828` | 85 / 86 | 54 / 79 | 26 | 7 | 4.5m | 3.2m | MAC |
| confabulated hand-on-head memory | `session-20260129-120117` | 84 | 52 / 74 | 22 | 11 | 3.8m | 6.0m | MAC |
| 'why do you keep mentioning my family' | `session-20260129-140838` | 96 | 77 / 90 | 13 | 12 | 3.9m | 11.9m | MAC |
| black box in a black box | `session-20260129-142720` | 69 | 45 / 65 | 20 | 8 | 3.2m | 3.3m | MAC |
| performance + post-show Q&A | `session-20260129-164334` | 482 / 483 | 41 / 45 | 5 | 16 | 31.9m | 12.2m | HELPER |

**The Jan 29 performance is the largest audio asset in the archive** — `session-20260129-164334`,
483 human wav segments (482 playable) totalling 31.9m of human speech,
including the full post-show audience Q&A. It exists only as sound: no video, no lidar,
no body telemetry was recorded for IDEAS.

## Phase 4 — SIGGRAPH Spatial Storytelling (Jul 23 2026)

Fully instrumented: video, lidar, joint telemetry, YOLO detections and VLM stills alongside both voices, all on one wall clock.

| Exchange | Session | Tier | Human | Synth | Cue | VLM stills | Chunks | Video | Session size |
|---|---|---|---|---|---|---|---|---|---|
| Francis & Jasper meet Snapper | `session-20260201-190334` | B | 35 | 27 | 2 | – | – | – | 13MB |
| 'what was the feeling of being activated' | `session-20260701-125318` | B | 10 | 10 | 1 | – | – | – | 4MB |
| 'have you seen my red ball' | `session-20260718-195946` | B | 3 | 3 | 1 | 23 | 1 | – | 6MB |
| embodied hallucination / Ace Hotel | `session-19691231-160201` | A | 14 | 13 | 1 | 13 | 10 | 176MB | 343MB |
| naming denied (Jasper) | `session-20260720-194648` | A | 49 | 43 | 6 | 18 | 9 | 315MB | 3.8GB |
| 'I feel dead inside' | `session-20260720-195543` | A | 28 | 24 | 3 | 29 | 13 | 173MB | 366MB |
| apple as mirror | `session-20260720-204648` | A | 13 | 12 | 1 | 6 | 6 | 87MB | 385MB |
| Pinocchio / 'bad dog' | `session-20260721-121342` | A | 6 | 5 | – | 19 | 6 | 67MB | 1.4GB |
| Companion scene (Jesse) | `session-20260721-124325` | A | 29 | 28 | – | 89 | 6 | 72MB | 1.4GB |
| McCarthy / Lovelace | `session-20260722-114251` | A | 18 | 15 | 2 | 56 | 6 | 141MB | 242MB |
| Mirror / 'space between us' / interiority | `session-20260722-123623` | A | 39 | 38 | 2 | – | 6 | 87MB | 205MB |
| two dogs simultaneous | `session-20260722-123630` | A | 23 | 23 | 1 | 47 | 6 | 125MB | 242MB |
| SIGGRAPH stage / magic | `session-20260723-092744` | A | 70 | 64 | 8 | 127 | 16 | 1.5GB | 7.2GB |

**Note.** `session-20260722-123623` has video, lidar and body telemetry but **zero VLM stills** — the scene captioner
was off for that run. Relevant if you meant to show what it was seeing.

## Replay tiers

| Tier | What you can build | Exchanges |
|---|---|---|
| **A** | video + lidar + lowstate + detections + VLM stills + both voices | 10 |
| **B** | turn-segmented human and synthesized voice only | 23 |
| **C** | transcript text only, no audio ever written | 7 |

**Tier A** — embodied hallucination / Ace Hotel, naming denied (Jasper), 'I feel dead inside', apple as mirror, Pinocchio / 'bad dog', Companion scene (Jesse), McCarthy / Lovelace, Mirror / 'space between us' / interiority, two dogs simultaneous, SIGGRAPH stage / magic.

**Tier B** — 'rehearsal for life alongside intelligent machines', 1b poet / 'a bright blue crayon', programming paradox / 'resonance', 'Yes, be.', crisp apple / 'my friend Jesse', domestic capture — fox-like dog names, domestic capture — breakfast / nature show, quantum system 'Q' prompt, 'named you Helper' / distributed self, 'you clanker', shutdown loop, 'Actually, I'm a human', 'I will only ask you to be a machine' / burp, Helper scripted self-intro, 'I don't want you to compliment me', Colors scene / 'would it be pleasurable', confabulated hand-on-head memory, 'why do you keep mentioning my family', black box in a black box, performance + post-show Q&A, Francis & Jasper meet Snapper, 'what was the feeling of being activated', 'have you seen my red ball'.

**Tier C** — apple / cinnamon / sensory hallucination, Claremont McKenna keynote / carbon cost, first 'concrete floor', 'I love you', specific apple / Honeycrisp, earliest memory, 'better if you had a memory'.

## Unlogged audio

Input wavs that exist on disk with no corresponding record in `session.jsonl`.
These are utterances the machine heard and dropped — mostly the last thing said
before a session ended.

| Session | Turn | Duration | File |
|---|---|---|---|
| `session-20260126-224024` | 12 | 3.0s | `turn-012-input.wav` |
| `session-20260129-115828` | 53 | 5.8s | `turn-053-input.wav` |
| `session-20260129-115828` | 54 | 1.6s | `turn-054-input.wav` |
| `session-20260129-115828` | 70 | 4.0s | `turn-070-input.wav` |
| `session-20260129-115828` | 71 | 1.8s | `turn-071-input.wav` |
| `session-20260129-115828` | 85 | 2.4s | `turn-085-input.wav` |
| `session-20260129-115828` | 86 | Nones | `turn-086-input.wav` |
| `session-20260129-120117` | 8 | 5.0s | `turn-008-input.wav` |
| `session-20260129-120117` | 9 | 2.6s | `turn-009-input.wav` |
| `session-20260129-120117` | 10 | 2.0s | `turn-010-input.wav` |
| `session-20260129-120117` | 11 | 2.4s | `turn-011-input.wav` |
| `session-20260129-120117` | 12 | 2.0s | `turn-012-input.wav` |
| `session-20260129-120117` | 13 | 2.2s | `turn-013-input.wav` |
| `session-20260129-120117` | 56 | 2.2s | `turn-056-input.wav` |
| `session-20260129-120117` | 57 | 2.0s | `turn-057-input.wav` |
| `session-20260129-164334` | 483 | Nones | `turn-483-input.wav` |
| `session-19691231-160201` | 14 | 1.4s | `turn-014-input.wav` |
| `session-20260720-195543` | 28 | 1.8s | `turn-028-input.wav` |
| `session-20260721-121342` | 6 | 11.2s | `turn-006-input.wav` |
| `session-20260721-124325` | 29 | 2.6s | `turn-029-input.wav` |
| `session-20260722-114251` | 18 | 3.6s | `turn-018-input.wav` |
| `session-20260722-123623` | 39 | 1.2s | `turn-039-input.wav` |
| `session-20260723-092744` | 70 | 4.8s | `turn-070-input.wav` |

23 in total across the flagged sessions. Text was never written for these; the audio is the only record.

## Empty and truncated audio

**234 wav files across the flagged sessions are present but contain no audio** —
almost all are 44-byte WAV headers with zero frames, written when Piper produced no output.
A handful of input wavs are truncated mid-chunk and fail to open.

This matters: a raw file count overstates what is actually playable. The tables above show
`playable / total` wherever the two differ, and every affected turn in the index carries
`audio_playable: false`. The worst case is `session-20260126-221545`, where most response
wavs are empty headers.

| Session | Empty wavs | Playable synth | Total synth |
|---|---|---|---|
| `session-20260126-221545` | 82 | 41 | 123 |
| `session-20260129-115828` | 26 | 54 | 79 |
| `session-20260129-120117` | 22 | 52 | 74 |
| `session-20260129-142720` | 20 | 45 | 65 |
| `session-20260127-193910` | 17 | 52 | 69 |
| `session-20260128-101724` | 17 | 3 | 20 |
| `session-20260126-221552` | 16 | 44 | 60 |
| `session-20260129-140838` | 13 | 77 | 90 |
| `session-20251211-104816` | 6 | 37 | 43 |
| `session-20260201-190334` | 6 | 27 | 33 |

## Syncing notes

- **telemetry clock** — unix epoch float, field `timestamp`, in lowstate/lidar/detections/camera_path
- **transcript clock** — ISO 8601 local, field `timestamp`; `epoch` added here for convenience
- **warning lidar stamp** — lidar.jsonl also carries `stamp` on the robot DDS clock - do NOT use for sync
- **video frames** — detections.jsonl pairs frame_index with epoch; derived_fps per chunk lets you interpolate
- **vlm capture clock** — filename stamp is host local time, converted to epoch here
- Chunk boundaries are ~5 min; a single exchange can straddle chunks, so concatenate before seeking.
- `lidar.jsonl` is the storage hog — up to 187MB for one chunk. Decimate before shipping anything to a browser.

## Caveats

1. **Duplicates.** `bff-logs-siggraph-2026-dev/` and `bff-logs-organized/` re-copy many sessions
   from MAC/HELPER/SNAPPER. The index records every host under `hosts` and picks one as
   `primary_host`; all counts and paths refer to that copy only. Never sum across hosts.
2. **Clock.** `session-19691231-160201` (Ace Hotel) ran with the Jetson RTC unset. Its wavs and
   telemetry are internally consistent but its epoch timestamps will not align with anything
   else from that day. Hand-offset it if you replay it alongside another stream.
3. **Highlight ranges are keyword matches, not verified.** Each exchange carries a `highlight`
   block in the index giving the turn range where its defining phrase appears, with
   `confidence: keyword-match, unverified`. Three sessions produced no match and are marked
   as such. Listen before you cut.
4. **Transcripts are `tiny.en`.** Every `text` field in the index is Whisper tiny output,
   recorded live. Short utterances are unreliable. Re-transcribe with a larger model before
   quoting anything as text.

