# BFF — Dialogue & Media Report

Curated exchanges from the BFF development and performance archive, with an
inventory of every recorded stream that survives for each one.

- **Generated:** 2026-08-01T12:32:53
- **Archive:** `/Volumes/Cohab2024/BFF/logs-all`
- **Derived:** `/Volumes/Cohab2024/BFF/processed`
- **Companion index:** [`bff-replay-index.json`](bff-replay-index.json) — per-turn timestamps,
  wav paths, video frame offsets. [`bff-sessions.json`](bff-sessions.json) catalogs all
  688 sessions at summary level. This report is generated from them.

## Read this first

`text` is re-transcribed with distil-large-v3 and attributed by file kind (response.wav = the dog, definitively). `logged_text` is the original live tiny.en text from session.jsonl, kept for comparison. Where they disagree, prefer `text`: ~1-2% of assistant records point at the wrong wav.

Of 41 curated exchanges: **10** fully instrumented (tier A), **24** voice-only (tier B), **7** text-only (tier C).

**Video, lidar and body telemetry exist only from July 2026 onward.** Everything from the
CMC keynote through IDEAS is audio-only, and CMC is transcript-only — that log format never
wrote audio.

## Machine, persona and voice are three different things

By July 2026 the **helper machine was running the SNAPPER persona prompt**, so the name in the
system prompt no longer tells the dogs apart. The Piper voice does: `aru` is snapper, `alan` is
helper. The index carries `group` (machine), `persona` (prompt) and `voice` separately.

## Sessions run long and overlap

A session id is a *start* timestamp. Sessions routinely run for hours, sit idle, and overlap
each other. Use the `blocks` field, which gives the actually-active spans separated by silence.
The clearest case is the Snapper IDEAS performance below: a 2.7-hour session whose show is an
8-minute block at the end.

## Phase 1 — CMC WIN keynote (Nov 2025)

| Exchange | Session | Persona |
|---|---|---|
| apple / cinnamon / sensory hallucination | `chat_session_19691231_191711` | SNAPPER |
| Claremont McKenna keynote / carbon cost | `chat_session_20251105_232211` | SNAPPER |
| first 'concrete floor' | `chat_session_20251106_194243` | SNAPPER |
| 'I love you' | `chat_session_20251106_145039` | SNAPPER |
| specific apple / Honeycrisp | `chat_session_20251106_194552` | SNAPPER |
| earliest memory | `chat_session_20251101_184352` | SNAPPER |
| 'better if you had a memory' | `chat_session_20251106_100247` | SNAPPER |

Transcript only — no audio was ever written for these.

## Phase 2 — NeurIPS (Dec 2025)

| Exchange | Session | Machine | Voice | Tier | In | Out | Empty | **Recovered** | Bleed | Audio | Video |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 'rehearsal for life alongside intelligent machines' | `session-20251203-114233` | helper | alan | B | 28 | 9 | – | **0** | – | 3.4m | – |
| 1b poet / 'a bright blue crayon' | `session-20251207-154426` | snapper | alan | B | 13 | 13 | – | **1** | 2 | 1.4m | – |
| programming paradox / 'resonance' | `session-20251210-133949` | snapper | aru | B | 10 | 10 | – | **4** | – | 2.1m | – |
| 'Yes, be.' | `session-20251211-140606` | mac | aru | B | 7 | 7 | – | **1** | – | 56s | – |
| crisp apple / 'my friend Jesse' | `session-20251211-104816` | snapper | aru | B | 45 | 43 | 6 | **11** | – | 4.8m | – |
| domestic capture - fox-like dog names | `session-20251207-081946` | snapper | aru | B | 63 | 0 | – | **3** | – | 2.4m | – |
| domestic capture - breakfast / nature show | `session-20251207-072459` | snapper | alan | B | 6 | 0 | – | **0** | – | 17s | – |
| quantum system 'Q' prompt | `session-20251219-100935` | mac | aru | B | 1 | 1 | 1 | **1** | – | 4s | – |

## Phase 3 — IDEAS performance (Jan 29 2026)

| Exchange | Session | Machine | Voice | Tier | In | Out | Empty | **Recovered** | Bleed | Audio | Video |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 'named you Helper' / distributed self | `session-20260126-221545` | snapper | aru | B | 124 | 123 | 82 | **31** | – | 12.4m | – |
| 'you clanker' | `session-20260126-221552` | snapper | aru | B | 60 | 60 | 16 | **29** | 13 | 4.1m | – |
| shutdown loop | `session-20260126-224024` | snapper | aru | B | 12 | 11 | 2 | **6** | 1 | 50s | – |
| 'Actually, I'm a human' | `session-20260127-193910` | helper | aru | B | 69 | 69 | 17 | **20** | 8 | 6.5m | – |
| 'I will only ask you to be a machine' / burp | `session-20260128-100651` | helper | aru | B | 155 | 52 | 1 | **25** | 3 | 25.8m | – |
| Helper scripted self-intro | `session-20260128-101724` | helper | aru | B | 20 | 20 | 17 | **3** | – | 3.4m | – |
| 'I don't want you to compliment me' | `session-20260128-105613` | helper | aru | B | 151 | 21 | – | **33** | 12 | 9.7m | – |
| Colors scene / 'would it be pleasurable' | `session-20260129-115828` | snapper | aru | B | 86 | 79 | 26 | **50** | 11 | 7.9m | – |
| confabulated hand-on-head memory | `session-20260129-120117` | snapper | aru | B | 84 | 74 | 22 | **33** | – | 10.0m | – |
| 'why do you keep mentioning my family' | `session-20260129-140838` | snapper | aru | B | 96 | 90 | 13 | **35** | 1 | 16.2m | – |
| black box in a black box | `session-20260129-142720` | snapper | aru | B | 69 | 65 | 20 | **15** | 1 | 6.7m | – |
| *** SNAPPER IDEAS performance (block 2 is the show) | `session-20260129-144803` | snapper | aru | B | 176 | 171 | 64 | **98** | 59 | 15.4m | – |
| *** HELPER IDEAS performance + post-show Q&A | `session-20260129-164334` | helper | aru | B | 483 | 45 | 5 | **40** | 6 | 44.5m | – |

## Phase 4 — SIGGRAPH Spatial Storytelling (Jul 23 2026)

| Exchange | Session | Machine | Voice | Tier | In | Out | Empty | **Recovered** | Bleed | Audio | Video |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Francis & Jasper meet Snapper | `session-20260201-190334` | snapper | aru | B | 35 | 33 | 6 | **25** | 1 | 5.5m | – |
| 'what was the feeling of being activated' | `session-20260701-125318` | mac | aru | B | 10 | 10 | – | **2** | – | 1.8m | – |
| 'have you seen my red ball' | `session-20260718-195946` | mac | aru | B | 3 | 3 | – | **2** | – | 44s | – |
| embodied hallucination / Ace Hotel | `session-19691231-160201` | snapper | aru | A | 14 | 13 | – | **12** | – | 16.3m | 176MB |
| naming denied (Jasper) | `session-20260720-194648` | snapper | aru | A | 49 | 43 | – | **15** | – | 30.7m | 315MB |
| 'I feel dead inside' | `session-20260720-195543` | snapper | aru | A | 28 | 24 | – | **17** | – | 21.8m | 173MB |
| apple as mirror | `session-20260720-204648` | snapper | aru | A | 13 | 12 | – | **6** | – | 19.8m | 87MB |
| Pinocchio / 'bad dog' | `session-20260721-121342` | helper | alan | A | 6 | 5 | – | **7** | – | 7.8m | 67MB |
| Companion scene (Jesse) | `session-20260721-124325` | helper | alan | A | 29 | 28 | – | **7** | – | 16.4m | 72MB |
| McCarthy / Lovelace | `session-20260722-114251` | snapper | aru | A | 18 | 15 | – | **9** | – | 10.4m | 141MB |
| Mirror / 'space between us' / interiority | `session-20260722-123623` | helper | alan | A | 39 | 38 | – | **9** | – | 20.7m | 87MB |
| two dogs simultaneous | `session-20260722-123630` | snapper | aru | A | 23 | 23 | – | **7** | – | 15.6m | 125MB |
| *** SIGGRAPH stage / magic | `session-20260723-092744` | snapper | aru | A | 70 | 64 | – | **25** | 1 | 95.2m | 1.5GB |

## The two IDEAS performances

Both dogs performed, simultaneously.

**`session-20260129-144803`** — machine `snapper`, voice `en_GB-aru-medium`, persona SNAPPER · *** SNAPPER IDEAS performance (block 2 is the show)

- Wall span `14:48:03` → `17:29:10` (2.69 h)
- Block 1: `14:48:10` → `14:56:19` — 188 wavs
- Block 2: `17:21:34` → `17:29:32` — 167 wavs
- 176 mic / 171 synth wavs, 98 recovered, 59 speaker bleed

**`session-20260129-164334`** — machine `helper`, voice `en_GB-aru-medium`, persona HELPER · *** HELPER IDEAS performance + post-show Q&A

- Wall span `16:43:34` → `18:20:31` (1.62 h)
- Block 1: `16:43:37` → `18:20:42` — 545 wavs
- 483 mic / 45 synth wavs, 40 recovered, 6 speaker bleed

## Exchanges with a processed media bundle

Session-level concatenations under `processed/` — far easier to drive a presentation from
than the raw per-chunk files.

| Exchange | Session | Clean video | Overlay | Subtitles |
|---|---|---|---|---|
| Companion scene (Jesse) | `session-20260721-124325` | 80MB | 144MB | `speech.srt`, `yolo+vlm.srt`, `bodystate.srt`, `speech+yolo+vlm+bodystate.srt` |
| McCarthy / Lovelace | `session-20260722-114251` | 149MB | 197MB | `yolo+vlm.srt`, `bodystate.srt`, `speech+yolo+vlm+bodystate.srt` |
| Mirror / 'space between us' / interiority | `session-20260722-123623` | 100MB | 162MB | `speech.srt`, `yolo+vlm.srt`, `bodystate.srt`, `speech+yolo+vlm+bodystate.srt` |
| two dogs simultaneous | `session-20260722-123630` | 133MB | 181MB | `yolo+vlm.srt`, `bodystate.srt`, `speech+yolo+vlm+bodystate.srt` |
| *** SIGGRAPH stage / magic | `session-20260723-092744` | 1.6GB | 2.6GB | `speech.srt`, `yolo+vlm.srt`, `bodystate.srt`, `speech+yolo+vlm+bodystate.srt` |

## Replay tiers

| Tier | What you can build | Count |
|---|---|---|
| **A** | video + lidar + lowstate + detections + VLM stills + both voices | 10 |
| **B** | turn-segmented human and synthesized voice only | 24 |
| **C** | transcript text only, no audio ever written | 7 |

## Archive at a glance

| Group | Sessions | SNAPPER | HELPER | Unattributed |
|---|---|---|---|---|
| `_review-unattributed-robot` | 62 | 40 | 0 | 22 |
| `_review-unknown` | 41 | 34 | 0 | 7 |
| `helper` | 121 | 28 | 85 | 8 |
| `mac` | 135 | 132 | 0 | 3 |
| `snapper` | 329 | 280 | 0 | 49 |
| **total** | **688** | | | |

## Syncing notes

- **telemetry clock** — unix epoch float, field `timestamp`, in lowstate/lidar/detections/camera_path
- **turn clock** — wav mtime, given as `epoch` and `iso` per turn
- **warning lidar stamp** — lidar.jsonl also carries `stamp` on the robot DDS clock - do NOT use for sync
- **video frames** — detections.jsonl pairs frame_index with epoch; derived_fps per chunk lets you interpolate
- **blocks** — sessions often run for hours with long idle gaps; `blocks` gives the actually-active spans
- `lidar.jsonl` is the storage hog — up to 187MB per chunk. Decimate before shipping to a browser.

## Caveats

1. **`_conflicts` holds partial duplicate copies**, flagged `is_variant` and excluded from counts.
   Their canonical copy lives in a machine folder.
2. **Clock.** `session-19691231-160201` (Ace Hotel) ran with the Jetson RTC unset. Internally
   consistent, but its epoch will not align with anything else from that day.
3. **Highlight ranges are keyword matches, not verified** — see `confidence` on each. Listen before cutting.
4. **`logged_text` is live `tiny.en`** and is often wrong or missing entirely; `text` is the
   re-transcription. Roughly 1–2% of assistant records point at the wrong wav.

