# BFF — Unitree Go2 Capture, Voice, & Post-Processing Suite

A data-acquisition, real-time telemetry, and conversational-AI system for the
Unitree Go2 quadruped. It captures synchronized video, audio, joint/body state,
and LiDAR over WebRTC; runs a local voice assistant (STT → LLM → TTS) with a
live web dashboard; and provides a post-processing toolchain that turns raw
sessions into clean, annotated, subtitled, multi-track media.

The suite runs on two robots — **snapper** and **helper** (both NVIDIA Jetson) —
and on a **macOS** workstation used for development and post-processing.

---

## Overview

The system has three layers:

1. **Capture** — [capture_go2_data.py](capture_go2_data.py) decodes the Go2's
   WebRTC feeds and writes synchronized `video.mp4`, `audio.wav`,
   `lowstate.jsonl` (IMU/battery/motor/`sport_state`), and `lidar.jsonl`,
   segmented into `chunk_N/` directories. YOLO object detection runs once in a
   background worker and is logged to `detections.jsonl`.
2. **Voice & dashboard** — [chat-manager.py](chat-manager.py) runs the
   voice-assistant loop (Whisper STT, Ollama LLM, Moondream/gemma VLM, Piper
   TTS) and launches [dashboard_server.py](dashboard_server.py), a Flask +
   Three.js dashboard streaming live video, telemetry, and a 3D LiDAR map. Chat
   transcript, VLM snapshots, and capture data all share one
   `session-YYYYMMDD-HHMMSS/` directory.
3. **Post-processing** — [fix_recordings.py](fix_recordings.py) and the
   [post_processing/](post_processing/) tools repair, concatenate, annotate,
   subtitle, and re-mux recorded sessions into finished assets. See
   [Helper Scripts](#helper-scripts).

### Recording modes

Capture defaults to **full recording** (`BFF_RECORD_BY_DEFAULT=true`): 5-minute
chunks, nothing pruned, so an entire session is retained. Set it to `false` for
the older **circular buffer** — 1-minute chunks with only the last six kept
(~6 minutes rolling), deleted live on the device. Either way the full chat
transcript is always preserved in `session.jsonl`, even when the video for a
turn has been pruned.

---

## Clean Install

The robots are Jetsons (aarch64 + CUDA); the workstation is macOS (Apple
Silicon / MPS). **We standardize on `venv`, not conda** — it's the only thing
that works cleanly on Jetson, where the GPU builds of `torch`/`onnxruntime`
come from NVIDIA's system wheels rather than PyPI or conda-forge, and it works
on the Mac too. (Conda still works fine on the Mac if you prefer it there, but
the robots have no conda.)

### 1. System prerequisites

| Component | macOS | Jetson (Ubuntu) |
| --- | --- | --- |
| ffmpeg / ffprobe | `brew install ffmpeg` | `sudo apt install ffmpeg` |
| Ollama (LLM server) | `brew install ollama` | [ollama.com/download/linux](https://ollama.com/download/linux) |
| PortAudio (mic/spkr) | `brew install portaudio` | `sudo apt install portaudio19-dev` |
| Python | 3.10+ | system 3.10 (with NVIDIA CUDA wheels) |

Pull the models Ollama serves (the LLM and the VLM):

```bash
ollama pull gemma4:e2b
```

### 2. Clone and create the environment

```bash
git clone https://github.com/roberttwomey/bff-code2.git
cd bff-code2
```

**macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt        # torch/onnxruntime resolve to MPS/CPU builds
```

**Jetson** — create the venv **with system site-packages** so it inherits
NVIDIA's CUDA-enabled `torch`/`torchvision`/`onnxruntime` (do **not** let pip
replace them with CPU wheels from PyPI):
```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print('cuda', torch.cuda.is_available())"   # expect: cuda True
```

> If `requirements.txt` is absent, the core Python deps are: `aiortc`,
> `opencv-python`, `flask`, `flask-socketio`, `python-dotenv`, `numpy`,
> `ultralytics`, `faster-whisper`, `piper-tts`, `ollama`, `sounddevice`,
> `unitree_sdk2py`. On Jetson, install `torch`/`torchvision`/`onnxruntime-gpu`
> from NVIDIA's wheels, not PyPI.

**Episodic memory (optional).** The cross-session recall database needs a few
extra packages on top of the above:
```bash
pip install -r memory/requirements.txt
python -m memory.db                    # expect: sqlite-vec <version> loaded OK
```
The subsystem is fail-soft in two independent stages — if `sqlite-vec` is
missing or this Python can't load SQLite extensions, memory degrades to a
no-op; if the embedder is unavailable, raw events still record without
vectors. Either way the voice loop runs unaffected, so skipping this step
costs recall, not stability.

### 3. Speech + vision models

- **Piper voices** (TTS): place `.onnx` voice files under `speech/piper/`
  (e.g. `en_GB-alan-medium.onnx`).
- **YOLO weights**: `yolo11n.pt` ships in the repo root; Ultralytics downloads
  others on first use.
- **Whisper**: `faster-whisper` downloads the chosen model
  (`BFF_WHISPER_MODEL`, e.g. `tiny.en`) on first run.
- **Memory embedder** (only if you installed `memory/requirements.txt`):
  `all-MiniLM-L6-v2` is pulled from Hugging Face on first use and cached on
  disk — the int8 `arm64` export on Jetson/Apple Silicon, fp32 elsewhere. It
  runs CPU-only on purpose, so it never competes with Ollama for the Jetson's
  shared unified memory.

> **Jetson caveat — pre-stage the embedding model, and don't let pip touch
> `numpy`.** Two things bite on snapper and helper specifically:
>
> **1. Warm the model cache before you go on site.** The Hugging Face fetch
> happens on the *first event the robot records*, and robot internet access is
> **conditional on where you are**: at home the housemachine network has an
> uplink and both Jetsons can reach the Hub, but on site that network is a
> local access point (GL.iNet Beryl AX3000) plus internal-net with **no route
> out**. A first run on site therefore logs `[Memory] Failed to embed ...` for
> the whole session — events still land in the database, but without vectors,
> so they are invisible to semantic recall until backfilled.
>
> While still on an uplinked network, warm each robot once:
> ```bash
> ssh cohab@snapper.local "cd code/bff-code2 && source venv/bin/activate && \
>   python -c 'from memory.embedder import embed; embed(\"warm the cache\")'"
> ```
> Adjust user, host, and path per robot (see [Fleet Notes](#fleet-notes)) — on
> helper that is `jesse@helper.local` and `code/bff-code2-main`. One-time per
> robot per account; `~/.cache/huggingface` persists across runs. Verify with
> `python -m memory.inspect`, which reports an `N/N embedded` count per event
> type.
>
> If you are already on site with a cold cache, copy it in from the Mac
> instead — the Mac is Apple Silicon and the Jetsons are aarch64, so both
> resolve to the same int8 `arm64` export and the ~22 MB cache is portable:
> ```bash
> tar -czf /tmp/hf-minilm.tgz -C ~/.cache/huggingface \
>   hub/models--sentence-transformers--all-MiniLM-L6-v2
> scp /tmp/hf-minilm.tgz cohab@snapper.local:/tmp/
> ssh cohab@snapper.local "mkdir -p ~/.cache/huggingface && \
>   tar -xzf /tmp/hf-minilm.tgz -C ~/.cache/huggingface"
> ```
>
> **2. Install with `numpy` pinned.** `onnxruntime` will happily pull `numpy`
> 2.x, which **breaks the Jetson `torch`/`cv2` builds**. Pin it on both robots:
> ```bash
> pip install -r memory/requirements.txt "numpy<2"
> python -c "import numpy, torch, cv2; print(numpy.__version__, torch.cuda.is_available())"
> ```

### 4. Configure `.env`

`.env` is **gitignored** (per-machine). Start from a deploy reference and edit:

```bash
cp deploy-reference/helper.env .env     # or snapper.env
```

Essential keys:

```env
UNITREE_ROBOT_IP=192.168.4.30
UNITREE_AES_KEY=your_32_hex_character_key_here
BFF_OUTPUT_DIR=captures
BFF_LOG_ROOT=~/bff/logs
BFF_RECORD_BY_DEFAULT=true              # full recording; false = circular buffer

# Voice assistant
BFF_OLLAMA_MODEL=gemma4:e2b
BFF_WHISPER_MODEL=tiny.en
BFF_PIPER_VOICE=speech/piper/en_GB-alan-medium.onnx
BFF_INPUT_DEVICE_KEYWORD="Wireless Mic Rx, DJI"
```

> **Jetson clock caveat:** Jetsons have no RTC and may boot with a wrong clock,
> which misdates session folders. `chat-manager.py` corrects this from the
> robot's own clock at startup — see [Known Issue: Jetson Clock Drift](#known-issue-jetson-clock-drift).

---

## Typical Usage

### Voice assistant (the usual entry point)

```bash
./run-chat-manager.sh
```

The wrapper activates the project venv and launches
[chat-manager.py](chat-manager.py), which — when the robot is reachable —
starts the dashboard itself and waits for the WebRTC feed to actually deliver
frames and telemetry (not just for Flask to answer). The Go2 often refuses the
first WebRTC offer after boot; rather than needing a manual relaunch, the
dashboard is restarted and retried:

| Variable | Default | Purpose |
| --- | --- | --- |
| `BFF_DASHBOARD_START_ATTEMPTS` | `3` | Times to (re)start the dashboard waiting for the feed |
| `BFF_DASHBOARD_HTTP_TIMEOUT` | `15` | Seconds to wait for the dashboard index per attempt |
| `BFF_DASHBOARD_STREAM_TIMEOUT` | `30` | Seconds to wait for camera/telemetry before restarting |

Streams disabled for the run (`BFF_CAPTURE_VIDEO=false`, etc.) are not waited
on; if every attempt fails, the session continues on the webcam fallback.

Direct invocation with overrides:
```bash
python3 chat-manager.py --require-wakeword
python3 chat-manager.py --piper-voice speech/piper/en_GB-alan-medium.onnx
```

**Scene prompts** (persona + trigger-phrase scenes) come from
`performance-script.json`, which is untracked (edited per machine/show). Start
from the template:
```bash
cp performance-script.example.json performance-script.json
```

### Dashboard only

```bash
python dashboard_server.py               # live: connect over WebRTC + record
python dashboard_server.py --simulate    # replay the newest captures/ session
```
Open `http://localhost:8080`. Toggle **YOLO DETECT** in the video panel to
pause/resume inference; use **RECORD** to switch recording mode at runtime.

### After a session

Recorded sessions live under `BFF_LOG_ROOT` (default `~/bff/logs`) as
`session-YYYYMMDD-HHMMSS/`. Turn them into finished media with
[fix_recordings.py](fix_recordings.py) — see below.

---

## Helper Scripts

### `fix_recordings.py` — session repair & media rendering

The main post-processing tool. It walks recorded sessions and produces a clean
set of derived outputs under a separate directory, **never modifying the
originals**. Runs on the macOS workstation against local or archived sessions.

```bash
# Most recent session under ~/bff/logs, all steps
./fix_recordings.py

# Specific sessions from an archive, into a chosen output root
./fix_recordings.py --logs-dir /Volumes/Cohab2024/bff-logs-HELPER \
                    --out-dir  /Volumes/Cohab2024/bff-logs-HELPER-processed \
                    --session session-20260722-125445

./fix_recordings.py --last 5            # five most recent sessions
./fix_recordings.py --dry-run           # report chunk health, write nothing
```

**Pipeline steps** (`--steps`, default runs all):

| Step | What it does |
| --- | --- |
| `repair` | Rebuilds the index (`moov` atom) of a truncated final chunk — a hard shutdown leaves OpenCV's `video.mp4` as `ftyp`+`mdat` with no index. Reconstructs it from a healthy sibling chunk (or a donor from another session). |
| `plate` | Concatenates all chunks into one clean `video_clean.mp4`, stream-copied (no re-encode). |
| `overlay` | Renders YOLO boxes over the clean plate → `video_overlay.mp4`. Handles the shared global `frame_index` counter, including circular-buffer sessions where early chunks were pruned. |
| `srt` | Subtitle tracks synced to the plate — see `--srt-tracks`. |
| `speech` | Gathers all turn wavs into `speech/`, relabeling the synthesized-speech headers (Piper writes 22050 Hz headers over 48000 Hz samples). |
| `audio` | Builds two audio tracks and muxes both into the videos: **track 1** onboard mic (per-chunk `audio.wav`), **track 2** dialogue (startup/input/response/wake/reset clips placed at their true times, level-matched to −20 dBFS). |
| `fulldialogue` | Whole-conversation `audio_dialogue_full.wav` + `dialogue_full.srt` on the session's own clock — recovers the *entire* conversation even when the circular buffer pruned most of the video. |

**Subtitle tracks** (`--srt-tracks`) combine four sources — `speech`, `vlm`,
`yolo`, `bodystate` — with `+` to stack them into one file. Default:
`speech,yolo+vlm,bodystate,speech+yolo+vlm+bodystate`.

**Key options:** `--overlay-seconds N` (quick preview render), `--dialogue-db`
(dialogue level target), `--no-normalize`, `--yolo-min-conf`, `--body-interval`.
Run `./fix_recordings.py --help` for the full list.

**Typical outputs** per session:
```yaml
video_clean.mp4              # concatenated plate (+ mic & dialogue audio tracks)
video_overlay.mp4            # plate with YOLO boxes (+ both audio tracks)
audio_mic.wav                # onboard mic, full timeline
audio_dialogue.wav           # dialogue synced to the plate window
audio_dialogue_full.wav      # whole conversation, session clock
speech.srt / yolo+vlm.srt / bodystate.srt / speech+yolo+vlm+bodystate.srt
dialogue_full.srt            # full-conversation transcript
speech/                      # every turn wav (relabeled / copied)
detections_combined.jsonl    # detections remapped onto the plate timeline
report.json                  # what was repaired / produced
```

### `post_processing/overlay_detections.py` — standalone YOLO overlay

Re-run the bounding-box overlay directly on a raw capture directory (a single
chunk, or a session with `chunk_*` subfolders):
```bash
python3 post_processing/overlay_detections.py captures/session-20260704-130112
```
Produces `video_annotated.mp4` with a persistence buffer that keeps boxes from
flickering on frames that had no detection. (`fix_recordings.py`'s `overlay`
step drives this over the clean plate; use this for a quick one-off.)

### `post_processing/view_lidar.py` — interactive 3D LiDAR viewer

Replay recorded LiDAR voxel clouds in a Three.js/WebGL scene and compile a
`lidar_render.mp4`:
```bash
python3 post_processing/view_lidar.py                                   # newest session
python3 post_processing/view_lidar.py captures/session-20260704-130112  # specific
python3 post_processing/view_lidar.py --step 10                         # more decimation
```
Controls: left-drag rotate, right-drag pan, scroll zoom, PLAY/PAUSE/PREV/NEXT,
and **ACCUMULATE MAP** to stitch all frames into the full traversed map.

### Clock helpers — `set_time.py` / `set_time_from_mac.sh`

Push a correct clock to a Jetson from a synchronized peer (the Mac). See
[Known Issue: Jetson Clock Drift](#known-issue-jetson-clock-drift) for how
`chat-manager.py` self-corrects at startup; these are for manual/boot-time use.

```bash
./set_time_from_mac.sh                 # set the local robot's clock from the Mac
```

### `run-chat-manager.sh` — venv launcher

Activates the project-local venv and execs `chat-manager.py`, passing through
any arguments. This is the intended way to start the assistant on the robots
(they have no conda).

---

## Output Capture File Tree

Captured directories are saved under `session-YYYYMMDD-HHMMSS/`. When launched
via `chat-manager.py`, chat transcript, VLM snapshots, and capture data share
this directory:

```yaml
session-YYYYMMDD-HHMMSS/
├── session.jsonl         # chat transcript, VLM queries, session events
├── startup.wav           # synthesized greeting
├── turn-NNN-input.wav    # recorded user speech (16 kHz)
├── turn-NNN-response.wav # synthesized reply (48 kHz samples; see note)
├── turn-NNN-wake.wav     # wake-word acknowledgements
├── turn-NNN-reset.wav    # reset chimes
├── vlm_captures/         # VLM snapshot frames + descriptions
└── chunk_N/
    ├── video.mp4         # segment video (mpeg4, 1280×720 @ 30 fps)
    ├── audio.wav         # segment audio (48 kHz stereo)
    ├── detections.jsonl  # YOLO detections (shared global frame_index)
    ├── lowstate.jsonl    # IMU, battery, motor temps, sport_state (velocity)
    └── lidar.jsonl       # 3D point-cloud snapshots
```

> **Note on `turn-NNN-response.wav`:** Piper writes a 22050 Hz header over
> samples that are actually 48000 Hz, so these play ~2.2× too slow until
> relabeled. `fix_recordings.py`'s `speech` step fixes this (header only —
> sample data is untouched).

---

## Known Issue: Jetson Clock Drift

The Jetsons have no working RTC battery (`timedatectl` reports an RTC of
`1970-01-01`), and on the robot they sit on a network with no reachable NTP
server — `systemd-timesyncd` runs but never synchronizes. Each boot restores a
stale timestamp and counts forward from there, so the clock can be hours or
days behind. Sessions started with no network come out named
`session-19691231-*` (epoch).

This matters because session directories are named `session-YYYYMMDD-HHMMSS`
from the local clock, and `lowstate.jsonl` timestamps come from `time.time()`
on whichever machine runs the capturer. A skewed clock misdates captures and
makes any freshness check meaningless.

Check for skew against a synchronized machine:
```bash
echo "skew: $(( $(date +%s) - $(ssh cohab@snapper.local date +%s) )) seconds"
```

### How chat-manager fixes it

At startup, if the local clock is clearly wrong (before 2025),
`chat-manager.py` reads the robot's own wall clock from the `stamp` on its
`sportmodestate` messages over DDS. The Go2 keeps real time across our reboots,
needs no internet, and is already on the wire before anything gets named.

It then tries `sudo -n date`, which fixes file mtimes and every other program
on the box. Without the sudoers drop-in below, that call fails immediately
rather than blocking on a password nobody is there to type, and an in-process
offset is applied instead — session folders, VLM snapshots and log timestamps
come out right; file mtimes do not.

Time is only ever corrected **forwards**. A machine that booted without a clock
is behind real time, never ahead, so a robot stamp older than what the machine
already believes means the robot is wrong rather than the machine. This is not
hypothetical: helper's Go2 reports 2023, and is correctly refused.

`BFF_CLOCK_SYNC=0` disables the whole mechanism; `BFF_DDS_INTERFACE` overrides
the interface (default `enP8p1s0`).

### Granting permission to set the clock

Install a sudoers drop-in granting exactly one command, substituting the
machine's user for `cohab`:

```
cohab ALL=(root) NOPASSWD: /usr/bin/date -u -s @*
```

```bash
ssh -t cohab@snapper.local "sudo install -m 440 -o root -g root /tmp/99-bff-clock /etc/sudoers.d/99-bff-clock && sudo visudo -c"
```

`visudo -c` validates every sudoers file afterwards, so a syntax error cannot
lock you out of sudo. Both Jetsons have this installed.

### When the robot's clock is also wrong

Then nothing automatic can help — helper is in this position. Its own clock and
its robot's are both untrustworthy, so it comes up misdated after every reboot.
With the drop-in installed, correcting it from a synchronized machine needs no
password:

```bash
ssh jesse@helper.local "sudo -n date -u -s @$(date -u +%s)"
```

The cleanest durable fix remains giving the machine a reachable NTP server —
one served on the LAN or a route out — after which `systemd-timesyncd` corrects
itself at every boot with no further intervention.

---

## Fleet Notes

- **snapper** (`cohab@snapper.local`, `/home/cohab/code/bff-code2`) and
  **helper** (`jesse@helper.local`, `/home/jesse/code/bff-code2-main`) both run
  the assistant from a project-local venv; the macOS workstation is used for
  development and post-processing.
- Session archives are copied to external storage for processing — e.g.
  `/Volumes/Cohab2024/bff-logs-SNAPPER` and `bff-logs-HELPER`, with
  `fix_recordings.py` output alongside in `*-processed` directories.
- `.env` and `performance-script.json` are gitignored and per-machine; the
  committed `deploy-reference/*.env` and `performance-script.example.json` are
  the starting points.
