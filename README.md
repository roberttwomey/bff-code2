# BFF Unitree Go2 Data Capture & Visualization Suite

A robust data acquisition, real-time telemetry, and visualization system for the Unitree Go2 quadruped robot. The project stream-captures low-latency video, audio, joint-states, and LiDAR voxel maps over WebRTC, exposing a live dashboard and saving synchronized files for post-processing and analysis.

---

## Key Features

* **Voice Chat Assistant:** An interactive voice-based assistant ([chat-manager.py](file:///Volumes/Work/Projects/bff/code/bff-code2/chat-manager.py)) running locally on the robot:
  * **Speech-to-Text (STT):** Real-time voice segmenting and transcription via Whisper.
  * **Local LLM:** Conversation generation using Ollama (`gemma4:e2b`).
  * **Text-to-Speech (TTS):** High-quality voice synthesis via Piper.
  * **Bluetooth/PulseAudio Integration:** Automatic pairing and sound routing to a Bluetooth headset (e.g., Shokz OpenRun Pro 2).
* **Multi-modal WebRTC Capture:** Decodes and records synchronized high-fidelity feeds:
  * **Video:** `video.mp4` (H.264 decoded and saved at a steady target FPS, matching elapsed session duration).
  * **Audio:** `audio.wav` (2 channels, 16-bit PCM at 48000 Hz).
  * **LiDAR:** `lidar.jsonl` (raw 3D point cloud snapshots).
  * **LowState:** `lowstate.jsonl` (IMU, battery voltage, motor temperatures, joint positions, etc.).
* **Accelerated YOLOv8 Integration:** Real-time object detection using a decoupled worker thread:
  * **GPU Acceleration:** Auto-selects and initializes Apple Silicon GPU (**MPS**) or NVIDIA GPU (**CUDA**) on boot, executing in ~23ms.
  * **Unified Inference:** YOLO runs only once in the background capturer; detections are piped directly to the live dashboard MJPEG stream and logged to `detections.jsonl`.
  * **Interactive Toggle:** Enable/disable YOLO detections instantly via the web dashboard interface.
* **Live Camera Path Recording:** Tracks manual camera adjustments (rotation, target pan, zoom) on the dashboard's Three.js OrbitControls, saving them to `camera_path.jsonl` at 10 FPS.
* **Auto Post-Processing:** Automatically runs the overlay annotator to compile `video_annotated.mp4` with persistent, flicker-free bounding boxes immediately upon server shutdown (`Ctrl+C`).
* **Interactive 3D LiDAR Playback & Rendering:** Replay 3D voxel clouds inside your browser with the option to follow the exact camera paths you recorded during the session, and automatically compiles `lidar_render.mp4` upon execution.

---

## Installation & Setup

1. **Activate Environment:** Ensure you are using the correct `bff` Conda environment:
   ```bash
   conda activate bff
   ```

2. **Configuration (`.env`):** Create or edit the `.env` file in the root directory:
   ```env
   UNITREE_ROBOT_IP=192.168.4.30
   UNITREE_AES_KEY=your_32_hex_character_key_here
   BFF_OUTPUT_DIR=captures
   BFF_VIDEO_FPS=30
   
   # Voice Assistant Configuration
   BFF_OLLAMA_MODEL=gemma4:e2b
   BFF_WHISPER_MODEL=tiny
   BFF_PIPER_VOICE=speech/piper/en_GB-aru-medium.onnx
   BFF_INPUT_DEVICE_KEYWORD="OpenRun Pro 2 by Shokz"
   ```

---

## Usage Instructions

### 1. Launch the Dashboard Server
Start the main dashboard in live connection mode (which connects to the robot over WebRTC, streams telemetry/media, and starts recording):
```bash
python dashboard_server.py
```

Or launch it in **Simulation Mode** (which replays the most recent session from the `captures/` folder):
```bash
python dashboard_server.py --simulate
```

* **Access Dashboard:** Open your browser and navigate to `http://localhost:8080` (or the local network URL printed on boot).
* **Control YOLO:** Toggle the **YOLO DETECT** checkbox in the Video Feed panel to pause or resume inference.

### 2. Launch the Voice Chat Assistant
Run the voice-activated LLM chat loop using the wrapper script (which ensures correct environment/Python path configuration):
```bash
./run-chat-manager.sh
```

Or invoke the Python script directly with custom arguments:
```bash
# Require a wake word ("Snapper") to prompt
python3 chat-manager.py --require-wakeword

# Override the TTS voice model
python3 chat-manager.py --piper-voice speech/piper/en_GB-alan-medium.onnx
```

### 3. Interactive 3D LiDAR Viewer
Replay and explore recorded LiDAR voxel data in an interactive 3D scene (rotatable/pannable) using Three.js and WebGL:
```bash
# Auto-detects and plays the newest session folder in captures/
python3 post_processing/view_lidar.py

# Or run targeting a specific capture directory
python3 post_processing/view_lidar.py captures/session-20260704-130112

# Adjust decimation/downsampling rate (default is 5; increase for faster loads)
python3 post_processing/view_lidar.py --step 10
```
* **Viewer UI Controls:** 
  * Drag with Left Mouse Button to rotate.
  * Drag with Right Mouse Button to pan.
  * Scroll wheel to zoom.
  * **PLAY/PAUSE/PREV/NEXT** timeline controls.
  * **ACCUMULATE MAP** checkbox to stitch all frames together, recreating the 3D map traversed by the robot.

### 4. Manual Bounding Box Overlay
If you ever want to re-run the YOLO overlay post-processing on a raw capture directory:
```bash
python3 post_processing/overlay_detections.py captures/session-20260704-130112
```

---

## Output Capture File Tree
Captured directories are saved inside `captures/session-YYYYMMDD-HHMMSS/`. When launched via `chat-manager.py`, chat transcript, VLM snapshots, and AV/LiDAR capture data all share this same session directory. Contents include:
```yaml
.
├── audio.wav             # Recorded audio (48kHz stereo wave)
├── video.mp4             # Raw video (steady FPS matching audio length)
├── detections.jsonl      # YOLO detection coordinates & confidence logs
├── video_annotated.mp4   # Video file with bounding boxes drawn over it
├── lowstate.jsonl        # Robot joint states & body sensors telemetry
├── lidar.jsonl           # Raw 3D point cloud snapshots
├── camera_path.jsonl     # Recorded camera movements (positions and targets)
├── lidar_render.mp4      # Compiled 3D WebGL session video (generated by post_processing/view_lidar.py)
└── lidar_view.html       # Self-contained interactive 3D WebGL player
```
