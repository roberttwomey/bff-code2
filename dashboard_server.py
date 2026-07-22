#!/usr/bin/env python3
"""
Unified real-time web dashboard server for Unitree Go2.
Streams camera feed (MJPEG), LiDAR 2D scans (WebSockets),
body telemetry data (WebSockets), and Chat-Manager logs (WebSockets).
"""

import os
import sys
import socket
import time
import json
import queue
import argparse
import threading
import asyncio
from pathlib import Path
import cv2
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import dotenv

# Load .env file
dotenv.load_dotenv()

# Import Go2DataCapturer from local script
try:
    from capture_go2_data import Go2DataCapturer
except ImportError as e:
    print(f"Error: Failed to import Go2DataCapturer from capture_go2_data.py: {e}")
    sys.exit(1)

# Global video frame state
latest_frame = None
frame_lock = threading.Lock()
capturer = None

def video_callback(img):
    """Callback triggered whenever a new camera frame is decoded."""
    global latest_frame
    with frame_lock:
        latest_frame = img

# Live detection overlay state — updated by the capturer's _yolo_worker via listener
latest_detections = []
latest_detection_time = 0.0
detections_lock = threading.Lock()
yolo_enabled = True   # toggled via /toggle_yolo; also propagated to capturer.yolo_enabled
is_recording = False  # toggled via /toggle_record; also propagated to capturer.is_recording

def detections_callback(payload):
    """Receives detection results from _yolo_worker and stores them for the MJPEG overlay."""
    global latest_detections, latest_detection_time
    with detections_lock:
        dets = payload.get('detections', [])
        if dets:
            latest_detections = dets
            latest_detection_time = payload.get('timestamp', time.time())
        else:
            # Keep previous detections active; draw_detections will automatically
            # clear them after 0.5 seconds of silence (missed frames).
            pass

def draw_detections(frame):
    """Draw bounding boxes from the latest _yolo_worker results onto frame (in-place copy)."""
    global latest_detections, latest_detection_time
    with detections_lock:
        # Clear/ignore detections if they are older than 1.0 seconds (stale)
        # used to be 0.5 seconds but I edited by hand
        if time.time() - latest_detection_time > 1.0:
            latest_detections = []
        dets = list(latest_detections)
    if not dets or not yolo_enabled:
        return frame
    frame = frame.copy()
    for det in dets:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # display tag above
        # cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        # cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # display tag inside
        cv2.rectangle(frame, (x1, y1), (x1 + tw + 4, y1+th+6), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1+th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame

def generate_mjpeg():
    """Generator function that yields JPEG-encoded frames at a throttled rate."""
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                frame = None
            else:
                frame = latest_frame.copy()

        if frame is not None:
            # Overlay detection annotations from _yolo_worker results
            frame = draw_detections(frame)
            
            # Compress NumPy frame to JPEG
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            # Yield brief wait if no frame is received yet
            time.sleep(0.05)
        
        # Throttled delay to enforce maximum ~25 FPS to conserve local host bandwidth
        time.sleep(0.04)

# Initialize Flask & SocketIO
app = Flask(__name__, template_folder='templates')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

@app.route('/')
def index():
    """Render the dashboard front-page UI."""
    lidar_settings = None
    settings_file = Path(__file__).resolve().parent / 'lidar_settings.json'
    if settings_file.exists():
        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                lidar_settings = json.load(f)
        except Exception as e:
            print(f"[Dashboard Server] Error loading lidar_settings.json: {e}")
    # BFF_SPEAKER is the name chat-manager already labels its own turns with,
    # so the dashboard borrows it rather than inventing a second setting.
    # BFF_DEVICE_NAME is just the host shown in the header; it defaults to this
    # machine's own hostname, which is right on both Jetsons.
    robot_name = os.environ.get("BFF_SPEAKER", "SNAPPER").upper()
    device_name = os.environ.get("BFF_DEVICE_NAME") or f"{socket.gethostname()}.local"
    return render_template(
        'dashboard.html',
        server_lidar_settings=lidar_settings,
        robot_name=robot_name,
        device_name=device_name,
    )

@app.route('/video_feed')
def video_feed():
    """HTTP streaming endpoint returning MJPEG multipart response."""
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snapshot')
def snapshot():
    """Returns the latest captured frame as a JPEG image."""
    global latest_frame
    with frame_lock:
        if latest_frame is None:
            return "No frame available", 503
        ret, jpeg = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ret:
            return "Encoding error", 500
        return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route('/detections')
def detections():
    """Returns the latest YOLO detections as JSON."""
    global latest_detections, latest_detection_time
    from flask import jsonify
    with detections_lock:
        return jsonify({
            'detections': latest_detections,
            'timestamp': latest_detection_time
        })

@app.route('/lowstate')
def lowstate():
    """Returns the latest body-state telemetry payload as JSON."""
    from flask import jsonify
    with lowstate_lock:
        if latest_lowstate is None:
            return jsonify({}), 503
        return jsonify(latest_lowstate)


@app.route('/history')
def history():
    """Returns the conversation so far, so a page that opens (or reloads) mid
    session isn't blank. The tail worker only relays lines written while it is
    watching - everything before that lives here."""
    from flask import jsonify, request
    try:
        limit = int(request.args.get('limit', 200))
    except ValueError:
        limit = 200

    log_file = get_current_log_file()
    if not log_file or not log_file.exists():
        return jsonify({"entries": []})

    entries = []
    with log_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            # assistant_chunk lines are the progressive stream of a reply that
            # also lands as a final 'assistant' record - replaying both would
            # print every answer twice.
            if record.get("type") in ("user", "assistant"):
                entries.append(record)

    return jsonify({"entries": entries[-limit:]})


@app.route('/toggle_yolo', methods=['POST'])
def toggle_yolo():
    """Enable or disable YOLO inference and bounding-box overlays.
    Propagates to capturer.yolo_enabled so _yolo_worker pauses inference
    rather than just hiding the overlay.
    """
    global yolo_enabled, capturer
    from flask import request, jsonify
    data = request.get_json(silent=True) or {}
    if 'enabled' in data:
        yolo_enabled = bool(data['enabled'])
    else:
        yolo_enabled = not yolo_enabled
    # Propagate to the capturer so inference actually stops
    if capturer is not None:
        capturer.yolo_enabled = yolo_enabled
    print(f"[YOLO] Detection {'enabled' if yolo_enabled else 'disabled'}.")
    return jsonify({'yolo_enabled': yolo_enabled})

@app.route('/toggle_record', methods=['POST'])
def toggle_record():
    """Enable or disable continual 5-minute chunk recording.
    Propagates to capturer.is_recording.
    """
    global is_recording, capturer
    from flask import request, jsonify
    data = request.get_json(silent=True) or {}
    if 'enabled' in data:
        is_recording = bool(data['enabled'])
    else:
        is_recording = not is_recording
    # Propagate to the capturer
    if capturer is not None:
        capturer.set_recording(is_recording)
    print(f"[Record] Recording {'enabled' if is_recording else 'disabled'}.")
    return jsonify({'is_recording': is_recording})

@app.route('/get_record_status', methods=['GET'])
def get_record_status():
    """Returns the current recording status."""
    global is_recording
    from flask import jsonify
    return jsonify({'is_recording': is_recording})

@app.route('/audio_start', methods=['POST'])
def audio_start():
    """Relays audio start event to web clients via SocketIO."""
    from flask import request, jsonify
    data = request.get_json(silent=True) or {}
    socketio.emit('audio_start', data)
    return jsonify({'status': 'ok'})

@app.route('/audio_chunk', methods=['POST'])
def audio_chunk():
    """Relays PCM audio chunk payload to web clients via SocketIO."""
    from flask import request, jsonify
    data = request.get_json(silent=True) or {}
    socketio.emit('audio_chunk', data)
    return jsonify({'status': 'ok'})

@app.route('/audio_end', methods=['POST'])
def audio_end():
    """Relays audio end event to web clients via SocketIO."""
    from flask import request, jsonify
    data = request.get_json(silent=True) or {}
    socketio.emit('audio_end', data)
    return jsonify({'status': 'ok'})

@app.route('/audio_interrupt', methods=['POST'])
def audio_interrupt():
    """Relays audio interrupt event to web clients via SocketIO."""
    from flask import request, jsonify
    data = request.get_json(silent=True) or {}
    socketio.emit('audio_interrupt', data)
    return jsonify({'status': 'ok'})

@socketio.on('ping_latency')
def handle_ping():
    """Latency calculation handshake."""
    socketio.emit('pong_latency')

@socketio.on('camera_move')
def handle_camera_move(payload):
    """Logs Three.js orbit controls camera moves to camera_path.jsonl."""
    global capturer
    if capturer and hasattr(capturer, 'output_dir') and capturer.output_dir:
        filepath = os.path.join(capturer.output_dir, "camera_path.jsonl")
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                payload["timestamp"] = time.time()
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            print(f"[Dashboard Server] Failed to log camera move: {e}")

# Telemetry callbacks with rate-limiting
last_lowstate_time = 0
latest_lowstate = None
lowstate_lock = threading.Lock()

def on_lowstate_received(payload):
    """Caches body state telemetry for HTTP polling and relays it via WebSockets."""
    global last_lowstate_time, latest_lowstate
    with lowstate_lock:
        latest_lowstate = payload
    now = time.time()
    if now - last_lowstate_time < 0.1:  # Limit socket emissions to 10Hz
        return
    last_lowstate_time = now
    socketio.emit('telemetry_data', payload)

last_lidar_time = 0
def on_lidar_received(payload):
    """Downsamples and broadcasts LiDAR coordinates via WebSockets."""
    global last_lidar_time
    now = time.time()
    if now - last_lidar_time < 0.2:  # Limit map updates to 5Hz
        return
    last_lidar_time = now

    points = payload.get('points', [])
    max_points = 1500  # Downsample threshold to prevent interface lag
    if len(points) > max_points:
        step = len(points) // max_points
        points = points[::step]

    socketio.emit('lidar_data', {
        'points': points,
        'point_count': len(points)
    })

def get_session_root() -> Path:
    """Shared session root - same BFF_LOG_ROOT variable chat-manager.py uses, so
    chat/VLM history and AV/LiDAR captures always land under one directory."""
    default_root = Path(__file__).resolve().parent / "captures"
    return Path(os.environ.get("BFF_LOG_ROOT", default_root)).expanduser()

# Logs Watcher: Watches and tails JSONL logs written by chat-manager.py
def get_latest_log_file():
    """Scans and retrieves the latest chat-manager log file."""
    log_dir = get_session_root()
    if not log_dir.exists():
        return None
    # session.jsonl lives one level down, inside each session-{id}/ directory
    log_files = list(log_dir.glob("session-*/session.jsonl"))
    if not log_files:
        return None
    # Sort files by modification date (newest first)
    log_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return log_files[0]

def get_current_log_file():
    """The session.jsonl belonging to this dashboard's session.

    chat-manager.py sets BFF_SESSION_ID when it spawns us, which pins the right
    session even before its first line is written - picking by mtime instead
    would latch onto the previous session's log and stay there until the new
    one happens to be written. Standalone runs have no id and fall back to
    whichever session was written most recently."""
    session_id = os.environ.get("BFF_SESSION_ID")
    if session_id:
        pinned = get_session_root() / f"session-{session_id}" / "session.jsonl"
        return pinned if pinned.exists() else None
    return get_latest_log_file()

def tail_logs_worker():
    """Background worker that tails new log lines and relays them via WebSockets."""
    print("[Log Watcher] Logging tail worker started.")
    current_file = None
    file_handle = None

    while True:
        try:
            latest_file = get_current_log_file()
            if latest_file != current_file:
                if file_handle:
                    file_handle.close()
                current_file = latest_file
                if current_file:
                    print(f"[Log Watcher] Found newer log file: {current_file}")
                    file_handle = open(current_file, "r", encoding="utf-8")
                    # Seek to the end of the file on startup so we only relay active logs
                    file_handle.seek(0, 2)
                else:
                    file_handle = None

            if file_handle:
                line = file_handle.readline()
                if line:
                    try:
                        data = json.loads(line.strip())
                        socketio.emit('log_data', data)
                    except json.JSONDecodeError:
                        socketio.emit('log_data', {'raw': line.strip()})
                else:
                    time.sleep(0.1)
            else:
                time.sleep(1.0)
        except Exception as err:
            print(f"[Log Watcher] Exception encountered: {err}")
            time.sleep(1.0)

def start_capturer_async(ip, aes_key, no_video, no_audio, no_lowstate, no_lidar):
    """Sets up the WebRTC capturer loop in a separate daemon thread."""
    captures_dir = get_session_root()
    captures_dir.mkdir(parents=True, exist_ok=True)

    # BFF_SESSION_ID is set by chat-manager.py so this capture lands in the same
    # session-{id} directory as the chat/VLM history it was spawned alongside.
    # Standalone runs (no parent session) generate their own id, same tree shape.
    session_id = os.getenv("BFF_SESSION_ID") or time.strftime("%Y%m%d-%H%M%S")
    session_dir = captures_dir / f"session-{session_id}"

    capturer = Go2DataCapturer(
        ip=ip,
        aes_key=aes_key,
        output_dir=str(session_dir),
        video_fps=30,
        capture_video=not no_video,
        capture_audio=not no_audio,
        capture_lowstate=not no_lowstate,
        capture_lidar=not no_lidar
    )
    capturer.set_recording(is_recording)

    # Register listener callbacks
    if not no_video:
        capturer.add_listener('video', video_callback)
        capturer.add_listener('detection', detections_callback)
    if not no_lowstate:
        capturer.add_listener('lowstate', on_lowstate_received)
    if not no_lidar:
        capturer.add_listener('lidar', on_lidar_received)

    def run_connection_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(capturer.run())
        except Exception as e:
            print(f"[Capturer] Connection thread failure: {e}")

    capturer_thread = threading.Thread(target=run_connection_loop, daemon=True)
    capturer_thread.start()
    print("[Capturer] WebRTC connection thread spawned successfully.")
    return capturer, capturer_thread
def _session_has_replayable_video(session_dir: Path) -> bool:
    """True if this session has a video.mp4 to replay, either chunked
    (chunk_*/video.mp4) or in the legacy single-directory layout."""
    if any(session_dir.glob("chunk_*/video.mp4")):
        return True
    return (session_dir / "video.mp4").exists()


def find_latest_capture_session():
    """Scans the session root and returns the absolute path of the newest
    session that actually has video to replay. Skips sessions with no
    recorded video yet -- notably the current run's own session directory,
    which chat-manager.py --simulate creates (empty) before spawning this
    process, and which would otherwise always sort as "latest"."""
    captures_dir = get_session_root()
    if not captures_dir.exists():
        return None
    session_dirs = [d for d in captures_dir.glob("session-*") if d.is_dir()]
    if not session_dirs:
        return None
    session_dirs.sort(key=lambda p: p.name, reverse=True)
    for session_dir in session_dirs:
        if _session_has_replayable_video(session_dir):
            return str(session_dir)
    return None

def simulation_worker(session_dir, stop_event):
    """Background worker that reads recorded capture session logs and plays them back in real time."""
    print(f"\n[Simulation] Initializing real-time playback for session: {session_dir}")
    
    session_path = Path(session_dir)
    chunk_dirs = sorted([d for d in session_path.glob("chunk_*") if d.is_dir()], key=lambda d: int(d.name.split('_')[1]))
    
    if not chunk_dirs:
        chunk_dirs = [session_path]
        print("[Simulation] No chunk subdirectories found. Playing legacy single-directory session.")
    else:
        print(f"[Simulation] Found {len(chunk_dirs)} chunks: {[d.name for d in chunk_dirs]}")

    while not stop_event.is_set():
        for chunk_dir in chunk_dirs:
            if stop_event.is_set():
                break
                
            chunk_dir_str = str(chunk_dir)
            print(f"\n[Simulation] Playing chunk: {chunk_dir.name}")
            video_path = os.path.join(chunk_dir_str, "video.mp4")
            lowstate_path = os.path.join(chunk_dir_str, "lowstate.jsonl")
            lidar_path = os.path.join(chunk_dir_str, "lidar.jsonl")
            detections_path = os.path.join(chunk_dir_str, "detections.jsonl")

            # Load telemetry logs
            lowstate_data = []
            if os.path.exists(lowstate_path):
                with open(lowstate_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            lowstate_data.append(json.loads(line.strip()))
                        except Exception:
                            pass
            print(f"[Simulation] Loaded {len(lowstate_data)} telemetry logs.")

            # Load lidar scans
            lidar_data = []
            if os.path.exists(lidar_path):
                with open(lidar_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            lidar_data.append(json.loads(line.strip()))
                        except Exception:
                            pass
            print(f"[Simulation] Loaded {len(lidar_data)} LiDAR scans.")

            # Load YOLO detections
            detections_data = []
            if os.path.exists(detections_path):
                with open(detections_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            detections_data.append(json.loads(line.strip()))
                        except Exception:
                            pass
            print(f"[Simulation] Loaded {len(detections_data)} detection logs.")

            # Determine base log start time to sync streams
            timestamps = []
            if lowstate_data:
                timestamps.append(lowstate_data[0]["timestamp"])
            if lidar_data:
                timestamps.append(lidar_data[0]["timestamp"])
            if detections_data:
                timestamps.append(detections_data[0]["timestamp"])

            if not timestamps:
                print(f"[Simulation] Warning: No logged timestamps found in chunk {chunk_dir.name}. Skipping chunk.")
                continue

            start_log_time = min(timestamps)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[Simulation] Error: Unable to open video capture file: {video_path}")
                time.sleep(2.0)
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_delay = 1.0 / fps

            # Copy data queues for this run iteration
            lowstate_queue = list(lowstate_data)
            lidar_queue = list(lidar_data)
            detections_queue = list(detections_data)

            # Ensure correct chronological order
            lowstate_queue.sort(key=lambda x: x["timestamp"])
            lidar_queue.sort(key=lambda x: x["timestamp"])
            detections_queue.sort(key=lambda x: x["timestamp"])

            start_wall_time = time.monotonic()
            print(f"[Simulation] Playback loop active for chunk {chunk_dir.name}.")

            frame_idx = 0
            while cap.isOpened() and not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                # Update live frame buffer
                global latest_frame
                with frame_lock:
                    latest_frame = frame

                # Align log timeline with current wall time elapsed
                elapsed = time.monotonic() - start_wall_time
                current_log_time = start_log_time + elapsed

                # Dispatch telemetry packets
                while lowstate_queue and lowstate_queue[0]["timestamp"] <= current_log_time:
                    item = lowstate_queue.pop(0)
                    on_lowstate_received(item)

                # Dispatch LiDAR voxel scans
                while lidar_queue and lidar_queue[0]["timestamp"] <= current_log_time:
                    item = lidar_queue.pop(0)
                    on_lidar_received(item)

                # Dispatch YOLO detections (refresh timestamp to prevent caching eviction)
                while detections_queue and detections_queue[0]["timestamp"] <= current_log_time:
                    item = detections_queue.pop(0)
                    item["timestamp"] = time.time()
                    detections_callback(item)

                frame_idx += 1

                # Precision frame sleep delay
                expected_time = start_wall_time + (frame_idx * frame_delay)
                sleep_time = expected_time - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    time.sleep(0.001)

            cap.release()

        if not stop_event.is_set():
            print("[Simulation] Replay of all chunks complete. Looping simulation in 1.0s...")
            time.sleep(1.0)

def main():
    global capturer
    parser = argparse.ArgumentParser(description="BFF Go2 Dashboard Server")
    parser.add_argument("--ip", type=str, default=None, help="Go2 IP Address")
    parser.add_argument("--aes-key", type=str, default=None, help="Go2 WebRTC AES Key")
    parser.add_argument("--port", type=int, default=None, help="Dashboard port")
    parser.add_argument("--no-video", action="store_true", default=None, help="Disable camera stream")
    parser.add_argument("--no-audio", action="store_true", default=None, help="Disable audio capture and recording")
    parser.add_argument("--no-lowstate", action="store_true", default=None, help="Disable telemetry data")
    parser.add_argument("--no-lidar", action="store_true", default=None, help="Disable LiDAR mapping")
    parser.add_argument("--simulate", action="store_true", help="Load the latest capture session and play it back in realtime")
    args = parser.parse_args()

    # Fallback to env vars or default values
    if args.ip is None:
        args.ip = os.getenv("UNITREE_ROBOT_IP", "192.168.4.30")
    if args.aes_key is None:
        args.aes_key = os.getenv("UNITREE_AES_KEY")
    if args.port is None:
        args.port = int(os.getenv("BFF_DASHBOARD_PORT", "8080"))

    def get_env_bool(name, default_val):
        val = os.getenv(name)
        if val is None:
            return default_val
        return val.lower() in ("true", "1", "yes", "on")

    if args.no_video is None:
        args.no_video = not get_env_bool("BFF_CAPTURE_VIDEO", True)
    if args.no_audio is None:
        args.no_audio = not get_env_bool("BFF_CAPTURE_AUDIO", True)
    if args.no_lowstate is None:
        args.no_lowstate = not get_env_bool("BFF_CAPTURE_LOWSTATE", True)
    if args.no_lidar is None:
        args.no_lidar = not get_env_bool("BFF_CAPTURE_LIDAR", True)

    # Start Go2 connection or simulation
    if args.simulate:
        latest_session = find_latest_capture_session()
        if not latest_session:
            print("[Simulation] Error: No capture sessions found in captures/ folder.")
            sys.exit(1)
        print(f"[Simulation] Using latest capture session: {latest_session}")
        simulation_stop_event = threading.Event()
        sim_thread = threading.Thread(
            target=simulation_worker,
            args=(latest_session, simulation_stop_event),
            daemon=True
        )
        sim_thread.start()
        capturer_thread = None
    else:
        print(f"Connecting to Go2 client at {args.ip}...")
        capturer, capturer_thread = start_capturer_async(
            ip=args.ip,
            aes_key=args.aes_key,
            no_video=args.no_video,
            no_audio=args.no_audio,
            no_lowstate=args.no_lowstate,
            no_lidar=args.no_lidar
        )

    # Start logs tailing monitor
    tail_thread = threading.Thread(target=tail_logs_worker, daemon=True)
    tail_thread.start()

    print(f"\n=======================================================")
    print(f"BFF Go2 Dashboard serving at: http://localhost:{args.port}")
    if args.simulate:
        print(f"RUNNING IN SIMULATION MODE (PLAYBACK)")
    print(f"=======================================================\n")

    try:
        socketio.run(app, host='0.0.0.0', port=args.port, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nShutdown signals received. Stopping server.")
    finally:
        if args.simulate:
            print("Stopping simulation playback...")
            simulation_stop_event.set()
        else:
            if capturer:
                print("Stopping robot capture streams and finalizing files...")
                capturer.stop_event.set()
                if capturer_thread:
                    capturer_thread.join(timeout=5.0)
                print("Dashboard shutdown completed cleanly.")

                # Automatically run overlay_detections immediately on exit
                if hasattr(capturer, 'output_dir') and capturer.output_dir:
                    try:
                        from post_processing.overlay_detections import overlay_detections
                        print(f"\n[Dashboard Server] Post-processing: Overlaying YOLO detections on recorded video chunks...")
                        import glob
                        chunk_paths = sorted(glob.glob(os.path.join(capturer.output_dir, "chunk_*")))
                        if chunk_paths:
                            for chunk_path in chunk_paths:
                                print(f"[Dashboard Server] Overlaying detections on chunk: {os.path.basename(chunk_path)}")
                                overlay_detections(chunk_path)
                        else:
                            overlay_detections(capturer.output_dir)
                    except Exception as e:
                        print(f"[Dashboard Server] Failed to run overlay_detections: {e}")

if __name__ == "__main__":
    main()
