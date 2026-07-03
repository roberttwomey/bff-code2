#!/usr/bin/env python3
"""
Unified real-time web dashboard server for Unitree Go2.
Streams camera feed (MJPEG), LiDAR 2D scans (WebSockets),
body telemetry data (WebSockets), and Chat-Manager logs (WebSockets).
"""

import os
import sys
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

# Optional YOLO Object Detection integration
YOLO_AVAILABLE = False
yolo_model = None
try:
    from ultralytics import YOLO
    # Try loading a local YOLO model if present
    yolo_model = YOLO("yolov8n.pt")
    YOLO_AVAILABLE = True
    print("[YOLO] ultralytics package loaded. Bounding boxes enabled.")
except ImportError:
    print("[YOLO] ultralytics/YOLO not installed. Stream will show raw frames.")

def annotate_frame(frame):
    """Run real-time object detection overlays on the frame if YOLO is available."""
    if YOLO_AVAILABLE and yolo_model is not None:
        try:
            results = yolo_model(frame, verbose=False)
            frame = results[0].plot()
        except Exception as e:
            # Silently fallback to raw frame on any inference error
            pass
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
            # Overlay detection annotations
            frame = annotate_frame(frame)
            
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
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    """HTTP streaming endpoint returning MJPEG multipart response."""
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('ping_latency')
def handle_ping():
    """Latency calculation handshake."""
    socketio.emit('pong_latency')

@socketio.on('lidar_recording_chunk')
def handle_lidar_recording_chunk(data):
    global capturer
    if capturer and hasattr(capturer, 'output_dir') and capturer.output_dir:
        filepath = os.path.join(capturer.output_dir, "lidar_render.webm")
        try:
            with open(filepath, "ab") as f:
                f.write(data)
        except Exception as e:
            print(f"[Dashboard Server] Failed to write LiDAR WebGL video chunk: {e}")

# Telemetry callbacks with rate-limiting
last_lowstate_time = 0
def on_lowstate_received(payload):
    """Relays body state telemetry JSON packet via WebSockets."""
    global last_lowstate_time
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

# Logs Watcher: Watches and tails JSONL logs written by chat-manager.py
def get_latest_log_file():
    """Scans and retrieves the latest chat-manager log file."""
    log_dir = Path(os.environ.get("BFF_LOG_ROOT", Path.home() / "bff" / "logs")).expanduser()
    if not log_dir.exists():
        return None
    log_files = list(log_dir.glob("*.jsonl"))
    if not log_files:
        return None
    # Sort files by modification date (newest first)
    log_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return log_files[0]

def tail_logs_worker():
    """Background worker that tails new log lines and relays them via WebSockets."""
    print("[Log Watcher] Logging tail worker started.")
    current_file = None
    file_handle = None

    while True:
        try:
            latest_file = get_latest_log_file()
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
    workspace_dir = Path(__file__).resolve().parent
    captures_dir = workspace_dir / "captures"
    captures_dir.mkdir(exist_ok=True)

    capturer = Go2DataCapturer(
        ip=ip,
        aes_key=aes_key,
        output_dir=str(captures_dir),
        video_fps=30,
        capture_video=not no_video,
        capture_audio=not no_audio,
        capture_lowstate=not no_lowstate,
        capture_lidar=not no_lidar
    )

    # Register listener callbacks
    if not no_video:
        capturer.add_listener('video', video_callback)
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

    # Start Go2 connection
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
    print(f"=======================================================\n")

    try:
        socketio.run(app, host='0.0.0.0', port=args.port, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutdown signals received. Stopping server.")
    finally:
        if capturer:
            print("Stopping robot capture streams and finalizing files...")
            capturer.stop_event.set()
            if capturer_thread:
                capturer_thread.join(timeout=5.0)
            print("Dashboard shutdown completed cleanly.")

if __name__ == "__main__":
    main()
