#!/usr/bin/env python3
import asyncio
import logging
import os
import sys
import time
import wave
import queue
import threading
import json
import argparse
from datetime import datetime
import numpy as np
import cv2
import dotenv

# Load .env file
dotenv.load_dotenv()

# Import SDK modules
try:
    from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
    from unitree_webrtc_connect.constants import RTC_TOPIC
    from aiortc import MediaStreamTrack
except ImportError as e:
    print(f"Error: Failed to import unitree_webrtc_connect packages. Make sure you are in the directory containing the package or it is installed: {e}")
    sys.exit(1)

# Optional YOLO Object Detection integration
YOLO_AVAILABLE = False
yolo_model = None
try:
    from ultralytics import YOLO
    import torch
    if os.path.exists("yolov8n.pt"):
        yolo_model = YOLO("yolov8n.pt")
        # Automatically move model to best GPU if available (MPS on macOS, CUDA on Linux)
        if torch.backends.mps.is_available():
            yolo_model.to("mps")
            # Warm up model to compile Metal shaders now instead of blocking later
            print("[YOLO] Warming up model on MPS...")
            dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
            yolo_model(dummy, verbose=False)
            print("[YOLO] Warmup complete.")
        elif torch.cuda.is_available():
            yolo_model.to("cuda")
            # Warm up model to initialize CUDA context now instead of blocking later
            print("[YOLO] Warming up model on CUDA...")
            dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
            yolo_model(dummy, verbose=False)
            print("[YOLO] Warmup complete.")
        YOLO_AVAILABLE = True
except Exception as e:
    print(f"[YOLO] Warning: failed to load or warm up YOLO model: {e}")

# Configure logging
logging.basicConfig(level=logging.FATAL)

# Suppress PyAV/libav noise (e.g. 'No start code is found')
try:
    import av
    av.logging.set_level(av.logging.ERROR)
except Exception:
    pass

class Go2DataCapturer:
    def __init__(self, ip, aes_key, output_dir, video_fps, capture_video, capture_audio, capture_lowstate, capture_lidar):
        self.ip = ip
        self.aes_key = aes_key
        self.output_root = output_dir
        self.video_fps = video_fps
        self.capture_video = capture_video
        self.capture_audio = capture_audio
        self.capture_lowstate = capture_lowstate
        self.capture_lidar = capture_lidar

        # Stats counters
        self.video_count = 0
        self.audio_frames = 0
        self.lowstate_count = 0
        self.lidar_count = 0

        # Throttle timestamps – limit lowstate/lidar to video_fps rate
        self._last_lowstate_time = 0.0
        self._last_lidar_time = 0.0

        # Queues and control
        self.video_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.lowstate_queue = queue.Queue()
        self.lidar_queue = queue.Queue()
        self.yolo_queue = queue.Queue(maxsize=1)   # capacity 1 to prevent backlog latency
        self.stop_event = threading.Event()
        self.threads = []

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.output_root, f"go2_capture_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Saving data to: {self.output_dir}")

        self.conn = None

        # Real-time listener callbacks
        self.video_listeners = []
        self.lowstate_listeners = []
        self.lidar_listeners = []
        self.detection_listeners = []   # fired by _yolo_worker with each inference result

        # YOLO enable flag — set to False to pause inference and overlay
        self.yolo_enabled = True
        self.start_time = None

    def add_listener(self, listener_type, callback):
        """Add a callback listener for real-time streaming.
        listener_type: 'video', 'lowstate', 'lidar', or 'detection'
        """
        if listener_type == 'video':
            self.video_listeners.append(callback)
        elif listener_type == 'lowstate':
            self.lowstate_listeners.append(callback)
        elif listener_type == 'lidar':
            self.lidar_listeners.append(callback)
        elif listener_type == 'detection':
            self.detection_listeners.append(callback)

    def start_writers(self):
        self.start_time = time.monotonic()
        if self.capture_video:
            t = threading.Thread(target=self._video_writer_worker, daemon=True, name="VideoWriter")
            t.start()
            self.threads.append(t)

        if self.capture_video and YOLO_AVAILABLE and yolo_model:
            t = threading.Thread(target=self._yolo_worker, daemon=True, name="YoloWorker")
            t.start()
            self.threads.append(t)

        if self.capture_audio:
            t = threading.Thread(target=self._audio_writer_worker, daemon=True, name="AudioWriter")
            t.start()
            self.threads.append(t)

        if self.capture_lowstate:
            t = threading.Thread(
                target=self._jsonl_writer_worker, 
                args=(self.lowstate_queue, os.path.join(self.output_dir, "lowstate.jsonl")), 
                daemon=True,
                name="LowStateWriter"
            )
            t.start()
            self.threads.append(t)

        if self.capture_lidar:
            t = threading.Thread(
                target=self._jsonl_writer_worker, 
                args=(self.lidar_queue, os.path.join(self.output_dir, "lidar.jsonl")), 
                daemon=True,
                name="LidarWriter"
            )
            t.start()
            self.threads.append(t)

    def stop_writers(self):
        self.stop_event.set()
        for t in self.threads:
            try:
                t.join()
            except BaseException as e:
                logging.error(f"Interrupted while joining thread {t.name}: {e}")
        print("\nAll background writers stopped and files closed successfully.")

    def _video_writer_worker(self):
        """Write video frames to disk at a steady real-time rate matching the unified clock.

        Driven by the unified self.start_time clock so that the total number of
        written frames strictly matches the elapsed session time. When no new
        frame arrives, the last received frame is repeated. If there is startup
        latency (e.g. WebRTC negotiation), the first received frame is repeated
        to fill the initial gap, ensuring the video duration is identical to the
        session duration (and approximately identical to the audio duration).
        """
        writer = None
        last_frame = None   # most recently received frame; repeated when queue is dry
        frame_idx = 0
        start_time = self.start_time if self.start_time is not None else time.monotonic()

        while not self.stop_event.is_set():
            now = time.monotonic()

            # Drain all pending frames; keep only the most recent one
            new_frame = None
            while True:
                try:
                    f = self.video_queue.get_nowait()
                    self.video_queue.task_done()
                    new_frame = f
                except queue.Empty:
                    break

            if new_frame is not None:
                last_frame = new_frame

            if last_frame is None:
                # No frame received yet — wait briefly before retrying
                time.sleep(0.005)
                continue

            # Initialise writer on the very first frame
            if writer is None:
                height, width = last_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_path = os.path.join(self.output_dir, "video.mp4")
                writer = cv2.VideoWriter(video_path, fourcc, self.video_fps, (width, height))
                print(f"\nInitialized VideoWriter: {video_path} ({width}x{height} @ {self.video_fps} FPS)")

            # Calculate expected frames based on unified session start time
            elapsed = now - start_time
            expected_frames = int(elapsed * self.video_fps)

            # Write missing frames to keep up with the elapsed time
            written_in_loop = 0
            while self.video_count < expected_frames and written_in_loop < 10:
                writer.write(last_frame)

                # Only queue YOLO for genuinely new frames, and only on the first write of this frame
                if new_frame is not None and written_in_loop == 0:
                    if YOLO_AVAILABLE and yolo_model and self.yolo_enabled:
                        try:
                            self.yolo_queue.put_nowait((frame_idx, last_frame))
                        except queue.Full:
                            pass

                self.video_count += 1
                frame_idx += 1
                written_in_loop += 1

            # Sleep a short duration to remain responsive and not peg CPU
            time.sleep(0.005)

        # Drain queue on shutdown (frames discarded; file is already at real time)
        while True:
            try:
                self.video_queue.get_nowait()
                self.video_queue.task_done()
            except queue.Empty:
                break

        if writer is not None:
            writer.release()
            print("VideoWriter released.")

    def _yolo_worker(self):
        """Run YOLO inference in a dedicated thread, independent of video writing.
        Reads (frame_idx, frame) pairs from yolo_queue and writes results to
        detections.jsonl without ever blocking the VideoWriter thread.
        """
        detections_path = os.path.join(self.output_dir, "detections.jsonl")
        try:
            detections_file = open(detections_path, "w", encoding="utf-8")
            print(f"Initialized Detections JSONL Logger: {detections_path}")
        except Exception as e:
            logging.error(f"Failed to open detections.jsonl: {e}")
            return

        while not self.stop_event.is_set() or not self.yolo_queue.empty():
            try:
                frame_idx, frame = self.yolo_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                results = yolo_model(frame, verbose=False)
                frame_detections = []
                for r in results:
                    for box in r.boxes:
                        xyxy = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = yolo_model.names[cls_id] if hasattr(yolo_model, 'names') else str(cls_id)
                        frame_detections.append({
                            "class": cls_name,
                            "confidence": conf,
                            "bbox": [round(val, 1) for val in xyxy]
                        })

                # Always fire detection listeners (even with empty list) so the
                # dashboard can clear stale boxes when nothing is detected
                detection_payload = {
                    "frame_index": frame_idx,
                    "timestamp": time.time(),
                    "detections": frame_detections
                }
                for cb in self.detection_listeners:
                    try:
                        cb(detection_payload)
                    except Exception as cb_err:
                        logging.error(f"Error in detection listener callback: {cb_err}")

                if frame_detections:
                    detections_file.write(json.dumps(detection_payload) + "\n")
                    detections_file.flush()
            except Exception as yolo_err:
                logging.error(f"YOLO inference error on frame {frame_idx}: {yolo_err}")
            finally:
                self.yolo_queue.task_done()

        detections_file.close()
        print("Detections JSONL file closed.")

    def _audio_writer_worker(self):
        wf = None
        while not self.stop_event.is_set() or not self.audio_queue.empty():
            try:
                audio_bytes = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if wf is None:
                audio_path = os.path.join(self.output_dir, "audio.wav")
                wf = wave.open(audio_path, 'wb')
                wf.setnchannels(2)
                wf.setsampwidth(2) # 16-bit PCM (2 bytes)
                wf.setframerate(48000)
                print(f"\nInitialized AudioWriter: {audio_path} (2 channels @ 48000 Hz)")

            wf.writeframes(audio_bytes)
            # 2 channels, 2 bytes per sample -> 4 bytes per stereo sample
            self.audio_frames += len(audio_bytes) // 4
            self.audio_queue.task_done()

        if wf is not None:
            wf.close()
            print("AudioWriter closed.")

    def _jsonl_writer_worker(self, data_queue, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            while not self.stop_event.is_set() or not data_queue.empty():
                try:
                    item = data_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                f.write(json.dumps(item) + "\n")
                f.flush()
                
                # Increment statistics based on the file name
                if "lowstate.jsonl" in file_path:
                    self.lowstate_count += 1
                elif "lidar.jsonl" in file_path:
                    self.lidar_count += 1
                    
                data_queue.task_done()

    async def run(self):
        # 1. Connect to the WebRTC connection
        self.conn = UnitreeWebRTCConnection(
            WebRTCConnectionMethod.LocalSTA, 
            ip=self.ip, 
            aes_128_key=self.aes_key
        )
        print(f"Connecting to Go2 at {self.ip}...")
        await self.conn.connect()
        print("Connected to Go2!")

        # Start writer threads
        self.start_writers()

        # 2. Setup video callback
        if self.capture_video:
            async def recv_camera_stream(track: MediaStreamTrack):
                while True:
                    try:
                        frame = await track.recv()
                        img = frame.to_ndarray(format="bgr24")
                        self.video_queue.put(img)
                        for cb in self.video_listeners:
                            try:
                                cb(img)
                            except Exception as cb_err:
                                logging.error(f"Error in video listener callback: {cb_err}")
                    except Exception as e:
                        if not self.stop_event.is_set():
                            logging.error(f"Error in video track receive: {e}")
                        break

            self.conn.video.switchVideoChannel(True)
            self.conn.video.add_track_callback(recv_camera_stream)
            print("Video stream enabled.")

        # 3. Setup audio callback
        if self.capture_audio:
            async def recv_audio_stream(frame):
                try:
                    audio_data = np.frombuffer(frame.to_ndarray(), dtype=np.int16)
                    self.audio_queue.put(audio_data.tobytes())
                except Exception as e:
                    logging.error(f"Error in audio track receive: {e}")

            self.conn.audio.switchAudioChannel(True)
            self.conn.audio.add_track_callback(recv_audio_stream)
            print("Audio stream enabled.")

        # 4. Setup lowstate callback
        if self.capture_lowstate:
            lowstate_interval = 1.0 / self.video_fps

            def lowstate_callback(message):
                try:
                    now = time.time()
                    if now - self._last_lowstate_time < lowstate_interval:
                        return  # throttle to video FPS rate
                    self._last_lowstate_time = now

                    current_message = message.get('data')
                    if current_message:
                        payload = {
                            "timestamp": now,
                            "data": current_message
                        }
                        self.lowstate_queue.put(payload)
                        for cb in self.lowstate_listeners:
                            try:
                                cb(payload)
                            except Exception as cb_err:
                                logging.error(f"Error in lowstate listener callback: {cb_err}")
                except Exception as e:
                    logging.error(f"Error in lowstate callback: {e}")

            self.conn.datachannel.pub_sub.subscribe(RTC_TOPIC['LOW_STATE'], lowstate_callback)
            print("Low-level body state subscription enabled.")

        # 5. Setup LiDAR callback
        if self.capture_lidar:
            # Disable traffic saving mode on the data channel
            await self.conn.datachannel.disableTrafficSaving(True)
            # Use native decoder for points coordinates
            self.conn.datachannel.set_decoder(decoder_type='native')
            # Turn LiDAR sensor on
            self.conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")

            lidar_interval = 1.0 / self.video_fps

            def lidar_callback(message):
                try:
                    now = time.time()
                    if now - self._last_lidar_time < lidar_interval:
                        return  # throttle to video FPS rate
                    self._last_lidar_time = now

                    data_field = message.get("data", {})
                    inner_data = data_field.get("data", {})
                    points = inner_data.get("points")
                    
                    if points is not None:
                        # Convert numpy array points to standard list structure
                        points_list = points.tolist() if hasattr(points, "tolist") else list(points)
                        payload = {
                            "timestamp": now,
                            "stamp": data_field.get("stamp"),
                            "frame_id": data_field.get("frame_id"),
                            "resolution": data_field.get("resolution"),
                            "origin": data_field.get("origin"),
                            "point_count": len(points_list),
                            "points": points_list
                        }
                        self.lidar_queue.put(payload)
                        for cb in self.lidar_listeners:
                            try:
                                cb(payload)
                            except Exception as cb_err:
                                logging.error(f"Error in lidar listener callback: {cb_err}")
                except Exception as e:
                    logging.error(f"Error in lidar callback: {e}")

            self.conn.datachannel.pub_sub.subscribe("rt/utlidar/voxel_map_compressed", lidar_callback)
            print("LiDAR snapshots subscription enabled.")

        print("\n=== Capturing Data (Press Ctrl+C to Stop) ===")
        try:
            while not self.stop_event.is_set():
                # Check if WebRTC connection is lost
                if self.conn and hasattr(self.conn, 'pc') and self.conn.pc:
                    if self.conn.pc.connectionState in ('closed', 'failed'):
                        print("\n[Capturer] WebRTC connection lost (state: closed/failed). Stopping capture...")
                        self.stop_event.set()
                        break

                audio_sec = self.audio_frames / 48000.0
                sys.stdout.write(
                    f"\rRecorded: Video={self.video_count} frames | "
                    f"Audio={audio_sec:.1f}s | "
                    f"LowState={self.lowstate_count} samples | "
                    f"LiDAR={self.lidar_count} snapshots"
                )
                sys.stdout.flush()
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            pass
        finally:
            print("\nShutting down stream capture...")
            try:
                # Turn off LiDAR sensor
                if self.capture_lidar and self.conn and self.conn.datachannel:
                    try:
                        self.conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "off")
                        print("Sent switch off command to LiDAR.")
                    except Exception as e:
                        logging.error(f"Failed to turn off LiDAR: {e}")
                
                # Disconnect WebRTC connection
                if self.conn:
                    try:
                        await self.conn.disconnect()
                    except BaseException as e:
                        logging.error(f"Disconnect interrupted: {e}")
            finally:
                # Stop background threads
                self.stop_writers()

def parse_args():
    parser = argparse.ArgumentParser(description="Go2 Multi-modal Data Capture Tool")
    parser.add_argument("--ip", type=str, default=None, help="Robot local IP address")
    parser.add_argument("--aes-key", type=str, default=None, help="16-byte AES key (32 hex characters) for authentication on newer firmware")
    parser.add_argument("--output-dir", type=str, default=None, help="Base directory to save captured data")
    parser.add_argument("--fps", type=int, default=None, help="Target frame rate for output video file")
    parser.add_argument("--no-video", action="store_true", default=None, help="Disable video stream capture")
    parser.add_argument("--no-audio", action="store_true", default=None, help="Disable audio stream capture")
    parser.add_argument("--no-lowstate", action="store_true", default=None, help="Disable lowstate data capture")
    parser.add_argument("--no-lidar", action="store_true", default=None, help="Disable LiDAR snapshots capture")
    args = parser.parse_args()

    # Fallback to env vars or default values
    if args.ip is None:
        args.ip = os.getenv("UNITREE_ROBOT_IP", "192.168.4.30")
    if args.aes_key is None:
        args.aes_key = os.getenv("UNITREE_AES_KEY")
    if args.output_dir is None:
        args.output_dir = os.getenv("BFF_OUTPUT_DIR", "captures")
    if args.fps is None:
        args.fps = int(os.getenv("BFF_VIDEO_FPS", "30"))

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

    return args

def main():
    args = parse_args()
    
    capturer = Go2DataCapturer(
        ip=args.ip,
        aes_key=args.aes_key,
        output_dir=args.output_dir,
        video_fps=args.fps,
        capture_video=not args.no_video,
        capture_audio=not args.no_audio,
        capture_lowstate=not args.no_lowstate,
        capture_lidar=not args.no_lidar
    )

    try:
        asyncio.run(capturer.run())
    except KeyboardInterrupt:
        print("\nCapture stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred during capture execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
