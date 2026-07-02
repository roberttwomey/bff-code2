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

# Import SDK modules
try:
    from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
    from unitree_webrtc_connect.constants import RTC_TOPIC
    from aiortc import MediaStreamTrack
except ImportError as e:
    print(f"Error: Failed to import unitree_webrtc_connect packages. Make sure you are in the directory containing the package or it is installed: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.FATAL)

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

        # Queues and control
        self.video_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.lowstate_queue = queue.Queue()
        self.lidar_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.threads = []

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.output_root, f"go2_capture_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Saving data to: {self.output_dir}")

        self.conn = None

    def start_writers(self):
        if self.capture_video:
            t = threading.Thread(target=self._video_writer_worker, daemon=True, name="VideoWriter")
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
        writer = None
        while not self.stop_event.is_set() or not self.video_queue.empty():
            try:
                frame = self.video_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_path = os.path.join(self.output_dir, "video.mp4")
                writer = cv2.VideoWriter(video_path, fourcc, self.video_fps, (width, height))
                print(f"\nInitialized VideoWriter: {video_path} ({width}x{height} @ {self.video_fps} FPS)")

            writer.write(frame)
            self.video_count += 1
            self.video_queue.task_done()

        if writer is not None:
            writer.release()
            print("VideoWriter released.")

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
            def lowstate_callback(message):
                try:
                    current_message = message.get('data')
                    if current_message:
                        payload = {
                            "timestamp": time.time(),
                            "data": current_message
                        }
                        self.lowstate_queue.put(payload)
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

            def lidar_callback(message):
                try:
                    data_field = message.get("data", {})
                    inner_data = data_field.get("data", {})
                    points = inner_data.get("points")
                    
                    if points is not None:
                        # Convert numpy array points to standard list structure
                        points_list = points.tolist() if hasattr(points, "tolist") else list(points)
                        payload = {
                            "timestamp": time.time(),
                            "stamp": data_field.get("stamp"),
                            "frame_id": data_field.get("frame_id"),
                            "resolution": data_field.get("resolution"),
                            "origin": data_field.get("origin"),
                            "point_count": len(points_list),
                            "points": points_list
                        }
                        self.lidar_queue.put(payload)
                except Exception as e:
                    logging.error(f"Error in lidar callback: {e}")

            self.conn.datachannel.pub_sub.subscribe("rt/utlidar/voxel_map_compressed", lidar_callback)
            print("LiDAR snapshots subscription enabled.")

        # Keep running and printing stats
        print("\n=== Capturing Data (Press Ctrl+C to Stop) ===")
        try:
            while True:
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
    parser.add_argument("--ip", type=str, default="192.168.4.30", help="Robot local IP address")
    parser.add_argument("--aes-key", type=str, default=None, help="16-byte AES key (32 hex characters) for authentication on newer firmware")
    parser.add_argument("--output-dir", type=str, default="captures", help="Base directory to save captured data")
    parser.add_argument("--fps", type=int, default=30, help="Target frame rate for output video file")
    parser.add_argument("--no-video", action="store_true", help="Disable video stream capture")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio stream capture")
    parser.add_argument("--no-lowstate", action="store_true", help="Disable lowstate data capture")
    parser.add_argument("--no-lidar", action="store_true", help="Disable LiDAR snapshots capture")
    return parser.parse_args()

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
