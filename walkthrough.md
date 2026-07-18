# Walkthrough - Go2 Multi-modal Data Capture Script

I have implemented a clean and robust python script called [capture_go2_data.py](file:///Volumes/Work/Projects/bff/code/bff-code2/capture_go2_data.py) to connect to your Go2 robot and log its video, audio, lowstate, and LiDAR data.

## Features Implemented
- **Concurrent Stream Capture**: Uses an `asyncio` event loop to manage WebRTC connections and stream data channels.
- **Multithreaded I/O Logging**: Video, audio, and state logs are queued and written to disk in separate background threads to ensure high loop performance and prevent packet drops.
- **Graceful Shutdown**: Catches `KeyboardInterrupt` (Ctrl+C), disconnects WebRTC, turns off the LiDAR sensor, and safely flushes and closes all files.
- **Git Tracked**: Staged and ready in the `bff-code2` repository.

---

## Output Structure

When run, the script creates a folder like `captures/session-YYYYMMDD-HHMMSS/` containing:
- `video.mp4` – Video stream from the robot camera.
- `audio.wav` – Audio stream recorded from the robot mic (stereo, 48kHz, 16-bit PCM).
- `lowstate.jsonl` – Low-level body/motor states with local computer timestamps.
- `lidar.jsonl` – LiDAR snapshots containing metadata and a list of `[x, y, z]` point coordinates for each scan.

---

## How to Run

1. Navigate to the repository directory:
   ```bash
   cd /Volumes/Work/Projects/bff/code/bff-code2
   ```

2. Run the script:
   ```bash
   python3 capture_go2_data.py --ip 192.168.4.30
   ```

3. If your robot firmware is newer (Go2 >= 1.1.15) and requires the AES-128 authentication key, provide it with the `--aes-key` option:
   ```bash
   python3 capture_go2_data.py --ip 192.168.4.30 --aes-key <32-hex-characters>
   ```

4. You can selectively disable modalities if desired:
   ```bash
   # Capture everything EXCEPT video
   python3 capture_go2_data.py --ip 192.168.4.30 --no-video
   ```

5. Stop capturing by pressing **Ctrl+C**. The script will shut down cleanly and print completion statistics.
