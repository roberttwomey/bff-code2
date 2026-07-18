#!/usr/bin/env python3
import os
import sys
import time
import asyncio
from pathlib import Path
import cv2
import ollama
import dotenv

# Load environment variables (e.g., to read BFF_OLLAMA_MODEL)
dotenv.load_dotenv()

async def grab_go2_frame(ip, aes_key, timeout=5.0):
    from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
    from aiortc import MediaStreamTrack
    
    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA, 
        ip=ip, 
        aes_128_key=aes_key
    )
    
    frame_future = asyncio.get_event_loop().create_future()
    
    async def recv_camera_stream(track: MediaStreamTrack):
        try:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            if not frame_future.done():
                frame_future.set_result(img)
        except Exception as e:
            if not frame_future.done():
                frame_future.set_exception(e)
                
    print(f"Connecting to Go2 WebRTC at {ip}...")
    await conn.connect()
    print("Connected to Go2 WebRTC! Enabling video channel...")
    
    conn.video.add_track_callback(recv_camera_stream)
    conn.video.switchVideoChannel(True)
    
    try:
        img = await asyncio.wait_for(frame_future, timeout=timeout)
        return img
    finally:
        print("Disconnecting from Go2 WebRTC...")
        try:
            await conn.disconnect()
        except Exception as close_err:
            print(f"Error disconnecting: {close_err}", file=sys.stderr)

def main():
    # 1. Try to grab image from Go2 WebRTC
    ip = os.getenv("UNITREE_ROBOT_IP", "192.168.4.30")
    aes_key = os.getenv("UNITREE_AES_KEY", "")
    if not aes_key:
        aes_key = None
        
    frame = None
    if ip:
        try:
            print("Attempting to capture image from Go2 robot via WebRTC...")
            # Run the connection and capture with a strict timeout
            frame = asyncio.run(asyncio.wait_for(grab_go2_frame(ip, aes_key, timeout=5.0), timeout=8.0))
            print("Successfully captured frame from Go2 robot WebRTC.")
        except Exception as e:
            print(f"Go2 WebRTC capture failed or timed out: {e}")
            print("Falling back to built-in webcam...")
            
    if frame is None:
        # Fallback to local webcam
        camera_index = 0
        print(f"Opening local webcam at index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open local webcam at index {camera_index}", file=sys.stderr)
            sys.exit(1)
            
        # Read a few frames to let auto-exposure adjust
        print("Warming up camera...")
        for i in range(5):
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Failed to read warm-up frame {i+1}", file=sys.stderr)
            
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error: Could not read frame from local webcam", file=sys.stderr)
            sys.exit(1)
            
    output_path = Path("snapshot.jpg")
    cv2.imwrite(str(output_path), frame)
    print(f"Successfully saved snapshot to {output_path}")
    
    # 2. Resolve model name from env or default
    model_name = os.environ.get("BFF_OLLAMA_MODEL", "gemma4:e2b")
    print(f"Querying Ollama model '{model_name}' (keeping model warm)...")
    
    # prompt = "Describe what you see in this image. Tell me what objects/people you recognize and what kind of room or space this is."
    prompt = """You are the visual processing unit for the SNAPPER robot dog. Analyze the input image and output a dense, flat list of semantic tags, objects, spatial layout, and environmental context. 

Strict constraints:
1. No conversational filler, intro, or outro text.
2. No markdown formatting, bullet points, or line breaks.
3. Output a single, continuous paragraph of comma-separated descriptions.
4. Prioritize: Exact object names, spatial relationships (e.g., "chair left of table"), room type, lighting conditions, and human presence/actions.

Example Format:
[room type], [primary lighting], [object 1 with location], [object 2], [detected person with posture/action]"""
    
    try:
        client = ollama.Client()
        query_start = time.perf_counter()
        response = client.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(output_path)]
                }
            ],
            keep_alive=-1
        )
        query_duration = time.perf_counter() - query_start
        
        description = response["message"]["content"]
        print("\n=== Model Description ===")
        print(description)
        print("=========================")
        print(f"\nProcessing took: {query_duration:.2f} seconds.")
    except Exception as e:
        print(f"Error querying Ollama: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
