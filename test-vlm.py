#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path
import cv2
import ollama
import dotenv

# Load environment variables (e.g., to read BFF_OLLAMA_MODEL)
dotenv.load_dotenv()

def main():
    # 1. Grab image from webcam
    camera_index = 0
    print(f"Opening camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}", file=sys.stderr)
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
        print("Error: Could not read frame from camera", file=sys.stderr)
        sys.exit(1)
        
    output_path = Path("snapshot.jpg")
    cv2.imwrite(str(output_path), frame)
    print(f"Successfully captured image and saved to {output_path}")
    
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
