#!/usr/bin/env python3
import os
import sys
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
    print(f"Querying Ollama model '{model_name}'...")
    
    prompt = "Describe what you see in this image. Tell me what objects/people you recognize and what kind of room or space this is."
    
    try:
        client = ollama.Client()
        response = client.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(output_path)]
                }
            ]
        )
        description = response["message"]["content"]
        print("\n=== Model Description ===")
        print(description)
        print("=========================")
    except Exception as e:
        print(f"Error querying Ollama: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
