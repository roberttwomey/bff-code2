#!/usr/bin/env python3
import os
import sys
import json
import argparse
import cv2

def overlay_detections(capture_dir):
    video_path = os.path.join(capture_dir, "video.mp4")
    jsonl_path = os.path.join(capture_dir, "detections.jsonl")
    output_path = os.path.join(capture_dir, "video_annotated.mp4")

    if not os.path.exists(video_path):
        print(f"Error: video.mp4 not found in {capture_dir}")
        return False
    if not os.path.exists(jsonl_path):
        print(f"Error: detections.jsonl not found in {capture_dir}")
        return False

    print(f"Loading detections from {jsonl_path}...")
    detections_map = {}
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                detections_map[data["frame_index"]] = data.get("detections", [])
    except Exception as e:
        print(f"Failed to read detections log: {e}")
        return False

    print(f"Opening source video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open source video file.")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video specs: {w}x{h} @ {fps} FPS | Total frames: {total_frames}")
    print(f"Writing annotated video to: {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    current_detections = []
    frames_since_last_detection = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Draw detections (with a persistence buffer to prevent blinking on skipped frames)
            if frame_idx in detections_map:
                current_detections = detections_map[frame_idx]
                frames_since_last_detection = 0
            else:
                frames_since_last_detection += 1
                if frames_since_last_detection > 8:  # persist for up to 8 frames (~266ms)
                    current_detections = []

            for det in current_detections:
                bbox = det.get("bbox", [0, 0, 0, 0])
                x1, y1, x2, y2 = [int(val) for val in bbox]
                label = f"{det.get('class', 'object')} ({det.get('confidence', 0.0):.2f})"
                
                # Draw a nice green rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label background box
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 6), (x1 + label_size[0] + 6, y1), (0, 255, 0), -1)
                
                # Write label text
                cv2.putText(frame, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames...")
    finally:
        cap.release()
        writer.release()

    print(f"Success! Annotated video saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Overlay saved YOLO detections over recorded Go2 stream video")
    parser.add_argument("capture_dir", type=str, help="Path to the capture session folder containing video.mp4 and detections.jsonl")
    args = parser.parse_args()

    if not os.path.isdir(args.capture_dir):
        print(f"Error: {args.capture_dir} is not a valid directory.")
        sys.exit(1)

    overlay_detections(args.capture_dir)

if __name__ == "__main__":
    main()
