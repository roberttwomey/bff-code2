#!/usr/bin/env python3
import os
import re
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

def natural_sort_key(path):
    """Sort chunk_0, chunk_1, ..., chunk_10 in numeric rather than string order."""
    name = os.path.basename(os.path.normpath(path))
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]

def find_chunk_dirs(root_dir):
    """Find immediate subdirectories of root_dir that contain both video.mp4
    and detections.jsonl, naturally sorted (chunk_0, chunk_1, ..., chunk_10)."""
    chunk_dirs = []
    for entry in os.listdir(root_dir):
        full_path = os.path.join(root_dir, entry)
        if not os.path.isdir(full_path):
            continue
        if os.path.exists(os.path.join(full_path, "video.mp4")) and \
           os.path.exists(os.path.join(full_path, "detections.jsonl")):
            chunk_dirs.append(full_path)
    chunk_dirs.sort(key=natural_sort_key)
    return chunk_dirs

def concatenate_videos(video_paths, output_path):
    """Append a list of videos together, frame by frame, into a single output file."""
    if not video_paths:
        print("No videos to concatenate.")
        return False

    print(f"\nConcatenating {len(video_paths)} video(s) into: {output_path}")

    target_fps = None
    target_w = None
    target_h = None
    writer = None

    try:
        for video_path in video_paths:
            if not os.path.exists(video_path):
                print(f"  Skipping missing video: {video_path}")
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"  Skipping unreadable video: {video_path}")
                cap.release()
                continue

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if writer is None:
                target_fps = cap.get(cv2.CAP_PROP_FPS)
                target_w, target_h = w, h
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, target_fps, (target_w, target_h))
                print(f"  Combined video specs: {target_w}x{target_h} @ {target_fps} FPS")

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if (w, h) != (target_w, target_h):
                    frame = cv2.resize(frame, (target_w, target_h))
                writer.write(frame)
                frame_count += 1
            cap.release()
            print(f"  Appended {frame_count} frames from {os.path.basename(os.path.dirname(video_path))}")
    finally:
        if writer is not None:
            writer.release()

    if writer is None:
        print("Failed to open any videos for concatenation.")
        return False

    print(f"Success! Combined video saved to: {output_path}")
    return True

def process_all_chunks(root_dir):
    """Overlay detections on every chunk subdirectory found under root_dir,
    then append all of the resulting annotated videos into one combined file."""
    chunk_dirs = find_chunk_dirs(root_dir)
    if not chunk_dirs:
        print(f"No chunk subdirectories with video.mp4 + detections.jsonl found under {root_dir}")
        return False

    print(f"Found {len(chunk_dirs)} chunk(s) under {root_dir}:")
    for chunk_dir in chunk_dirs:
        print(f"  - {os.path.basename(chunk_dir)}")

    annotated_paths = []
    for chunk_dir in chunk_dirs:
        print(f"\n--- Processing {os.path.basename(chunk_dir)} ---")
        if overlay_detections(chunk_dir):
            annotated_paths.append(os.path.join(chunk_dir, "video_annotated.mp4"))

    combined_output = os.path.join(root_dir, "video_annotated_combined.mp4")
    concatenate_videos(annotated_paths, combined_output)
    return True

def main():
    parser = argparse.ArgumentParser(description="Overlay saved YOLO detections over recorded Go2 stream video")
    parser.add_argument(
        "capture_dir",
        type=str,
        help="Path to a single chunk folder (containing video.mp4 and detections.jsonl), "
             "or a session folder containing multiple chunk_* subfolders"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.capture_dir):
        print(f"Error: {args.capture_dir} is not a valid directory.")
        sys.exit(1)

    # Single-chunk mode: the given directory directly contains video.mp4 + detections.jsonl
    if os.path.exists(os.path.join(args.capture_dir, "video.mp4")) and \
       os.path.exists(os.path.join(args.capture_dir, "detections.jsonl")):
        overlay_detections(args.capture_dir)
    else:
        # Batch mode: find every chunk_* subdirectory, annotate each one, and
        # append the results into a single combined video.
        process_all_chunks(args.capture_dir)

if __name__ == "__main__":
    main()
