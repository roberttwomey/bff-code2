#!/usr/bin/env python3
import os
import sys
import json
import argparse
import webbrowser
from datetime import datetime

# HTML Template with Embedded Three.js and OrbitControls
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unitree Go2 LiDAR 3D Playback</title>
    <!-- Three.js and OrbitControls via CDN -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        :root {
            --bg-color: #0a0a0c;
            --panel-bg: rgba(16, 16, 20, 0.85);
            --panel-border: rgba(255, 255, 255, 0.08);
            --accent-color: #00ff66;
            --text-color: #ffffff;
            --text-muted: #8e8e93;
            --font-mono: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        }

        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            user-select: none;
        }

        #canvas-container {
            width: 100vw;
            height: 100vh;
            position: absolute;
            top: 0;
            left: 0;
            z-index: 1;
        }

        /* Float Panel Control Board */
        .control-panel {
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 10;
            width: 320px;
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--panel-border);
            padding-bottom: 10px;
            margin-bottom: 14px;
        }

        .panel-title {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--accent-color);
        }

        .info-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            margin-bottom: 6px;
            font-family: var(--font-mono);
        }

        .info-label {
            color: var(--text-muted);
        }

        .info-value {
            color: #ffffff;
        }

        /* Timeline and buttons */
        .timeline-section {
            margin-top: 16px;
            border-top: 1px solid var(--panel-border);
            padding-top: 14px;
        }

        .slider-container {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }

        input[type="range"] {
            flex: 1;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            height: 6px;
            outline: none;
            accent-color: var(--accent-color);
            cursor: pointer;
        }

        .playback-controls {
            display: flex;
            gap: 8px;
            justify-content: space-between;
            margin-bottom: 14px;
        }

        button {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--panel-border);
            color: #ffffff;
            font-size: 0.7rem;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-family: var(--font-mono);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            transition: all 0.2s ease;
        }

        button:hover {
            background: rgba(0, 255, 102, 0.15);
            border-color: var(--accent-color);
            color: var(--accent-color);
        }

        button.active {
            background: var(--accent-color);
            color: #000000;
            border-color: var(--accent-color);
            font-weight: bold;
        }

        .checkbox-container {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.75rem;
            color: var(--text-muted);
            cursor: pointer;
        }

        .checkbox-container input {
            accent-color: var(--accent-color);
            cursor: pointer;
        }

        /* Orbit Instructions */
        .instructions {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 10;
            font-size: 0.7rem;
            color: var(--text-muted);
            font-family: var(--font-mono);
            background: rgba(10, 10, 12, 0.7);
            padding: 6px 16px;
            border-radius: 20px;
            border: 1px solid var(--panel-border);
            pointer-events: none;
        }
    </style>
</head>
<body>

    <div id="canvas-container"></div>

    <div class="control-panel">
        <div class="panel-header">
            <span class="panel-title">LiDAR 3D Viewer</span>
            <span style="font-size: 0.65rem; color: var(--text-muted); font-family: var(--font-mono);">3D CLOUD</span>
        </div>

        <div class="info-section">
            <div class="info-row">
                <span class="info-label">Capture Date:</span>
                <span class="info-value" id="captureDate">--</span>
            </div>
            <div class="info-row">
                <span class="info-label">Frame Index:</span>
                <span class="info-value" id="frameIdxDisplay">0 / 0</span>
            </div>
            <div class="info-row">
                <span class="info-label">Frame Time:</span>
                <span class="info-value" id="frameTime">0.00s</span>
            </div>
            <div class="info-row">
                <span class="info-label">Points Count:</span>
                <span class="info-value" id="pointCount">0</span>
            </div>
        </div>

        <div class="timeline-section">
            <div class="playback-controls">
                <button id="prevBtn">PREV</button>
                <button id="playBtn">PLAY</button>
                <button id="nextBtn">NEXT</button>
            </div>
            <div class="slider-container">
                <input type="range" id="timelineSlider" min="0" max="0" value="0">
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                <label class="checkbox-container">
                    <input type="checkbox" id="accumulateCheck">
                    <span>ACCUMULATE MAP</span>
                </label>
                <button id="resetViewBtn" style="font-size: 0.6rem; padding: 2px 6px;">RESET CAMERA</button>
            </div>
        </div>
    </div>

    <div class="instructions">
        Left Click + Drag to Rotate | Right Click + Drag to Pan | Scroll to Zoom
    </div>

    <script>
        // Data injected by python compiler
        // Format: Array of { time: float, points: Array of [x, y, z] }
        const snapshots = _DATA_PLACEHOLDER_;

        let scene, camera, renderer, controls;
        let pointsObject = null;
        let accumulatedPointsObject = null;
        
        let currentFrameIndex = 0;
        let isPlaying = false;
        let playInterval = null;

        // UI references
        const dateDisplay = document.getElementById('captureDate');
        const frameIdxDisplay = document.getElementById('frameIdxDisplay');
        const frameTimeDisplay = document.getElementById('frameTime');
        const pointCountDisplay = document.getElementById('pointCount');
        const playBtn = document.getElementById('playBtn');
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        const slider = document.getElementById('timelineSlider');
        const accumulateCheck = document.getElementById('accumulateCheck');
        const resetViewBtn = document.getElementById('resetViewBtn');

        function getHeightColorRGB(z) {
            const zMin = -0.8;
            const zMax = 1.2;
            let ratio = (z - zMin) / (zMax - zMin);
            ratio = Math.max(0, Math.min(1, ratio));
            
            // Hue shifts: 240 (blue) -> 120 (green) -> 0 (red)
            const hue = (1.0 - ratio) * 240;
            const color = new THREE.Color();
            color.setHSL(hue / 360, 0.9, 0.55);
            return color;
        }

        function init3D() {
            const container = document.getElementById('canvas-container');
            
            // Scene
            scene = new THREE.Scene();
            scene.fog = new THREE.FogExp2(0x0a0a0c, 0.015);

            // Camera
            camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
            camera.position.set(0, 5, 8);

            // Renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.setClearColor(scene.fog.color);
            container.appendChild(renderer.domElement);

            // Orbit Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.maxPolarAngle = Math.PI / 2 + 0.1; // allow slightly below grid level
            controls.minDistance = 1;
            controls.maxDistance = 50;

            // Grid Helper
            const gridHelper = new THREE.GridHelper(40, 40, 0x00ff66, 0x222226);
            gridHelper.position.y = -0.5; // aligned with standard dog ground plane
            scene.add(gridHelper);

            // Initialize Point Cloud Geometries
            const geometry = new THREE.BufferGeometry();
            const material = new THREE.PointsMaterial({
                size: 0.06,
                vertexColors: true,
                transparent: true,
                opacity: 0.9,
                sizeAttenuation: true
            });
            pointsObject = new THREE.Points(geometry, material);
            scene.add(pointsObject);

            // Initialize Accumulated Point Cloud
            const accGeometry = new THREE.BufferGeometry();
            const accMaterial = new THREE.PointsMaterial({
                size: 0.04,
                vertexColors: true,
                transparent: true,
                opacity: 0.45,
                sizeAttenuation: true
            });
            accumulatedPointsObject = new THREE.Points(accGeometry, accMaterial);
            scene.add(accumulatedPointsObject);

            // Window Resize
            window.addEventListener('resize', onWindowResize);
        }

        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }

        function updatePointCloud() {
            if (!snapshots || snapshots.length === 0) return;

            const snapshot = snapshots[currentFrameIndex];
            const points = snapshot.points;

            // Update Single Frame Points
            const positions = new Float32Array(points.length * 3);
            const colors = new Float32Array(points.length * 3);

            for (let i = 0; i < points.length; i++) {
                const pt = points[i];
                positions[i * 3] = -pt[1];      // Three.js X = -y_robot
                positions[i * 3 + 1] = pt[2];  // Three.js Y = z_robot (up)
                positions[i * 3 + 2] = -pt[0]; // Three.js Z = -x_robot

                const color = getHeightColorRGB(pt[2]);
                colors[i * 3] = color.r;
                colors[i * 3 + 1] = color.g;
                colors[i * 3 + 2] = color.b;
            }

            pointsObject.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            pointsObject.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            pointsObject.geometry.computeBoundingSphere();

            // Handle Accumulate Map Mode
            if (accumulateCheck.checked) {
                pointsObject.visible = false;
                accumulatedPointsObject.visible = true;

                // Collect points from start to current frame
                let totalAccumulated = 0;
                for (let f = 0; f <= currentFrameIndex; f++) {
                    totalAccumulated += snapshots[f].points.length;
                }

                const accPositions = new Float32Array(totalAccumulated * 3);
                const accColors = new Float32Array(totalAccumulated * 3);
                
                let idx = 0;
                for (let f = 0; f <= currentFrameIndex; f++) {
                    const snapPoints = snapshots[f].points;
                    for (let p = 0; p < snapPoints.length; p++) {
                        const pt = snapPoints[p];
                        accPositions[idx * 3] = -pt[1];      // Three.js X = -y_robot
                        accPositions[idx * 3 + 1] = pt[2];  // Three.js Y = z_robot (up)
                        accPositions[idx * 3 + 2] = -pt[0]; // Three.js Z = -x_robot

                        const color = getHeightColorRGB(pt[2]);
                        accColors[idx * 3] = color.r;
                        accColors[idx * 3 + 1] = color.g;
                        accColors[idx * 3 + 2] = color.b;
                        idx++;
                    }
                }

                accumulatedPointsObject.geometry.setAttribute('position', new THREE.BufferAttribute(accPositions, 3));
                accumulatedPointsObject.geometry.setAttribute('color', new THREE.BufferAttribute(accColors, 3));
                accumulatedPointsObject.geometry.computeBoundingSphere();
                pointCountDisplay.textContent = totalAccumulated.toLocaleString();
            } else {
                pointsObject.visible = true;
                accumulatedPointsObject.visible = false;
                pointCountDisplay.textContent = points.length.toLocaleString();
            }

            // Update Metadata Info Displays
            frameIdxDisplay.textContent = `${currentFrameIndex + 1} / ${snapshots.length}`;
            const elapsed = snapshot.time - snapshots[0].time;
            frameTimeDisplay.textContent = `${elapsed.toFixed(2)}s`;
            slider.value = currentFrameIndex;
        }

        // Playback operations
        function setPlaying(state) {
            isPlaying = state;
            if (isPlaying) {
                playBtn.textContent = 'PAUSE';
                playBtn.classList.add('active');
                playInterval = setInterval(() => {
                    currentFrameIndex = (currentFrameIndex + 1) % snapshots.length;
                    updatePointCloud();
                }, 100); // play at ~10 FPS
            } else {
                playBtn.textContent = 'PLAY';
                playBtn.classList.remove('active');
                if (playInterval) {
                    clearInterval(playInterval);
                    playInterval = null;
                }
            }
        }

        // Setup Actions & Listeners
        function setupControls() {
            slider.max = snapshots.length - 1;
            
            // Format capture date from first timestamp
            if (snapshots.length > 0) {
                const date = new Date(snapshots[0].time * 1000);
                dateDisplay.textContent = date.toLocaleTimeString() + ' (' + date.toLocaleDateString() + ')';
            }

            playBtn.addEventListener('click', () => setPlaying(!isPlaying));
            
            prevBtn.addEventListener('click', () => {
                setPlaying(false);
                currentFrameIndex = (currentFrameIndex - 1 + snapshots.length) % snapshots.length;
                updatePointCloud();
            });

            nextBtn.addEventListener('click', () => {
                setPlaying(false);
                currentFrameIndex = (currentFrameIndex + 1) % snapshots.length;
                updatePointCloud();
            });

            slider.addEventListener('input', (e) => {
                setPlaying(false);
                currentFrameIndex = parseInt(e.target.value);
                updatePointCloud();
            });

            accumulateCheck.addEventListener('change', () => {
                updatePointCloud();
            });

            resetViewBtn.addEventListener('click', () => {
                controls.reset();
                camera.position.set(0, 5, 8);
                controls.target.set(0, 0, 0);
            });
        }

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }

        // Start
        if (snapshots && snapshots.length > 0) {
            init3D();
            setupControls();
            updatePointCloud();
            animate();
        } else {
            document.body.innerHTML = '<div style="display:flex;justify-content:center;align-items:center;height:100vh;color:var(--accent-color);font-family:var(--font-mono);">No LiDAR data snapshots found in JSONL.</div>';
        }
    </script>
</body>
</html>
"""

def main():
    parser = argparse.ArgumentParser(description="Create interactive 3D WebGL playback viewer from Unitree Go2 lidar.jsonl log")
    parser.add_argument("capture_dir", type=str, nargs="?", default=None, 
                        help="Path to the capture session folder containing lidar.jsonl (default: newest in captures/)")
    parser.add_argument("--step", type=int, default=5, 
                        help="Point downsample factor (e.g. 5 means keep every 5th point to keep HTML load fast, default: 5)")
    args = parser.parse_args()

    capture_dir = args.capture_dir
    # Auto-find newest capture directory if none provided
    if not capture_dir:
        workspace_dir = os.path.dirname(os.path.abspath(__file__))
        captures_root = os.path.join(workspace_dir, "captures")
        if not os.path.exists(captures_root):
            print(f"Error: captures/ directory not found in {workspace_dir}")
            sys.exit(1)
        
        folders = [os.path.join(captures_root, f) for f in os.listdir(captures_root)]
        folders = [f for f in folders if os.path.isdir(f) and f.split("/")[-1].startswith("go2_capture_")]
        if not folders:
            print("Error: No capture directories found in captures/")
            sys.exit(1)
        
        folders.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        capture_dir = folders[0]

    jsonl_path = os.path.join(capture_dir, "lidar.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"Error: lidar.jsonl not found in {capture_dir}")
        sys.exit(1)

    print(f"Reading LiDAR data from: {jsonl_path}")
    print(f"Downsampling points step: {args.step} (keeping 1 out of every {args.step} points)")
    
    snapshots_data = []
    
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    raw_points = record.get("points", [])
                    timestamp = record.get("timestamp", 0.0)
                    
                    # Decimate points to keep HTML lightweight
                    decimated_points = raw_points[::args.step]
                    
                    snapshots_data.append({
                        "time": timestamp,
                        "points": decimated_points
                    })
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Failed to read lidar log: {e}")
        sys.exit(1)

    print(f"Loaded {len(snapshots_data)} snapshots.")
    if not snapshots_data:
        print("No valid LiDAR snapshots found.")
        sys.exit(1)

    # Ingest data into the HTML template
    data_json = json.dumps(snapshots_data)
    html_content = HTML_TEMPLATE.replace("_DATA_PLACEHOLDER_", data_json)

    # Write output html file to the capture directory
    output_filename = "lidar_view.html"
    output_path = os.path.join(capture_dir, output_filename)
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:
        print(f"Failed to write output HTML: {e}")
        sys.exit(1)

    print(f"Success! 3D LiDAR Viewer saved to: {output_path}")
    
    # Open the generated HTML in default web browser
    print("Opening 3D WebGL viewer in browser...")
    webbrowser.open(f"file://{output_path}")

if __name__ == "__main__":
    main()
