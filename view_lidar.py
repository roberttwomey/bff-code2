#!/usr/bin/env python3
import os
import sys
import json
import argparse
import webbrowser
import time
import socket
import threading
import http.server
import socketserver
import subprocess
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

        /* Recording overlay indicator */
        #recording-overlay {
            display: none;
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 100;
            background: rgba(255, 0, 0, 0.85);
            border: 1px solid #ff3333;
            color: #ffffff;
            font-family: var(--font-mono);
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
            box-shadow: 0 4px 16px rgba(255, 0, 0, 0.4);
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { opacity: 0.8; }
            50% { opacity: 1.0; }
            100% { opacity: 0.8; }
        }
    </style>
</head>
<body>

    <div id="canvas-container"></div>
    <div id="recording-overlay">REC: ACCUMULATING VIDEO...</div>

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
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; gap: 6px;">
                <label class="checkbox-container">
                    <input type="checkbox" id="accumulateCheck">
                    <span>ACCUMULATE MAP</span>
                </label>
                <button id="resetViewBtn" style="font-size: 0.55rem; padding: 2px 6px;">RESET</button>
            </div>
            <div style="display: flex; gap: 6px; margin-top: 8px; justify-content: space-between; width: 100%;">
                <button id="saveLidarSettings" style="font-size: 0.6rem; padding: 4px 6px; flex: 1;">SAVE SETTINGS</button>
                <button id="importLidarSettingsBtn" style="font-size: 0.6rem; padding: 4px 6px; flex: 1;">IMPORT</button>
                <input type="file" id="importLidarSettingsFile" accept=".json" style="display: none;">
            </div>
        </div>
    </div>

    <div class="instructions">
        Left Click + Drag to Rotate | Right Click + Drag to Pan | Scroll to Zoom
    </div>

    <script>
        // Data injected by python compiler
        // Format: Array of { time: float, points: Array of [x, y, z], camera_pos: [x,y,z]/null, camera_target: [x,y,z]/null }
        const snapshots = _DATA_PLACEHOLDER_;

        let scene, camera, renderer, controls;
        let pointsObject = null;
        let accumulatedPointsObject = null;
        let lidarRangeCutoff = 12;
        let pointSizeVal = 4;
        
        let currentFrameIndex = 0;
        let isPlaying = false;
        let playInterval = null;

        // URL arguments
        const urlParams = new URLSearchParams(window.location.search);
        const shouldRecord = urlParams.get('record') === 'true';

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
        const recOverlay = document.getElementById('recording-overlay');

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
            
            // Load default settings from server or localStorage if available
            const serverSettings = _SETTINGS_PLACEHOLDER_;
            const savedSettingsStr = localStorage.getItem('lidarSettings');
            let initialSettings = null;
            if (serverSettings) {
                initialSettings = serverSettings;
            } else if (savedSettingsStr) {
                try {
                    initialSettings = JSON.parse(savedSettingsStr);
                } catch (e) {
                    console.warn("Failed to parse saved settings from localStorage:", e);
                }
            }

            if (initialSettings) {
                try {
                    if (initialSettings.camera && initialSettings.camera.position) {
                        camera.position.set(initialSettings.camera.position.x, initialSettings.camera.position.y, initialSettings.camera.position.z);
                    } else {
                        camera.position.set(0, 5, 8);
                    }
                    if (initialSettings.camera && initialSettings.camera.zoom !== undefined) {
                        camera.zoom = initialSettings.camera.zoom;
                    }
                    if (initialSettings.accumulate !== undefined) {
                        document.getElementById('accumulateCheck').checked = initialSettings.accumulate;
                    }
                    if (initialSettings.cutoff !== undefined) {
                        lidarRangeCutoff = initialSettings.cutoff;
                    }
                    if (initialSettings.size !== undefined) {
                        pointSizeVal = initialSettings.size;
                    }
                } catch (e) {
                    console.warn("Failed to apply initial settings:", e);
                    camera.position.set(0, 5, 8);
                }
            } else {
                camera.position.set(0, 5, 8);
            }

            // Renderer - force preserveDrawingBuffer for canvas capture if recording
            renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: shouldRecord });
            renderer.setSize(window.innerWidth, window.innerHeight);
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

            if (initialSettings && initialSettings.controls && initialSettings.controls.target) {
                controls.target.set(initialSettings.controls.target.x, initialSettings.controls.target.y, initialSettings.controls.target.z);
            } else {
                controls.target.set(0, 0, 0);
            }
            controls.update();

            // Grid Helper
            const gridHelper = new THREE.GridHelper(40, 40, 0x00ff66, 0x222226);
            gridHelper.position.y = -0.5; // aligned with standard dog ground plane
            scene.add(gridHelper);

            // Initialize Point Cloud Geometries
            const geometry = new THREE.BufferGeometry();
            const material = new THREE.PointsMaterial({
                size: 0.015 * pointSizeVal,
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
                size: 0.01 * pointSizeVal,
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
            const filteredPos = [];
            const filteredColors = [];
            for (let i = 0; i < points.length; i++) {
                const pt = points[i];
                const x = pt[0];
                const y = pt[1];
                const z = pt[2];
                const distance = Math.sqrt(x * x + y * y + z * z);
                if (distance > lidarRangeCutoff) continue;

                filteredPos.push(-y, z, -x);
                const color = getHeightColorRGB(z);
                filteredColors.push(color.r, color.g, color.b);
            }
            const positions = new Float32Array(filteredPos);
            const colors = new Float32Array(filteredColors);

            pointsObject.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            pointsObject.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            pointsObject.geometry.computeBoundingSphere();

            // Handle Accumulate Map Mode
            if (accumulateCheck.checked) {
                pointsObject.visible = false;
                accumulatedPointsObject.visible = true;

                // Collect points from start to current frame
                const accPos = [];
                const accColorsList = [];
                let totalAccumulated = 0;
                for (let f = 0; f <= currentFrameIndex; f++) {
                    const snapPoints = snapshots[f].points;
                    for (let p = 0; p < snapPoints.length; p++) {
                        const pt = snapPoints[p];
                        const x = pt[0];
                        const y = pt[1];
                        const z = pt[2];
                        const distance = Math.sqrt(x * x + y * y + z * z);
                        if (distance > lidarRangeCutoff) continue;

                        accPos.push(-y, z, -x);
                        const color = getHeightColorRGB(z);
                        accColorsList.push(color.r, color.g, color.b);
                        totalAccumulated++;
                    }
                }

                const accPositions = new Float32Array(accPos);
                const accColors = new Float32Array(accColorsList);

                accumulatedPointsObject.geometry.setAttribute('position', new THREE.BufferAttribute(accPositions, 3));
                accumulatedPointsObject.geometry.setAttribute('color', new THREE.BufferAttribute(accColors, 3));
                accumulatedPointsObject.geometry.computeBoundingSphere();
                pointCountDisplay.textContent = totalAccumulated.toLocaleString();
            } else {
                pointsObject.visible = true;
                accumulatedPointsObject.visible = false;
                pointCountDisplay.textContent = (filteredPos.length / 3).toLocaleString();
            }

            // Update Metadata Info Displays
            frameIdxDisplay.textContent = `${currentFrameIndex + 1} / ${snapshots.length}`;
            const elapsed = snapshot.time - snapshots[0].time;
            frameTimeDisplay.textContent = `${elapsed.toFixed(2)}s`;
            slider.value = currentFrameIndex;

            // Align pointsObject and accumulatedPointsObject to the dog's body frame
            const alignCheckEl = document.getElementById('alignCheck');
            const alignCheck = alignCheckEl ? alignCheckEl.checked : true;
            if (alignCheck && snapshot.slam_pose) {
                const pos = snapshot.slam_pose.position;
                const ori = snapshot.slam_pose.orientation;

                // Transform dog pose from ROS coordinates to Three.js coordinates
                const posThree = new THREE.Vector3(-pos[1], pos[2], -pos[0]);

                // Extract Euler angles from ROS quaternion (ZYX order)
                const qRos = new THREE.Quaternion(ori[0], ori[1], ori[2], ori[3]);
                const eulerRos = new THREE.Euler().setFromQuaternion(qRos, 'ZYX');
                const yaw = eulerRos.z;
                const pitch = eulerRos.y;
                const roll = eulerRos.x;

                // Map to Three.js rotation convention
                const quatThree = new THREE.Quaternion().setFromEuler(
                    new THREE.Euler(pitch, yaw, roll, 'YXZ')
                );

                // Build the global world matrix for the dog
                const dogMatrix = new THREE.Matrix4().compose(
                    posThree,
                    quatThree,
                    new THREE.Vector3(1, 1, 1)
                );

                // Invert it to map points from global (odom) space to dog-local space
                const invMatrix = new THREE.Matrix4().copy(dogMatrix).invert();

                pointsObject.matrix.copy(invMatrix);
                pointsObject.matrixAutoUpdate = false;
                accumulatedPointsObject.matrix.copy(invMatrix);
                accumulatedPointsObject.matrixAutoUpdate = false;
            } else {
                pointsObject.matrix.identity();
                pointsObject.matrixAutoUpdate = false;
                accumulatedPointsObject.matrix.identity();
                accumulatedPointsObject.matrixAutoUpdate = false;
            }
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

        // Offline recording pipeline
        async function runOfflineRecording() {
            console.log("[Lidar Recorder] Initializing offline canvas recording server...");
            recOverlay.style.display = 'block';
            accumulateCheck.checked = true; // Record the accumulated map builder
            
            // Set up stream capture from WebGL canvas at 30 FPS
            const stream = renderer.domElement.captureStream(30);
            
            let options = { mimeType: 'video/webm; codecs=vp9', videoBitsPerSecond: 4000000 };
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options = { mimeType: 'video/webm; codecs=vp8', videoBitsPerSecond: 4000000 };
            }
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options = { mimeType: 'video/webm', videoBitsPerSecond: 4000000 };
            }

            const recorder = new MediaRecorder(stream, options);
            const chunks = [];
            
            recorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) chunks.push(e.data);
            };

            recorder.onstop = async () => {
                console.log("[Lidar Recorder] Compiling recorded video blob...");
                const blob = new Blob(chunks, { type: 'video/webm' });
                
                recOverlay.textContent = "REC: UPLOADING VIDEO DATA...";
                try {
                    const res = await fetch('/upload', {
                        method: 'POST',
                        body: blob
                    });
                    if (res.ok) {
                        recOverlay.textContent = "REC: SUCCESS. CLOSING...";
                        console.log("[Lidar Recorder] Upload successful!");
                    } else {
                        recOverlay.textContent = "REC: UPLOAD FAILED.";
                        console.error("[Lidar Recorder] Upload failed.");
                    }
                } catch (err) {
                    recOverlay.textContent = "REC: ERROR UPLOADING.";
                    console.error("[Lidar Recorder] Connection error during upload:", err);
                }
            };

            recorder.start();

            // Run step-by-step frame rendering to guarantee zero frame drop
            currentFrameIndex = 0;
            
            // Use MessageChannel to bypass background tab timer/rAF throttling
            const channel = new MessageChannel();
            const frameDelay = 33; // ~30 FPS (33ms per frame)
            let nextFrameTime = performance.now();

            channel.port1.onmessage = () => {
                const now = performance.now();
                if (now >= nextFrameTime) {
                    step();
                    nextFrameTime = now + frameDelay;
                } else {
                    // Not time for next frame yet, yield control and check again immediately
                    channel.port2.postMessage(null);
                }
            };

            async function step() {
                if (currentFrameIndex >= snapshots.length) {
                    // Let final frames bake in for a second, then stop
                    setTimeout(() => recorder.stop(), 1000);
                    return;
                }

                // Update Camera Pos from recorded path
                const snap = snapshots[currentFrameIndex];
                if (snap.camera_pos) {
                    camera.position.set(snap.camera_pos[0], snap.camera_pos[1], snap.camera_pos[2]);
                    controls.target.set(snap.camera_target[0], snap.camera_target[1], snap.camera_target[2]);
                } else {
                    // Slow fallback orbital rotation
                    const angle = currentFrameIndex * 0.03;
                    camera.position.x = Math.sin(angle) * 12;
                    camera.position.z = Math.cos(angle) * 12;
                    camera.position.y = 6;
                    controls.target.set(0, 0, 0);
                }

                updatePointCloud();
                renderer.render(scene, camera);
                controls.update();

                currentFrameIndex++;
                recOverlay.textContent = `REC: CAPTURING FRAME ${currentFrameIndex} / ${snapshots.length}`;
                
                // Yield to event loop to allow encoder to consume canvas frame
                channel.port2.postMessage(null);
            }

            // Start step-by-step playback after a short initialization delay
            setTimeout(() => channel.port2.postMessage(null), 500);
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
                const savedSettingsStr = localStorage.getItem('lidarSettings');
                if (savedSettingsStr) {
                    try {
                        const settings = JSON.parse(savedSettingsStr);
                        if (settings.camera && settings.camera.position) {
                            camera.position.set(settings.camera.position.x, settings.camera.position.y, settings.camera.position.z);
                        } else {
                            camera.position.set(0, 5, 8);
                        }
                        if (settings.camera && settings.camera.zoom !== undefined) {
                            camera.zoom = settings.camera.zoom;
                            camera.updateProjectionMatrix();
                        }
                        if (settings.controls && settings.controls.target) {
                            controls.target.set(settings.controls.target.x, settings.controls.target.y, settings.controls.target.z);
                        } else {
                            controls.target.set(0, 0, 0);
                        }
                        if (settings.accumulate !== undefined) {
                            document.getElementById('accumulateCheck').checked = settings.accumulate;
                            updatePointCloud();
                        }
                    } catch (e) {
                        controls.reset();
                        camera.position.set(0, 5, 8);
                        controls.target.set(0, 0, 0);
                    }
                } else {
                    controls.reset();
                    camera.position.set(0, 5, 8);
                    controls.target.set(0, 0, 0);
                }
                controls.update();
            });

            // Save LiDAR Settings (Download & LocalStorage)
            document.getElementById('saveLidarSettings').addEventListener('click', (e) => {
                e.preventDefault();
                const settings = {
                    camera: {
                        position: { x: camera.position.x, y: camera.position.y, z: camera.position.z },
                        rotation: { x: camera.rotation.x, y: camera.rotation.y, z: camera.rotation.z },
                        zoom: camera.zoom
                    },
                    controls: {
                        target: { x: controls.target.x, y: controls.target.y, z: controls.target.z }
                    },
                    accumulate: document.getElementById('accumulateCheck').checked,
                    cutoff: lidarRangeCutoff,
                    size: pointSizeVal
                };
                
                // Save to localStorage as the default for future page loads
                localStorage.setItem('lidarSettings', JSON.stringify(settings));
                
                // Trigger JSON download
                const blob = new Blob([JSON.stringify(settings, null, 4)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'lidar_settings.json';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            });

            // Import LiDAR Settings
            document.getElementById('importLidarSettingsBtn').addEventListener('click', (e) => {
                e.preventDefault();
                document.getElementById('importLidarSettingsFile').click();
            });

            document.getElementById('importLidarSettingsFile').addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = function(evt) {
                    try {
                        const settings = JSON.parse(evt.target.result);
                        if (settings.camera && settings.camera.position) {
                            camera.position.set(settings.camera.position.x, settings.camera.position.y, settings.camera.position.z);
                        }
                        if (settings.camera && settings.camera.zoom !== undefined) {
                            camera.zoom = settings.camera.zoom;
                            camera.updateProjectionMatrix();
                        }
                        if (settings.controls && settings.controls.target) {
                            controls.target.set(settings.controls.target.x, settings.controls.target.y, settings.controls.target.z);
                        }
                        controls.update();
                        
                        if (settings.accumulate !== undefined) {
                            document.getElementById('accumulateCheck').checked = settings.accumulate;
                        }
                        if (settings.cutoff !== undefined) {
                            lidarRangeCutoff = settings.cutoff;
                        }
                        if (settings.size !== undefined) {
                            pointSizeVal = settings.size;
                            if (pointsObject && pointsObject.material) {
                                pointsObject.material.size = 0.015 * pointSizeVal;
                            }
                            if (accumulatedPointsObject && accumulatedPointsObject.material) {
                                accumulatedPointsObject.material.size = 0.01 * pointSizeVal;
                            }
                        }
                        updatePointCloud();
                        
                        // Also save to localStorage as the new default
                        localStorage.setItem('lidarSettings', JSON.stringify(settings));
                        console.log("LiDAR settings imported and saved to localStorage.");
                    } catch (err) {
                        alert("Failed to parse settings JSON: " + err.message);
                    }
                };
                reader.readAsText(file);
                e.target.value = ''; // Clear value
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
            
            if (shouldRecord) {
                runOfflineRecording();
            } else {
                updatePointCloud();
                animate();
            }
        } else {
            document.body.innerHTML = '<div style="display:flex;justify-content:center;align-items:center;height:100vh;color:var(--accent-color);font-family:var(--font-mono);">No LiDAR data snapshots found in JSONL.</div>';
        }
    </script>
</body>
</html>
"""

class LidarHTTPServerHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP server to handle WebM uploads from the browser context."""
    def __init__(self, *args, **kwargs):
        server = args[2]
        directory = getattr(server, "capture_dir", os.getcwd())
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format, *args):
        pass  # Suppress server terminal request noise
        
    def do_POST(self):
        if self.path == "/upload":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Save raw recorded WebM bytes to disk
            webm_path = os.path.join(self.server.capture_dir, "lidar_render.webm")
            try:
                with open(webm_path, "wb") as f:
                    f.write(post_data)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode())
                
                # Trigger server shutdown flag
                self.server.upload_received = True
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                print(f"[HTTP Server] Error writing WebM: {e}")
            return
        super().do_POST()

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    pass

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

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
    
    # Load camera path if it exists
    camera_path = []
    camera_path_file = os.path.join(capture_dir, "camera_path.jsonl")
    if os.path.exists(camera_path_file):
        try:
            with open(camera_path_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        camera_path.append(json.loads(line))
            print(f"Loaded {len(camera_path)} recorded camera movement states.")
        except Exception as e:
            print(f"Failed to read camera path: {e}")

    # Load lowstate log if it exists to retrieve dog SLAM positions and orientations
    lowstate_path = []
    lowstate_file = os.path.join(capture_dir, "lowstate.jsonl")
    if os.path.exists(lowstate_file):
        try:
            with open(lowstate_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        lowstate_path.append(json.loads(line))
            print(f"Loaded {len(lowstate_path)} telemetry logs for dog pose alignment.")
        except Exception as e:
            print(f"Failed to read lowstate log: {e}")

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
                    
                    # Find closest camera move if available
                    camera_pos = None
                    camera_target = None
                    if camera_path:
                        closest = min(camera_path, key=lambda c: abs(c.get("timestamp", 0.0) - timestamp))
                        if abs(closest.get("timestamp", 0.0) - timestamp) < 1.5:  # within 1.5s tolerance
                            camera_pos = closest["position"]
                            camera_target = closest["target"]

                    # Find closest lowstate SLAM pose if available
                    slam_pose = {
                        "position": [0.0, 0.0, 0.0],
                        "orientation": [0.0, 0.0, 0.0, 1.0]
                    }
                    if lowstate_path:
                        closest_low = min(lowstate_path, key=lambda l: abs(l.get("timestamp", 0.0) - timestamp))
                        if abs(closest_low.get("timestamp", 0.0) - timestamp) < 1.5:  # within 1.5s tolerance
                            low_data = closest_low.get("data", {})
                            slam_pose = low_data.get("slam_pose", slam_pose)

                    snapshots_data.append({
                        "time": timestamp,
                        "points": decimated_points,
                        "camera_pos": camera_pos,
                        "camera_target": camera_target,
                        "slam_pose": slam_pose
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

    # Load settings from lidar_settings.json if it exists
    settings_data = None
    possible_paths = [
        os.path.join(capture_dir, "lidar_settings.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "lidar_settings.json")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    settings_data = json.load(f)
                    print(f"Loaded LiDAR settings from: {p}")
                    break
            except Exception as e:
                print(f"Failed to read settings from {p}: {e}")

    # Ingest data into the HTML template
    data_json = json.dumps(snapshots_data)
    settings_json = json.dumps(settings_data) if settings_data else "null"
    html_content = HTML_TEMPLATE.replace("_DATA_PLACEHOLDER_", data_json)
    html_content = html_content.replace("_SETTINGS_PLACEHOLDER_", settings_json)

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

    # Launch local web server and trigger recording
    port = find_free_port()
    server = ThreadedHTTPServer(('localhost', port), LidarHTTPServerHandler)
    server.capture_dir = capture_dir
    server.upload_received = False

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Open HTML page with recording enabled in default web browser
    url = f"http://localhost:{port}/{output_filename}?record=true"
    print(f"Starting background recording server at: {url}")
    print("Opening 3D WebGL renderer in browser to compile video...")
    webbrowser.open(url)

    # Wait for browser upload completion
    print("Waiting for browser rendering and upload to complete (max 90s)...")
    start_time = time.time()
    try:
        while not server.upload_received:
            time.sleep(0.5)
            if time.time() - start_time > 90:
                print("Timeout waiting for browser upload.")
                server.shutdown()
                sys.exit(1)
    except KeyboardInterrupt:
        print("\nRecording aborted by user.")
        server.shutdown()
        sys.exit(0)

    server.shutdown()
    print("LiDAR WebM rendering received successfully.")

    # Convert WebM to high-quality MP4 using FFmpeg
    webm_path = os.path.join(capture_dir, "lidar_render.webm")
    mp4_path = os.path.join(capture_dir, "lidar_render.mp4")
    if os.path.exists(webm_path):
        print("Converting WebM to MP4 using FFmpeg...")
        try:
            # -y overwrites existing, -c:v libx264 encodes H.264, -pix_fmt yuv420p ensures compatibility
            subprocess.run([
                "ffmpeg", "-y", "-i", webm_path, 
                "-c:v", "libx264", "-pix_fmt", "yuv420p", 
                mp4_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Success! Final LiDAR render video saved to: {mp4_path}")
            os.remove(webm_path)  # Delete intermediate WebM file
        except Exception as e:
            print(f"FFmpeg conversion failed: {e}. Keeping raw WebM file: {webm_path}")

if __name__ == "__main__":
    main()
