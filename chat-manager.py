#!/usr/bin/env python3
"""Voice chat assistant using Whisper STT, Ollama Gemma 3 Nano, and Piper TTS.

This script performs continuous voice activity detection (VAD) on microphone
audio, automatically segments speech, transcribes each utterance with Whisper,
sends the resulting text to an Ollama model (`gemma3n:e2b` by default), and
plays back the assistant response via Piper text-to-speech using the Python
`piper-tts` library.

Requirements:
    - ollama (Python package) with the `gemma3n:e2b` model pulled locally
    - openai-whisper
    - sounddevice
    - soundfile
    - numpy
    - piper-tts (Python package) and at least one Piper voice model file

Example usage:
    python local/bff-voice-chat.py --piper-voice piper/en_GB-alan-medium.onnx
    
    python local/bff-voice-chat.py --piper-voice local/piper/en_GB-alan-medium.onnx --show-levels

To test just ollama: 
ollama run gemma3n:e2b

Environment variables:
    BFF_OLLAMA_MODEL   override Ollama model name for text chat (default: gemma3n:e2b)
    BFF_VLM_MODEL      override Ollama model name for VLM scene captioning (default: moondream)
    BFF_VLM_NUM_PREDICT override max tokens generated per VLM scene description (default: 50)
    BFF_VLM_INTERVAL   override minimum seconds between VLM capture starts (default: 1.5)
    BFF_WHISPER_MODEL  override Whisper model size (default: tiny.en)
    BFF_PIPER_VOICE    override Piper voice path if --piper-voice not provided
    BFF_INTERRUPTABLE  override interruptable behavior (default: true)
    BFF_LOG_ROOT       override session history root (default: ./captures, alongside AV/lidar capture data)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, List
import wave

import numpy as np
import ollama
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import torch
import dotenv

import cv2
import asyncio
import logging
from contextlib import contextmanager

# Suppress aiortc and pyav loggers
logging.getLogger("aiortc").setLevel(logging.ERROR)
logging.getLogger("aiortc.rtp").setLevel(logging.ERROR)
logging.getLogger("aiortc.codecs.h264").setLevel(logging.ERROR)

@contextmanager
def silence_outputs():
    # Redirect python stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    null_file = open(os.devnull, 'w')
    sys.stdout = null_file
    sys.stderr = null_file
    
    # Redirect C-level file descriptors (FD 1 and FD 2)
    try:
        fd_out = os.dup(1)
        fd_err = os.dup(2)
        null_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        os.close(null_fd)
    except Exception:
        fd_out = None
        fd_err = None
        
    try:
        yield
    finally:
        # Restore python streams
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        null_file.close()
        
        # Restore C file descriptors
        if fd_out is not None:
            try:
                os.dup2(fd_out, 1)
                os.close(fd_out)
            except Exception:
                pass
        if fd_err is not None:
            try:
                os.dup2(fd_err, 2)
                os.close(fd_err)
            except Exception:
                pass



# --- Global VLM States ---
vlm_lock = threading.Lock()
latest_scene_description = ""
last_vlm_query_time = 0.0
is_dialogue_active = False
last_interaction_time = time.time()

# --- Global Body State ---
body_state_lock = threading.Lock()
latest_body_state = ""

from piper import PiperVoice
try:  # Optional type that some versions expose
    from piper import AudioChunk  # type: ignore
except ImportError:  # pragma: no cover - older library versions
    AudioChunk = None



dotenv.load_dotenv()

def fix_user_paths() -> None:
    """
    If /home/cohab does not exist, replace /home/cohab prefixes in environment variables
    with the current user's home directory.
    """
    # Check if we are likely on a different machine (i.e., /home/cohab missing)
    # or just want to be safe and use the current user's home.
    cohab_home = Path("/home/cohab")
    if cohab_home.exists():
        return

    current_home = Path.home()
    print(f"Notice: {cohab_home} not found. Remapping paths to {current_home}...", file=sys.stderr)

    for key, value in os.environ.items():
        if value and "/home/cohab" in value:
            new_value = value.replace("/home/cohab", str(current_home))
            os.environ[key] = new_value
            # print(f"  Remapped {key}: {value} -> {new_value}", file=sys.stderr)

fix_user_paths()


DEFAULT_SYSTEM_PROMPT = (
    os.environ.get(
        "BFF_SYSTEM_PROMPT",
        "you are SNAPPER a robot dog. you do not say woof, whir, tail wag. answer in 2 sentences or less.",
    )
)

DEFAULT_OLLAMA_MODEL = os.environ.get("BFF_OLLAMA_MODEL", "gemma3n:e2b")
DEFAULT_VLM_MODEL = os.environ.get("BFF_VLM_MODEL", "moondream")
DEFAULT_WHISPER_MODEL = os.environ.get("BFF_WHISPER_MODEL", "tiny.en")
DEFAULT_SAMPLE_RATE = int(os.environ.get("BFF_SAMPLE_RATE", "16000"))
DEFAULT_PLAYBACK_SPEED = float(os.environ.get("BFF_PLAYBACK_SPEED", "1.0"))
DEFAULT_INPUT_DEVICE_KEYWORD = os.environ.get(
    "BFF_INPUT_DEVICE_KEYWORD", "OpenRun Pro 2 by Shokz"
)
DEFAULT_ACTIVATION_THRESHOLD = float(os.environ.get("BFF_ACTIVATION_THRESHOLD", "0.03"))
DEFAULT_SILENCE_THRESHOLD = float(os.environ.get("BFF_SILENCE_THRESHOLD", "0.015"))
DEFAULT_SILENCE_DURATION = float(os.environ.get("BFF_SILENCE_DURATION", "0.8"))
DEFAULT_MIN_PHRASE_SECONDS = float(os.environ.get("BFF_MIN_PHRASE_SECONDS", "0.5"))
DEFAULT_BLOCK_DURATION = float(os.environ.get("BFF_BLOCK_DURATION", "0.2"))
DEFAULT_INTERRUPTABLE_ENV = os.environ.get("BFF_INTERRUPTABLE", "true").lower()
DEFAULT_INTERRUPTABLE = DEFAULT_INTERRUPTABLE_ENV in ("true", "1", "yes", "on")
DEFAULT_FLUSH_ON_INTERRUPT_ENV = os.environ.get("BFF_FLUSH_ON_INTERRUPT", "false").lower()
DEFAULT_FLUSH_ON_INTERRUPT = DEFAULT_FLUSH_ON_INTERRUPT_ENV in ("true", "1", "yes", "on")
LOG_ROOT = Path(
    os.environ.get("BFF_LOG_ROOT", Path(__file__).resolve().parent / "captures")
).expanduser()
DEFAULT_HISTORY_TRUNCATION_LIMIT = int(os.environ.get("BFF_HISTORY_TRUNCATION_LIMIT", "11"))
DEFAULT_OLLAMA_TEMPERATURE = float(os.environ.get("BFF_OLLAMA_TEMPERATURE", "0.7"))
DEFAULT_OLLAMA_TOP_P = float(os.environ.get("BFF_OLLAMA_TOP_P", "0.9"))
DEFAULT_OLLAMA_TOP_K = int(os.environ.get("BFF_OLLAMA_TOP_K", "40"))
DEFAULT_OLLAMA_NUM_PREDICT = int(os.environ.get("BFF_OLLAMA_NUM_PREDICT", "100"))
DEFAULT_OLLAMA_NUM_CTX = int(os.environ.get("BFF_OLLAMA_NUM_CTX", "2048"))
DEFAULT_VLM_NUM_PREDICT = int(os.environ.get("BFF_VLM_NUM_PREDICT", "50"))
DEFAULT_OLLAMA_THINK_ENV = os.environ.get("BFF_OLLAMA_THINK", "false").lower()
DEFAULT_OLLAMA_THINK = DEFAULT_OLLAMA_THINK_ENV in ("true", "1", "yes", "on")
DEFAULT_REQUIRE_WAKEWORD_ENV = os.environ.get("BFF_REQUIRE_WAKEWORD", "false").lower()
DEFAULT_REQUIRE_WAKEWORD = DEFAULT_REQUIRE_WAKEWORD_ENV in ("true", "1", "yes", "on")
DEFAULT_WAKE_PHRASES_ENV = os.environ.get("BFF_WAKE_PHRASES", "ok snapper, okay snapper, hey snapper, snapper")
DEFAULT_OUTPUT_USB_KEYWORD = os.environ.get("BFF_OUTPUT_USB_KEYWORD", "USB")
DEFAULT_OUTPUT_BT_KEYWORD = os.environ.get("BFF_OUTPUT_BT_KEYWORD")
DEFAULT_OUTPUT_SAMPLE_RATE_ENV = os.environ.get("BFF_OUTPUT_SAMPLE_RATE")
DEFAULT_OUTPUT_SAMPLE_RATE = (
    int(DEFAULT_OUTPUT_SAMPLE_RATE_ENV) if DEFAULT_OUTPUT_SAMPLE_RATE_ENV else None
)
DEFAULT_PULSE_SINKS = os.environ.get("BFF_PULSE_SINKS")
DEFAULT_PULSE_COMBINED_SINK_NAME = os.environ.get(
    "BFF_PULSE_COMBINED_SINK_NAME", "bff_combined"
)
DEFAULT_PULSE_DEVICE_NAME = os.environ.get("BFF_PULSE_DEVICE_NAME", "pulse")
LAST_HEADSET_MAC = os.environ.get("LAST_HEADSET_MAC")
LAST_HEADSET_NAME = os.environ.get("LAST_HEADSET_NAME")

@dataclass
class ConversationConfig:
    """Runtime configuration for the voice chat assistant."""

    ollama_model: str = DEFAULT_OLLAMA_MODEL
    vlm_model: str = DEFAULT_VLM_MODEL
    whisper_model: str = DEFAULT_WHISPER_MODEL
    whisper_compute_type: str = "int8"  # optimized for Jetson
    piper_voice: Path | None = None
    piper_config: Path | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    sample_rate: int = DEFAULT_SAMPLE_RATE
    max_record_seconds: int = 20
    piper_length_scale: float | None = None
    piper_noise_scale: float | None = None
    piper_noise_w: float | None = None
    activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD
    silence_duration: float = DEFAULT_SILENCE_DURATION
    min_phrase_seconds: float = DEFAULT_MIN_PHRASE_SECONDS
    block_duration: float = DEFAULT_BLOCK_DURATION
    show_levels: bool = True
    input_device_keyword: str | None = DEFAULT_INPUT_DEVICE_KEYWORD
    input_device_index: int | None = None
    output_usb_keyword: str | None = DEFAULT_OUTPUT_USB_KEYWORD
    output_bt_keyword: str | None = DEFAULT_OUTPUT_BT_KEYWORD
    output_sample_rate: int | None = DEFAULT_OUTPUT_SAMPLE_RATE
    pulse_sinks: list[str] = field(default_factory=list)
    pulse_combined_sink_name: str = DEFAULT_PULSE_COMBINED_SINK_NAME
    pulse_device_name: str = DEFAULT_PULSE_DEVICE_NAME
    output_device_indices: list[str] = field(default_factory=list)
    interruptable: bool = DEFAULT_INTERRUPTABLE
    flush_on_interrupt: bool = DEFAULT_FLUSH_ON_INTERRUPT
    history_truncation_limit: int = DEFAULT_HISTORY_TRUNCATION_LIMIT
    ollama_temperature: float = DEFAULT_OLLAMA_TEMPERATURE
    ollama_top_p: float = DEFAULT_OLLAMA_TOP_P
    ollama_top_k: int = DEFAULT_OLLAMA_TOP_K
    ollama_num_predict: int = DEFAULT_OLLAMA_NUM_PREDICT
    ollama_num_ctx: int = DEFAULT_OLLAMA_NUM_CTX
    ollama_think: bool = DEFAULT_OLLAMA_THINK
    require_wakeword: bool = DEFAULT_REQUIRE_WAKEWORD
    wake_phrases: list[str] = field(default_factory=list)


@dataclass
class Scene:
    """Represents a conversational scene with a specific system prompt."""
    name: str
    system_prompt: str
    trigger: str | None = None


def parse_args() -> ConversationConfig:
    parser = argparse.ArgumentParser(description="Interactive voice chat assistant")
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama model name to use (default: %(default)s)",
    )
    parser.add_argument(
        "--vlm-model",
        default=DEFAULT_VLM_MODEL,
        help="Ollama model name to use for VLM scene captioning (default: %(default)s)",
    )
    parser.add_argument(
        "--ollama-think",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_OLLAMA_THINK,
        help="Enable/disable thinking/reasoning outputs for Ollama models (default: from env or false)",
    )
    parser.add_argument(
        "--require-wakeword",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REQUIRE_WAKEWORD,
        help="Require wake word to trigger LLM prompts (default: from env or false)",
    )
    parser.add_argument(
        "--wake-phrases",
        default=DEFAULT_WAKE_PHRASES_ENV,
        help="Comma-separated list of wake phrases (default: from env or 'ok snapper, okay snapper, hey snapper, snapper')",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help="Whisper model size to load (default: %(default)s)",
    )
    parser.add_argument(
        "--whisper-compute-type",
        default="int8",
        help="Quantization type for Whisper (default: int8, options: float16, int8_float16, int8)",
    )
    parser.add_argument(
        "--piper-voice",
        default=os.environ.get("BFF_PIPER_VOICE"),
        type=Path,
        help="Path to Piper voice model (*.onnx) (default: env BFF_PIPER_VOICE)",
    )
    parser.add_argument(
        "--piper-config",
        type=Path,
        help="Optional path to Piper voice config (*.json); defaults to <voice>.json",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt sent with each conversation",
    )
    parser.add_argument(
        "--max-record-seconds",
        type=int,
        default=20,
        help="Maximum seconds to record per turn (default: %(default)s)",
    )
    parser.add_argument(
        "--piper-length-scale",
        type=float,
        help="Override Piper config length_scale (lower=faster)",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=DEFAULT_PLAYBACK_SPEED,
        help="Audio playback speed multiplier (default: %(default)s). Overrides length_scale if length_scale is not set.",
    )
    parser.add_argument(
        "--piper-noise-scale",
        type=float,
        help="Override Piper config noise_scale",
    )
    parser.add_argument(
        "--piper-noise-w",
        type=float,
        help="Override Piper config noise_w",
    )
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=DEFAULT_ACTIVATION_THRESHOLD,
        help="RMS amplitude that starts a speech segment (default: %(default)s)",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=DEFAULT_SILENCE_THRESHOLD,
        help="RMS amplitude below which audio counts as silence (default: %(default)s)",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=DEFAULT_SILENCE_DURATION,
        help="Seconds of silence that end a speech segment (default: %(default)s)",
    )
    parser.add_argument(
        "--min-phrase-seconds",
        type=float,
        default=DEFAULT_MIN_PHRASE_SECONDS,
        help="Discard segments shorter than this many seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--block-duration",
        type=float,
        default=DEFAULT_BLOCK_DURATION,
        help="Processing block size in seconds for VAD (default: %(default)s)",
    )
    parser.add_argument(
        "--show-levels",
        action="store_true",
        default=True,
        help="Print live RMS level meter to stderr (default: enabled)",
    )
    parser.add_argument(
        "--no-show-levels",
        dest="show_levels",
        action="store_false",
        help="Disable live RMS level meter",
    )
    parser.add_argument(
        "--input-device-keyword",
        default=DEFAULT_INPUT_DEVICE_KEYWORD,
        help="Substring to match desired input device (default: %(default)s)",
    )
    parser.add_argument(
        "--output-usb-keyword",
        default=DEFAULT_OUTPUT_USB_KEYWORD,
        help="Substring to match USB output device (default: %(default)s)",
    )
    parser.add_argument(
        "--output-bt-keyword",
        default=DEFAULT_OUTPUT_BT_KEYWORD,
        help="Substring to match Bluetooth output device (default: env BFF_OUTPUT_BT_KEYWORD or headset name)",
    )
    parser.add_argument(
        "--output-sample-rate",
        type=int,
        default=DEFAULT_OUTPUT_SAMPLE_RATE,
        help="Override playback sample rate (default: env BFF_OUTPUT_SAMPLE_RATE or voice rate)",
    )
    parser.add_argument(
        "--pulse-sinks",
        default=DEFAULT_PULSE_SINKS,
        help="Comma-separated PulseAudio sink names to combine for output (default: env BFF_PULSE_SINKS)",
    )
    parser.add_argument(
        "--pulse-combined-sink-name",
        default=DEFAULT_PULSE_COMBINED_SINK_NAME,
        help="PulseAudio combined sink name (default: %(default)s)",
    )
    parser.add_argument(
        "--pulse-device-name",
        default=DEFAULT_PULSE_DEVICE_NAME,
        help="PulseAudio output device name for PortAudio (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Audio sample rate for recording and playback (default: %(default)s)",
    )
    parser.add_argument(
        "--no-interruptable",
        action="store_true",
        help="Disable interruptable behavior for Ollama queries and audio playback (default: from BFF_INTERRUPTABLE env or enabled)",
    )
    args = parser.parse_args()

    if args.piper_voice is None:
        parser.error("Piper voice model must be provided via --piper-voice or BFF_PIPER_VOICE")
    if not args.piper_voice.exists():
        parser.error(f"Piper voice model not found: {args.piper_voice}")

    input_keyword = args.input_device_keyword.strip() if args.input_device_keyword else None
    if input_keyword == "":
        input_keyword = None

    # Calculate length_scale from playback_speed if not explicitly provided
    length_scale = args.piper_length_scale
    if length_scale is None:
        # length_scale of 1.0 is normal speed. Lower is faster.
        # speed = 1.25 -> length_scale = 1/1.25 = 0.8
        speed = args.playback_speed
        if speed <= 0:
            speed = 1.0
        length_scale = 1.0 / speed

    pulse_sinks = [s.strip() for s in (args.pulse_sinks or "").split(",") if s.strip()]

    return ConversationConfig(
        ollama_model=args.ollama_model,
        vlm_model=args.vlm_model,
        whisper_model=args.whisper_model,
        whisper_compute_type=args.whisper_compute_type,
        piper_voice=args.piper_voice,
        piper_config=args.piper_config,
        system_prompt=args.system_prompt,
        sample_rate=args.sample_rate,
        max_record_seconds=args.max_record_seconds,
        piper_length_scale=length_scale,
        piper_noise_scale=args.piper_noise_scale,
        piper_noise_w=args.piper_noise_w,
        activation_threshold=args.activation_threshold,
        silence_threshold=args.silence_threshold,
        silence_duration=args.silence_duration,
        min_phrase_seconds=args.min_phrase_seconds,
        block_duration=args.block_duration,
        show_levels=args.show_levels,
        input_device_keyword=input_keyword,
        interruptable=False if args.no_interruptable else DEFAULT_INTERRUPTABLE,
        output_usb_keyword=args.output_usb_keyword.strip() if args.output_usb_keyword else None,
        output_bt_keyword=args.output_bt_keyword.strip() if args.output_bt_keyword else None,
        output_sample_rate=args.output_sample_rate,
        pulse_sinks=pulse_sinks,
        pulse_combined_sink_name=args.pulse_combined_sink_name,
        pulse_device_name=args.pulse_device_name,
        history_truncation_limit=DEFAULT_HISTORY_TRUNCATION_LIMIT,
        ollama_temperature=DEFAULT_OLLAMA_TEMPERATURE,
        ollama_top_p=DEFAULT_OLLAMA_TOP_P,
        ollama_top_k=DEFAULT_OLLAMA_TOP_K,
        ollama_num_predict=DEFAULT_OLLAMA_NUM_PREDICT,
        ollama_num_ctx=DEFAULT_OLLAMA_NUM_CTX,
        ollama_think=args.ollama_think,
        require_wakeword=args.require_wakeword,
        wake_phrases=[p.strip().lower() for p in args.wake_phrases.split(",") if p.strip()],
    )


def load_whisper_model(name: str, compute_type: str = "int8") -> WhisperModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Faster Whisper model '{name}' on {device} ({compute_type})…", file=sys.stderr)
    return WhisperModel(name, device=device, compute_type=compute_type)


def resolve_piper_config_path(model_path: Path, config_path: Path | None) -> Path:
    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"Piper config not found: {config_path}")
        return config_path

    candidate = model_path.with_suffix(model_path.suffix + ".json")
    if candidate.exists():
        return candidate

    alt = model_path.with_suffix(".json")
    if alt.exists():
        return alt

    raise FileNotFoundError(
        "Could not infer Piper config JSON. Provide --piper-config explicitly."
    )
    raise FileNotFoundError(
        "Could not infer Piper config JSON. Provide --piper-config explicitly."
    )


def load_scenes(script_path: Path) -> list[Scene]:
    """Load scenes from a JSON script file."""
    if not script_path.exists():
        print(f"Warning: Script file not found at {script_path}. Using default scene only.", file=sys.stderr)
        return []
        
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        scenes = []
        for item in data:
            scenes.append(Scene(
                name=item["name"],
                system_prompt=item["system_prompt"],
                trigger=item.get("trigger")
            ))
        print(f"Loaded {len(scenes)} scenes from {script_path}", file=sys.stderr)
        return scenes
    except Exception as e:
        print(f"Error loading script file: {e}", file=sys.stderr)
        return []

def load_piper_voice(
    model_path: Path,
    config_path: Path | None,
    *,
    length_scale: float | None = None,
    noise_scale: float | None = None,
    noise_w: float | None = None,
) -> PiperVoice:
    resolved = resolve_piper_config_path(model_path, config_path)
    with open(resolved, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    overrides = {
        "length_scale": length_scale,
        "noise_scale": noise_scale,
        "noise_w": noise_w,
    }

    applied = {k: v for k, v in overrides.items() if v is not None}
    tmp_path: Path | None = None

    if applied:
        config_data.update(applied)
        # Some Piper voices (e.g. aru) have params inside an 'inference' block
        if "inference" in config_data:
             config_data["inference"].update(applied)

        tmp_file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        tmp_path = Path(tmp_file.name)
        json.dump(config_data, tmp_file)
        tmp_file.flush()
        tmp_file.close()
        config_to_use = tmp_path
        print(
            "Loading Piper voice '{}' with overrides {}".format(
                model_path.name,
                ", ".join(f"{k}={v}" for k, v in applied.items()),
            ),
            file=sys.stderr,
        )
    else:
        config_to_use = resolved
        config_to_use = resolved
        # print(
        #     f"Loading Piper voice '{model_path.name}' with config '{resolved.name}'…",
        #     file=sys.stderr,
        # )

    try:
        voice = PiperVoice.load(str(model_path), config_path=str(config_to_use))
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return voice


def ensure_log_dir() -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    return LOG_ROOT


def is_jetson() -> bool:
    """Detect Jetson hardware via the Tegra release marker file NVIDIA ships on L4T."""
    return Path("/etc/nv_tegra_release").exists()


def is_process_running(pattern: str) -> bool:
    """Check whether a process matching `pattern` (as in `pgrep -f`) is currently running."""
    try:
        result = subprocess.run(["pgrep", "-f", pattern], capture_output=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def append_log_line(log_path: Path, payload: dict[str, Any]) -> None:
    record = {"timestamp": datetime.now().isoformat(), **payload}
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def meter_break(show_levels: bool) -> None:
    if show_levels:
        sys.stderr.write("\n") #\n
        sys.stderr.flush()


def check_bluetooth_connection_status(mac: str) -> bool:
    """Check if a Bluetooth device is connected."""
    if sys.platform == "darwin":
        if LAST_HEADSET_NAME:
            try:
                for device in sd.query_devices():
                    if LAST_HEADSET_NAME.lower() in device.get("name", "").lower():
                        return True
            except Exception:
                pass
        return False

    try:
        result = subprocess.run(
            ["bluetoothctl", "info", mac],
            capture_output=True,
            text=True,
            check=False,
        )
        return "Connected: yes" in result.stdout
    except Exception:
        return False


def find_pulseaudio_card_by_mac(mac: str, max_retries: int = 5, retry_delay: float = 1.0) -> str | None:
    """Find PulseAudio card by MAC address, with retries."""
    for attempt in range(max_retries):
        pactl_result = subprocess.run(
            ["pactl", "list", "cards", "short"],
            capture_output=True,
            text=True,
            check=False,
        )
        
        for line in pactl_result.stdout.splitlines():
            if mac.lower() in line.lower():
                card_id = line.split()[0]
                return card_id
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    return None


def try_auto_connect_headset(mac: str, name: str) -> bool:
    """Attempt to auto-connect to a saved headset."""
    if not mac or not name:
        return False
    
    print(f"Attempting to auto-connect to last headset: {name} ({mac})", file=sys.stderr)
    
    try:
        # Check if already connected
        if check_bluetooth_connection_status(mac):
            print(f"Device {name} ({mac}) is already connected.", file=sys.stderr)
            # Try to find and configure PulseAudio card
            card_id = find_pulseaudio_card_by_mac(mac, max_retries=3, retry_delay=0.5)
            if card_id:
                subprocess.run(
                    ["pactl", "set-card-profile", card_id, "headset-head-unit"],
                    capture_output=True,
                    check=False,
                )
                print(f"Auto-connect successful! {name} is connected and configured.", file=sys.stderr)
                return True
            else:
                print(f"Device is connected via Bluetooth, but PulseAudio card not yet available.", file=sys.stderr)
                print(f"This is normal - the device should work as the system default audio device.", file=sys.stderr)
                return True
        
        # Initialize Bluetooth
        subprocess.run(["bluetoothctl", "power", "on"], capture_output=True, check=False)
        subprocess.run(["bluetoothctl", "agent", "on"], capture_output=True, check=False)
        subprocess.run(["bluetoothctl", "default-agent"], capture_output=True, check=False)
        
        # Trust and connect
        bluetoothctl_input = f"trust {mac}\nconnect {mac}\n"
        result = subprocess.run(
            ["bluetoothctl"],
            input=bluetoothctl_input.encode(),
            capture_output=True,
            check=False,
            timeout=10,
        )
        
        # Wait for connection to establish and check status
        for attempt in range(5):
            time.sleep(1)
            if check_bluetooth_connection_status(mac):
                break
        else:
            print(f"Auto-connect failed: Could not establish Bluetooth connection to {name} ({mac}).", file=sys.stderr)
            return False
        
        # Try to find and configure PulseAudio card (with retries)
        card_id = find_pulseaudio_card_by_mac(mac, max_retries=5, retry_delay=1.0)
        if card_id:
            subprocess.run(
                ["pactl", "set-card-profile", card_id, "headset-head-unit"],
                capture_output=True,
                check=False,
            )
            print(f"Auto-connect successful! Connected {name} ({mac}) in headset (HFP/HSP) mode.", file=sys.stderr)
            return True
        else:
            print(f"Bluetooth connection established to {name} ({mac}), but PulseAudio card not yet available.", file=sys.stderr)
            print(f"This is normal - the device should work as the system default audio device.", file=sys.stderr)
            return True
        
    except Exception as exc:
        print(f"Auto-connect error: {exc}", file=sys.stderr)
        return False


def scan_bluetooth_devices() -> list[tuple[str, str]]:
    """Scan for Bluetooth devices and return list of (mac, name) tuples."""
    devices: list[tuple[str, str]] = []
    
    try:
        # Initialize Bluetooth
        subprocess.run(["bluetoothctl", "power", "on"], capture_output=True, check=False)
        subprocess.run(["bluetoothctl", "agent", "on"], capture_output=True, check=False)
        subprocess.run(["bluetoothctl", "default-agent"], capture_output=True, check=False)
        
        # Start scanning
        scan_process = subprocess.Popen(
            ["bluetoothctl", "scan", "on"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Wait for devices to be discovered
        time.sleep(5)
        
        # Stop scanning
        scan_process.terminate()
        subprocess.run(["bluetoothctl", "scan", "off"], capture_output=True, check=False)
        
        # Get list of devices
        result = subprocess.run(
            ["bluetoothctl", "devices"],
            capture_output=True,
            text=True,
            check=False,
        )
        
        # Filter for audio devices first
        audio_devices = []
        all_devices = []
        
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.split(None, 2)
            if len(parts) >= 3:
                mac = parts[1]
                name = parts[2]
                device_tuple = (mac, name)
                all_devices.append(device_tuple)
                # Check if it's an audio device
                name_lower = name.lower()
                if any(keyword in name_lower for keyword in ["headset", "audio", "speaker", "earbuds", "airpods"]):
                    audio_devices.append(device_tuple)
        
        # Return audio devices if found, otherwise all devices
        return audio_devices if audio_devices else all_devices
    except Exception as exc:
        print(f"Bluetooth scan error: {exc}", file=sys.stderr)
        return []


def connect_to_headset(mac: str, name: str) -> bool:
    """Connect to a specific Bluetooth headset."""
    try:
        print(f"Connecting to: {name} ({mac})", file=sys.stderr)
        
        # Check if already connected
        if check_bluetooth_connection_status(mac):
            print(f"Device {name} ({mac}) is already connected via Bluetooth.", file=sys.stderr)
        else:
            # Trust and connect
            bluetoothctl_input = f"trust {mac}\nconnect {mac}\n"
            subprocess.run(
                ["bluetoothctl"],
                input=bluetoothctl_input.encode(),
                capture_output=True,
                check=False,
                timeout=10,
            )
            
            # Wait for connection to establish and verify
            for attempt in range(5):
                time.sleep(1)
                if check_bluetooth_connection_status(mac):
                    break
            else:
                print(f"Warning: Could not verify Bluetooth connection to {name} ({mac}).", file=sys.stderr)
        
        # Try to find and configure PulseAudio card (with retries)
        card_id = find_pulseaudio_card_by_mac(mac, max_retries=5, retry_delay=1.0)
        if card_id:
            subprocess.run(
                ["pactl", "set-card-profile", card_id, "headset-head-unit"],
                capture_output=True,
                check=False,
            )
            print(f"Connected {name} ({mac}) in headset (HFP/HSP) mode.", file=sys.stderr)
        else:
            print(f"Bluetooth connection established to {name} ({mac}), but PulseAudio card not yet available.", file=sys.stderr)
            print(f"This is normal - the device should work as the system default audio device.", file=sys.stderr)
        
        # Save to environment (will be saved to .env by caller)
        os.environ["LAST_HEADSET_MAC"] = mac
        os.environ["LAST_HEADSET_NAME"] = name
        return True
    except Exception as exc:
        print(f"Connection error: {exc}", file=sys.stderr)
        return False


def ensure_headset_connected() -> None:
    """Ensure a Bluetooth headset is connected, auto-connecting or prompting user if needed."""
    if sys.platform == "darwin":
        if LAST_HEADSET_NAME:
            print(f"On macOS, assuming headset '{LAST_HEADSET_NAME}' is managed by the OS.", file=sys.stderr)
            print("Please ensure it is connected in macOS Bluetooth Settings.", file=sys.stderr)
        return

    # Try auto-connect first
    if LAST_HEADSET_MAC and LAST_HEADSET_NAME:
        if try_auto_connect_headset(LAST_HEADSET_MAC, LAST_HEADSET_NAME):
            return
        # Auto-connect failed, but device might still be available - continue to scan
        print("Auto-connect did not succeed. Scanning for available devices...", file=sys.stderr)
        print("", file=sys.stderr)
    
    # Scan for devices
    print("Scanning for Bluetooth devices...", file=sys.stderr)
    devices = scan_bluetooth_devices()
    
    if not devices:
        print("No Bluetooth devices found.", file=sys.stderr)
        return
    
    # Display menu
    print("", file=sys.stderr)
    print("Available Bluetooth devices:", file=sys.stderr)
    print("=" * 28, file=sys.stderr)
    for idx, (mac, name) in enumerate(devices, 1):
        print(f"{idx}) {name} ({mac})", file=sys.stderr)
    
    # Get user selection
    print("", file=sys.stderr)
    try:
        selection = input("Select a device (1-{}) or 'q' to quit: ".format(len(devices)))
        if selection.lower() == 'q':
            print("Cancelled.", file=sys.stderr)
            return
        
        idx = int(selection)
        if idx < 1 or idx > len(devices):
            print("Invalid selection.", file=sys.stderr)
            return
        
        mac, name = devices[idx - 1]
        if connect_to_headset(mac, name):
            # Save to .env file (look for it in current dir or parent dirs, like dotenv does)
            env_path = None
            current = Path.cwd()
            for parent in [current] + list(current.parents):
                candidate = parent / ".env"
                if candidate.exists():
                    env_path = candidate
                    break
            
            # If not found, use .env in current directory
            if env_path is None:
                env_path = Path(".env")
            
            # Read existing .env or create new content
            if env_path.exists():
                with open(env_path, "r") as f:
                    content = f.read()
                lines = content.splitlines()
            else:
                lines = []
            
            # Update or add headset info
            updated_mac = False
            updated_name = False
            for i, line in enumerate(lines):
                if line.startswith("LAST_HEADSET_MAC="):
                    lines[i] = f'LAST_HEADSET_MAC={mac}'
                    updated_mac = True
                elif line.startswith("LAST_HEADSET_NAME="):
                    lines[i] = f'LAST_HEADSET_NAME="{name}"'
                    updated_name = True
            
            if not updated_mac:
                lines.append(f"LAST_HEADSET_MAC={mac}")
            if not updated_name:
                lines.append(f'LAST_HEADSET_NAME="{name}"')
            
            with open(env_path, "w") as f:
                f.write("\n".join(lines) + "\n")
    except (ValueError, KeyboardInterrupt, EOFError):
        print("Cancelled or invalid input.", file=sys.stderr)


def find_input_device(keyword: str, min_channels: int = 1) -> int | None:
    keyword_lower = keyword.lower()
    for idx, device in enumerate(sd.query_devices()):
        name = device.get("name", "")
        if keyword_lower in name.lower() and device.get("max_input_channels", 0) >= min_channels:
            return idx
    return None


def is_pulseaudio_available() -> bool:
    if sys.platform == "darwin":
        return False
    import shutil
    return shutil.which("pactl") is not None


def find_output_device_by_keyword(keyword: str, min_channels: int = 1) -> str | None:
    if not keyword:
        return None
    keyword_lower = keyword.lower()
    try:
        for device in sd.query_devices():
            name = device.get("name", "")
            if keyword_lower in name.lower() and device.get("max_output_channels", 0) >= min_channels:
                return name
    except Exception:
        pass
    return None


def ensure_pulse_combined_sink(sinks: list[str], sink_name: str) -> str | None:
    if not sinks:
        return None

    try:
        modules = subprocess.run(
            ["pactl", "list", "short", "modules"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in modules.stdout.splitlines():
            if "module-combine-sink" in line and f"sink_name={sink_name}" in line:
                return sink_name
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                "pactl",
                "load-module",
                "module-combine-sink",
                f"sink_name={sink_name}",
                f"slaves={','.join(sinks)}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return sink_name
        print(
            f"PulseAudio combine-sink error: {result.stderr.strip()}",
            file=sys.stderr,
        )
    except Exception as exc:
        print(f"PulseAudio combine-sink error: {exc}", file=sys.stderr)

    return None


def list_pulse_sinks() -> list[str]:
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []

    sinks: list[str] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            sinks.append(parts[1])
    return sinks


def auto_detect_pulse_sinks(
    usb_keyword: str | None,
    bt_keyword: str | None,
) -> list[str]:
    sinks = list_pulse_sinks()
    if not sinks:
        return []

    usb_matches: list[str] = []
    bt_matches: list[str] = []
    bluez_matches: list[str] = []

    for sink in sinks:
        sink_lower = sink.lower()
        if usb_keyword and usb_keyword.lower() in sink_lower:
            usb_matches.append(sink)
        if bt_keyword and bt_keyword.lower() in sink_lower:
            bt_matches.append(sink)
        if "bluez_sink" in sink_lower:
            bluez_matches.append(sink)

    # Prioritize USB audio output, only fall back to Bluetooth if USB is not available
    if usb_matches:
        return usb_matches
    if bt_matches:
        return bt_matches
    if bluez_matches:
        return bluez_matches
    return []


def set_default_pulse_sink(sink_name: str) -> None:
    try:
        subprocess.run(
            ["pactl", "set-default-sink", sink_name],
            capture_output=True,
            check=False,
        )
    except Exception:
        pass


def resolve_output_devices(config: ConversationConfig) -> list[str]:
    # 1. If PulseAudio is available, use the PulseAudio setup.
    if is_pulseaudio_available():
        output_indices: list[str] = []
        pulse_sinks = config.pulse_sinks
        if not pulse_sinks:
            pulse_sinks = auto_detect_pulse_sinks(
                config.output_usb_keyword, config.output_bt_keyword
            )
        combined_sink = ensure_pulse_combined_sink(
            pulse_sinks, config.pulse_combined_sink_name
        )
        if combined_sink:
            set_default_pulse_sink(combined_sink)
        output_indices.append(config.pulse_device_name)
        return output_indices

    # 2. On macOS/non-PulseAudio platforms, search for matched hardware output devices.
    if sys.platform == "darwin":
        return []

    matched_devices = []
    seen = set()
    
    if config.output_bt_keyword:
        name = find_output_device_by_keyword(config.output_bt_keyword)
        if name and name not in seen:
            matched_devices.append(name)
            seen.add(name)
            
    if config.output_usb_keyword:
        name = find_output_device_by_keyword(config.output_usb_keyword)
        if name and name not in seen:
            matched_devices.append(name)
            seen.add(name)
            
    return matched_devices


def log_audio_devices() -> None:
    """Print full list of audio devices with channel info."""
    try:
        devices = sd.query_devices()
    except Exception as exc:
        print(f"Audio device query failed: {exc}", file=sys.stderr)
        return

    try:
        default_input, default_output = sd.default.device
    except Exception:
        default_input, default_output = None, None

    print("", file=sys.stderr)
    print("Available audio devices:", file=sys.stderr)
    print("=" * 26, file=sys.stderr)
    for idx, device in enumerate(devices):
        name = device.get("name", "unknown")
        max_in = device.get("max_input_channels", 0)
        max_out = device.get("max_output_channels", 0)
        default_flags = []
        if default_input is not None and idx == default_input:
            default_flags.append("default input")
        if default_output is not None and idx == default_output:
            default_flags.append("default output")
        suffix = f" [{', '.join(default_flags)}]" if default_flags else ""
        print(f"{idx}) {name} (in={max_in}, out={max_out}){suffix}", file=sys.stderr)
    print("", file=sys.stderr)


def rms_amplitude(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block))))


def resample_audio(
    data: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    if source_rate == target_rate:
        return data
    ratio = target_rate / source_rate
    new_length = max(1, int(round(data.shape[0] * ratio)))
    x_old = np.linspace(0.0, 1.0, num=data.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=new_length, endpoint=False)

    if data.ndim == 1:
        return np.interp(x_new, x_old, data).astype(np.float32)

    channels = data.shape[1]
    resampled = np.empty((new_length, channels), dtype=np.float32)
    for ch in range(channels):
        resampled[:, ch] = np.interp(x_new, x_old, data[:, ch])
    return resampled


class UnifiedAudioSystem:
    def __init__(self):
        self.active_buffer: np.ndarray = np.array([], dtype=np.float32)
        self.buffer_mutex = threading.Lock()
        self.interrupt_event = threading.Event()
        self.is_playing = False

    def play_chunk(self, chunk: np.ndarray):
        with self.buffer_mutex:
            if chunk.ndim > 1:
                chunk = chunk.flatten()
            self.active_buffer = np.concatenate((self.active_buffer, chunk.astype(np.float32)))
            self.is_playing = True

    def clear(self):
        with self.buffer_mutex:
            self.active_buffer = np.array([], dtype=np.float32)
            self.is_playing = False

    def fill_output(self, outdata: np.ndarray, frames: int):
        with self.buffer_mutex:
            if self.interrupt_event.is_set():
                self.active_buffer = np.array([], dtype=np.float32)
                self.is_playing = False
                self.interrupt_event.clear()

            n_samples = len(self.active_buffer)
            if n_samples == 0:
                outdata.fill(0.0)
                self.is_playing = False
                return

            if n_samples >= frames:
                outdata[:, 0] = self.active_buffer[:frames]
                self.active_buffer = self.active_buffer[frames:]
            else:
                outdata[:n_samples, 0] = self.active_buffer
                outdata[n_samples:, 0] = 0.0
                self.active_buffer = np.array([], dtype=np.float32)
                self.is_playing = False


unified_audio = UnifiedAudioSystem()


def phrase_stream(
    config: ConversationConfig, 
    stop_event: threading.Event | None = None,
    on_voice_activity: Callable[[], None] | None = None,
) -> Iterable[np.ndarray]:
    """Yield successive speech segments detected from the microphone.
    
    Args:
        config: Conversation configuration
        stop_event: Event to stop the stream
        on_voice_activity: Callback called immediately when voice activity is detected
    """

    channels = 1
    block_size = max(1, int(config.sample_rate * config.block_duration))
    silence_blocks_required = max(1, int(config.silence_duration / config.block_duration))
    max_blocks = max(1, int(config.max_record_seconds / config.block_duration))
    min_blocks = max(1, int(config.min_phrase_seconds / config.block_duration))

    q: queue.Queue[np.ndarray] = queue.Queue()

    def audio_callback(indata, outdata, frames, time_info, status):
        # if status:
        #     print(f"[vad] {status}", file=sys.stderr)
        q.put(indata.copy())
        unified_audio.fill_output(outdata, frames)

    print("Listening continuously… (Ctrl+C to exit)")
    with sd.Stream(
        samplerate=config.sample_rate,
        channels=(channels, 1),
        dtype="float32",
        blocksize=block_size,
        callback=audio_callback,
        device=(
            config.input_device_index,
            config.output_device_indices[0] if config.output_device_indices else None
        ),
    ):
        recording = False
        silence_blocks = 0
        collected: List[np.ndarray] = []
        block_counter = 0

        while True:
            if stop_event and stop_event.is_set():
                break
            try:
                block = q.get(timeout=0.1)
            except queue.Empty:
                continue
            block_counter += 1 if recording else 0
            amp = rms_amplitude(block)

            if config.show_levels:
                meter_width = 40
                normalized = min(1.0, amp / max(config.activation_threshold, 1e-6))
                filled = int(normalized * meter_width)
                bar = "#" * filled + "-" * (meter_width - filled)
                suffix = "REC"
                if recording and amp >= config.activation_threshold:
                    suffix = "REC (*)"
                sys.stderr.write(
                    f"\rLevel {amp:0.3f} |{bar}| {suffix}"
                )
                sys.stderr.flush()

            if not recording:
                if amp >= config.activation_threshold:
                    # Voice activity detected - pause immediately
                    if on_voice_activity:
                        on_voice_activity()
                    recording = True
                    collected = [block]
                    silence_blocks = 0
                    block_counter = 1
            else:
                collected.append(block)
                if amp < config.silence_threshold:
                    silence_blocks += 1
                else:
                    silence_blocks = 0

                if silence_blocks >= silence_blocks_required or block_counter >= max_blocks:
                    duration = len(collected) * config.block_duration
                    recording = False
                    silence_blocks = 0
                    block_counter = 0

                    if len(collected) < min_blocks:
                        print("Discarded short segment.", file=sys.stderr)
                        collected = []
                        meter_break(config.show_levels)
                        continue

                    audio = np.concatenate(collected, axis=0)
                    collected = []
                    meter_break(config.show_levels)
                    yield audio



def transcribe_audio(model: WhisperModel, audio_path: Path, show_levels: bool) -> str:
    print("Transcribing with Faster Whisper…", file=sys.stderr)
    start_time = time.perf_counter()
    
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=1,  # Greedy decoding for speed
        temperature=0,
    )
    
    # faster-whisper returns a generator, so we must iterate to get results
    text_segments = []
    for segment in segments:
        text_segments.append(segment.text)
        
    text = " ".join(text_segments).strip()
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    # print(f"Whisper transcription completed in {duration:.2f} seconds", file=sys.stderr)
    meter_break(show_levels)
    print(f"You said: {text}")
    return text


class SentenceAccumulator:
    """Accumulates text chunks and yields complete sentences."""
    def __init__(self):
        self.buffer = ""
        # Only split on clear sentence endings.
        self.endings = {'.', '!', '?'}
        
    def add(self, text: str) -> Iterable[str]:
        self.buffer += text
        while True:
            # Find the first sentence ending
            earliest_end = -1
            best_mark = None
            
            for mark in self.endings:
                start_search = 0
                while True:
                    idx = self.buffer.find(mark, start_search)
                    if idx == -1:
                        break
                    # Ignore dots preceded by digits (decimals/list numbers like 1. or 3.14)
                    if mark == '.' and idx > 0 and self.buffer[idx - 1].isdigit():
                        start_search = idx + 1
                        continue
                    # Ignore dots preceded by common abbreviations
                    if mark == '.' and idx > 0:
                        words = self.buffer[:idx].split()
                        if words:
                            last_word = words[-1].lower().strip(" \t\n\r*_-")
                            if last_word in ["mr", "ms", "mrs", "dr", "vs", "eg", "ie"]:
                                start_search = idx + 1
                                continue
                    if earliest_end == -1 or idx < earliest_end:
                        earliest_end = idx
                        best_mark = mark
                    break
            
            if earliest_end == -1:
                break
                
            candidate = self.buffer[:earliest_end+1]
            remainder = self.buffer[earliest_end+1:]
            
            yield candidate.strip()
            self.buffer = remainder

    def flush(self) -> Iterable[str]:
        if self.buffer.strip():
            yield self.buffer.strip()
        self.buffer = ""


def query_ollama_streaming(
    model_name: str,
    messages: list[dict[str, str]],
    interruptable: bool = True,
    stop_event: threading.Event | None = None,
    options: dict[str, Any] | None = None,
    think: bool = False,
    print_stream: bool = True,
) -> Iterable[str]:
    """
    Yields complete sentences from Ollama.
    """
    client = ollama.Client()
    
    # Extract just the new user message for logging
    user_content = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "...")
    print(f"Querying Ollama '{model_name}': {user_content[:60]}...", file=sys.stderr)

    try:
        stream = client.chat(
            model=model_name,
            messages=messages,
            stream=True,
            keep_alive=-1,
            think=think,
            options=options or {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 100,
                "num_ctx": 2048,
            },
        )
    except Exception as e:
        print(f"Ollama error: {e}", file=sys.stderr)
        return

    accumulator = SentenceAccumulator()
    first_chunk = True
    in_thinking = False

    for chunk in stream:
        if stop_event and stop_event.is_set():
            # print("Ollama query interrupted.", file=sys.stderr)
            return

        content = ""
        thinking = ""
        # Extract content from various chunk formats
        if isinstance(chunk, dict):
            msg = chunk.get("message", {})
            content = msg.get("content", "")
            thinking = msg.get("thinking", "")
        else:
             # Object access
            msg = getattr(chunk, "message", None)
            if msg:
                content = getattr(msg, "content", "")
                thinking = getattr(msg, "thinking", "")
        
        text_to_process = ""
        is_thought = False
        if content:
            text_to_process = content
        elif think and thinking:
            text_to_process = thinking
            is_thought = True

        if text_to_process:
            if print_stream:
                if first_chunk:
                    sys.stdout.write("Assistant: ")
                    sys.stdout.flush()
                    first_chunk = False
                
                # Apply ANSI styling: gray italics for thinking blocks
                if is_thought and not in_thinking:
                    sys.stdout.write("\033[3;90m")
                    sys.stdout.flush()
                    in_thinking = True
                elif not is_thought and in_thinking:
                    sys.stdout.write("\033[0m")
                    sys.stdout.flush()
                    in_thinking = False
                
                sys.stdout.write(text_to_process)
                sys.stdout.flush()
            for sentence in accumulator.add(text_to_process):
                yield sentence
    
    if print_stream and in_thinking:
        sys.stdout.write("\033[0m")
        sys.stdout.flush()
    
    for sentence in accumulator.flush():
        yield sentence


def synthesize_with_piper(
    voice: PiperVoice, text: str, output_wav: Path
) -> None:
    print("Synthesizing speech with Piper…", file=sys.stderr)
    audio_iter = voice.synthesize(text)
    base_sample_rate = int(
        getattr(voice, "sample_rate", getattr(getattr(voice, "config", {}), "sample_rate", DEFAULT_SAMPLE_RATE))
    )

    def extract_audio_field(obj: Any) -> Any | None:
        field_candidates = (
            "audio",
            "_audio",
            "buffer",
            "data",
            "pcm",
            "samples",
            "wave",
            "waveform",
            "frames",
            "chunk",
            "audio_int16_bytes",
            "audio_int16_array",
            "audio_float_array",
            "_audio_int16_bytes",
            "_audio_int16_array",
        )
        for attr in field_candidates:
            value = getattr(obj, attr, None)
            if value is not None:
                return value
        return None

    def to_bytes_and_rate(chunk: Any) -> tuple[bytes, int | None]:
        current_rate: int | None = None
        data: Any = chunk

        if AudioChunk is not None and isinstance(chunk, AudioChunk):
            maybe = extract_audio_field(chunk)
            if maybe is not None:
                data = maybe
            current_rate = getattr(chunk, "sample_rate", None)
        elif isinstance(chunk, dict):
            if "audio" in chunk:
                data = chunk["audio"]
            else:
                for key in ("buffer", "data", "pcm", "samples"):
                    if key in chunk:
                        data = chunk[key]
                        break
            current_rate = chunk.get("sample_rate")
        else:
            maybe = extract_audio_field(chunk)
            if maybe is not None:
                data = maybe
                current_rate = getattr(chunk, "sample_rate", None)

        if isinstance(data, np.ndarray):
            return data.astype(np.int16).tobytes(), current_rate
        if isinstance(data, (bytes, bytearray, memoryview)):
            return bytes(data), current_rate
        if isinstance(data, (tuple, list)) and data:
            first = data[0]
            if isinstance(first, np.ndarray):
                return first.astype(np.int16).tobytes(), current_rate
            if isinstance(first, (bytes, bytearray, memoryview)):
                return bytes(first), current_rate
        if data is chunk and hasattr(chunk, "__iter__") and not isinstance(
            chunk, (str, bytes, bytearray, memoryview)
        ):
            try:
                arr = np.fromiter(chunk, dtype=np.int16)
                return arr.tobytes(), current_rate
            except TypeError:
                pass

        # Fall back to generic bytes conversion if possible
        try:
            return bytes(data), current_rate
        except Exception as exc:
            raise TypeError(
                f"Unsupported Piper chunk type: {type(chunk)!r} (available attrs: {dir(chunk)})"
            ) from exc

    with wave.open(str(output_wav), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(base_sample_rate)

        for chunk in audio_iter:
            data, maybe_rate = to_bytes_and_rate(chunk)
            if maybe_rate and maybe_rate != base_sample_rate:
                wav_file.setframerate(maybe_rate)
            wav_file.writeframes(data)


def _is_punctuation_only(text: str) -> bool:
    """Return True if text is empty or contains only punctuation/whitespace (no speech content)."""
    t = text.strip()
    if not t:
        return True
    word_chars_only = re.sub(r"[^\w]", "", t, flags=re.ASCII)  # keep only letters, digits, underscore
    return len(word_chars_only) == 0


class TTSWorker:
    """
    Handles background TTS synthesis and audio buffering.
    Input: Text sentences
    Output: Audio chunks in a thread-safe queue for the player
    """
    def __init__(self, voice: PiperVoice, sample_rate: int):
        self.voice = voice
        self.sample_rate = sample_rate
        self.input_queue: queue.Queue[str | None] = queue.Queue()
        self.audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()  # For pausing synthesis
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.started = False

    def start(self):
        if not self.started:
            self.stop_event.clear()
            self.pause_event.clear()
            self.thread.start()
            self.started = True

    def stop(self):
        self.stop_event.set()
        self.pause_event.clear()  # Unpause to allow thread to exit
        # Drain queues to unblock
        while not self.input_queue.empty():
            try: self.input_queue.get_nowait()
            except queue.Empty: pass
        self.input_queue.put(None) # Sentinel

    def pause(self):
        """Pause TTS synthesis (stops processing new text, but keeps worker alive)"""
        self.pause_event.set()

    def resume(self):
        """Resume TTS synthesis"""
        self.pause_event.clear()

    def flush(self):
        """Flush all queued audio chunks and pending text"""
        # Clear audio queue
        while not self.audio_queue.empty():
            try: 
                self.audio_queue.get_nowait()
            except queue.Empty: 
                pass
        # Clear input queue
        while not self.input_queue.empty():
            try: 
                self.input_queue.get_nowait()
            except queue.Empty: 
                pass

    def put_text(self, text: str):
        self.input_queue.put(text)

    def _worker_loop(self):
        while not self.stop_event.is_set():
            # Check if paused
            if self.pause_event.is_set():
                time.sleep(0.1)
                continue
                
            try:
                text = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if text is None:
                break

            # Synthesize
            try:
                # print(f"[TTS] Synthesizing: {text[:30]}...", file=sys.stderr)
                stream = self.voice.synthesize(text)
                for chunk in stream:
                    if self.stop_event.is_set():
                        break
                    # Check if paused during synthesis
                    if self.pause_event.is_set():
                        break
                    
                    audio_array = None
                    
                    # 1. Try to get float array directly (most efficient)
                    if hasattr(chunk, "audio_float_array") and chunk.audio_float_array is not None:
                        audio_array = chunk.audio_float_array.astype(np.float32)
                    
                    # 2. Try to get bytes
                    elif hasattr(chunk, "audio_int16_bytes") and chunk.audio_int16_bytes is not None:
                         audio_data = chunk.audio_int16_bytes
                         audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    elif hasattr(chunk, "bytes") and chunk.bytes is not None:
                         audio_data = chunk.bytes
                         audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                     # 3. Fallback to generic bytes conversion
                    else:
                        try:
                            audio_data = bytes(chunk)
                            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        except Exception:
                            pass

                    if audio_array is None:
                         print(f"TTS Warning: Could not extract audio from {type(chunk)}: {dir(chunk)}", file=sys.stderr)
                         continue

                    self.audio_queue.put(audio_array)
            except Exception as e:
                print(f"TTS Error: {e}", file=sys.stderr)
        
        self.audio_queue.put(None) # End of audio stream


def play_audio_stream(
    audio_queue: queue.Queue[np.ndarray | None],
    sample_rate: int,
    interrupt_event: threading.Event,
    interruptable: bool = True,
    save_path: Path | None = None,
    output_device_indices: list[str] | None = None,
    output_sample_rate: int | None = None,
) -> None:
    try:
        wav_file = None
        if save_path:
            wav_file = wave.open(str(save_path), "wb")
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2) # 16-bit PCM
            wav_file.setframerate(sample_rate)

        target_rate = 16000
        
        while True:
            if interruptable and interrupt_event.is_set():
                unified_audio.clear()
                break

            try:
                chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk is None:
                # End of stream
                break

            # Process chunk for playing
            play_chunk = chunk.copy()
            if sample_rate != target_rate:
                play_chunk = resample_audio(play_chunk, sample_rate, target_rate)

            # Write chunk to unified audio system
            unified_audio.play_chunk(play_chunk)

            # Write original chunk to wav file if needed
            if wav_file:
                if chunk.ndim == 1:
                    chunk = chunk[:, np.newaxis]
                clipped = np.clip(chunk, -1.0, 1.0)
                int16_data = (clipped * 32767).astype(np.int16)
                wav_file.writeframes(int16_data.tobytes())

        if wav_file:
            wav_file.close()
                
    except Exception as e:
        print(f"Playback Error: {e}", file=sys.stderr)


def play_audio(
    audio_path: Path,
    interrupt_event: threading.Event,
    interruptable: bool = True,
    output_device_indices: list[str] | None = None,
    output_sample_rate: int | None = None,
) -> bool:
    try:
        data, samplerate = sf.read(audio_path, dtype="float32")
        
        # Always resample to 16000Hz (the bidirectional stream rate)
        target_rate = 16000
        if samplerate != target_rate:
            data = resample_audio(data, samplerate, target_rate)
            samplerate = target_rate
            
        if data.ndim > 1:
            data = data.mean(axis=1) # mix down to mono
            
        interrupt_event.clear()
        
        # Clear any active playback and queue the new sound
        unified_audio.clear()
        unified_audio.play_chunk(data)
        
        # Wait until playback is finished or interrupted
        while unified_audio.is_playing:
            if interruptable and interrupt_event.is_set():
                unified_audio.clear()
                return False
            time.sleep(0.05)
            
        return True
    except Exception as e:
        print(f"Playback Error: {e}", file=sys.stderr)
        return False


def is_lets_stop_command(text: str) -> bool:
    """Check if the transcribed text is a command to exit the program. Requires 'snapper' in input."""
    text_lower = text.lower().strip()
    # if "snapper" not in text_lower:
        # return False
    stop_phrases = [
        "let's stop the conversation",
        "lets stop the conversation",
        "i'd like to end the conversation",
        "i would like to end the conversation",
        "id like to end the conversation",
        "let's stop talking now",
        "lets stop talking now",
        "stop the conversation",
        "end the conversation",
        "stop talking now",
    ]
    return any(phrase in text_lower for phrase in stop_phrases)


def is_stop_listening_command(text: str) -> bool:
    """Check if the transcribed text is a command to stop sending input to the LLM. Requires 'snapper' in input."""
    text_lower = text.lower().strip()
    # if "snapper" not in text_lower:
        # return False
    stop_phrases = [
        "stop listening",
        "please stop listening",
    ]
    return any(phrase in text_lower for phrase in stop_phrases)


def is_start_listening_command(text: str) -> bool:
    """Check if the transcribed text is a command to resume sending input to the LLM. Requires 'snapper' in input."""
    text_lower = text.lower().strip()
    # if "snapper" not in text_lower:
        # return False
    start_phrases = [
        "start listening",
        "please start listening",
    ]
    return any(phrase in text_lower for phrase in start_phrases)


def is_reset_command(text: str) -> bool:
    """Check if the transcribed text is a command to reset the conversation."""
    text_lower = text.lower().strip()
    reset_phrases = [
        "start over",
        "lets start over",
        "let's start over",
        "let us start over",
        "reset",
        "clear",
        "new conversation",
        "begin again",
        "start fresh",
        "restart",
        "forget everything",
        "forget that",
    ]
    return any(phrase in text_lower for phrase in reset_phrases)


def build_initial_messages(system_prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_prompt}]



class VLMBackgroundWorker:
    def __init__(self, config: ConversationConfig, session_dir: Path, log_file: Path):
        self.config = config
        self.session_dir = session_dir
        self.log_file = log_file
        self.stop_event = threading.Event()
        self.thread = None
        self.ip = os.getenv("UNITREE_ROBOT_IP", "192.168.4.30")
        self.aes_key = os.getenv("UNITREE_AES_KEY", None)
        if not self.aes_key:
            self.aes_key = None
        
        # Minimum gap between capture starts. Scene descriptions don't need
        # sub-second freshness, so pace queries to leave GPU headroom for the
        # interactive chat/whisper turns rather than running back-to-back.
        # Set to 0 to restore continuous back-to-back capture.
        self.vlm_interval = float(os.getenv("BFF_VLM_INTERVAL", "1.5"))

        # Previous SLAM sample, used to estimate velocity by position delta -
        # LowState_ has no velocity field on this hardware.
        self._last_slam_position = None
        self._last_slam_time = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def _run(self):
        global last_vlm_query_time
        print("[VLM Worker] Started background VLM worker thread (continuous capture).", file=sys.stderr)

        while not self.stop_event.is_set():
            # Don't start a new capture while the chat model is generating a
            # response - both share one GPU, so overlapping inference slows
            # down the interactive turn, which matters more than VLM freshness.
            if is_dialogue_active:
                time.sleep(0.1)
                continue

            self._capture_and_query()
            self._fetch_body_state()

            wait_until = last_vlm_query_time + self.vlm_interval
            while time.time() < wait_until and not self.stop_event.is_set():
                time.sleep(0.1)

    def _fetch_body_state(self):
        """Poll the dashboard's /lowstate endpoint and cache a compact summary
        (battery, velocity, tilt) for injection into the LLM prompt. Foot
        contact is deliberately excluded - it doesn't return valid data on
        the Go2 Pro. LowState_ has no velocity field on this hardware either,
        so velocity is estimated from slam_pose position deltas between polls."""
        global latest_body_state
        dashboard_port = os.getenv("BFF_DASHBOARD_PORT", "8080")
        url = f"http://localhost:{dashboard_port}/lowstate"
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=2.0) as response:
                payload = json.loads(response.read().decode())

            data = payload.get("data")
            if not data:
                return

            bms_soc = (data.get("bms_state") or {}).get("soc")
            if bms_soc is not None:
                battery_pct = float(bms_soc)
            else:
                # Fallback voltage-based estimate if bms_state.soc isn't available
                power_v = data.get("power_v", 0.0)
                battery_pct = max(0.0, min(100.0, ((power_v - 22.0) / (29.6 - 22.0)) * 100))

            rpy = (data.get("imu_state") or {}).get("rpy") or [0.0, 0.0, 0.0]
            pitch_deg = math.degrees(rpy[0])
            roll_deg = math.degrees(rpy[1])

            # Velocity via SLAM position delta between consecutive polls
            speed = 0.0
            position = (data.get("slam_pose") or {}).get("position")
            sample_time = payload.get("timestamp")
            if position is not None and sample_time is not None:
                if self._last_slam_position is not None and self._last_slam_time is not None:
                    dt = sample_time - self._last_slam_time
                    if dt > 0.05:
                        dx = position[0] - self._last_slam_position[0]
                        dy = position[1] - self._last_slam_position[1]
                        candidate_speed = math.hypot(dx, dy) / dt
                        # Guard against SLAM relocalization jumps producing bogus spikes
                        if candidate_speed <= 5.0:
                            speed = candidate_speed
                self._last_slam_position = position
                self._last_slam_time = sample_time

            summary = (
                f"battery {battery_pct:.0f}%, velocity {speed:.2f} m/s, "
                f"tilt pitch {pitch_deg:+.1f}° roll {roll_deg:+.1f}°"
            )
            with body_state_lock:
                latest_body_state = summary
            print(f"[Body State] {summary}", file=sys.stderr)
        except Exception as e:
            print(f"[Body State] Failed to fetch lowstate: {e}", file=sys.stderr)

    def _capture_and_query(self):
        global latest_scene_description, last_vlm_query_time
        
        # Update query time immediately to prevent overlapping runs or spamming if connection hangs
        last_vlm_query_time = time.time()
        
        frame = None
        dashboard_port = os.getenv("BFF_DASHBOARD_PORT", "8080")
        url = f"http://localhost:{dashboard_port}/snapshot"
        try:
            import urllib.request
            import numpy as np
            print(f"[VLM Worker] Fetching frame from dashboard server at {url}...", file=sys.stderr)
            with urllib.request.urlopen(url, timeout=3.0) as response:
                img_data = response.read()
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                print(f"[VLM Worker] Successfully retrieved frame from dashboard server.", file=sys.stderr)
        except Exception as e:
            print(f"[VLM Worker] Could not fetch frame from dashboard server: {e}. Falling back to webcam...", file=sys.stderr)
                
        if frame is None:
            # Fallback to local webcam
            camera_index = 0
            try:
                ret = False
                with silence_outputs():
                    cap = cv2.VideoCapture(camera_index)
                    if cap.isOpened():
                        # Warm up camera exposure
                        for _ in range(5):
                            cap.read()
                        ret, frame = cap.read()
                        cap.release()
                if not ret:
                    print("[VLM Worker] Failed to read frame from local webcam index 0.", file=sys.stderr)
                    frame = None
            except Exception as webcam_err:
                print(f"[VLM Worker] Webcam fallback error: {webcam_err}", file=sys.stderr)
                
        if frame is None:
            print("[VLM Worker] Capture failed. No image retrieved.", file=sys.stderr)
            return
            
        # Create output directory for VLM captures inside session directory
        vlm_dir = self.session_dir / "vlm_captures"
        vlm_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        image_path = vlm_dir / f"snapshot_{timestamp_str}.jpg"
        cv2.imwrite(str(image_path), frame)
        print(f"[VLM Worker] Snapshot saved to {image_path}", file=sys.stderr)
        
        # Query Ollama VLM
        model_name = self.config.vlm_model
        prompt = (
            "Describe the scene in two short sentences: setting, objects present, "
            "lighting, and any people and what they are doing."
        )

        try:
            client = ollama.Client()
            query_start = time.perf_counter()
            stream = client.chat(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [str(image_path)]
                    }
                ],
                stream=True,
                keep_alive=-1,
                options={
                    "num_predict": DEFAULT_VLM_NUM_PREDICT,
                },
            )
            
            description_chunks = []
            for chunk in stream:
                if self.stop_event.is_set():
                    try:
                        stream.close()
                    except Exception:
                        pass
                    return

                content = ""
                if isinstance(chunk, dict):
                    content = chunk.get("message", {}).get("content", "")
                else:
                    msg = getattr(chunk, "message", None)
                    if msg:
                        content = getattr(msg, "content", "")
                description_chunks.append(content)

            query_duration = time.perf_counter() - query_start
            description = "".join(description_chunks).strip()
            
            with vlm_lock:
                latest_scene_description = description
                last_vlm_query_time = time.time()
                
            print(f"[VLM Worker] VLM query finished in {query_duration:.2f}s.", file=sys.stderr)
            print(f"[VLM Worker] Description: {description}", file=sys.stderr)
            
            # Save companion description text file
            desc_path = vlm_dir / f"description_{timestamp_str}.txt"
            with open(desc_path, "w", encoding="utf-8") as desc_file:
                desc_file.write(description + "\n")
                
            # Log in session.jsonl
            append_log_line(
                self.log_file,
                {
                    "type": "vlm_query",
                    "timestamp": timestamp_str,
                    "image_path": str(image_path),
                    "description": description,
                    "duration_seconds": query_duration
                }
            )
        except Exception as e:
            print(f"[VLM Worker] Ollama vision query failed: {e}", file=sys.stderr)



def run_conversation(config: ConversationConfig) -> None:
    dashboard_process = None
    dashboard_log = None

    # Create session directory first so dashboard log can write to it
    log_dir = ensure_log_dir()
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir = log_dir / f"session-{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # Check if the robot dog is on and available
    robot_ip = os.getenv("UNITREE_ROBOT_IP", "192.168.4.30")
    dog_available = False
    if robot_ip:
        print(f"Checking if robot dog is available at {robot_ip}...", file=sys.stderr)
        cmd = ["ping", "-c", "1", "-t", "1" if sys.platform == "darwin" else "1", robot_ip]
        if sys.platform != "darwin":
            cmd = ["ping", "-c", "1", "-W", "1", robot_ip]
        try:
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            dog_available = (res.returncode == 0)
        except Exception:
            dog_available = False

    if dog_available:
        print(f"Robot dog is on and available. Starting dashboard_server.py...", file=sys.stderr)
        try:
            script_dir = Path(__file__).resolve().parent
            dashboard_server_path = script_dir / "dashboard_server.py"
            dashboard_log = open(session_dir / "dashboard_server.log", "w", encoding="utf-8")
            dashboard_env = os.environ.copy()
            dashboard_env["BFF_SESSION_ID"] = session_id
            dashboard_process = subprocess.Popen(
                [sys.executable, str(dashboard_server_path)],
                cwd=str(script_dir),
                stdout=dashboard_log,
                stderr=subprocess.STDOUT,
                env=dashboard_env
            )

            # Wait until the dashboard server is live
            dashboard_port = os.getenv("BFF_DASHBOARD_PORT", "8080")
            url = f"http://localhost:{dashboard_port}/"
            print(f"Waiting for dashboard server to become live at {url}...", file=sys.stderr)

            import urllib.request
            start_wait = time.time()
            is_live = False
            while time.time() - start_wait < 15.0:
                if dashboard_process.poll() is not None:
                    print("[Chat Manager] Dashboard server subprocess exited unexpectedly. Check dashboard_server.log for errors.", file=sys.stderr)
                    break
                try:
                    # Query the dashboard server index route
                    with urllib.request.urlopen(url, timeout=1.0) as response:
                        if response.status == 200:
                            is_live = True
                            break
                except Exception:
                    pass
                time.sleep(0.5)

            if is_live:
                print(f"Dashboard server is live! Proceeding with conversation...", file=sys.stderr)
            else:
                print("[Warning] Dashboard server did not respond in time. Proceeding without it.", file=sys.stderr)
        except Exception as e:
            print(f"Failed to start dashboard_server.py: {e}", file=sys.stderr)
    else:
        print("Robot dog is offline or unavailable. Not starting dashboard server.", file=sys.stderr)

    # Launch behavioral-state-machine.py on Jetson hardware, unless it's already running
    bsm_process = None
    bsm_log = None
    if is_jetson():
        if is_process_running("behavioral-state-machine.py"):
            print("[Chat Manager] behavioral-state-machine.py is already running. Not starting another instance.", file=sys.stderr)
        else:
            print("[Chat Manager] Jetson detected. Starting behavioral-state-machine.py...", file=sys.stderr)
            try:
                script_dir = Path(__file__).resolve().parent
                bsm_path = script_dir / "behavioral-state-machine.py"
                bsm_log = open(session_dir / "behavioral_state_machine.log", "w", encoding="utf-8")
                bsm_process = subprocess.Popen(
                    [sys.executable, str(bsm_path)],
                    cwd=str(script_dir),
                    stdout=bsm_log,
                    stderr=subprocess.STDOUT
                )
                print(f"[Chat Manager] behavioral-state-machine.py started (PID {bsm_process.pid}).", file=sys.stderr)
            except Exception as e:
                print(f"[Chat Manager] Failed to start behavioral-state-machine.py: {e}", file=sys.stderr)
    else:
        print("[Chat Manager] Not running on Jetson hardware. Not starting behavioral-state-machine.py.", file=sys.stderr)

    whisper_model = load_whisper_model(config.whisper_model, config.whisper_compute_type)
    messages = build_initial_messages(config.system_prompt)
    assert config.piper_voice is not None
    piper_voice = load_piper_voice(
        config.piper_voice,
        config.piper_config,
        length_scale=config.piper_length_scale,
        noise_scale=config.piper_noise_scale,
        noise_w=config.piper_noise_w,
    )

    # Load scenes
    script_path = Path("performance-script.json")
    scenes = load_scenes(script_path)
    
    # Set initial system prompt
    current_system_prompt = config.system_prompt
    
    # If "Default" scene exists, use its prompt as the base
    default_scene = next((s for s in scenes if s.name == "Default"), None)
    if default_scene:
        print(f"Using 'Default' scene system prompt.", file=sys.stderr)
        current_system_prompt = default_scene.system_prompt
    
    messages = build_initial_messages(current_system_prompt)
    log_file = session_dir / "session.jsonl"
    append_log_line(
        log_file,
        {
            "type": "session_start",
            "session_id": session_id,
            "config": asdict(config),
            "env_overrides": {
                k: v for k, v in os.environ.items() if k.startswith("BFF_")
            }
        },
    )

    # Initialize and start VLM background worker
    vlm_worker = VLMBackgroundWorker(config, session_dir, log_file)
    vlm_worker.start()


    # Ensure Bluetooth headset is connected
    ensure_headset_connected()
    # Reload environment variables in case headset info was updated
    dotenv.load_dotenv(override=True)
    # Update module-level variables after reload
    global LAST_HEADSET_MAC, LAST_HEADSET_NAME
    LAST_HEADSET_MAC = os.environ.get("LAST_HEADSET_MAC")
    LAST_HEADSET_NAME = os.environ.get("LAST_HEADSET_NAME")
    # Log audio devices after Bluetooth connect/reload
    log_audio_devices()
    # Update the input/output device keywords if we have a saved headset name
    if LAST_HEADSET_NAME and not config.input_device_keyword:
        config.input_device_keyword = LAST_HEADSET_NAME
    if LAST_HEADSET_NAME and not config.output_bt_keyword:
        config.output_bt_keyword = LAST_HEADSET_NAME

    if sys.platform == "darwin":
        try:
            default_device = sd.query_devices(kind='input')
            default_name = default_device.get('name', 'unknown') if default_device else 'unknown'
            print(
                f"On macOS, using system default input device: '{default_name}'.",
                file=sys.stderr,
            )
        except Exception:
            print("On macOS, using system default input device.", file=sys.stderr)
        config.input_device_index = None
    elif config.input_device_keyword:
        device_index = find_input_device(config.input_device_keyword)
        if device_index is not None:
            config.input_device_index = device_index
            dev_info = sd.query_devices(device_index)
            print(
                f"Using input device #{device_index}: {dev_info['name']}",
                file=sys.stderr,
            )
        else:
            # Check if we just connected a Bluetooth device
            bluetooth_connected = False
            if LAST_HEADSET_MAC:
                bluetooth_connected = check_bluetooth_connection_status(LAST_HEADSET_MAC)
            
            # Get the default input device to see what we're actually using
            try:
                default_device = sd.query_devices(kind='input')
                default_name = default_device.get('name', 'unknown') if default_device else 'unknown'
                
                if bluetooth_connected:
                    print(
                        f"Note: Bluetooth device '{config.input_device_keyword}' is connected, "
                        f"but not found by exact name match in audio devices.",
                        file=sys.stderr,
                    )
                    print(
                        f"Using system default input device: '{default_name}' "
                        f"(this is likely the Bluetooth headset).",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"Note: no input device found matching '{config.input_device_keyword}'. "
                        f"Falling back to system default: '{default_name}'.",
                        file=sys.stderr,
                    )
            except Exception:
                if bluetooth_connected:
                    print(
                        f"Note: Bluetooth device '{config.input_device_keyword}' is connected, "
                        f"but not found by exact name match. Using system default audio device.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"Note: no input device found matching '{config.input_device_keyword}'. "
                        "Falling back to system default.",
                        file=sys.stderr,
                    )

    # Resolve output devices (USB + Bluetooth headset)
    config.output_device_indices = resolve_output_devices(config)
    if config.output_device_indices:
        for device_idx in config.output_device_indices:
            if device_idx == "pulse":
                print(
                    f"Using output device '{device_idx}' (PulseAudio device).",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Using output device '{device_idx}'.",
                    file=sys.stderr,
                )
    else:
        if sys.platform == "darwin":
            try:
                default_device = sd.query_devices(kind='output')
                default_name = default_device.get('name', 'unknown') if default_device else 'unknown'
                print(
                    f"On macOS, using system default output device: '{default_name}'.",
                    file=sys.stderr,
                )
                # Force output sample rate to match system default output device rate
                config.output_sample_rate = int(default_device.get('default_samplerate', 16000))
                print(
                    f"On macOS, resampling output to device default rate: {config.output_sample_rate}Hz.",
                    file=sys.stderr,
                )
            except Exception:
                print(
                    "On macOS, using system default output device.",
                    file=sys.stderr,
                )
        else:
            print(
                "Using default system output device.",
                file=sys.stderr,
            )

    stop_event = threading.Event()
    segment_queue: queue.Queue[np.ndarray] = queue.Queue()
    pending_segments: list[np.ndarray] = []
    playback_interrupt = threading.Event()
    pending_concatenation = ""
    
    # Shared references to current TTS worker and playback thread for interruption
    current_tts_worker: TTSWorker | None = None
    current_playback_thread: threading.Thread | None = None
    current_abort_event: threading.Event | None = None

    def on_voice_activity_detected():
        """Called immediately when voice activity is detected (before phrase is complete)"""
        if config.interruptable:
            # Immediately pause TTS synthesis and flush audio queue
            if current_tts_worker is not None:
                current_tts_worker.pause()
                current_tts_worker.flush()  # Flush queued audio chunks
            playback_interrupt.set()
            # Also abort current LLM generation if in progress
            if current_abort_event is not None:
                current_abort_event.set()

    def producer() -> None:
        try:
            for segment in phrase_stream(config, stop_event=stop_event, on_voice_activity=on_voice_activity_detected):
                segment_queue.put(segment)
                # Note: pausing already happened in on_voice_activity_detected when voice was first detected
        except Exception as exc:
            print(f"Phrase producer error: {exc}", file=sys.stderr)

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    # Use the session directory for audio files instead of a temp dir
    try:
        # Initialize TTS Worker
        tts_worker = TTSWorker(piper_voice, config.sample_rate)
        tts_worker.start()

        # Announce readiness
        print("Ready to chat...", file=sys.stderr)
        
        startup_intro = ""#Bluetooth connected, ready to chat."
        startup_prompt = "Give a short friendly greeting to start the conversation. One sentence."
        startup_line = ""
        for sentence in query_ollama_streaming(
            config.ollama_model,
            build_initial_messages(config.system_prompt)
            + [{"role": "user", "content": startup_prompt}],
            interruptable=config.interruptable,
            options={
                "temperature": config.ollama_temperature,
                "top_p": config.ollama_top_p,
                "top_k": config.ollama_top_k,
                "num_predict": min(config.ollama_num_predict, 60),
                "num_ctx": config.ollama_num_ctx,
            },
            think=False,
            print_stream=False,
        ):
            startup_line = sentence.strip()
            if startup_line:
                break

        startup_text = startup_intro
        if startup_line:
            startup_text = f"{startup_intro} {startup_line}"
        
        startup_audio = session_dir / "startup.wav"
        synthesize_with_piper(
            piper_voice,
            startup_text,
            startup_audio,
        )
        # Clear interrupt before playing startup sound
        playback_interrupt.clear()
        play_audio(
            startup_audio,
            playback_interrupt,
            interruptable=False,
            output_device_indices=config.output_device_indices,
            output_sample_rate=config.output_sample_rate,
        )

        # --- Wake word + special commands ---
        # Usage:
        # - Say "ok/okay/hey snapper", then say a command like "shutdown"
        # - Or say "ok snapper shutdown" (etc) in one utterance
        WAKE_PHRASES = config.wake_phrases
        WAKE_WINDOW_SECONDS = 8.0
        wake_armed_until = 0.0

        def normalize_command(text: str) -> str:
            lowered = text.lower().strip()
            lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
            return " ".join(lowered.split())

        def find_wake_phrase(norm_text: str) -> str | None:
            padded = f" {norm_text} "
            for phrase in WAKE_PHRASES:
                if f" {phrase} " in padded:
                    return phrase
            return None

        def strip_wake_phrase(norm_text: str, phrase: str) -> str:
            padded = f" {norm_text} "
            stripped = padded.replace(f" {phrase} ", " ", 1)
            return " ".join(stripped.split())

        def run_special_command(cmd: str, raw_text: str) -> bool:
            """
            Returns True if handled (and caller should continue/exit),
            False if not a recognized special command.
            """
            nonlocal wake_armed_until, listening_active, current_system_prompt, messages
            cmd_norm = normalize_command(cmd)
            if not cmd_norm:
                return False
            cmd_compact = cmd_norm.replace(" ", "")

            def speak(text: str) -> None:
                try:
                    out = session_dir / f"turn-{turn:03d}-special.wav"
                    synthesize_with_piper(piper_voice, text, out)
                    playback_interrupt.clear()
                    play_audio(
                        out,
                        playback_interrupt,
                        interruptable=False,
                        output_device_indices=config.output_device_indices,
                        output_sample_rate=config.output_sample_rate,
                    )
                except Exception as exc:
                    print(f"Special command TTS/playback error: {exc}", file=sys.stderr)

            # Clear any pending wake state once we attempt a command
            wake_armed_until = 0.0

            # Accept broader shutdown phrases:
            # - "shutdown" / "shut down"
            # - "shut it down", "shut down now", etc.
            if (
                cmd_compact == "shutdown"
                or "shutdown" in cmd_compact
                or " shut down " in f" {cmd_norm} "
                or " shut it down " in f" {cmd_norm} "
            ):
                append_log_line(
                    log_file,
                    {"type": "special_command", "turn": turn, "command": "shutdown", "text": raw_text},
                )
                speak("Shutting down.")
                shutdown_script = Path(__file__).resolve().parent / "shutdown.py"
                try:
                    subprocess.Popen(
                        [sys.executable, str(shutdown_script), "--yes"],
                        cwd=str(shutdown_script.parent),
                    )
                except Exception as exc:
                    print(f"Failed to launch shutdown command: {exc}", file=sys.stderr)
                stop_event.set()
                return True

            if cmd_norm == "dream":
                append_log_line(
                    log_file,
                    {"type": "special_command", "turn": turn, "command": "dream", "text": raw_text},
                )
                speak("Okay.")
                return True

            if cmd_norm in ("follow me", "follow"):
                append_log_line(
                    log_file,
                    {"type": "special_command", "turn": turn, "command": "follow_me", "text": raw_text},
                )
                speak("Okay, follow me.")
                return True

            # Stop conversation: stop listening, reset to default system prompt, wait until "snapper start listening"
            if is_lets_stop_command(cmd_norm) or is_lets_stop_command(raw_text):
                listening_active = False
                if default_scene is not None:
                    current_system_prompt = default_scene.system_prompt
                else:
                    current_system_prompt = config.system_prompt
                messages = build_initial_messages(current_system_prompt)
                append_log_line(
                    log_file,
                    {"type": "stop_conversation", "turn": turn, "command": "lets_stop", "text": raw_text},
                )
                speak("Ok, conversation stopped. Say snapper start listening when you're ready.")
                return False

            return False

        turn = 1
        listening_active = True  # When False, ignore all transcribed input until "start listening"
        global is_dialogue_active, last_interaction_time
        is_dialogue_active = True
        last_interaction_time = time.time()

        while True:
            try:
                # Prioritize any pending messages (e.g. from scene switch context)
                # No, we don't have a pending message queue for the loop, we rely on segment_queue mostly.
                # But if we just switched scenes and added to messages, we should fall through to generation.
                # Wait... the loop structure expects to Get Audio -> Transcribe -> messages.append -> Generate.
                # If we switched scene, we have `user_text` (the trigger).
                # We appended it to `messages` in the new context.
                # So we just need to NOT `continue` if it's a matched_scene, but instead proceed to the "messages.append" (which we essentially did manually) and then generation.
                
                # Careful: The original code appends `user_text` at line 1861.
                # In my replacement above, I added logic:
                # if matched_scene: messages.append(...)
                # So I should Skip the standard append at 1861 if I already did it.
                
                # Let's adjust the flow in the replacement above to set a flag 'skip_standard_append' or just proceed carefully.
                is_dialogue_active = False
                
                if pending_segments:
                    phrase = pending_segments.pop(0)
                else:
                    phrase = segment_queue.get(timeout=0.1)
                
                is_dialogue_active = True
                last_interaction_time = time.time()
            except queue.Empty:
                continue

            raw_audio = session_dir / f"turn-{turn:03d}-input.wav"
            sf.write(raw_audio, phrase, config.sample_rate)

            # If this segment interrupted a previous turn, ensure previous TTS is fully stopped
            if current_tts_worker is not None:
                current_tts_worker.stop()
                current_tts_worker = None
            if current_playback_thread is not None and current_playback_thread.is_alive():
                playback_interrupt.set()
                current_playback_thread.join(timeout=0.5)
                current_playback_thread = None
            if current_abort_event is not None:
                current_abort_event.set()
                current_abort_event = None

            user_text = transcribe_audio(whisper_model, raw_audio, config.show_levels)
            if not user_text:
                # Prompt the user via TTS instead of printing.
                try:
                    reprompt_text = "what did you say?"
                    reprompt_audio = session_dir / f"turn-{turn:03d}-reprompt.wav"
                    synthesize_with_piper(piper_voice, reprompt_text, reprompt_audio)
                    # Clear interrupt flag since this didn't result in a query
                    playback_interrupt.clear()
                    play_audio(
                        reprompt_audio,
                        playback_interrupt,
                        interruptable=config.interruptable,
                        output_device_indices=config.output_device_indices,
                        output_sample_rate=config.output_sample_rate,
                    )
                except Exception as exc:
                    print(f"Reprompt TTS/playback error: {exc}", file=sys.stderr)
                continue

            # Stop conversation: stop listening, reset to default system prompt, wait until "snapper start listening"
            if is_lets_stop_command(user_text):
                listening_active = False
                # Reset to default system prompt (same as "let's start over")
                if default_scene is not None:
                    current_system_prompt = default_scene.system_prompt
                else:
                    current_system_prompt = config.system_prompt
                messages = build_initial_messages(current_system_prompt)
                print("Conversation stopped (reset to default). Say snapper start listening when ready.", file=sys.stderr)
                append_log_line(
                    log_file,
                    {"type": "stop_conversation", "turn": turn, "text": user_text},
                )
                try:
                    stop_conv_audio = session_dir / f"turn-{turn:03d}-stop-conversation.wav"
                    synthesize_with_piper(
                        piper_voice,
                        "Ok, conversation stopped. Say snapper start listening when you're ready.",
                        stop_conv_audio,
                    )
                    playback_interrupt.clear()
                    play_audio(
                        stop_conv_audio,
                        playback_interrupt,
                        interruptable=False,
                        output_device_indices=config.output_device_indices,
                        output_sample_rate=config.output_sample_rate,
                    )
                except Exception as exc:
                    print(f"Stop-conversation TTS/playback error: {exc}", file=sys.stderr)
                turn += 1
                continue

            # Stop/start listening (no LLM prompting when stopped) — works with or without "snapper"
            if is_stop_listening_command(user_text):
                listening_active = False
                print("Stopped listening (no LLM prompting until you say 'start listening').", file=sys.stderr)
                try:
                    stop_audio = session_dir / f"turn-{turn:03d}-stop-listening.wav"
                    synthesize_with_piper(
                        piper_voice,
                        "Ok, stopped listening.",
                        stop_audio,
                    )
                    playback_interrupt.clear()
                    play_audio(
                        stop_audio,
                        playback_interrupt,
                        interruptable=config.interruptable,
                        output_device_indices=config.output_device_indices,
                        output_sample_rate=config.output_sample_rate,
                    )
                except Exception as exc:
                    print(f"Stop-listening TTS/playback error: {exc}", file=sys.stderr)
                turn += 1
                continue
            if is_start_listening_command(user_text):
                listening_active = True
                print("Listening again.", file=sys.stderr)
                try:
                    start_audio = session_dir / f"turn-{turn:03d}-start-listening.wav"
                    synthesize_with_piper(
                        piper_voice,
                        "Ok, listening again.",
                        start_audio,
                    )
                    playback_interrupt.clear()
                    play_audio(
                        start_audio,
                        playback_interrupt,
                        interruptable=config.interruptable,
                        output_device_indices=config.output_device_indices,
                        output_sample_rate=config.output_sample_rate,
                    )
                except Exception as exc:
                    print(f"Start-listening TTS/playback error: {exc}", file=sys.stderr)
                turn += 1
                continue
            if not listening_active:
                # Ignore all other input until "start listening" is heard
                continue

            if pending_concatenation:
                if config.flush_on_interrupt:
                    print(f"Flushing previous input: '{pending_concatenation}' (flush on interrupt enabled)", file=sys.stderr)
                    pending_concatenation = ""
                else:
                    print(f"Concatenating previous input: '{pending_concatenation}' + '{user_text}'", file=sys.stderr)
                    user_text = f"{pending_concatenation} {user_text}"
                    pending_concatenation = ""

            # Wake word handling: "ok/okay/hey snapper" (or "snapper") arms a short window
            # where the next utterance is processed. Optionally ignores all input if require_wakeword is True.
            now = time.time()
            user_norm = normalize_command(user_text)
            matched_wake = find_wake_phrase(user_norm)
            
            if matched_wake:
                # Remove the wake phrase and attempt to treat remaining text as an inline command.
                inline = strip_wake_phrase(user_norm, matched_wake).strip()
                if inline:
                    if run_special_command(inline, user_text):
                        return
                    # Not a special command: treat inline text as the user prompt,
                    # arm the wake window, and fall through to normal LLM generation.
                    user_text = inline
                    wake_armed_until = now + WAKE_WINDOW_SECONDS
                else:
                    wake_armed_until = now + WAKE_WINDOW_SECONDS
                    try:
                        ack_audio = session_dir / f"turn-{turn:03d}-wake.wav"
                        synthesize_with_piper(piper_voice, "Yes?", ack_audio)
                        playback_interrupt.clear()
                        play_audio(
                            ack_audio,
                            playback_interrupt,
                            interruptable=False,
                            output_device_indices=config.output_device_indices,
                            output_sample_rate=config.output_sample_rate,
                        )
                    except Exception as exc:
                        print(f"Wake-word ack TTS/playback error: {exc}", file=sys.stderr)
                    continue
            else:
                if wake_armed_until and now <= wake_armed_until:
                    if run_special_command(user_text, user_text):
                        return
                    # Not a special command: disarm wake window and fall through to normal chat handling.
                    wake_armed_until = 0.0
                else:
                    # No wake word detected, and not in the armed wake window.
                    if config.require_wakeword:
                        print(f"Ignoring input: Wake word not detected in '{user_text}'", file=sys.stderr)
                        continue

            # Check for scene triggers
            matched_scene = next(
                (s for s in scenes if s.trigger and s.trigger.lower() in user_text.lower()),
                None,
            )
            
            # Check for reset command
            is_reset = is_reset_command(user_text)
            if is_reset or matched_scene:
                if matched_scene:
                    print(f"Scene trigger detected: '{matched_scene.name}'", file=sys.stderr)
                    current_system_prompt = matched_scene.system_prompt
                    reset_message = f"Switching to scene: {matched_scene.name}"
                    
                    # Log the scene switch
                    append_log_line(
                        log_file,
                        {
                            "type": "scene_switch",
                            "turn": turn,
                            "scene": matched_scene.name,
                            "trigger_text": user_text,
                        }
                    )
                else:
                    # Manual reset command: reload performance script from disk
                    scenes = load_scenes(script_path)
                    default_scene = next((s for s in scenes if s.name == "Default"), None)
                    print("Conversation reset. Starting fresh.", file=sys.stderr)
                    # For manual resets, always return to the Default scene if it exists;
                    # otherwise fall back to the original configured system prompt.
                    if default_scene is not None:
                        current_system_prompt = default_scene.system_prompt
                    else:
                        current_system_prompt = config.system_prompt
                    reset_message = "Ok, starting over"
                
                messages = build_initial_messages(current_system_prompt)
                
                # If it was a scene match, we might want to also add the user's text as the first message
                # or just let the reset happen and wait for next input?
                # "when a trigger is detected it should set the llm system prompt to the new prompt and continue the conversation"
                # Usually implies we respond to the trigger phrase in the NEW context.
                
                if matched_scene:
                     pass
                
                if not matched_scene: # Only log reset if it was a manual reset command
                    append_log_line(
                        log_file,
                        {
                            "type": "reset",
                            "turn": turn,
                            "text": user_text,
                            "audio_path": str(raw_audio),
                        },
                    )

                # Synthesize and play confirmation/response (only for manual reset for now?)
                # If it's a scene switch, we probably want to Generate a response to the trigger phrase immediately
                # instead of just saying "Ok starting over". 
                
                if matched_scene:
                     # Fall through to normal generation loop with the new messages
                     pass 
                else:
                    # Manual reset behavior
                    reset_audio = session_dir / f"turn-{turn:03d}-reset.wav"
                    synthesize_with_piper(
                        piper_voice,
                        reset_message,
                        reset_audio,
                    )
                    play_audio(
                        reset_audio,
                        playback_interrupt,
                        interruptable=config.interruptable,
                        output_device_indices=config.output_device_indices,
                        output_sample_rate=config.output_sample_rate,
                    )
                    turn += 1
                    continue

            # --- VLM Visual Context & Body State Injection ---
            # The background worker captures continuously (~1-2s cadence with a fast
            # VLM like moondream), so the cache is always near-fresh; no on-demand
            # synchronous capture is needed here.
            with vlm_lock:
                current_description = latest_scene_description
            with body_state_lock:
                current_body_state = latest_body_state

            # Clean up old visual context / body state messages from dialogue history to avoid context bloating
            messages = [
                m for m in messages
                if not (m["role"] == "system" and (
                    m["content"].startswith("Visual context (what you see):")
                    or m["content"].startswith("Body state:")
                ))
            ]

            # Inject new visual context and body state if available
            if current_description:
                messages.append({"role": "system", "content": f"Visual context (what you see): {current_description}"})
            if current_body_state:
                messages.append({"role": "system", "content": f"Body state: {current_body_state}"})

            messages.append({"role": "user", "content": user_text})
            
            # Truncate history: Keep generic system prompt + last N messages
            # This prevents the context from growing indefinitely and slowing down prefill.
            limit = config.history_truncation_limit
            if len(messages) > limit:
                # Keep system prompt (index 0) and the last (limit - 1) messages
                # We subtract 1 to account for the system prompt
                num_to_keep = limit - 1
                messages = [messages[0]] + messages[-num_to_keep:]

            append_log_line(
                log_file,
                {
                    "type": "user",
                    "turn": turn,
                    "text": user_text,
                    "audio_path": str(raw_audio),
                },
            )

            abort_event = threading.Event()
            
            # --- New Streaming Implementation ---
            
            # 1. Start TTS Worker
            tts_worker = TTSWorker(piper_voice, config.sample_rate)
            tts_worker.start()
            
            # Store references for interruption handling
            current_tts_worker = tts_worker
            current_abort_event = abort_event
            
            # 2. Check interruption function
            def check_interrupt():
                if not config.interruptable:
                    return False
                try:
                    new_segment = segment_queue.get_nowait()
                    pending_segments.append(new_segment)
                    # Pause and flush TTS (already done by producer, but ensure it here too)
                    tts_worker.pause()
                    tts_worker.flush()
                    playback_interrupt.set()
                    abort_event.set()
                    tts_worker.stop()
                    return True
                except queue.Empty:
                    return False

            # 3. Stream from Ollama -> TTS Worker
            full_assistant_text = ""
            interrupted = False
            
            # Start a thread to monitor interruptions during LLM generation?
            # Or just check periodically in the generator loop? 
            # The generator loop runs in main thread, so we can check there.
            
            # Temporarily disable level meter display during query and playback
            original_show_levels = config.show_levels
            config.show_levels = False
            # Clear level meter line
            sys.stderr.write("\r" + " " * 80 + "\r")
            sys.stderr.flush()

            # Start Playback Thread immediately
            # Use the TTS voice's sample rate for playback
            tts_sample_rate = tts_worker.voice.config.sample_rate if tts_worker.voice else config.sample_rate
            
            # Clear any stale interrupt from the user's input
            playback_interrupt.clear()
            
            response_audio_path = session_dir / f"turn-{turn:03d}-response.wav"

            playback_thread = threading.Thread(
                target=play_audio_stream,
                args=(
                    tts_worker.audio_queue,
                    tts_sample_rate,
                    playback_interrupt,
                    config.interruptable,
                    response_audio_path,
                    config.output_device_indices,
                    config.output_sample_rate,
                ),
                daemon=True
            )
            current_playback_thread = playback_thread
            playback_thread.start()

            # Track timing for each sentence
            prev_sentence_time = time.perf_counter()
            
            for sentence in query_ollama_streaming(
                config.ollama_model, 
                messages,
                interruptable=config.interruptable,
                stop_event=abort_event,
                options={
                    "temperature": config.ollama_temperature,
                    "top_p": config.ollama_top_p,
                    "top_k": config.ollama_top_k,
                    "num_predict": config.ollama_num_predict,
                    "num_ctx": config.ollama_num_ctx,
                },
                think=config.ollama_think,
                print_stream=True,
            ):
                if check_interrupt():
                    interrupted = True
                    break
                
                # Calculate time elapsed since previous sentence
                current_time = time.perf_counter()
                elapsed = current_time - prev_sentence_time
                prev_sentence_time = current_time
                
                # Print timing info (text is already streamed in real-time)
                print(f" ({elapsed:.2f}s) ", end="", flush=True)
                full_assistant_text += sentence + " "
                
                # --- CLEANING ---
                # Remove asterisks, headers, bullets, parenthesis/brackets blocks (e.g. [laughs])
                cleaned_sentence = re.sub(r'[\*#_`~]', '', sentence)                   # Remove markdown chars
                cleaned_sentence = re.sub(r'\([^\)]*\)', '', cleaned_sentence)        # Remove (content)
                cleaned_sentence = re.sub(r'\[[^\]]*\]', '', cleaned_sentence)        # Remove [content]
                cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()     # Normalize whitespace

                if cleaned_sentence and not _is_punctuation_only(cleaned_sentence):
                    tts_worker.put_text(cleaned_sentence)
                
            tts_worker.put_text(None) # End of input
            print() # Newline after response
            
            if interrupted:
                # New audio input detected that resulted in interruption
                # Flush TTS queue completely and cancel everything
                tts_worker.flush()
                tts_worker.stop()
                # print("Interrupted during generation.", file=sys.stderr)
                append_log_line(log_file, {"type": "assistant_cancelled", "turn": turn})
                 # Rollback
                if messages and messages[-1]["role"] == "user":
                    messages.pop()
                if messages and messages[-1]["role"] == "assistant": # Should not happen yet
                    messages.pop()
                pending_concatenation = user_text
                
                # Wait for playback thread to die (it should see interrupt event)
                playback_thread.join(timeout=1.0)
                # Clear references
                current_tts_worker = None
                current_playback_thread = None
                current_abort_event = None
                turn += 1
                config.show_levels = original_show_levels
                continue

            # Wait for playback to finish naturally
            
            while playback_thread.is_alive():
                if config.interruptable and playback_interrupt.is_set():
                    interrupted = True
                    # Flush TTS queue when interrupted during playback
                    tts_worker.flush()
                    tts_worker.stop()
                    break
                playback_thread.join(timeout=0.1)

            if interrupted:
                 # Interruption during tail playback
                 # Treat as interruption: rollback
                 if messages and messages[-1]["role"] == "user":
                    messages.pop()
                 pending_concatenation = user_text
                 # Clear references
                 current_tts_worker = None
                 current_playback_thread = None
                 current_abort_event = None
                 turn += 1
                 config.show_levels = original_show_levels
                 continue
            
            messages.append({"role": "assistant", "content": full_assistant_text.strip()})
            
            append_log_line(
                log_file,
                {
                    "type": "assistant",
                    "turn": turn,
                    "text": full_assistant_text.strip(),
                    "audio_path": str(response_audio_path),
                },
            )
            
            tts_worker.stop() # Cleanup
            # Clear references after successful completion
            current_tts_worker = None
            current_playback_thread = None
            current_abort_event = None

            # Save full response audio for logging (non-blocking or post-hoc?)
            # Re-synthesizing for logs is expensive. Ideally we'd capture the stream.
            # For now, let's skip re-synthesis to save time/resources on Jetson.
            
            turn += 1
            config.show_levels = original_show_levels
    except KeyboardInterrupt:
        print("\nExiting conversation.")
    finally:
        # Stop dashboard server if running
        if 'dashboard_process' in locals() and dashboard_process is not None:
            print("[Chat Manager] Terminating dashboard_server.py...", file=sys.stderr)
            dashboard_process.terminate()
            try:
                dashboard_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                print("[Chat Manager] Force killing dashboard_server.py...", file=sys.stderr)
                dashboard_process.kill()
                dashboard_process.wait()
            finally:
                if 'dashboard_log' in locals() and dashboard_log is not None:
                    dashboard_log.close()

        # behavioral-state-machine.py is intentionally left running - it controls
        # the robot's physical stand-up/stand-down state, which shouldn't be
        # tied to chat-manager.py's own lifecycle.

        # Stop VLM background worker
        if 'vlm_worker' in locals():
            vlm_worker.stop()
        stop_event.set()
        producer_thread.join(timeout=1.0)
        append_log_line(
            log_file,
            {"type": "session_end"},
        )


def main() -> None:
    config = parse_args()
    run_conversation(config)


if __name__ == "__main__":
    main()

