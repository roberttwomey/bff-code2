#!/usr/bin/env python3
"""Voice chat assistant using Whisper STT, Ollama Gemma 4 LLM, Moondream VLM, and Piper TTS.

This script performs continuous voice activity detection (VAD) on microphone
audio, automatically segments speech, transcribes each utterance with Whisper,
maintains continuous visual context via a background Moondream VLM worker,
sends the resulting text and scene descriptions to an Ollama model (`gemma4:e2b` by default), and
plays back the assistant response via Piper text-to-speech using the Python
`piper-tts` library.

Requirements:
    - ollama (Python package) with the configured LLM (`gemma4:e2b`) and VLM (`moondream`) models pulled locally
    - openai-whisper / faster-whisper
    - sounddevice, soundfile, numpy
    - piper-tts (Python package) and at least one Piper voice model file
    - silero-vad (optional; speech-aware segmentation. torchaudio must match
      the installed torch version. Falls back to RMS gating if missing)

Example usage:
    python chat-manager.py --piper-voice speech/piper/en_GB-aru-medium.onnx
    
    python chat-manager.py --show-levels

Environment variables:
    BFF_OLLAMA_MODEL        override Ollama model name for text chat (default: gemma4:e2b)
    BFF_OLLAMA_TEMPERATURE  override Ollama sampling temperature (default: 0.7)
    BFF_OLLAMA_TOP_P        override Ollama top_p (default: 0.9)
    BFF_OLLAMA_TOP_K        override Ollama top_k (default: 40)
    BFF_OLLAMA_NUM_PREDICT  override max tokens generated per LLM response (default: 200)
    BFF_OLLAMA_NUM_CTX      override context size for Ollama (default: 2048)
    BFF_OLLAMA_THINK        enable thinking token parsing for reasoning models (default: false)
    BFF_SYSTEM_PROMPT       override default system prompt for the chat assistant
    BFF_HISTORY_TRUNCATION_LIMIT override conversation history truncation limit (default: 11)
    BFF_VLM_MODEL           override Ollama model name for VLM scene captioning (default: moondream)
    BFF_VLM_NUM_PREDICT     override max tokens generated per VLM scene description (default: 50)
    BFF_VLM_NUM_CTX         override context size for the VLM model (default: 2048)
    BFF_VLM_TEMPERATURE     override VLM sampling temperature (default: 0.4)
    BFF_VLM_CHANGE_THRESHOLD override mean grayscale pixel-diff (0-255) required to trigger a re-description (default: 12.0)
    BFF_VLM_INTERVAL        override minimum seconds between VLM capture starts (default: 1.5)
    BFF_WHISPER_MODEL       override Whisper model size (default: tiny.en)
    BFF_WHISPER_DEVICE      override Whisper device, e.g. cpu or cuda (default: cuda if available)
    BFF_PIPER_VOICE         override Piper voice path if --piper-voice not provided
    BFF_PLAYBACK_SPEED      override playback speed multiplier (default: 1.0)
    BFF_PLAYBACK_TAIL_SILENCE seconds of silence appended after the last speech chunk so the sink can't clip it (default: 0.1)
    BFF_INTERRUPTABLE       override interruptable behavior (default: true)
    BFF_FLUSH_ON_INTERRUPT  override audio queue flushing on user interruption (default: false)
    BFF_LOG_ROOT            override session history root (default: ./captures)
    BFF_VAD_BACKEND         speech gating backend: silero (default) or rms
    BFF_VAD_THRESHOLD       Silero speech probability that starts a segment (default: 0.5)
    BFF_INPUT_DEVICE_KEYWORD match input device name substring (default: OpenRun Pro 2 by Shokz)
    BFF_ACTIVATION_THRESHOLD RMS activation threshold for RMS gating (default: 0.03)
    BFF_SILENCE_THRESHOLD    RMS threshold for silence detection (default: 0.015)
    BFF_SILENCE_DURATION     silence duration in seconds to trigger phrase end (default: 0.8)
    BFF_MIN_PHRASE_SECONDS   minimum phrase length in seconds (default: 0.5)
    BFF_REQUIRE_WAKEWORD    require wake phrase to activate (default: false)
    BFF_WAKE_PHRASES        comma-separated list of wake phrases
    BFF_PULSE_SINKS         comma-separated PulseAudio sink names to combine for output
    BFF_PULSE_COMBINED_SINK_NAME combined PulseAudio sink name (default: bff_combined)
    BFF_PULSE_DEVICE_NAME   PulseAudio device name for PortAudio (default: pulse)
    BFF_PULSE_SOURCE_VOLUME override mic capture volume percentage (e.g. 60%) via pactl
    BFF_DASHBOARD_START_ATTEMPTS how many times to (re)start dashboard_server.py while waiting for the WebRTC feed (default: 3)
    BFF_DASHBOARD_HTTP_TIMEOUT   seconds to wait for the dashboard to serve its index page (default: 15)
    BFF_DASHBOARD_STREAM_TIMEOUT seconds to wait for the robot camera/telemetry feeds before restarting the dashboard (default: 30)
"""

from __future__ import annotations

import argparse
import base64
import http.client
import collections
import difflib
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
import urllib.request
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, List
import shutil
import subprocess
import wave

import numpy as np
import ollama

# ALSA's pulse plugin reads this when a stream opens; PortAudio's default
# latency there is tight enough to underrun whenever Whisper/Piper spike the CPU.
# Must comfortably exceed the stream block size (BFF_BLOCK_DURATION, 200 ms
# default) or every block underruns.
os.environ.setdefault("PULSE_LATENCY_MSEC", "400")

import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import torch
import dotenv

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

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
CURRENT_STREAM_SAMPLERATE = 16000
is_assistant_speaking = False
last_assistant_speech_time = 0.0

# --- Global Body State ---
body_state_lock = threading.Lock()
latest_body_state = ""

# Motor order as the SDK reports it: three joints per leg, front-right first.
# motor_state pads out to 20 entries; only these first 12 are real motors.
LEG_NAMES = ("FR", "FL", "RR", "RL")
JOINT_NAMES = tuple(f"{leg}_{joint}" for leg in LEG_NAMES for joint in ("hip", "thigh", "calf"))

# Prefixes of the system messages the background workers inject. Shared so the
# injection, the stale-message cleanup and the console log can't drift apart.
VISUAL_CONTEXT_PREFIX = "Visual context (what you see):"
BODY_STATE_PREFIX = "Body state:"

# What counts as cool / warm / hot for each kind of sensor, since a small model
# has no reference for these numbers and reads magnitude as alarm - it called a
# 24°C pack "quite hot" when the charge beside it read 97%. Bands are (warm
# above, hot above) in °C, against each part's own normal operating range.
TEMP_BANDS = {
    "cell": (38.0, 45.0),      # pack thermistors; sits near ambient
    "joint": (55.0, 70.0),     # motors idle around 35-50 and climb under load
    "chassis": (60.0, 75.0),   # main board NTC
    "core": (88.0, 95.0),      # IMU die runs ~79 all the time
}


def describe_temperature(value: float, kind: str) -> str:
    """Cool / warm / hot for a reading, judged against its own normal range."""
    warm_above, hot_above = TEMP_BANDS[kind]
    if value > hot_above:
        return "hot"
    if value > warm_above:
        return "warm"
    return "cool"

# Per-capture chatter from the background workers (snapshot paths, YOLO hints,
# query timings). Off by default - the packet lines below are what matters.
WORKER_VERBOSE = os.getenv("BFF_WORKER_VERBOSE", "0").lower() in ("1", "true", "yes")


def log_context_packet(prefix: str, content: str) -> None:
    """Print a context packet exactly as the model will receive it.

    Workers call this only when the packet's meaning changes, so the console
    shows the model's view of the world rather than a sensor tick."""
    print(f"[LLM context] {prefix} {content}", file=sys.stderr)

from piper import PiperVoice
try:  # Optional type that some versions expose
    from piper import AudioChunk  # type: ignore
except ImportError:  # pragma: no cover - older library versions
    AudioChunk = None



dotenv.load_dotenv()

def fix_user_paths() -> None:
    """
    Check environment variables for non-existent file paths and attempt to resolve them
    relative to repository relative subpaths or current user's home directory.
    """
    cohab_home = Path("/home/cohab")
    if not cohab_home.exists():
        print(f"Notice: {cohab_home} not found. Remapping paths to {Path.home()}...", file=sys.stderr)

    current_home = Path.home()
    repo_root = SCRIPT_DIR

    for key, value in list(os.environ.items()):
        if not value or not isinstance(value, str):
            continue
        p = Path(value)
        if len(value) < 1000 and (value.startswith("/") or "speech" in value or "captures" in value or "logs" in value):
            try:
                exists = p.exists()
            except (OSError, ValueError):
                exists = False
            if not exists:
                parts = p.parts
                found_rel = False
                for sub in ("speech", "captures", "logs"):
                    if sub in parts:
                        idx = parts.index(sub)
                        rel_path = Path(*parts[idx:])
                        candidate = repo_root / rel_path
                        if candidate.exists():
                            os.environ[key] = str(candidate)
                            found_rel = True
                            break
                if not found_rel and "/home/cohab" in value:
                    new_value = value.replace("/home/cohab", str(current_home))
                    os.environ[key] = new_value

fix_user_paths()


DEFAULT_SYSTEM_PROMPT = (
    os.environ.get(
        "BFF_SYSTEM_PROMPT",
        "you are SNAPPER a robot dog. you do not say woof, whir, tail wag. answer in 2 sentences or less.",
    )
)

DEFAULT_OLLAMA_MODEL = os.environ.get("BFF_OLLAMA_MODEL", "gemma4:e2b")
DEFAULT_VLM_MODEL = os.environ.get("BFF_VLM_MODEL", "moondream")
DEFAULT_WHISPER_MODEL = os.environ.get("BFF_WHISPER_MODEL", "tiny.en")
DEFAULT_SAMPLE_RATE = int(os.environ.get("BFF_SAMPLE_RATE", "16000"))
DEFAULT_PLAYBACK_SPEED = float(os.environ.get("BFF_PLAYBACK_SPEED", "1.0"))
# Silence appended after the last synthesized chunk so the release of the final
# word survives whatever the sink discards when the stream is torn down.
PLAYBACK_TAIL_SILENCE = float(os.environ.get("BFF_PLAYBACK_TAIL_SILENCE", "0.1"))
DEFAULT_INPUT_DEVICE_KEYWORD = os.environ.get(
    "BFF_INPUT_DEVICE_KEYWORD", "Wireless Mic Rx"
)
DEFAULT_ACTIVATION_THRESHOLD = float(os.environ.get("BFF_ACTIVATION_THRESHOLD", "0.03"))
DEFAULT_SILENCE_THRESHOLD = float(os.environ.get("BFF_SILENCE_THRESHOLD", "0.015"))
DEFAULT_SILENCE_DURATION = float(os.environ.get("BFF_SILENCE_DURATION", "0.8"))
DEFAULT_MIN_PHRASE_SECONDS = float(os.environ.get("BFF_MIN_PHRASE_SECONDS", "0.5"))
DEFAULT_BLOCK_DURATION = float(os.environ.get("BFF_BLOCK_DURATION", "0.2"))
DEFAULT_VAD_BACKEND = os.environ.get("BFF_VAD_BACKEND", "silero").lower()
DEFAULT_VAD_THRESHOLD = float(os.environ.get("BFF_VAD_THRESHOLD", "0.5"))
DEFAULT_INTERRUPTABLE_ENV = os.environ.get("BFF_INTERRUPTABLE", "true").lower()
DEFAULT_INTERRUPTABLE = DEFAULT_INTERRUPTABLE_ENV in ("true", "1", "yes", "on")
DEFAULT_FLUSH_ON_INTERRUPT_ENV = os.environ.get("BFF_FLUSH_ON_INTERRUPT", "false").lower()
DEFAULT_FLUSH_ON_INTERRUPT = DEFAULT_FLUSH_ON_INTERRUPT_ENV in ("true", "1", "yes", "on")
LOG_ROOT = Path(
    os.environ.get("BFF_LOG_ROOT", SCRIPT_DIR / "captures")
).expanduser()
DEFAULT_HISTORY_TRUNCATION_LIMIT = int(os.environ.get("BFF_HISTORY_TRUNCATION_LIMIT", "11"))
DEFAULT_OLLAMA_TEMPERATURE = float(os.environ.get("BFF_OLLAMA_TEMPERATURE", "0.7"))
DEFAULT_OLLAMA_TOP_P = float(os.environ.get("BFF_OLLAMA_TOP_P", "0.9"))
DEFAULT_OLLAMA_TOP_K = int(os.environ.get("BFF_OLLAMA_TOP_K", "40"))
DEFAULT_OLLAMA_NUM_PREDICT = int(os.environ.get("BFF_OLLAMA_NUM_PREDICT", "100"))
DEFAULT_OLLAMA_NUM_CTX = int(os.environ.get("BFF_OLLAMA_NUM_CTX", "2048"))
DEFAULT_VLM_NUM_PREDICT = int(os.environ.get("BFF_VLM_NUM_PREDICT", "50"))
DEFAULT_VLM_NUM_CTX = int(os.environ.get("BFF_VLM_NUM_CTX", "2048"))
DEFAULT_VLM_TEMPERATURE = float(os.environ.get("BFF_VLM_TEMPERATURE", "0.4"))
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
DEFAULT_PULSE_SOURCE_VOLUME = os.environ.get("BFF_PULSE_SOURCE_VOLUME")
HEADSET_STATE_FILE = SCRIPT_DIR / "headset_state.json"


def load_headset_state() -> tuple[str | None, str | None]:
    """Last-connected headset (mac, name). Env vars override the state file."""
    mac = os.environ.get("LAST_HEADSET_MAC")
    name = os.environ.get("LAST_HEADSET_NAME")
    if mac and name:
        return mac, name
    try:
        state = json.loads(HEADSET_STATE_FILE.read_text())
        return mac or state.get("mac"), name or state.get("name")
    except (OSError, json.JSONDecodeError):
        return mac, name


def save_headset_state(mac: str, name: str) -> None:
    try:
        HEADSET_STATE_FILE.write_text(
            json.dumps({"mac": mac, "name": name}, indent=2) + "\n"
        )
    except OSError as exc:
        print(f"Could not save headset state: {exc}", file=sys.stderr)
    # Keep the in-process env in sync for anything reading it later this run
    os.environ["LAST_HEADSET_MAC"] = mac
    os.environ["LAST_HEADSET_NAME"] = name


LAST_HEADSET_MAC, LAST_HEADSET_NAME = load_headset_state()

@dataclass
class ConversationConfig:
    """Runtime configuration for the voice chat assistant."""

    ollama_model: str = DEFAULT_OLLAMA_MODEL
    vlm_model: str = DEFAULT_VLM_MODEL
    no_vlm: bool = False
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
    vad_backend: str = DEFAULT_VAD_BACKEND
    vad_threshold: float = DEFAULT_VAD_THRESHOLD
    show_levels: bool = True
    input_device_keyword: str | None = DEFAULT_INPUT_DEVICE_KEYWORD
    input_device_index: int | None = None
    output_usb_keyword: str | None = DEFAULT_OUTPUT_USB_KEYWORD
    output_bt_keyword: str | None = DEFAULT_OUTPUT_BT_KEYWORD
    output_sample_rate: int | None = DEFAULT_OUTPUT_SAMPLE_RATE
    pulse_sinks: list[str] = field(default_factory=list)
    pulse_combined_sink_name: str = DEFAULT_PULSE_COMBINED_SINK_NAME
    pulse_device_name: str = DEFAULT_PULSE_DEVICE_NAME
    pulse_source_volume: str | None = DEFAULT_PULSE_SOURCE_VOLUME
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
    simulate: bool = False
    speaker: str = os.environ.get("BFF_SPEAKER", "SNAPPER")


@dataclass
class Scene:
    """Represents a conversational scene with a specific system prompt."""
    name: str
    system_prompt: str
    trigger: str | list[str] | None = None
    speaker: str | None = None


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, stripping non-alphanumeric characters, and collapsing whitespace."""
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return " ".join(lowered.split())


def stem_text(text: str) -> str:
    """Simple stemmer removing trailing 's' from words (len > 3) for basic plural/singular matching."""
    words = text.split()
    stemmed = [w[:-1] if (len(w) > 3 and w.endswith('s') and not w.endswith('ss')) else w for w in words]
    return " ".join(stemmed)


def matches_trigger_phrase(trigger_phrase: str, user_text: str) -> bool:
    """Check if user_text matches trigger_phrase using exact, normalized, stemmed, or fuzzy matching."""
    if not trigger_phrase or not user_text:
        return False

    # 1. Direct lowercase substring match
    tp_lower = trigger_phrase.lower().strip()
    u_lower = user_text.lower().strip()
    if tp_lower in u_lower:
        return True

    # 2. Normalized substring match (ignores punctuation/apostrophes)
    norm_tp = normalize_text(trigger_phrase)
    norm_u = normalize_text(user_text)
    if not norm_tp or not norm_u:
        return False
    if norm_tp in norm_u:
        return True

    # 3. Stemmed substring match (handles color vs colors, number vs numbers)
    stem_tp = stem_text(norm_tp)
    stem_u = stem_text(norm_u)
    if stem_tp in stem_u:
        return True

    # 4. Fuzzy sliding window match for STT mishearings
    words_tp = norm_tp.split()
    words_u = norm_u.split()
    n = len(words_tp)
    if n > 1 and len(words_u) >= n:
        for i in range(len(words_u) - n + 1):
            window = " ".join(words_u[i:i + n])
            if difflib.SequenceMatcher(None, norm_tp, window).ratio() >= 0.82:
                return True

    return False


def is_scene_triggered(scene: Scene, user_text: str) -> bool:
    """Check if any trigger defined in scene matches the user's text."""
    if not scene.trigger:
        return False
    if isinstance(scene.trigger, list):
        triggers = scene.trigger
    elif isinstance(scene.trigger, str):
        triggers = [t.strip() for t in scene.trigger.split("|") if t.strip()]
    else:
        return False

    return any(matches_trigger_phrase(t, user_text) for t in triggers)


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
        "--no-vlm",
        action="store_true",
        help="Disable VLM scene captioning entirely: no VLM background worker and no VLM model preload "
        "(frees the VLM model's memory on constrained devices)",
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
        "--simulate",
        action="store_true",
        help="Skip the live robot dog check and spawn dashboard_server.py in --simulate mode "
        "instead, replaying the most recent captures/session-* recording for the VLM/telemetry feed",
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
        "--speaker",
        default=os.environ.get("BFF_SPEAKER", "SNAPPER"),
        help="Speaker name displayed for assistant responses (default: %(default)s)",
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
        "--vad-backend",
        choices=("silero", "rms"),
        default=DEFAULT_VAD_BACKEND,
        help="Speech gating backend: silero neural VAD or plain RMS amplitude "
        "(default: %(default)s; silero falls back to rms if unavailable)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=DEFAULT_VAD_THRESHOLD,
        help="Silero speech probability that starts a segment; speech ends below "
        "threshold minus 0.15 (default: %(default)s)",
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
        "--pulse-source-volume",
        default=DEFAULT_PULSE_SOURCE_VOLUME,
        help="PulseAudio input source capture volume percentage (e.g. 60%% or 50) set via pactl (default: env BFF_PULSE_SOURCE_VOLUME)",
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

    repo_root = SCRIPT_DIR
    if args.piper_voice is None:
        default_voice = repo_root / "speech/piper/en_GB-aru-medium.onnx"
        if default_voice.exists():
            args.piper_voice = default_voice
        else:
            parser.error("Piper voice model must be provided via --piper-voice or BFF_PIPER_VOICE")
    if not args.piper_voice.exists():
        rel_voice = repo_root / args.piper_voice
        if rel_voice.exists():
            args.piper_voice = rel_voice
        else:
            parts = args.piper_voice.parts
            found = False
            if "speech" in parts:
                idx = parts.index("speech")
                candidate = repo_root / Path(*parts[idx:])
                if candidate.exists():
                    args.piper_voice = candidate
                    found = True
            if not found:
                default_voice = repo_root / "speech/piper/en_GB-aru-medium.onnx"
                if default_voice.exists():
                    args.piper_voice = default_voice
                else:
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
        no_vlm=args.no_vlm,
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
        vad_backend=args.vad_backend,
        vad_threshold=args.vad_threshold,
        show_levels=args.show_levels,
        input_device_keyword=input_keyword,
        interruptable=False if args.no_interruptable else DEFAULT_INTERRUPTABLE,
        output_usb_keyword=args.output_usb_keyword.strip() if args.output_usb_keyword else None,
        output_bt_keyword=args.output_bt_keyword.strip() if args.output_bt_keyword else None,
        output_sample_rate=args.output_sample_rate,
        pulse_sinks=pulse_sinks,
        pulse_combined_sink_name=args.pulse_combined_sink_name,
        pulse_device_name=args.pulse_device_name,
        pulse_source_volume=args.pulse_source_volume,
        history_truncation_limit=DEFAULT_HISTORY_TRUNCATION_LIMIT,
        ollama_temperature=DEFAULT_OLLAMA_TEMPERATURE,
        ollama_top_p=DEFAULT_OLLAMA_TOP_P,
        ollama_top_k=DEFAULT_OLLAMA_TOP_K,
        ollama_num_predict=DEFAULT_OLLAMA_NUM_PREDICT,
        ollama_num_ctx=DEFAULT_OLLAMA_NUM_CTX,
        ollama_think=args.ollama_think,
        require_wakeword=args.require_wakeword,
        wake_phrases=[p.strip().lower() for p in args.wake_phrases.split(",") if p.strip()],
        simulate=args.simulate,
    )


def load_whisper_model(name: str, compute_type: str = "int8") -> WhisperModel:
    device = os.environ.get("BFF_WHISPER_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Faster Whisper model '{name}' on {device} ({compute_type})…", file=sys.stderr)
    try:
        return WhisperModel(name, device=device, compute_type=compute_type)
    except ValueError as exc:
        # e.g. the Jetson's CUDA-built ctranslate2 has no efficient int8
        # kernels for the ARM CPU backend; let ctranslate2 pick a supported type
        print(
            f"compute_type '{compute_type}' unsupported on {device} ({exc}); retrying with 'auto'…",
            file=sys.stderr,
        )
        return WhisperModel(name, device=device, compute_type="auto")


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
                trigger=item.get("trigger"),
                speaker=item.get("speaker")
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


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean env var using the same spelling dashboard_server.py accepts."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes", "on")


def dashboard_url(path: str = "/") -> str:
    port = os.getenv("BFF_DASHBOARD_PORT", "8080")
    return f"http://localhost:{port}{path}"


def dashboard_endpoint_ok(path: str, timeout: float = 2.0) -> bool:
    """True when the dashboard answers `path` with 200. A 503 means Flask is up
    but the WebRTC feed behind that endpoint hasn't delivered anything yet."""
    import urllib.request
    try:
        with urllib.request.urlopen(dashboard_url(path), timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def kill_stale_dashboards() -> None:
    """Clear any dashboard orphaned by a previous crash or a failed start attempt:
    it still holds the port, so a new instance dies at bind while the orphan's dead
    feed answers the index liveness check with a healthy-looking 200."""
    try:
        stale = subprocess.run(
            ["pgrep", "-f", "dashboard_server.py"],
            capture_output=True,
            text=True,
            check=False,
        )
        if stale.stdout.strip():
            print(
                f"[Chat Manager] Killing stale dashboard_server.py (PID {' '.join(stale.stdout.split())})...",
                file=sys.stderr,
            )
            subprocess.run(["pkill", "-f", "dashboard_server.py"], capture_output=True, check=False)
            time.sleep(1.0)
    except Exception:
        pass


def required_stream_endpoints(simulate: bool) -> list[tuple[str, str]]:
    """The (label, path) endpoints that must return 200 before the robot feed
    counts as usable. Only the streams this run actually enabled are checked -
    with video off, /snapshot never returns a frame and waiting on it would
    restart the dashboard forever."""
    endpoints: list[tuple[str, str]] = []
    if env_bool("BFF_CAPTURE_VIDEO", True):
        endpoints.append(("camera", "/snapshot"))
    # Playback sessions carry telemetry only if it was recorded, so in simulate
    # mode the video frames alone decide whether the feed came up.
    if not simulate and env_bool("BFF_CAPTURE_LOWSTATE", True):
        endpoints.append(("telemetry", "/lowstate"))
    return endpoints


def spawn_dashboard_server(
    session_dir: Path, session_id: str, extra_args: list[str]
) -> tuple[subprocess.Popen, Any]:
    """Launch dashboard_server.py against the current session, returning
    (process, open log file). The log is appended to so a restarted attempt
    doesn't erase the failure that caused the restart."""
    script_dir = Path(__file__).resolve().parent
    dashboard_log = open(session_dir / "dashboard_server.log", "a", encoding="utf-8")
    dashboard_env = os.environ.copy()
    dashboard_env["BFF_SESSION_ID"] = session_id
    process = subprocess.Popen(
        [sys.executable, str(script_dir / "dashboard_server.py")] + extra_args,
        cwd=str(script_dir),
        stdout=dashboard_log,
        stderr=subprocess.STDOUT,
        env=dashboard_env,
    )
    return process, dashboard_log


def wait_for_dashboard(
    process: subprocess.Popen,
    endpoints: list[tuple[str, str]],
    http_timeout: float,
    stream_timeout: float,
) -> bool:
    """Wait for the dashboard to serve its index, then for every endpoint in
    `endpoints` to stop returning 503. Returns False as soon as the subprocess
    dies or either wait times out, so the caller can restart it."""
    def process_alive() -> bool:
        if process.poll() is None:
            return True
        print(
            "[Chat Manager] Dashboard server subprocess exited unexpectedly. "
            "Check dashboard_server.log for errors.",
            file=sys.stderr,
        )
        return False

    print(f"Waiting for dashboard server to become live at {dashboard_url()}...", file=sys.stderr)
    deadline = time.time() + http_timeout
    while time.time() < deadline:
        if not process_alive():
            return False
        if dashboard_endpoint_ok("/"):
            break
        time.sleep(0.5)
    else:
        print("[Chat Manager] Dashboard server did not answer in time.", file=sys.stderr)
        return False

    if not endpoints:
        return True

    labels = ", ".join(label for label, _ in endpoints)
    print(f"Dashboard server is live. Waiting for the robot feed ({labels})...", file=sys.stderr)
    pending = list(endpoints)
    deadline = time.time() + stream_timeout
    while time.time() < deadline:
        if not process_alive():
            return False
        still_pending = []
        for label, path in pending:
            if dashboard_endpoint_ok(path):
                print(f"[Chat Manager] Robot {label} feed is up.", file=sys.stderr)
            else:
                still_pending.append((label, path))
        pending = still_pending
        if not pending:
            return True
        time.sleep(1.0)

    print(
        f"[Chat Manager] WebRTC feed never came up ({', '.join(label for label, _ in pending)} "
        f"still 503 after {stream_timeout:.0f}s).",
        file=sys.stderr,
    )
    return False


def start_dashboard_with_retries(
    session_dir: Path, session_id: str, extra_args: list[str], simulate: bool
) -> tuple[subprocess.Popen | None, Any]:
    """Start dashboard_server.py and confirm the WebRTC streams actually come up,
    restarting it when they don't. The Go2 regularly refuses the first WebRTC
    offer after boot: the capturer thread dies, Flask keeps serving 503s, and
    every consumer silently degrades (VLM to the webcam, body state to nothing).
    A restart almost always fixes it, so do it here instead of making the
    operator relaunch the whole program. Returns (process, log file); the
    process may be a feed-less dashboard if every attempt failed."""
    attempts = max(1, int(os.getenv("BFF_DASHBOARD_START_ATTEMPTS", "3")))
    http_timeout = float(os.getenv("BFF_DASHBOARD_HTTP_TIMEOUT", "15"))
    stream_timeout = float(os.getenv("BFF_DASHBOARD_STREAM_TIMEOUT", "30"))
    endpoints = required_stream_endpoints(simulate)

    process: subprocess.Popen | None = None
    dashboard_log: Any = None
    for attempt in range(1, attempts + 1):
        kill_stale_dashboards()
        print(
            f"Starting dashboard_server.py{' (simulate mode)' if simulate else ''} "
            f"(attempt {attempt}/{attempts})...",
            file=sys.stderr,
        )
        try:
            process, dashboard_log = spawn_dashboard_server(session_dir, session_id, extra_args)
        except Exception as e:
            print(f"Failed to start dashboard_server.py: {e}", file=sys.stderr)
            return None, None

        if wait_for_dashboard(process, endpoints, http_timeout, stream_timeout):
            print("Robot feed is live! Proceeding with conversation...", file=sys.stderr)
            return process, dashboard_log

        if attempt < attempts:
            print("[Chat Manager] Restarting dashboard_server.py to retry the WebRTC connection...", file=sys.stderr)
            stop_dashboard_server(process, dashboard_log)
            process, dashboard_log = None, None
            time.sleep(2.0)

    print(
        "[Warning] Robot feed did not come up after "
        f"{attempts} attempt(s). Continuing without it - the VLM will fall back to the webcam.",
        file=sys.stderr,
    )
    return process, dashboard_log


def stop_dashboard_server(process: subprocess.Popen | None, dashboard_log: Any) -> None:
    """Terminate dashboard_server.py and close its log, escalating to SIGKILL."""
    if process is not None:
        print("[Chat Manager] Terminating dashboard_server.py...", file=sys.stderr)
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            print("[Chat Manager] Force killing dashboard_server.py...", file=sys.stderr)
            process.kill()
            process.wait()
    if dashboard_log is not None:
        dashboard_log.close()


# Seconds to add to this machine's clock to get real wall time. The Jetsons
# have no RTC battery and, on the robot's network, no reachable NTP server, so
# they boot at the epoch and count forward - session folders come out named
# session-19700101-*. sync_clock_from_robot() fills this in at startup when it
# can't set the system clock outright. Zero on a correctly-timed machine.
CLOCK_OFFSET_SECONDS = 0.0


def wall_now() -> datetime:
    """Wall-clock now, corrected for a Jetson that booted without a clock.

    Named wall_now rather than now because run_conversation() already binds a
    local `now` for elapsed-time math, which would shadow this for the whole
    function and fail at the session-naming call above it."""
    return datetime.now() + timedelta(seconds=CLOCK_OFFSET_SECONDS)


def read_robot_clock(interface: str, timeout: float = 4.0) -> float | None:
    """The robot's own wall clock, from the stamp on its sportmodestate
    messages. The Go2 keeps real time across our reboots and is reachable over
    DDS on the internal network, which makes it the one time source that needs
    no internet, no laptop and no human. Returns epoch seconds, or None."""
    try:
        from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    except ImportError:
        return None

    stamped: list[float] = []

    def handler(msg) -> None:
        if not stamped:
            stamped.append(msg.stamp.sec + msg.stamp.nanosec / 1e9)

    try:
        ChannelFactoryInitialize(0, interface)
        subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        subscriber.Init(handler, 10)
        deadline = time.time() + timeout
        while not stamped and time.time() < deadline:
            time.sleep(0.05)
    except Exception as e:
        print(f"[Clock] Could not read the robot's clock over DDS: {e}", file=sys.stderr)
        return None

    if not stamped:
        return None
    # A robot that also booted without a clock is no help.
    return stamped[0] if stamped[0] > 1_600_000_000 else None


def clock_is_broken(value: float) -> bool:
    """Whether an epoch reading is too early to be now. 2025-01-01: this code
    did not exist before then, so anything earlier is a machine that booted
    without a clock."""
    return value < 1_735_689_600


def sync_clock_from_robot() -> None:
    """Correct this machine's sense of time before anything is named by it.

    Sets the system clock when that is permitted (passwordless sudo for date,
    which also fixes file mtimes and every other program on the box) and
    otherwise records an offset that now() applies. Skipped entirely when the
    local clock already looks sane, so this is a no-op on a laptop."""
    global CLOCK_OFFSET_SECONDS

    if os.getenv("BFF_CLOCK_SYNC", "true").lower() in ("0", "false", "no", "off"):
        return

    local = time.time()
    if not clock_is_broken(local):
        return

    interface = os.getenv("BFF_DDS_INTERFACE", "enP8p1s0")
    print(f"[Clock] Local clock reads {datetime.fromtimestamp(local):%Y-%m-%d %H:%M} - "
          f"asking the robot for the time on {interface}...", file=sys.stderr)
    robot_time = read_robot_clock(interface)
    if robot_time is None:
        print("[Clock] No usable time from the robot; timestamps will be wrong.", file=sys.stderr)
        return

    # Not every robot knows what year it is - helper's reports 2023. Only ever
    # correct forwards: a machine that booted without a clock is behind, never
    # ahead, so a stamp older than what we already believe is the robot being
    # wrong rather than us.
    if clock_is_broken(robot_time) or robot_time <= local:
        print(f"[Clock] The robot's clock reads {datetime.fromtimestamp(robot_time):%Y-%m-%d %H:%M}, "
              f"which is no better than ours - leaving the clock alone.", file=sys.stderr)
        return

    # Prefer the real fix. -n means never prompt: if the sudoers drop-in isn't
    # installed this fails immediately rather than blocking startup on a
    # password nobody is there to type.
    result = subprocess.run(
        ["sudo", "-n", "date", "-u", "-s", f"@{robot_time:.0f}"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 0:
        print(f"[Clock] System clock set from the robot: {datetime.now():%Y-%m-%d %H:%M:%S}", file=sys.stderr)
        return

    CLOCK_OFFSET_SECONDS = robot_time - time.time()
    print(f"[Clock] Cannot set the system clock ({result.stderr.strip() or 'sudo denied'}); "
          f"applying a {CLOCK_OFFSET_SECONDS / 3600:.1f}h offset instead - session folders and "
          f"logs will be dated correctly, file mtimes will not.", file=sys.stderr)
    print(f"[Clock] Corrected time is now {wall_now():%Y-%m-%d %H:%M:%S}.", file=sys.stderr)


def append_log_line(log_path: Path, payload: dict[str, Any]) -> None:
    record = {"timestamp": wall_now().isoformat(), **payload}
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


def set_default_bluez_source(mac: str, max_retries: int = 5, retry_delay: float = 1.0, volume: str | None = None) -> bool:
    """Make the headset's Bluetooth mic the default PulseAudio source."""
    mac_token = mac.replace(":", "_").lower()
    for attempt in range(max_retries):
        result = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            source_name = parts[1]
            if mac_token in source_name.lower() and ".monitor" not in source_name:
                subprocess.run(
                    ["pactl", "set-default-source", source_name],
                    capture_output=True,
                    check=False,
                )
                print(f"Default input source set to headset mic: {source_name}", file=sys.stderr)
                if volume:
                    set_pulse_source_volume(volume, source_name)
                return True
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    return False


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
            card_id = find_pulseaudio_card_by_mac(mac, max_retries=6, retry_delay=1.0)
            if card_id:
                subprocess.run(
                    ["pactl", "set-card-profile", card_id, "handsfree_head_unit"],
                    capture_output=True,
                    check=False,
                )
                if not set_default_bluez_source(mac, max_retries=3, retry_delay=0.5):
                    print("Headset mic source not found; input stays on the previous default.", file=sys.stderr)
                print(f"Auto-connect successful! {name} is connected and configured.", file=sys.stderr)
                return True
            # Connected at the BlueZ level but PulseAudio never attached a card:
            # cycle the connection so PulseAudio's bluez5 discovery picks it up.
            print(
                "Bluetooth is connected but PulseAudio has no card for it; cycling the connection…",
                file=sys.stderr,
            )
            subprocess.run(
                ["bluetoothctl", "disconnect", mac],
                capture_output=True,
                check=False,
                timeout=10,
            )
            time.sleep(2)

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
                ["pactl", "set-card-profile", card_id, "handsfree_head_unit"],
                capture_output=True,
                check=False,
            )
            if not set_default_bluez_source(mac):
                print("Headset mic source not found; input stays on the previous default.", file=sys.stderr)
            print(f"Auto-connect successful! Connected {name} ({mac}) in headset (HFP/HSP) mode.", file=sys.stderr)
            return True
        else:
            print(f"Bluetooth connection established to {name} ({mac}), but PulseAudio card not yet available.", file=sys.stderr)
            print(
                "Warning: the headset microphone will NOT be used; input stays on the previous default source.",
                file=sys.stderr,
            )
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
                ["pactl", "set-card-profile", card_id, "handsfree_head_unit"],
                capture_output=True,
                check=False,
            )
            print(f"Connected {name} ({mac}) in headset (HFP/HSP) mode.", file=sys.stderr)
        else:
            print(f"Bluetooth connection established to {name} ({mac}), but PulseAudio card not yet available.", file=sys.stderr)
            print(f"This is normal - the device should work as the system default audio device.", file=sys.stderr)
        
        save_headset_state(mac, name)
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
        connect_to_headset(mac, name)  # persists headset state on success
    except (ValueError, KeyboardInterrupt, EOFError):
        print("Cancelled or invalid input.", file=sys.stderr)


def find_input_device(keyword: str, min_channels: int = 1) -> int | None:
    if not keyword:
        return None
    keywords = [k.strip().lower() for k in keyword.replace("|", ",").split(",") if k.strip()]
    
    for idx, device in enumerate(sd.query_devices()):
        name = device.get("name", "").lower()
        if device.get("max_input_channels", 0) >= min_channels:
            if any(kw in name for kw in keywords):
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
                # Reuse only if it still points at the sinks we want now — a
                # combined sink left over from a run with different hardware
                # (e.g. USB card since unplugged) would silently route nowhere.
                current_slaves: set[str] = set()
                for field_ in line.split():
                    if field_.startswith("slaves="):
                        current_slaves = set(field_[len("slaves="):].split(","))
                if current_slaves == set(sinks):
                    return sink_name
                print(
                    f"Recreating PulseAudio combined sink '{sink_name}': "
                    f"slaves {sorted(current_slaves)} -> {sorted(sinks)}",
                    file=sys.stderr,
                )
                subprocess.run(
                    ["pactl", "unload-module", line.split()[0]],
                    capture_output=True,
                    check=False,
                )
                break
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
        kw = (usb_keyword or "usb").lower()
        if kw in sink_lower or "usb" in sink_lower:
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


def set_pulse_source_volume(volume: str | None, source_name: str = "@DEFAULT_SOURCE@") -> bool:
    """Set PulseAudio input source capture volume via pactl (e.g. '60%' or '50')."""
    if not volume or not is_pulseaudio_available():
        return False
    vol_str = str(volume).strip()
    if vol_str.isdigit():
        vol_str = f"{vol_str}%"
    try:
        res = subprocess.run(
            ["pactl", "set-source-volume", source_name, vol_str],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            print(f"PulseAudio input source '{source_name}' volume set to {vol_str}", file=sys.stderr)
            return True
        else:
            print(f"Warning: Failed to set PulseAudio source volume for '{source_name}': {res.stderr.strip()}", file=sys.stderr)
            return False
    except Exception as exc:
        print(f"Warning: Error setting PulseAudio source volume: {exc}", file=sys.stderr)
        return False


def resolve_output_devices(config: ConversationConfig) -> list[str]:
    # 1. If PulseAudio is available, use the PulseAudio setup.
    if is_pulseaudio_available():
        output_indices: list[str] = []
        pulse_sinks = config.pulse_sinks
        if not pulse_sinks:
            pulse_sinks = auto_detect_pulse_sinks(
                config.output_usb_keyword, config.output_bt_keyword
            )
        if pulse_sinks:
            if len(pulse_sinks) == 1:
                target_sink = pulse_sinks[0]
            else:
                target_sink = ensure_pulse_combined_sink(
                    pulse_sinks, config.pulse_combined_sink_name
                ) or pulse_sinks[0]

            if target_sink:
                set_default_pulse_sink(target_sink)
                subprocess.run(["pactl", "set-sink-mute", target_sink, "0"], capture_output=True, check=False)
                subprocess.run(["pactl", "set-sink-volume", target_sink, "80%"], capture_output=True, check=False)
                print(f"[Audio] Set default PulseAudio output sink to: '{target_sink}'", file=sys.stderr)

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


VAD_SAMPLE_RATE = 16000
VAD_FRAME_SAMPLES = 512  # Silero v5 accepts only 512-sample frames at 16 kHz


class SileroSpeechDetector:
    """Streaming speech probability from the Silero VAD model (CPU, ~1 ms/frame).

    The model is stateful (LSTM) and consumes fixed 512-sample frames at
    16 kHz, so incoming mic blocks are buffered into frames here and state
    must be cleared between utterances via reset().
    """

    def __init__(self):
        try:
            from silero_vad import load_silero_vad

            self.model = load_silero_vad()
        except ImportError:
            # Package not installed; torch.hub downloads to ~/.cache once.
            self.model = torch.hub.load(
                "snakers4/silero-vad", "silero_vad", trust_repo=True
            )[0]
        self._pending = np.empty(0, dtype=np.float32)
        self._last_prob = 0.0

    def speech_probability(self, block: np.ndarray, source_rate: int) -> float:
        """Max speech probability over the complete frames in this block.

        Returns the previous probability if the block is shorter than one
        frame (the remainder stays buffered for the next call).
        """
        mono = block[:, 0] if block.ndim > 1 else block
        mono = resample_audio(mono.astype(np.float32), source_rate, VAD_SAMPLE_RATE)
        self._pending = np.concatenate((self._pending, mono))
        prob: float | None = None
        with torch.inference_mode():
            while len(self._pending) >= VAD_FRAME_SAMPLES:
                frame = torch.from_numpy(self._pending[:VAD_FRAME_SAMPLES].copy())
                self._pending = self._pending[VAD_FRAME_SAMPLES:]
                p = float(self.model(frame, VAD_SAMPLE_RATE).item())
                prob = p if prob is None else max(prob, p)
        if prob is not None:
            self._last_prob = prob
        return self._last_prob

    def reset(self) -> None:
        self.model.reset_states()
        self._pending = np.empty(0, dtype=np.float32)
        self._last_prob = 0.0


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
    """Queue of pending playback chunks, drained by the audio callback.

    Chunks are kept as a deque rather than one concatenated array so neither
    side ever holds the mutex for a large copy — the callback runs on the
    audio thread and must not block.
    """

    def __init__(self):
        self.chunks: collections.deque[np.ndarray] = collections.deque()
        self.buffer_mutex = threading.Lock()
        self.interrupt_event = threading.Event()
        self.is_playing = False

    def play_chunk(self, chunk: np.ndarray):
        if chunk.ndim > 1:
            chunk = chunk.flatten()
        chunk = chunk.astype(np.float32)
        with self.buffer_mutex:
            self.chunks.append(chunk)
            self.is_playing = True

    def clear(self):
        with self.buffer_mutex:
            self.chunks.clear()
            self.is_playing = False

    def queued_frames(self) -> int:
        """Frames still pending in the buffer, i.e. how far playback is ahead."""
        with self.buffer_mutex:
            return sum(len(c) for c in self.chunks)

    def fill_output(self, outdata: np.ndarray, frames: int):
        with self.buffer_mutex:
            if self.interrupt_event.is_set():
                self.chunks.clear()
                self.is_playing = False
                self.interrupt_event.clear()

            offset = 0
            while offset < frames and self.chunks:
                head = self.chunks[0]
                take = min(len(head), frames - offset)
                outdata[offset:offset + take, 0] = head[:take]
                if take == len(head):
                    self.chunks.popleft()
                else:
                    self.chunks[0] = head[take:]
                offset += take

            if offset < frames:
                outdata[offset:, 0] = 0.0
            if not self.chunks:
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

    # Only the duplex (CoreAudio) path shares one device with other audio clients.
    # The PulseAudio path goes through the sound server, which does its own mixing
    # and rate conversion, so leave the Jetson deployments on their existing rate.
    prefer_native_rate = os.environ.get(
        "BFF_NATIVE_STREAM_RATE",
        "false" if is_pulseaudio_available() else "true",
    ).lower() in ("true", "1", "yes", "on")

    stream_samplerate = config.sample_rate
    input_channels = 1

    # CoreAudio accepts an off-native rate without complaint and resamples inside the
    # IO proc, so check_input_settings alone never steers us to the native rate. Running
    # the duplex stream off-native makes the hardware rate disagree with other clients on
    # the device (notably the dashboard's 48kHz AudioContext), and each reconciliation is
    # audible as a dropout. Prefer the device rate and resample down to config.sample_rate
    # for VAD/STT below. On macOS input_device_index is deliberately None (system default
    # device), so resolve the rate from the default input rather than skipping this.
    if prefer_native_rate:
        try:
            rate_probe = sd.query_devices(
                config.input_device_index if config.input_device_index is not None else None,
                kind="input",
            )
            native_sr = int(rate_probe.get("default_samplerate", 0))
            if native_sr and native_sr != config.sample_rate:
                sd.check_input_settings(device=config.input_device_index, samplerate=native_sr)
                stream_samplerate = native_sr
                print(
                    f"[Audio] Running input '{rate_probe.get('name', 'default')}' at its native "
                    f"{native_sr}Hz; resampling to {config.sample_rate}Hz for VAD/STT.",
                    file=sys.stderr,
                )
        except Exception as probe_err:
            print(f"[Audio] Native rate probe failed, using {config.sample_rate}Hz: {probe_err}", file=sys.stderr)

    if config.input_device_index is not None:
        try:
            dev_info = sd.query_devices(config.input_device_index)
            input_channels = max(1, min(int(dev_info.get("max_input_channels", 1)), 2))
            sd.check_input_settings(device=config.input_device_index, samplerate=stream_samplerate)
        except Exception:
            dev_info = sd.query_devices(config.input_device_index)
            native_sr = int(dev_info.get("default_samplerate", 48000))
            try:
                sd.check_input_settings(device=config.input_device_index, samplerate=native_sr)
                stream_samplerate = native_sr
                print(
                    f"[Audio] Input device #{config.input_device_index} ('{dev_info['name']}') "
                    f"does not support {config.sample_rate}Hz; recording at native {stream_samplerate}Hz and resampling to {config.sample_rate}Hz.",
                    file=sys.stderr,
                )
            except Exception as sr_err:
                print(f"[Audio] Input device sample rate check warning: {sr_err}", file=sys.stderr)

    global CURRENT_STREAM_SAMPLERATE
    CURRENT_STREAM_SAMPLERATE = stream_samplerate

    block_size = max(1, int(stream_samplerate * config.block_duration))
    silence_blocks_required = max(1, int(config.silence_duration / config.block_duration))
    max_blocks = max(1, int(config.max_record_seconds / config.block_duration))
    min_blocks = max(1, int(config.min_phrase_seconds / config.block_duration))

    q: queue.Queue[np.ndarray] = queue.Queue()

    def audio_callback(indata, outdata, frames, time_info, status):
        now = time.time()
        mute_mic = is_assistant_speaking or (now - last_assistant_speech_time < 0.4)
        if not mute_mic or config.interruptable:
            q.put(indata.copy())
        if outdata is not None:
            unified_audio.fill_output(outdata, frames)

    vad: SileroSpeechDetector | None = None
    if config.vad_backend == "silero":
        try:
            print("Loading Silero VAD…", file=sys.stderr)
            vad = SileroSpeechDetector()
        except Exception as exc:
            print(
                f"Silero VAD unavailable ({exc}); falling back to RMS gating.",
                file=sys.stderr,
            )
    # Hysteresis: speech starts above vad_threshold, only counts as silence
    # well below it, mirroring the activation/silence RMS threshold pair.
    vad_end_threshold = max(0.15, config.vad_threshold - 0.15)

    print("Listening continuously… (Ctrl+C to exit)")
    if is_pulseaudio_available():
        stream_ctx = sd.InputStream(
            samplerate=stream_samplerate,
            channels=input_channels,
            dtype="float32",
            blocksize=block_size,
            callback=lambda indata, frames, time_info, status: audio_callback(indata, None, frames, time_info, status),
            device=config.input_device_index,
        )
    else:
        stream_ctx = sd.Stream(
            samplerate=stream_samplerate,
            channels=(input_channels, 1),
            dtype="float32",
            blocksize=block_size,
            callback=audio_callback,
            device=(
                config.input_device_index,
                config.output_device_indices[0] if config.output_device_indices else None
            ),
        )

    with stream_ctx:
        recording = False
        silence_blocks = 0
        collected: List[np.ndarray] = []
        block_counter = 0

        while True:
            if stop_event and stop_event.is_set():
                break
            try:
                raw_block = q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Mix multi-channel input to mono if needed
            if raw_block.ndim > 1:
                raw_block = raw_block.mean(axis=1)

            # Resample hardware stream rate down to 16kHz for VAD and STT
            if stream_samplerate != config.sample_rate:
                block = resample_audio(raw_block, stream_samplerate, config.sample_rate)
            else:
                block = raw_block
            block_counter += 1 if recording else 0
            amp = rms_amplitude(block)

            if vad is not None:
                prob = vad.speech_probability(block, config.sample_rate)
                voice_detected = prob >= config.vad_threshold
                silence_detected = prob < vad_end_threshold
            else:
                prob = None
                voice_detected = amp >= config.activation_threshold
                silence_detected = amp < config.silence_threshold

            if config.show_levels:
                meter_width = 40
                if prob is not None:
                    normalized = min(1.0, prob)
                    label = f"Speech {prob:0.2f}"
                else:
                    normalized = min(1.0, amp / max(config.activation_threshold, 1e-6))
                    label = f"Level {amp:0.3f}"
                filled = int(normalized * meter_width)
                bar = "#" * filled + "-" * (meter_width - filled)
                suffix = "REC"
                if recording and voice_detected:
                    suffix = "REC (*)"
                sys.stderr.write(
                    f"\r{label} |{bar}| {suffix}"
                )
                sys.stderr.flush()

            if not recording:
                if voice_detected:
                    # Voice activity detected - pause immediately
                    if on_voice_activity:
                        on_voice_activity()
                    recording = True
                    collected = [block]
                    silence_blocks = 0
                    block_counter = 1
            else:
                collected.append(block)
                if silence_detected:
                    silence_blocks += 1
                else:
                    silence_blocks = 0

                if silence_blocks >= silence_blocks_required or block_counter >= max_blocks:
                    duration = len(collected) * config.block_duration
                    recording = False
                    silence_blocks = 0
                    block_counter = 0
                    if vad is not None:
                        vad.reset()

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
                "num_ctx": DEFAULT_OLLAMA_NUM_CTX,
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
    voice_config = getattr(voice, "config", None)
    base_sample_rate = int(
        getattr(voice_config, "sample_rate", getattr(voice, "sample_rate", 22050))
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
                is_first_chunk = True
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
                    
                    text_label = text if is_first_chunk else None
                    self.audio_queue.put((text_label, audio_array))
                    is_first_chunk = False
            except Exception as e:
                print(f"TTS Error: {e}", file=sys.stderr)
        
        self.audio_queue.put(None) # End of audio stream


class DashboardAudioRelay:
    """Non-blocking asynchronous worker queue for sending PCM audio streams to dashboard server."""
    def __init__(self):
        self.queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=300)
        self.worker_thread = threading.Thread(target=self._run, daemon=True)
        self.worker_thread.start()

    def start_stream(self, sample_rate: int) -> None:
        self.clear()
        try:
            self.queue.put_nowait(("start", sample_rate))
        except queue.Full:
            pass

    def send_chunk(self, chunk: np.ndarray, sample_rate: int) -> None:
        try:
            self.queue.put_nowait(("chunk", (chunk.copy(), sample_rate)))
        except queue.Full:
            pass

    def end_stream(self) -> None:
        try:
            self.queue.put_nowait(("end", None))
        except queue.Full:
            pass

    def interrupt_stream(self) -> None:
        self.clear()
        try:
            self.queue.put_nowait(("interrupt", None))
        except queue.Full:
            pass

    def clear(self) -> None:
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def _run(self) -> None:
        # One keep-alive connection for the whole run. A fresh TCP connect per 80ms
        # chunk makes the worker fall behind in bursts, and the browser schedules those
        # bursts back-to-back into the future — which is what pushes web playback out of
        # sync with the local device. Reusing the socket keeps delivery paced with audio.
        conn: http.client.HTTPConnection | None = None
        dashboard_port = int(os.getenv("BFF_DASHBOARD_PORT", "8080"))

        while True:
            cmd, data = self.queue.get()
            try:
                if cmd == "start":
                    path = "/audio_start"
                    payload = json.dumps({"sample_rate": data}).encode("utf-8")
                elif cmd == "chunk":
                    chunk, sample_rate = data
                    if chunk.ndim > 1:
                        chunk = chunk.flatten()
                    clipped = np.clip(chunk, -1.0, 1.0)
                    int16_pcm = (clipped * 32767).astype(np.int16)
                    b64_data = base64.b64encode(int16_pcm.tobytes()).decode("ascii")
                    path = "/audio_chunk"
                    payload = json.dumps({"data": b64_data, "sample_rate": sample_rate}).encode("utf-8")
                elif cmd == "end":
                    path = "/audio_end"
                    payload = b"{}"
                elif cmd == "interrupt":
                    path = "/audio_interrupt"
                    payload = b"{}"
                else:
                    continue

                if conn is None:
                    conn = http.client.HTTPConnection("127.0.0.1", dashboard_port, timeout=1.0)
                conn.request("POST", path, body=payload,
                             headers={"Content-Type": "application/json"})
                # Drain the body or the socket cannot be reused for the next chunk.
                conn.getresponse().read()
            except Exception:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None


dashboard_audio_relay = DashboardAudioRelay()


def send_dashboard_audio_start(sample_rate: int) -> None:
    """Notify dashboard server that speech audio streaming is starting (non-blocking)."""
    dashboard_audio_relay.start_stream(sample_rate)


def send_dashboard_audio_chunk(chunk: np.ndarray, sample_rate: int) -> None:
    """Send a PCM audio chunk to dashboard server for real-time Web Audio playback (non-blocking)."""
    dashboard_audio_relay.send_chunk(chunk, sample_rate)


def send_dashboard_audio_end() -> None:
    """Notify dashboard server that speech audio streaming has completed (non-blocking)."""
    dashboard_audio_relay.end_stream()


def send_dashboard_audio_interrupt() -> None:
    """Notify dashboard server that speech audio playback was interrupted (non-blocking)."""
    dashboard_audio_relay.interrupt_stream()


def drain_playback_tail(
    output_stream: Any | None,
    target_rate: int,
    frames_written: int = 0,
    stream_started_at: float | None = None,
    interrupt_event: threading.Event | None = None,
) -> None:
    """Let the last word finish before the caller tears the stream down.

    Writing the final chunk only hands it to the sink. sounddevice documents
    stop() as waiting for pending buffers, but it does not on the Jetson's
    ALSA->PulseAudio sink: measured there, stop() returned with 0.41 s of a
    1.5 s tone still unplayed, which is exactly the clipped tail. The stream's
    reported latency (0.035 s) understates the server-side buffer by an order
    of magnitude and is useless for sizing the wait, so derive what is still in
    flight from how much audio has been written against how long the stream has
    been open. On the unified_audio path nothing waits for the callback to drain
    the deque at all, so wait on the queue itself there.

    Also keeps the caller's `last_assistant_speech_time` honest - barge-in echo
    gating measures from when sound stopped, not from when the last chunk was
    queued. Returns early on interrupt: a barge-in *wants* the tail cut."""
    pad_frames = int(target_rate * PLAYBACK_TAIL_SILENCE)
    silence = np.zeros(max(pad_frames, 0), dtype=np.float32)

    if output_stream is not None:
        # A short pad first, so anything the sink drops at teardown (a partial
        # final period) is silence rather than the release of the last word.
        if pad_frames > 0:
            try:
                output_stream.write(silence)
                frames_written += pad_frames
            except Exception:
                pass
        if stream_started_at is None:
            return
        pending = frames_written / float(target_rate) - (time.time() - stream_started_at)
        deadline = time.time() + min(max(pending, 0.0), 5.0)
        while time.time() < deadline:
            if interrupt_event is not None and interrupt_event.is_set():
                return
            time.sleep(0.02)
        return

    if pad_frames > 0:
        unified_audio.play_chunk(silence)
    deadline = time.time() + PLAYBACK_TAIL_SILENCE + 2.0
    pending = unified_audio.queued_frames()
    last_progress = time.time()
    while pending > 0 and time.time() < deadline:
        if interrupt_event is not None and interrupt_event.is_set():
            return
        time.sleep(0.02)
        remaining = unified_audio.queued_frames()
        if remaining < pending:
            last_progress = time.time()
        elif time.time() - last_progress > 0.5:
            # Nobody is draining the deque - the duplex callback only exists on
            # the CoreAudio path, so if the dedicated stream failed to open on a
            # PulseAudio host there is no consumer. Don't stall every turn.
            return
        pending = remaining


def play_audio_stream(
    audio_queue: queue.Queue[tuple[str | None, np.ndarray] | np.ndarray | None],
    sample_rate: int,
    interrupt_event: threading.Event,
    interruptable: bool = True,
    save_path: Path | None = None,
    output_device_indices: list[str] | None = None,
    output_sample_rate: int | None = None,
    on_chunk_playback: Callable[[str], None] | None = None,
) -> None:
    global is_assistant_speaking, last_assistant_speech_time
    is_assistant_speaking = True
    output_stream = None
    was_interrupted = False
    try:
        wav_file = None
        if save_path:
            wav_file = wave.open(str(save_path), "wb")
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2) # 16-bit PCM
            wav_file.setframerate(sample_rate)

        target_rate = CURRENT_STREAM_SAMPLERATE
        send_dashboard_audio_start(target_rate)
        # Written-vs-elapsed bookkeeping for drain_playback_tail.
        stream_started_at: float | None = None
        frames_written = 0

        # On the PulseAudio path phrase_stream opens an input-only InputStream, so
        # nothing else owns the output device and we need our own stream. On the
        # duplex path (macOS) phrase_stream already holds an sd.Stream on this same
        # device and drains unified_audio from its callback — opening a second
        # stream there makes the two contend and drop out.
        if is_pulseaudio_available():
            try:
                dev_idx = output_device_indices[0] if output_device_indices else None
                output_stream = sd.OutputStream(
                    samplerate=target_rate,
                    channels=1,
                    dtype="float32",
                    device=dev_idx,
                    latency="high",
                )
                output_stream.start()
                stream_started_at = time.time()
            except Exception as stream_err:
                print(f"[Audio] Warning: Could not open dedicated output stream: {stream_err}", file=sys.stderr)
                output_stream = None

        # Slicing chunk into 50-100ms blocks (~80ms) for tight hardware/web audio sync
        chunk_size_samples = max(256, int(target_rate * 0.08))
        # How far unified_audio playback may run ahead of this loop. Writing to the
        # dedicated stream blocks for the chunk duration and paces itself; the
        # callback-driven path does not, so throttle on buffer depth instead to keep
        # on_chunk_playback text logging aligned with what is actually audible.
        max_backlog_frames = int(target_rate * 0.25)

        while True:
            if interruptable and interrupt_event.is_set():
                was_interrupted = True
                unified_audio.clear()
                send_dashboard_audio_interrupt()
                break

            try:
                item = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                # End of stream
                break

            if isinstance(item, tuple):
                text_label, chunk = item
            else:
                text_label, chunk = None, item

            if text_label and on_chunk_playback:
                try:
                    on_chunk_playback(text_label)
                except Exception as cb_err:
                    print(f"on_chunk_playback callback error: {cb_err}", file=sys.stderr)

            # Process chunk for playing
            play_chunk = chunk.copy()
            if sample_rate != target_rate:
                play_chunk = resample_audio(play_chunk, sample_rate, target_rate)

            # Write original chunk to wav file if needed
            if wav_file:
                w_chunk = play_chunk
                if w_chunk.ndim == 1:
                    w_chunk = w_chunk[:, np.newaxis]
                clipped = np.clip(w_chunk, -1.0, 1.0)
                int16_data = (clipped * 32767).astype(np.int16)
                wav_file.writeframes(int16_data.tobytes())

            # Sub-chunk streaming in ~80ms blocks to align USB hardware & Web Audio clocks
            if play_chunk.ndim > 1:
                play_chunk = play_chunk.flatten()

            for start_i in range(0, len(play_chunk), chunk_size_samples):
                if interruptable and interrupt_event.is_set():
                    was_interrupted = True
                    unified_audio.clear()
                    send_dashboard_audio_interrupt()
                    break

                sub_chunk = play_chunk[start_i : start_i + chunk_size_samples]

                # 1. Relays PCM to web dashboard (queued, never blocks this thread)
                send_dashboard_audio_chunk(sub_chunk, target_rate)

                # 2. Output to local audio device
                if output_stream is not None:
                    output_stream.write(sub_chunk.astype(np.float32))
                    frames_written += len(sub_chunk)
                else:
                    unified_audio.play_chunk(sub_chunk)
                    while unified_audio.queued_frames() > max_backlog_frames:
                        if interruptable and interrupt_event.is_set():
                            break
                        time.sleep(0.02)

            if was_interrupted:
                break

        if not was_interrupted:
            drain_playback_tail(
                output_stream,
                target_rate,
                frames_written=frames_written,
                stream_started_at=stream_started_at,
                interrupt_event=interrupt_event if interruptable else None,
            )

        if wav_file:
            wav_file.close()

    except Exception as e:
        print(f"Playback Error: {e}", file=sys.stderr)
    finally:
        if output_stream is not None:
            try:
                output_stream.stop()
                output_stream.close()
            except Exception:
                pass
        if was_interrupted:
            send_dashboard_audio_interrupt()
        else:
            send_dashboard_audio_end()
        is_assistant_speaking = False
        last_assistant_speech_time = time.time()


def play_audio(
    audio_path: Path,
    interrupt_event: threading.Event,
    interruptable: bool = True,
    output_device_indices: list[str] | None = None,
    output_sample_rate: int | None = None,
) -> bool:
    global is_assistant_speaking, last_assistant_speech_time
    is_assistant_speaking = True
    was_interrupted = False
    try:
        data, samplerate = sf.read(audio_path, dtype="float32")

        target_rate = CURRENT_STREAM_SAMPLERATE
        if samplerate != target_rate:
            data = resample_audio(data, samplerate, target_rate)
            samplerate = target_rate

        if data.ndim > 1:
            data = data.mean(axis=1) # mix down to mono

        send_dashboard_audio_start(samplerate)
        interrupt_event.clear()

        chunk_size = max(256, int(samplerate * 0.08))

        if is_pulseaudio_available() and shutil.which("paplay"):
            cmd = ["paplay"]
            try:
                sinks = list_pulse_sinks()
                usb_sinks = [s for s in sinks if "usb" in s.lower()]
                if usb_sinks:
                    cmd.append(f"--device={usb_sinks[0]}")
            except Exception:
                pass
            cmd.append(str(audio_path))

            proc = subprocess.Popen(cmd)
            # Send chunks to dashboard in background while paplay plays locally
            for i in range(0, len(data), chunk_size):
                if interruptable and interrupt_event.is_set():
                    proc.terminate()
                    was_interrupted = True
                    send_dashboard_audio_interrupt()
                    return False
                sub_chunk = data[i : i + chunk_size]
                send_dashboard_audio_chunk(sub_chunk, samplerate)
                time.sleep(len(sub_chunk) / float(samplerate))
            proc.wait()
            return proc.returncode == 0
        else:
            stream = sd.OutputStream(
                samplerate=samplerate,
                channels=1,
                dtype="float32",
                device=output_device_indices[0] if output_device_indices else None,
            )
            with stream:
                stream_started_at = time.time()
                frames_written = 0
                for i in range(0, len(data), chunk_size):
                    if interruptable and interrupt_event.is_set():
                        was_interrupted = True
                        send_dashboard_audio_interrupt()
                        return False
                    sub_chunk = data[i : i + chunk_size]
                    send_dashboard_audio_chunk(sub_chunk, samplerate)
                    stream.write(sub_chunk)
                    frames_written += len(sub_chunk)
                drain_playback_tail(
                    stream,
                    samplerate,
                    frames_written=frames_written,
                    stream_started_at=stream_started_at,
                )
            return True
    except Exception as e:
        print(f"Playback Error: {e}", file=sys.stderr)
        return False
    finally:
        if was_interrupted:
            send_dashboard_audio_interrupt()
        else:
            send_dashboard_audio_end()
        is_assistant_speaking = False
        last_assistant_speech_time = time.time()


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
        
        # Minimum gap between capture checks. Set to 0 for back-to-back checking.
        self.vlm_interval = float(os.getenv("BFF_VLM_INTERVAL", "1.5"))

        # Only actually query the VLM (and save a snapshot) when the scene has
        # visibly changed since the last description, checked via a cheap
        # grayscale frame-difference on every capture tick - the VLM query
        # itself is the expensive step, so most ticks should just look and
        # skip. Mean per-pixel difference (0-255 scale) above this threshold
        # counts as "changed"; higher = less sensitive to lighting/noise.
        self.change_threshold = float(os.getenv("BFF_VLM_CHANGE_THRESHOLD", "12.0"))
        self._last_compared_frame = None  # small grayscale frame, for change detection

        # Fallback velocity estimate by SLAM position delta, used only when
        # sport_state is missing from the payload (e.g. replaying an older
        # session capture) - LowState_ has no velocity field on this hardware.
        self._last_slam_position = None
        self._last_slam_time = None

        # Console-logging state. The body state is polled every couple of
        # seconds but only printed when it says something new, or when a fresh
        # scene description means the model's whole context has moved.
        self._last_logged_state_key = None
        self._last_logged_scene = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def _run(self):
        global last_vlm_query_time
        print("[VLM Worker] Started background VLM worker thread (change-triggered capture).", file=sys.stderr)

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
        for injection into the LLM prompt: charge, stance, movement, and the
        thermal gradient from battery to IMU. Deliberately narrow - this is
        what the robot should be able to feel and talk about, not everything
        the payload carries. Foot contact is excluded because it returns
        nothing valid on the Go2 Pro. LowState_ has no velocity field on this
        hardware either, so velocity, stance height and IMU temperature come
        from the sportmodestate subscription injected as sport_state, falling
        back to slam_pose position deltas for speed when absent."""
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

            bms = data.get("bms_state") or {}
            power_v = data.get("power_v", 0.0)
            bms_soc = bms.get("soc")
            if bms_soc is not None:
                battery_pct = float(bms_soc)
            else:
                # Fallback voltage-based estimate if bms_state.soc isn't
                # available. Overstates a lithium pack badly - it reads ~97%
                # at an actual 55% - so it really is a last resort.
                battery_pct = max(0.0, min(100.0, ((power_v - 22.0) / (29.6 - 22.0)) * 100))

            # Body-frame velocity straight from the sport controller's state
            # estimator - no SLAM jumps, so no spike guard needed.
            speed = 0.0
            velocity = (data.get("sport_state") or {}).get("velocity")
            if velocity is not None:
                speed = math.hypot(velocity[0], velocity[1])
            else:
                # Older captures have no sport_state: estimate by SLAM position
                # delta between consecutive polls instead.
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

            sport_state = data.get("sport_state") or {}

            # Posture and movement as words, with the number only where it
            # carries information. A bare "0.00 m/s" invites the model to
            # narrate zeros; "still" is what the body actually feels.
            stance = "standing" if (sport_state.get("body_height") or 0.0) > 0.2 else "lying down"
            motion = "still" if speed < 0.05 else f"moving at {speed:.2f} m/s"

            # Thermal gradient across the whole frame, sorted coolest to
            # hottest so the ordering *is* the gradient - if load ever pushes a
            # joint past the chassis, the sentence reorders itself rather than
            # asserting a fixed story. Nothing here shares a word with the
            # charge reading above: "battery 97%" next to "battery pack 24°C"
            # is what made the model call a cool pack hot. motor_state pads out
            # to 20 entries; only the first 12 are real motors.
            readings = []
            bq_ntc = bms.get("bq_ntc") or []
            if bq_ntc:
                # Hottest of the two pack thermistors - heat matters where it peaks.
                readings.append((float(max(bq_ntc)), "power cell", "cell"))
            motors = (data.get("motor_state") or [])[:12]
            if len(motors) == 12:
                temps = [m.get("temperature", 0) for m in motors]
                hottest = max(range(12), key=lambda i: temps[i])
                coolest = min(range(12), key=lambda i: temps[i])
                readings.append((float(temps[coolest]), f"coolest joint {JOINT_NAMES[coolest]}", "joint"))
                readings.append((float(temps[hottest]), f"warmest joint {JOINT_NAMES[hottest]}", "joint"))
            board_temp = data.get("temperature_ntc1")
            if board_temp:
                readings.append((float(board_temp), "chassis", "chassis"))
            imu_temp = sport_state.get("imu_temperature")
            if imu_temp:
                readings.append((float(imu_temp), "sensing core", "core"))

            warmth_part = ""
            if readings:
                readings.sort()
                items = []
                running_hot = []
                for value, label, kind in readings:
                    verdict = describe_temperature(value, kind)
                    if verdict == "hot":
                        running_hot.append(label)
                    if kind == "core" and verdict == "cool":
                        # The IMU die sits at ~79°C forever. Calling that "cool"
                        # invites alarm; calling it usual is what it is.
                        items.append(f"{label} at its usual {value:.0f}°C")
                    elif kind == "cell" or verdict != "cool":
                        # Always qualify the cell - it is the reading that gets
                        # misread - and anything that isn't cool.
                        items.append(f"{label} {verdict} at {value:.0f}°C")
                    else:
                        items.append(f"{label} {value:.0f}°C")
                headline = f"{', '.join(running_hot)} running hot" if running_hot else "nothing running hot"
                warmth_part = f", {headline}: {', '.join(items)}"

            summary = f"charge {battery_pct:.0f}%, {stance}, {motion}{warmth_part}"
            with body_state_lock:
                latest_body_state = summary

            # Log only when the packet means something new. Raw temperatures
            # jitter by a degree between polls, so compare the shape of the
            # state - charge to the nearest 5%, posture, moving or not, and
            # which joints are the extremes - rather than the string itself.
            with vlm_lock:
                current_scene = latest_scene_description
            state_key = (
                round(battery_pct / 5.0),
                stance,
                motion != "still",
                readings[0][1] if readings else None,
                readings[-1][1] if readings else None,
            )
            if state_key != self._last_logged_state_key or current_scene != self._last_logged_scene:
                self._last_logged_state_key = state_key
                self._last_logged_scene = current_scene
                log_context_packet(BODY_STATE_PREFIX, summary)
        except Exception as e:
            print(f"[Body State] Failed to fetch lowstate: {e}", file=sys.stderr)

    def _acquire_frame(self):
        """Fetch a frame from the dashboard's /snapshot, falling back to the local webcam."""
        frame = None
        dashboard_port = os.getenv("BFF_DASHBOARD_PORT", "8080")
        url = f"http://localhost:{dashboard_port}/snapshot"
        try:
            import urllib.request
            import numpy as np
            with urllib.request.urlopen(url, timeout=3.0) as response:
                img_data = response.read()
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
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

        return frame

    def _scene_has_changed(self, frame) -> bool:
        """Cheap grayscale frame-difference check against the last frame that
        was actually described. Keeps the (slow) VLM query from re-describing
        a static scene on every tick - only fires when something's different."""
        small = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(small, (64, 64), interpolation=cv2.INTER_AREA)

        if self._last_compared_frame is None:
            self._last_compared_frame = small
            return True  # nothing to compare against yet - always describe the first frame

        mean_diff = float(cv2.absdiff(small, self._last_compared_frame).mean())
        changed = mean_diff >= self.change_threshold
        if changed:
            self._last_compared_frame = small
        return changed

    def _capture_and_query(self):
        global latest_scene_description, last_vlm_query_time

        # Update immediately to pace the next check, regardless of whether
        # this tick ends up actually querying the VLM.
        last_vlm_query_time = time.time()

        frame = self._acquire_frame()
        if frame is None:
            print("[VLM Worker] Capture failed. No image retrieved.", file=sys.stderr)
            return

        if not self._scene_has_changed(frame):
            return

        # Create output directory for VLM captures inside session directory
        vlm_dir = self.session_dir / "vlm_captures"
        vlm_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = wall_now().strftime("%Y%m%d-%H%M%S")
        image_path = vlm_dir / f"snapshot_{timestamp_str}.jpg"
        cv2.imwrite(str(image_path), frame)
        if WORKER_VERBOSE:
            print(f"[VLM Worker] Scene changed - snapshot saved to {image_path}", file=sys.stderr)

        # Fetch latest YOLO detections from local dashboard server if available
        yolo_summary = ""
        try:
            import urllib.request
            dashboard_port = getattr(self.config, "dashboard_port", 8080)
            det_url = f"http://127.0.0.1:{dashboard_port}/detections"
            with urllib.request.urlopen(det_url, timeout=0.5) as resp:
                det_json = json.loads(resp.read().decode("utf-8"))
                dets = det_json.get("detections", [])
                if dets:
                    classes = [
                        d["class"] for d in dets 
                        if isinstance(d, dict) and d.get("class") and d.get("confidence", 0.0) >= 0.3
                    ]
                    if classes:
                        unique_classes = list(dict.fromkeys(classes))
                        yolo_summary = ", ".join(unique_classes)
        except Exception:
            pass

        # Query Ollama VLM
        model_name = self.config.vlm_model
        prompt = (
            "You are the visual processing unit for the SNAPPER robot dog. Analyze the input image and output a dense, flat list of semantic tags, objects, spatial layout, and environmental context.\n\n"
            "Strict constraints:\n"
            "1. No conversational filler, intro, or outro text.\n"
            "2. No markdown formatting, bullet points, or line breaks.\n"
            "3. Output a single, continuous paragraph of comma-separated descriptions.\n"
            "4. Prioritize: Exact object names, spatial relationships (e.g., 'chair left of table'), room type, lighting conditions, and human presence/actions.\n\n"
            "Example Format:\n"
            "[room type], [primary lighting], [object 1 with location], [object 2], [detected person with posture/action]"
        )
        if yolo_summary:
            prompt += f"\n\nHigh-confidence bounding-box objects detected: {yolo_summary}"
            if WORKER_VERBOSE:
                print(f"[VLM Worker] Incorporating YOLO hints: {yolo_summary}", file=sys.stderr)

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
                think=False,
                options={
                    "num_predict": DEFAULT_VLM_NUM_PREDICT,
                    "num_ctx": DEFAULT_VLM_NUM_CTX,
                    "temperature": DEFAULT_VLM_TEMPERATURE,
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
                    msg = chunk.get("message", {})
                    content = msg.get("content", "") or msg.get("thinking", "")
                else:
                    msg = getattr(chunk, "message", None)
                    if msg:
                        content = getattr(msg, "content", "") or getattr(msg, "thinking", "")
                if content:
                    description_chunks.append(content)

            query_duration = time.perf_counter() - query_start
            description = "".join(description_chunks).strip()
            
            with vlm_lock:
                previous_description = latest_scene_description
                latest_scene_description = description
                last_vlm_query_time = time.time()

            if WORKER_VERBOSE:
                print(f"[VLM Worker] VLM query finished in {query_duration:.2f}s.", file=sys.stderr)
            # Same rule as the body state: print the packet the model receives,
            # and only when it actually says something new. The scene-change
            # check upstream still fires on lighting shifts that describe
            # identically, so compare the text rather than trusting the trigger.
            if description and description != previous_description:
                log_context_packet(VISUAL_CONTEXT_PREFIX, description)
            
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



def preload_ollama_models(config: ConversationConfig) -> None:
    """Warm the chat and VLM models in Ollama before anything touches CUDA.

    Ollama decides CPU/GPU placement when a model loads and keeps it for as
    long as later requests use the same runner settings (num_ctx). Loading
    both models while the GPU is still empty — before Whisper, the dashboard's
    TensorRT engine, etc. — gives them the best possible split, and matching
    num_ctx here to what the real calls send means those calls never trigger
    a reload that would re-place the models under memory pressure.
    """
    try:
        client = ollama.Client()
    except Exception as exc:
        print(f"[Preload] Ollama unavailable, skipping model preload: {exc}", file=sys.stderr)
        return

    # A VLM runner left over from a previous run keeps whatever (possibly
    # CPU-heavy) placement it got under that run's memory pressure — unload it
    # so placement is re-decided now, while the GPU is free. With --no-vlm the
    # unload still happens (reclaiming the model's memory) but nothing reloads it.
    if config.vlm_model != config.ollama_model:
        try:
            client.generate(model=config.vlm_model, prompt="", keep_alive=0)
        except Exception:
            pass

    # Chat model first so it claims full GPU; the VLM takes what remains.
    models_to_load = [(config.ollama_model, config.ollama_num_ctx)]
    if not config.no_vlm:
        models_to_load.append((config.vlm_model, DEFAULT_VLM_NUM_CTX))
    for model_name, num_ctx in models_to_load:
        try:
            start = time.perf_counter()
            client.chat(
                model=model_name,
                messages=[{"role": "user", "content": "ping"}],
                keep_alive=-1,
                options={"num_ctx": num_ctx, "num_predict": 1},
            )
            print(
                f"[Preload] {model_name} loaded (num_ctx={num_ctx}) "
                f"in {time.perf_counter() - start:.1f}s",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"[Preload] Could not preload {model_name}: {exc}", file=sys.stderr)


def run_conversation(config: ConversationConfig) -> None:
    dashboard_process = None
    dashboard_log = None

    preload_ollama_models(config)

    # Before anything is named after the current time. The Jetson may have
    # booted at the epoch; the robot knows what time it is.
    sync_clock_from_robot()

    # Create session directory first so dashboard log can write to it
    log_dir = ensure_log_dir()
    session_id = wall_now().strftime("%Y%m%d-%H%M%S")
    session_dir = log_dir / f"session-{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # Check if the robot dog is on and available (skipped entirely in --simulate
    # mode, which always starts dashboard_server.py itself, in playback mode)
    dashboard_extra_args: list[str] = []
    if config.simulate:
        print(
            "[Chat Manager] --simulate: skipping robot dog check, will start "
            "dashboard_server.py --simulate to replay the most recent session.",
            file=sys.stderr,
        )
        start_dashboard = True
        dashboard_extra_args = ["--simulate"]
    else:
        robot_ip = os.getenv("UNITREE_ROBOT_IP", "192.168.4.30")
        start_dashboard = False
        if robot_ip:
            print(f"Checking if robot dog is available at {robot_ip}...", file=sys.stderr)
            cmd = ["ping", "-c", "1", "-t", "1" if sys.platform == "darwin" else "1", robot_ip]
            if sys.platform != "darwin":
                cmd = ["ping", "-c", "1", "-W", "1", robot_ip]
            # A dog still joining the wifi drops the first ping or two, and one
            # miss here means no dashboard at all for the whole session.
            for ping_attempt in range(3):
                try:
                    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    start_dashboard = (res.returncode == 0)
                except Exception:
                    start_dashboard = False
                if start_dashboard:
                    break
                if ping_attempt < 2:
                    print(f"No reply from {robot_ip}, retrying...", file=sys.stderr)
                    time.sleep(1.0)

    dashboard_process = None
    dashboard_log = None
    if start_dashboard:
        dashboard_process, dashboard_log = start_dashboard_with_retries(
            session_dir, session_id, dashboard_extra_args, config.simulate
        )
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
                # start_new_session so terminal Ctrl-C (SIGINT to the foreground
                # process group) never reaches it — its own KeyboardInterrupt
                # handler runs the robot POWER_OFF/stand-down sequence.
                bsm_process = subprocess.Popen(
                    [sys.executable, str(bsm_path)],
                    cwd=str(script_dir),
                    stdout=bsm_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
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
    
    # Set initial system prompt and speaker
    current_system_prompt = config.system_prompt
    current_speaker = config.speaker
    
    # If "Default" scene exists, use its prompt as the base
    default_scene = next((s for s in scenes if s.name == "Default"), None)
    if default_scene:
        print(f"Using 'Default' scene system prompt.", file=sys.stderr)
        current_system_prompt = default_scene.system_prompt
        if default_scene.speaker:
            current_speaker = default_scene.speaker
    
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
    if config.no_vlm:
        print("[VLM Worker] Disabled via --no-vlm; skipping scene captioning.", file=sys.stderr)
        vlm_worker = None
    else:
        vlm_worker = VLMBackgroundWorker(config, session_dir, log_file)
        vlm_worker.start()


    # Check if a matching input device (e.g. USB DJI Mic Mini / Wireless Mic Rx) is already present
    usb_input_index = None
    if config.input_device_keyword:
        usb_input_index = find_input_device(config.input_device_keyword)

    if usb_input_index is not None:
        dev_info = sd.query_devices(usb_input_index)
        print(
            f"[Audio] Found matching USB input device #{usb_input_index}: '{dev_info['name']}'. Skipping Bluetooth scan.",
            file=sys.stderr,
        )
        config.input_device_index = usb_input_index
    else:
        # Fall back to Bluetooth headset connection if no USB input match was found
        ensure_headset_connected()

    # Pick up headset state in case it was updated during connect
    LAST_HEADSET_MAC, LAST_HEADSET_NAME = load_headset_state()
    # Apply input source capture volume if configured
    if config.pulse_source_volume:
        set_pulse_source_volume(config.pulse_source_volume)
    # Log audio devices after device scan/connect
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
            send_dashboard_audio_interrupt()
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
        
        startup_intro = ""
        #Bluetooth connected, ready to chat."
        startup_prompt = "Give a short greeting to start the conversation, one sentence."
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
        startup_text = startup_text.strip()
        if startup_text:
            append_log_line(
                log_file,
                {
                    "type": "assistant",
                    "turn": 0,
                    "text": startup_text,
                    "audio_path": str(startup_audio),
                    "speaker": current_speaker,
                },
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
                (s for s in scenes if is_scene_triggered(s, user_text)),
                None,
            )
            
            # Check for reset command
            is_reset = is_reset_command(user_text)
            if is_reset or matched_scene:
                if matched_scene:
                    print(f"Scene trigger detected: '{matched_scene.name}'", file=sys.stderr)
                    current_system_prompt = matched_scene.system_prompt
                    if matched_scene.speaker:
                        current_speaker = matched_scene.speaker
                    elif default_scene and default_scene.speaker:
                        current_speaker = default_scene.speaker
                    else:
                        current_speaker = config.speaker
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
                        current_speaker = default_scene.speaker or config.speaker
                    else:
                        current_system_prompt = config.system_prompt
                        current_speaker = config.speaker
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
                    m["content"].startswith(VISUAL_CONTEXT_PREFIX)
                    or m["content"].startswith(BODY_STATE_PREFIX)
                ))
            ]

            # The user's turn goes in first, then the context, so perception
            # sits at the very end of the prompt. A small model weights the
            # tail of its context most heavily, and these packets describe the
            # moment being asked about - they should read as what the body
            # notices while answering, not as background stated earlier.
            messages.append({"role": "user", "content": user_text})

            if current_description:
                messages.append({"role": "system", "content": f"{VISUAL_CONTEXT_PREFIX} {current_description}"})
            if current_body_state:
                messages.append({"role": "system", "content": f"{BODY_STATE_PREFIX} {current_body_state}"})
            
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
                    "speaker": "USER",
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

            def on_chunk_playback(chunk_text: str):
                append_log_line(
                    log_file,
                    {
                        "type": "assistant_chunk",
                        "turn": turn,
                        "text": chunk_text,
                        "speaker": current_speaker,
                    },
                )

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
                kwargs={"on_chunk_playback": on_chunk_playback},
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
                append_log_line(log_file, {"type": "assistant_cancelled", "turn": turn, "speaker": current_speaker})
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
                    "speaker": current_speaker,
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
        if 'dashboard_process' in locals():
            stop_dashboard_server(dashboard_process, dashboard_log if 'dashboard_log' in locals() else None)

        # behavioral-state-machine.py is intentionally left running - it controls
        # the robot's physical stand-up/stand-down state, which shouldn't be
        # tied to chat-manager.py's own lifecycle.

        # Stop VLM background worker
        if 'vlm_worker' in locals() and vlm_worker is not None:
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

