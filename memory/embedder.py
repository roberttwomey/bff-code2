"""Small-footprint ONNX text embedder (all-MiniLM-L6-v2, 384-dim).

Runs CPU-only via onnxruntime, deliberately kept off the GPU so it never
competes with Ollama's gemma4/moondream for the Jetson's shared unified
memory (see the branch's earlier resource-contention analysis). Downloads
and caches the model + tokenizer from Hugging Face on first use via
huggingface_hub's standard on-disk cache.

The model is picked by CPU architecture: the ARM-optimized int8 quantized
export on arm64/aarch64 (Jetson, Apple Silicon), the full fp32 model
elsewhere. The AVX2/AVX512 quantized variants in the same repo are
x86-only and deliberately not used here.

Loaded lazily as a module-level singleton: an onnxruntime
InferenceSession supports concurrent Run() calls from multiple threads,
so one instance is shared by every MemoryStore rather than reloaded per
thread/session.
"""

from __future__ import annotations

import platform
import threading

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
_ARM_FILENAME = "onnx/model_qint8_arm64.onnx"
_DEFAULT_FILENAME = "onnx/model.onnx"
MAX_LENGTH = 256
EMBEDDING_DIM = 384

_lock = threading.Lock()
_session: ort.InferenceSession | None = None
_tokenizer: Tokenizer | None = None


def _model_filename() -> str:
    return _ARM_FILENAME if platform.machine() in ("arm64", "aarch64") else _DEFAULT_FILENAME


def _load() -> tuple[ort.InferenceSession, Tokenizer]:
    global _session, _tokenizer
    with _lock:
        if _session is None:
            model_path = hf_hub_download(MODEL_REPO, _model_filename())
            tokenizer_path = hf_hub_download(MODEL_REPO, "tokenizer.json")
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            tokenizer = Tokenizer.from_file(tokenizer_path)
            tokenizer.enable_truncation(max_length=MAX_LENGTH)
            _session, _tokenizer = session, tokenizer
        return _session, _tokenizer


def embed(text: str) -> bytes:
    """Return a 384-dim, L2-normalized embedding as packed float32 bytes --
    sqlite-vec's expected format for a FLOAT[384] column."""
    session, tokenizer = _load()

    encoding = tokenizer.encode(text)
    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

    feed = {"input_ids": input_ids, "attention_mask": attention_mask}
    input_names = {i.name for i in session.get_inputs()}
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = np.zeros_like(input_ids)

    token_embeddings = session.run(None, feed)[0]  # [1, seq_len, EMBEDDING_DIM]

    mask = attention_mask[..., None].astype(np.float32)
    summed = (token_embeddings * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    pooled = summed / counts

    normalized = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
    return normalized.astype(np.float32).tobytes()
