"""
Subs Pipeline: An Automated Video Transcription and Translation Pipeline.

This pipeline automates transcription using local Faster Whisper engines and translates
subtitles through leading API endpoints (Gemini, OpenAI, Anthropic, DeepL, Google, Ollama).
It handles batch processing, filesystem watching, error recovery, and robust video muxing.
"""

import os
import re
import sys
import gc
import json
import time
import uuid
import shutil
import argparse
import logging
import platform
import tempfile
import threading
import subprocess
import multiprocessing
import queue
import random
import contextlib
import gzip
import ssl
import urllib.request
import urllib.parse
import urllib.error
import difflib
from pathlib import Path
from typing import Any, Optional, Union, Tuple, List, Dict, Set, Generator, Final
from dataclasses import dataclass, field, asdict

# ── Dynamic OpenMP and SSL Environment Overrides ──────────────────
# Prevents crashes when duplicate OpenMP runtimes are loaded in the same process
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure frozen executables find SSL certificates for API requests
try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    _CERTIFI_AVAILABLE = True
except ImportError:
    certifi = None
    _CERTIFI_AVAILABLE = False

# ── Module Level Declarations ───────────────────────────
__all__ = [
    "Context",
    "TranscriptionManager",
    "is_valid_srt",
    "translate_srt_native",
    "find_source_srt",
    "process_file",
    "verify_translation_quality",
    "get_local_ollama_models",
    "pull_ollama_model",
    "main",
]

# ── Safe Terminal Encoding ──────────────────────────────
def setup_terminal_encoding() -> None:
    """Configure sys.stdout to safely support UTF-8 formatting."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception as e:
            logging.getLogger("subs_pipeline").debug("Terminal encoding config skipped: %s", e)

setup_terminal_encoding()

# ── Torch — imported once at module level ───────────────
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None
    _TORCH_AVAILABLE = False


# ════════════════════════════════════════════════════════════
#  ANSI COLORS  (with NO_COLOR support)
# ════════════════════════════════════════════════════════════
_NO_COLOR = os.environ.get("NO_COLOR", "").strip() not in ("", "0", "false", "False")
COLOR_ENABLED = not _NO_COLOR


class C:
    """ANSI color codes with NO_COLOR environment variable support."""

    CYAN: Final[str] = "\033[96m" if COLOR_ENABLED else ""
    GREEN: Final[str] = "\033[92m" if COLOR_ENABLED else ""
    YELLOW: Final[str] = "\033[93m" if COLOR_ENABLED else ""
    RED: Final[str] = "\033[91m" if COLOR_ENABLED else ""
    RESET: Final[str] = "\033[0m" if COLOR_ENABLED else ""
    BOLD: Final[str] = "\033[1m" if COLOR_ENABLED else ""
    DIM: Final[str] = "\033[2m" if COLOR_ENABLED else ""


_ANSI_RE = re.compile(r"\033(?:\[[0-9;]*[A-Za-z]|\][^\007]*\007)")


def strip_ansi(s: str) -> str:
    """Remove ANSI escape codes from a string."""
    return _ANSI_RE.sub("", s)


def style(text: str, *codes: str) -> str:
    """Wrap text in ANSI styling codes, ensuring reset safety."""
    if not COLOR_ENABLED or not codes:
        return text
    joined = "".join(codes)
    return f"{joined}{text}{C.RESET}"


def enable_windows_ansi() -> None:
    """Enable VT100 ANSI support on Windows platforms."""
    if platform.system() == "Windows":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception as e:
            logging.getLogger("subs_pipeline").debug("Console VT100 initialization bypassed: %s", e)
            try:
                os.system("")
            except Exception:
                pass


# ════════════════════════════════════════════════════════════
#  MODULE METADATA
# ════════════════════════════════════════════════════════════
__version__ = "1.5"
__author__ = "Subs Pipeline Team"

# ════════════════════════════════════════════════════════════
#  CONSTANTS & DEFAULTS
# ════════════════════════════════════════════════════════════
APP_DIR = (
    Path(sys.executable).parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)
CONFIG_PATH: Path = APP_DIR / "subs_pipeline_settings.json"

MEDIA_EXTS: Final[Tuple[str, ...]] = (
    ".mkv",
    ".mp4",
    ".webm",
    ".avi",
    ".mov",
    ".m4v",
    ".flv",
    ".ts",
    ".wmv",
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".opus",
)
AUDIO_EXTS: Final[Tuple[str, ...]] = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".opus")

# API Configuration Defaults
DEFAULT_GEMINI_MODEL: Final[str] = "gemini-3.5-flash"
GEMINI_URL_TEMPLATE: Final[str] = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)
API_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.+/\-=@#$!%^&*()]{10,120}$")

# VRAM Requirements (in GB)
VRAM_REQUIREMENTS: Final[Dict[str, float]] = {
    "tiny": 1.0,
    "base": 1.5,
    "small": 2.5,
    "medium": 5.0,
    "large-v3-turbo": 6.0,
    "large-v3": 10.0,
}

MODEL_MAP: Final[Dict[str, str]] = {
    "0": "tiny",
    "1": "base",
    "2": "small",
    "3": "medium",
    "4": "large-v3-turbo",
    "5": "large-v3",
}

# Dynamic Fallbacks for Translation Services
FALLBACK_MODELS: Final[Dict[str, str]] = {
    "gemini": "gemini-3.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "ollama": "qwen2.5:7b",
}

# Timings & Chunking Constants
GEMINI_CHUNK_SIZE: int = 80
GEMINI_TIMEOUT: int = 30
GEMINI_INTER_CHUNK_DELAY: int = 2
WATCHER_SETTLE_SECS: int = 5
MUXED_MIN_BYTES: int = 1000
FILE_SETTLE_MAX_RETRIES: int = 6
FILE_SETTLE_DELAY: float = 1.5
WHISPER_BEAM_SIZE: int = 5
WHISPER_TRANSCRIBE_TIMEOUT: int = 10800  # 3 hours max for transcription

# Named constants to avoid raw values inside execution pathways
WORKER_INIT_TIMEOUT: Final[float] = 45.0
WATCHER_IN_FLIGHT_EXPIRY: Final[float] = 1800.0
RATE_LIMITER_POLL_INTERVAL: Final[float] = 0.05
STDERR_TRUNCATION_LIMIT: Final[int] = 500
FFMPEG_PROCESS_TIMEOUT: Final[float] = 600.0  # Kept for backward compatibility references
FFMPEG_STREAM_TIMEOUT: Final[float] = 600.0   # 10 minutes max for fast stream operations
FFMPEG_TRANSCODE_TIMEOUT: Final[float] = 3600.0 # 1 hour max for rendering heavy hardsubs
MAX_CHUNK_CHAR_LIMIT: Final[int] = 5000  # Dialogue character length limit per translation request

# Sensitive Key substrings
SENSITIVE_KEY_SUBSTRINGS: Final[List[str]] = [
    "api_key", "token", "secret", "password", "api_url", "gateway", "key="
]

# Audio extraction settings
AUDIO_SAMPLE_RATE: int = 16000
AUDIO_CODEC: str = "pcm_s16le"

# Batch Quota Safety Brakes
CONSECUTIVE_429_LIMIT: int = 3
CONSECUTIVE_TOTAL_FAIL_LIMIT: int = 5

# Disk space safety margin (bytes)
MIN_FREE_DISK_BYTES: int = 500 * 1024 * 1024  # 500 MB

# Audit log controls
MAX_AUDIT_LOGS: int = 30
AUDIT_LOG_MAX_AGE_DAYS: int = 30

# Retry configuration
MAX_TRANSCRIPTION_RETRIES: int = 3
TRANSCRIPTION_RETRY_BASE_DELAY: float = 2.0

# Translation retry
MAX_TRANSLATION_CHUNK_RETRIES: int = 3
TRANSLATION_RETRY_BASE_DELAY: float = 2.0

# Validation Regex (Allows variable hour digit counts)
SRT_TIMESTAMP_PATTERN = re.compile(
    r"(\d+:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d+:\d{2}:\d{2}[,\.]\d{3})"
)

# Standard multilingual trivial words
MULTILINGUAL_TRIVIAL_WORDS: Final[Dict[str, Set[str]]] = {
    "en": {"oh", "ah", "okay", "ok", "yeah", "yes", "no", "uh", "hmm", "ha", "hey", "wow"},
    "ja": {"あの", "ええと", "はい", "いいえ", "うん", "うーん", "まぁ", "そう"},
    "es": {"oh", "ah", "bueno", "sí", "no", "oye", "vaya", "eh", "hola"},
    "fr": {"ah", "oh", "oui", "non", "bon", "bah", "hein", "ouais", "allô"},
    "zh": {"嗯", "啊", "哦", "好的", "那个", "就是", "哈"},
    "ko": {"아", "어", "예", "아니오", "음", "와", "네", "그래"}
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "schema_version": 2,
    "api_key": "",
    "tgt_lang": "English",
    "tgt_ext": "en",
    "src_lang": "",
    "min_blocks": 3,
    "model": "small",
    "device": "auto",
    "no_cleanup": False,
    "skip_migration": False,
    "explain_summary": True,
    "srt_max_avg_duration": 10.0,
    "srt_min_avg_duration": 0.1,
    "srt_dup_ratio": 0.6,
    "fallback_match_threshold": 0.95,
    "max_audit_logs": MAX_AUDIT_LOGS,
    "gemini_model": DEFAULT_GEMINI_MODEL,
    "translator": "gemini",
    "translation_model": "gemini-3.5-flash",
    "api_url": "",
    "whisper_beam_size": WHISPER_BEAM_SIZE,
}

# Unified Box-Drawing Constants
BOX_TL = "\u2554"
BOX_TR = "\u2557"
BOX_HL = "\u2550"
BOX_VL = "\u2551"
BOX_ML = "\u2560"
BOX_MR = "\u2563"
BOX_BL = "\u255a"
BOX_BR = "\u255d"

_config_save_lock = threading.Lock()


# ════════════════════════════════════════════════════════════
#  PIPELINE CONFIG & STATUS SCHEMAS
# ════════════════════════════════════════════════════════════
@dataclass
class PipelineConfig:
    schema_version: int = 2
    folder: str = ""
    api_key: str = ""
    tgt_lang: str = "English"
    tgt_ext: str = "en"
    src_lang: Optional[str] = None
    min_blocks: int = 3
    model: str = "small"
    device: str = "auto"
    skip_migration: bool = False
    explain_summary: bool = True
    srt_max_avg_duration: float = 10.0
    srt_min_avg_duration: float = 0.1
    srt_dup_ratio: float = 0.6
    fallback_match_threshold: float = 0.95
    max_audit_logs: int = 30
    gemini_model: str = "gemini-3.5-flash"
    translator: str = "gemini"
    translation_model: str = "gemini-3.5-flash"
    api_url: str = ""
    whisper_beam_size: int = 5
    
    # Executable flags derived during parsed argument mergers
    headless: bool = False
    watch: bool = False
    dry_run: bool = False
    no_audit: bool = False
    verbose_summary: bool = False
    recursive: bool = False
    quiet: bool = False
    verbose: bool = False
    test: bool = False
    
    # Skip flags parsed from options
    skip_transcribe: bool = False
    skip_translate: bool = False
    skip_embed: bool = False
    no_cleanup: bool = False
    
    # Final operational targets
    transcribe: bool = True
    translate: bool = True
    embed: bool = True
    hardsub: bool = False


@dataclass
class FileStatus:
    transcribed: bool = False
    translated: bool = False
    reused_srt: bool = False
    reused_all: bool = False
    mixed_language: bool = False
    partial_success: bool = False
    fallback_count: int = 0
    muxed: bool = False
    skipped: bool = False
    error: bool = False
    audio_failed: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ════════════════════════════════════════════════════════════
#  CUSTOM LOGGING AND EXCEPTIONS
# ════════════════════════════════════════════════════════════
class PipelineError(Exception):
    """Base exception class for all pipeline-related failures."""
    pass

class TranscriptionError(PipelineError):
    """Exception raised during audio transcription steps."""
    pass

class TranslationError(PipelineError):
    """Exception raised during API translation steps."""
    pass

class MuxError(PipelineError):
    """Exception raised during FFmpeg muxing or burning operations."""
    pass


class ColorLogFormatter(logging.Formatter):
    """Custom formatter to render log records with color safely depending on state."""
    def format(self, record: logging.LogRecord) -> str:
        res = super().format(record)
        if record.levelno >= logging.ERROR:
            return style(res, C.RED, C.BOLD)
        elif record.levelno >= logging.WARNING:
            return style(res, C.YELLOW)
        return res


# ════════════════════════════════════════════════════════════
#  LOGGER SETUP
# ════════════════════════════════════════════════════════════
def setup_logging(quiet: bool = False, verbose: bool = False) -> logging.Logger:
    """Configure Python logging framework with appropriate verbosity."""
    logger = logging.getLogger("subs_pipeline")
    if quiet:
        logger.setLevel(logging.WARNING)
    elif verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logger.level)
        fmt = ColorLogFormatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


logger: logging.Logger = setup_logging()


# ════════════════════════════════════════════════════════════
#  THREAD-SAFE CONTEXT SINGLETON STRUCTURE
# ════════════════════════════════════════════════════════════
class Context:
    """Thread-safe global application context registry containing operational flags."""

    quiet: bool = False
    ffmpeg_cmd: Optional[str] = None
    ffprobe_cmd: Optional[str] = None
    _translation_disabled: bool = False
    _consecutive_429s: int = 0
    _consecutive_total_failures: int = 0
    _state_lock: threading.Lock = threading.Lock()
    config_warning: str = ""
    active_temp_files: Set[Path] = set()
    temp_lock: threading.Lock = threading.Lock()
    failed_cleanups: List[str] = []
    migration_status: str = "none"
    provenance: Dict[str, str] = {}

    @classmethod
    def is_translation_disabled(cls) -> bool:
        """Get thread-safe translation disabled status."""
        with cls._state_lock:
            return cls._translation_disabled

    @classmethod
    def set_translation_disabled(cls, val: bool) -> None:
        """Set thread-safe translation disabled status."""
        with cls._state_lock:
            cls._translation_disabled = val

    @classmethod
    def add_failed_cleanup(cls, msg: str) -> None:
        """Add clean up error message safely."""
        with cls._state_lock:
            cls.failed_cleanups.append(msg)

    @classmethod
    def get_consecutive_429s(cls) -> int:
        """Get the thread-safe count of consecutive 429 errors."""
        with cls._state_lock:
            return cls._consecutive_429s

    @classmethod
    def increment_consecutive_429s(cls) -> None:
        """Increment the consecutive 429 error counter in a thread-safe manner."""
        with cls._state_lock:
            cls._consecutive_429s += 1

    @classmethod
    def reset_consecutive_429s(cls) -> None:
        """Reset the consecutive 429 error counter to zero in a thread-safe manner."""
        with cls._state_lock:
            cls._consecutive_429s = 0

    @classmethod
    def get_consecutive_total_failures(cls) -> int:
        """Get the thread-safe count of total consecutive API failures."""
        with cls._state_lock:
            return cls._consecutive_total_failures

    @classmethod
    def increment_consecutive_total_failures(cls) -> None:
        """Increment the consecutive total failures counter in a thread-safe manner."""
        with cls._state_lock:
            cls._consecutive_total_failures += 1

    @classmethod
    def reset_consecutive_total_failures(cls) -> None:
        """Reset the consecutive total failures counter to zero in a thread-safe manner."""
        with cls._state_lock:
            cls._consecutive_total_failures = 0

    @classmethod
    def reset_all_counters(cls) -> None:
        """Reset all active execution and error counters to their initial values."""
        with cls._state_lock:
            cls._consecutive_429s = 0
            cls._consecutive_total_failures = 0
            cls._translation_disabled = False

    @classmethod
    def clear_mutable_states(cls) -> None:
        """Fully restore active mutable storage environments safely."""
        with cls.temp_lock:
            cls.active_temp_files.clear()
        with cls._state_lock:
            cls.failed_cleanups.clear()
            cls.provenance.clear()

    @classmethod
    def reset(cls) -> None:
        """Complete structural context reset to initial default state."""
        with cls._state_lock:
            cls.quiet = False
            cls.ffmpeg_cmd = None
            cls.ffprobe_cmd = None
            cls._translation_disabled = False
            cls._consecutive_429s = 0
            cls._consecutive_total_failures = 0
            cls.config_warning = ""
            cls.migration_status = "none"
            cls.failed_cleanups.clear()
            cls.provenance.clear()
        with cls.temp_lock:
            cls.active_temp_files.clear()


def qprint(*args, **kwargs) -> None:
    """Stdout writer that respects the global quiet setting and logs messages."""
    msg = " ".join(str(a) for a in args)
    if not Context.quiet:
        print(*args, **kwargs)
    logger.info(strip_ansi(msg))


# ── Dependency Check & FFmpeg Finder ────────────────────
REQUIRED_PACKAGES: List[str] = ["faster_whisper"]


def check_dependencies(headless: bool = False) -> None:
    """Verify all required Python packages are installed."""
    missing: List[str] = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"{C.RED}[!] Missing required packages: {', '.join(missing)}{C.RESET}")
        print(f"    Please run: pip install {' '.join(missing)}")
        if not headless:
            try:
                input("\nPress Enter to exit...")
            except Exception:
                pass
        sys.exit(1)


def setup_ffmpeg() -> bool:
    """Locate ffmpeg and ffprobe executables in system PATH or program directory."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    
    # Fallback to local program folder directory to assist portable distributions
    if not ffmpeg or not ffprobe:
        ext = ".exe" if platform.system() == "Windows" else ""
        local_ffmpeg = APP_DIR / f"ffmpeg{ext}"
        local_ffprobe = APP_DIR / f"ffprobe{ext}"
        if local_ffmpeg.exists() and local_ffprobe.exists():
            ffmpeg = str(local_ffmpeg)
            ffprobe = str(local_ffprobe)

    if ffmpeg and ffprobe:
        Context.ffmpeg_cmd = ffmpeg
        Context.ffprobe_cmd = ffprobe
        logger.debug("Located FFmpeg: %s, FFprobe: %s", ffmpeg, ffprobe)
        return True
    return False


def get_available_vram_gb() -> float:
    """Query available GPU VRAM in gigabytes."""
    if _TORCH_AVAILABLE and _torch.cuda.is_available():
        try:
            t_vram = _torch.cuda.get_device_properties(0).total_memory
            a_vram = t_vram - _torch.cuda.memory_allocated(0)
            return a_vram / (1024**3)
        except Exception as e:
            logger.debug("Failed to query VRAM: %s", e)
            return 0.0
    return 0.0


def resolve_device_and_compute(mode: str = "auto") -> Tuple[str, str]:
    mode = (mode or "auto").lower().strip()
    if mode == "cpu":
        return "cpu", "int8"
    if mode == "cuda":
        return ("cuda", "float16") if _TORCH_AVAILABLE and _torch.cuda.is_available() else ("cpu", "int8")
    if _TORCH_AVAILABLE and _torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "int8"


def get_model_recommendation(vram_gb: float, is_cpu: bool) -> str:
    """Return recommended model based on computed VRAM or CPU state."""
    if is_cpu:
        return "1"  # "base"
    if vram_gb >= 10.0:
        return "5"  # "large-v3"
    elif vram_gb >= 6.0:
        return "4"  # "large-v3-turbo"
    elif vram_gb >= 5.0:
        return "3"  # "medium"
    elif vram_gb >= 2.5:
        return "2"  # "small"
    elif vram_gb >= 1.5:
        return "1"  # "base"
    return "0"  # "tiny"


def recommend_whisper_model() -> str:
    """Recommend a Whisper model based on available hardware."""
    device, _ = resolve_device_and_compute()
    is_cpu = (device == "cpu")
    vram = get_available_vram_gb()
    return get_model_recommendation(vram, is_cpu)


def download_whisper_model_if_needed(model_name: str) -> bool:
    """Locate or download the Whisper model to cache directories."""
    try:
        from faster_whisper.utils import download_model
        download_model(model_name)
        return True
    except Exception as e:
        logger.debug("Model retrieval failed for %s: %s", model_name, e)
        return False


# ════════════════════════════════════════════════════════════
#  PERSISTENT BOUNDED TRANSCRIBER MOTOR
# ════════════════════════════════════════════════════════════
def _transcribe_worker_loop(
    req_queue: multiprocessing.Queue,
    res_queue: multiprocessing.Queue,
    model_name: str,
    device: str,
    compute_type: str,
) -> None:
    """Top-level worker function running inside the spawned subprocess."""
    try:
        # Avoid thread contention in deep CPU loops
        os.environ["OMP_NUM_THREADS"] = "4"
        from faster_whisper import WhisperModel
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        res_queue.put(("init_ok", None))
    except BaseException as e:
        import traceback
        tb = traceback.format_exc()
        try:
            res_queue.put(("init_error", f"{e}\n{tb}"))
        except Exception:
            pass
        return

    while True:
        try:
            task = req_queue.get()
            if task is None:
                break
                
            task_type = "transcribe"
            if len(task) == 4:
                audio_path, lang_hint, beam_size, task_type = task
            else:
                audio_path, lang_hint, beam_size = task
                
            segments, info = model.transcribe(
                audio_path,
                beam_size=beam_size,
                language=lang_hint,
                task=task_type,
            )
            res_queue.put(("info", (info.language, info.language_probability)))
            
            try:
                for seg in segments:
                    res_queue.put(("segment", (seg.start, seg.end, seg.text)))
                res_queue.put(("done", None))
            except BaseException as loop_err:
                try:
                    res_queue.put(("error", f"Fault encountered during active audio segmentation: {loop_err}"))
                except Exception:
                    pass
        except BaseException as e:
            try:
                res_queue.put(("error", str(e)))
            except Exception:
                pass


class TranscriptionManager:
    """Bounded model manager processing audio across file loops."""

    _process: Optional[multiprocessing.Process] = None
    _req_queue: Optional[multiprocessing.Queue] = None
    _res_queue: Optional[multiprocessing.Queue] = None
    _lock: threading.RLock = threading.RLock()
    _transcribe_mutex: threading.Lock = threading.Lock()
    _is_running: bool = False
    _current_model: Optional[Tuple[str, str, str]] = None

    @classmethod
    def _start_process(cls, model_name: str, device: str, compute_type: str) -> None:
        """Launch the worker process and configure queues."""
        ctx = multiprocessing.get_context("spawn")
        cls._req_queue = ctx.Queue()
        cls._res_queue = ctx.Queue()
        cls._process = ctx.Process(
            target=_transcribe_worker_loop,
            args=(cls._req_queue, cls._res_queue, model_name, device, compute_type),
            daemon=False
        )
        cls._process.start()

    @classmethod
    def _drain_queue(cls, q: Optional[multiprocessing.Queue]) -> None:
        """Exhaust all remaining elements from a multiprocessing queue."""
        if q is None:
            return
        while True:
            try:
                q.get_nowait()
            except Exception:
                break

    @classmethod
    def transcribe(
        cls,
        audio_path: str,
        model_name: str,
        device: str,
        compute_type: str,
        lang_hint: Optional[str],
        beam_size: int,
        timeout: float,
        task_type: str = "transcribe",
    ) -> Generator[Tuple[str, Any], None, None]:
        """Transcribe audio using the persistent process, serialization locks handle concurrency."""
        with cls._transcribe_mutex:
            with cls._lock:
                if cls._is_running:
                    raise TranscriptionError("Another transcription is currently in progress.")
                cls._is_running = True

                target_model = (model_name, device, compute_type)
                if (
                    cls._process is None
                    or not cls._process.is_alive()
                    or cls._current_model != target_model
                ):
                    if cls._process and cls._process.is_alive():
                        cls.terminate_with_lock()
                    
                    logger.debug("Spawning child transcription worker using model: %s", model_name)
                    cls._start_process(model_name, device, compute_type)
                    cls._current_model = target_model

                    init_timeout = False
                    msg_type, payload = None, None
                    try:
                        msg_type, payload = cls._res_queue.get(timeout=WORKER_INIT_TIMEOUT)
                    except queue.Empty:
                        init_timeout = True

                    if init_timeout:
                        cls.terminate_with_lock()
                        cls._is_running = False
                        raise TranscriptionError("Failed to communicate with transcription worker (initialization timeout).")
                    elif msg_type == "init_error":
                        cls.terminate_with_lock()
                        cls._is_running = False
                        raise TranscriptionError(f"Transcription worker initialization failed: {payload}")
                    elif msg_type != "init_ok":
                        cls.terminate_with_lock()
                        cls._is_running = False
                        raise TranscriptionError(f"Unexpected response from transcription worker initialization: {msg_type}")

                cls._drain_queue(cls._res_queue)
                cls._req_queue.put((audio_path, lang_hint, beam_size, task_type))

            deadline = time.monotonic() + timeout
            try:
                while True:
                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        cls.terminate()
                        raise TranscriptionError(f"Transcription timed out after {timeout} seconds")

                    try:
                        msg_type, data = cls._res_queue.get(timeout=min(rem, 1.0))
                        if msg_type == "info":
                            yield ("info", data)
                        elif msg_type == "segment":
                            yield ("segment", data)
                        elif msg_type == "done":
                            break
                        elif msg_type == "error":
                            raise TranscriptionError(data)
                    except queue.Empty:
                        with cls._lock:
                            if cls._process is not None and not cls._process.is_alive():
                                raise TranscriptionError("Transcription worker process terminated unexpectedly")
                        continue
            finally:
                with cls._lock:
                    cls._is_running = False
                perform_vram_gc()

    @classmethod
    def terminate(cls) -> None:
        """Safely terminate child process and release active memory under lock."""
        with cls._lock:
            cls.terminate_with_lock()

    @classmethod
    def terminate_with_lock(cls) -> None:
        """Internal termination handler executing under active lock contexts."""
        cls._is_running = False
        if cls._process:
            logger.debug("Terminating transcription child process")
            try:
                if cls._req_queue:
                    try:
                        cls._req_queue.put_nowait(None)
                    except Exception:
                        pass
                cls._process.join(timeout=2.0)
                if cls._process.is_alive():
                    cls._process.terminate()
                    cls._process.join(timeout=2.0)
                if cls._process.is_alive():
                    cls._process.kill()
                    cls._process.join(timeout=1.0)
            except Exception as e:
                logger.debug("Non-fatal termination error: %s", e)
            
            # Close queues securely without blocking on lingering background pipe threads
            for q in (cls._req_queue, cls._res_queue):
                if q:
                    try:
                        q.cancel_join_thread()
                        q.close()
                    except Exception:
                        pass
            
            cls._process = None
            cls._req_queue = None
            cls._res_queue = None
            cls._current_model = None


# ════════════════════════════════════════════════════════════
#  TEMP FILE MANAGER & SCOPED TEMP GUARDS
# ════════════════════════════════════════════════════════════
def register_temp_file(path: Union[str, Path]) -> None:
    """Register a temporary file for later cleanup."""
    with Context.temp_lock:
        Context.active_temp_files.add(Path(path).resolve())


def unregister_temp_file(path: Union[str, Path]) -> None:
    """Unregister a temporary file from cleanup tracking."""
    with Context.temp_lock:
        p = Path(path).resolve()
        Context.active_temp_files.discard(p)


def safe_remove(path: Union[str, Path]) -> None:
    """Safely remove a file, unregistering from temp tracking."""
    p = Path(path)
    try:
        if p.exists():
            p.unlink(missing_ok=True)
    except OSError as e:
        Context.add_failed_cleanup(f"{path} ({e.strerror or str(e)})")
    finally:
        unregister_temp_file(p)


@contextlib.contextmanager
def temp_file_guard(path: Union[str, Path]) -> Generator[Path, None, None]:
    """Context manager to guarantee the registration and removal of temp files."""
    p = Path(path).resolve()
    register_temp_file(p)
    try:
        yield p
    finally:
        safe_remove(p)


def cleanup_all_temp_files() -> None:
    """Remove all registered temporary files with race-condition safety."""
    with Context.temp_lock:
        for p in list(Context.active_temp_files):
            try:
                p.unlink(missing_ok=True)
            except OSError as e:
                logger.debug("Failed to clean up temp file %s: %s", p, e)
            Context.active_temp_files.discard(p)


# ════════════════════════════════════════════════════════════
#  GARBAGE COLLECTION
# ════════════════════════════════════════════════════════════
def startup_garbage_collection(
    folder_path: Union[str, Path], skip_cleanup: bool = False
) -> None:
    """Clean up stale temporary files from previous runs."""
    if skip_cleanup:
        return
    targets = [Path(folder_path), APP_DIR]
    cleaned_count = 0
    for target_dir in targets:
        if not target_dir.is_dir():
            continue
        try:
            for item in target_dir.iterdir():
                if item.is_file():
                    is_stale_hardsub = (
                        "temp_hardsub_" in item.name
                        and item.suffix.lower() == ".srt"
                    )
                    is_stale_audio = (
                        item.name.startswith("temp_") and item.name.endswith("_audio.wav")
                    )
                    if is_stale_hardsub or is_stale_audio:
                        try:
                            item.unlink()
                            cleaned_count += 1
                        except OSError as e:
                            logger.debug("Failed to delete %s: %s", item, e)
        except PermissionError as e:
            logger.debug("Permission error during garbage collection of folder %s: %s", target_dir, e)
    if cleaned_count > 0:
        qprint(
            f"  {style('[~]', C.DIM)} Swept workspaces: Purged {cleaned_count} stale temp "
            f"file(s)."
        )


# ════════════════════════════════════════════════════════════
#  DISK SPACE CHECK
# ════════════════════════════════════════════════════════════
def check_disk_space(path: Union[str, Path], required_bytes: int = MIN_FREE_DISK_BYTES) -> bool:
    """Verify disk space margin is sufficient."""
    try:
        p = Path(path)
        target = p if p.is_dir() else p.parent
        free = shutil.disk_usage(target).free
        if free < required_bytes:
            qprint(
                f"{style('[!]', C.YELLOW)} Low disk space: {free // (1024**2)} MB available, "
                f"{required_bytes // (1024**2)} MB recommended."
            )
            return False
        return True
    except Exception as e:
        logger.warning("Disk space analysis could not be completed: %s", e)
        return True


# ════════════════════════════════════════════════════════════
#  SAFE EXIT
# ════════════════════════════════════════════════════════════
def exit_app(code: int = 0) -> None:
    """Perform clean shutdown with temp file cleanup and worker termination."""
    cleanup_all_temp_files()
    TranscriptionManager.terminate()
    if Context.failed_cleanups:
        qprint(
            f"\n{style('[~]', C.DIM)} System cleanup complete. Some active lock files were "
            f"bypassed:"
        )
        for item in set(Context.failed_cleanups):
            qprint(f"      · {item}")
    if "--headless" not in sys.argv:
        try:
            input(f"\n{C.DIM}Press Enter to exit...{C.RESET}")
        except Exception:
            pass
    sys.exit(code)


# ════════════════════════════════════════════════════════════
#  CONFIG & DIAGNOSTICS
# ════════════════════════════════════════════════════════════
def verify_config_status() -> Union[bool, str]:
    """Verify configuration directory and file are accessible."""
    parent_dir = CONFIG_PATH.parent
    if not parent_dir.exists():
        return "Configuration directory does not exist."
    if not os.access(parent_dir, os.W_OK):
        return "Configuration directory is not writeable."
    if CONFIG_PATH.exists():
        if not os.access(CONFIG_PATH, os.R_OK):
            return "Configuration file is not readable."
        if not os.access(CONFIG_PATH, os.W_OK):
            return "Configuration file is not writeable."
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                json.load(f)
            return True
        except json.JSONDecodeError as e:
            return f"Configuration file has invalid JSON formatting: {e}"
        except Exception as e:
            return f"Configuration file access failure: {e}"
    return True


def validate_schema(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Sanity-check loaded configuration values, enforce ranges, maintain schema versioning."""
    if "schema_version" not in cfg_dict:
        cfg_dict["schema_version"] = DEFAULT_CONFIG["schema_version"]
    elif cfg_dict["schema_version"] < DEFAULT_CONFIG["schema_version"]:
        qprint(
            f"  {style('[~]', C.YELLOW)} Legacy config schema version "
            f"{cfg_dict.get('schema_version')} detected. Updating to version "
            f"{DEFAULT_CONFIG['schema_version']}."
        )
        cfg_dict["schema_version"] = DEFAULT_CONFIG["schema_version"]

    range_rules: Dict[str, Tuple[Union[int, float], Union[int, float], Any]] = {
        "min_blocks": (1, 1000, 3),
        "srt_max_avg_duration": (0.5, 300.0, 10.0),
        "srt_min_avg_duration": (0.01, 10.0, 0.1),
        "srt_dup_ratio": (0.01, 1.0, 0.6),
        "fallback_match_threshold": (0.1, 1.0, 0.95),
        "max_audit_logs": (1, 500, 30),
    }

    for key, default_val in DEFAULT_CONFIG.items():
        if key in cfg_dict:
            if isinstance(default_val, bool):
                if not isinstance(cfg_dict[key], bool):
                    val_str = str(cfg_dict[key]).lower()
                    if val_str in ("true", "1", "yes", "on"):
                        cfg_dict[key] = True
                    elif val_str in ("false", "0", "no", "off"):
                        cfg_dict[key] = False
                    else:
                        cfg_dict[key] = bool(cfg_dict[key])
            elif default_val is not None and type(cfg_dict[key]) is not type(default_val):
                try:
                    if isinstance(default_val, float):
                        cfg_dict[key] = float(cfg_dict[key])
                    elif isinstance(default_val, int):
                        cfg_dict[key] = int(cfg_dict[key])
                except Exception as e:
                    logger.debug("Type conversion fail: key %s, error %s", key, e)
                    cfg_dict[key] = default_val

            if key in range_rules:
                min_val, max_val, fallback = range_rules[key]
                try:
                    if cfg_dict[key] < min_val or cfg_dict[key] > max_val:
                        qprint(
                            f"  {style('[!]', C.YELLOW)} Config key '{key}' out of range "
                            f"[{min_val} - {max_val}]. Resetting to default '{fallback}'."
                        )
                        cfg_dict[key] = fallback
                except TypeError:
                    qprint(
                        f"  {style('[!]', C.YELLOW)} Config key '{key}' has invalid type compatibility. "
                        f"Resetting to default '{fallback}'."
                    )
                    cfg_dict[key] = fallback
        else:
            cfg_dict[key] = default_val
    return cfg_dict


def load_config() -> Dict[str, Any]:
    """Load configuration with reversible base64 obfuscation used to prevent plaintext exposure in settings."""
    logger.debug("Loading configuration parameters from: %s", CONFIG_PATH)
    if CONFIG_PATH.exists():
        Context.migration_status = "loaded"

    cfg_dict = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                
                # Map legacy skip_cleanup key down to no_cleanup parameters
                if "skip_cleanup" in loaded:
                    loaded["no_cleanup"] = loaded.pop("skip_cleanup")
                
                if loaded.get("api_key") and loaded["api_key"].startswith("obf:"):
                    import base64
                    try:
                        loaded["api_key"] = base64.b64decode(loaded["api_key"][4:].encode("utf-8")).decode("utf-8")
                    except Exception:
                        pass
                
                cfg_dict.update(loaded)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Failed to parse config. Restoring defaults: %s", e)
        except Exception as e:
            logger.debug("Config load exception: %s", e)
    return validate_schema(cfg_dict)


def save_config(conf: Dict[str, Any]) -> None:
    """Save configuration atomically with basic base64 obfuscation to prevent direct exposure of credential keys."""
    with _config_save_lock:
        parent_dir = CONFIG_PATH.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=parent_dir, suffix=".tmp", encoding="utf-8", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                
                conf_copy = dict(conf)
                if conf_copy.get("api_key"):
                    import base64
                    try:
                        conf_copy["api_key"] = "obf:" + base64.b64encode(conf_copy["api_key"].encode("utf-8")).decode("utf-8")
                    except Exception:
                        pass
                
                json.dump(conf_copy, tmp_file, indent=4, ensure_ascii=False)
                
            with open(tmp_path, "r", encoding="utf-8") as f:
                json.load(f)
                
            os.replace(str(tmp_path), str(CONFIG_PATH))
            tmp_path = None
            
            if platform.system() != "Windows":
                try:
                    os.chmod(CONFIG_PATH, 0o600)
                except Exception as e:
                    logger.debug("Failed to set file permissions on config path: %s", e)
        except Exception as e:
            qprint(f"\n  {style('[!]', C.YELLOW)} Config save failed: {e}")
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass


# Global configuration reference (deferred initialization)
global_cfg: Dict[str, Any] = {}


# ════════════════════════════════════════════════════════════
#  UTILITIES & HEURISTIC FALLBACK DETECTION
# ════════════════════════════════════════════════════════════
def is_safe_relative(path: Union[str, Path], base: Union[str, Path]) -> bool:
    """Safely verify path remains inside folder constraints without traversal."""
    try:
        r_path = Path(path).resolve()
        r_base = Path(base).resolve()
        r_path.relative_to(r_base)
        return True
    except ValueError:
        return False


def escape_ffmpeg_filter_path(path: Union[str, Path]) -> str:
    """Escape filenames for use inside FFmpeg filter syntax specifications."""
    p_str = str(Path(path).resolve()).replace("\\", "/")
    p_str = p_str.replace(":", "\\:")
    p_str = p_str.replace("'", "'\\\\''")
    p_str = p_str.replace("[", "\\[").replace("]", "\\]")
    p_str = p_str.replace("%", "\\%")
    p_str = p_str.replace(";", "\\;")
    p_str = p_str.replace(",", "\\,")
    p_str = p_str.replace("\n", "")
    return f"'{p_str}'"


def natural_keys(text: Union[str, Path]) -> List[Tuple[int, Union[int, str]]]:
    """Split text into natural sort key components (numbers sorted as numbers, text lowercase)
    safely avoiding cross-type comparison errors.
    """
    return [
        (0, int(c)) if c.isdigit() else (1, c.lower())
        for c in re.split(r"(\d+)", str(text))
        if c
    ]


def fmt_time(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    seconds = max(0.0, float(seconds))
    if seconds == 0:
        return "0s"
    if seconds < 1:
        return "<1s"
    h, rem = divmod(int(seconds), 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


def fmt_srt_ts(t: float) -> str:
    """Format a timestamp in seconds to SRT format (HH:MM:SS,mmm)."""
    t = max(0.0, float(t))
    ms = round(t * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def get_duration(media_path: Union[str, Path]) -> float:
    """Get the duration of a media file using ffprobe."""
    if not Context.ffprobe_cmd:
        return 0.0
    try:
        cmd = [
            Context.ffprobe_cmd,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ]
        result = subprocess.check_output(
            cmd,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.DEVNULL,
            timeout=30.0
        ).strip()
        val = float(result)
        return val if val > 0 else 0.0
    except Exception:
        return 0.0


def perform_vram_gc() -> None:
    """Free GPU VRAM by clearing CUDA cache and running garbage collection."""
    if _TORCH_AVAILABLE:
        try:
            if _torch.cuda.is_available():
                gc.collect()
                _torch.cuda.empty_cache()
        except Exception as e:
            logger.debug("CUDA empty cache failed: %s", e)


def normalize_dialogue(text: Optional[str], lang: str = "en") -> str:
    """Normalize subtitle dialogue for comparison by removing markup and trivial words."""
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"\[[^\]]*\]", "", t)
    t = re.sub(r"\([^\)]*\)", "", t)
    t = re.sub(r"<[^>]*>", "", t)
    t = re.sub(r"[^\w\s]", "", t)
    
    lang_clean = (lang or "en").lower()[:2]
    trivial_set = MULTILINGUAL_TRIVIAL_WORDS.get(lang_clean, MULTILINGUAL_TRIVIAL_WORDS["en"])
    
    words = [w for w in t.split() if w not in trivial_set]
    return " ".join(words)


def parse_srt_dialogue(path: Union[str, Path]) -> List[str]:
    """Parse dialogue lines from an SRT file."""
    try:
        content = Path(path).read_text(encoding="utf-8", errors="ignore")
        blocks = [b.strip() for b in re.split(r"\n\n+", content.strip()) if b.strip()]
        dialogues: List[str] = []
        for b in blocks:
            lines = b.splitlines()
            if len(lines) >= 3:
                dialogues.append("\n".join(lines[2:]).strip())
            else:
                dialogues.append("")
        return dialogues
    except Exception:
        return []


def parse_srt_to_dict(path: Union[str, Path]) -> Dict[int, str]:
    """Parse an SRT file to a mapping dictionary of block ID to raw dialogue text."""
    sub_map: Dict[int, str] = {}
    try:
        content = Path(path).read_text(encoding="utf-8", errors="ignore")
        blocks = [b.strip() for b in re.split(r"\n\n+", content.strip()) if b.strip()]
        for b in blocks:
            lines = b.splitlines()
            if len(lines) >= 3:
                raw_id = lines[0].strip()
                if raw_id.isdigit():
                    b_id = int(raw_id)
                    sub_map[b_id] = "\n".join(lines[2:]).strip()
    except Exception as e:
        logger.debug("Failed to parse SRT to dictionary layout: %s", e)
    return sub_map


def detect_fallbacks(
    src_path: Union[str, Path],
    tgt_path: Union[str, Path],
    fallback_match_threshold: float,
    lang_code: str = "en"
) -> Tuple[int, str]:
    """Detect translation fallback blocks where target matches source via ID matching."""
    src_map = parse_srt_to_dict(src_path)
    tgt_map = parse_srt_to_dict(tgt_path)
    if not src_map or not tgt_map:
        return 0, "No blocks parsed"

    match_count = 0

    for b_id, src_txt in src_map.items():
        if b_id not in tgt_map:
            continue
        tgt_txt = tgt_map[b_id]

        s_clean = src_txt.replace("\u266a", "").strip()
        t_clean = tgt_txt.replace("\u266a", "").strip()

        s_norm = normalize_dialogue(s_clean, lang_code)
        t_norm = normalize_dialogue(t_clean, lang_code)

        if len(s_norm) < 4 or len(t_norm) < 4:
            continue

        if s_norm == t_norm:
            match_count += 1
        else:
            ratio = difflib.SequenceMatcher(None, s_norm, t_norm).ratio()
            if ratio >= fallback_match_threshold:
                match_count += 1

    return match_count, f"{match_count} blocks matching source"


def find_source_srt(media_path: Union[str, Path], tgt_ext: str, src_lang: Optional[str] = None) -> Tuple[Path, str]:
    """Locate and return an existing source subtitle file associated with a media file."""
    media_path = Path(media_path)
    parent = media_path.parent
    stem = media_path.stem
    tgt_ext_clean = tgt_ext.strip().lower()
    
    candidates: List[Tuple[Path, str]] = []
    try:
        for item in parent.iterdir():
            if item.is_file() and item.suffix.lower() == ".srt":
                name = item.name
                if name == f"{stem}.srt":
                    candidates.append((item, "und"))
                elif name.startswith(f"{stem}."):
                    suffix_len = len(".srt")
                    mid_part = name[len(stem) + 1 : -suffix_len]
                    if mid_part and "." not in mid_part:
                        if mid_part.lower() != tgt_ext_clean:
                            candidates.append((item, mid_part.lower()))
    except Exception as e:
        logger.debug("Error searching for source SRT: %s", e)

    if src_lang:
        src_lang_clean = src_lang.strip().lower()
        for path, lang in candidates:
            if lang == src_lang_clean:
                return path, lang
        # If explicitly requested, an unrelated candidate language must not fall through
        return parent / f"{stem}.{src_lang_clean}.srt", src_lang_clean
    
    for path, lang in candidates:
        if lang != "und":
            return path, lang
            
    for path, lang in candidates:
        if lang == "und":
            return path, lang

    default_lang = src_lang if src_lang else "und"
    if default_lang != "und":
        default_path = parent / f"{stem}.{default_lang}.srt"
    else:
        default_path = parent / f"{stem}.subs-pipeline.srt"
    return default_path, default_lang


def _prepare_translation_prompt(chunk: List[str], tgt_lang: str) -> str:
    """Prepare a structured translation prompt for translation APIs."""
    prompt_lines = [
        f"You are a professional film/media translator. Translate the following subtitle blocks into accurate, natural, and context-aware {tgt_lang}.",
        "Preserve the tone, colloquialisms, and original meaning as closely as possible.",
        "Strictly follow these formatting constraints:",
        "1. Respond ONLY with the translations, using exactly the block format template shown below.",
        "2. Do not write any introduction, commentary, explanations, or Markdown code block wrappers. Only output the formatted translation blocks.",
        "3. Each block output must strictly follow this pattern: 'Block #[ID]: [translated_text]'",
        "4. If a block contains multiple lines, keep them on separate lines under the same Block ID.",
        "5. Do not alter, merge, or omit any Block IDs.",
        "\nInput blocks to translate:\n"
    ]
    
    for block in chunk:
        lines = block.splitlines()
        if len(lines) >= 3:
            try:
                b_id = int(lines[0])
            except ValueError:
                b_id = 0
            dialogue = "\n".join(lines[2:])
            prompt_lines.append(f"Block #{b_id}:\n{dialogue}\n")
            
    return "\n".join(prompt_lines)


# ════════════════════════════════════════════════════════════
#  SRT HEALTH CHECK & QUALITY ASSURANCE
# ════════════════════════════════════════════════════════════
def is_valid_srt(
    srt_path: Union[str, Path],
    media_duration: float = 0.0,
    min_blocks: int = 3,
    args_ref: Optional[PipelineConfig] = None,
) -> Tuple[bool, str]:
    """Validate an SRT file for structural integrity and quality."""
    if args_ref is None:
        args_ref = PipelineConfig()

    max_duration: float = args_ref.srt_max_avg_duration
    min_duration: float = args_ref.srt_min_avg_duration
    dup_threshold: float = args_ref.srt_dup_ratio

    try:
        p = Path(srt_path)
        if not p.exists() or p.stat().st_size == 0:
            return False, "File is missing or empty"
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return False, f"Cannot read file: {e}"

    blocks = [b.strip() for b in re.split(r"\n\n+", text.strip()) if b.strip()]
    if len(blocks) < min_blocks:
        return (
            False,
            f"Only {len(blocks)} block(s) -- minimum health threshold is {min_blocks} block(s).",
        )

    def ts_to_sec(ts: str) -> Optional[float]:
        try:
            ts = ts.replace(",", ".")
            parts = ts.split(":")
            if len(parts) != 3:
                return None
            h, m, r = parts
            return int(h) * 3600 + int(m) * 60 + float(r)
        except (ValueError, AttributeError):
            return None

    durations: List[float] = []
    lines: List[str] = []
    block_numbers: List[int] = []

    for idx, block in enumerate(blocks, 1):
        block_lines = block.splitlines()
        if not block_lines:
            return False, f"Empty block structure parsed at block sequence index {idx}."
        
        first_line = block_lines[0].strip()
        if not first_line.isdigit():
            return False, f"Block missing numeric index header: '{first_line[:20]}' at parsed segment {idx}."
        else:
            block_numbers.append(int(first_line))

        match = SRT_TIMESTAMP_PATTERN.search(block)
        if match:
            t_start = ts_to_sec(match.group(1))
            t_end = ts_to_sec(match.group(2))
            if t_start is not None and t_end is not None:
                if t_start > t_end:
                    return False, f"Inverted timestamp detected: start {match.group(1)} > end {match.group(2)}."
                durations.append(max(0.0, t_end - t_start))
        
        for ln in block_lines[2:]:
            stripped = ln.strip()
            if stripped:
                lines.append(stripped.lower())

    if block_numbers:
        expected = list(range(block_numbers[0], block_numbers[0] + len(block_numbers)))
        if block_numbers != expected:
            return False, "Non-sequential block numbering detected"

    if not durations:
        return False, "No valid timestamps found."

    avg = sum(durations) / len(durations)
    if avg > max_duration:
        return (
            False,
            f"Avg block duration {avg:.1f}s > {max_duration}s -- likely hallucination.",
        )
    if avg < min_duration:
        return (
            False,
            f"Avg block duration {avg:.3f}s < {min_duration}s -- flash hallucination.",
        )

    if lines:
        dup = 1.0 - len(set(lines)) / len(lines)
    else:
        dup = 0.0

    if dup > dup_threshold:
        return (
            False,
            f"Duplicate line ratio {dup * 100:.0f}% > {dup_threshold * 100:.0f}% "
            f"-- looping hallucination.",
        )

    return True, "OK"


def verify_translation_quality(
    src_path: Union[str, Path],
    tgt_path: Union[str, Path],
    tgt_ext: str,
) -> Tuple[bool, str]:
    """
    Validate translation quality by comparing target blocks against source blocks.
    Flags files where average text length deviates excessively or contains garbled blocks.
    """
    src_map = parse_srt_to_dict(src_path)
    tgt_map = parse_srt_to_dict(tgt_path)
    if not src_map or not tgt_map:
        return False, "Empty or unparseable translation mapping."
    
    length_deviations = 0
    empty_blocks = 0
    total_blocks = len(tgt_map)
    
    for b_id, src_txt in src_map.items():
        if b_id not in tgt_map:
            continue
        tgt_txt = tgt_map[b_id]
        
        if not tgt_txt.strip():
            empty_blocks += 1
            continue
        
        s_len = len(src_txt)
        t_len = len(tgt_txt)
        
        # Check for extreme length deviation (e.g., target is > 4x longer or < 1/4 of source)
        # indicating hallucinated expansion or truncated/dropped output.
        if s_len > 10:  # Only flag significant length blocks
            if t_len > s_len * 4.0 or t_len < s_len * 0.25:
                length_deviations += 1
    
    # If more than 35% of blocks deviate excessively or are empty, trigger warnings
    if total_blocks > 0:
        deviation_ratio = length_deviations / total_blocks
        empty_ratio = empty_blocks / total_blocks
        if deviation_ratio > 0.35:
            return False, f"{deviation_ratio:.0%} of blocks have excessive translation length deviation (possible hallucination)."
        if empty_ratio > 0.2:
            return False, f"{empty_ratio:.0%} of blocks were left blank or dropped."
            
    return True, "OK"


# ════════════════════════════════════════════════════════════
#  OLLAMA HELPER PROCEDURES
# ════════════════════════════════════════════════════════════
def get_local_ollama_models(api_url: str) -> List[str]:
    """Query local Ollama instance for pulled models to verify availability and prevent 404s."""
    try:
        raw_url = api_url.strip() if api_url else "http://localhost:11434"
        if not raw_url.startswith("http://") and not raw_url.startswith("https://"):
            raw_url = f"http://{raw_url}"
        parsed = urllib.parse.urlparse(raw_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        tags_url = f"{base_url}/api/tags"
        
        req = urllib.request.Request(tags_url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]
            return models
    except Exception as e:
        logger.debug("Failed to fetch local Ollama models list: %s", e)
        return []


def pull_ollama_model(api_url: str, model: str) -> bool:
    """Programmatically pull an Ollama model if it's missing from the device."""
    try:
        raw_url = api_url.strip() if api_url else "http://localhost:11434"
        if not raw_url.startswith("http://") and not raw_url.startswith("https://"):
            raw_url = f"http://{raw_url}"
        parsed = urllib.parse.urlparse(raw_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        pull_url = f"{base_url}/api/pull"
        
        qprint(f"  {style('[~]', C.CYAN)} Model '{model}' not found on device. Pulling from Ollama registry...")
        
        # Pull the model payload stream (use stream=False for atomic, high-timeout fetch)
        payload = json.dumps({"name": model, "stream": False}).encode("utf-8")
        req = urllib.request.Request(
            pull_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=3600) as response:
            if response.status == 200:
                qprint(f"  {style('[+]', C.GREEN)} Successfully pulled local Ollama model '{model}'!")
                return True
    except Exception as e:
        logger.debug("Failed to pull local Ollama model %s: %s", model, e)
        qprint(f"  {style('[x]', C.RED)} Failed to programmatically download local model '{model}': {e}")
    return False


# ════════════════════════════════════════════════════════════
#  BOX RENDERING AND FORMATTING TOOLS
# ════════════════════════════════════════════════════════════
def render_box(lines: List[str], width: int = 72) -> str:
    """Safely render lines wrapped inside a box structure of explicit width, with truncation guards."""
    max_line_len = max((len(strip_ansi(ln)) for ln in lines), default=0)
    if max_line_len > width:
        width = max_line_len

    rendered_lines = []
    rendered_lines.append(style(BOX_TL + BOX_HL * width + BOX_TR, C.CYAN, C.BOLD))
    for ln in lines:
        plain_len = len(strip_ansi(ln))
        if plain_len > width:
            ln = ln[:width-3] + "..."
            plain_len = len(strip_ansi(ln))
        padding = " " * (width - plain_len)
        rendered_lines.append(f"{style(BOX_VL, C.CYAN, C.BOLD)}{ln}{padding}{style(BOX_VL, C.CYAN, C.BOLD)}")
    rendered_lines.append(style(BOX_BL + BOX_HL * width + BOX_BR, C.CYAN, C.BOLD))
    return "\n".join(rendered_lines)


# ════════════════════════════════════════════════════════════
#  API KEY VALIDATION
# ════════════════════════════════════════════════════════════
def validate_api_key(key: str) -> Tuple[bool, str]:
    """Validate general key structure, permitting standard API credential symbols."""
    if not key or not key.strip():
        return False, "API key is empty"
    if len(key) < 10:
        return False, f"API key too short ({len(key)} chars, minimum 10)"
    if not API_KEY_PATTERN.match(key):
        return False, "API key contains invalid characters"
    return True, ""


def validate_tgt_ext(ext: str) -> Tuple[bool, str]:
    """Validate a target language extension string."""
    if not ext or not ext.strip():
        return False, "Extension is empty"
    clean = ext.strip().lower()
    if not re.match(r"^[a-z\-]+$", clean):
        return False, f"Extension must contain only letters and hyphens, got: {clean}"
    if len(clean) < 1 or len(clean) > 10:
        return False, f"Extension length must be 1-10 characters, got: {len(clean)}"
    return True, ""


# ════════════════════════════════════════════════════════════
#  SENSITIVE INFORMATION MASKING
# ════════════════════════════════════════════════════════════
def should_mask(key_name: str, value_str: str = "") -> bool:
    """Determine if a config key or string value contains sensitive components."""
    k = key_name.lower()
    if any(substr in k for substr in SENSITIVE_KEY_SUBSTRINGS):
        return True
    if value_str:
        val_lower = value_str.lower()
        if "key=" in val_lower or "api_key=" in val_lower or "token=" in val_lower:
            return True
    return False


# ════════════════════════════════════════════════════════════
#  VALIDATION & AUDITING
# ════════════════════════════════════════════════════════════
def resolve_pipeline_steps(args: PipelineConfig, respect_interactive: bool = False) -> None:
    """Consistently resolve active steps based on CLI constraints, key validation, and skip options."""
    if not respect_interactive:
        args.transcribe = not getattr(args, "skip_transcribe", False)
    else:
        if getattr(args, "skip_transcribe", False):
            args.transcribe = False

    has_api_key = bool(getattr(args, "api_key", ""))
    is_google = (getattr(args, "translator", "gemini") == "google")
    is_ollama = (getattr(args, "translator", "gemini") == "ollama")
    
    if not respect_interactive:
        args.translate = (has_api_key or is_google or is_ollama) and not getattr(args, "skip_translate", False)
    else:
        if getattr(args, "skip_translate", False):
            args.translate = False
        else:
            args.translate = args.translate and (has_api_key or is_google or is_ollama)

    if not respect_interactive:
        args.embed = not getattr(args, "skip_embed", False)
    else:
        if getattr(args, "skip_embed", False):
            args.embed = False
    
    if getattr(args, "hardsub", False) and not args.embed:
        args.embed = True


def validate_args(args: PipelineConfig) -> None:
    """Resolve configuration conflicts and apply automatic overrides."""
    adjustments: List[str] = []
    
    if args.translator:
        args.translator = args.translator.lower().strip()
        supported = {"gemini", "openai", "anthropic", "deepl", "google", "ollama"}
        if args.translator not in supported:
            adjustments.append(f"Invalid translator '{args.translator}' detected. Resetting to default 'gemini'.")
            args.translator = "gemini"
            args.translation_model = "gemini-3.5-flash"
            Context.provenance["translator"] = "Auto-Override"
            Context.provenance["translation_model"] = "Auto-Override"

    # Auto-align the translation model defaults on mismatch to avoid endpoint errors
    default_models = {
        "gemini": "gemini-3.5-flash",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-haiku-4-5",
        "deepl": "deepl-translator",
        "google": "google-v1-free",
        "ollama": "qwen2.5:7b",
    }
    
    if args.translate or args.translator:
        current_trans = args.translator.lower().strip() if args.translator else "gemini"
        current_model = getattr(args, "translation_model", "")
        is_mismatch = False
        if current_trans == "gemini" and "gemini" not in current_model.lower():
            is_mismatch = True
        elif current_trans == "openai" and "gpt" not in current_model.lower() and "o1" not in current_model.lower():
            is_mismatch = True
        elif current_trans == "anthropic" and "claude" not in current_model.lower():
            is_mismatch = True
            
        if is_mismatch:
            fallback_m = default_models.get(current_trans, "gemini-3.5-flash")
            adjustments.append(f"Model mismatch detected for '{current_trans}'. Aligning translation model to '{fallback_m}'.")
            args.translation_model = fallback_m
            Context.provenance["translation_model"] = "Auto-Override"

    if args.hardsub and not args.embed:
        adjustments.append(
            "Hardsub is enabled but Muxing is disabled. "
            "(Burning subtitles requires muxing; auto-enabling Mux.)"
        )
        args.embed = True
        Context.provenance["embed"] = "Auto-Override"

    if args.translate and not args.api_key and args.translator not in ("google", "ollama"):
        adjustments.append(
            "Translation is requested, but no API key was configured. "
            "(Disabling translate step.)"
        )
        args.translate = False
        Context.provenance["translate"] = "Auto-Override"
    elif args.translate and args.api_key:
        is_valid, err_msg = validate_api_key(args.api_key)
        if not is_valid:
            adjustments.append(f"Invalid API key format: {err_msg}. (Disabling translate.)")
            args.translate = False
            Context.provenance["translate"] = "Auto-Override"

    if hasattr(args, "tgt_ext") and args.tgt_ext:
        is_valid, err_msg = validate_tgt_ext(args.tgt_ext)
        if not is_valid:
            adjustments.append(f"Invalid target extension: {err_msg}. Using default 'en'.")
            args.tgt_ext = "en"

    if adjustments:
        qprint(
            f"\n{style('[!]', C.YELLOW)} Configuration Conflicts Resolved "
            f"(Overriding Variables):"
        )
        for msg in adjustments:
            qprint(f"    · {msg}")
        qprint()


def write_audit_log(
    args: PipelineConfig, summary: List[Tuple[str, FileStatus, float]], total_elapsed: float
) -> None:
    """Write a structured JSON audit log of the pipeline run with rotation."""
    if args.no_audit:
        return

    logs_dir = APP_DIR / "logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = logs_dir / f"run-{timestamp}.json"

        serializable_args: Dict[str, Any] = {}
        for k, v in vars(args).items():
            val = str(v) if isinstance(v, Path) else v
            if should_mask(k, str(val) if val else ""):
                val = "[PRESENT]" if val else "[ABSENT]"
            serializable_args[k] = val

        log_data = {
            "timestamp": timestamp,
            "total_elapsed_seconds": total_elapsed,
            "configuration": serializable_args,
            "processed_files": [
                {
                    "file_name": name,
                    "pipeline_status": status.as_dict() if isinstance(status, FileStatus) else status,
                    "elapsed_seconds": elapsed,
                }
                for name, status, elapsed in summary
            ],
        }
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)

        qprint(f"  {style('[+]', C.GREEN)} Audit log recorded to: {log_file}")

        now = time.time()
        for f in logs_dir.glob("run-*.json"):
            if now - f.stat().st_mtime > AUDIT_LOG_MAX_AGE_DAYS * 86400:
                try:
                    f.unlink(missing_ok=True)
                except OSError:
                    pass

        log_files = sorted(
            list(logs_dir.glob("run-*.json")),
            key=lambda p: p.stat().st_mtime,
        )
        max_logs = max(1, getattr(args, "max_audit_logs", MAX_AUDIT_LOGS))
        while len(log_files) > max_logs:
            oldest = log_files.pop(0)
            if oldest.resolve() != log_file.resolve():
                try:
                    oldest.unlink(missing_ok=True)
                except OSError:
                    pass
            else:
                break
    except Exception as e:
        logger.debug("Failed to write audit log: %s", e)


def print_effective_settings(args: PipelineConfig) -> None:
    """Display the effective pipeline configuration."""

    def get_src(key_name: str) -> str:
        src = Context.provenance.get(key_name, "Default")
        return f"[{src}]"

    qprint(
        f"\n{C.BOLD}── Active Pipeline Parameters "
        f"────────────────────────────{C.RESET}"
    )
    qprint(
        f"  Target Folder:     {style(args.folder, C.CYAN):<49} "
        f"{style(get_src('folder'), C.DIM)}"
    )
    qprint(
        f"  Transcription:     "
        f"{style('Enabled' if args.transcribe else 'Disabled', C.CYAN):<49} "
        f"{style(get_src('model') + ' Model: ' + args.model.upper(), C.DIM)}"
    )
    qprint(
        f"  Compute Device:    "
        f"{style(args.device.upper(), C.CYAN):<49} "
        f"{style(get_src('device'), C.DIM)}"
    )
    qprint(
        f"  Translation:       "
        f"{style('Enabled' if args.translate else 'Disabled', C.CYAN):<49} "
        f"{style(get_src('tgt_lang') + ' Target: ' + args.tgt_lang + ' [.' + args.tgt_ext + ']', C.DIM)}"
    )
    qprint(
        f"  Translator Provider: {style(args.translator.upper(), C.CYAN):<49} "
        f"{style('Model: ' + args.translation_model, C.DIM)}"
    )
    hardsub_label = "Hardsub (Burn-in)" if args.hardsub else (
        "Softsub (Mux)" if args.embed else "Disabled"
    )
    embed_src = get_src("hardsub") if args.hardsub else get_src("embed")
    qprint(
        f"  Final Muxing:      {style(hardsub_label, C.CYAN):<49} "
        f"{style(embed_src, C.DIM)}"
    )
    qprint(
        f"  Watch Mode:        "
        f"{style('Enabled' if args.watch else 'Disabled', C.CYAN):<49} "
        f"{style('[CLI]', C.DIM)}"
    )
    qprint(
        f"  Dry Run Mode:      "
        f"{style('Active' if args.dry_run else 'Inactive', C.CYAN):<49} "
        f"{style('[CLI]', C.DIM)}"
    )
    qprint(
        f"  Audit Logs:        "
        f"{style('Disabled' if args.no_audit else 'Enabled', C.CYAN):<49} "
        f"{style('[CLI]', C.DIM)}"
    )
    qprint(
        f"  Min Blocks Req:    "
        f"{style(str(args.min_blocks), C.CYAN):<49} "
        f"{style(get_src('min_blocks'), C.DIM)}"
    )
    qprint(f"{C.BOLD}───────────────────────────────────────────────────────────{C.RESET}\n")


# ════════════════════════════════════════════════════════════
#  BATCH SUMMARY FLAG PARSER
# ════════════════════════════════════════════════════════════
def _determine_flag(st: FileStatus) -> Tuple[str, str, Optional[int]]:
    """Determine the status label, ANSI color code, and an optional fallback block count."""
    if st.error:
        return "FAULT", C.RED, None
    if st.audio_failed:
        return "AUDIO_FAIL", C.RED, None
    if st.skipped:
        return "SKIPPED", C.YELLOW, None
    if st.reused_all:
        return "REUSED", C.GREEN, None
    if st.translated and st.muxed:
        return "DONE_TRN", C.GREEN, None
    if st.mixed_language and st.muxed:
        return "MIXED", C.YELLOW, st.fallback_count
    if st.reused_srt and st.muxed:
        return "DONE_MUX", C.GREEN, None
    if st.transcribed and (st.translated or st.mixed_language):
        if st.mixed_language:
            return "TXT+MIX", C.CYAN, st.fallback_count
        return "TXT+TRN", C.CYAN, None
    if st.transcribed:
        return "TXT_ONLY", C.CYAN, None
    if st.translated or st.reused_srt or st.mixed_language:
        if st.mixed_language:
            return "MIX_TRN", C.CYAN, st.fallback_count
        return "TRN_ONLY", C.CYAN, None
    return "NO-OP", C.DIM, None


def print_summary(
    summary: List[Tuple[str, FileStatus, float]],
    total_elapsed: float,
    args: PipelineConfig,
) -> None:
    """Print a formatted batch completion summary table with aligned headers."""
    W = 72
    title_text = " BATCH REPORT "
    rem = W - len(title_text)
    left_dashes = rem // 2
    right_dashes = rem - left_dashes
    top = BOX_TL + BOX_HL * left_dashes + title_text + BOX_HL * right_dashes + BOX_TR
    mid = BOX_ML + BOX_HL * W + BOX_MR
    bot = BOX_BL + BOX_HL * W + BOX_BR

    print(f"\n{style(top, C.CYAN, C.BOLD)}")
    title = f"  Batch Complete -- {fmt_time(total_elapsed)}"
    print(f"{style(BOX_VL, C.CYAN, C.BOLD)}{title:<{W}}{style(BOX_VL, C.CYAN, C.BOLD)}")
    print(f"{style(mid, C.CYAN, C.BOLD)}")

    def pad_flag(label: str, color: str, num: Optional[int] = None) -> str:
        if num is not None:
            num_str = "999+" if num > 999 else str(num)
            content = f"{label:<7}{num_str:>3}"
        else:
            content = f"{label:<10}"
        return f"{color}[{content}]{C.RESET}"

    for name, st, t in summary:
        t_str = fmt_time(t).rjust(6)
        
        label, color, num = _determine_flag(st)
        flag = pad_flag(label, color, num)

        flag_raw = strip_ansi(flag)
        name_width = max(20, W - 2 - len(flag_raw) - 1 - len(t_str) - 1)
        if len(name) > name_width:
            name_trunc = name[: name_width - 1] + "~"
        else:
            name_trunc = name.ljust(name_width)

        print(f"{style(BOX_VL, C.CYAN, C.BOLD)}  {flag} {name_trunc} {t_str} {style(BOX_VL, C.CYAN, C.BOLD)}")

    print(f"{style(bot, C.CYAN, C.BOLD)}")

    if getattr(args, "verbose_summary", False):
        print(
            f"\n{C.BOLD}── Verbose Execution Details "
            f"─────────────────────────────{C.RESET}"
        )
        for name, st, t in summary:
            details: List[str] = []
            if st.error:
                details.append("Execution encountered system errors.")
            if st.audio_failed:
                details.append("Audio preprocessing/extraction failed completely.")
            if st.skipped:
                details.append("Health check failed; file skipped.")
            if st.reused_all:
                details.append("Output file already exists; skipped rerun.")
            if st.transcribed:
                details.append("Transcribed audio using local Whisper engine.")
            if st.translated:
                details.append("Translated subtitles using translation API.")
            if st.mixed_language:
                details.append(
                    f"Completed with {st.fallback_count} likely fallback "
                    f"blocks matching original dialogue."
                )
            if st.reused_srt:
                details.append("Reused existing source subtitles.")
            if st.muxed:
                details.append("Muxed tracks into video container.")
            print(
                f"  * {name:<30} -> "
                f"{', '.join(details) if details else 'No actions executed.'}"
            )

    if getattr(args, "explain_summary", False):
        print(f"\n{C.DIM}Status Explanations:")
        print("  FAULT       - Processing failed or error occurred")
        print("  AUDIO_FAIL  - Audio extraction failed (unable to run transcription)")
        print("  SKIPPED     - File skipped (e.g. failed health check)")
        print("  REUSED      - Reused output from previous run")
        print("  DONE_TRN    - Fully transcribed and translated")
        print("  MIXED(N)    - Translation completed, N blocks fell back")
        print("  TXT+MIX(N)  - Transcribed and translated with N fallback blocks")
        print("  TXT_ONLY    - Transcribed only (no translation applied)")
        print("  TRN_ONLY    - Translated only (existing source SRT reused)")
        print("  MIX_TRN(N)  - Reused source SRT, translated with N fallback blocks")
        print("  DONE_MUX    - Reused existing SRT and muxed into video")
        print("  NO-OP       - No operations performed on this file")


# ════════════════════════════════════════════════════════════
#  TRANSLATION PROVIDER ENVELOPE BUILDERS
# ════════════════════════════════════════════════════════════
def _build_gemini_payload(model: str, prompt_or_list: Union[str, List[str]], api_key: str, url_override: str) -> Tuple[str, bytes, Dict[str, str]]:
    req_url = url_override.strip() if url_override else GEMINI_URL_TEMPLATE.format(model=model, key=api_key)
    if url_override and "?" not in req_url:
        req_url = f"{req_url}?key={api_key}"
    payload_dict = {"contents": [{"parts": [{"text": prompt_or_list}]}]}
    headers = {"Content-Type": "application/json"}
    return req_url, json.dumps(payload_dict).encode("utf-8"), headers


def _build_openai_payload(model: str, prompt_or_list: Union[str, List[str]], api_key: str, url_override: str, tgt_lang: str) -> Tuple[str, bytes, Dict[str, str]]:
    req_url = url_override.strip() if url_override else "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload_dict = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are a professional translator specializing in film subtitles. "
                    f"Translate the provided blocks accurately to {tgt_lang}. "
                    "Do not output conversational text or wrapper blocks. Maintain raw format: 'Block #[ID]: [translation]'"
                )
            },
            {"role": "user", "content": prompt_or_list}
        ],
        "temperature": 0.3
    }
    return req_url, json.dumps(payload_dict).encode("utf-8"), headers


def _build_anthropic_payload(model: str, prompt_or_list: Union[str, List[str]], api_key: str, url_override: str, tgt_lang: str) -> Tuple[str, bytes, Dict[str, str]]:
    req_url = url_override.strip() if url_override else "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "Anthropic-Version": "2023-06-01"
    }
    payload_dict = {
        "model": model,
        "max_tokens": 4000,
        "messages": [
            {"role": "user", "content": f"You are a professional translator. Translate these blocks to {tgt_lang}. Maintain raw formats. Only output translated blocks formatted as 'Block #[ID]: [translated_text]':\n\n{prompt_or_list}"}
        ],
        "temperature": 0.3
    }
    return req_url, json.dumps(payload_dict).encode("utf-8"), headers


def _build_deepl_payload(model: str, prompt_or_list: Union[str, List[str]], api_key: str, url_override: str, tgt_ext: str) -> Tuple[str, bytes, Dict[str, str]]:
    req_url = url_override.strip() if url_override else ("https://api-free.deepl.com/v2/translate" if api_key.endswith(":fx") else "https://api.deepl.com/v2/translate")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"DeepL-Auth-Key {api_key}"
    }
    lang_upper = tgt_ext.upper()
    if lang_upper == "EN":
        lang_upper = "EN-US"
    elif lang_upper == "PT":
        lang_upper = "PT-BR"
    payload_dict = {
        "text": prompt_or_list,
        "target_lang": lang_upper
    }
    return req_url, json.dumps(payload_dict).encode("utf-8"), headers


def _build_google_payload(prompt_or_list: Union[str, List[str]], url_override: str, tgt_ext: str) -> Tuple[str, bytes, Dict[str, str]]:
    req_url = url_override.strip() if url_override else f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={tgt_ext}&dt=t"
    joined_text = "\n###\n".join(prompt_or_list)
    payload_dict = {"q": joined_text}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = urllib.parse.urlencode(payload_dict).encode("utf-8")
    return req_url, payload, headers


def _build_ollama_payload(model: str, prompt_or_list: Union[str, List[str]], url_override: str, tgt_lang: str) -> Tuple[str, bytes, Dict[str, str]]:
    # Normalize the host target url to ensure endpoints are combined cleanly
    raw_url = url_override.strip() if url_override else "http://localhost:11434"
    if not raw_url.startswith("http://") and not raw_url.startswith("https://"):
        raw_url = f"http://{raw_url}"
        
    parsed = urllib.parse.urlparse(raw_url)
    path = parsed.path if parsed.path else "/api/chat"
    if path in ("", "/"):
        path = "/api/chat"
        
    req_url = f"{parsed.scheme}://{parsed.netloc}{path}"
    headers = {"Content-Type": "application/json"}
    
    model_lower = model.lower()
    # Structural adaptation depending on dedicated NMT vs general LLM instruction alignments
    if "nllb" in model_lower or "opus" in model_lower or "helsinki" in model_lower:
        system_instruction = (
            f"Translate the following raw subtitle blocks directly into accurate and natural {tgt_lang}. "
            "Only output the translated blocks following the template 'Block #[ID]: [translation]'."
        )
    else:
        system_instruction = (
            f"You are a professional film subtitle translator. "
            f"Translate the provided blocks accurately into highly natural and context-aware {tgt_lang}. "
            "Do not output conversational filler, introductions, explanations, or markdown wrappers. "
            "Strictly follow this exact output template: 'Block #[ID]: [translated_text]'"
        )

    payload_dict = {
        "model": model if model else "qwen2.5:7b",
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_or_list}
        ],
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }
    return req_url, json.dumps(payload_dict).encode("utf-8"), headers


# ════════════════════════════════════════════════════════════
#  TRANSLATION MOTOR MODULES
# ════════════════════════════════════════════════════════════
def _execute_translator_request(
    translator: str,
    model: str,
    url_override: str,
    api_key: str,
    prompt_or_list: Union[str, List[str]],
    tgt_lang: str,
    tgt_ext: str,
) -> Union[str, Dict[int, str]]:
    """Execute raw HTTP request targeting the selected translation provider with fallbacks."""
    translator = translator.lower().strip()
    supported_translators = {"gemini", "openai", "anthropic", "deepl", "google", "ollama"}
    if translator not in supported_translators:
        raise ValueError(f"Unsupported translator provider choice: {translator}")

    # Check model tags locally for Ollama tasks to verify pull statuses or auto-fallback
    if translator == "ollama":
        local_models = get_local_ollama_models(url_override)
        target_clean = model.strip().lower()
        matched_model = None
        target_has_tag = ":" in target_clean

        if local_models:
            for lm in local_models:
                lm_clean = lm.strip().lower()
                if target_clean == lm_clean:
                    matched_model = lm
                    break
                if not target_has_tag and lm_clean == f"{target_clean}:latest":
                    matched_model = lm
                    break
        
        if matched_model:
            model = matched_model
        else:
            # Model not on device -> Programmatically pull
            pull_success = pull_ollama_model(url_override, model)
            if pull_success:
                local_models = get_local_ollama_models(url_override)
                for lm in local_models:
                    lm_clean = lm.strip().lower()
                    if target_clean == lm_clean or (not target_has_tag and lm_clean == f"{target_clean}:latest"):
                        model = lm
                        break
            else:
                if local_models:
                    fallback_model = local_models[0]
                    qprint(
                        f"  {style('[!]', C.YELLOW)} Model pull failed or was bypassed. "
                        f"Using existing model '{fallback_model}' to prevent execution crash."
                    )
                    model = fallback_model
                else:
                    raise TranslationError(
                        f"Model '{model}' is not installed locally and pulling it failed. "
                        "No local Ollama models are available to fall back to."
                    )

    if not global_rate_limiter.acquire(timeout=30):
        raise RuntimeError("API request blocked: Rate limiter token acquisition timeout.")

    success = False
    response_text = ""
    translated_map: Dict[int, str] = {}

    ssl_context = None
    try:
        if _CERTIFI_AVAILABLE and certifi:
            ssl_context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass

    for attempt in range(MAX_TRANSLATION_CHUNK_RETRIES):
        try:
            if translator == "gemini":
                req_url, payload, headers = _build_gemini_payload(model, prompt_or_list, api_key, url_override)
            elif translator == "openai":
                req_url, payload, headers = _build_openai_payload(model, prompt_or_list, api_key, url_override, tgt_lang)
            elif translator == "anthropic":
                req_url, payload, headers = _build_anthropic_payload(model, prompt_or_list, api_key, url_override, tgt_lang)
            elif translator == "deepl":
                req_url, payload, headers = _build_deepl_payload(model, prompt_or_list, api_key, url_override, tgt_ext)
            elif translator == "google":
                req_url, payload, headers = _build_google_payload(prompt_or_list, url_override, tgt_ext)
            elif translator == "ollama":
                req_url, payload, headers = _build_ollama_payload(model, prompt_or_list, url_override, tgt_lang)

            headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SubsPipeline/1.3"

            req = urllib.request.Request(req_url, data=payload, headers=headers, method="POST")
            
            # Apply a generous timeout limit for Ollama execution requests to handle cold loading
            req_timeout = 600 if translator == "ollama" else GEMINI_TIMEOUT
            
            with urllib.request.urlopen(req, timeout=req_timeout, context=ssl_context) as response:
                resp_bytes = response.read()
                if response.info().get("Content-Encoding") == "gzip":
                    resp_bytes = gzip.decompress(resp_bytes)
                resp_decoded = resp_bytes.decode("utf-8", errors="replace")

                if translator == "google":
                    resp_data = json.loads(resp_decoded)
                    segments = resp_data[0] if resp_data and isinstance(resp_data, list) else []
                    parts_translated = []
                    if segments:
                        for segment in segments:
                            if segment and isinstance(segment, list) and len(segment) > 0 and segment[0]:
                                parts_translated.append(segment[0])
                    
                    stitched_translation = "".join(parts_translated)
                    split_pattern = re.compile(r"\s*#\s*#\s*#\s*")
                    parts = split_pattern.split(stitched_translation)
                    
                    for idx, part in enumerate(parts):
                        translated_map[idx] = part.strip()
                    success = True
                else:
                    resp_data = json.loads(resp_decoded)
                    if translator == "gemini":
                        candidates = resp_data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                response_text = parts[0].get("text", "")
                                if response_text.strip():
                                    success = True
                    elif translator == "openai":
                        choices = resp_data.get("choices", [])
                        if choices:
                            response_text = choices[0].get("message", {}).get("content", "")
                            if response_text.strip():
                                success = True
                    elif translator == "anthropic":
                        content_list = resp_data.get("content", [])
                        if content_list:
                            response_text = content_list[0].get("text", "")
                            if response_text.strip():
                                success = True
                    elif translator == "deepl":
                        translations = resp_data.get("translations", [])
                        if translations:
                            for idx, item in enumerate(translations):
                                translated_map[idx] = item.get("text", "")
                            success = True
                    elif translator == "ollama":
                        message = resp_data.get("message", {})
                        response_text = message.get("content", "")
                        if not response_text:
                            # Fallback configuration check in the event an OpenAI proxy target configuration was mapped
                            choices = resp_data.get("choices", [])
                            if choices:
                                response_text = choices[0].get("message", {}).get("content", "")
                        if response_text.strip():
                            success = True

                if success:
                    Context.reset_consecutive_429s()
                    Context.reset_consecutive_total_failures()
                    break

        except urllib.error.HTTPError as e:
            if e.code == 404 and translator == "ollama":
                raise TranslationError(
                    f"Ollama returned HTTP 404 (Not Found).\n"
                    f"      This usually means model '{model}' is not installed.\n"
                    f"      Please open a terminal and run: ollama pull {model}"
                )

            if e.code == 400 and model != FALLBACK_MODELS.get(translator):
                fallback = FALLBACK_MODELS.get(translator)
                if fallback:
                    logger.warning("API returned 400. Attempting fallback model: %s", fallback)
                    model = fallback
                    time.sleep(TRANSLATION_RETRY_BASE_DELAY)
                    continue

            if e.code == 429:
                Context.increment_consecutive_429s()
                if Context.get_consecutive_429s() >= CONSECUTIVE_429_LIMIT:
                    Context.set_translation_disabled(True)
                    raise TranslationError("Rate limits (429) hit consecutively. Suspending API translation.")
                time.sleep(5 * attempt + 5)
            elif e.code == 503:
                logger.warning("Transient backend error 503 received. Retrying with delay...")
                if attempt == MAX_TRANSLATION_CHUNK_RETRIES - 1:
                    Context.increment_consecutive_total_failures()
                    raise TranslationError(f"API Request failed after retries with HTTP Error {e.code}.")
                time.sleep(TRANSLATION_RETRY_BASE_DELAY * (3**attempt) + random.uniform(1, 3))
            elif e.code >= 500:
                logger.warning("Transient backend error %d received. Retrying...", e.code)
                if attempt == MAX_TRANSLATION_CHUNK_RETRIES - 1:
                    Context.increment_consecutive_total_failures()
                    raise TranslationError(f"API Request failed after retries with HTTP Error {e.code}")
                time.sleep(TRANSLATION_RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1))
            else:
                Context.increment_consecutive_total_failures()
                raise TranslationError(f"API Request failed with HTTP Error {e.code}")
        except urllib.error.URLError as e:
            if translator == "ollama":
                raise TranslationError(
                    f"Failed to connect to local Ollama server at '{url_override or 'http://localhost:11434'}'.\n"
                    f"      Ensure the Ollama application is running and accessible."
                )
                
            logger.warning("Connection failure encountered: %s. Retrying...", e.reason)
            if attempt == MAX_TRANSLATION_CHUNK_RETRIES - 1:
                Context.increment_consecutive_total_failures()
                raise TranslationError(f"API Request failed after retries with URLError: {e.reason}")
            time.sleep(TRANSLATION_RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1))
        except Exception as e:
            logger.warning("Unexpected error during API request: %s. Retrying...", e)
            if attempt == MAX_TRANSLATION_CHUNK_RETRIES - 1:
                Context.increment_consecutive_total_failures()
                raise TranslationError(f"API Request failed after retries with unexpected error: {e}")
            time.sleep(TRANSLATION_RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1))

    if not success:
        raise TranslationError("API translation request sequence exhausted without retrieving structured data.")

    return translated_map if translator in ("deepl", "google") else response_text


def _parse_translation_response(response_text: str) -> Dict[int, str]:
    """Parse output text back into structured layout components mapping IDs to text with regex boundary isolation."""
    parsed_translations: Dict[int, str] = {}
    if response_text.strip():
        # Match "Block #ID:" or similar specific pattern cleanly, preventing split errors on text matching "Block"
        parts = re.split(r"Block\s*(?:#|No\s*|\s)\s*(\d+)\s*:?", response_text, flags=re.IGNORECASE)
        
        if len(parts) >= 3:
            for idx in range(1, len(parts), 2):
                try:
                    b_num = int(parts[idx])
                    body = parts[idx + 1].strip() if idx + 1 < len(parts) else ""
                    parsed_translations[b_num] = body
                except ValueError:
                    pass
        else:
            # Fallback to lines parsing if the regex split pattern did not match correctly
            for part in re.split(r"Block\s*(?:#|No\s*|\s)\s*", response_text, flags=re.IGNORECASE):
                part = part.strip()
                if not part:
                    continue
                lines = part.splitlines()
                header = lines[0] if lines else ""
                m = re.match(r"^(\d+):?", header)
                if m:
                    b_num = int(m.group(1))
                    body = "\n".join(lines[1:]) if len(lines) > 1 else ""
                    if body.startswith(":"):
                        body = body[1:].strip()
                    parsed_translations[b_num] = body.strip()
    return parsed_translations


def translate_srt_native(
    srt_src: Union[str, Path],
    srt_tgt: Union[str, Path],
    tgt_lang: str,
    api_key: str,
    translator: str = "gemini",
    translation_model: Optional[str] = None,
    api_url: str = "",
    fallback_match_threshold: float = 0.95,
    tgt_ext: str = "en",
    src_lang: Optional[str] = None,
) -> Tuple[bool, str, int]:
    """Translate an SRT file using the selected translation router API."""
    srt_src = Path(srt_src)
    srt_tgt = Path(srt_tgt)

    is_valid, reason = is_valid_srt(srt_src)
    if not is_valid:
        return False, f"Source SRT validation failed: {reason}", 0

    try:
        content = srt_src.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return False, f"Read error: {e}", 0

    blocks = [b.strip() for b in re.split(r"\n\n+", content.strip()) if b.strip()]
    if not blocks:
        return False, "Empty SRT file.", 0

    translator = translator.lower().strip()
    
    if translator not in ("google", "ollama"):
        key_valid, key_err = validate_api_key(api_key)
        if not key_valid:
            return False, f"Invalid API key: {key_err}", 0

    model = translation_model or (
        "gemini-3.5-flash" if translator == "gemini" else
        "gpt-4o-mini" if translator == "openai" else
        "claude-haiku-4-5" if translator == "anthropic" else
        "google-v1-free" if translator == "google" else
        "qwen2.5:7b" if translator == "ollama" else
        "deepl-translator"
    )

    chunk_size = GEMINI_CHUNK_SIZE
    
    # Character-length aware chunk allocation to protect provider contexts
    chunks: List[List[str]] = []
    current_chunk: List[str] = []
    current_chars = 0
    
    for b in blocks:
        b_len = len(b)
        if len(current_chunk) >= chunk_size or (current_chars + b_len > MAX_CHUNK_CHAR_LIMIT and current_chunk):
            chunks.append(current_chunk)
            current_chunk = []
            current_chars = 0
        current_chunk.append(b)
        current_chars += b_len
        
    if current_chunk:
        chunks.append(current_chunk)
        
    total_chunks = len(chunks)
    tmp_tgt = srt_tgt.with_suffix(".tmp")
    translated_count = 0

    try:
        # Use temp_file_guard for guaranteed automatic cleanup of tmp_tgt
        with temp_file_guard(tmp_tgt) as guarded_tmp:
            with open(guarded_tmp, "w", encoding="utf-8") as tmp_fh:

                for chunk_idx, chunk in enumerate(chunks, 1):
                    qprint(
                        f"  {style('[~]', C.DIM)} Translating chunk {chunk_idx}/{total_chunks} "
                        f"({len(chunk)} blocks) using {translator.upper()}...{C.RESET}"
                    )

                    if translator in ("deepl", "google"):
                        diags = []
                        for b in chunk:
                            lines = b.splitlines()
                            diag = "\n".join(lines[2:]) if len(lines) >= 3 else ""
                            diags.append(diag)
                        try:
                            translated_dict = _execute_translator_request(
                                translator, model, api_url, api_key, diags, tgt_lang, tgt_ext
                            )
                        except Exception as api_err:
                            return False, f"{translator.upper()} translation chunk {chunk_idx}/{total_chunks} failed: {api_err}", 0
                    else:
                        prompt = _prepare_translation_prompt(chunk, tgt_lang)
                        try:
                            response_text = _execute_translator_request(
                                translator, model, api_url, api_key, prompt, tgt_lang, tgt_ext
                            )
                            translated_dict = _parse_translation_response(response_text)
                        except Exception as api_err:
                            return False, f"Translation chunk {chunk_idx}/{total_chunks} failed: {api_err}", 0

                    for idx_in_chunk, b in enumerate(chunk):
                        lines = b.splitlines()
                        if not lines:
                            continue
                        b_idx = int(lines[0]) if lines[0].isdigit() else (translated_count + 1)
                        ts = (
                            lines[1]
                            if len(lines) >= 2
                            else "00:00:00,000 --> 00:00:00,000"
                        )

                        orig_diag = "\n".join(lines[2:]) if len(lines) >= 3 else ""
                        
                        if translator in ("deepl", "google"):
                            translated_diag = translated_dict.get(idx_in_chunk, orig_diag)
                        else:
                            translated_diag = translated_dict.get(b_idx, orig_diag)

                        if not translated_diag.strip() and orig_diag.strip():
                            translated_diag = orig_diag

                        tmp_fh.write(f"{b_idx}\n{ts}\n{translated_diag}\n\n")
                        translated_count += 1

                    if chunk_idx < total_chunks:
                        time.sleep(GEMINI_INTER_CHUNK_DELAY)

            os.replace(str(guarded_tmp), str(srt_tgt))

        fallbacks, _ = detect_fallbacks(srt_src, srt_tgt, fallback_match_threshold, lang_code=(src_lang or "en"))
        return True, "Success", fallbacks

    except KeyboardInterrupt:
        logger.warning("Translation interrupted by user")
        return False, "Translation interrupted by user", 0
    except Exception as e:
        logger.error("Translation error: %s", e)
        return False, f"Write error: {e}", 0


# ════════════════════════════════════════════════════════════
#  DIRECTORY WATCHER
# ════════════════════════════════════════════════════════════
def run_watcher(args: PipelineConfig) -> None:
    """Run a filesystem watcher for automatic processing of new media files."""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        qprint(f"\n{C.YELLOW}  [!] watchdog not installed: pip install watchdog{C.RESET}")
        return

    in_flight: Dict[Path, float] = {}
    lock = threading.Lock()
    folder_root = Path(args.folder).resolve()

    def clean_expired_in_flight() -> None:
        now = time.monotonic()
        for path_obj, ts in list(in_flight.items()):
            if now - ts > WATCHER_IN_FLIGHT_EXPIRY:  # Expire stale lock files
                in_flight.pop(path_obj, None)

    class WatchHandler(FileSystemEventHandler):

        def _should_process(self, path: str) -> bool:
            resolved = Path(path).resolve()
            if not str(resolved).lower().endswith(MEDIA_EXTS):
                return False
            # Check for local processing temp files explicitly to prevent infinite feedback loops
            if resolved.name.startswith("temp_"):
                return False
            if not is_safe_relative(resolved, folder_root):
                return False
            try:
                rel = resolved.relative_to(folder_root)
                if rel.parts and any(p.startswith("muxed_") for p in rel.parts):
                    return False
            except ValueError:
                return False
            return True

        def _dispatch(self, path: str) -> None:
            if not self._should_process(path):
                return

            resolved = Path(path).resolve()
            with lock:
                clean_expired_in_flight()
                if resolved in in_flight:
                    return
                in_flight[resolved] = time.monotonic()

            def handle() -> None:
                try:
                    name = resolved.name
                    clamped_name = (name[:35] + "...") if len(name) > 38 else name
                    qprint(f"\n{C.CYAN}  [*] Detected: {clamped_name}{C.RESET}")
                    if not wait_for_file_settle(resolved):
                        qprint(
                            f"  {C.YELLOW}[!] Warning: File {clamped_name} is "
                            f"locked/busy. Skipping watch thread execution.{C.RESET}"
                        )
                        return
                    time.sleep(WATCHER_SETTLE_SECS)
                    process_file(resolved, args)
                    qprint(f"{C.GREEN}  [*] Idle -- awaiting files...{C.RESET}")
                except KeyboardInterrupt:
                    qprint(f"\n{C.YELLOW}[!] Watch thread interrupted.{C.RESET}")
                except Exception as e:
                    qprint(f"{C.RED}[x] Watch processing error: {e}{C.RESET}")
                finally:
                    with lock:
                        in_flight.pop(resolved, None)

            threading.Thread(target=handle, daemon=True).start()

        def on_created(self, event) -> None:
            if not event.is_directory:
                self._dispatch(event.src_path)

        def on_modified(self, event) -> None:
            if not event.is_directory:
                self._dispatch(event.src_path)

        def on_moved(self, event) -> None:
            if not event.is_directory:
                self._dispatch(event.dest_path)

    observer = Observer()
    observer.schedule(WatchHandler(), path=str(args.folder), recursive=False)
    observer.start()

    qprint(f"\n{C.CYAN}  [*] Watch Mode -- '{Path(args.folder).name}'{C.RESET}")
    qprint(f"  {C.DIM}Ctrl+C to stop.{C.RESET}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# ════════════════════════════════════════════════════════════
#  RATE LIMITER (Token Bucket)
# ════════════════════════════════════════════════════════════
class TokenBucketRateLimiter:
    """Token bucket rate limiter for proactive API throttling."""

    def __init__(self, rate: float = 1.0, capacity: int = 2) -> None:
        """Initialize the rate limiter."""
        self.rate = rate
        self.capacity = capacity
        self.tokens: float = float(capacity)
        self.last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """Acquire a token from the bucket."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True

            if not blocking:
                return False

            if deadline is not None and time.monotonic() >= deadline:
                return False

            time.sleep(RATE_LIMITER_POLL_INTERVAL)


# Global rate limiter: max 1 request per second with burst of 2
global_rate_limiter = TokenBucketRateLimiter(rate=1.0, capacity=2)


# ════════════════════════════════════════════════════════════
#  INTERACTIVE WIZARD
# ════════════════════════════════════════════════════════════
def interactive_wizard(
    args: PipelineConfig, cfg_memory: Dict[str, Any]
) -> None:
    """Run the interactive configuration wizard with API key validation checks."""
    print()
    banner_lines = [
        f"  Subs Pipeline v{__version__}",
        "  Multi-Language Media Transcription & Translation Pipeline"
    ]
    print(render_box(banner_lines, 60))
    print()

    print(
        f"  {style('[~]', C.DIM)} This utility automatically handles local multi-language "
        f"media pipeline runs.\n  Use the steps below to initialize models and "
        f"folders.\n"
    )

    if Context.migration_status == "loaded":
        print(f"  {style('[~]', C.GREEN)} Loaded saved profile settings.\n")

    if Context.config_warning:
        print(f"  {style('[!]', C.RED)} CONFIG SYSTEM: {Context.config_warning}\n")

    if not setup_ffmpeg():
        print(f"{C.RED}  [!] FFmpeg is required to continue.{C.RESET}")
        exit_app(1)

    # Step 1: Folder
    current_dir = os.getcwd()
    print(f"{C.BOLD}> Step 1: Media Directory{C.RESET}")
    print(f"  Current: {style(current_dir, C.CYAN)}")
    
    while True:
        f_in = input(f"  [Enter = current  |  path]: ").strip()
        if not f_in:
            args.folder = current_dir
            Context.provenance["folder"] = "Interactive"
            break
        else:
            resolved_p = Path(f_in).resolve()
            if resolved_p.is_dir():
                args.folder = str(resolved_p)
                Context.provenance["folder"] = "Interactive"
                break
            else:
                print(f"  {style('[!]', C.YELLOW)} Path does not exist or is not a directory. Try again.")

    print(f"  {style('->', C.GREEN)} {args.folder}\n")

    startup_garbage_collection(args.folder, skip_cleanup=args.no_cleanup)

    # Step 2: Saved Profiles Fast Path
    if cfg_memory:
        print(f"\n{C.BOLD}> Step 2: Use Saved Settings Profile?{C.RESET}")
        print(
            f"  Target Language:     {style(cfg_memory.get('tgt_lang', 'English'), C.CYAN)}"
        )
        print(
            f"  Subtitle Extension:  {style(cfg_memory.get('tgt_ext', 'en'), C.CYAN)}"
        )
        print(
            f"  Translation Service: {style(cfg_memory.get('translator', 'gemini').upper(), C.CYAN)} "
            f"(Model: {cfg_memory.get('translation_model', DEFAULT_GEMINI_MODEL)})"
        )
        print(
            f"  Min Blocks Required: {style(str(cfg_memory.get('min_blocks', 3)), C.CYAN)}"
        )
        print(
            f"  Saved Whisper Model: {style((cfg_memory.get('model') or 'None').upper(), C.CYAN)}"
        )
        args.device = cfg_memory.get("device", "auto")
        device, compute = resolve_device_and_compute(args.device)
        print(f"  Configured Device:   {style(device + ' (' + compute + ')', C.CYAN)}")

        fast_path = (
            input(
                f"\n  Use saved settings for {Path(args.folder).name}? [Y/n]: "
            )
            .strip()
            .lower()
            != "n"
        )
        if fast_path:
            args.tgt_lang = cfg_memory.get("tgt_lang", "English")
            args.tgt_ext = cfg_memory.get("tgt_ext", "en")
            args.src_lang = cfg_memory.get("src_lang") or None
            if args.src_lang and not args.src_lang.strip():
                args.src_lang = None
            
            args.translator = cfg_memory.get("translator", "gemini")
            args.translation_model = cfg_memory.get("translation_model", DEFAULT_GEMINI_MODEL)
            args.api_url = cfg_memory.get("api_url", "")
            
            args.api_key = cfg_memory.get("api_key", "")
            if not args.api_key and args.translator not in ("google", "ollama"):
                env_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "") or os.environ.get("DEEPL_API_KEY", "")
                if env_key:
                    is_valid, _ = validate_api_key(env_key)
                    if is_valid:
                        args.api_key = env_key
                    else:
                        qprint(f"  {style('[!]', C.YELLOW)} Ignored invalid credentials environment variable.")
                
            args.translate = (bool(args.api_key) or args.translator in ("google", "ollama")) and not getattr(args, "skip_translate", False)
            args.min_blocks = int(cfg_memory.get("min_blocks", 3))

            saved_model = cfg_memory.get("model")
            if saved_model:
                args.model = saved_model
            else:
                recommended = recommend_whisper_model()
                args.model = MODEL_MAP.get(recommended, "small")

            for k in [
                "tgt_lang",
                "tgt_ext",
                "src_lang",
                "translator",
                "translation_model",
                "api_url",
                "api_key",
                "min_blocks",
                "model",
                "device",
            ]:
                Context.provenance[k] = "Saved Profile"
            Context.provenance["translate"] = "Saved Profile"
            Context.provenance["transcribe"] = "Saved Profile"
            Context.provenance["embed"] = "Saved Profile"

            resolve_pipeline_steps(args, respect_interactive=True)

            print(
                f"\n  {style('Applying Saved Configuration Profile & Workflow Overrides:', C.CYAN)}"
            )
            print(
                f"    - Transcription:     "
                f"{style('Enabled' if args.transcribe else 'Disabled', C.CYAN)}"
                f"  (Model: {args.model.upper()} on {device.upper()})"
            )
            print(
                f"    - Translation:       "
                f"{style('Enabled' if args.translate else 'Disabled', C.CYAN)}"
                f" ({args.translator.upper()} Model: {args.translation_model})"
            )
            print(
                f"    - Final Muxing:      "
                f"{style('Enabled (Hardsub)' if args.hardsub else 'Enabled (Softsub)' if args.embed else 'Disabled', C.CYAN)}"
            )
            print(
                f"  {style('[+]', C.GREEN)} Loaded saved profile. Proceeding directly...\n"
            )
            return

    # Step 3: Target Translation settings
    print(f"{C.BOLD}> Step 3: Translation (Output) Settings{C.RESET}")
    args.tgt_lang = (
        input(f"  Language to TRANSLATE TO (e.g. Arabic) [{cfg_memory.get('tgt_lang', 'English')}]: ").strip()
        or cfg_memory.get("tgt_lang", "English")
    )
    Context.provenance["tgt_lang"] = "Interactive"

    if len(args.tgt_lang) < 2 or not args.tgt_lang.replace(" ", "").replace("-", "").isalpha():
        print(
            f"  {style('[!]', C.YELLOW)} Warning: '{args.tgt_lang}' might not be a "
            f"valid language name."
        )

    while True:
        ext_input = (
            input(
                f"  Subtitle Extension (e.g. en, ar) "
                f"[{cfg_memory.get('tgt_ext', 'en')}]: "
            )
            .strip()
            .lower()
            or cfg_memory.get("tgt_ext", "en")
        )
        valid, err = validate_tgt_ext(ext_input)
        if valid:
            args.tgt_ext = ext_input
            break
        print(f"  {style('[!]', C.YELLOW)} {err} Please try again.")
    Context.provenance["tgt_ext"] = "Interactive"

    # Step 4: Source language
    saved_src = cfg_memory.get("src_lang") or ""
    print(f"\n{C.BOLD}> Step 4: Source Language (What is spoken in the video){C.RESET}")
    print(
        f"  {C.DIM}Blank = auto-detect  |  ISO codes: ja  en  ko  zh  ar  es ...{C.RESET}"
    )
    src_in = input(f" Source Language [{saved_src or 'auto'}]: ").strip().lower()
    args.src_lang = src_in if src_in else (saved_src if saved_src else None)
    Context.provenance["src_lang"] = "Interactive"

    # Step 5a: Compute Device selection
    print(f"\n{C.BOLD}> Step 5a: Compute Device{C.RESET}")
    device_detected, _ = resolve_device_and_compute("auto")
    print(f"  Detected support: {style(device_detected.upper(), C.CYAN)}")
    print(f"    {style('[0]', C.CYAN)} Auto-detect device")
    print(f"    {style('[1]', C.CYAN)} Force CPU Mode")
    print(f"    {style('[2]', C.CYAN)} Force CUDA GPU Mode (Requires NVIDIA GPU)")

    saved_dev = cfg_memory.get("device", "auto")
    dev_default_num = "0" if saved_dev == "auto" else "1" if saved_dev == "cpu" else "2"
    dev_in = input(f"  Selection [{dev_default_num}]: ").strip() or dev_default_num
    
    dev_map = {"0": "auto", "1": "cpu", "2": "cuda"}
    args.device = dev_map.get(dev_in, saved_dev)
    Context.provenance["device"] = "Interactive"
    
    device_resolved, compute_resolved = resolve_device_and_compute(args.device)
    print(f"  -> Resolved Device: {style(device_resolved.upper() + ' (' + compute_resolved + ')', C.GREEN)}")

    # Step 5b: Whisper model (Re-evaluated based on selected device)
    vram_avail = get_available_vram_gb() if device_resolved == "cuda" else 0.0
    vram_label = f"{vram_avail:.1f} GB free" if device_resolved == "cuda" else "CPU mode"
    recommended = get_model_recommendation(vram_avail, device_resolved == "cpu")

    print(
        f"\n{C.BOLD}> Step 5b: Transcription Model "
        f"(Recommended: {MODEL_MAP[recommended].upper()}){C.RESET}"
    )
    print(f"  {C.DIM}Selected Device: {device_resolved.upper()}  ({vram_label}){C.RESET}")
    print(f"    {style('[0]', C.CYAN)} Tiny           ~1.0 GB  Fastest")
    print(f"    {style('[1]', C.CYAN)} Base           ~1.5 GB  Fast")
    print(f"    {style('[2]', C.CYAN)} Small          ~2.5 GB  Recommended")
    print(f"    {style('[3]', C.CYAN)} Medium         ~5.0 GB  Better accuracy")
    print(f"    {style('[4]', C.CYAN)} Large-v3 Turbo ~6.0 GB  Fast + accurate")
    print(f"    {style('[5]', C.CYAN)} Large-v3       ~10  GB  Best accuracy")

    _model_input = input(f"  Selection [{recommended}]: ").strip() or recommended
    args.model = MODEL_MAP.get(_model_input)
    if args.model is None:
        qprint(f"  {style('[!]', C.YELLOW)} Invalid selection '{_model_input}', defaulting to 'small'.")
        args.model = "small"
    Context.provenance["model"] = "Interactive"
    print(f"  {style('->', C.GREEN)} Selected Model: {args.model.upper()}")

    # Step 6: Translation Service Router Choice
    saved_translator = cfg_memory.get("translator", "gemini")
    print(f"\n{C.BOLD}> Step 6: Translation Service Provider{C.RESET}")
    print(f"    {style('[0]', C.CYAN)} Gemini           (Free/Flash options)")
    print(f"    {style('[1]', C.CYAN)} OpenAI           (GPT models / Custom compatibles)")
    print(f"    {style('[2]', C.CYAN)} Anthropic        (Claude-4 models)")
    print(f"    {style('[3]', C.CYAN)} DeepL            (Dedicated plain text translation)")
    print(f"    {style('[4]', C.CYAN)} Google Translate (Free / Unofficial v1 API)")
    print(f"    {style('[5]', C.CYAN)} Ollama           (Local self-hosted inference)")
    
    trans_in = input(f"  Selection [{saved_translator}]: ").strip().lower()
    trans_map = {"0": "gemini", "1": "openai", "2": "anthropic", "3": "deepl", "4": "google", "5": "ollama"}
    args.translator = trans_map.get(trans_in, trans_in if trans_in in trans_map.values() else saved_translator)
    Context.provenance["translator"] = "Interactive"

    model_defaults = {
        "gemini": "gemini-3.5-flash",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-haiku-4-5",
        "deepl": "deepl-translator",
        "google": "google-v1-free",
        "ollama": "qwen2.5:7b",
    }

    if args.translator == "google":
        args.translation_model = "google-v1-free"
        args.api_url = ""
        args.api_key = ""
        qprint(f"  {style('[~]', C.GREEN)} Google Translate (Free) selected. API key and model selection bypassed.")
    elif args.translator == "ollama":
        print(f"\n{C.BOLD}── Local Translation Models sub-wizard ─────────────────────{C.RESET}")
        print(f"    {style('[0]', C.CYAN)} Qwen2.5-7B (qwen2.5:7b)")
        print(f"    {style('[1]', C.CYAN)} Llama-3-8B (llama3:8b)")
        print(f"    {style('[2]', C.CYAN)} NLLB-200 (nllb)")
        print(f"    {style('[3]', C.CYAN)} Opus-MT / Helsinki-NLP (opus-mt)")
        print(f"    {style('[4]', C.CYAN)} Mistral-7B (mistral:7b)")
        print(f"    {style('[5]', C.CYAN)} DeepSeek-R1-8B (deepseek-r1:8b)")
        print(f"    {style('[6]', C.CYAN)} Gemma-2-9B (gemma2:9b)")
        print(f"    {style('[7]', C.CYAN)} Phi-3.5 (phi3.5)")
        print(f"    {style('[8]', C.CYAN)} Custom model tag")
        
        local_choice = input("  Selection [0]: ").strip()
        local_map = {
            "0": "qwen2.5:7b",
            "1": "llama3:8b",
            "2": "nllb",
            "3": "opus-mt",
            "4": "mistral:7b",
            "5": "deepseek-r1:8b",
            "6": "gemma2:9b",
            "7": "phi3.5"
        }
        if local_choice in local_map:
            args.translation_model = local_map[local_choice]
        elif local_choice == "8":
            args.translation_model = input("  Custom model tag: ").strip() or "qwen2.5:7b"
        else:
            args.translation_model = "qwen2.5:7b"
            
        saved_url = cfg_memory.get("api_url") or "http://localhost:11434/api/chat"
        args.api_url = input(f"  Local Ollama Endpoint URL [{saved_url}]: ").strip() or saved_url
        args.api_key = ""
        qprint(f"  {style('[~]', C.GREEN)} Ollama translator configured with model: {args.translation_model}")
    else:
        saved_t_model = cfg_memory.get("translation_model") or model_defaults.get(args.translator, "gemini-3.5-flash")
        args.translation_model = input(f"  Translation Model [{saved_t_model}]: ").strip() or saved_t_model
        Context.provenance["translation_model"] = "Interactive"

        saved_url = cfg_memory.get("api_url", "")
        args.api_url = input(f"  Custom API Gateway URL (Leave blank for default) [{saved_url}]: ").strip() or saved_url
        Context.provenance["api_url"] = "Interactive"

        saved_key = cfg_memory.get("api_key", "")
        if not saved_key:
            env_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "") or os.environ.get("DEEPL_API_KEY", "")
            if env_key:
                is_valid, _ = validate_api_key(env_key)
                if is_valid:
                    saved_key = env_key

        print(f"\n{C.BOLD}> Step 7: API Credentials Key{C.RESET}")
        if saved_key:
            print(f"  {C.DIM}Stored key context found. Press Enter to reuse.{C.RESET}")
        while True:
            key_input = input("  Key: ").strip() or saved_key
            if not key_input:
                args.api_key = ""
                break
            valid, err = validate_api_key(key_input)
            if valid:
                args.api_key = key_input
                break
            print(f"  {style('[!]', C.YELLOW)} {err} Please try again.")
            
    args.translate = bool(args.api_key) or args.translator in ("google", "ollama")
    Context.provenance["api_key"] = "Interactive"
    Context.provenance["translate"] = "Interactive"

    # Step 8: Output format
    print(f"\n{C.BOLD}> Step 8: Output Format{C.RESET}")
    print(f"    {style('[0]', C.CYAN)} Softsub -- toggleable track  (default)")
    print(f"    {style('[1]', C.CYAN)} Hardsub -- burned into video")
    args.hardsub = input("  Selection [0]: ").strip() == "1"
    Context.provenance["hardsub"] = "Interactive"

    # Step 9: Subtitle Validation
    print(f"\n{C.BOLD}> Step 9: Health Check Settings{C.RESET}")
    saved_min = cfg_memory.get("min_blocks", 3)
    try:
        val_blocks = input(f"  Minimum valid blocks [{saved_min}]: ").strip()
        args.min_blocks = int(val_blocks) if val_blocks else int(saved_min)
    except ValueError:
        args.min_blocks = 3
    Context.provenance["min_blocks"] = "Interactive"

    # Step 10: Advanced Tuning Options
    print(
        f"\n{C.BOLD}> Step 10: Tune Advanced SRT & Similarity Thresholds?{C.RESET} [y/N]"
    )
    tune = input("  Selection: ").strip().lower() == "y"
    if tune:
        try:
            args.srt_max_avg_duration = float(
                input(
                    f"    Max block duration seconds "
                    f"[{getattr(args, 'srt_max_avg_duration', cfg_memory.get('srt_max_avg_duration', 10.0))}]: "
                ).strip()
                or getattr(args, 'srt_max_avg_duration', cfg_memory.get('srt_max_avg_duration', 10.0))
            )
            args.srt_min_avg_duration = float(
                input(
                    f"    Min block duration seconds "
                    f"[{getattr(args, 'srt_min_avg_duration', cfg_memory.get('srt_min_avg_duration', 0.1))}]: "
                ).strip()
                or getattr(args, 'srt_min_avg_duration', cfg_memory.get('srt_min_avg_duration', 0.1))
            )
            args.srt_dup_ratio = float(
                input(
                    f"    Duplicate loop ratio threshold "
                    f"[{getattr(args, 'srt_dup_ratio', cfg_memory.get('srt_dup_ratio', 0.6))}]: "
                ).strip()
                or getattr(args, 'srt_dup_ratio', cfg_memory.get('srt_dup_ratio', 0.6))
            )
            args.fallback_match_threshold = float(
                input(
                    f"    Fuzzy match ratio (0.0 - 1.0) "
                    f"[{getattr(args, 'fallback_match_threshold', cfg_memory.get('fallback_match_threshold', 0.95))}]: "
                ).strip()
                or getattr(args, 'fallback_match_threshold', cfg_memory.get('fallback_match_threshold', 0.95))
            )
            for k in [
                "srt_max_avg_duration",
                "srt_min_avg_duration",
                "srt_dup_ratio",
                "fallback_match_threshold",
            ]:
                Context.provenance[k] = "Interactive"
        except ValueError:
            print(
                f"    {style('[!]', C.YELLOW)} Invalid numeric values. "
                f"Standard defaults retained."
            )
    else:
        args.srt_max_avg_duration = getattr(args, "srt_max_avg_duration", cfg_memory.get("srt_max_avg_duration", 10.0))
        args.srt_min_avg_duration = getattr(args, "srt_min_avg_duration", cfg_memory.get("srt_min_avg_duration", 0.1))
        args.srt_dup_ratio = getattr(args, "srt_dup_ratio", cfg_memory.get("srt_dup_ratio", 0.6))
        args.fallback_match_threshold = getattr(args, "fallback_match_threshold", cfg_memory.get("fallback_match_threshold", 0.95))

    # Step 11: Pipeline Steps
    print(f"\n{C.BOLD}> Step 11: Pipeline Steps{C.RESET}")
    if getattr(args, "skip_transcribe", False):
        args.transcribe = False
        print("  Transcribe? [Disabled via CLI]")
    else:
        args.transcribe = input("  Transcribe? [Y/n]: ").strip().lower() != "n"
    Context.provenance["transcribe"] = "Interactive"
    
    if args.translate:
        if getattr(args, "skip_translate", False):
            args.translate = False
            print("  Translate?  [Disabled via CLI]")
        else:
            args.translate = input("  Translate?  [Y/n]: ").strip().lower() != "n"
            Context.provenance["translate"] = "Interactive"
            
    if getattr(args, "skip_embed", False):
        args.embed = False
        print("  Mux?        [Disabled via CLI]")
    else:
        args.embed = input("  Mux?        [Y/n]: ").strip().lower() != "n"
    Context.provenance["embed"] = "Interactive"

    if args.hardsub and not args.embed:
        print(
            f"  {style('[!]', C.YELLOW)} Override: Hardsub is active. Soft muxing step "
            f"enabled to complete burning action."
        )
        args.embed = True
        Context.provenance["embed"] = "Auto-Override"

    # Step 12: Watch mode
    args.watch = (
        input(f"\n{C.BOLD}> Step 12: Watch Mode?{C.RESET} [y/N]: ").strip().lower()
        == "y"
    )

    resolve_pipeline_steps(args, respect_interactive=True)

    save_config({
        "schema_version": DEFAULT_CONFIG["schema_version"],
        "api_key": args.api_key,
        "tgt_lang": args.tgt_lang,
        "tgt_ext": args.tgt_ext,
        "src_lang": args.src_lang or "",
        "min_blocks": args.min_blocks,
        "model": args.model,
        "device": args.device,
        "no_cleanup": args.no_cleanup,
        "skip_migration": args.skip_migration,
        "explain_summary": args.explain_summary,
        "srt_max_avg_duration": args.srt_max_avg_duration,
        "srt_min_avg_duration": args.srt_min_avg_duration,
        "srt_dup_ratio": args.srt_dup_ratio,
        "fallback_match_threshold": args.fallback_match_threshold,
        "translator": args.translator,
        "translation_model": args.translation_model,
        "api_url": args.api_url,
        "max_audit_logs": cfg_memory.get("max_audit_logs", MAX_AUDIT_LOGS),
        "gemini_model": cfg_memory.get("gemini_model", DEFAULT_GEMINI_MODEL),
        "whisper_beam_size": cfg_memory.get("whisper_beam_size", WHISPER_BEAM_SIZE),
    })


# ════════════════════════════════════════════════════════════
#  HARDSUB HELPER
# ════════════════════════════════════════════════════════════
def run_ffmpeg_hardsub(
    media_path: Path,
    target_srt: Path,
    out_path: Path,
    cwd_path: Path,
) -> subprocess.CompletedProcess:
    """Execute FFmpeg hardsub command with safe path escaping."""
    base = media_path.stem
    unique_id = uuid.uuid4().hex
    temp_srt = f"temp_hardsub_{base}_{unique_id}.srt"
    temp_srt_path = cwd_path / temp_srt

    abs_media = str(media_path.resolve())
    abs_out = str(out_path.resolve())

    if abs_media.startswith("-"):
        abs_media = f"./{abs_media}"
    if abs_out.startswith("-"):
        abs_out = f"./{abs_out}"

    try:
        # Guarantee removal of temporary srt using temp_file_guard
        with temp_file_guard(temp_srt_path) as guarded_tmp:
            shutil.copy(target_srt, guarded_tmp)
            escaped_srt = escape_ffmpeg_filter_path(guarded_tmp)
            cmd = [
                Context.ffmpeg_cmd,
                "-y",
                "-v",
                "error",
                "-i",
                abs_media,
                "-vf",
                f"subtitles={escaped_srt}",
                "-c:a",
                "copy",
                abs_out,
            ]
            logger.debug("Executing local FFmpeg hardsub: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, cwd=cwd_path, encoding="utf-8", errors="replace", timeout=FFMPEG_TRANSCODE_TIMEOUT)
            return result
    except (OSError, PermissionError) as err1:
        try:
            alt_temp = Path(tempfile.gettempdir()) / temp_srt
            with temp_file_guard(alt_temp) as guarded_alt:
                shutil.copy(target_srt, guarded_alt)
                escaped_alt = escape_ffmpeg_filter_path(guarded_alt)
                cmd = [
                    Context.ffmpeg_cmd,
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    abs_media,
                    "-vf",
                    f"subtitles={escaped_alt}",
                    "-c:a",
                    "copy",
                    abs_out,
                ]
                logger.debug("Executing alt FFmpeg hardsub: %s", " ".join(cmd))
                result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=FFMPEG_TRANSCODE_TIMEOUT)
                return result
        except (OSError, PermissionError) as err2:
            qprint(
                f"  {style('[x]', C.RED)} Severe Hardsub Error: Temporary subtitle file "
                f"could not be written."
            )
            qprint(f"      Workspace Error: {err1}")
            qprint(f"      System Temp Error: {err2}")
            return subprocess.CompletedProcess(args=[], returncode=1)


# ════════════════════════════════════════════════════════════
#  MUX OUTPUT VALIDATION
# ════════════════════════════════════════════════════════════
def verify_mux_output(path: Union[str, Path], hardsub: bool = False) -> Tuple[bool, str]:
    """Verify the output of a muxing/hardsub operation."""
    p = Path(path)
    if not p.exists():
        return False, "Output container was not generated."
    file_size = p.stat().st_size
    if file_size < MUXED_MIN_BYTES:
        return (
            False,
            f"Output file size ({file_size} bytes) is below safe processing bounds.",
        )
    
    if Context.ffprobe_cmd:
        try:
            cmd = [
                Context.ffprobe_cmd,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(p),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30.0)
            logger.debug("Container structure integrity diagnostic complete: returncode %d", result.returncode)
            if result.returncode != 0:
                err_msg = result.stderr.strip() if result.stderr else "Corrupt stream layout"
                return False, f"FFprobe validation check failure: {err_msg}"
            
            duration_str = result.stdout.strip()
            if duration_str:
                try:
                    duration_val = float(duration_str)
                    if duration_val <= 0:
                        return False, "FFprobe reported non-positive media duration."
                except ValueError:
                    return False, f"Non-numeric duration output reported: '{duration_str}'"
            else:
                return False, "FFprobe returned empty duration metadata."
        except subprocess.TimeoutExpired:
            return False, "FFprobe validation check timed out."
        except Exception as e:
            logger.debug("Non-fatal verification error: %s", e)
            
    return True, "OK"


# ════════════════════════════════════════════════════════════
#  MODULAR PIPELINE COMPONENT METHODS
# ════════════════════════════════════════════════════════════
def _extract_audio(media_path: Path, temp_audio: Path) -> bool:
    """Extract audio track to monaural PCM format at 16kHz."""
    if media_path.name.startswith("-"):
        raise ValueError(f"Target media path starts with options flag parameter: {media_path}")

    abs_media = str(media_path.resolve())
    if abs_media.startswith("-"):
        abs_media = f"./{abs_media}"

    cmd = [
        Context.ffmpeg_cmd,
        "-y",
        "-v",
        "error",
        "-i",
        abs_media,
        "-vn",
        "-acodec",
        AUDIO_CODEC,
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-ac",
        "1",
        str(temp_audio),
    ]
    logger.debug("Executing local FFmpeg extraction: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=FFMPEG_STREAM_TIMEOUT
    )
    if result.returncode != 0 or not temp_audio.exists():
        stderr_msg = result.stderr.strip() if result.stderr else ""
        qprint(f"  {style('[x]', C.RED)} Audio extraction failed.")
        if stderr_msg:
            qprint(f"      {C.DIM}{stderr_msg[:STDERR_TRUNCATION_LIMIT]}{C.RESET}")
        return False
    return True


def _transcribe_audio(
    temp_audio: Path,
    srt_src: Path,
    duration: float,
    args: PipelineConfig,
    task_type: str = "transcribe",
) -> Tuple[bool, str]:
    """Run core transcribe functions against active audio samples."""
    transcription_success = False
    detected_lang = "und"
    beam_size = getattr(args, "whisper_beam_size", WHISPER_BEAM_SIZE)

    for attempt in range(MAX_TRANSCRIPTION_RETRIES):
        tmp_srt = srt_src.with_suffix(".tmp")
        try:
            qprint(
                f"  {style('>', C.CYAN)} Transcribing ({task_type})..."
                f"{f' (attempt {attempt + 1}/{MAX_TRANSCRIPTION_RETRIES})' if attempt > 0 else ''}"
            )
            
            lang_hint = args.src_lang if args.src_lang else None
            device, compute = resolve_device_and_compute(args.device)
            
            generator = TranscriptionManager.transcribe(
                audio_path=str(temp_audio),
                model_name=args.model,
                device=device,
                compute_type=compute,
                lang_hint=lang_hint,
                beam_size=beam_size,
                timeout=WHISPER_TRANSCRIBE_TIMEOUT,
                task_type=task_type,
            )

            # Ensure cleanup of tmp_srt in the event of partial transcription errors
            with temp_file_guard(tmp_srt) as guarded_tmp:
                idx = 1
                with open(guarded_tmp, "w", encoding="utf-8") as f:
                    for event, data in generator:
                        if event == "info":
                            detected_lang, prob = data
                        elif event == "segment":
                            start, end, text = data
                            cleaned_text = text.strip()
                            if cleaned_text:
                                f.write(
                                    f"{idx}\n"
                                    f"{fmt_srt_ts(start)} --> {fmt_srt_ts(end)}\n"
                                    f"{cleaned_text}\n\n"
                                )
                                idx += 1
                            if not Context.quiet:
                                if duration > 0:
                                    pct = min(end / duration * 100, 100.0)
                                    sys.stdout.write(
                                        f"\r    {C.DIM}{fmt_time(end)} / "
                                        f"{fmt_time(duration)} ({pct:.1f}%)"
                                        f"{C.RESET}   "
                                    )
                                else:
                                    sys.stdout.write(
                                        f"\r    {C.DIM}{fmt_time(end)}{C.RESET}   "
                                    )
                                sys.stdout.flush()

                os.replace(str(guarded_tmp), str(srt_src))

            qprint(
                f"\n  {style('[+]', C.GREEN)} Transcription done "
                f"(detected: {detected_lang})"
            )
            transcription_success = True
            break

        except Exception as e:
            qprint(f"\n  {style('[x]', C.RED)} Transcription error: {e}")
            if attempt < MAX_TRANSCRIPTION_RETRIES - 1:
                delay = TRANSCRIPTION_RETRY_BASE_DELAY * (2**attempt)
                qprint(f"  {style('[~]', C.YELLOW)} Retrying in {delay:.0f}s...")
                time.sleep(delay)
                perform_vram_gc()
            else:
                raise TranscriptionError(f"All {MAX_TRANSCRIPTION_RETRIES} transcription attempts failed. Context error: {e}")

    return transcription_success, detected_lang


def wait_for_file_settle(
    path: Union[str, Path],
    max_retries: int = FILE_SETTLE_MAX_RETRIES,
    delay: float = FILE_SETTLE_DELAY,
) -> bool:
    """Wait for a file to stabilize (stop changing size and mtime)."""
    p = Path(path)
    if not p.exists():
        return False

    last_size = -1
    last_mtime = -1.0
    for _ in range(max_retries):
        try:
            stat = p.stat()
            current_size = stat.st_size
            current_mtime = stat.st_mtime
            
            if current_size == last_size and current_mtime == last_mtime and current_size > 0:
                # Perform access test on Windows to respect file sharing violations
                if platform.system() == "Windows":
                    try:
                        with open(p, "rb+") as f:
                            pass
                    except IOError:
                        time.sleep(delay)
                        continue
                return True
            
            last_size = current_size
            last_mtime = current_mtime
        except (IOError, OSError):
            pass
        time.sleep(delay)
    return False


# ════════════════════════════════════════════════════════════
#  CORE PROCESS FILE PATHWAY
# ════════════════════════════════════════════════════════════
def process_file(
    media_path: Union[str, Path],
    args: PipelineConfig,
    file_index: int = 1,
    total_files: int = 1,
) -> Tuple[FileStatus, float]:
    """Process a single media file through the full pipeline with container validations."""
    media_p = Path(media_path)
    base = media_p.stem
    t_start = time.time()
    
    unique_id = uuid.uuid4().hex[:16]
    temp_audio = media_p.parent / f"temp_{base}_{unique_id}_audio.wav"

    status = FileStatus()

    qprint(f"\n{C.BOLD}── [{file_index}/{total_files}] {base}{C.RESET}")
    logger.debug("Processing file: %s", media_p)

    if not media_p.exists():
        qprint(f"  {style('[x]', C.RED)} File no longer exists -- skipping.")
        status.error = True
        return status, 0.0

    if not is_safe_relative(media_p, args.folder):
        qprint(
            f"  {style('[x]', C.RED)} Path traversal detected via symlink -- "
            f"file resolves outside target directory."
        )
        status.error = True
        return status, 0.0

    if str(media_path).startswith("-") or media_p.name.startswith("-"):
        qprint(f"  {style('[x]', C.RED)} Error: Media path starts with an options flag parameter.")
        status.error = True
        return status, 0.0

    srt_src, src_lang_code = find_source_srt(media_p, args.tgt_ext, args.src_lang)

    # Legacy File Conversion / Migration Paths
    legacy_old_srt = media_p.parent / f"{base}.auto.srt"
    legacy_subs_pipeline = media_p.parent / f"{base}.subs-pipeline.srt"
    standard_src_path = media_p.parent / f"{base}.{src_lang_code}.srt" if src_lang_code != "und" else legacy_subs_pipeline

    if legacy_old_srt.exists() and not standard_src_path.exists():
        if not args.skip_migration:
            try:
                legacy_old_srt.rename(standard_src_path)
                srt_src = standard_src_path
                qprint(f"  {style('[~]', C.DIM)} Converted legacy format: '{legacy_old_srt.name}' -> '{standard_src_path.name}'.")
            except OSError as e:
                logger.debug("Migration of %s failed: %s", legacy_old_srt, e)

    if legacy_subs_pipeline.exists() and src_lang_code != "und" and not standard_src_path.exists():
        if not args.skip_migration:
            try:
                legacy_subs_pipeline.rename(standard_src_path)
                srt_src = standard_src_path
                qprint(f"  {style('[~]', C.DIM)} Migrated generic source subtitle: '{legacy_subs_pipeline.name}' -> '{standard_src_path.name}'.")
            except OSError as e:
                logger.debug("Migration of %s failed: %s", legacy_subs_pipeline, e)

    srt_tgt = media_p.parent / f"{base}.{args.tgt_ext}.srt"
    out_ext = "mp4" if args.hardsub else "mkv"
    duration = get_duration(media_path)

    # Enforce quality validation of the target SRT if it exists on disk
    tgt_exists_and_healthy = False
    if srt_tgt.exists():
        ok, reason = is_valid_srt(srt_tgt, duration, args.min_blocks, args)
        if ok:
            tgt_exists_and_healthy = True
        else:
            qprint(f"  {style('[!]', C.YELLOW)} Existing target SRT '{srt_tgt.name}' is invalid ({reason}). Re-generating.")
            safe_remove(srt_tgt)

    existing_out_path = None
    if args.translate:
        target_dir = media_p.parent / f"muxed_{args.tgt_ext}"
        candidate = target_dir / f"{base}.{out_ext}"
        if candidate.exists():
            ok, _ = verify_mux_output(candidate, hardsub=args.hardsub)
            if ok:
                existing_out_path = candidate
            else:
                qprint(f"  {style('[!]', C.YELLOW)} Existing container output failed integrity checks. Cleaning up.")
                safe_remove(candidate)
    else:
        for parent_dir in media_p.parent.glob("muxed_*"):
            if parent_dir.is_dir():
                candidate = parent_dir / f"{base}.{out_ext}"
                if candidate.exists():
                    ok, _ = verify_mux_output(candidate, hardsub=args.hardsub)
                    if ok:
                        existing_out_path = candidate
                        break
                    else:
                        qprint(f"  {style('[!]', C.YELLOW)} Existing container output failed integrity checks. Cleaning up.")
                        safe_remove(candidate)

    if existing_out_path:
        out_path = existing_out_path
    else:
        out_path = media_p.parent / f"muxed_{args.tgt_ext}" / f"{base}.{out_ext}"

    if not check_disk_space(media_p.parent):
        qprint(
            f"  {style('[!]', C.YELLOW)} Insufficient disk space. "
            f"Attempting to continue..."
        )

    src_exists_and_healthy = False
    if srt_src.exists():
        ok, _ = is_valid_srt(srt_src, duration, args.min_blocks, args)
        if ok:
            src_exists_and_healthy = True

    # Check if target language is English to bypass API translator and utilize Whisper's native English translate task
    native_whisper_translate = False
    if args.translate and args.tgt_ext.strip().lower() == "en" and not tgt_exists_and_healthy:
        native_whisper_translate = True

    # Transcription is only skipped if we have a healthy source or target file
    need_transcribe = args.transcribe and not src_exists_and_healthy and not tgt_exists_and_healthy

    # ── DRY RUN SIMULATION PATHWAY ───────────────────────
    if args.dry_run:
        qprint(f"  {style('[DRY-RUN]', C.YELLOW)} Planning execution for file: {base}")

        if srt_src.exists():
            qprint(f"    - Existing source subtitle '{srt_src.name}' detected.")
            if src_exists_and_healthy:
                qprint(
                    f"      [Health Check] PASS: Reusing '{srt_src.name}' "
                    f"(transcription bypassed)."
                )
                status.reused_srt = True
            else:
                qprint(
                    f"      {style('[Health Check]', C.YELLOW)} FAIL: '{srt_src.name}' "
                    f"is invalid."
                )
                qprint(
                    f"      -> Simulated Action: Re-extract audio & transcribe "
                    f"(using model {args.model.upper()})."
                )
                status.transcribed = True
        else:
            qprint(f"    - No source subtitle exists.")
            qprint(
                f"    - Simulated Action: Extract audio and transcribe using "
                f"local {args.model.upper()} engine."
            )
            status.transcribed = True

        if srt_tgt.exists():
            qprint(f"    - Existing target subtitle '{srt_tgt.name}' detected.")
            fallback_match_threshold = getattr(args, "fallback_match_threshold", 0.95)
            if srt_src.exists():
                fallbacks, _ = detect_fallbacks(srt_src, srt_tgt, fallback_match_threshold, src_lang_code)
            else:
                fallbacks = 0
            if fallbacks > 0:
                qprint(f"      [Status Check] Target file contains {fallbacks} fallback block(s).")
                status.mixed_language = True
                status.fallback_count = fallbacks
            else:
                qprint(f"      [Status Check] Target file looks fully translated.")
                status.translated = True
        elif args.translate:
            if native_whisper_translate:
                qprint(
                    f"    - Simulated Action: Translate dialogue to English "
                    f"locally using Whisper native 'translate' pipeline task."
                )
            else:
                qprint(
                    f"    - Simulated Action: Translate dialogue to {args.tgt_lang} "
                    f"using {args.translator.upper()} API."
                )
            status.translated = True

        if args.embed:
            if out_path.exists():
                qprint(f"    - Output video already exists at '{out_path.name}'.")
                status.reused_all = True
            else:
                qprint(
                    f"    - Simulated Action: Mux subtitles into final '{out_ext}' "
                    f"container ({'Hardsub' if args.hardsub else 'Softsub'})."
                )
                status.muxed = True
        return status, 0.02

    if out_path.exists() and out_path.stat().st_size >= MUXED_MIN_BYTES:
        qprint(
            f"  {style('[+]', C.GREEN)} Output file already exists. "
            f"Skipping processing."
        )
        status.reused_all = True
        return status, 0.0

    if duration <= 0:
        qprint(
            f"  {style('[!]', C.YELLOW)} Could not determine duration -- "
            f"progress % unavailable."
        )

    try:
        # ── STEP 1: AUDIO EXTRACTION ──────────────────────
        audio_extracted_successfully = False
        if need_transcribe:
            qprint(f"  {style('[~]', C.DIM)} Ensuring Whisper model '{args.model}' is downloaded...")
            download_whisper_model_if_needed(args.model)
            
            qprint(f"  {style('>', C.CYAN)} Extracting audio...")
            try:
                # Scoped file guard guarantees cleanup of intermediate audio
                with temp_file_guard(temp_audio) as guarded_audio:
                    audio_extracted_successfully = _extract_audio(media_p, guarded_audio)
                    if audio_extracted_successfully:
                        # ── STEP 2: TRANSCRIPTION ─────────────────────────
                        target_output_srt = srt_tgt if native_whisper_translate else srt_src
                        task_type = "translate" if native_whisper_translate else "transcribe"
                        
                        transcription_success, detected_lang = _transcribe_audio(
                            guarded_audio, target_output_srt, duration, args, task_type=task_type
                        )
                        if not transcription_success:
                            status.error = True
                            return status, time.time() - t_start
                        
                        status.transcribed = True
                        if native_whisper_translate:
                            status.translated = True
                            # Preserve translated English output as source reference as well
                            try:
                                shutil.copy(srt_tgt, srt_src)
                            except Exception:
                                pass

                        if src_lang_code == "und" and detected_lang != "unknown" and not native_whisper_translate:
                            final_src_path = media_p.parent / f"{base}.{detected_lang}.srt"
                            if srt_src.exists() and srt_src != final_src_path:
                                try:
                                    os.replace(str(srt_src), str(final_src_path))
                                    srt_src = final_src_path
                                    src_lang_code = detected_lang
                                    qprint(f"  {style('[~]', C.DIM)} Standardized source subtitle to language code: '{srt_src.name}'")
                                except Exception as e:
                                    logger.debug("Failed to standardize dynamic source SRT name: %s", e)
                    else:
                        status.audio_failed = True
                        status.error = True
                        return status, time.time() - t_start
            except ValueError as val_err:
                qprint(f"  {style('[x]', C.RED)} Extraction blocked: {val_err}")
                status.audio_failed = True
                status.error = True
                return status, time.time() - t_start

        # ── SRT HEALTH CHECK ──────────────────────────────
        srt_src_healthy = True
        if srt_src.exists() and not srt_tgt.exists():
            ok, reason = is_valid_srt(srt_src, duration, args.min_blocks, args)
            if not ok:
                qprint(
                    f"  {style('[!]', C.YELLOW)} SRT health check failed: {reason}"
                )
                qprint(f"  {style('[!]', C.YELLOW)}     Translation skipped.")
                srt_src_healthy = False
                status.skipped = True

        if not srt_src_healthy and not tgt_exists_and_healthy:
            qprint(f"  {style('[x]', C.RED)} Error: No healthy subtitle file available, and transcription is disabled/unavailable.")
            status.error = True
            return status, time.time() - t_start

        # ── STEP 3: TRANSLATION ───────────────────────────
        if (
            args.translate
            and not native_whisper_translate
            and not Context.is_translation_disabled()
            and not srt_tgt.exists()
            and srt_src.exists()
            and srt_src_healthy
            and (args.api_key or args.translator in ("google", "ollama"))
        ):
            if (
                Context.get_consecutive_total_failures()
                >= CONSECUTIVE_TOTAL_FAIL_LIMIT
            ):
                Context.set_translation_disabled(True)
                qprint(
                    f"  {style('[!]', C.RED)} Translation suspended due to persistent "
                    f"communication failures."
                )
            else:
                qprint(f"  {style('>', C.CYAN)} Translating -> {args.tgt_lang}...")
                
                translator = getattr(args, "translator", "gemini")
                trans_model = getattr(args, "translation_model", None)
                api_url_val = getattr(args, "api_url", "")
                
                fallback_match_threshold = getattr(args, "fallback_match_threshold", 0.95)
                success, msg, fallbacks = translate_srt_native(
                    srt_src,
                    srt_tgt,
                    args.tgt_lang,
                    args.api_key,
                    translator=translator,
                    translation_model=trans_model,
                    api_url=api_url_val,
                    fallback_match_threshold=fallback_match_threshold,
                    tgt_ext=args.tgt_ext,
                    src_lang=src_lang_code,
                )
                if success:
                    # Enforce quality validation checks
                    quality_ok, quality_msg = verify_translation_quality(srt_src, srt_tgt, args.tgt_ext)
                    if not quality_ok:
                        qprint(f"  {style('[!]', C.YELLOW)} Translation Quality warning: {quality_msg}")
                        status.partial_success = True

                    if fallbacks > 0:
                        status.mixed_language = True
                        status.partial_success = True
                        status.fallback_count = fallbacks
                        qprint(
                            f"  {style('[~]', C.YELLOW)} Partial Success: {fallbacks} "
                            f"block(s) fell back to source language."
                        )
                    else:
                        status.translated = True
                        qprint(f"  {style('[+]', C.GREEN)} Translation done.")
                else:
                    qprint(
                        f"  {style('[x]', C.RED)} Translation skipped/failed: {msg}"
                    )

        # ── STEP 4: EMBED ─────────────────────────────────
        is_audio = media_p.suffix.lower() in AUDIO_EXTS
        target_srt = None
        fallback_match_threshold = getattr(args, "fallback_match_threshold", 0.95)

        if srt_tgt.exists():
            target_srt = srt_tgt
            if not status.translated and not status.mixed_language:
                if srt_src.exists():
                    fallback_count, _ = detect_fallbacks(srt_src, srt_tgt, fallback_match_threshold, src_lang_code)
                    if fallback_count > 0:
                        status.mixed_language = True
                        status.partial_success = True
                        status.fallback_count = fallback_count
                        qprint(
                            f"  {style('[~]', C.YELLOW)} Target file contains {fallback_count} "
                            f"unchanged fallback blocks."
                        )
                    else:
                        status.reused_srt = True
                else:
                    status.reused_srt = True

        elif srt_src.exists() and srt_src_healthy:
            target_srt = srt_src
            if args.translate:
                qprint(
                    f"  {style('[!]', C.YELLOW)} Using source SRT -- "
                    f"translated SRT unavailable."
                )

        actual_lang = args.tgt_ext if target_srt == srt_tgt else src_lang_code
        out_path = media_p.parent / f"muxed_{actual_lang}" / f"{base}.{out_ext}"

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            qprint(
                f"  {style('[x]', C.RED)} Failed to create output directory: {e}"
            )
            status.error = True
            return status, time.time() - t_start

        if out_path.exists():
            if out_path.stat().st_size < MUXED_MIN_BYTES:
                qprint(
                    f"  {style('[!]', C.YELLOW)} Removing incomplete output from "
                    f"previous run."
                )
                safe_remove(out_path)

        if args.embed and target_srt and not out_path.exists():
            if is_audio:
                qprint(
                    f"  {style('[~]', C.DIM)} Audio-only file -- muxing skipped."
                )
            else:
                qprint(f"  {style('>', C.CYAN)} Muxing...")

                try:
                    if args.hardsub:
                        cwd_path = media_p.parent
                        result = run_ffmpeg_hardsub(
                            media_p, target_srt, out_path, cwd_path
                        )
                    else:
                        abs_media = str(media_p.resolve())
                        abs_tgt_srt = str(target_srt.resolve())
                        abs_out = str(out_path.resolve())

                        cmd = [
                            Context.ffmpeg_cmd,
                            "-y",
                            "-v",
                            "error",
                            "-i",
                            abs_media,
                            "-i",
                            abs_tgt_srt,
                            "-c:v",
                            "copy",
                            "-c:a",
                            "copy",
                            "-c:s",
                            "srt",
                            "-metadata:s:s:0",
                            f"language={actual_lang}",
                            abs_out,
                        ]
                        logger.debug("Executing Softsub Mux: %s", " ".join(cmd))
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            encoding="utf-8",
                            errors="replace",
                            timeout=FFMPEG_STREAM_TIMEOUT
                        )

                    if result.returncode != 0:
                        err = (
                            result.stderr.strip()[:STDERR_TRUNCATION_LIMIT]
                            if result.stderr
                            else ""
                        )
                        if err:
                            qprint(f"  {C.DIM}    FFmpeg: {err}{C.RESET}")
                        raise MuxError(f"Muxing failed with exit code {result.returncode}. Error: {err}")

                    ok, reason = verify_mux_output(out_path, hardsub=args.hardsub)
                    if ok:
                        qprint(
                            f"  {style('[+]', C.GREEN)} -> "
                            f"{out_path.parent.name}\\{out_path.name}"
                        )
                        status.muxed = True
                    else:
                        raise MuxError(f"Mux validation failed: {reason}")

                except MuxError:
                    raise
                except Exception as e:
                    raise MuxError(f"Muxing process error: {e}")

    except TranscriptionError as te:
        qprint(f"  {style('[x]', C.RED)} Transcription error: {te}")
        status.error = True
    except MuxError as me:
        qprint(f"  {style('[x]', C.RED)} Muxing error: {me}")
        status.error = True
    except TranslationError as tre:
        qprint(f"  {style('[x]', C.RED)} Translation error: {tre}")
        status.error = True
    except KeyboardInterrupt:
        qprint(
            f"\n{style('[!]', C.YELLOW)} Interrupted during processing of {base}."
        )
        cleanup_all_temp_files()
        raise
    except Exception as e:
        qprint(
            f"  {style('[x]', C.RED)} Unexpected pipeline processing error: {e}"
        )
        status.error = True

    finally:
        perform_vram_gc()

    return status, time.time() - t_start


# ════════════════════════════════════════════════════════════
#  FILE ENUMERATION (with recursive support)
# ════════════════════════════════════════════════════════════
def enumerate_media_files(folder: Union[str, Path], recursive: bool = False) -> List[Path]:
    """Enumerate media files in the target folder, preventing internal output indexing."""
    folder_path = Path(folder).resolve()
    media_files: List[Path] = []

    if recursive:
        files = folder_path.rglob("*")
    else:
        files = folder_path.iterdir()

    for p in files:
        if p.is_file() and p.suffix.lower() in MEDIA_EXTS:
            try:
                real_p = p.resolve()
                if not is_safe_relative(real_p, folder_path):
                    logger.warning("Skipping file resolving outside target root: %s", p)
                    continue
                
                # Exclude any files residing inside generated output directories
                skip_file = False
                for part in real_p.relative_to(folder_path).parts[:-1]:
                    if part.startswith("muxed_"):
                        skip_file = True
                        break
                if skip_file:
                    continue
            except (OSError, RuntimeError, ValueError):
                continue
            media_files.append(p)

    media_files.sort(key=lambda x: natural_keys(x.name))
    return media_files


# ════════════════════════════════════════════════════════════
#  INTERNAL TEST RUNNER
# ════════════════════════════════════════════════════════════
def run_self_tests() -> bool:
    """Execute built-in test suite to verify critical pipeline modules."""
    print(f"\n{C.CYAN}── Running Internal Self-Test Suite ───────────────────────{C.RESET}")
    failed = 0
    
    if platform.system() == "Windows":
        test_path = "C:\\path's with spaces\\video:sub.srt"
        escaped = escape_ffmpeg_filter_path(test_path)
        expected = "'C:/path'\\\\''s with spaces/video\\:sub.srt'"
    else:
        test_path = "/tmp/path's with spaces/video:sub.srt"
        escaped = escape_ffmpeg_filter_path(test_path)
        expected = "'/tmp/path'\\\\''s with spaces/video\\:sub.srt'"

    if escaped == expected:
        print(f"  {C.GREEN}[PASS]{C.RESET} FFmpeg filter path escaping")
    else:
        print(f"  {C.RED}[FAIL]{C.RESET} FFmpeg filter path escaping (Got: {escaped}, Expected: {expected})")
        failed += 1
        
    limiter = TokenBucketRateLimiter(rate=100.0, capacity=2)
    if limiter.acquire(blocking=False) and limiter.acquire(blocking=False):
        if not limiter.acquire(blocking=False):
            print(f"  {C.GREEN}[PASS]{C.RESET} Token Bucket rate limiter capacity constraints")
        else:
            print(f"  {C.RED}[FAIL]{C.RESET} Token Bucket rate limiter (allowed more than capacity)")
            failed += 1
    else:
        print(f"  {C.RED}[FAIL]{C.RESET} Token Bucket rate limiter (failed initial acquisition)")
        failed += 1
        
    with tempfile.NamedTemporaryFile("w", suffix=".srt", delete=False, encoding="utf-8") as tmp:
        tmp.write("1\n00:00:01,000 --> 00:00:03,000\nHello world\n\n2\n00:00:04,000 --> 00:00:06,000\nTest srt\n\n3\n00:00:07,000 --> 00:00:09,000\nSelf check\n")
        tmp_path = tmp.name
    
    try:
        ok, reason = is_valid_srt(tmp_path, min_blocks=3)
        if ok:
            print(f"  {C.GREEN}[PASS]{C.RESET} SRT validation (valid case)")
        else:
            print(f"  {C.RED}[FAIL]{C.RESET} SRT validation valid case (Reason: {reason})")
            failed += 1
            
        with open(tmp_path, "w", encoding="utf-8") as tmp:
            tmp.write("1\n00:00:01,000 --> 00:00:03,000\nHello world\n\n3\n00:00:04,000 --> 00:00:06,000\nTest srt\n")
        ok, reason = is_valid_srt(tmp_path, min_blocks=2)
        if not ok and "Non-sequential" in reason:
            print(f"  {C.GREEN}[PASS]{C.RESET} SRT validation sequential numbering check")
        else:
            print(f"  {C.RED}[FAIL]{C.RESET} SRT validation sequential numbering (Passed unexpectedly or wrong reason: {reason})")
            failed += 1
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Test translation parser robustness against textual noise
    sample_response = "Introduction context.\nBlock #1: Translated Hello\nBlock #2:\nTranslated World\nBlock #3: This block has conversational noise."
    parsed_res = _parse_translation_response(sample_response)
    if parsed_res.get(1) == "Translated Hello" and parsed_res.get(2) == "Translated World":
        print(f"  {C.GREEN}[PASS]{C.RESET} Translation response block parsing")
    else:
        print(f"  {C.RED}[FAIL]{C.RESET} Translation response block parsing (Got: {parsed_res})")
        failed += 1

    # Test programmatic srt file prefix search without glob character class escaping issues
    with tempfile.TemporaryDirectory() as tmp_dir:
        dir_path = Path(tmp_dir)
        media_file = dir_path / "Movie [2023] [1080p].mp4"
        media_file.touch()
        srt_file = dir_path / "Movie [2023] [1080p].ja.srt"
        srt_file.touch()
        found_srt, srt_lang = find_source_srt(media_file, "en")
        if found_srt == srt_file and srt_lang == "ja":
            print(f"  {C.GREEN}[PASS]{C.RESET} Source SRT localization with path character classes")
        else:
            print(f"  {C.RED}[FAIL]{C.RESET} Source SRT localization with path character classes (Got: {found_srt}, lang: {srt_lang})")
            failed += 1

    # Test natural_keys mixed type comparisons
    try:
        test_list = ["file123", "123file", "file2"]
        test_list.sort(key=natural_keys)
        print(f"  {C.GREEN}[PASS]{C.RESET} Natural keys mixed sorting logic")
    except TypeError as e:
        print(f"  {C.RED}[FAIL]{C.RESET} Natural keys mixed sorting logic (Error: {e})")
        failed += 1

    # Test Quality Assurance Module with simulated excessive length deviation
    with tempfile.NamedTemporaryFile("w", suffix=".srt", delete=False, encoding="utf-8") as s_tmp, \
         tempfile.NamedTemporaryFile("w", suffix=".srt", delete=False, encoding="utf-8") as t_tmp:
        s_tmp.write("1\n00:00:01,000 --> 00:00:03,000\nHello, how are you today?\n")
        t_tmp.write("1\n00:00:01,000 --> 00:00:03,000\nThis is a massive hallucinated expansion of a very small source text which should fail the translation length check.\n")
        s_path = s_tmp.name
        t_path = t_tmp.name
        
    try:
        ok, reason = verify_translation_quality(s_path, t_path, "en")
        if not ok and "excessive translation length deviation" in reason:
            print(f"  {C.GREEN}[PASS]{C.RESET} Translation quality check (hallucination detection)")
        else:
            print(f"  {C.RED}[FAIL]{C.RESET} Translation quality check (Got: {ok}, Reason: {reason})")
            failed += 1
    finally:
        Path(s_path).unlink(missing_ok=True)
        Path(t_path).unlink(missing_ok=True)

    # Verify local Ollama model payload creation constructs formatted variables without issue
    url, payload, headers = _build_ollama_payload("qwen2.5:7b", "Block #1:\nHello\n", "", "Spanish")
    if b"qwen2.5:7b" in payload and b"Spanish" in payload:
        print(f"  {C.GREEN}[PASS]{C.RESET} Ollama local model payload builder")
    else:
        print(f"  {C.RED}[FAIL]{C.RESET} Ollama local model payload builder")
        failed += 1
        
    print(f"{C.CYAN}───────────────────────────────────────────────────────────{C.RESET}")
    if failed == 0:
        print(f"{C.GREEN}[+] All tests completed successfully!{C.RESET}\n")
        return True
    else:
        print(f"{C.RED}[x] Self-test suite failed with {failed} failures.{C.RESET}\n")
        return False


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════
def main() -> None:
    """Main entry point for the Subs Pipeline application."""
    parser = argparse.ArgumentParser(
        description="Media Transcription & Translation Pipeline (v{})".format(__version__),
        epilog=(
            "Examples:\n"
            "  %(prog)s                                   # Interactive wizard mode\n"
            "  %(prog)s --headless --folder ./videos      # Process all videos\n"
            "  %(prog)s --headless --folder ./videos --watch  # Watch mode\n"
            "  %(prog)s --headless --folder . --model small --device cuda\n"
            "  %(prog)s --version                         # Show version"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--test", action="store_true", help="Run the internal self-test suite")
    parser.add_argument("--headless", action="store_true", help="Run without interactive prompts")
    parser.add_argument("--folder", type=str, help="Target directory containing media files")
    parser.add_argument("--model", type=str, default=None, help="Whisper model to use")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default=None, help="Compute device for Whisper (auto, cpu, cuda)")
    parser.add_argument("--src-lang", type=str, default=None, dest="src_lang", help="Source language code")
    parser.add_argument("--tgt-lang", type=str, default=None, dest="tgt_lang", help="Target language name")
    parser.add_argument("--tgt-ext", type=str, default=None, dest="tgt_ext", help="Target subtitle extension (e.g., en, ar)")
    parser.add_argument("--api-key", type=str, default=None, dest="api_key", help="Gemini/OpenAI/Anthropic/DeepL API key")
    parser.add_argument("--hardsub", action="store_true", help="Burn subtitles into video (hardsub)")
    parser.add_argument("--min-blocks", type=int, default=None, dest="min_blocks", help="Minimum valid subtitle blocks")
    parser.add_argument("--skip-transcribe", action="store_true", dest="skip_transcribe", help="Skip transcription step")
    parser.add_argument("--skip-translate", action="store_true", dest="skip_translate", help="Skip translation step")
    parser.add_argument("--skip-embed", action="store_true", dest="skip_embed", help="Skip muxing step")
    parser.add_argument("--watch", action="store_true", help="Enable filesystem watch mode")
    
    parser.add_argument("--no-cleanup", action="store_const", const=True, default=None, dest="no_cleanup", help="Skip temp file cleanup")
    parser.add_argument("--cleanup", action="store_const", const=False, default=None, dest="no_cleanup", help="Perform temp file cleanup")
    parser.add_argument("--skip-migration", action="store_const", const=True, default=None, dest="skip_migration", help="Skip legacy file migration")
    parser.add_argument("--migration", action="store_const", const=False, default=None, dest="skip_migration", help="Perform legacy file migration")
    parser.add_argument("--explain-summary", action="store_const", const=True, default=None, dest="explain_summary", help="Show status code explanations")
    parser.add_argument("--no-explain-summary", action="store_const", const=False, default=None, dest="explain_summary", help="Do not show status code explanations")
    
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="Simulate without processing")
    parser.add_argument("--no-audit", action="store_true", dest="no_audit", help="Disable audit logging")
    parser.add_argument("--verbose-summary", action="store_true", dest="verbose_summary", help="Show detailed processing info")
    parser.add_argument("--recursive", action="store_true", dest="recursive", help="Include subdirectories")
    parser.add_argument("--quiet", action="store_true", dest="quiet", help="Suppress non-error output")
    parser.add_argument("--verbose", action="store_true", dest="verbose", help="Enable verbose/debug output")
    parser.add_argument("--gemini-model", type=str, default=None, dest="gemini_model", help="Gemini model name (Legacy option map)")
    
    parser.add_argument("--translator", type=str, default=None, choices=["gemini", "openai", "anthropic", "deepl", "google", "ollama"], dest="translator", help="Translator provider choice (gemini, openai, anthropic, deepl, google, ollama)")
    parser.add_argument("--translation-model", type=str, default=None, dest="translation_model", help="Specific translation model to invoke")
    parser.add_argument("--api-url", type=str, default=None, dest="api_url", help="Custom base API gateway endpoint override")

    parser.add_argument(
        "--srt-max-avg-duration",
        type=float,
        dest="srt_max_avg_duration",
        help="Maximum average block duration in seconds",
    )
    parser.add_argument(
        "--srt-min-avg-duration",
        type=float,
        dest="srt_min_avg_duration",
        help="Minimum average block duration in seconds",
    )
    parser.add_argument(
        "--srt-dup-ratio",
        type=float,
        dest="srt_dup_ratio",
        help="Duplicate line ratio threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--fallback-match-threshold",
        type=float,
        dest="fallback_match_threshold",
        help="Fallback fuzzy match threshold (0.0-1.0)",
    )
    args_parsed = parser.parse_args()

    if args_parsed.test:
        success = run_self_tests()
        sys.exit(0 if success else 1)

    enable_windows_ansi()

    global logger
    logger = setup_logging(quiet=args_parsed.quiet, verbose=args_parsed.verbose)
    Context.quiet = args_parsed.quiet

    Context.clear_mutable_states()
    Context.reset_all_counters()

    global global_cfg
    global_cfg = load_config()

    config_fields = PipelineConfig.__dataclass_fields__.keys()
    merged_dict = {}
    for key in config_fields:
        if hasattr(args_parsed, key) and getattr(args_parsed, key) is not None:
            merged_dict[key] = getattr(args_parsed, key)
            Context.provenance[key] = "CLI"
        elif key in global_cfg:
            merged_dict[key] = global_cfg[key]
            Context.provenance[key] = "Config File"
        else:
            merged_dict[key] = DEFAULT_CONFIG.get(key, PipelineConfig.__dataclass_fields__[key].default)
            Context.provenance[key] = "Default"

    args = PipelineConfig(**merged_dict)

    if args.gemini_model and getattr(args, "translation_model", None) == DEFAULT_CONFIG["translation_model"]:
        args.translation_model = args.gemini_model
        args.translator = "gemini"

    if not args.api_key:
        env_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "") or os.environ.get("DEEPL_API_KEY", "")
        if env_key:
            is_valid, _ = validate_api_key(env_key)
            if is_valid:
                args.api_key = env_key
            else:
                qprint(f"  {style('[!]', C.YELLOW)} Ignored invalid API_KEY environment variable.")
                args.api_key = ""

    args.no_cleanup = args.no_cleanup if args.no_cleanup is not None else global_cfg.get("no_cleanup", False)
    args.skip_migration = args.skip_migration if args.skip_migration is not None else global_cfg.get("skip_migration", False)
    args.explain_summary = args.explain_summary if args.explain_summary is not None else global_cfg.get("explain_summary", True)

    if args.model:
        args.model = args.model.lower().strip()
        if args.model in MODEL_MAP:
            args.model = MODEL_MAP[args.model]
            
        model_aliases = {
            "large-v3-turbo": "large-v3-turbo",
            "largev3turbo": "large-v3-turbo",
            "turbo": "large-v3-turbo",
            "large-v3": "large-v3",
            "largev3": "large-v3",
            "large": "large-v3",
        }
        if args.model in model_aliases:
            args.model = model_aliases[args.model]

    diag_status = verify_config_status()
    if diag_status is not True:
        Context.config_warning = str(diag_status)

    if args.src_lang and not args.src_lang.strip():
        args.src_lang = None

    resolve_pipeline_steps(args)

    check_dependencies(headless=args.headless)

    if not args.headless:
        interactive_wizard(args, global_cfg)
    else:
        if Context.config_warning:
            print(f"  [!] Startup error: {Context.config_warning}")
        if not setup_ffmpeg():
            print(
                f"{C.RED}[!] Headless Failure: FFmpeg is missing from "
                f"systemic paths.{C.RESET}"
            )
            sys.exit(1)
        if not args.folder or not Path(args.folder).is_dir():
            print(
                f"{C.RED}[!] Headless Failure: Target directory path is "
                f"invalid.{C.RESET}"
            )
            sys.exit(1)
        startup_garbage_collection(args.folder, skip_cleanup=args.no_cleanup)

    validate_args(args)
    if not args.quiet:
        print_effective_settings(args)

    try:
        media_files = enumerate_media_files(args.folder, recursive=args.recursive)
    except PermissionError:
        print(
            f"{C.RED}[!] Permission denied accessing target folder: "
            f"{args.folder}{C.RESET}"
        )
        sys.exit(1)

    summary: List[Tuple[str, FileStatus, float]] = []
    batch_start = time.time()

    if media_files:
        for i, file in enumerate(media_files, 1):
            try:
                status, elapsed = process_file(file, args, i, len(media_files))
                summary.append((Path(file).name, status, elapsed))
            except KeyboardInterrupt:
                qprint(
                    f"\n{style('[!]', C.YELLOW)} Interrupted during batch loop "
                    f"processing."
                )
                summary.append((Path(file).name, FileStatus(error=True), 0.0))
                break
            except Exception as e:
                qprint(
                    f"{C.RED}  [x] Processing fault encountered on "
                    f"{Path(file).name}: {e}{C.RESET}"
                )
                summary.append((Path(file).name, FileStatus(error=True), 0.0))

        total_time = time.time() - batch_start
        if not args.quiet:
            print_summary(summary, total_time, args)
        if not args.dry_run:
            write_audit_log(args, summary, total_time)
    else:
        qprint(
            f"{style('[!]', C.YELLOW)}\n  No compatible media files found inside target "
            f"location."
        )

    if args.watch:
        run_watcher(args)
    else:
        exit_app(0)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{style('[!]', C.YELLOW)} Force terminating...")
        exit_app(0)
    except Exception as e:
        print(f"\n{C.RED}[x] FATAL RUN ERROR: {e}{C.RESET}")
        import traceback
        traceback.print_exc()
        exit_app(1)
