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
import base64
import atexit
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
import getpass
import gzip
import ssl
import http.client
import urllib.request
import urllib.parse
import urllib.error
import difflib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Union, Tuple, List, Dict, Set, Generator, Final
from dataclasses import dataclass, asdict

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

# ── OS Keyring for API Key Storage ──────────────────────
_KEYRING_SERVICE: Final[str] = "subs_pipeline"
_KEYRING_PROBED: bool = False
_KEYRING_AVAILABLE: bool = False
_keyring_probe_lock: threading.Lock = threading.Lock()
try:
    import keyring as _keyring_mod
except ImportError:
    _keyring_mod = None  # type: ignore[assignment]


def _ensure_keyring_probed() -> bool:
    """Lazily probe the keyring backend on first use.

    The probe (``get_password``) is deferred from import time because on
    Linux with a D-Bus secret-service daemon it can block for 1-3 seconds.
    """
    global _KEYRING_PROBED, _KEYRING_AVAILABLE
    if _KEYRING_PROBED:
        return _KEYRING_AVAILABLE
    with _keyring_probe_lock:
        if _KEYRING_PROBED:
            return _KEYRING_AVAILABLE
        _KEYRING_PROBED = True
        if _keyring_mod is None:
            _KEYRING_AVAILABLE = False
            return False
        try:
            _keyring_mod.get_password(_KEYRING_SERVICE, "__probe__")
            _KEYRING_AVAILABLE = True
        except Exception:
            _KEYRING_AVAILABLE = False
        return _KEYRING_AVAILABLE

# ── Cached SSL Context for API Requests ─────────────────
_cached_ssl_context: Optional[ssl.SSLContext] = None
_ssl_context_lock = threading.Lock()

def _get_ssl_context() -> Optional[ssl.SSLContext]:
    """Return a cached SSL context, creating it once if certifi is available."""
    global _cached_ssl_context
    if _cached_ssl_context is None:
        with _ssl_context_lock:
            if _cached_ssl_context is None:
                if _CERTIFI_AVAILABLE and certifi:
                    try:
                        _cached_ssl_context = ssl.create_default_context(cafile=certifi.where())
                    except Exception as e:
                        logger.warning("Failed to create SSL context from certifi: %s", e)
                else:
                    logger.warning("certifi not available — SSL connections will use the system trust store. "
                                   "Install 'certifi' for up-to-date CA certificates: pip install certifi")
    return _cached_ssl_context


# ── Keyring Helpers for API Key Storage ─────────────────
def _keyring_store(api_key: str) -> bool:
    """Store the API key in the OS keyring. Returns True on success."""
    if not _ensure_keyring_probed():
        return False
    try:
        _keyring_mod.set_password(_KEYRING_SERVICE, "api_key", api_key)  # type: ignore[union-attr]
        return True
    except Exception as e:
        logger.warning("Keyring store failed: %s", e)
        return False


def _keyring_load() -> str:
    """Load the API key from the OS keyring. Returns empty string on failure."""
    if not _ensure_keyring_probed():
        return ""
    try:
        return _keyring_mod.get_password(_KEYRING_SERVICE, "api_key") or ""  # type: ignore[union-attr]
    except Exception as e:
        logger.debug("Keyring load failed: %s", e)
        return ""


def _keyring_delete() -> None:
    """Remove the API key from the OS keyring (best-effort)."""
    if not _ensure_keyring_probed():
        return
    try:
        _keyring_mod.delete_password(_KEYRING_SERVICE, "api_key")  # type: ignore[union-attr]
    except Exception as e:
        logger.debug("Keyring delete failed (non-fatal): %s", e)


# ── Module Level Declarations ───────────────────────────
__all__ = [
    "context",
    "transcription_manager",
    "PipelineConfig",
    "FileStatus",
    "OllamaConnectionError",
    "is_valid_srt",
    "translate_srt_native",
    "find_source_srt",
    "process_file",
    "verify_translation_quality",
    "enumerate_media_files",
    "validate_args",
    "resolve_pipeline_steps",
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
_NO_COLOR = os.environ.get("NO_COLOR") is not None
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
            except OSError:
                pass


# ════════════════════════════════════════════════════════════
#  MODULE METADATA
# ════════════════════════════════════════════════════════════
__version__ = "1.6"
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
    "{model}:generateContent"
)
API_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_\-./+=@!*$]{10,120}$")

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

# Default translation models per provider (consolidated)
DEFAULT_TRANSLATION_MODELS: Final[Dict[str, str]] = {
    "gemini": "gemini-3.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "deepl": "deepl-translator",
    "google": "google-v1-free",
    "ollama": "qwen2.5:7b",
}

# Timings & Chunking Constants
GEMINI_CHUNK_SIZE: int = 80
GEMINI_INTER_CHUNK_DELAY: int = 2  # Legacy alias — use PROVIDER_INTER_CHUNK_DELAYS

# Per-provider inter-chunk delays (seconds). Local providers need no rate limiting.
PROVIDER_INTER_CHUNK_DELAYS: Final[Dict[str, int]] = {
    "gemini": 2,
    "openai": 2,
    "anthropic": 2,
    "deepl": 1,
    "google": 1,
    "ollama": 0,
}

# Per-provider translation request timeouts (seconds)
PROVIDER_TIMEOUTS: Final[Dict[str, int]] = {
    "gemini": 60,
    "openai": 120,
    "anthropic": 120,
    "deepl": 60,
    "google": 30,
    "ollama": 600,
}

# Default hardsub font — Cairo Bold (open-source, full Arabic/CJK/Latin support)
# Cairo only ships as a variable font (weights 200-1000). We download it and
# use Bold=1 + Weight=700 in ASS force_style to select the bold weight.
DEFAULT_HARDSUB_FONT_NAME: Final[str] = "Cairo"
DEFAULT_HARDSUB_FONT_FILE: Final[str] = "Cairo.ttf"
CAIRO_BOLD_GITHUB_URL: Final[str] = (
    "https://github.com/Gue3bara/Cairo/raw/master/fonts/Cairo/variable/Cairo%5Bslnt,wght%5D.ttf"
)

# ISO 639-1 extension to human-readable language name (for translation prompts)
EXT_TO_LANG_NAME: Final[Dict[str, str]] = {
    "en": "English", "ar": "Arabic", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "hi": "Hindi",
    "th": "Thai", "vi": "Vietnamese", "tr": "Turkish", "pl": "Polish",
    "nl": "Dutch", "sv": "Swedish", "da": "Danish", "fi": "Finnish",
    "no": "Norwegian", "cs": "Czech", "el": "Greek", "he": "Hebrew",
    "uk": "Ukrainian", "ro": "Romanian", "hu": "Hungarian", "id": "Indonesian",
    "ms": "Malay", "tl": "Filipino", "bn": "Bengali", "ur": "Urdu",
    "fa": "Persian", "sw": "Swahili", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi", "si": "Sinhala", "my": "Myanmar", "ka": "Georgian",
    "am": "Amharic", "ne": "Nepali", "km": "Khmer", "lo": "Lao",
}

# Right-to-Left languages — require special subtitle rendering
RTL_LANGUAGES: Final[Set[str]] = {
    "ar", "he", "fa", "ur", "yi", "ps", "sd", "ug", "ku", "ckb",
}

# Config schema validation ranges: (min, max, fallback_default)
CONFIG_RANGE_RULES: Final[Dict[str, Tuple[Union[int, float], Union[int, float], Any]]] = {
    "min_blocks": (1, 1000, 3),
    "srt_max_avg_duration": (0.5, 300.0, 10.0),
    "srt_min_avg_duration": (0.01, 10.0, 0.1),
    "srt_dup_ratio": (0.01, 1.0, 0.6),
    "fallback_match_threshold": (0.1, 1.0, 0.95),
    "max_audit_logs": (1, 500, 30),
}

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

# Translation Quality Thresholds
TRANSLATION_DEVIATION_RATIO_LIMIT: Final[float] = 0.35
TRANSLATION_EMPTY_RATIO_LIMIT: Final[float] = 0.2
TRANSLATION_LENGTH_EXPANSION_FACTOR: Final[float] = 4.0
TRANSLATION_LENGTH_TRUNCATION_FACTOR: Final[float] = 0.25
TRANSLATION_MIN_SOURCE_LENGTH: Final[int] = 10

# Watcher thread pool
WATCHER_MAX_WORKERS: Final[int] = 4

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
    "gpu_accel": "auto",
    "preset": "fast",
    "hardsub_fontsize": 22,
    "hardsub_outline": 2,
    "hardsub_shadow": 1,
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

    # Multi-language support
    tgt_langs: Optional[str] = None
    
    # Final operational targets
    transcribe: bool = True
    translate: bool = True
    embed: bool = True
    hardsub: bool = False
    font_path: Optional[str] = None
    gpu_accel: str = "auto"  # auto, nvenc, amf, qsv, none
    preset: str = "fast"     # ultrafast, fast, medium, slow
    hardsub_fontsize: int = 22
    hardsub_outline: int = 2
    hardsub_shadow: int = 1


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

class ConfigError(PipelineError):
    """Exception raised when configuration files or directories are inaccessible."""
    pass

class DiskSpaceError(PipelineError):
    """Exception raised when disk space checks fail or return errors."""
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
    else:
        for handler in logger.handlers:
            handler.setLevel(logger.level)
    return logger


logger: logging.Logger = setup_logging()


# ════════════════════════════════════════════════════════════
#  THREAD-SAFE CONTEXT SINGLETON STRUCTURE
# ════════════════════════════════════════════════════════════
class Context:
    """Thread-safe application context singleton.

    All state lives on the single ``context`` instance created at module
    level.  Access via ``context.XXX`` — never instantiate this class.
    """

    _instance: "Optional[Context]" = None

    def __new__(cls) -> "Context":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        # Settings (set once at startup)
        self.quiet: bool = False
        self.ffmpeg_cmd: Optional[str] = None
        self.ffprobe_cmd: Optional[str] = None
        self.config_warning: str = ""
        self.migration_status: str = "none"
        self._settings_lock: threading.Lock = threading.Lock()

        # Mutable state (thread-safe access via methods)
        self.provenance: Dict[str, str] = {}
        self._translation_disabled: bool = False
        self._consecutive_429s: int = 0
        self._consecutive_total_failures: int = 0
        self._state_lock: threading.Lock = threading.Lock()

        # Temp-file tracking
        self.active_temp_files: Set[Path] = set()
        self.temp_lock: threading.Lock = threading.Lock()
        self.failed_cleanups: List[str] = []

    def is_translation_disabled(self) -> bool:
        with self._state_lock:
            return self._translation_disabled

    def set_translation_disabled(self, val: bool) -> None:
        with self._state_lock:
            self._translation_disabled = val

    def add_failed_cleanup(self, msg: str) -> None:
        with self._state_lock:
            self.failed_cleanups.append(msg)

    def get_consecutive_429s(self) -> int:
        with self._state_lock:
            return self._consecutive_429s

    def increment_consecutive_429s(self) -> None:
        with self._state_lock:
            self._consecutive_429s += 1

    def reset_consecutive_429s(self) -> None:
        with self._state_lock:
            self._consecutive_429s = 0

    def get_consecutive_total_failures(self) -> int:
        with self._state_lock:
            return self._consecutive_total_failures

    def increment_consecutive_total_failures(self) -> None:
        with self._state_lock:
            self._consecutive_total_failures += 1

    def reset_consecutive_total_failures(self) -> None:
        with self._state_lock:
            self._consecutive_total_failures = 0

    def reset_all_counters(self) -> None:
        with self._state_lock:
            self._consecutive_429s = 0
            self._consecutive_total_failures = 0
            self._translation_disabled = False

    def clear_mutable_states(self) -> None:
        with self.temp_lock:
            self.active_temp_files.clear()
        with self._state_lock:
            self.failed_cleanups.clear()
            self.provenance.clear()

    def reset(self) -> None:
        global _gpu_encoder_cache
        _gpu_encoder_cache = None
        with self._settings_lock:
            self.quiet = False
            self.ffmpeg_cmd = None
            self.ffprobe_cmd = None
            self.config_warning = ""
            self.migration_status = "none"
        with self._state_lock:
            self._translation_disabled = False
            self._consecutive_429s = 0
            self._consecutive_total_failures = 0
            self.failed_cleanups.clear()
            self.provenance.clear()
        with self.temp_lock:
            self.active_temp_files.clear()


# Module-level singleton — the only way to use Context.
context = Context()


def qprint(*args, **kwargs) -> None:
    """Stdout writer that respects the global quiet setting and logs messages."""
    msg = " ".join(str(a) for a in args)
    if not context.quiet:
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
        context.ffmpeg_cmd = ffmpeg
        context.ffprobe_cmd = ffprobe
        logger.debug("Located FFmpeg: %s, FFprobe: %s", ffmpeg, ffprobe)
        return True
    return False


# GPU encoder detection cache
_gpu_encoder_cache: Optional[str] = None
_gpu_encoder_lock: threading.Lock = threading.Lock()


def detect_gpu_encoder() -> str:
    """Detect available hardware video encoder. Returns 'nvenc', 'amf', 'qsv', or 'none'."""
    global _gpu_encoder_cache
    if _gpu_encoder_cache is not None:
        return _gpu_encoder_cache

    with _gpu_encoder_lock:
        # Double-check after acquiring lock
        if _gpu_encoder_cache is not None:
            return _gpu_encoder_cache

        if not context.ffmpeg_cmd:
            _gpu_encoder_cache = "none"
            return "none"

        try:
            result = subprocess.run(
                [context.ffmpeg_cmd, "-encoders"],
                capture_output=True, encoding="utf-8", errors="replace", timeout=10,
            )
            encoders = result.stdout.lower()
            if "h264_nvenc" in encoders:
                _gpu_encoder_cache = "nvenc"
                return "nvenc"
            elif "h264_amf" in encoders:
                _gpu_encoder_cache = "amf"
                return "amf"
            elif "h264_qsv" in encoders:
                _gpu_encoder_cache = "qsv"
                return "qsv"
        except Exception as e:
            logger.debug("GPU encoder detection failed: %s", e)

        _gpu_encoder_cache = "none"
        return "none"


def get_encoder_args(user_choice: str, preset: str) -> List[str]:
    """Build FFmpeg encoder flags based on user preference and detected hardware.
    Returns a list of extra args to insert before the output filename."""
    if user_choice == "none":
        return ["-preset", preset]

    if user_choice == "auto":
        detected = detect_gpu_encoder()
    else:
        # Validate user-specified encoder is actually available
        actual = detect_gpu_encoder()
        if actual != user_choice:
            logger.warning("Requested encoder '%s' not available (detected: '%s'), falling back", user_choice, actual)
            detected = actual
        else:
            detected = user_choice

    encoder_map = {
        "nvenc": "h264_nvenc",
        "amf":  "h264_amf",
        "qsv":  "h264_qsv",
    }
    # Per-encoder preset mapping (AMF uses different names)
    amf_preset_map = {"ultrafast": "speed", "fast": "speed", "medium": "balanced", "slow": "quality"}

    if detected in encoder_map:
        enc_preset = amf_preset_map.get(preset, preset) if detected == "amf" else preset
        return ["-c:v", encoder_map[detected], "-preset", enc_preset]
    # Fallback: software encode
    return ["-preset", preset]


def select_preset_for_duration(duration: float, user_preset: str = "fast") -> str:
    """Auto-select encoding preset based on video duration.

    Short clips (<30 min) -> ultrafast (fast encode, acceptable quality)
    Medium (30min-2hr)    -> fast (default, good balance)
    Long (>2hr)          -> medium (better compression for large files)

    The user's explicit preset is used as an upper bound — auto-selection
    never picks a *faster* preset than the user requested, but may pick a
    *slower* one for better compression on long videos.
    """
    _PRESET_ORDER = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "veryslow"]
    user_idx = _PRESET_ORDER.index(user_preset) if user_preset in _PRESET_ORDER else _PRESET_ORDER.index("fast")

    if duration <= 0:
        return user_preset
    if duration < 1800:
        auto = "ultrafast"
    elif duration < 7200:
        auto = "fast"
    else:
        auto = "medium"

    auto_idx = _PRESET_ORDER.index(auto)
    # Never go faster than what the user asked for
    return _PRESET_ORDER[max(auto_idx, user_idx)]


def get_available_vram_gb() -> float:
    """Query available GPU VRAM in gigabytes."""
    if _TORCH_AVAILABLE and _torch.cuda.is_available():
        try:
            t_vram = _torch.cuda.get_device_properties(0).total_memory
            a_vram = t_vram - _torch.cuda.memory_allocated(0)
            return a_vram / (1024**3)
        except Exception as e:
            logger.warning("Failed to query VRAM: %s", e)
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
    worker_logger = logging.getLogger("subs_pipeline.worker")
    try:
        # Avoid thread contention in deep CPU loops
        os.environ["OMP_NUM_THREADS"] = "4"
        from faster_whisper import WhisperModel
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as load_err:
            # If load fails due to lack of connection or metadata sync offline, retry forcing local-only configurations
            worker_logger.debug("Initial WhisperModel load failed, retrying with offline flags: %s", load_err)
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            model = WhisperModel(model_name, device=device, compute_type=compute_type, local_files_only=True)
            
        res_queue.put(("init_ok", None))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            res_queue.put(("init_error", f"{e}\n{tb}"))
        except Exception as queue_err:
            logger.error("Failed to send init_error to queue (original: %s): %s", e, queue_err)
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
            except Exception as loop_err:
                try:
                    res_queue.put(("error", f"Fault encountered during active audio segmentation: {loop_err}"))
                except Exception as queue_err:
                    logger.error("Failed to send segment error to queue (original: %s): %s", loop_err, queue_err)
        except Exception as e:
            try:
                res_queue.put(("error", str(e)))
            except Exception as queue_err:
                logger.error("Failed to send error to queue (original: %s): %s", e, queue_err)


class TranscriptionManager:
    """Singleton managing a persistent transcription worker process.

    Usage::

        generator = transcription_manager.transcribe(...)
        transcription_manager.terminate()

    Never instantiate this class — use the module-level ``transcription_manager``.
    """

    _instance: "Optional[TranscriptionManager]" = None

    def __new__(cls) -> "TranscriptionManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._process: Optional[multiprocessing.Process] = None
        self._req_queue: Optional[multiprocessing.Queue] = None
        self._res_queue: Optional[multiprocessing.Queue] = None
        self._lock: threading.RLock = threading.RLock()
        self._transcribe_mutex: threading.Lock = threading.Lock()
        self._is_running: bool = False
        self._current_model: Optional[Tuple[str, str, str]] = None

    def _start_process(self, model_name: str, device: str, compute_type: str) -> None:
        ctx = multiprocessing.get_context("spawn")
        self._req_queue = ctx.Queue()
        self._res_queue = ctx.Queue()
        self._process = ctx.Process(
            target=_transcribe_worker_loop,
            args=(self._req_queue, self._res_queue, model_name, device, compute_type),
            daemon=True
        )
        self._process.start()

    def _drain_queue(self, q: Optional[multiprocessing.Queue]) -> None:
        if q is None:
            return
        while True:
            try:
                q.get_nowait()
            except Exception:
                break

    def transcribe(
        self,
        audio_path: str,
        model_name: str,
        device: str,
        compute_type: str,
        lang_hint: Optional[str],
        beam_size: int,
        timeout: float,
        task_type: str = "transcribe",
    ) -> Generator[Tuple[str, Any], None, None]:
        with self._transcribe_mutex:
            with self._lock:
                if self._is_running:
                    raise TranscriptionError("Another transcription is currently in progress.")
                self._is_running = True

                target_model = (model_name, device, compute_type)
                if (
                    self._process is None
                    or not self._process.is_alive()
                    or self._current_model != target_model
                ):
                    if self._process and self._process.is_alive():
                        self._terminate_under_lock()

                    logger.debug("Spawning child transcription worker using model: %s", model_name)
                    self._start_process(model_name, device, compute_type)
                    self._current_model = target_model

                    init_timeout = False
                    msg_type, payload = None, None
                    try:
                        msg_type, payload = self._res_queue.get(timeout=WORKER_INIT_TIMEOUT)
                    except queue.Empty:
                        init_timeout = True

                    if init_timeout:
                        self._terminate_under_lock()
                        self._is_running = False
                        raise TranscriptionError("Failed to communicate with transcription worker (initialization timeout).")
                    elif msg_type == "init_error":
                        self._terminate_under_lock()
                        self._is_running = False
                        raise TranscriptionError(f"Transcription worker initialization failed: {payload}")
                    elif msg_type != "init_ok":
                        self._terminate_under_lock()
                        self._is_running = False
                        raise TranscriptionError(f"Unexpected response from transcription worker initialization: {msg_type}")

                self._drain_queue(self._res_queue)
                self._req_queue.put((audio_path, lang_hint, beam_size, task_type))

            deadline = time.monotonic() + timeout
            try:
                while True:
                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        self.terminate()
                        raise TranscriptionError(f"Transcription timed out after {timeout} seconds")

                    try:
                        msg_type, data = self._res_queue.get(timeout=min(rem, 1.0))
                        if msg_type == "info":
                            yield ("info", data)
                        elif msg_type == "segment":
                            yield ("segment", data)
                        elif msg_type == "done":
                            break
                        elif msg_type == "error":
                            raise TranscriptionError(data)
                    except queue.Empty:
                        with self._lock:
                            if self._process is not None and not self._process.is_alive():
                                raise TranscriptionError("Transcription worker process terminated unexpectedly")
                        continue
            finally:
                with self._lock:
                    self._is_running = False

    def terminate(self) -> None:
        with self._lock:
            self._terminate_under_lock()

    def _terminate_under_lock(self) -> None:
        self._is_running = False
        if self._process:
            logger.debug("Terminating transcription child process")
            try:
                if self._req_queue:
                    try:
                        self._req_queue.put_nowait(None)
                    except Exception as e:
                        logger.debug("Queue sentinel error (non-fatal): %s", e)
                self._process.join(timeout=2.0)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=2.0)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=1.0)
            except Exception as e:
                logger.debug("Non-fatal termination error: %s", e)

            for q in (self._req_queue, self._res_queue):
                if q:
                    try:
                        q.cancel_join_thread()
                        q.close()
                    except Exception as e:
                        logger.debug("Queue cleanup error (non-fatal): %s", e)

            self._process = None
            self._req_queue = None
            self._res_queue = None
            self._current_model = None


# Module-level singleton.
transcription_manager = TranscriptionManager()


# ════════════════════════════════════════════════════════════
#  TEMP FILE MANAGER & SCOPED TEMP GUARDS
# ════════════════════════════════════════════════════════════
def register_temp_file(path: Union[str, Path]) -> None:
    """Register a temporary file for later cleanup."""
    with context.temp_lock:
        context.active_temp_files.add(Path(path).resolve())


def unregister_temp_file(path: Union[str, Path]) -> None:
    """Unregister a temporary file from cleanup tracking."""
    with context.temp_lock:
        p = Path(path).resolve()
        context.active_temp_files.discard(p)


def safe_remove(path: Union[str, Path]) -> None:
    """Safely remove a file, unregistering from temp tracking."""
    p = Path(path)
    try:
        if p.exists():
            p.unlink(missing_ok=True)
    except OSError as e:
        context.add_failed_cleanup(f"{path} ({e.strerror or str(e)})")
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
    """Remove all registered temporary files with race-condition safety.

    The set is snapshot and cleared under the lock, but the actual disk I/O
    (unlink) happens outside the lock to avoid blocking other threads during
    slow or failing file removals.
    """
    with context.temp_lock:
        pending = list(context.active_temp_files)
        context.active_temp_files.clear()

    for p in pending:
        try:
            p.unlink(missing_ok=True)
        except OSError as e:
            context.add_failed_cleanup(f"{p} ({e.strerror or str(e)})")


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
    """Verify disk space margin is sufficient.

    Returns True if enough space, False if low (with a warning printed).
    Raises DiskSpaceError if the check itself fails (e.g. permission denied).
    """
    p = Path(path)
    target = p if p.is_dir() else p.parent
    try:
        free = shutil.disk_usage(target).free
    except Exception as e:
        raise DiskSpaceError(f"Disk space check failed: {e}") from e
    if free < required_bytes:
        qprint(
            f"{style('[!]', C.YELLOW)} Low disk space: {free // (1024**2)} MB available, "
            f"{required_bytes // (1024**2)} MB recommended."
        )
        return False
    return True


# ════════════════════════════════════════════════════════════
#  SAFE EXIT
# ════════════════════════════════════════════════════════════
# Global flag for headless mode (set in main)
_headless_mode: bool = False


def exit_app(code: int = 0) -> None:
    """Perform clean shutdown with temp file cleanup and worker termination."""
    _http_pool.close_all()
    cleanup_all_temp_files()
    transcription_manager.terminate()
    if context.failed_cleanups:
        qprint(
            f"\n{style('[~]', C.DIM)} System cleanup complete. Some active lock files were "
            f"bypassed:"
        )
        for item in set(context.failed_cleanups):
            qprint(f"      · {item}")
    if not _headless_mode:
        try:
            input(f"\n{C.DIM}Press Enter to exit...{C.RESET}")
        except Exception:
            pass
    sys.exit(code)


# ════════════════════════════════════════════════════════════
#  CONFIG & DIAGNOSTICS
# ════════════════════════════════════════════════════════════
def verify_config_status() -> None:
    """Verify configuration directory and file are accessible.

    Raises ConfigError with a descriptive message on any failure.
    Returns None on success.
    """
    parent_dir = CONFIG_PATH.parent
    if not parent_dir.exists():
        raise ConfigError("Configuration directory does not exist.")
    if not os.access(parent_dir, os.W_OK):
        raise ConfigError("Configuration directory is not writeable.")
    if CONFIG_PATH.exists():
        if not os.access(CONFIG_PATH, os.R_OK):
            raise ConfigError("Configuration file is not readable.")
        if not os.access(CONFIG_PATH, os.W_OK):
            raise ConfigError("Configuration file is not writeable.")
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigError(f"Configuration file has invalid JSON formatting: {e}") from e
        except Exception as e:
            raise ConfigError(f"Configuration file access failure: {e}") from e


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
                    elif isinstance(default_val, str):
                        cfg_dict[key] = str(cfg_dict[key])
                except Exception as e:
                    logger.warning("Type conversion fail: key %s, error %s", key, e)
                    cfg_dict[key] = default_val

            if key in CONFIG_RANGE_RULES:
                min_val, max_val, fallback = CONFIG_RANGE_RULES[key]
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
    """Load configuration.

    API key priority:
      1. OS keyring (encrypted at rest by the OS)
      2. Legacy Base64-obfuscated value in the config file (migrated to keyring on load)
      3. Empty string (no key configured)
    """
    logger.debug("Loading configuration parameters from: %s", CONFIG_PATH)
    if CONFIG_PATH.exists():
        context.migration_status = "loaded"

    cfg_dict = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                
                # Map legacy skip_cleanup key down to no_cleanup parameters
                if "skip_cleanup" in loaded:
                    loaded["no_cleanup"] = loaded.pop("skip_cleanup")
                
                # ── API key resolution ────────────────────────────
                file_key = loaded.get("api_key", "")

                # 1) Try OS keyring first
                keyring_key = _keyring_load()

                if keyring_key:
                    # Keyring has the authoritative key — use it and
                    # silently drop any stale copy in the config file.
                    loaded["api_key"] = keyring_key
                elif file_key.startswith("obf:"):
                    # 2) Legacy Base64 in config — decode and migrate
                    try:
                        decoded = base64.b64decode(file_key[4:].encode("utf-8")).decode("utf-8")
                    except Exception as e:
                        logger.warning("Failed to decode legacy API key: %s", e)
                        decoded = ""
                    loaded["api_key"] = decoded
                    if decoded:
                        if _keyring_store(decoded):
                            logger.info("Migrated API key from config file to OS keyring.")
                        else:
                            logger.warning("Keyring unavailable; API key remains in config file (Base64).")
                else:
                    # 3) No key anywhere
                    loaded["api_key"] = file_key or ""

                cfg_dict.update(loaded)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Failed to parse config. Restoring defaults: %s", e)
        except Exception as e:
            logger.debug("Config load exception: %s", e)
    return validate_schema(cfg_dict)


def save_config(conf: Dict[str, Any]) -> None:
    """Save configuration atomically.

    The API key is stored in the OS keyring when available.  When the
    keyring is unavailable the key is written to the config file with
    Base64 obfuscation (cosmetic only — NOT secure on shared machines).
    """
    with _config_save_lock:
        parent_dir = CONFIG_PATH.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=parent_dir, suffix=".tmp", encoding="utf-8", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                
                conf_copy = dict(conf)
                raw_key = conf_copy.get("api_key", "")

                if raw_key:
                    if _keyring_store(raw_key):
                        # Key secured in OS keyring — omit from config file
                        conf_copy["api_key"] = ""
                    else:
                        # Keyring unavailable — fall back to Base64 in config
                        try:
                            conf_copy["api_key"] = "obf:" + base64.b64encode(raw_key.encode("utf-8")).decode("utf-8")
                        except Exception as e:
                            logger.warning("API key obfuscation failed; key stored in plaintext: %s", e)
                
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
                except OSError as e:
                    logger.debug("Failed to remove config temp file %s: %s", tmp_path, e)


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
    """Escape filenames for use inside FFmpeg filter syntax specifications.

    Wraps the result in single quotes so that colons, commas, brackets and
    other special characters are treated as literals.  Only the single-quote
    character itself needs escaping (``'\\'''`` pattern).
    """
    p_str = str(Path(path).resolve()).replace("\\", "/")
    p_str = p_str.replace("'", "'\\''")
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
    if not context.ffprobe_cmd:
        return 0.0
    try:
        cmd = [
            context.ffprobe_cmd,
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
    except subprocess.TimeoutExpired:
        logger.warning("ffprobe timed out for %s", media_path)
        return 0.0
    except Exception as e:
        logger.warning("Could not determine duration of %s: %s", media_path, e)
        return 0.0


def get_video_resolution(media_path: Union[str, Path]) -> Tuple[int, int]:
    """Get the width and height of the first video stream using ffprobe."""
    if not context.ffprobe_cmd:
        return 0, 0
    try:
        cmd = [
            context.ffprobe_cmd,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            str(media_path),
        ]
        result = subprocess.check_output(
            cmd, encoding="utf-8", errors="replace",
            stderr=subprocess.DEVNULL, timeout=30.0,
        ).strip()
        if "," in result:
            parts = result.split(",")
            return int(parts[0].strip()), int(parts[1].strip())
    except (subprocess.TimeoutExpired, ValueError, Exception) as e:
        logger.warning("Could not determine resolution of %s: %s", media_path, e)
    return 0, 0


def _parse_ffmpeg_time(time_str: str) -> float:
    """Parse FFmpeg time string (HH:MM:SS.ms) to seconds."""
    try:
        parts = time_str.strip().split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
    except (ValueError, AttributeError):
        pass
    return 0.0


def _format_eta(seconds: float) -> str:
    """Format seconds into human-readable ETA string."""
    if seconds < 0 or seconds > 86400:
        return "??:??"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def perform_vram_gc() -> None:
    """Free GPU VRAM by clearing CUDA cache and running garbage collection."""
    if _TORCH_AVAILABLE:
        try:
            if _torch.cuda.is_available():
                gc.collect()
                _torch.cuda.empty_cache()
        except Exception as e:
            logger.warning("CUDA empty cache failed: %s", e)


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
        content = Path(path).read_text(encoding="utf-8-sig", errors="ignore")
        blocks = [b.strip() for b in re.split(r"\n\n+", content.strip()) if b.strip()]
        dialogues: List[str] = []
        for b in blocks:
            lines = b.splitlines()
            if len(lines) >= 3:
                dialogues.append("\n".join(lines[2:]).strip())
            else:
                dialogues.append("")
        return dialogues
    except Exception as e:
        logger.warning("Failed to parse SRT dialogue from %s: %s", path, e)
        return []


def parse_srt_to_dict(path: Union[str, Path]) -> Dict[int, str]:
    """Parse an SRT file to a mapping dictionary of block ID to raw dialogue text."""
    sub_map: Dict[int, str] = {}
    try:
        content = Path(path).read_text(encoding="utf-8-sig", errors="ignore")
        blocks = [b.strip() for b in re.split(r"\n\n+", content.strip()) if b.strip()]
        for b in blocks:
            lines = b.splitlines()
            if len(lines) >= 3:
                raw_id = lines[0].strip()
                if raw_id.isdigit():
                    b_id = int(raw_id)
                    sub_map[b_id] = "\n".join(lines[2:]).strip()
    except Exception as e:
        logger.warning("Failed to parse SRT to dictionary layout: %s", e)
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
            len_s, len_t = len(s_norm), len(t_norm)
            # Skip SequenceMatcher for very short strings (unreliable ratios)
            if len_s < 10 or len_t < 10:
                continue
            # Quick length-ratio pre-check: if lengths differ by >2x, ratio can't be high
            min_len = min(len_s, len_t)
            max_len = max(len_s, len_t)
            if min_len / max_len < fallback_match_threshold:
                continue
            ratio = difflib.SequenceMatcher(None, s_norm, t_norm, autojunk=False).ratio()
            if ratio >= fallback_match_threshold:
                match_count += 1

    return match_count, f"{match_count} blocks matching source"


def find_source_srt(media_path: Union[str, Path], tgt_ext: Optional[str], src_lang: Optional[str] = None) -> Tuple[Path, str]:
    """Locate and return an existing source subtitle file associated with a media file."""
    media_path = Path(media_path)
    parent = media_path.parent
    stem = media_path.stem
    tgt_ext_clean = (tgt_ext or "").strip().lower()
    
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
        logger.warning("Error searching for source SRT: %s", e)

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

    # If we reach here, src_lang is falsy (was already checked above)
    default_lang = "und"
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
    
    for idx_in_chunk, block in enumerate(chunk):
        lines = block.splitlines()
        if len(lines) >= 3:
            try:
                b_id = int(lines[0])
            except ValueError:
                b_id = idx_in_chunk + 1
                logger.warning("Non-numeric block ID in SRT, using sequence index %d", b_id)
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
        text = p.read_text(encoding="utf-8-sig", errors="ignore")
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
        if s_len > TRANSLATION_MIN_SOURCE_LENGTH:  # Only flag significant length blocks
            if t_len > s_len * TRANSLATION_LENGTH_EXPANSION_FACTOR or t_len < s_len * TRANSLATION_LENGTH_TRUNCATION_FACTOR:
                length_deviations += 1
    
    # If more than 35% of blocks deviate excessively or are empty, trigger warnings
    if total_blocks > 0:
        deviation_ratio = length_deviations / total_blocks
        empty_ratio = empty_blocks / total_blocks
        if deviation_ratio > TRANSLATION_DEVIATION_RATIO_LIMIT:
            return False, f"{deviation_ratio:.0%} of blocks have excessive translation length deviation (possible hallucination)."
        if empty_ratio > TRANSLATION_EMPTY_RATIO_LIMIT:
            return False, f"{empty_ratio:.0%} of blocks were left blank or dropped."
            
    return True, "OK"


def _normalize_ollama_url(api_url: str) -> str:
    """Normalize an Ollama API URL to a valid base URL."""
    raw_url = api_url.strip() if api_url else "http://localhost:11434"
    if not raw_url.startswith("http://") and not raw_url.startswith("https://"):
        raw_url = f"http://{raw_url}"
    parsed = urllib.parse.urlparse(raw_url)
    return f"{parsed.scheme}://{parsed.netloc}"


# ════════════════════════════════════════════════════════════
#  OLLAMA HELPER PROCEDURES
# ════════════════════════════════════════════════════════════
class OllamaConnectionError(TranslationError):
    """Raised when the Ollama server cannot be reached."""
    pass


def try_start_ollama() -> bool:
    """Attempt to launch the Ollama app silently. Returns True if started or already running."""
    # Check if Ollama is already responding
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        pass

    if platform.system() == "Windows":
        candidate_paths = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama app.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
            Path(os.environ.get("PROGRAMFILES", "")) / "Ollama" / "ollama app.exe",
            Path(os.environ.get("PROGRAMFILES", "")) / "Ollama" / "ollama.exe",
        ]
        for exe_path in candidate_paths:
            if exe_path.exists():
                try:
                    qprint(f"  {style('[~]', C.CYAN)} Starting Ollama application...")
                    startup_info = subprocess.STARTUPINFO()
                    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startup_info.wShowWindow = 6  # SW_MINIMIZE
                    subprocess.Popen(
                        [str(exe_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
                        startupinfo=startup_info,
                    )
                    return _wait_for_ollama_ready()
                except Exception as e:
                    qprint(f"  {style('[!]', C.YELLOW)} Could not start Ollama: {e}")
                    return False
    else:
        # Linux / macOS — look for ollama on PATH or common install locations
        ollama_bin = shutil.which("ollama")
        if not ollama_bin and platform.system() == "Darwin":
            # Homebrew on Apple Silicon and Intel
            for candidate in ("/opt/homebrew/bin/ollama", "/usr/local/bin/ollama"):
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    ollama_bin = candidate
                    break
        if ollama_bin:
            try:
                qprint(f"  {style('[~]', C.CYAN)} Starting Ollama via background process...")
                subprocess.Popen(
                    [ollama_bin, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return _wait_for_ollama_ready()
            except Exception as e:
                qprint(f"  {style('[!]', C.YELLOW)} Could not start Ollama: {e}")
                return False
        else:
            qprint(
                f"  {style('[!]', C.YELLOW)} Ollama executable not found on PATH."
                f" Start Ollama manually or install it: https://ollama.com/download"
            )
            return False


def _wait_for_ollama_ready(timeout: float = 15.0) -> bool:
    """Block until Ollama responds on localhost:11434 or *timeout* seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(1)
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3):
                qprint(f"  {style('[+]', C.GREEN)} Ollama started successfully.")
                return True
        except Exception:
            continue
    qprint(f"  {style('[!]', C.YELLOW)} Ollama started but not yet responsive after {timeout:.0f}s. Continuing anyway.")
    return False


# Translation quality ranking for Ollama models (lower = better for multilingual tasks)
# Based on benchmark performance on translation, multilingual understanding, and instruction following.
OLLAMA_TRANSLATION_RANK: Final[Dict[str, int]] = {
    "qwen2.5:7b": 1,
    "gemma4:12b": 2,
    "llama3.1:8b": 3,
    "llama3.1:8b-instruct-q4_k_m": 3,
    "gemma4:e4b-it-qat": 4,
    "phi3.5": 5,
    "dolphin-llama3:8b": 6,
    "llama3.2-vision:11b": 7,
}

OLLAMA_CONNECT_RETRIES: Final[int] = 5
OLLAMA_CONNECT_DELAY: Final[float] = 3.0


def get_local_ollama_models(api_url: str, retries: int = OLLAMA_CONNECT_RETRIES) -> List[str]:
    """Query local Ollama instance for pulled models, retrying if server is starting up."""
    base_url = _normalize_ollama_url(api_url)
    tags_url = f"{base_url}/api/tags"
    last_err: Optional[Exception] = None

    for attempt in range(retries):
        try:
            req = urllib.request.Request(tags_url, method="GET")
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                models_list = data.get("models", [])
                if not isinstance(models_list, list):
                    return []
                models = [m.get("name", "") for m in models_list if isinstance(m, dict)]
                return [m for m in models if m]
        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError) as e:
            last_err = e
            if attempt < retries - 1:
                logger.debug("Ollama not ready yet (attempt %d/%d): %s", attempt + 1, retries, e)
                time.sleep(OLLAMA_CONNECT_DELAY)

    raise OllamaConnectionError(
        f"Cannot connect to Ollama server at '{base_url}' after {retries} attempts.\n"
        f"      Ensure the Ollama application is running and accessible.\n"
        f"      Last error: {last_err}"
    )


def display_ollama_models(api_url: str, retries: int = OLLAMA_CONNECT_RETRIES) -> Optional[List[str]]:
    """Fetch and display installed Ollama models with translation recommendations.
    
    Returns the list of model names on success, None if Ollama is not reachable.
    """
    base_url = _normalize_ollama_url(api_url)

    try:
        models = get_local_ollama_models(api_url, retries=retries)
    except OllamaConnectionError:
        return None

    if not models:
        qprint(f"\n  {style('[~]', C.YELLOW)} Ollama is running but no models are installed.")
        qprint(f"  {C.DIM}Run 'ollama pull <model>' to download a model, e.g.:")
        qprint(f"    ollama pull qwen2.5:7b{C.RESET}")
        return models

    # Rank and sort by translation quality
    def _rank(name: str) -> int:
        base = name.split(":")[0]
        return OLLAMA_TRANSLATION_RANK.get(name, OLLAMA_TRANSLATION_RANK.get(base, 99))

    ranked = sorted(models, key=_rank)
    top3 = [m for m in ranked[:3] if _rank(m) <= 3]

    qprint(f"\n  {style('[+]', C.GREEN)} Found {len(models)} installed Ollama model(s):")
    for i, m in enumerate(ranked, 1):
        base = m.split(":")[0]
        rank = _rank(m)
        tag = ""
        if i <= 3 and rank <= 3:
            tag = f"  {style('[BEST FOR TRANSLATION]', C.GREEN + C.BOLD)}"
        qprint(f"    {i}. {m}{tag}")

    if top3:
        qprint(f"\n  {style('[!]', C.CYAN)} Top pick{'s' if len(top3) > 1 else ''} for translation/multilingual: "
               f"{', '.join(top3)}")

    return models


def pull_ollama_model(api_url: str, model: str) -> bool:
    """Programmatically pull an Ollama model if it's missing from the device."""
    base_url = _normalize_ollama_url(api_url)
    pull_url = f"{base_url}/api/pull"

    qprint(f"  {style('[~]', C.CYAN)} Model '{model}' not found locally. Pulling from Ollama registry...")

    payload = json.dumps({"name": model, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        pull_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=3600) as response:
                if response.status == 200:
                    qprint(f"  {style('[+]', C.GREEN)} Successfully pulled local Ollama model '{model}'!")
                    return True
                else:
                    logger.warning("Ollama pull returned HTTP %d for model '%s'", response.status, model)
        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError) as e:
            if attempt == 0:
                qprint(f"  {style('[!]', C.YELLOW)} Ollama pull failed ({e}). Trying to start Ollama...")
                if try_start_ollama():
                    continue  # Retry after starting
            qprint(f"  {style('[x]', C.RED)} Failed to pull model '{model}' from Ollama: {e}")
            logger.warning("Ollama pull failed for model '%s': %s", model, e)
    return False


def resolve_ollama_model(api_url: str, requested_model: str) -> str:
    """
    Robustly resolve the requested Ollama model tag.
    If the model exists exactly or by base name, use it.
    If it's missing, try to pull it.
    If pull fails, falls back to the best available model on the device.
    """
    req_clean = requested_model.strip()
    req_lower = req_clean.lower()
    req_base = req_lower.split(":")[0] if ":" in req_lower else req_lower

    base_url = _normalize_ollama_url(api_url)

    try:
        local_models = get_local_ollama_models(api_url)
    except OllamaConnectionError:
        # Try to start Ollama automatically before giving up
        qprint(f"  {style('[!]', C.YELLOW)} Cannot reach Ollama at '{base_url}'. Attempting to start...")
        if try_start_ollama():
            try:
                local_models = get_local_ollama_models(api_url)
            except Exception as e:
                raise OllamaConnectionError(
                    f"Cannot connect to Ollama server at '{base_url}'.\n"
                    f"      Ollama was started but is still not responsive.\n"
                    f"      Error: {e}"
                )
        else:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama server at '{base_url}'.\n"
                f"      Ensure the Ollama application is running and accessible.\n"
                f"      You can start it manually from the Start menu."
            )

    # 1. Exact Match Check
    for lm in local_models:
        if lm.lower() == req_lower:
            return lm

    # 2. Base Name Match Check (e.g., requested "qwen2.5" and "qwen2.5:7b" is installed)
    for lm in local_models:
        lm_lower = lm.lower()
        lm_base = lm_lower.split(":")[0] if ":" in lm_lower else lm_lower
        if lm_base == req_base:
            return lm

    # 3. Pull from registry if missing entirely
    try:
        pull_success = pull_ollama_model(api_url, req_clean)
    except Exception as e:
        qprint(f"  {style('[x]', C.RED)} Failed to pull model '{req_clean}': {e}")
        pull_success = False

    if pull_success:
        try:
            updated_models = get_local_ollama_models(api_url)
        except Exception:
            updated_models = local_models
        for lm in updated_models:
            if lm.lower() == req_lower:
                return lm
        if updated_models:
            for lm in updated_models:
                lm_lower = lm.lower()
                lm_base = lm_lower.split(":")[0] if ":" in lm_lower else lm_lower
                if lm_base == req_base:
                    return lm

    # 4. Fallback of last resort: if offline or download failed, use best installed model
    if local_models:
        # Try to find a family match first (e.g. "gemma" or "aya" prefix)
        family_matches = [lm for lm in local_models if lm.lower().startswith(req_base)]
        if family_matches:
            family_matches.sort(key=natural_keys, reverse=True)
            fallback_model = family_matches[0]
            qprint(
                f"  {style('[!]', C.YELLOW)} Failed to acquire requested model '{requested_model}'. "
                f"Gracefully falling back to matching family model '{fallback_model}'."
            )
            return fallback_model

        # Try to look for highly capable general translation models in local inventory
        preferred_patterns = ["aya", "llama3.1", "qwen2.5", "gemma4", "gemma2", "qwen", "llama3", "llama", "mistral", "phi"]
        for pattern in preferred_patterns:
            matches = [lm for lm in local_models if pattern in lm.lower()]
            if matches:
                matches.sort(key=natural_keys, reverse=True)
                fallback_model = matches[0]
                qprint(
                    f"  {style('[!]', C.YELLOW)} Failed to acquire requested model '{requested_model}'. "
                    f"Gracefully falling back to capable local model '{fallback_model}'."
                )
                return fallback_model

        fallback_model = local_models[0]
        qprint(
            f"  {style('[!]', C.YELLOW)} Failed to acquire requested model '{requested_model}'. "
            f"Gracefully falling back to first installed model '{fallback_model}'."
        )
        return fallback_model
    else:
        raise TranslationError(
            f"Requested model '{requested_model}' is not installed on Ollama.\n"
            f"      No local Ollama models found at '{base_url}'.\n"
            f"      Please run: ollama pull {requested_model}"
        )


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
    if len(clean) > 10:
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
            args.translation_model = DEFAULT_GEMINI_MODEL
            context.provenance["translator"] = "Auto-Override"
            context.provenance["translation_model"] = "Auto-Override"

    # Auto-align the translation model defaults on mismatch to avoid endpoint errors
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
            fallback_m = DEFAULT_TRANSLATION_MODELS.get(current_trans, DEFAULT_GEMINI_MODEL)
            adjustments.append(f"Model mismatch detected for '{current_trans}'. Aligning translation model to '{fallback_m}'.")
            args.translation_model = fallback_m
            context.provenance["translation_model"] = "Auto-Override"

    if args.hardsub and not args.embed:
        adjustments.append(
            "Hardsub is enabled but Muxing is disabled. "
            "(Burning subtitles requires muxing; auto-enabling Mux.)"
        )
        args.embed = True
        context.provenance["embed"] = "Auto-Override"

    if args.translate and not args.api_key and args.translator not in ("google", "ollama"):
        adjustments.append(
            "Translation is requested, but no API key was configured. "
            "(Disabling translate step.)"
        )
        args.translate = False
        context.provenance["translate"] = "Auto-Override"
    elif args.translate and args.api_key:
        is_valid, err_msg = validate_api_key(args.api_key)
        if not is_valid:
            adjustments.append(f"Invalid API key format: {err_msg}. (Disabling translate.)")
            args.translate = False
            context.provenance["translate"] = "Auto-Override"

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
        logger.warning("Failed to write audit log: %s", e)


def print_effective_settings(args: PipelineConfig) -> None:
    """Display the effective pipeline configuration."""

    def get_src(key_name: str) -> str:
        src = context.provenance.get(key_name, "Default")
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
    if args.hardsub:
        detected = detect_gpu_encoder()
        gpu_label = detected.upper() if detected != "none" else "Software (libx264)"
        qprint(
            f"  GPU Encoder:       {style(gpu_label, C.CYAN):<49} "
            f"{style('Preset: ' + args.preset, C.DIM)}"
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
    req_url = url_override.strip() if url_override else GEMINI_URL_TEMPLATE.format(model=model)
    if url_override and "?" in req_url:
        req_url = req_url.split("?")[0]
    text = "\n".join(prompt_or_list) if isinstance(prompt_or_list, list) else prompt_or_list
    payload_dict = {"contents": [{"parts": [{"text": text}]}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
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
        "Anthropic-Version": "2024-10-01"
    }
    payload_dict = {
        "model": model,
        "max_tokens": 16000,
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
    if isinstance(prompt_or_list, list):
        joined_text = "\n###\n".join(prompt_or_list)
    else:
        joined_text = prompt_or_list
    payload_dict = {"q": joined_text}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = urllib.parse.urlencode(payload_dict).encode("utf-8")
    return req_url, payload, headers


def _build_ollama_payload(model: str, prompt_or_list: Union[str, List[str]], url_override: str, tgt_lang: str) -> Tuple[str, bytes, Dict[str, str]]:
    base_url = _normalize_ollama_url(url_override)
    parsed = urllib.parse.urlparse(base_url)
    path = parsed.path if parsed.path not in ("", "/") else "/api/chat"
    req_url = f"{base_url}{path}"
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
#  HTTP CONNECTION POOL FOR TRANSLATION API REQUESTS
# ════════════════════════════════════════════════════════════
class _HTTPPool:
    """Thread-safe HTTP connection pool.

    Each thread gets its own ``http.client.HTTP[S]Connection`` per host,
    so concurrent requests never share a socket.  Connections are reused
    across requests within the same thread when the socket is still alive.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    @staticmethod
    def _key(parsed: urllib.parse.ParseResult) -> str:
        port = parsed.port
        scheme = parsed.scheme.lower()
        default_port = 443 if scheme == "https" else 80
        host = parsed.hostname or ""
        return f"{host}:{port or default_port}"

    def _get_conns(self) -> Dict[str, http.client.HTTPConnection]:
        """Return the calling thread's connection dict (creating it if needed)."""
        try:
            return self._local.conns
        except AttributeError:
            self._local.conns = {}
            return self._local.conns

    def get(self, url: str, timeout: float = 60,
            ssl_ctx: Optional[ssl.SSLContext] = None) -> http.client.HTTPConnection:
        parsed = urllib.parse.urlparse(url)
        key = self._key(parsed)
        scheme = (parsed.scheme or "https").lower()
        host = parsed.hostname or ""
        port = parsed.port or (443 if scheme == "https" else 80)

        conns = self._get_conns()
        conn = conns.get(key)
        if conn is not None and conn.sock is None:
            # Socket was closed server-side; discard and reconnect.
            conns.pop(key, None)
            conn = None
        if conn is None:
            if scheme == "https":
                conn = http.client.HTTPSConnection(host, port, timeout=timeout, context=ssl_ctx)
            else:
                conn = http.client.HTTPConnection(host, port, timeout=timeout)
            conns[key] = conn
        return conn

    def evict(self, url: str) -> None:
        """Remove the pooled connection for *url* so the next request gets a fresh one."""
        parsed = urllib.parse.urlparse(url)
        key = self._key(parsed)
        conns = self._get_conns()
        conns.pop(key, None)

    def close_all(self) -> None:
        """Close connections owned by the calling thread.

        Because connections are stored per-thread via ``threading.local()``,
        this only closes sockets for the thread that calls it.  Connections
        held by other threads (e.g. watcher ``ThreadPoolExecutor`` workers)
        are not affected — the OS reclaims their sockets on process exit
        and servers time out idle connections.
        """
        conns = self._get_conns()
        for conn in conns.values():
            try:
                conn.close()
            except Exception:
                pass
        conns.clear()


_http_pool = _HTTPPool()


def _request_with_pool(
    url: str,
    data: bytes,
    headers: Dict[str, str],
    method: str = "POST",
    timeout: float = 60,
    ssl_ctx: Optional[ssl.SSLContext] = None,
) -> Tuple[int, Dict[str, str], bytes]:
    """POST *data* through the connection pool.

    Returns ``(status_code, response_headers, body_bytes)``.
    Retries once on ``ConnectionError`` / ``RemoteDisconnected``.
    """
    parsed = urllib.parse.urlparse(url)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    last_err: Optional[Exception] = None
    for attempt in range(2):
        conn = _http_pool.get(url, timeout=timeout, ssl_ctx=ssl_ctx)
        try:
            conn.request(method, path, body=data, headers=headers)
            resp = conn.getresponse()
            resp_headers = dict(resp.getheaders())
            resp_body = resp.read()
            return resp.status, resp_headers, resp_body
        except (http.client.RemoteDisconnected, ConnectionError, OSError) as e:
            last_err = e
            _http_pool.evict(url)
            if attempt == 0:
                continue
            raise

    raise last_err  # type: ignore[misc]


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

    # Resolve, Pull, and Fallback models programmatically on the device
    if translator == "ollama":
        model = resolve_ollama_model(url_override, model)

    success = False
    response_text = ""
    translated_map: Dict[int, str] = {}

    ssl_context = _get_ssl_context()

    for attempt in range(MAX_TRANSLATION_CHUNK_RETRIES):
        try:
            if not global_rate_limiter.acquire(timeout=30):
                if attempt == MAX_TRANSLATION_CHUNK_RETRIES - 1:
                    raise TranslationError("API request blocked: Rate limiter timeout after retries.")
                time.sleep(TRANSLATION_RETRY_BASE_DELAY * (2**attempt))
                continue

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

            headers["User-Agent"] = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) SubsPipeline/{__version__}"
            
            req_timeout = PROVIDER_TIMEOUTS.get(translator, 60)

            # Show progress for slow local providers
            _ollama_alive = None
            _dot_thread = None
            if translator == "ollama":
                ollama_start = time.monotonic()
                # Background spinner to show liveness during long Ollama inference
                _ollama_alive = threading.Event()
                _spin_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                _spin_idx = [0]
                def _ollama_spinner():
                    while not _ollama_alive.wait(0.4):
                        elapsed = time.monotonic() - ollama_start
                        ch = _spin_chars[_spin_idx[0] % len(_spin_chars)]
                        _spin_idx[0] += 1
                        sys.stdout.write(f"\r  {style(ch, C.CYAN)} Ollama thinking... {style(f'{elapsed:.0f}s', C.DIM)}  ")
                        sys.stdout.flush()
                _dot_thread = threading.Thread(target=_ollama_spinner, daemon=True)
                _dot_thread.start()

            try:
                resp_status, resp_headers, resp_bytes = _request_with_pool(
                    req_url, payload, headers, method="POST",
                    timeout=req_timeout, ssl_ctx=ssl_context,
                )

                if translator == "ollama" and _ollama_alive:
                    _ollama_alive.set()
                    _dot_thread.join(timeout=1)
                    elapsed = time.monotonic() - ollama_start
                    sys.stdout.write(f"\r  {style('[+]', C.GREEN)} Ollama translation complete ({elapsed:.1f}s)                    \n")
                    sys.stdout.flush()

                if resp_headers.get("Content-Encoding") == "gzip":
                    resp_bytes = gzip.decompress(resp_bytes)

                resp_decoded = resp_bytes.decode("utf-8", errors="replace")

                if resp_status >= 400:
                    raise urllib.error.HTTPError(req_url, resp_status, resp_decoded, resp_headers, None)

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
                                
                        # Filter out raw reasoning blocks (like DeepSeek-R1 <think> loops) from subtitle text
                        if response_text:
                            response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
                            
                        if response_text.strip():
                            success = True

                if success:
                    context.reset_consecutive_429s()
                    context.reset_consecutive_total_failures()
                    break

            finally:
                if _ollama_alive and not _ollama_alive.is_set():
                    _ollama_alive.set()
                    if _dot_thread:
                        _dot_thread.join(timeout=1)

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
                context.increment_consecutive_429s()
                if context.get_consecutive_429s() >= CONSECUTIVE_429_LIMIT:
                    context.set_translation_disabled(True)
                    raise TranslationError("Rate limits (429) hit consecutively. Suspending API translation.")
                if attempt == MAX_TRANSLATION_CHUNK_RETRIES - 1:
                    context.increment_consecutive_total_failures()
                    raise TranslationError(f"API Request failed after retries with HTTP Error 429.")
                time.sleep(5 * attempt + 5)
            elif e.code == 503:
                logger.warning("Transient backend error 503 received. Retrying with delay...")
                if attempt == MAX_TRANSLATION_CHUNK_RETRIES - 1:
                    context.increment_consecutive_total_failures()
                    raise TranslationError(f"API Request failed after retries with HTTP Error {e.code}.")
                time.sleep(TRANSLATION_RETRY_BASE_DELAY * (3**attempt) + random.uniform(1, 3))
            elif e.code >= 500:
                logger.warning("Transient backend error %d received. Retrying...", e.code)
                if attempt == MAX_TRANSLATION_CHUNK_RETRIES - 1:
                    context.increment_consecutive_total_failures()
                    raise TranslationError(f"API Request failed after retries with HTTP Error {e.code}")
                time.sleep(TRANSLATION_RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1))
            else:
                context.increment_consecutive_total_failures()
                raise TranslationError(f"API Request failed with HTTP Error {e.code}")
        except urllib.error.URLError as e:
            if translator == "ollama":
                raise TranslationError(
                    f"Failed to connect to local Ollama server at '{url_override or 'http://localhost:11434'}'.\n"
                    f"      Ensure the Ollama application is running and accessible."
                )
                
            logger.warning("Connection failure encountered: %s. Retrying...", e.reason)
            if attempt == MAX_TRANSLATION_CHUNK_RETRIES - 1:
                context.increment_consecutive_total_failures()
                raise TranslationError(f"API Request failed after retries with URLError: {e.reason}")
            time.sleep(TRANSLATION_RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1))
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.warning("Transient network error during API request: %s. Retrying...", e)
            if attempt == MAX_TRANSLATION_CHUNK_RETRIES - 1:
                context.increment_consecutive_total_failures()
                raise TranslationError(f"API Request failed after retries with network error: {e}")
            time.sleep(TRANSLATION_RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1))
        except json.JSONDecodeError as e:
            logger.warning("Malformed JSON response (transient). Retrying... %s", e)
            if attempt == MAX_TRANSLATION_CHUNK_RETRIES - 1:
                context.increment_consecutive_total_failures()
                raise TranslationError(f"API returned invalid JSON after retries: {e}")
            time.sleep(TRANSLATION_RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1))
        except TranslationError:
            raise
        except Exception as e:
            logger.error("Unexpected error during API request (not retried): %s", e, exc_info=True)
            context.increment_consecutive_total_failures()
            raise TranslationError(f"API Request failed with unexpected error: {e}")

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
                    logger.debug("Skipping malformed block ID in translation response: '%s'", parts[idx])
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
    args_ref: Optional[PipelineConfig] = None,
) -> int:
    """Translate an SRT file using the selected translation router API.

    Returns the number of fallback (untranslated) blocks.
    Raises TranslationError on any failure.
    """
    srt_src = Path(srt_src)
    srt_tgt = Path(srt_tgt)

    is_valid, reason = is_valid_srt(srt_src, args_ref=args_ref)
    if not is_valid:
        raise TranslationError(f"Source SRT validation failed: {reason}")

    try:
        content = srt_src.read_text(encoding="utf-8-sig", errors="ignore")
    except Exception as e:
        raise TranslationError(f"Read error: {e}") from e

    blocks = [b.strip() for b in re.split(r"\n\n+", content.strip()) if b.strip()]
    if not blocks:
        raise TranslationError("Empty SRT file.")

    translator = translator.lower().strip()
    
    if translator not in ("google", "ollama"):
        key_valid, key_err = validate_api_key(api_key)
        if not key_valid:
            raise TranslationError(f"Invalid API key: {key_err}")

    model = translation_model or DEFAULT_TRANSLATION_MODELS.get(translator, DEFAULT_GEMINI_MODEL)

    chunks: List[List[str]] = []
    current_chunk: List[str] = []
    current_chars = 0
    
    for b in blocks:
        b_len = len(b)
        # Primary constraint: character limit.  Flush before adding if the
        # chunk would exceed MAX_CHUNK_CHAR_LIMIT.  The `current_chunk`
        # guard ensures we never flush an empty chunk — an oversized single
        # block is always added to the (now-empty) chunk and flushed on the
        # *next* iteration, producing a deterministic one-block chunk.
        if current_chunk and current_chars + b_len > MAX_CHUNK_CHAR_LIMIT:
            chunks.append(current_chunk)
            current_chunk = []
            current_chars = 0
        # Secondary hard cap: block count.  Prevents unbounded growth when
        # all blocks are tiny (well under the char limit).
        elif current_chunk and len(current_chunk) >= GEMINI_CHUNK_SIZE:
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
        with temp_file_guard(tmp_tgt) as guarded_tmp:
            with open(guarded_tmp, "w", encoding="utf-8") as tmp_fh:

                for chunk_idx, chunk in enumerate(chunks, 1):
                    pct = (chunk_idx / total_chunks) * 100
                    qprint(
                        f"  {style('[~]', C.DIM)} Translating chunk {chunk_idx}/{total_chunks} "
                        f"({pct:.0f}%) ({len(chunk)} blocks) using {translator.upper()}...{C.RESET}"
                    )

                    if translator in ("deepl", "google"):
                        diags = []
                        for b in chunk:
                            lines = b.splitlines()
                            diag = "\n".join(lines[2:]) if len(lines) >= 3 else ""
                            diags.append(diag)
                        translated_dict = _execute_translator_request(
                            translator, model, api_url, api_key, diags, tgt_lang, tgt_ext
                        )
                    else:
                        prompt = _prepare_translation_prompt(chunk, tgt_lang)
                        response_text = _execute_translator_request(
                            translator, model, api_url, api_key, prompt, tgt_lang, tgt_ext
                        )
                        translated_dict = _parse_translation_response(response_text)

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
                        chunk_delay = PROVIDER_INTER_CHUNK_DELAYS.get(translator, GEMINI_INTER_CHUNK_DELAY)
                        if chunk_delay > 0:
                            time.sleep(chunk_delay)

            os.replace(str(guarded_tmp), str(srt_tgt))

        fallbacks, _ = detect_fallbacks(srt_src, srt_tgt, fallback_match_threshold, lang_code=(src_lang or "en"))
        return fallbacks

    except KeyboardInterrupt:
        logger.warning("Translation interrupted by user")
        raise TranslationError("Translation interrupted by user")


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
    executor = ThreadPoolExecutor(max_workers=WATCHER_MAX_WORKERS)

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

            executor.submit(handle)

        def on_created(self, event) -> None:
            if not event.is_directory:
                self._dispatch(event.src_path)

        def on_modified(self, event) -> None:
            if not event.is_directory:
                self._dispatch(event.src_path)

        def on_moved(self, event) -> None:
            if not event.is_directory:
                self._dispatch(event.dest_path)

    try:
        observer = Observer()
        observer.schedule(WatchHandler(), path=str(args.folder), recursive=False)
        observer.start()
    except Exception as e:
        qprint(f"{C.RED}[x] Watch mode failed to start: {e}{C.RESET}")
        executor.shutdown(wait=True, cancel_futures=True)
        return

    qprint(f"\n{C.CYAN}  [*] Watch Mode -- '{Path(args.folder).name}'{C.RESET}")
    qprint(f"  {C.DIM}Ctrl+C to stop. Status updates every 30s.{C.RESET}")

    try:
        heartbeat_interval = 30
        last_heartbeat = time.monotonic()
        while True:
            time.sleep(1)
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_interval:
                with lock:
                    active = len(in_flight)
                qprint(f"  {C.DIM}[*] Heartbeat -- {active} file{'s' if active != 1 else ''} in flight{C.RESET}")
                last_heartbeat = now
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    executor.shutdown(wait=True, cancel_futures=True)
    cleanup_all_temp_files()
    transcription_manager.terminate()


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

    if context.migration_status == "loaded":
        print(f"  {style('[~]', C.GREEN)} Loaded saved profile settings.\n")

    if context.config_warning:
        print(f"  {style('[!]', C.RED)} CONFIG SYSTEM: {context.config_warning}\n")

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
            context.provenance["folder"] = "Interactive"
            break
        else:
            resolved_p = Path(f_in).resolve()
            if resolved_p.is_dir():
                args.folder = str(resolved_p)
                context.provenance["folder"] = "Interactive"
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
                context.provenance[k] = "Saved Profile"
            context.provenance["translate"] = "Saved Profile"
            context.provenance["transcribe"] = "Saved Profile"
            context.provenance["embed"] = "Saved Profile"

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
    context.provenance["tgt_lang"] = "Interactive"

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
    context.provenance["tgt_ext"] = "Interactive"

    # Step 4: Source language
    saved_src = cfg_memory.get("src_lang") or ""
    print(f"\n{C.BOLD}> Step 4: Source Language (What is spoken in the video){C.RESET}")
    print(
        f"  {C.DIM}Blank = auto-detect  |  ISO codes: ja  en  ko  zh  ar  es ...{C.RESET}"
    )
    src_in = input(f" Source Language [{saved_src or 'auto'}]: ").strip().lower()
    args.src_lang = src_in if src_in else (saved_src if saved_src else None)
    context.provenance["src_lang"] = "Interactive"

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
    context.provenance["device"] = "Interactive"
    
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
    context.provenance["model"] = "Interactive"
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
    context.provenance["translator"] = "Interactive"

    if args.translator == "google":
        args.translation_model = "google-v1-free"
        args.api_url = ""
        args.api_key = ""
        qprint(f"  {style('[~]', C.GREEN)} Google Translate (Free) selected. API key and model selection bypassed.")
    elif args.translator == "ollama":
        print(f"\n{C.BOLD}── Local Translation Models sub-wizard ─────────────────────{C.RESET}")
        print(f"    {style('[0]', C.CYAN)} Qwen2.5-7B (qwen2.5:7b)          - Good balanced speed/quality")
        print(f"    {style('[1]', C.CYAN)} Llama-3.1-8B (llama3.1:8b)        - Strong multilingual capability")
        print(f"    {style('[2]', C.CYAN)} NLLB-200 (nllb)                  - Dedicated translation model")
        print(f"    {style('[3]', C.CYAN)} Aya Expanse 8B (aya-expanse:8b)   - Highly optimized multilingual")
        print(f"    {style('[4]', C.CYAN)} Mistral-7B (mistral:7b)          - Efficient logical reasoning")
        print(f"    {style('[5]', C.CYAN)} DeepSeek-R1-8B (deepseek-r1:8b)  - Reasoning model structure")
        print(f"    {style('[6]', C.CYAN)} Gemma-2-9B (gemma2:9b)            - Extremely high quality 9B")
        print(f"    {style('[7]', C.CYAN)} Gemma-2-27B (gemma2:27b)          - State-of-the-art offline translation")
        print(f"    {style('[8]', C.CYAN)} Phi-3.5 (phi3.5)                 - Lightweight and fast")
        print(f"    {style('[9]', C.CYAN)} Custom model tag")
        
        local_choice = input("  Selection [0]: ").strip()
        local_map = {
            "0": "qwen2.5:7b",
            "1": "llama3.1:8b",
            "2": "nllb",
            "3": "aya-expanse:8b",
            "4": "mistral:7b",
            "5": "deepseek-r1:8b",
            "6": "gemma2:9b",
            "7": "gemma2:27b",
            "8": "phi3.5"
        }
        if local_choice in local_map:
            args.translation_model = local_map[local_choice]
        elif local_choice == "9":
            # Try to start Ollama if not running, then show installed models
            saved_url = cfg_memory.get("api_url") or "http://localhost:11434/api/chat"
            base_url = _normalize_ollama_url(saved_url)
            ollama_running = False
            try:
                req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=2):
                    ollama_running = True
            except Exception:
                pass

            if not ollama_running:
                qprint(f"  {style('[~]', C.CYAN)} Ollama not running. Starting...")
                try_start_ollama()

            # Quick single attempt to show models (no retries)
            try:
                req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    models_list = data.get("models", [])
                    if models_list:
                        print(f"\n  {style('[+]', C.GREEN)} Your installed Ollama models:")
                        for i, m in enumerate(models_list, 1):
                            name = m.get("name", "?")
                            print(f"    {i}. {name}")
                        print(f"\n  {C.DIM}Enter a model tag from above, or type a custom one - The name not the number before it -:{C.RESET}")
                    else:
                        print(f"\n  {C.DIM}Ollama is running but has no models. Type a model tag:{C.RESET}")
            except Exception:
                print(f"\n  {C.DIM}Could not list models. Type a model tag to use:{C.RESET}")

            args.translation_model = input("  Model tag: ").strip() or "qwen2.5:7b"
        elif local_choice: # Allows typed entries like gemma4:12b directly at Selection prompt
            args.translation_model = local_choice
        else:
            args.translation_model = "qwen2.5:7b"
            
        saved_url = cfg_memory.get("api_url") or "http://localhost:11434/api/chat"
        args.api_url = input(f"  Local Ollama Endpoint URL [{saved_url}]: ").strip() or saved_url
        args.api_key = ""
        qprint(f"  {style('[~]', C.GREEN)} Ollama translator configured with model: {args.translation_model}")
    else:
        saved_t_model = cfg_memory.get("translation_model") or DEFAULT_TRANSLATION_MODELS.get(args.translator, DEFAULT_GEMINI_MODEL)
        args.translation_model = input(f"  Translation Model [{saved_t_model}]: ").strip() or saved_t_model
        context.provenance["translation_model"] = "Interactive"

        saved_url = cfg_memory.get("api_url", "")
        args.api_url = input(f"  Custom API Gateway URL (Leave blank for default) [{saved_url}]: ").strip() or saved_url
        context.provenance["api_url"] = "Interactive"

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
            key_input = getpass.getpass("  Key: ").strip() or saved_key
            if not key_input:
                args.api_key = ""
                break
            valid, err = validate_api_key(key_input)
            if valid:
                args.api_key = key_input
                break
            print(f"  {style('[!]', C.YELLOW)} {err} Please try again.")
            
    args.translate = bool(args.api_key) or args.translator in ("google", "ollama")
    context.provenance["api_key"] = "Interactive"
    context.provenance["translate"] = "Interactive"

    # Step 8: Output format
    print(f"\n{C.BOLD}> Step 8: Output Format{C.RESET}")
    print(f"    {style('[0]', C.CYAN)} Softsub -- toggleable track  (default)")
    print(f"    {style('[1]', C.CYAN)} Hardsub -- burned into video")
    args.hardsub = input("  Selection [0]: ").strip() == "1"
    context.provenance["hardsub"] = "Interactive"

    # Step 9: Subtitle Validation
    print(f"\n{C.BOLD}> Step 9: Health Check Settings{C.RESET}")
    saved_min = cfg_memory.get("min_blocks", 3)
    try:
        val_blocks = input(f"  Minimum valid blocks [{saved_min}]: ").strip()
        args.min_blocks = int(val_blocks) if val_blocks else int(saved_min)
    except ValueError:
        args.min_blocks = 3
    context.provenance["min_blocks"] = "Interactive"

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
                context.provenance[k] = "Interactive"
        except ValueError:
            print(
                f"    {style('[!]', C.YELLOW)} Invalid numeric values. "
                f"Standard defaults retained."
            )
    else:
        # When the user declines to tune, use saved profile values if present,
        # otherwise fall back to current (dataclass default) values.
        args.srt_max_avg_duration = cfg_memory.get("srt_max_avg_duration", args.srt_max_avg_duration)
        args.srt_min_avg_duration = cfg_memory.get("srt_min_avg_duration", args.srt_min_avg_duration)
        args.srt_dup_ratio = cfg_memory.get("srt_dup_ratio", args.srt_dup_ratio)
        args.fallback_match_threshold = cfg_memory.get("fallback_match_threshold", args.fallback_match_threshold)

    # Step 11: Pipeline Steps
    print(f"\n{C.BOLD}> Step 11: Pipeline Steps{C.RESET}")
    if getattr(args, "skip_transcribe", False):
        args.transcribe = False
        print("  Transcribe? [Disabled via CLI]")
    else:
        args.transcribe = input("  Transcribe? [Y/n]: ").strip().lower() != "n"
    context.provenance["transcribe"] = "Interactive"
    
    if args.translate:
        if getattr(args, "skip_translate", False):
            args.translate = False
            print("  Translate?  [Disabled via CLI]")
        else:
            args.translate = input("  Translate?  [Y/n]: ").strip().lower() != "n"
            context.provenance["translate"] = "Interactive"
            
    if getattr(args, "skip_embed", False):
        args.embed = False
        print("  Mux?        [Disabled via CLI]")
    else:
        args.embed = input("  Mux?        [Y/n]: ").strip().lower() != "n"
    context.provenance["embed"] = "Interactive"

    if args.hardsub and not args.embed:
        print(
            f"  {style('[!]', C.YELLOW)} Override: Hardsub is active. Soft muxing step "
            f"enabled to complete burning action."
        )
        args.embed = True
        context.provenance["embed"] = "Auto-Override"

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
        "max_audit_logs": getattr(args, "max_audit_logs", cfg_memory.get("max_audit_logs", MAX_AUDIT_LOGS)),
        "gemini_model": getattr(args, "gemini_model", cfg_memory.get("gemini_model", DEFAULT_GEMINI_MODEL)),
        "whisper_beam_size": getattr(args, "whisper_beam_size", cfg_memory.get("whisper_beam_size", WHISPER_BEAM_SIZE)),
        "tgt_langs": getattr(args, "tgt_langs", cfg_memory.get("tgt_langs", "")),
        "gpu_accel": getattr(args, "gpu_accel", cfg_memory.get("gpu_accel", "auto")),
        "preset": getattr(args, "preset", cfg_memory.get("preset", "fast")),
        "hardsub_fontsize": getattr(args, "hardsub_fontsize", cfg_memory.get("hardsub_fontsize", 22)),
        "hardsub_outline": getattr(args, "hardsub_outline", cfg_memory.get("hardsub_outline", 2)),
        "hardsub_shadow": getattr(args, "hardsub_shadow", cfg_memory.get("hardsub_shadow", 1)),
        "target_res": getattr(args, "target_res", cfg_memory.get("target_res", "")),
        "hardsub_mkv": getattr(args, "hardsub_mkv", cfg_memory.get("hardsub_mkv", False)),
        "vmaf_check": getattr(args, "vmaf_check", cfg_memory.get("vmaf_check", False)),
        "preset_auto": getattr(args, "preset_auto", cfg_memory.get("preset_auto", False)),
    })


# ════════════════════════════════════════════════════════════
#  ASS STYLE BUILDER
# ════════════════════════════════════════════════════════════
def _build_force_style(
    tgt_lang: str,
    fontsize: int = 22,
    outline: int = 2,
    shadow: int = 1,
) -> str:
    """Build FFmpeg force_style string with Cairo Bold font and lower-center alignment.

    Cairo Bold is *always* specified so that even when the font file cannot be
    resolved, libass will look for a font named Cairo on the system.

    RTL languages get: Alignment=3 (bottom-right), MarginR increased.
    LTR languages get: Alignment=2 (bottom-center), default margins.
    """
    parts: List[str] = []
    # Always specify Cairo Bold — the font file in fontsdir or system fonts
    parts.append("FontName=Cairo")
    parts.append("Bold=1")
    parts.append("Weight=700")
    parts.append(f"FontSize={fontsize}")
    parts.append("PrimaryColour=&H00FFFFFF")  # White text
    parts.append("OutlineColour=&H00000000")  # Black outline
    parts.append(f"Outline={outline}")
    parts.append(f"Shadow={shadow}")

    lang_ext = tgt_lang.lower().strip() if tgt_lang else ""
    # Also check full language names mapped to extensions
    for ext, name in EXT_TO_LANG_NAME.items():
        if name.lower() == lang_ext:
            lang_ext = ext
            break

    if lang_ext in RTL_LANGUAGES:
        # RTL: right-aligned, push subtitles slightly right to avoid edge clipping
        parts.append("Alignment=3")   # bottom-right in ASS (numpad layout)
        parts.append("MarginL=20")
        parts.append("MarginR=40")
    else:
        # LTR: center-aligned at bottom
        parts.append("Alignment=2")   # bottom-center
        parts.append("MarginL=20")
        parts.append("MarginR=20")
    # Push subtitles up from the very bottom so they sit in the lower third
    parts.append("MarginV=30")

    return ",".join(parts)


# ════════════════════════════════════════════════════════════
#  FONT INSTALLER
# ════════════════════════════════════════════════════════════
def ensure_font_installed() -> Optional[str]:
    """Ensure Cairo Bold font is available. Returns the font file path or None."""
    # 1. Check Windows Fonts directory
    win_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidate = win_fonts / DEFAULT_HARDSUB_FONT_FILE
    if candidate.is_file():
        return str(candidate)

    # 2. Check local pipeline directory
    local_font = Path(__file__).parent / DEFAULT_HARDSUB_FONT_FILE
    if local_font.is_file():
        return str(local_font)

    # 3. Download and install
    qprint(f"  {style('>', C.CYAN)} Cairo Bold font not found. Downloading...")
    try:
        req = urllib.request.Request(CAIRO_BOLD_GITHUB_URL, headers={"User-Agent": f"SubsPipeline/{__version__}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        # Validate: font files start with TrueType/OpenType magic bytes or are >100KB
        if len(data) < 100000 or data[:4] not in (b"\x00\x01\x00\x00", b"OTTO", b"ttcf", b"wOFF"):
            raise ValueError(f"Downloaded file is not a valid font ({len(data)} bytes)")
        local_font.write_bytes(data)
        qprint(f"  {style('+', C.GREEN)} Downloaded to {local_font}")
        return str(local_font)
    except Exception as e:
        logger.warning("Font download failed: %s", e)
        qprint(f"  {style('[!]', C.YELLOW)} Could not download Cairo Bold: {e}")
        qprint(f"  {style('>', C.DIM)} Install manually: https://fonts.google.com/specimen/Cairo")
        return None


# ════════════════════════════════════════════════════════════
#  INTERRUPTIBLE SUBPROCESS RUNNER
# ════════════════════════════════════════════════════════════
def _run_subprocess_interruptible(
    cmd: List[str],
    timeout: float = 600.0,
    cwd: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess with reader threads so the main thread stays
    responsive to Ctrl+C on Windows (where blocking pipe reads prevent it).
    
    Behaves like subprocess.run(cmd, capture_output=True, ...) but uses
    daemon threads to drain stdout/stderr, allowing signal delivery.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        encoding="utf-8",
        errors="replace",
    )

    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []

    def _drain(stream: Any, sink: List[str]) -> None:
        try:
            for chunk in iter(lambda: stream.read(4096), ""):
                sink.append(chunk)
        except Exception:
            pass

    t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_chunks), daemon=True)
    t_err = threading.Thread(target=_drain, args=(proc.stderr, stderr_chunks), daemon=True)
    t_out.start()
    t_err.start()

    try:
        proc.wait(timeout=timeout)
        t_out.join(timeout=5)
        t_err.join(timeout=5)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
        )
    except KeyboardInterrupt:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        t_out.join(timeout=2)
        t_err.join(timeout=2)
        raise
    except subprocess.TimeoutExpired:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        t_out.join(timeout=2)
        t_err.join(timeout=2)
        raise
    except Exception:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        t_out.join(timeout=2)
        t_err.join(timeout=2)
        raise


# ════════════════════════════════════════════════════════════
#  FFMPEG PROGRESS TRACKER
# ════════════════════════════════════════════════════════════
def _run_ffmpeg_with_progress(
    cmd: List[str],
    cwd_path: Optional[Path],
    total_duration: float,
    label: str,
) -> subprocess.CompletedProcess:
    """Run FFmpeg with real-time progress display on stderr.
    
    Uses a reader thread so the main thread stays responsive to Ctrl+C
    (on Windows, blocking I/O in the main thread prevents KeyboardInterrupt).
    """
    start_time = time.time()
    last_pct = -1
    spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    spin_idx = 0

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=cwd_path,
        encoding="utf-8",
        errors="replace",
    )

    if proc.stderr is None:
        raise MuxError("FFmpeg stderr pipe not available")

    # Drain stderr in a background thread so the main thread can receive
    # KeyboardInterrupt on Windows (where blocking pipe reads prevent it).
    _stderr_queue: "queue.Queue[Optional[str]]" = queue.Queue()

    def _reader() -> None:
        try:
            for line in proc.stderr:  # type: ignore[union-attr]
                _stderr_queue.put(line)
        except Exception:
            pass
        finally:
            _stderr_queue.put(None)  # sentinel

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    try:
        stderr_lines: List[str] = []
        while True:
            try:
                line = _stderr_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if line is None:
                break
            line = line.strip()
            if not line:
                continue
            stderr_lines.append(line)
            current_time = 0.0
            if line.startswith("out_time="):
                current_time = _parse_ffmpeg_time(line.split("=", 1)[1])
            elif line.startswith("out_time_ms="):
                try:
                    current_time = int(line.split("=", 1)[1]) / 1_000_000
                except (ValueError, IndexError):
                    pass

            if total_duration > 0 and current_time > 0:
                pct = min(100, int(current_time / total_duration * 100))
                if pct != last_pct:
                    elapsed = time.time() - start_time
                    speed = current_time / elapsed if elapsed > 0 else 0
                    remaining = (total_duration - current_time) / speed if speed > 0 else 0
                    spin = spinner_chars[spin_idx % len(spinner_chars)]
                    spin_idx += 1
                    sys.stdout.write(
                        f"\r  {style(spin, C.CYAN)} {label} "
                        f"{style(f'{pct}%', C.GREEN + C.BOLD)} "
                        f"{style(f'({_format_eta(remaining)} left)', C.DIM)}"
                    )
                    sys.stdout.flush()
                    last_pct = pct

        proc.wait(timeout=FFMPEG_TRANSCODE_TIMEOUT)
        sys.stdout.write(f"\r{' ' * 80}\r")
        sys.stdout.flush()
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout="",
            stderr="\n".join(stderr_lines),
        )
    except KeyboardInterrupt:
        sys.stdout.write(f"\r{' ' * 80}\r")
        sys.stdout.flush()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        raise
    except subprocess.TimeoutExpired:
        sys.stdout.write(f"\r{' ' * 80}\r")
        sys.stdout.flush()
        proc.kill()
        proc.wait()
        raise
    except Exception:
        sys.stdout.write(f"\r{' ' * 80}\r")
        sys.stdout.flush()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        raise


# ════════════════════════════════════════════════════════════
#  HARDSUB HELPER
# ════════════════════════════════════════════════════════════
def run_ffmpeg_hardsub(
    media_path: Path,
    target_srt: Path,
    out_path: Path,
    cwd_path: Path,
    font_path: Optional[str] = None,
    gpu_accel: str = "auto",
    preset: str = "fast",
    tgt_lang: str = "",
    fontsize: int = 22,
    outline: int = 2,
    shadow: int = 1,
    target_res: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Execute FFmpeg hardsub command with safe path escaping.

    The ``subtitles`` filter's filename is a positional argument whose
    colon-separated option parsing cannot handle Windows drive letters
    (``C:``) even with backslash escaping.  We avoid this entirely by
    passing **only the bare filename** (no directory) into the filter
    string and relying on FFmpeg's ``cwd`` to locate the file.

    Args:
        target_res: Optional resolution to scale to, format "WxH" (e.g. "1280x720").
                    Font size is auto-adjusted proportionally if provided.
    """
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

    # Copy font file alongside SRT so libass can locate it.
    # We use fontsdir="." — a bare relative path — so the filter string
    # never contains a Windows drive-letter colon.
    font_dir = ""
    if font_path and Path(font_path).is_file():
        try:
            shutil.copy(font_path, cwd_path)
            font_dir = ":fontsdir=."
        except (OSError, PermissionError):
            # Fallback: keep font in its original location.  If that
            # directory also contains a colon (Windows drive letter) the
            # mux will still fail, but the user at least sees the error.
            font_parent = Path(font_path).parent
            if font_parent.is_dir():
                font_dir = f":fontsdir={escape_ffmpeg_filter_path(font_parent)}"
                logger.warning("Could not copy font to work dir, using original dir: %s", font_parent)
            else:
                logger.warning("Could not copy font file %s to working directory", font_path)

    try:
        encoder_args = get_encoder_args(gpu_accel, preset)
        total_duration = get_duration(media_path)

        # Auto-adjust fontsize based on target resolution
        actual_fontsize = fontsize
        if target_res:
            try:
                parts = target_res.lower().split("x")
                target_h = int(parts[1])
                src_w, src_h = get_video_resolution(media_path)
                if src_h > 0 and target_h > 0:
                    scale_factor = target_h / src_h
                    actual_fontsize = max(8, int(fontsize * scale_factor))
                    logger.info("Resolution scaling: %dx%d -> %s, font size %d -> %d",
                                src_w, src_h, target_res, fontsize, actual_fontsize)
            except (ValueError, IndexError):
                logger.warning("Invalid target_res format '%s', expected WxH", target_res)

        # Build force_style with font, bold, and RTL alignment
        force_style = _build_force_style(tgt_lang, actual_fontsize, outline, shadow)
        style_arg = f":force_style='{force_style}'" if force_style else ""

        # Guarantee removal of temporary srt using temp_file_guard
        with temp_file_guard(temp_srt_path) as guarded_tmp:
            shutil.copy(target_srt, guarded_tmp)
            # Use ONLY the bare filename — never an absolute path — so the
            # subtitles filter never sees a Windows drive-letter colon.
            escaped_srt = guarded_tmp.name

            # Build filter chain
            sub_filter = f"subtitles='{escaped_srt}'{font_dir}{style_arg}"
            if target_res:
                # Convert WxH (user input) to W:H (FFmpeg filter syntax)
                ff_scale = target_res.lower().replace("x", ":")
                vf_chain = f"scale={ff_scale},{sub_filter}"
            else:
                vf_chain = sub_filter

            cmd = [
                context.ffmpeg_cmd,
                "-y",
                "-v",
                "error",
                "-stats",
                "-stats_period",
                "0.5",
                "-i",
                abs_media,
                "-vf",
                vf_chain,
                "-c:a",
                "copy",
            ]
            # Conditionally add faststart (MP4-only, not MKV)
            if out_path.suffix.lower() == ".mp4":
                cmd.extend(["-movflags", "+faststart"])
            cmd.extend(encoder_args + [abs_out])
            logger.debug("Executing local FFmpeg hardsub: %s", " ".join(cmd))
            result = _run_ffmpeg_with_progress(cmd, cwd_path, total_duration, base)
            return result
    except (OSError, PermissionError, subprocess.TimeoutExpired) as err1:
        try:
            alt_temp = Path(tempfile.gettempdir()) / temp_srt
            with temp_file_guard(alt_temp) as guarded_alt:
                shutil.copy(target_srt, guarded_alt)
                # Same bare-filename strategy for the fallback path.
                escaped_alt = guarded_alt.name

                sub_filter = f"subtitles='{escaped_alt}'{font_dir}{style_arg}"
                if target_res:
                    ff_scale = target_res.lower().replace("x", ":")
                    vf_chain = f"scale={ff_scale},{sub_filter}"
                else:
                    vf_chain = sub_filter

                cmd = [
                    context.ffmpeg_cmd,
                    "-y",
                    "-v",
                    "error",
                    "-stats",
                    "-stats_period",
                    "0.5",
                    "-i",
                    abs_media,
                    "-vf",
                    vf_chain,
                    "-c:a",
                    "copy",
                ]
                if out_path.suffix.lower() == ".mp4":
                    cmd.extend(["-movflags", "+faststart"])
                cmd.extend(encoder_args + [abs_out])
                logger.debug("Executing alt FFmpeg hardsub: %s", " ".join(cmd))
                result = _run_ffmpeg_with_progress(cmd, cwd_path, total_duration, base)
                return result
        except (OSError, PermissionError) as err2:
            qprint(
                f"  {style('[x]', C.RED)} Severe Hardsub Error: Temporary subtitle file "
                f"could not be written."
            )
            qprint(f"      Workspace Error: {err1}")
            qprint(f"      System Temp Error: {err2}")
            return subprocess.CompletedProcess(
                args=[], returncode=1,
                stdout="", stderr=f"Hardsub SRT write failed: {err1} / {err2}"
            )


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
    
    if context.ffprobe_cmd:
        try:
            # Check for video stream (hardsub must have one)
            if hardsub:
                stream_cmd = [
                    context.ffprobe_cmd,
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=codec_type",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(p),
                ]
                stream_result = subprocess.run(stream_cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=30.0)
                if stream_result.returncode != 0 or "video" not in stream_result.stdout.lower():
                    return False, "Hardsub output missing video stream."

            cmd = [
                context.ffprobe_cmd,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(p),
            ]
            result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=30.0)
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


def verify_vmaf_quality(
    reference: Union[str, Path],
    distorted: Union[str, Path],
    threshold: float = 0.80,
) -> Tuple[bool, float, str]:
    """Run VMAF quality check between original and hardsubbed video.
    
    Returns (passed, vmaf_score, message).
    Requires FFmpeg compiled with --enable-libvmaf.
    """
    if not context.ffmpeg_cmd:
        return True, 0.0, "FFmpeg not available; skipping VMAF check."

    try:
        cmd = [
            context.ffmpeg_cmd,
            "-i", str(distorted),
            "-i", str(reference),
            "-lavfi",
            "libvmaf=log_path=-:log_fmt=json",
            "-f", "null",
            "-",
        ]
        result = _run_subprocess_interruptible(
            cmd,
            timeout=FFMPEG_TRANSCODE_TIMEOUT,
        )
        # Parse JSON from stderr (libvmaf outputs to stderr)
        vmaf_score = 0.0
        try:
            # Try to find JSON block in stderr
            stderr_text = result.stderr or ""
            json_start = stderr_text.rfind("{")
            json_end = stderr_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(stderr_text[json_start:json_end])
                vmaf_score = data.get("pooled_metrics", {}).get("vmaf", {}).get("mean", 0.0)
        except (ValueError, KeyError):
            pass

        if vmaf_score >= threshold:
            return True, vmaf_score, f"VMAF {vmaf_score:.2f} >= {threshold:.2f}"
        return False, vmaf_score, f"VMAF {vmaf_score:.2f} < {threshold:.2f} (quality degraded)"
    except FileNotFoundError:
        return True, 0.0, "FFmpeg not found; skipping VMAF."
    except subprocess.TimeoutExpired:
        return True, 0.0, "VMAF check timed out."
    except Exception as e:
        logger.debug("VMAF check failed: %s", e)
        return True, 0.0, f"VMAF check error: {e}"


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
        context.ffmpeg_cmd,
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
    result = _run_subprocess_interruptible(
        cmd,
        timeout=FFMPEG_STREAM_TIMEOUT,
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
                f"  {style('>', C.CYAN)} Transcribing ({task_type})...."
                f"{f' (attempt {attempt + 1}/{MAX_TRANSCRIPTION_RETRIES})' if attempt > 0 else ''}"
            )
            
            lang_hint = args.src_lang if args.src_lang else None
            device, compute = resolve_device_and_compute(args.device)
            
            generator = transcription_manager.transcribe(
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
                            if not context.quiet:
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
                        with open(p, "rb") as f:
                            pass
                    except IOError:
                        time.sleep(delay)
                        continue
                return True
            
            last_size = current_size
            last_mtime = current_mtime
        except (IOError, OSError) as e:
            logger.debug("File settle stat error for %s: %s", path, e)
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
    # #15: MKV stream-copy output for hardsub (avoids MP4 remux overhead)
    if args.hardsub and getattr(args, "hardsub_mkv", False):
        out_ext = "mkv"
    else:
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

    # #9: Incremental hardsub manifest check
    manifest = load_manifest(media_p.parent)
    try:
        source_mtime = media_p.stat().st_mtime
    except OSError:
        source_mtime = 0.0
    if manifest_entry_ok(manifest, base, out_path, source_mtime):
        qprint(
            f"  {style('[+]', C.GREEN)} Manifest confirms output is current. "
            f"Skipping hardsub."
        )
        status.reused_all = True
        return status, 0.0

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
            if not download_whisper_model_if_needed(args.model):
                qprint(f"  {style('[!]', C.YELLOW)} Warning: Whisper model '{args.model}' "
                       f"may need manual download.")
            
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
                            except Exception as e:
                                logger.warning("Could not copy translated SRT to source path: %s", e)

                        if src_lang_code == "und" and detected_lang != "unknown" and not native_whisper_translate:
                            final_src_path = media_p.parent / f"{base}.{detected_lang}.srt"
                            if srt_src.exists() and srt_src != final_src_path:
                                try:
                                    os.replace(str(srt_src), str(final_src_path))
                                    srt_src = final_src_path
                                    src_lang_code = detected_lang
                                    qprint(f"  {style('[~]', C.DIM)} Standardized source subtitle to language code: '{srt_src.name}'")
                                except Exception as e:
                                    logger.warning("Failed to standardize dynamic source SRT name: %s", e)
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
            and not context.is_translation_disabled()
            and not srt_tgt.exists()
            and srt_src.exists()
            and srt_src_healthy
            and (args.api_key or args.translator in ("google", "ollama"))
        ):
            if (
                context.get_consecutive_total_failures()
                >= CONSECUTIVE_TOTAL_FAIL_LIMIT
            ):
                context.set_translation_disabled(True)
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
                try:
                    fallbacks = translate_srt_native(
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
                        args_ref=args,
                    )
                except TranslationError as e:
                    qprint(f"  {style('[x]', C.RED)} Translation skipped/failed: {e}")
                else:
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
                        # #12: Smart preset selection based on duration
                        active_preset = args.preset
                        if getattr(args, "preset_auto", False) and duration > 0:
                            active_preset = select_preset_for_duration(duration, args.preset)
                            if active_preset != args.preset:
                                qprint(f"  {style('[~]', C.DIM)} Auto-preset: {active_preset} (duration: {duration:.0f}s)")
                        result = run_ffmpeg_hardsub(
                            media_p, target_srt, out_path, cwd_path,
                            font_path=args.font_path,
                            gpu_accel=args.gpu_accel,
                            preset=active_preset,
                            tgt_lang=args.tgt_lang,
                            fontsize=args.hardsub_fontsize,
                            outline=args.hardsub_outline,
                            shadow=args.hardsub_shadow,
                            target_res=getattr(args, "target_res", None),
                        )
                    else:
                        abs_media = str(media_p.resolve())
                        abs_tgt_srt = str(target_srt.resolve())
                        abs_out = str(out_path.resolve())

                        cmd = [
                            context.ffmpeg_cmd,
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
                        result = _run_subprocess_interruptible(
                            cmd,
                            timeout=FFMPEG_STREAM_TIMEOUT,
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
                            f"{out_path.parent.name}{os.sep}{out_path.name}"
                        )
                        status.muxed = True
                        # #8: VMAF quality validation for hardsub
                        if args.hardsub and getattr(args, "vmaf_check", False):
                            vmaf_ok, vmaf_score, vmaf_msg = verify_vmaf_quality(
                                media_p, out_path, threshold=0.80,
                            )
                            if not vmaf_ok:
                                qprint(f"  {style('[!]', C.YELLOW)} VMAF warning: {vmaf_msg}")
                            else:
                                logger.info("VMAF check passed: %s", vmaf_msg)
                        # #9: Update manifest after successful hardsub
                        if args.hardsub:
                            update_manifest(media_p.parent, manifest, base, out_path, source_mtime)
                    else:
                        raise MuxError(f"Mux validation failed: {reason}")

                except MuxError:
                    raise
                except Exception as e:
                    raise MuxError(f"Muxing process error: {e}") from e

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
        logger.error("Unexpected pipeline error for %s: %s", base, e, exc_info=True)
        qprint(
            f"  {style('[x]', C.RED)} Unexpected pipeline processing error: {e}"
        )
        status.error = True

    finally:
        perform_vram_gc()

    return status, time.time() - t_start


# ════════════════════════════════════════════════════════════
#  HARDSUB MANIFEST (#9: Incremental skip tracking)
# ════════════════════════════════════════════════════════════
def _manifest_path(folder: Path) -> Path:
    """Return the manifest file path for a given folder."""
    return folder / ".hardsub_manifest.json"


_manifest_locks: Dict[str, threading.Lock] = {}
_manifest_locks_guard: threading.Lock = threading.Lock()


def _manifest_lock(folder: Path) -> threading.Lock:
    """Return a per-folder lock for manifest read/write concurrency."""
    key = str(folder.resolve())
    with _manifest_locks_guard:
        if key not in _manifest_locks:
            _manifest_locks[key] = threading.Lock()
        return _manifest_locks[key]


def load_manifest(folder: Path) -> Dict[str, Any]:
    """Load hardsub manifest from disk. Returns empty dict on failure."""
    mp = _manifest_path(folder)
    if not mp.exists():
        return {}
    try:
        with open(mp, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Manifest load failed for %s: %s", mp, e)
        return {}


def save_manifest(folder: Path, manifest: Dict[str, Any]) -> None:
    """Save hardsub manifest atomically with per-folder locking."""
    lock = _manifest_lock(folder)
    with lock:
        mp = _manifest_path(folder)
        try:
            tmp = mp.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            os.replace(str(tmp), str(mp))
        except OSError as e:
            logger.debug("Manifest save failed for %s: %s", mp, e)


def manifest_entry_ok(
    manifest: Dict[str, Any],
    base: str,
    out_path: Path,
    source_mtime: float,
) -> bool:
    """Check if a manifest entry indicates the output is still valid.
    
    An entry is valid if:
    - The output file exists and is non-empty
    - The source modification time matches
    - The output path matches
    """
    entry = manifest.get(base)
    if not entry:
        return False
    try:
        if entry.get("source_mtime") != source_mtime:
            return False
        if entry.get("out_path") != str(out_path):
            return False
        cached_out = Path(entry["out_path"])
        return cached_out.exists() and cached_out.stat().st_size >= MUXED_MIN_BYTES
    except (KeyError, OSError):
        return False


def update_manifest(
    folder: Path,
    manifest: Dict[str, Any],
    base: str,
    out_path: Path,
    source_mtime: float,
) -> None:
    """Add or update a manifest entry and save."""
    manifest[base] = {
        "out_path": str(out_path),
        "source_mtime": source_mtime,
        "timestamp": time.time(),
    }
    save_manifest(folder, manifest)


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
                # Only resolve symlinks — avoid expensive resolve() on every file
                real_p = p.resolve() if p.is_symlink() else p
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
            except (OSError, RuntimeError, ValueError) as e:
                logger.debug("Skipping unreadable file %s: %s", p, e)
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
    
    with tempfile.TemporaryDirectory() as _tmp_dir:
        test_path = os.path.join(_tmp_dir, "path's with spaces", "video:sub.srt")
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        Path(test_path).touch()
        escaped = escape_ffmpeg_filter_path(test_path)
        # Verify key escaping rules are applied
        # Single-quotes inside the path are escaped via '\'' pattern
        pass_escape = "'\\''" in escaped
        # Colons are literal inside single-quotes (no backslash needed)
        colon_literal = ":" in escaped
        wrapped = escaped.startswith("'") and escaped.endswith("'")
        if pass_escape and colon_literal and wrapped:
            print(f"  {C.GREEN}[PASS]{C.RESET} FFmpeg filter path escaping")
        else:
            print(f"  {C.RED}[FAIL]{C.RESET} FFmpeg filter path escaping (checks: pass={pass_escape}, colon={colon_literal}, wrapped={wrapped})")
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

    # Validate the Ollama model resolution trace
    try:
        res_m = resolve_ollama_model("http://localhost:11434", "qwen2.5:7b")
        print(f"  {C.GREEN}[PASS]{C.RESET} resolve_ollama_model execution trace")
    except (TranslationError, OllamaConnectionError) as e:
        # Only skip when the server is unreachable — fail on actual resolution errors
        if "404" not in str(e) and "not installed" not in str(e).lower():
            print(f"  {C.DIM}[SKIP]{C.RESET} resolve_ollama_model — Ollama offline/unavailable")
        else:
            print(f"  {C.RED}[FAIL]{C.RESET} resolve_ollama_model: {e}")
            failed += 1
    except Exception as e:
        print(f"  {C.RED}[FAIL]{C.RESET} resolve_ollama_model error: {e}")
        failed += 1
        
    print(f"{C.CYAN}───────────────────────────────────────────────────────────{C.RESET}")
    if failed == 0:
        print(f"{C.GREEN}[+] All tests completed successfully!{C.RESET}\n")
        return True
    else:
        print(f"{C.RED}[x] Self-test suite failed with {failed} failures.{C.RESET}\n")
        return False


def _clone_args_for_lang(args: PipelineConfig, lang_ext: str) -> PipelineConfig:
    """Create a shallow copy of PipelineConfig with tgt_ext overridden for multi-language mode."""
    import copy
    new_args = copy.copy(args)
    new_args.tgt_ext = lang_ext
    # Resolve human-readable language name from extension for better translation quality
    new_args.tgt_lang = EXT_TO_LANG_NAME.get(lang_ext, lang_ext)
    return new_args


def _run_transcription_only(
    file_path: Path, args: PipelineConfig, file_idx: int, total_files: int
) -> None:
    """Run only the transcription step for a file (used in multi-language mode)."""
    base = file_path.stem
    # Pass None for tgt_ext since transcription doesn't need to exclude any target language
    srt_src_path, _ = find_source_srt(file_path, None, args.src_lang)

    if srt_src_path.exists():
        ok, _ = is_valid_srt(srt_src_path, 0.0, args.min_blocks, args)
        if ok:
            return  # Already transcribed

    qprint(f"  {style('>', C.CYAN)} [{file_idx}/{total_files}] Transcribing {base}...")
    # Use existing process_file with translation/embed disabled
    import copy
    transcribe_args = copy.copy(args)
    transcribe_args.translate = False
    transcribe_args.embed = False
    transcribe_args.skip_translate = True
    transcribe_args.skip_embed = True
    status, _ = process_file(file_path, transcribe_args, file_idx, total_files)
    if status.error:
        raise RuntimeError(f"Transcription failed for {base}")


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
            "  %(prog)s --headless --folder ./videos --tgt-langs en,ar,ja  # Multi-language\n"
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
    parser.add_argument("--tgt-langs", type=str, default=None, dest="tgt_langs", help="Comma-separated target languages for multi-language output (e.g., en,ar,ja)")
    parser.add_argument("--api-key", type=str, default=None, dest="api_key", help="Gemini/OpenAI/Anthropic/DeepL API key")
    parser.add_argument("--hardsub", action="store_true", help="Burn subtitles into video (hardsub)")
    parser.add_argument("--font", default=None, dest="font_path", help="Font for hardsub (default: Cairo Bold, auto-installed if missing)")
    parser.add_argument("--gpu-accel", default=None, dest="gpu_accel", choices=["auto", "nvenc", "amf", "qsv", "none"], help="GPU video encoder for hardsub (default: auto-detect)")
    parser.add_argument("--preset", default=None, dest="preset", choices=["ultrafast", "fast", "medium", "slow"], help="FFmpeg encoding preset (default: fast)")
    parser.add_argument("--hardsub-fontsize", type=int, default=None, dest="hardsub_fontsize", help="Subtitle font size (default: 22)")
    parser.add_argument("--hardsub-outline", type=int, default=None, dest="hardsub_outline", help="Subtitle outline thickness (default: 2)")
    parser.add_argument("--hardsub-shadow", type=int, default=None, dest="hardsub_shadow", help="Subtitle shadow depth (default: 1)")
    parser.add_argument("--target-res", type=str, default=None, dest="target_res", help="Scale video to WxH before hardsub (e.g., 1280x720). Subtitle font auto-adjusts.")
    parser.add_argument("--hardsub-mkv", action="store_true", dest="hardsub_mkv", help="Output MKV instead of MP4 for hardsub (stream-copy friendly)")
    parser.add_argument("--vmaf-check", action="store_true", dest="vmaf_check", help="Run VMAF quality check after hardsub (requires libvmaf in FFmpeg)")
    parser.add_argument("--preset-auto", action="store_true", dest="preset_auto", help="Auto-select preset based on video duration (ultrafast for short, medium for long)")
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

    global _headless_mode
    _headless_mode = args_parsed.headless

    atexit.register(transcription_manager.terminate)
    atexit.register(cleanup_all_temp_files)

    enable_windows_ansi()

    global logger
    logger = setup_logging(quiet=args_parsed.quiet, verbose=args_parsed.verbose)
    context.quiet = args_parsed.quiet

    context.clear_mutable_states()
    context.reset_all_counters()

    global global_cfg
    global_cfg = load_config()

    config_fields = PipelineConfig.__dataclass_fields__.keys()
    merged_dict = {}
    for key in config_fields:
        if hasattr(args_parsed, key) and getattr(args_parsed, key) is not None:
            merged_dict[key] = getattr(args_parsed, key)
            context.provenance[key] = "CLI"
        elif key in global_cfg:
            merged_dict[key] = global_cfg[key]
            context.provenance[key] = "Config File"
        else:
            merged_dict[key] = DEFAULT_CONFIG.get(key, PipelineConfig.__dataclass_fields__[key].default)
            context.provenance[key] = "Default"

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

    try:
        verify_config_status()
    except ConfigError as e:
        context.config_warning = str(e)

    if args.src_lang and not args.src_lang.strip():
        args.src_lang = None

    # Auto-install Cairo Bold font when hardsub is enabled and no font specified
    if args.hardsub and not args.font_path:
        args.font_path = ensure_font_installed()

    resolve_pipeline_steps(args)

    check_dependencies(headless=args.headless)

    if not args.headless:
        interactive_wizard(args, global_cfg)
    else:
        if context.config_warning:
            print(f"  [!] Startup error: {context.config_warning}")
        if not setup_ffmpeg():
            print(
                f"{C.RED}[!] Headless Failure: FFmpeg is missing from "
                f"systemic paths.{C.RESET}"
            )
            sys.exit(1)
        if args.hardsub and args.gpu_accel != "none":
            detected = detect_gpu_encoder()
            if detected != "none":
                logger.info("GPU encoder detected: %s", detected)
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

    # Show installed Ollama models with translation ranking when Ollama is selected
    if getattr(args, "translator", "gemini") == "ollama" and not args.quiet:
        ollama_url = getattr(args, "api_url", "") or "http://localhost:11434/api/chat"
        display_ollama_models(ollama_url)
        print()

    try:
        media_files = enumerate_media_files(args.folder, recursive=args.recursive)
    except PermissionError:
        print(
            f"{C.RED}[!] Permission denied accessing target folder: "
            f"{args.folder}{C.RESET}"
        )
        sys.exit(1)

    # Check disk space once at batch start instead of per-file
    if media_files and not args.dry_run:
        try:
            check_disk_space(Path(args.folder).resolve())
        except DiskSpaceError as e:
            print(f"\n{C.RED}[!] {e}{C.RESET}")
            sys.exit(1)

    summary: List[Tuple[str, FileStatus, float]] = []
    batch_start = time.time()

    # Multi-language support: --tgt-langs overrides --tgt-lang/--tgt-ext
    multi_langs: List[str] = []
    if getattr(args, "tgt_langs", None):
        multi_langs = [ext.strip().lower() for ext in args.tgt_langs.split(",") if ext.strip()]
        if multi_langs:
            qprint(f"\n{C.CYAN}  [*] Multi-language mode: {', '.join(multi_langs)}{C.RESET}")

    if media_files:
        if multi_langs:
            # Multi-language: transcribe once per file, then translate+embed for each language
            for i, file in enumerate(media_files, 1):
                file_path = Path(file)
                file_args = _clone_args_for_lang(args, multi_langs[0])
                try:
                    # Step 1: Transcribe to source SRT
                    _run_transcription_only(file_path, file_args, i, len(media_files))
                    # Step 2: Translate + embed for each target language
                    for lang_ext in multi_langs:
                        try:
                            lang_args = _clone_args_for_lang(args, lang_ext)
                            lang_args.transcribe = False  # Already transcribed
                            status, elapsed = process_file(file, lang_args, i, len(media_files))
                            summary.append((f"{file_path.name} [{lang_ext}]", status, elapsed))
                        except Exception as lang_err:
                            logger.error("Language %s failed for %s: %s", lang_ext, file, lang_err, exc_info=True)
                            qprint(f"{C.RED}  [x] {lang_ext} failed for {file_path.name}: {lang_err}{C.RESET}")
                            summary.append((f"{file_path.name} [{lang_ext}]", FileStatus(error=True), 0.0))
                except KeyboardInterrupt:
                    qprint(f"\n{style('[!]', C.YELLOW)} Interrupted during multi-language batch processing.")
                    summary.append((file_path.name, FileStatus(error=True), 0.0))
                    break
                except Exception as e:
                    logger.error("Processing fault on %s: %s", file, e, exc_info=True)
                    qprint(f"{C.RED}  [x] Processing fault on {file_path.name}: {e}{C.RESET}")
                    summary.append((file_path.name, FileStatus(error=True), 0.0))
        else:
            # Single-language: original flow
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
                    logger.error("Processing fault on %s: %s", file, e, exc_info=True)
                    qprint(
                        f"{C.RED}  [x] Processing fault on "
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
