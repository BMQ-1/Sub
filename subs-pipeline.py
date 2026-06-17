import os
import re
import sys
import gc
import json
import time
import uuid
import shutil
import shlex
import argparse
import logging
import platform
import tempfile
import threading
import subprocess
import urllib.request
import urllib.error
import difflib
from pathlib import Path
from typing import Any, Optional, Union, Tuple, List, Dict, Set
from dataclasses import dataclass, field

# ── Safe Terminal Encoding for Arabic/UTF-8 ─────────────
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Torch — imported once at module level ───────────────
try:
    import torch as _torch

    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None
    _TORCH_AVAILABLE = False

if platform.system() == "Windows":
    os.system("color")

# ════════════════════════════════════════════════════════════
#  ANSI COLORS  (with NO_COLOR support)
# ════════════════════════════════════════════════════════════
_NO_COLOR = os.environ.get("NO_COLOR", "").strip() not in ("", "0", "false", "False")


class C:
    """ANSI color codes with NO_COLOR environment variable support."""

    CYAN: str = "" if _NO_COLOR else "\033[96m"
    GREEN: str = "" if _NO_COLOR else "\033[92m"
    YELLOW: str = "" if _NO_COLOR else "\033[93m"
    RED: str = "" if _NO_COLOR else "\033[91m"
    RESET: str = "" if _NO_COLOR else "\033[0m"
    BOLD: str = "" if _NO_COLOR else "\033[1m"
    DIM: str = "" if _NO_COLOR else "\033[2m"


_ANSI_RE = re.compile(r"\033(?:\[[0-9;]*[A-Za-z]|\][^\007]*\007)")


def strip_ansi(s: str) -> str:
    """Remove ANSI escape codes from a string."""
    return _ANSI_RE.sub("", s)


# ════════════════════════════════════════════════════════════
#  MODULE METADATA
# ════════════════════════════════════════════════════════════
__version__ = "2.0.0"
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
OLD_CONFIG_PATH: Path = APP_DIR / "autosubs_settings.json"

MEDIA_EXTS: Tuple[str, ...] = (
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
AUDIO_EXTS: Tuple[str, ...] = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".opus")

# API Configuration
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)
API_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{20,}\b")

# VRAM Requirements (in GB)
VRAM_REQUIREMENTS: Dict[str, float] = {
    "tiny": 1.0,
    "base": 1.5,
    "small": 2.5,
    "medium": 5.0,
    "large-v3-turbo": 6.0,
    "large-v3": 10.0,
}

MODEL_MAP: Dict[str, str] = {
    "0": "tiny",
    "1": "base",
    "2": "small",
    "3": "medium",
    "4": "large-v3-turbo",
    "5": "large-v3",
}

# Timings & Chunking
GEMINI_CHUNK_SIZE: int = 80
GEMINI_TIMEOUT: int = 30
GEMINI_INTER_CHUNK_DELAY: int = 2
WATCHER_SETTLE_SECS: int = 5
MUXED_MIN_BYTES: int = 1000
FILE_SETTLE_MAX_RETRIES: int = 6
FILE_SETTLE_DELAY: float = 1.5
WHISPER_BEAM_SIZE: int = 5
WHISPER_TRANSCRIBE_TIMEOUT: int = 3600  # 1 hour max for transcription

# Audio extraction settings
AUDIO_SAMPLE_RATE: int = 16000
AUDIO_CODEC: str = "pcm_s16le"

# Batch Quota Safety Brakes
CONSECUTIVE_429_LIMIT: int = 3
CONSECUTIVE_TOTAL_FAIL_LIMIT: int = 5

# Disk space safety margin (bytes)
MIN_FREE_DISK_BYTES: int = 500 * 1024 * 1024  # 500 MB

# Retry configuration
MAX_TRANSCRIPTION_RETRIES: int = 3
TRANSCRIPTION_RETRY_BASE_DELAY: float = 2.0

# Translation retry
MAX_TRANSLATION_CHUNK_RETRIES: int = 3
TRANSLATION_RETRY_BASE_DELAY: float = 2.0

# Logging
MAX_AUDIT_LOGS: int = 30
AUDIT_LOG_MAX_AGE_DAYS: int = 14

# Validation
SRT_TIMESTAMP_PATTERN = re.compile(
    r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})"
)
TRIVIAL_WORDS: Set[str] = {
    "oh",
    "ah",
    "okay",
    "ok",
    "yeah",
    "yes",
    "no",
    "uh",
    "hmm",
    "ha",
    "hey",
    "wow",
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "schema_version": 1,
    "api_key": "",
    "tgt_lang": "English",
    "tgt_ext": "en",
    "src_lang": "",
    "min_blocks": 3,
    "model": "small",
    "skip_cleanup": False,
    "skip_migration": False,
    "explain_summary": True,
    "srt_max_avg_duration": 10.0,
    "srt_min_avg_duration": 0.1,
    "srt_dup_ratio": 0.6,
    "fallback_match_threshold": 0.95,
    "max_audit_logs": 30,
    "gemini_model": DEFAULT_GEMINI_MODEL,
}

# Select device and compute type
if _TORCH_AVAILABLE and _torch.cuda.is_available():
    DEVICE: str = "cuda"
    COMPUTE: str = "float16"
else:
    DEVICE: str = "cpu"
    COMPUTE: str = "int8"

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
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


logger: logging.Logger = setup_logging()


# ════════════════════════════════════════════════════════════
#  THREAD-SAFE CONTEXT
# ════════════════════════════════════════════════════════════
class Context:
    """Thread-safe global application context.

    All mutable shared state is protected by threading locks to prevent
    race conditions when running in watcher mode with concurrent translations.
    """

    whisper_model: Optional[Any] = None
    whisper_loaded: bool = False
    whisper_lock: threading.Lock = threading.Lock()
    ffmpeg_cmd: Optional[str] = None
    ffprobe_cmd: Optional[str] = None
    translation_disabled: bool = False
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
    def get_consecutive_429s(cls) -> int:
        with cls._state_lock:
            return cls._consecutive_429s

    @classmethod
    def increment_consecutive_429s(cls) -> None:
        with cls._state_lock:
            cls._consecutive_429s += 1

    @classmethod
    def reset_consecutive_429s(cls) -> None:
        with cls._state_lock:
            cls._consecutive_429s = 0

    @classmethod
    def get_consecutive_total_failures(cls) -> int:
        with cls._state_lock:
            return cls._consecutive_total_failures

    @classmethod
    def increment_consecutive_total_failures(cls) -> None:
        with cls._state_lock:
            cls._consecutive_total_failures += 1

    @classmethod
    def reset_consecutive_total_failures(cls) -> None:
        with cls._state_lock:
            cls._consecutive_total_failures = 0

    @classmethod
    def reset_all_counters(cls) -> None:
        with cls._state_lock:
            cls._consecutive_429s = 0
            cls._consecutive_total_failures = 0


# ════════════════════════════════════════════════════════════
#  DEPENDENCY RESOLUTION & HARDWARE CHECKING
# ════════════════════════════════════════════════════════════
REQUIRED_PACKAGES: List[str] = ["faster_whisper"]
OPTIONAL_PACKAGES: Dict[str, str] = {"watchdog": "watch mode"}


def check_dependencies(headless: bool = False) -> None:
    """Verify all required Python packages are installed.

    Args:
        headless: If True, suppress interactive prompts on failure.
    """
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
    """Locate ffmpeg and ffprobe executables in system PATH.

    Returns:
        True if both executables are found, False otherwise.
    """
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        Context.ffmpeg_cmd = ffmpeg
        Context.ffprobe_cmd = ffprobe
        return True
    return False


def get_available_vram_gb() -> float:
    """Query available GPU VRAM in gigabytes.

    Returns:
        Available VRAM in GB, or 0.0 if no GPU is available.
    """
    if _TORCH_AVAILABLE and _torch.cuda.is_available():
        try:
            t_vram = _torch.cuda.get_device_properties(0).total_memory
            a_vram = t_vram - _torch.cuda.memory_allocated(0)
            return a_vram / (1024**3)
        except Exception:
            return 4.0
    return 0.0


def recommend_whisper_model() -> str:
    """Recommend a Whisper model based on available hardware.

    Returns:
        Model selection key ("0" through "5").
    """
    if DEVICE == "cpu":
        return "1"
    vram = get_available_vram_gb()
    if vram >= 10.0:
        return "5"
    elif vram >= 6.0:
        return "4"
    elif vram >= 5.0:
        return "3"
    elif vram >= 2.5:
        return "2"
    elif vram >= 1.5:
        return "1"
    return "0"


def check_model_exists(model_name: str) -> bool:
    """Check if a Whisper model has already been downloaded locally.

    Args:
        model_name: The model size identifier (e.g., "small", "medium").

    Returns:
        True if the model snapshot exists in the HuggingFace cache.
    """
    try:
        from faster_whisper.utils import download_model

        download_model(model_name)
        return True
    except Exception:
        return False


def load_whisper(model_name: str) -> None:
    """Load the Whisper model into memory with thread-safe locking.

    Args:
        model_name: The model size identifier to load.

    Raises:
        RuntimeError: If the model fails to load.
    """
    if Context.whisper_loaded:
        return
    with Context.whisper_lock:
        if Context.whisper_loaded:
            return
        try:
            from faster_whisper import WhisperModel

            print(
                f"  {C.CYAN}> Loading Whisper model '{model_name}' on "
                f"{DEVICE} ({COMPUTE})...{C.RESET}"
            )
            Context.whisper_model = WhisperModel(
                model_name, device=DEVICE, compute_type=COMPUTE
            )
            Context.whisper_loaded = True
        except Exception as e:
            logger.error("Failed to load Whisper model: %s", e)
            raise RuntimeError(f"Failed to load Whisper model '{model_name}': {e}") from e


# ════════════════════════════════════════════════════════════
#  TEMP FILE MANAGER
# ════════════════════════════════════════════════════════════
def register_temp_file(path: Union[str, Path]) -> None:
    """Register a temporary file for later cleanup.

    Args:
        path: Path to the temporary file.
    """
    with Context.temp_lock:
        Context.active_temp_files.add(Path(path).resolve())


def unregister_temp_file(path: Union[str, Path]) -> None:
    """Unregister a temporary file from cleanup tracking.

    Args:
        path: Path to the temporary file.
    """
    with Context.temp_lock:
        p = Path(path).resolve()
        Context.active_temp_files.discard(p)


def cleanup_all_temp_files() -> None:
    """Remove all registered temporary files with race-condition safety.

    Uses missing_ok=True to avoid TOCTOU race conditions.
    """
    with Context.temp_lock:
        for p in list(Context.active_temp_files):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
            Context.active_temp_files.discard(p)


# ════════════════════════════════════════════════════════════
#  GARBAGE COLLECTION
# ════════════════════════════════════════════════════════════
def startup_garbage_collection(
    folder_path: Union[str, Path], skip_cleanup: bool = False
) -> None:
    """Clean up stale temporary files from previous runs.

    Args:
        folder_path: The media folder to scan for stale files.
        skip_cleanup: If True, skip cleanup entirely.
    """
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
                        item.name.startswith("temp_hardsub_")
                        or "temp_hardsub_" in item.name
                    ) and item.suffix.lower() == ".srt"
                    is_stale_audio = (
                        item.name.startswith("temp_") and item.name.endswith("_audio.wav")
                    )
                    if is_stale_hardsub or is_stale_audio:
                        try:
                            item.unlink()
                            cleaned_count += 1
                        except OSError:
                            pass
        except PermissionError:
            pass
    if cleaned_count > 0:
        print(
            f"  {C.DIM}[~] Swept workspaces: Purged {cleaned_count} stale temp "
            f"file(s).{C.RESET}"
        )


# ════════════════════════════════════════════════════════════
#  DISK SPACE CHECK
# ════════════════════════════════════════════════════════════
def check_disk_space(path: Union[str, Path], required_bytes: int = MIN_FREE_DISK_BYTES) -> bool:
    """Verify sufficient disk space is available at the given path.

    Args:
        path: The path to check (uses parent directory if a file).
        required_bytes: Minimum required free space in bytes.

    Returns:
        True if sufficient space is available.
    """
    try:
        p = Path(path)
        target = p if p.is_dir() else p.parent
        stat = os.statvfs(target) if hasattr(os, "statvfs") else None
        if stat:
            free = stat.f_frsize * stat.f_bavail
            if free < required_bytes:
                print(
                    f"{C.YELLOW}[!] Low disk space: {free // (1024**2)} MB available, "
                    f"{required_bytes // (1024**2)} MB recommended.{C.RESET}"
                )
                return False
        return True
    except Exception:
        return True  # Assume OK if we can't check


# ════════════════════════════════════════════════════════════
#  SAFE EXIT
# ════════════════════════════════════════════════════════════
def exit_app(code: int = 0) -> None:
    """Perform clean shutdown with temp file cleanup.

    Args:
        code: The exit status code to return to the shell.
    """
    cleanup_all_temp_files()
    if Context.failed_cleanups:
        print(
            f"\n{C.DIM}  [~] System cleanup complete. Some active lock files were "
            f"bypassed:{C.RESET}"
        )
        for item in set(Context.failed_cleanups):
            print(f"      · {item}")
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
    """Verify configuration directory and file are accessible.

    Returns:
        True if everything is OK, or a descriptive error string.
    """
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
    """Sanity-check loaded configuration values, enforce ranges, maintain schema versioning.

    Args:
        cfg_dict: The loaded configuration dictionary.

    Returns:
        The validated and potentially modified configuration dictionary.
    """
    if "schema_version" not in cfg_dict:
        cfg_dict["schema_version"] = DEFAULT_CONFIG["schema_version"]
    elif cfg_dict["schema_version"] < DEFAULT_CONFIG["schema_version"]:
        print(
            f"  {C.YELLOW}[~] Legacy config schema version "
            f"{cfg_dict.get('schema_version')} detected. Updating to version "
            f"{DEFAULT_CONFIG['schema_version']}.{C.RESET}"
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
            if default_val is not None and type(cfg_dict[key]) is not type(default_val):
                try:
                    if isinstance(default_val, float):
                        cfg_dict[key] = float(cfg_dict[key])
                    elif isinstance(default_val, int):
                        cfg_dict[key] = int(cfg_dict[key])
                    elif isinstance(default_val, bool):
                        cfg_dict[key] = bool(cfg_dict[key])
                except Exception:
                    cfg_dict[key] = default_val

            if key in range_rules:
                min_val, max_val, fallback = range_rules[key]
                if cfg_dict[key] < min_val or cfg_dict[key] > max_val:
                    print(
                        f"  {C.YELLOW}[!] Config key '{key}' out of range "
                        f"[{min_val} - {max_val}]. Resetting to default '{fallback}'.{C.RESET}"
                    )
                    cfg_dict[key] = fallback
        else:
            cfg_dict[key] = default_val
    return cfg_dict


def load_config() -> Dict[str, Any]:
    """Load configuration from disk with migration support.

    Returns:
        The loaded and validated configuration dictionary.
    """
    if not CONFIG_PATH.exists() and OLD_CONFIG_PATH.exists():
        try:
            shutil.copy(OLD_CONFIG_PATH, CONFIG_PATH)
            Context.migration_status = "migrated"
        except Exception:
            pass
    elif CONFIG_PATH.exists():
        Context.migration_status = "loaded"

    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                cfg.update(loaded)
        except Exception:
            pass
    return validate_schema(cfg)


def save_config(conf: Dict[str, Any]) -> None:
    """Atomically save configuration to disk.

    Uses write-to-temp-then-rename pattern for atomicity on all platforms.
    On Windows, uses os.replace for proper atomic replacement.

    Args:
        conf: The configuration dictionary to save.
    """
    tmp = CONFIG_PATH.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(conf, f, indent=4, ensure_ascii=False)
        # Validate written JSON before replacing
        with open(tmp, "r", encoding="utf-8") as f:
            json.load(f)
        # Atomic replace (works on Windows and Unix)
        if sys.platform == "win32":
            if CONFIG_PATH.exists():
                CONFIG_PATH.unlink()
            os.replace(str(tmp), str(CONFIG_PATH))
        else:
            tmp.replace(CONFIG_PATH)
    except Exception as e:
        print(f"\n  {C.YELLOW}[!] Config save failed: {e}{C.RESET}")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


# Load global configuration
cfg: Dict[str, Any] = load_config()


# ════════════════════════════════════════════════════════════
#  UTILITIES & HEURISTIC FALLBACK DETECTION
# ════════════════════════════════════════════════════════════
def natural_keys(text: Union[str, Path]) -> List[Union[int, str]]:
    """Split text into natural sort key components (numbers as int, text lowercase).

    Args:
        text: The string or Path to create sort keys for.

    Returns:
        A list of integers and lowercase strings for natural sorting.
    """
    return [
        int(c) if c.isdigit() else c.lower()
        for c in re.split(r"(\d+)", str(text))
    ]


def fmt_time(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like "1h 23m 45s" or "<1s".
    """
    if seconds <= 0:
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
    """Format a timestamp in seconds to SRT format (HH:MM:SS,mmm).

    Args:
        t: Time in seconds.

    Returns:
        SRT-formatted timestamp string.
    """
    t = max(0.0, float(t))
    ms = round(t * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def get_duration(media_path: Union[str, Path]) -> float:
    """Get the duration of a media file using ffprobe.

    Args:
        media_path: Path to the media file.

    Returns:
        Duration in seconds, or 0.0 if unavailable.
    """
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
        ).strip()
        val = float(result)
        return val if val > 0 else 0.0
    except Exception:
        return 0.0


def safe_remove(path: Union[str, Path]) -> None:
    """Safely remove a file, unregistering from temp tracking.

    Args:
        path: Path to the file to remove.
    """
    try:
        p = Path(path)
        p.unlink(missing_ok=True)
        unregister_temp_file(p)
    except OSError as e:
        unregister_temp_file(path)
        Context.failed_cleanups.append(f"{path} ({e.strerror or str(e)})")


def perform_vram_gc() -> None:
    """Free GPU VRAM by clearing CUDA cache and running garbage collection."""
    if not Context.whisper_loaded:
        return
    if _TORCH_AVAILABLE:
        try:
            if _torch.cuda.is_available():
                gc.collect()
                _torch.cuda.empty_cache()
        except Exception:
            pass


def normalize_dialogue(text: Optional[str]) -> str:
    """Normalize subtitle dialogue for comparison by removing markup and trivial words.

    Args:
        text: The dialogue text to normalize.

    Returns:
        Lowercased, stripped text with formatting markers removed.
    """
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"\[[^\]]*\]", "", t)
    t = re.sub(r"\([^\)]*\)", "", t)
    t = re.sub(r"<[^>]*>", "", t)
    t = re.sub(r"[^\w\s]", "", t)
    words = [w for w in t.split() if w not in TRIVIAL_WORDS]
    return " ".join(words)


def parse_srt_dialogue(path: Union[str, Path]) -> List[str]:
    """Parse dialogue lines from an SRT file.

    Args:
        path: Path to the SRT file.

    Returns:
        A list of dialogue strings, one per subtitle block.
    """
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


def detect_fallbacks(
    src_path: Union[str, Path],
    tgt_path: Union[str, Path],
    args_ref: argparse.Namespace,
) -> Tuple[int, str]:
    """Detect translation fallback blocks where target matches source.

    Args:
        src_path: Path to source SRT.
        tgt_path: Path to target (translated) SRT.
        args_ref: Namespace containing fallback_match_threshold.

    Returns:
        Tuple of (match_count, description_string).
    """
    src_dialogues = parse_srt_dialogue(src_path)
    tgt_dialogues = parse_srt_dialogue(tgt_path)
    if not src_dialogues or not tgt_dialogues:
        return 0, "No blocks parsed"

    threshold: float = getattr(args_ref, "fallback_match_threshold", 0.95)
    match_count = 0
    for idx, src_txt in enumerate(src_dialogues):
        if idx >= len(tgt_dialogues):
            break
        tgt_txt = tgt_dialogues[idx]
        s_clean = src_txt.replace("\u266a", "").strip()
        t_clean = tgt_txt.replace("\u266a", "").strip()

        s_norm = normalize_dialogue(s_clean)
        t_norm = normalize_dialogue(t_clean)

        if len(s_norm) < 4 or len(t_norm) < 4:
            continue

        if s_norm == t_norm:
            match_count += 1
        else:
            ratio = difflib.SequenceMatcher(None, s_norm, t_norm).ratio()
            if ratio >= threshold:
                match_count += 1

    return match_count, f"{match_count} blocks matching source"


# ════════════════════════════════════════════════════════════
#  SRT HEALTH CHECK
# ════════════════════════════════════════════════════════════
def is_valid_srt(
    srt_path: Union[str, Path],
    media_duration: float = 0.0,
    min_blocks: int = 3,
    args_ref: Optional[argparse.Namespace] = None,
) -> Tuple[bool, str]:
    """Validate an SRT file for structural integrity and quality.

    Checks block count, timestamp validity, average duration, and duplicate ratio.
    Also validates sequential block numbering.

    Args:
        srt_path: Path to the SRT file.
        media_duration: Duration of the associated media for context.
        min_blocks: Minimum number of blocks required.
        args_ref: Namespace containing validation thresholds.

    Returns:
        Tuple of (is_valid, reason_string).
    """
    if args_ref is None:
        args_ref = argparse.Namespace(**DEFAULT_CONFIG)

    max_duration: float = getattr(args_ref, "srt_max_avg_duration", 10.0)
    min_duration: float = getattr(args_ref, "srt_min_avg_duration", 0.1)
    dup_threshold: float = getattr(args_ref, "srt_dup_ratio", 0.6)

    try:
        text = Path(srt_path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return False, f"Cannot read file: {e}"

    blocks = [b.strip() for b in re.split(r"\n\n+", text.strip()) if b.strip()]
    if len(blocks) < min_blocks:
        return (
            False,
            f"Only {len(blocks)} block(s) -- minimum health threshold is {min_blocks} block(s).",
        )

    def ts_to_sec(ts: str) -> Optional[float]:
        """Parse SRT timestamp string to seconds."""
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

    for block in blocks:
        match = SRT_TIMESTAMP_PATTERN.search(block)
        if match:
            t_start = ts_to_sec(match.group(1))
            t_end = ts_to_sec(match.group(2))
            if t_start is not None and t_end is not None:
                durations.append(max(0.0, t_end - t_start))
        block_lines = block.splitlines()
        # Validate block numbering
        if block_lines and block_lines[0].strip().isdigit():
            block_numbers.append(int(block_lines[0].strip()))
        for ln in block_lines[2:]:
            stripped = ln.strip()
            if stripped:
                lines.append(stripped.lower())

    # Check sequential block numbering
    if block_numbers:
        expected = list(range(1, len(block_numbers) + 1))
        if block_numbers != expected:
            return False, f"Non-sequential block numbering detected"

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
        if dup > dup_threshold:
            return (
                False,
                f"Duplicate line ratio {dup * 100:.0f}% > {dup_threshold * 100:.0f}% "
                f"-- looping hallucination.",
            )

    return True, "OK"


# ════════════════════════════════════════════════════════════
#  API KEY VALIDATION
# ════════════════════════════════════════════════════════════
def validate_api_key(key: str) -> Tuple[bool, str]:
    """Validate a Gemini API key format.

    Args:
        key: The API key string to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not key or not key.strip():
        return False, "API key is empty"
    if len(key) < 20:
        return False, f"API key too short ({len(key)} chars, minimum 20)"
    if not API_KEY_PATTERN.match(key):
        return False, "API key contains invalid characters"
    return True, ""


def validate_tgt_ext(ext: str) -> Tuple[bool, str]:
    """Validate a target language extension string.

    Args:
        ext: The extension string (e.g., 'en', 'ar').

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not ext or not ext.strip():
        return False, "Extension is empty"
    clean = ext.strip().lower()
    if not clean.isalpha():
        return False, f"Extension must contain only alphabetic characters, got: {clean}"
    if len(clean) < 1 or len(clean) > 10:
        return False, f"Extension length must be 1-10 characters, got: {len(clean)}"
    return True, ""


# ════════════════════════════════════════════════════════════
#  VALIDATION & AUDITING
# ════════════════════════════════════════════════════════════
def validate_args(args: argparse.Namespace) -> None:
    """Resolve configuration conflicts and apply automatic overrides.

    Args:
        args: The parsed argument namespace to validate and modify in-place.
    """
    adjustments: List[str] = []
    if args.hardsub and not args.embed:
        adjustments.append(
            "Hardsub is enabled but Muxing is disabled. "
            "(Burning subtitles requires muxing; auto-enabling Mux.)"
        )
        args.embed = True
        Context.provenance["embed"] = "Auto-Override"

    if args.translate and not args.api_key:
        adjustments.append(
            "Translation is requested, but no Gemini API key was configured. "
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
        print(
            f"\n{C.RED}{C.BOLD}[!] Configuration Conflicts Resolved "
            f"(Overriding Variables):{C.RESET}"
        )
        for msg in adjustments:
            print(f"    · {msg}")
        print()


def should_mask(key_name: str) -> bool:
    """Determine if a config key should be masked in logs.

    Args:
        key_name: The configuration key name.

    Returns:
        True if the key contains sensitive data.
    """
    k = key_name.lower()
    return any(word in k for word in ["key", "token", "secret", "password", "api"])


def write_audit_log(
    args: argparse.Namespace, summary: List[Tuple[str, Dict[str, Any], float]], total_elapsed: float
) -> None:
    """Write a structured JSON audit log of the pipeline run.

    API keys are NEVER logged -- only their presence/absence is recorded.

    Args:
        args: The effective pipeline arguments.
        summary: List of (filename, status_dict, elapsed_time) tuples.
        total_elapsed: Total batch processing time in seconds.
    """
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
            if should_mask(k):
                # Never log API keys, even masked -- only presence/absence
                val = "[PRESENT]" if val else "[ABSENT]"
            serializable_args[k] = val

        log_data = {
            "timestamp": timestamp,
            "total_elapsed_seconds": total_elapsed,
            "configuration": serializable_args,
            "processed_files": [
                {
                    "file_name": name,
                    "pipeline_status": status,
                    "elapsed_seconds": elapsed,
                }
                for name, status, elapsed in summary
            ],
        }
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)

        print(f"  {C.GREEN}[+] Audit log recorded to: {log_file}{C.RESET}")

        # Rotate old logs
        max_logs = getattr(args, "max_audit_logs", MAX_AUDIT_LOGS)
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
        while len(log_files) > max_logs:
            oldest = log_files.pop(0)
            try:
                oldest.unlink(missing_ok=True)
            except OSError:
                pass
    except Exception:
        pass


def print_effective_settings(args: argparse.Namespace) -> None:
    """Display the effective pipeline configuration to the user.

    Args:
        args: The effective pipeline arguments.
    """

    def get_src(key_name: str) -> str:
        src = Context.provenance.get(key_name, "Default")
        return f"[{src}]"

    print(
        f"\n{C.BOLD}── Active Pipeline Parameters "
        f"────────────────────────────{C.RESET}"
    )
    print(
        f"  Target Folder:     {C.CYAN}{args.folder:<40}{C.RESET} "
        f"{C.DIM}{get_src('folder')}{C.RESET}"
    )
    print(
        f"  Transcription:     "
        f"{C.CYAN}{'Enabled' if args.transcribe else 'Disabled':<40}{C.RESET} "
        f"{C.DIM}{get_src('model')} Model: {args.model.upper()}{C.RESET}"
    )
    print(
        f"  Translation:       "
        f"{C.CYAN}{'Enabled' if args.translate else 'Disabled':<40}{C.RESET} "
        f"{C.DIM}{get_src('tgt_lang')} Target: {args.tgt_lang} "
        f"[.{args.tgt_ext}]{C.RESET}"
    )
    hardsub_label = "Hardsub (Burn-in)" if args.hardsub else (
        "Softsub (Mux)" if args.embed else "Disabled"
    )
    embed_src = get_src("hardsub") if args.hardsub else get_src("embed")
    print(
        f"  Final Muxing:      {C.CYAN}{hardsub_label:<40}{C.RESET} "
        f"{C.DIM}{embed_src}{C.RESET}"
    )
    print(
        f"  Watch Mode:        "
        f"{C.CYAN}{'Enabled' if args.watch else 'Disabled':<40}{C.RESET} "
        f"{C.DIM}[CLI]{C.RESET}"
    )
    print(
        f"  Dry Run Mode:      "
        f"{C.CYAN}{'Active' if args.dry_run else 'Inactive':<40}{C.RESET} "
        f"{C.DIM}[CLI]{C.RESET}"
    )
    print(
        f"  Audit Logs:        "
        f"{C.CYAN}{'Disabled' if args.no_audit else 'Enabled':<40}{C.RESET} "
        f"{C.DIM}[CLI]{C.RESET}"
    )
    print(
        f"  Min Blocks Req:    "
        f"{C.CYAN}{args.min_blocks:<40}{C.RESET} "
        f"{C.DIM}{get_src('min_blocks')}{C.RESET}"
    )
    print(f"{C.BOLD}───────────────────────────────────────────────────────────{C.RESET}\n")


def wait_for_file_settle(
    path: Union[str, Path],
    max_retries: int = FILE_SETTLE_MAX_RETRIES,
    delay: float = FILE_SETTLE_DELAY,
) -> bool:
    """Wait for a file to stabilize (stop changing size).

    Properly closes file handles to prevent resource leaks.

    Args:
        path: Path to the file to monitor.
        max_retries: Maximum number of size-check iterations.
        delay: Seconds to wait between checks.

    Returns:
        True if the file stabilized, False otherwise.
    """
    p = Path(path)
    if not p.exists():
        return False

    last_size = -1
    for _ in range(max_retries):
        try:
            current_size = p.stat().st_size
            if current_size == last_size and current_size > 0:
                # Verify file is readable (not locked)
                try:
                    with open(p, "rb") as f:
                        f.read(1024)
                except IOError:
                    time.sleep(delay)
                    continue
                return True
            last_size = current_size
        except (IOError, OSError):
            pass
        time.sleep(delay)
    return False


# ════════════════════════════════════════════════════════════
#  BATCH SUMMARY
# ════════════════════════════════════════════════════════════
def print_summary(
    summary: List[Tuple[str, Dict[str, Any], float]],
    total_elapsed: float,
    args: argparse.Namespace,
) -> None:
    """Print a formatted batch completion summary table.

    Args:
        summary: List of (filename, status_dict, elapsed_time) tuples.
        total_elapsed: Total batch processing time in seconds.
        args: Pipeline arguments controlling output verbosity.
    """
    W = 72
    top = "\u2554" + "\u2550" * W + "\u2557"
    mid = "\u2560" + "\u2550" * W + "\u2563"
    bot = "\u255a" + "\u2550" * W + "\u255d"

    print(f"\n{C.BOLD}{C.CYAN}{top}")
    title = f"  Batch Complete -- {fmt_time(total_elapsed)}"
    print(f"\u2551{title:<{W}}\u2551")
    print(f"{mid}{C.RESET}")

    def pad_flag(label: str, color: str, num: Optional[int] = None) -> str:
        if num is not None:
            num_str = "999+" if num > 999 else str(num)
            content = f"{label:<7}{num_str:>3}"
        else:
            content = f"{label:<10}"
        return f"{color}[{content}]{C.RESET}"

    for name, st, t in summary:
        t_str = fmt_time(t).rjust(6)

        if st.get("error"):
            flag = pad_flag("FAULT", C.RED)
        elif st.get("audio_failed"):
            flag = pad_flag("AUDIO_FAIL", C.RED)
        elif st.get("skipped"):
            flag = pad_flag("SKIPPED", C.YELLOW)
        elif st.get("reused_all"):
            flag = pad_flag("REUSED", C.GREEN)
        elif st.get("translated") and st.get("muxed"):
            flag = pad_flag("DONE_TRN", C.GREEN)
        elif st.get("mixed_language") and st.get("muxed"):
            flag = pad_flag("MIXED", C.YELLOW, num=st.get("fallback_count", 0))
        elif st.get("reused_srt") and st.get("muxed"):
            flag = pad_flag("DONE_MUX", C.GREEN)
        elif st.get("transcribed") and (st.get("translated") or st.get("mixed_language")):
            if st.get("mixed_language"):
                flag = pad_flag("TXT+MIX", C.CYAN, num=st.get("fallback_count", 0))
            else:
                flag = pad_flag("TXT+TRN", C.CYAN)
        elif st.get("transcribed"):
            flag = pad_flag("TXT_ONLY", C.CYAN)
        elif st.get("translated") or st.get("reused_srt") or st.get("mixed_language"):
            if st.get("mixed_language"):
                flag = pad_flag("MIX_TRN", C.CYAN, num=st.get("fallback_count", 0))
            else:
                flag = pad_flag("TRN_ONLY", C.CYAN)
        else:
            flag = pad_flag("NO-OP", C.DIM)

        flag_raw = strip_ansi(flag)
        name_width = max(20, W - 2 - len(flag_raw) - 1 - len(t_str) - 1)
        if len(name) > name_width:
            name_trunc = name[: name_width - 1] + "~"
        else:
            name_trunc = name.ljust(name_width)

        print(f"\u2551  {flag} {name_trunc} {t_str} \u2551")

    print(f"{C.BOLD}{C.CYAN}{bot}{C.RESET}")

    if getattr(args, "verbose_summary", False):
        print(
            f"\n{C.BOLD}── Verbose Execution Details "
            f"─────────────────────────────{C.RESET}"
        )
        for name, st, t in summary:
            details: List[str] = []
            if st.get("error"):
                details.append("Execution encountered system errors.")
            if st.get("audio_failed"):
                details.append("Audio preprocessing/extraction failed completely.")
            if st.get("skipped"):
                details.append("Health check failed; file skipped.")
            if st.get("reused_all"):
                details.append("Output file already exists; skipped rerun.")
            if st.get("transcribed"):
                details.append("Transcribed audio using local Whisper engine.")
            if st.get("translated"):
                details.append("Translated subtitles using translation API.")
            if st.get("mixed_language"):
                details.append(
                    f"Completed with {st.get('fallback_count', 0)} likely fallback "
                    f"blocks matching original dialogue."
                )
            if st.get("reused_srt"):
                details.append("Reused existing source subtitles.")
            if st.get("muxed"):
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
#  DIRECTORY WATCHER
# ════════════════════════════════════════════════════════════
def run_watcher(args: argparse.Namespace) -> None:
    """Run a filesystem watcher for automatic processing of new media files.

    Uses watchdog to monitor the target directory and trigger processing
    for new or moved media files. Supports on_created and on_moved events.

    Args:
        args: Pipeline configuration arguments.
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print(f"\n{C.YELLOW}  [!] watchdog not installed: pip install watchdog{C.RESET}")
        return

    in_flight: Set[str] = set()
    lock = threading.Lock()
    muxed_folder = f"muxed_{args.tgt_ext}"
    folder_root = Path(args.folder).resolve()

    class WatchHandler(FileSystemEventHandler):
        """Handle filesystem events for the media directory."""

        def _should_process(self, path: str) -> bool:
            """Check if a file path should trigger processing."""
            resolved = str(Path(path).resolve())
            if not resolved.lower().endswith(MEDIA_EXTS):
                return False
            try:
                rel = Path(resolved).relative_to(folder_root)
                if rel.parts and rel.parts[0] == muxed_folder:
                    return False
            except ValueError:
                return False
            return True

        def _dispatch(self, path: str) -> None:
            """Process a single file event."""
            if not self._should_process(path):
                return

            resolved = str(Path(path).resolve())
            with lock:
                if resolved in in_flight:
                    return
                in_flight.add(resolved)

            def handle() -> None:
                try:
                    name = Path(resolved).name
                    clamped_name = (name[:35] + "...") if len(name) > 38 else name
                    print(f"\n{C.CYAN}  [*] Detected: {clamped_name}{C.RESET}")
                    if not wait_for_file_settle(resolved):
                        print(
                            f"  {C.YELLOW}[!] Warning: File {clamped_name} is "
                            f"locked/busy. Skipping watch thread execution.{C.RESET}"
                        )
                        return
                    time.sleep(WATCHER_SETTLE_SECS)
                    process_file(resolved, args)
                    print(f"{C.GREEN}  [*] Idle -- awaiting files...{C.RESET}")
                except KeyboardInterrupt:
                    print(f"\n{C.YELLOW}[!] Watch thread interrupted.{C.RESET}")
                except Exception as e:
                    print(f"{C.RED}[x] Watch processing error: {e}{C.RESET}")
                finally:
                    with lock:
                        in_flight.discard(resolved)

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
    observer.schedule(WatchHandler(), path=args.folder, recursive=False)
    observer.start()

    print(f"\n{C.CYAN}  [*] Watch Mode -- '{Path(args.folder).name}'{C.RESET}")
    print(f"  {C.DIM}Ctrl+C to stop.{C.RESET}")

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
    """Token bucket rate limiter for proactive API throttling.

    Prevents hammering the API by enforcing a minimum interval between requests.
    """

    def __init__(self, rate: float = 1.0, capacity: int = 2) -> None:
        """Initialize the rate limiter.

        Args:
            rate: Tokens added per second (requests/sec allowed).
            capacity: Maximum burst capacity.
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens: float = float(capacity)
        self.last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """Acquire a token, blocking if necessary.

        Args:
            blocking: If True, block until a token is available.
            timeout: Maximum seconds to wait if blocking.

        Returns:
            True if a token was acquired, False otherwise.
        """
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

            time.sleep(0.05)


# Global rate limiter: max 1 request per second with burst of 2
global_rate_limiter = TokenBucketRateLimiter(rate=1.0, capacity=2)


# ════════════════════════════════════════════════════════════
#  TRANSLATION MOTOR (Streaming)
# ════════════════════════════════════════════════════════════
def translate_srt_native(
    srt_src: Union[str, Path],
    srt_tgt: Union[str, Path],
    tgt_lang: str,
    api_key: str,
    gemini_model: Optional[str] = None,
) -> Tuple[bool, str, int]:
    """Translate an SRT file using the Gemini API with streaming output.

    Uses chunked processing with proactive rate limiting, retry logic,
    and streams results to disk to control memory usage. Handles
    KeyboardInterrupt gracefully by closing output files.

    Args:
        srt_src: Path to the source SRT file.
        srt_tgt: Path for the translated output SRT file.
        tgt_lang: Target language name (e.g., "English", "Arabic").
        api_key: Gemini API key.
        gemini_model: Gemini model name (defaults to gemini-1.5-flash).

    Returns:
        Tuple of (success, message, fallback_count).
    """
    srt_src = Path(srt_src)
    srt_tgt = Path(srt_tgt)

    # Validate SRT before translation
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

    # Validate API key format
    key_valid, key_err = validate_api_key(api_key)
    if not key_valid:
        return False, f"Invalid API key: {key_err}", 0

    model = gemini_model or DEFAULT_GEMINI_MODEL
    url = GEMINI_URL_TEMPLATE.format(model=model, key=api_key)

    chunk_size = GEMINI_CHUNK_SIZE
    total_chunks = (len(blocks) + chunk_size - 1) // chunk_size

    # Use streaming output to control memory
    tmp_tgt = srt_tgt.with_suffix(".tmp")
    tmp_fh = None
    translated_count = 0

    try:
        tmp_fh = open(tmp_tgt, "w", encoding="utf-8")

        for i in range(0, len(blocks), chunk_size):
            # Check for interrupt
            if threading.current_thread().is_alive():
                pass  # Normal operation

            chunk = blocks[i : i + chunk_size]
            chunk_idx = (i // chunk_size) + 1

            # Progress indicator
            print(
                f"  {C.DIM}[~] Translating chunk {chunk_idx}/{total_chunks} "
                f"({len(chunk)} blocks)...{C.RESET}"
            )

            prompt_lines: List[str] = []
            for b in chunk:
                lines = b.splitlines()
                idx = lines[0] if lines else str(len(prompt_lines) + 1)
                diag = "\n".join(lines[2:]) if len(lines) >= 3 else ""
                prompt_lines.append(f"Block #{idx}:\n{diag}")

            instruction = (
                f"Translate the following dialogue blocks to {tgt_lang}. "
                "Maintain the block IDs exactly. Preserve styling tags like <i> and </i>. "
                "Do not translate names or sound cues in brackets if they should remain native, "
                "but convert everything else naturally. Output ONLY translated blocks with matching IDs, "
                "with no extra conversational intro or outro text. Format exactly like "
                "'Block #[ID]: [translation]'\n\n"
            )
            prompt = instruction + "\n\n".join(prompt_lines)

            data = {"contents": [{"parts": [{"text": prompt}]}]}

            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            # Proactive rate limiting
            global_rate_limiter.acquire(timeout=30)

            response_text = ""
            success = False
            for attempt in range(MAX_TRANSLATION_CHUNK_RETRIES):
                try:
                    with urllib.request.urlopen(req, timeout=GEMINI_TIMEOUT) as response:
                        resp_data = json.loads(response.read().decode("utf-8"))
                        candidates = resp_data.get("candidates", [])
                        if candidates:
                            parts_list = candidates[0].get("content", {}).get("parts", [])
                            if parts_list:
                                response_text = parts_list[0].get("text", "")
                                success = True
                                Context.reset_consecutive_429s()
                                Context.reset_consecutive_total_failures()
                                break
                        else:
                            # Empty response -- log warning
                            logger.warning("Empty translation response for chunk %d", chunk_idx)
                            if attempt < MAX_TRANSLATION_CHUNK_RETRIES - 1:
                                time.sleep(TRANSLATION_RETRY_BASE_DELAY * (2**attempt))
                                continue
                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        Context.increment_consecutive_429s()
                        if Context.get_consecutive_429s() >= CONSECUTIVE_429_LIMIT:
                            return (
                                False,
                                "Rate limits (429) hit consecutively. "
                                "Suspending API translation.",
                                0,
                            )
                        time.sleep(5 * attempt + 5)
                    else:
                        Context.increment_consecutive_total_failures()
                        time.sleep(TRANSLATION_RETRY_BASE_DELAY * (2**attempt))
                except Exception:
                    Context.increment_consecutive_total_failures()
                    time.sleep(TRANSLATION_RETRY_BASE_DELAY * (2**attempt))

            if not success:
                return False, f"API chunk request {chunk_idx}/{total_chunks} failed.", 0

            # Handle empty response
            if not response_text or not response_text.strip():
                logger.warning(
                    "Empty response text for chunk %d/%d, using original dialogue",
                    chunk_idx,
                    total_chunks,
                )

            parsed_translations: Dict[int, str] = {}
            if response_text.strip():
                for part in re.split(r"Block\s*#", response_text):
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

            for idx_in_chunk, b in enumerate(chunk):
                lines = b.splitlines()
                if not lines:
                    continue
                b_idx = int(lines[0]) if lines[0].isdigit() else (i + idx_in_chunk + 1)
                ts = (
                    lines[1]
                    if len(lines) >= 2
                    else "00:00:00,000 --> 00:00:00,000"
                )

                orig_diag = "\n".join(lines[2:]) if len(lines) >= 3 else ""
                translated_diag = parsed_translations.get(b_idx, orig_diag)

                if not translated_diag.strip() and orig_diag.strip():
                    translated_diag = orig_diag

                tmp_fh.write(f"{b_idx}\n{ts}\n{translated_diag}\n\n")
                translated_count += 1

            if chunk_idx < total_chunks:
                time.sleep(GEMINI_INTER_CHUNK_DELAY)

        # Close file before rename
        tmp_fh.close()
        tmp_fh = None

        # Atomic rename
        if sys.platform == "win32" and srt_tgt.exists():
            srt_tgt.unlink()
        tmp_tgt.replace(srt_tgt)

        fallbacks, _ = detect_fallbacks(
            srt_src, srt_tgt, argparse.Namespace(fallback_match_threshold=0.95)
        )
        return True, "Success", fallbacks

    except KeyboardInterrupt:
        logger.warning("Translation interrupted by user")
        if tmp_fh:
            tmp_fh.close()
        try:
            tmp_tgt.unlink(missing_ok=True)
        except OSError:
            pass
        return False, "Translation interrupted by user", 0
    except Exception as e:
        logger.error("Translation error: %s", e)
        if tmp_fh:
            tmp_fh.close()
        try:
            tmp_tgt.unlink(missing_ok=True)
        except OSError:
            pass
        return False, f"Write error: {e}", 0
    finally:
        if tmp_fh:
            tmp_fh.close()


# ════════════════════════════════════════════════════════════
#  INTERACTIVE WIZARD
# ════════════════════════════════════════════════════════════
def interactive_wizard(
    args: argparse.Namespace, cfg_memory: Dict[str, Any]
) -> None:
    """Run the interactive configuration wizard.

    Guides the user through step-by-step configuration of the pipeline,
    with support for saved profile fast-path loading.

    Args:
        args: The argument namespace to populate.
        cfg_memory: Previously saved configuration dictionary.
    """
    print(f"\n{C.CYAN}{C.BOLD}", end="")
    print("\u2554" + "\u2550" * 60 + "\u2557")
    print("\u2551  Subs Pipeline v" + __version__ + " " * 37 + "\u2551")
    print("\u255a" + "\u2550" * 60 + "\u255d\n")

    print(
        f"  {C.DIM}This utility automatically handles local multi-language "
        f"media pipeline runs.\n  Use the steps below to initialize models and "
        f"folders.{C.RESET}\n"
    )

    if Context.migration_status == "migrated":
        print(f"  {C.GREEN}[~] Migrated legacy profile settings.{C.RESET}\n")
    elif Context.migration_status == "loaded":
        print(f"  {C.GREEN}[~] Loaded saved profile settings.{C.RESET}\n")

    if Context.config_warning:
        print(f"  {C.RED}[!] CONFIG SYSTEM: {Context.config_warning}{C.RESET}\n")

    if not setup_ffmpeg():
        print(f"{C.RED}  [!] FFmpeg is required to continue.{C.RESET}")
        exit_app(1)

    # Step 1: Folder
    current_dir = os.getcwd()
    print(f"{C.BOLD}> Step 1: Media Directory{C.RESET}")
    print(f"  Current: {C.CYAN}{current_dir}{C.RESET}")
    f_in = input(f"  [Enter = current  |  path  |  'b' = browse]: ").strip()

    if not f_in:
        args.folder = current_dir
        Context.provenance["folder"] = "Interactive"
    elif f_in.lower() == "b":
        args.folder = current_dir
        Context.provenance["folder"] = "Interactive"
        print(
            f"  {C.YELLOW}  Browsing disabled -- falling back to current directory.{C.RESET}"
        )
    else:
        args.folder = str(Path(f_in).resolve())
        Context.provenance["folder"] = "Interactive"

    if not args.folder or not Path(args.folder).is_dir():
        print(f"{C.RED}  [!] Invalid directory.{C.RESET}")
        exit_app(1)
    print(f"  {C.GREEN}-> {args.folder}{C.RESET}\n")

    startup_garbage_collection(args.folder, skip_cleanup=args.no_cleanup)

    # Step 2: Saved Profiles Fast Path
    if cfg_memory:
        print(f"\n{C.BOLD}> Step 2: Use Saved Settings Profile?{C.RESET}")
        print(
            f"  Target Language:     {C.CYAN}{cfg_memory.get('tgt_lang', 'English')}{C.RESET}"
        )
        print(
            f"  Subtitle Extension:  {C.CYAN}{cfg_memory.get('tgt_ext', 'en')}{C.RESET}"
        )
        print(
            f"  Min Blocks Required: {C.CYAN}{cfg_memory.get('min_blocks', 3)}{C.RESET}"
        )
        print(
            f"  Saved Whisper Model: {C.CYAN}"
            f"{(cfg_memory.get('model') or 'None').upper()}{C.RESET}"
        )
        print(f"  Detected Device:     {C.CYAN}{DEVICE} ({COMPUTE}){C.RESET}")

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
            args.api_key = cfg_memory.get("api_key", "")
            args.translate = bool(args.api_key)
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
                "api_key",
                "min_blocks",
                "model",
            ]:
                Context.provenance[k] = "Saved Profile"
            Context.provenance["translate"] = "Saved Profile"
            Context.provenance["transcribe"] = "Saved Profile"
            Context.provenance["embed"] = "Saved Profile"

            print(
                f"\n  {C.CYAN}Applying Saved Configuration Profile & "
                f"Workflow Overrides:{C.RESET}"
            )
            print(
                f"    - Transcription:     "
                f"{C.CYAN}{'Enabled' if args.transcribe else 'Disabled'}{C.RESET}"
                f"  (Model: {args.model.upper()})"
            )
            print(
                f"    - Translation:       "
                f"{C.CYAN}{'Enabled' if args.translate else 'Disabled'}{C.RESET}"
                f" (using stored API key)"
            )
            print(
                f"    - Final Muxing:      "
                f"{C.CYAN}"
                f"{'Enabled (Hardsub)' if args.hardsub else 'Enabled (Softsub)' if args.embed else 'Disabled'}"
                f"{C.RESET}"
            )
            print(
                f"  {C.GREEN}[+] Loaded saved profile. Proceeding directly...{C.RESET}\n"
            )
            return

    # Step 3: Translation settings
    print(f"{C.BOLD}> Step 3: Translation Settings{C.RESET}")
    args.tgt_lang = (
        input(f"  Target Language [{cfg_memory.get('tgt_lang', 'English')}]: ").strip()
        or cfg_memory.get("tgt_lang", "English")
    )
    Context.provenance["tgt_lang"] = "Interactive"

    if len(args.tgt_lang) < 2 or not args.tgt_lang.replace(" ", "").isalpha():
        print(
            f"  {C.YELLOW}  [!] Warning: '{args.tgt_lang}' might not be a "
            f"valid language name.{C.RESET}"
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
        print(f"  {C.YELLOW}[!] {err} Please try again.{C.RESET}")
    Context.provenance["tgt_ext"] = "Interactive"

    # Step 4: Source language
    saved_src = cfg_memory.get("src_lang") or ""
    print(f"\n{C.BOLD}> Step 4: Source Language{C.RESET}")
    print(
        f"  {C.DIM}Blank = auto-detect  |  ISO codes: ja  en  ko  zh  ar  es ...{C.RESET}"
    )
    src_in = input(f"  [{saved_src or 'auto'}]: ").strip().lower()
    args.src_lang = src_in if src_in else (saved_src if saved_src else None)
    Context.provenance["src_lang"] = "Interactive"

    # Step 5: Whisper model
    vram_avail = get_available_vram_gb()
    vram_label = f"{vram_avail:.1f} GB free" if DEVICE == "cuda" else "CPU mode"
    recommended = recommend_whisper_model()

    print(
        f"\n{C.BOLD}> Step 5: Transcription Model "
        f"(Recommended: {MODEL_MAP[recommended].upper()}){C.RESET}"
    )
    print(f"  {C.DIM}Device: {DEVICE}  ({vram_label}){C.RESET}")
    print(f"    {C.CYAN}[0]{C.RESET} Tiny           ~1.0 GB  Fastest")
    print(f"    {C.CYAN}[1]{C.RESET} Base           ~1.5 GB  Fast")
    print(f"    {C.CYAN}[2]{C.RESET} Small          ~2.5 GB  Recommended")
    print(f"    {C.CYAN}[3]{C.RESET} Medium         ~5.0 GB  Better accuracy")
    print(f"    {C.CYAN}[4]{C.RESET} Large-v3 Turbo ~6.0 GB  Fast + accurate")
    print(f"    {C.CYAN}[5]{C.RESET} Large-v3       ~10  GB  Best accuracy")

    args.model = MODEL_MAP.get(
        input(f"  Selection [{recommended}]: ").strip() or recommended, "small"
    )
    Context.provenance["model"] = "Interactive"
    print(f"  {C.GREEN}-> Selected Model: {args.model.upper()}{C.RESET}")

    # Step 6: API Key
    saved_key = cfg_memory.get("api_key", "")
    print(f"\n{C.BOLD}> Step 6: Gemini API Key{C.RESET}")
    if saved_key:
        print(f"  {C.DIM}Stored key found. Press Enter to reuse.{C.RESET}")
    while True:
        key_input = input("  Key: ").strip() or saved_key
        if not key_input:
            args.api_key = ""
            break
        valid, err = validate_api_key(key_input)
        if valid:
            args.api_key = key_input
            break
        print(f"  {C.YELLOW}[!] {err} Please try again.{C.RESET}")
    args.translate = bool(args.api_key)
    Context.provenance["api_key"] = "Interactive"
    Context.provenance["translate"] = "Interactive"

    # Step 7: Output format
    print(f"\n{C.BOLD}> Step 7: Output Format{C.RESET}")
    print(f"    {C.CYAN}[0]{C.RESET} Softsub -- toggleable track  (default)")
    print(f"    {C.CYAN}[1]{C.RESET} Hardsub -- burned into video")
    args.hardsub = input("  Selection [0]: ").strip() == "1"
    Context.provenance["hardsub"] = "Interactive"

    # Step 8: Subtitle Validation
    print(f"\n{C.BOLD}> Step 8: Health Check Settings{C.RESET}")
    saved_min = cfg_memory.get("min_blocks", 3)
    try:
        val_blocks = input(f"  Minimum valid blocks [{saved_min}]: ").strip()
        args.min_blocks = int(val_blocks) if val_blocks else int(saved_min)
    except ValueError:
        args.min_blocks = 3
    Context.provenance["min_blocks"] = "Interactive"

    # Step 8b: Advanced Tuning Options
    print(
        f"\n{C.BOLD}> Step 8b: Tune Advanced SRT & Similarity Thresholds?{C.RESET} [y/N]"
    )
    tune = input("  Selection: ").strip().lower() == "y"
    if tune:
        try:
            args.srt_max_avg_duration = float(
                input(
                    f"    Max block duration seconds "
                    f"[{cfg_memory.get('srt_max_avg_duration', 10.0)}]: "
                ).strip()
                or cfg_memory.get("srt_max_avg_duration", 10.0)
            )
            args.srt_min_avg_duration = float(
                input(
                    f"    Min block duration seconds "
                    f"[{cfg_memory.get('srt_min_avg_duration', 0.1)}]: "
                ).strip()
                or cfg_memory.get("srt_min_avg_duration", 0.1)
            )
            args.srt_dup_ratio = float(
                input(
                    f"    Duplicate loop ratio threshold "
                    f"[{cfg_memory.get('srt_dup_ratio', 0.6)}]: "
                ).strip()
                or cfg_memory.get("srt_dup_ratio", 0.6)
            )
            args.fallback_match_threshold = float(
                input(
                    f"    Fuzzy match ratio (0.0 - 1.0) "
                    f"[{cfg_memory.get('fallback_match_threshold', 0.95)}]: "
                ).strip()
                or cfg_memory.get("fallback_match_threshold", 0.95)
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
                f"    {C.YELLOW}[!] Invalid numeric values. "
                f"Standard defaults retained.{C.RESET}"
            )
    else:
        args.srt_max_avg_duration = cfg_memory.get("srt_max_avg_duration", 10.0)
        args.srt_min_avg_duration = cfg_memory.get("srt_min_avg_duration", 0.1)
        args.srt_dup_ratio = cfg_memory.get("srt_dup_ratio", 0.6)
        args.fallback_match_threshold = cfg_memory.get(
            "fallback_match_threshold", 0.95
        )

    # Step 9: Pipeline Steps
    print(f"\n{C.BOLD}> Step 9: Pipeline Steps{C.RESET}")
    args.transcribe = input("  Transcribe? [Y/n]: ").strip().lower() != "n"
    Context.provenance["transcribe"] = "Interactive"
    if args.translate:
        args.translate = input("  Translate?  [Y/n]: ").strip().lower() != "n"
        Context.provenance["translate"] = "Interactive"
    args.embed = input("  Mux?        [Y/n]: ").strip().lower() != "n"
    Context.provenance["embed"] = "Interactive"

    if args.hardsub and not args.embed:
        print(
            f"  {C.YELLOW}[!] Override: Hardsub is active. Soft muxing step "
            f"enabled to complete burning action.{C.RESET}"
        )
        args.embed = True
        Context.provenance["embed"] = "Auto-Override"

    # Step 10: Watch mode
    args.watch = (
        input(f"\n{C.BOLD}> Step 10: Watch Mode?{C.RESET} [y/N]: ").strip().lower()
        == "y"
    )

    # Save config with file permissions
    save_config({
        "schema_version": 1,
        "api_key": args.api_key,
        "tgt_lang": args.tgt_lang,
        "tgt_ext": args.tgt_ext,
        "src_lang": args.src_lang or "",
        "min_blocks": args.min_blocks,
        "model": args.model,
        "skip_cleanup": args.no_cleanup,
        "skip_migration": args.skip_migration,
        "explain_summary": args.explain_summary,
        "srt_max_avg_duration": args.srt_max_avg_duration,
        "srt_min_avg_duration": args.srt_min_avg_duration,
        "srt_dup_ratio": args.srt_dup_ratio,
        "fallback_match_threshold": args.fallback_match_threshold,
        "max_audit_logs": cfg_memory.get("max_audit_logs", MAX_AUDIT_LOGS),
        "gemini_model": cfg_memory.get("gemini_model", DEFAULT_GEMINI_MODEL),
    })


# ════════════════════════════════════════════════════════════
#  HARDSUB HELPER (Refactored duplicate code)
# ════════════════════════════════════════════════════════════
def run_ffmpeg_hardsub(
    media_path: Path,
    target_srt: Path,
    out_path: Path,
    cwd_path: Path,
) -> subprocess.CompletedProcess:
    """Execute FFmpeg hardsub command with safe path escaping.

    Uses shlex.quote to prevent command injection via filenames.
    Falls back to alternate temp directory on permission errors.

    Args:
        media_path: Path to the source media file.
        target_srt: Path to the subtitle file to burn.
        out_path: Path for the output video file.
        cwd_path: Working directory for FFmpeg execution.

    Returns:
        CompletedProcess instance with the FFmpeg result.
    """
    base = media_path.stem
    rel_media = media_path.name
    rel_out = out_path.relative_to(cwd_path)
    unique_id = uuid.uuid4().hex[:8]
    temp_srt = f"temp_hardsub_{base}_{unique_id}.srt"

    # Primary attempt: copy to working directory
    try:
        shutil.copy(target_srt, cwd_path / temp_srt)
        register_temp_file(cwd_path / temp_srt)
        # Use shlex.quote to prevent command injection
        safe_srt = shlex.quote(temp_srt)
        cmd = [
            Context.ffmpeg_cmd,
            "-y",
            "-v",
            "error",
            "-i",
            rel_media,
            "-vf",
            f"subtitles={safe_srt}",
            "-c:a",
            "copy",
            str(rel_out),
        ]
        result = subprocess.run(cmd, capture_output=True, cwd=cwd_path, encoding="utf-8", errors="replace")
        safe_remove(cwd_path / temp_srt)
        return result
    except (OSError, PermissionError) as err1:
        # Fallback: use system temp directory
        try:
            alt_temp = Path(tempfile.gettempdir()) / temp_srt
            shutil.copy(target_srt, alt_temp)
            register_temp_file(alt_temp)
            # Escape path for FFmpeg filter syntax (Windows path handling)
            escaped_alt = str(alt_temp).replace("\\", "/").replace(":", "\\:")
            safe_escaped = shlex.quote(escaped_alt)
            cmd = [
                Context.ffmpeg_cmd,
                "-y",
                "-v",
                "error",
                "-i",
                str(media_path),
                "-vf",
                f"subtitles={safe_escaped}",
                "-c:a",
                "copy",
                str(out_path),
            ]
            result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
            safe_remove(alt_temp)
            return result
        except (OSError, PermissionError) as err2:
            print(
                f"  {C.RED}[x] Severe Hardsub Error: Temporary subtitle file "
                f"could not be written.{C.RESET}"
            )
            print(f"      Workspace Error: {err1}")
            print(f"      System Temp Error: {err2}")
            return subprocess.CompletedProcess(args=[], returncode=1)


# ════════════════════════════════════════════════════════════
#  MUX OUTPUT VALIDATION (DEFINED BEFORE USE)
# ════════════════════════════════════════════════════════════
def verify_mux_output(path: Union[str, Path], hardsub: bool = False) -> Tuple[bool, str]:
    """Verify the output of a muxing/hardsub operation.

    Args:
        path: Path to the output file.
        hardsub: Whether hardsub mode was used.

    Returns:
        Tuple of (is_valid, reason_string).
    """
    p = Path(path)
    if not p.exists():
        return False, "Output container was not generated."
    file_size = p.stat().st_size
    if file_size < MUXED_MIN_BYTES:
        return (
            False,
            f"Output file size ({file_size} bytes) is below safe processing bounds.",
        )
    return True, "OK"


# ════════════════════════════════════════════════════════════
#  CORE PROCESS FILE PATHWAY
# ════════════════════════════════════════════════════════════
def process_file(
    media_path: Union[str, Path],
    args: argparse.Namespace,
    file_index: int = 1,
    total_files: int = 1,
) -> Tuple[Dict[str, Any], float]:
    """Process a single media file through the full pipeline.

    Pipeline steps: audio extraction -> transcription -> translation -> muxing.
    Includes retry logic, disk space checks, timeout handling, and safe
    subprocess execution with proper encoding.

    Args:
        media_path: Path to the media file.
        args: Pipeline configuration arguments.
        file_index: Current file index in the batch.
        total_files: Total number of files in the batch.

    Returns:
        Tuple of (status_dict, elapsed_seconds).
    """
    media_p = Path(media_path)
    base = media_p.stem
    t_start = time.time()
    temp_audio = media_p.parent / f"temp_{base}_audio.wav"
    status: Dict[str, Any] = {
        "transcribed": False,
        "translated": False,
        "reused_srt": False,
        "reused_all": False,
        "mixed_language": False,
        "partial_success": False,
        "fallback_count": 0,
        "muxed": False,
        "skipped": False,
        "error": False,
        "audio_failed": False,
    }

    print(f"\n{C.BOLD}── [{file_index}/{total_files}] {base}{C.RESET}")

    if not media_p.exists():
        print(f"  {C.RED}[x] File no longer exists -- skipping.{C.RESET}")
        status["error"] = True
        return status, 0.0

    # Check for path traversal via symlinks
    try:
        real_path = media_p.resolve()
        folder_real = Path(args.folder).resolve()
        if not str(real_path).startswith(str(folder_real)):
            print(
                f"  {C.RED}[x] Path traversal detected via symlink -- "
                f"file resolves outside target directory.{C.RESET}"
            )
            status["error"] = True
            return status, 0.0
    except (OSError, RuntimeError):
        pass

    safe_remove(temp_audio)

    old_srt = media_p.parent / f"{base}.auto.srt"
    srt_src = media_p.parent / f"{base}.subs-pipeline.srt"

    if old_srt.exists() and not srt_src.exists():
        if not args.skip_migration:
            try:
                old_srt.rename(srt_src)
                print(
                    f"  {C.DIM}[~] Converted legacy file format: "
                    f"'{old_srt.name}' -> '{srt_src.name}'.{C.RESET}"
                )
            except OSError:
                pass

    srt_tgt = media_p.parent / f"{base}.{args.tgt_ext}.srt"
    out_ext = "mp4" if args.hardsub else "mkv"
    out_path = media_p.parent / f"muxed_{args.tgt_ext}" / f"{base}.{out_ext}"
    duration = get_duration(media_path)

    # Check disk space before operations
    if not check_disk_space(media_p.parent):
        print(
            f"  {C.YELLOW}[!] Insufficient disk space. "
            f"Attempting to continue...{C.RESET}"
        )

    # ── DRY RUN SIMULATION PATHWAY ───────────────────────
    if args.dry_run:
        print(
            f"  {C.YELLOW}[DRY-RUN] Planning execution for file: {base}{C.RESET}"
        )

        if srt_src.exists():
            print(f"    - Existing source subtitle '{srt_src.name}' detected.")
            ok, reason = is_valid_srt(srt_src, duration, args.min_blocks, args)
            if ok:
                print(
                    f"      [Health Check] PASS: Reusing '{srt_src.name}' "
                    f"(transcription bypassed)."
                )
                status["reused_srt"] = True
            else:
                print(
                    f"      {C.YELLOW}[Health Check] FAIL: '{srt_src.name}' "
                    f"is invalid ({reason}).{C.RESET}"
                )
                print(
                    f"      -> Simulated Action: Re-extract audio & transcribe "
                    f"(using model {args.model.upper()})."
                )
                status["transcribed"] = True
        else:
            print(f"    - No source subtitle exists.")
            print(
                f"    - Simulated Action: Extract audio and transcribe using "
                f"local {args.model.upper()} engine."
            )
            status["transcribed"] = True

        if srt_tgt.exists():
            print(f"    - Existing target subtitle '{srt_tgt.name}' detected.")
            fallbacks, _ = detect_fallbacks(srt_src, srt_tgt, args)
            if fallbacks > 0:
                print(f"      [Status Check] Target file contains {fallbacks} fallback block(s).")
                status["mixed_language"] = True
                status["fallback_count"] = fallbacks
            else:
                print(f"      [Status Check] Target file looks fully translated.")
                status["translated"] = True
        elif args.translate:
            if not args.api_key:
                print(
                    "    - Translation requested, but API Key is missing. "
                    "Skipping translation step."
                )
            else:
                print(
                    f"    - Simulated Action: Translate dialogue to {args.tgt_lang} "
                    f"using Gemini API."
                )
                status["translated"] = True

        if args.embed:
            if out_path.exists():
                print(f"    - Output video already exists at '{out_path.name}'.")
                status["reused_all"] = True
            else:
                print(
                    f"    - Simulated Action: Mux subtitles into final '{out_ext}' "
                    f"container ({'Hardsub' if args.hardsub else 'Softsub'})."
                )
                status["muxed"] = True
        return status, 0.02

    # Check for existing output
    if out_path.exists() and out_path.stat().st_size >= MUXED_MIN_BYTES:
        print(
            f"  {C.GREEN}[+] Output file already exists. "
            f"Skipping processing.{C.RESET}"
        )
        status["reused_all"] = True
        return status, 0.0

    if duration <= 0:
        print(
            f"  {C.YELLOW}[!] Could not determine duration -- "
            f"progress % unavailable.{C.RESET}"
        )

    try:
        # ── STEP 1: AUDIO EXTRACTION ──────────────────────
        audio_extracted_successfully = False
        if args.transcribe and not srt_src.exists() and not srt_tgt.exists():
            print(f"  {C.CYAN}> Extracting audio...{C.RESET}")
            register_temp_file(temp_audio)
            result = subprocess.run(
                [
                    Context.ffmpeg_cmd,
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(media_path),
                    "-vn",
                    "-acodec",
                    AUDIO_CODEC,
                    "-ar",
                    str(AUDIO_SAMPLE_RATE),
                    "-ac",
                    "1",
                    str(temp_audio),
                ],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0 or not temp_audio.exists():
                stderr_msg = result.stderr.strip() if result.stderr else ""
                print(f"  {C.RED}[x] Audio extraction failed.{C.RESET}")
                if stderr_msg:
                    print(f"      {C.DIM}{stderr_msg[:500]}{C.RESET}")
                status["audio_failed"] = True
            else:
                audio_extracted_successfully = True

        # ── STEP 2: TRANSCRIPTION (with retry) ────────────
        if (
            args.transcribe
            and not srt_src.exists()
            and not srt_tgt.exists()
            and audio_extracted_successfully
        ):
            transcription_success = False
            last_error = ""

            for attempt in range(MAX_TRANSCRIPTION_RETRIES):
                try:
                    load_whisper(args.model)
                    print(
                        f"  {C.CYAN}> Transcribing..."
                        f"{f' (attempt {attempt + 1}/{MAX_TRANSCRIPTION_RETRIES})' if attempt > 0 else ''}"
                        f"{C.RESET}"
                    )
                    tmp_srt = srt_src.with_suffix(".tmp")
                    register_temp_file(tmp_srt)

                    lang_hint = args.src_lang if args.src_lang else None

                    # Whisper transcription with timeout using signal (Unix) or threading
                    import threading as _threading

                    whisper_result: List[Any] = [None, None, None]  # segments, info, error

                    def _transcribe() -> None:
                        try:
                            segments, info = Context.whisper_model.transcribe(
                                str(temp_audio),
                                beam_size=WHISPER_BEAM_SIZE,
                                language=lang_hint,
                            )
                            whisper_result[0] = list(segments)
                            whisper_result[1] = info
                        except Exception as e:
                            whisper_result[2] = e

                    t = _threading.Thread(target=_transcribe)
                    t.start()
                    t.join(timeout=WHISPER_TRANSCRIBE_TIMEOUT)

                    if t.is_alive():
                        logger.error("Whisper transcription timed out after %d seconds", WHISPER_TRANSCRIBE_TIMEOUT)
                        last_error = f"Transcription timed out after {WHISPER_TRANSCRIBE_TIMEOUT}s"
                        # Can't kill thread cleanly, mark for retry
                        safe_remove(tmp_srt)
                        if attempt < MAX_TRANSCRIPTION_RETRIES - 1:
                            delay = TRANSCRIPTION_RETRY_BASE_DELAY * (2**attempt)
                            print(f"  {C.YELLOW}[!] Timeout, retrying in {delay:.0f}s...{C.RESET}")
                            time.sleep(delay)
                            continue
                        else:
                            status["error"] = True
                            break

                    if whisper_result[2]:
                        raise whisper_result[2]

                    if whisper_result[0] is None:
                        raise RuntimeError("No transcription result")

                    with open(tmp_srt, "w", encoding="utf-8") as f:
                        for idx, seg in enumerate(whisper_result[0], 1):
                            f.write(
                                f"{idx}\n"
                                f"{fmt_srt_ts(seg.start)} --> {fmt_srt_ts(seg.end)}\n"
                                f"{seg.text.strip()}\n\n"
                            )
                            if duration > 0:
                                pct = min(seg.end / duration * 100, 100.0)
                                sys.stdout.write(
                                    f"\r    {C.DIM}{fmt_time(seg.end)} / "
                                    f"{fmt_time(duration)} ({pct:.1f}%)"
                                    f"{C.RESET}   "
                                )
                            else:
                                sys.stdout.write(
                                    f"\r    {C.DIM}{fmt_time(seg.end)}{C.RESET}   "
                                )
                            sys.stdout.flush()

                    tmp_srt.replace(srt_src)
                    unregister_temp_file(tmp_srt)
                    print(
                        f"\n  {C.GREEN}[+] Transcription done "
                        f"(detected: {whisper_result[1].language}){C.RESET}"
                    )
                    status["transcribed"] = True
                    transcription_success = True
                    break

                except Exception as e:
                    last_error = str(e)
                    print(f"\n  {C.RED}[x] Transcription error: {e}{C.RESET}")
                    safe_remove(tmp_srt)
                    if attempt < MAX_TRANSCRIPTION_RETRIES - 1:
                        delay = TRANSCRIPTION_RETRY_BASE_DELAY * (2**attempt)
                        print(
                            f"  {C.YELLOW}[~] Retrying in {delay:.0f}s...{C.RESET}"
                        )
                        time.sleep(delay)
                        perform_vram_gc()
                    else:
                        print(
                            f"  {C.RED}[x] All {MAX_TRANSCRIPTION_RETRIES} "
                            f"transcription attempts failed.{C.RESET}"
                        )

        # ── SRT HEALTH CHECK ──────────────────────────────
        srt_src_healthy = True
        if srt_src.exists() and not srt_tgt.exists():
            ok, reason = is_valid_srt(srt_src, duration, args.min_blocks, args)
            if not ok:
                print(
                    f"  {C.YELLOW}[!] SRT health check failed: {reason}{C.RESET}"
                )
                print(f"  {C.YELLOW}    Translation skipped.{C.RESET}")
                srt_src_healthy = False
                status["skipped"] = True

        # ── STEP 3: TRANSLATION ───────────────────────────
        if (
            args.translate
            and not Context.translation_disabled
            and not srt_tgt.exists()
            and srt_src.exists()
            and srt_src_healthy
            and args.api_key
        ):
            if (
                Context.get_consecutive_total_failures()
                >= CONSECUTIVE_TOTAL_FAIL_LIMIT
            ):
                Context.translation_disabled = True
                print(
                    f"  {C.RED}[!] Translation suspended due to persistent "
                    f"communication failures.{C.RESET}"
                )
            else:
                print(f"  {C.CYAN}> Translating -> {args.tgt_lang}...{C.RESET}")
                gemini_model = getattr(args, "gemini_model", DEFAULT_GEMINI_MODEL)
                success, msg, fallbacks = translate_srt_native(
                    srt_src,
                    srt_tgt,
                    args.tgt_lang,
                    args.api_key,
                    gemini_model=gemini_model,
                )
                if success:
                    if fallbacks > 0:
                        status["mixed_language"] = True
                        status["partial_success"] = True
                        status["fallback_count"] = fallbacks
                        print(
                            f"  {C.YELLOW}[~] Partial Success: {fallbacks} "
                            f"block(s) fell back to source language.{C.RESET}"
                        )
                    else:
                        status["translated"] = True
                        print(f"  {C.GREEN}[+] Translation done.{C.RESET}")
                else:
                    print(
                        f"  {C.RED}[x] Translation skipped/failed: {msg}{C.RESET}"
                    )

        # ── STEP 4: EMBED ─────────────────────────────────
        is_audio = media_p.suffix.lower() in AUDIO_EXTS
        target_srt = None

        if srt_tgt.exists():
            target_srt = srt_tgt
            if not status["translated"] and not status["mixed_language"]:
                if srt_src.exists():
                    fallback_count, _ = detect_fallbacks(srt_src, srt_tgt, args)
                    if fallback_count > 0:
                        status["mixed_language"] = True
                        status["partial_success"] = True
                        status["fallback_count"] = fallback_count
                        print(
                            f"  {C.YELLOW}[~] Target file contains {fallback_count} "
                            f"unchanged fallback blocks.{C.RESET}"
                        )
                    else:
                        status["reused_srt"] = True
                else:
                    status["reused_srt"] = True

        elif srt_src.exists() and srt_src_healthy:
            target_srt = srt_src
            if args.translate:
                print(
                    f"  {C.YELLOW}[!] Using source SRT -- "
                    f"translated SRT unavailable.{C.RESET}"
                )

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(
                f"  {C.RED}[x] Failed to create output directory: {e}{C.RESET}"
            )
            status["error"] = True
            return status, time.time() - t_start

        if out_path.exists():
            if out_path.stat().st_size < MUXED_MIN_BYTES:
                print(
                    f"  {C.YELLOW}[!] Removing incomplete output from "
                    f"previous run.{C.RESET}"
                )
                safe_remove(out_path)

        if args.embed and target_srt and not out_path.exists():
            if is_audio:
                print(
                    f"  {C.DIM}[~] Audio-only file -- muxing skipped.{C.RESET}"
                )
            else:
                print(f"  {C.CYAN}> Muxing...{C.RESET}")

                try:
                    if args.hardsub:
                        cwd_path = media_p.parent
                        result = run_ffmpeg_hardsub(
                            media_p, target_srt, out_path, cwd_path
                        )
                    else:
                        # Softsub: use shlex.quote for safety
                        cmd = [
                            Context.ffmpeg_cmd,
                            "-y",
                            "-v",
                            "error",
                            "-i",
                            str(media_path),
                            "-i",
                            str(target_srt),
                            "-c:v",
                            "copy",
                            "-c:a",
                            "copy",
                            "-c:s",
                            "srt",
                            "-metadata:s:s:0",
                            f"language={args.tgt_ext}",
                            str(out_path),
                        ]
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            encoding="utf-8",
                            errors="replace",
                        )

                    if result.returncode != 0:
                        err = (
                            result.stderr.strip()[:500]
                            if result.stderr
                            else ""
                        )
                        if err:
                            print(f"  {C.DIM}    FFmpeg: {err}{C.RESET}")

                    ok, reason = verify_mux_output(out_path, hardsub=args.hardsub)
                    if ok:
                        print(
                            f"  {C.GREEN}[+] -> "
                            f"{out_path.parent.name}\\{out_path.name}{C.RESET}"
                        )
                        status["muxed"] = True
                    else:
                        print(
                            f"  {C.YELLOW}[!] Mux validation failed: "
                            f"{reason}{C.RESET}"
                        )

                except Exception as e:
                    print(f"  {C.RED}[x] Muxing error: {e}{C.RESET}")

    except KeyboardInterrupt:
        print(
            f"\n{C.YELLOW}[!] Interrupted during processing of {base}.{C.RESET}"
        )
        raise
    except Exception as e:
        print(
            f"  {C.RED}[x] Unexpected pipeline processing error: {e}{C.RESET}"
        )
        status["error"] = True

    finally:
        safe_remove(temp_audio)
        perform_vram_gc()

    return status, time.time() - t_start


# ════════════════════════════════════════════════════════════
#  FILE ENUMERATION (with recursive support)
# ════════════════════════════════════════════════════════════
def enumerate_media_files(folder: Union[str, Path], recursive: bool = False) -> List[str]:
    """Enumerate media files in the target folder.

    Args:
        folder: The directory to scan.
        recursive: If True, include subdirectories.

    Returns:
        Sorted list of media file paths as strings.
    """
    folder_path = Path(folder)
    media_files: List[str] = []

    if recursive:
        files = folder_path.rglob("*")
    else:
        files = folder_path.iterdir()

    for p in files:
        if p.is_file() and p.suffix.lower() in MEDIA_EXTS:
            # Check for symlinks outside target
            try:
                if p.is_symlink():
                    real = p.resolve()
                    folder_real = folder_path.resolve()
                    if not str(real).startswith(str(folder_real)):
                        logger.warning("Skipping symlink outside target: %s", p)
                        continue
            except (OSError, RuntimeError):
                pass
            media_files.append(str(p))

    media_files.sort(key=lambda x: natural_keys(Path(x).name))
    return media_files


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
            "  %(prog)s --headless --folder . --model small --tgt-lang Arabic\n"
            "  %(prog)s --version                         # Show version"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--headless", action="store_true", help="Run without interactive prompts")
    parser.add_argument("--folder", type=str, help="Target directory containing media files")
    parser.add_argument("--model", type=str, default=None, help="Whisper model to use")
    parser.add_argument("--src-lang", type=str, default=None, dest="src_lang", help="Source language code")
    parser.add_argument("--tgt-lang", type=str, default=None, dest="tgt_lang", help="Target language name")
    parser.add_argument("--tgt-ext", type=str, default=None, dest="tgt_ext", help="Target subtitle extension (e.g., en, ar)")
    parser.add_argument("--api-key", type=str, default=None, dest="api_key", help="Gemini API key")
    parser.add_argument("--hardsub", action="store_true", help="Burn subtitles into video (hardsub)")
    parser.add_argument("--min-blocks", type=int, default=None, dest="min_blocks", help="Minimum valid subtitle blocks")
    parser.add_argument("--skip-transcribe", action="store_true", dest="skip_transcribe", help="Skip transcription step")
    parser.add_argument("--skip-translate", action="store_true", dest="skip_translate", help="Skip translation step")
    parser.add_argument("--skip-embed", action="store_true", dest="skip_embed", help="Skip muxing step")
    parser.add_argument("--watch", action="store_true", help="Enable filesystem watch mode")
    parser.add_argument("--no-cleanup", action="store_true", dest="no_cleanup", help="Skip temp file cleanup")
    parser.add_argument("--skip-migration", action="store_true", dest="skip_migration", help="Skip legacy file migration")
    parser.add_argument("--explain-summary", action="store_true", dest="explain_summary", help="Show status code explanations")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="Simulate without processing")
    parser.add_argument("--no-audit", action="store_true", dest="no_audit", help="Disable audit logging")
    parser.add_argument("--verbose-summary", action="store_true", dest="verbose_summary", help="Show detailed processing info")
    parser.add_argument("--recursive", action="store_true", dest="recursive", help="Include subdirectories")
    parser.add_argument("--quiet", action="store_true", dest="quiet", help="Suppress non-error output")
    parser.add_argument("--verbose", action="store_true", dest="verbose", help="Enable verbose/debug output")
    parser.add_argument("--gemini-model", type=str, default=None, dest="gemini_model", help="Gemini model name (default: gemini-1.5-flash)")

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
    args = parser.parse_args()

    # Setup logging based on verbosity
    global logger
    logger = setup_logging(quiet=args.quiet, verbose=args.verbose)

    Context.active_temp_files = set()
    Context.failed_cleanups = []
    Context.translation_disabled = False
    Context.reset_all_counters()
    Context.provenance = {}

    # Normalize model name (case-insensitive)
    if args.model:
        args.model = args.model.lower().strip()
        # Map common aliases
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

    for key, default_val in DEFAULT_CONFIG.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, cfg.get(key, default_val))

    args.no_cleanup = args.no_cleanup or cfg.get("skip_cleanup", False)
    args.skip_migration = args.skip_migration or cfg.get("skip_migration", False)
    # Fix: explain_summary should respect CLI explicit False, not always OR with True
    if not hasattr(args, "explain_summary") or args.explain_summary is None:
        args.explain_summary = cfg.get("explain_summary", True)

    diag_status = verify_config_status()
    if diag_status is not True:
        Context.config_warning = str(diag_status)

    if args.src_lang and not args.src_lang.strip():
        args.src_lang = None

    args.transcribe = not args.skip_transcribe
    args.translate = not args.skip_translate and bool(args.api_key)
    args.embed = not args.skip_embed

    check_dependencies(headless=args.headless)

    # Establish configuration parameter provenance
    for key in DEFAULT_CONFIG.keys():
        cli_flag_dash = f"--{key.replace('_', '-')}"
        if any(arg.startswith(cli_flag_dash) for arg in sys.argv):
            Context.provenance[key] = "CLI"
        elif key in cfg:
            Context.provenance[key] = "Config File"
        else:
            Context.provenance[key] = "Default"

    if not args.headless:
        interactive_wizard(args, cfg)
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

    summary: List[Tuple[str, Dict[str, Any], float]] = []
    batch_start = time.time()

    if media_files:
        for i, file in enumerate(media_files, 1):
            try:
                status, elapsed = process_file(file, args, i, len(media_files))
                summary.append((Path(file).name, status, elapsed))
            except KeyboardInterrupt:
                print(
                    f"\n{C.YELLOW}[!] Interrupted during batch loop "
                    f"processing.{C.RESET}"
                )
                summary.append((Path(file).name, {"error": True}, 0.0))
                break
            except Exception as e:
                print(
                    f"{C.RED}  [x] Processing fault encountered on "
                    f"{Path(file).name}: {e}{C.RESET}"
                )
                summary.append((Path(file).name, {"error": True}, 0.0))

        total_time = time.time() - batch_start
        if not args.quiet:
            print_summary(summary, total_time, args)
        if not args.dry_run:
            write_audit_log(args, summary, total_time)
    else:
        print(
            f"{C.YELLOW}\n  No compatible media files found inside target "
            f"location.{C.RESET}"
        )

    if args.watch:
        run_watcher(args)
    else:
        exit_app(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}[!] Interrupted.{C.RESET}")
        exit_app(0)
    except Exception as e:
        print(f"\n{C.RED}[x] FATAL RUN ERROR: {e}{C.RESET}")
        import traceback

        traceback.print_exc()
        exit_app(1)
