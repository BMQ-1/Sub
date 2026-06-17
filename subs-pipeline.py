import os
import re
import sys
import gc
import json
import time
import uuid
import shutil
import argparse
import platform
import zipfile
import threading
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

# ── Safe Terminal Encoding for Arabic/UTF-8 ─────────────
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ── Torch — imported once at module level ───────────────
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch           = None
    _TORCH_AVAILABLE = False

if platform.system() == "Windows":
    os.system("color")

# ════════════════════════════════════════════════════════════
#  ANSI COLORS
# ════════════════════════════════════════════════════════════
class C:
    CYAN   = '\033[96m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    RESET  = '\033[0m'
    BOLD   = '\033[1m'
    DIM    = '\033[2m'

_ANSI_RE = re.compile(r'\033(?:\[[0-9;]*[A-Za-z]|\][^\007]*\007)')

def strip_ansi(s):
    return _ANSI_RE.sub('', s)

# ════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════
APP_DIR     = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "subs_pipeline_settings.json"

MEDIA_EXTS = (
    ".mkv", ".mp4", ".webm", ".avi", ".mov", ".m4v",
    ".flv", ".ts",  ".wmv", ".mp3", ".wav", ".m4a",
    ".aac", ".flac", ".opus"
)
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".opus")

GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent?key={key}"
)

VRAM_REQUIREMENTS = {
    "tiny":            1.0,
    "base":            1.5,
    "small":           2.5,
    "medium":          5.0,
    "large-v3-turbo":  6.0,
    "large-v3":       10.0,
}

MODEL_MAP = {
    "0": "tiny",
    "1": "base",
    "2": "small",
    "3": "medium",
    "4": "large-v3-turbo",
    "5": "large-v3",
}

# SRT health thresholds
SRT_MAX_AVG_DURATION = 10.0   # seconds — longer avg = likely hallucination
SRT_MIN_AVG_DURATION = 0.1    # seconds — shorter avg = flash hallucination
SRT_DUP_RATIO        = 0.6    # 60 % duplicate lines triggers rejection

# Mux validation
MUXED_MIN_BYTES = 1024        # files smaller than 1 KB are treated as corrupt

# Timings & Chunking
GEMINI_CHUNK_SIZE        = 80 # 80 blocks ≈ 240 lines per request
GEMINI_TIMEOUT           = 30 # seconds per HTTP request
GEMINI_INTER_CHUNK_DELAY = 2  # seconds between chunks
WATCHER_SETTLE_SECS      = 5  # Give file writes time to settle

# Required / optional Python packages
REQUIRED_PACKAGES = ["faster_whisper"]
OPTIONAL_PACKAGES = {"watchdog": "watch mode"}

# Batch Quota Safety Brakes
CONSECUTIVE_429_LIMIT        = 3  # Disable translation if sequential rate limits are hit
CONSECUTIVE_TOTAL_FAIL_LIMIT = 5  # Disable translation if general failures hit this limit

# ════════════════════════════════════════════════════════════
#  GLOBAL CONTEXT
# ════════════════════════════════════════════════════════════
class Context:
    whisper_model                    = None
    whisper_loaded                   = False
    whisper_lock                     = threading.Lock()
    ffmpeg_cmd                       = None
    ffprobe_cmd                      = None
    translation_disabled             = False
    consecutive_429s                 = 0
    consecutive_total_failures       = 0
    config_warning                   = ""
    active_temp_files                = set()
    temp_lock                        = threading.Lock()
    failed_cleanups                  = []

# ════════════════════════════════════════════════════════════
#  TEMP FILE MANAGER
# ════════════════════════════════════════════════════════════
def register_temp_file(path):
    with Context.temp_lock:
        Context.active_temp_files.add(Path(path).resolve())

def unregister_temp_file(path):
    with Context.temp_lock:
        p = Path(path).resolve()
        Context.active_temp_files.discard(p)

def cleanup_all_temp_files():
    with Context.temp_lock:
        for p in list(Context.active_temp_files):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
            Context.active_temp_files.discard(p)

# ════════════════════════════════════════════════════════════
#  GARBAGE COLLECTION
# ════════════════════════════════════════════════════════════
def startup_garbage_collection(folder_path, skip_cleanup=False):
    """Cleans up stale temp files left over from prior abnormal terminations."""
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
                    is_stale_hardsub = item.name.startswith("temp_hardsub_") and item.suffix.lower() == ".srt"
                    is_stale_audio   = item.name.startswith("temp_") and item.name.endswith("_audio.wav")
                    if is_stale_hardsub or is_stale_audio:
                        try:
                            item.unlink()
                            cleaned_count += 1
                        except OSError:
                            pass
        except PermissionError:
            pass
    if cleaned_count > 0:
        print(f"  {C.DIM}[~] Swept workspaces: Purged {cleaned_count} stale temp file(s).{C.RESET}")

# ════════════════════════════════════════════════════════════
#  SAFE EXIT
# ════════════════════════════════════════════════════════════
def exit_app(code=0):
    cleanup_all_temp_files()
    if Context.failed_cleanups:
        print(f"\n{C.DIM}  [~] System cleanup complete. Some active lock files were bypassed:{C.RESET}")
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
def verify_config_status():
    """Diagnoses configuration system health non-invasively without disk writes."""
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

def load_config():
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_config(conf):
    tmp = CONFIG_PATH.with_suffix('.tmp')
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(conf, f, indent=4, ensure_ascii=False)
        with open(tmp, "r", encoding="utf-8") as f:
            json.load(f)
        tmp.replace(CONFIG_PATH)
    except Exception as e:
        print(f"\n  {C.YELLOW}[!] Config save failed: {e}{C.RESET}")
        try:
            tmp.unlink()
        except Exception:
            pass

cfg = load_config()

# ════════════════════════════════════════════════════════════
#  DEPENDENCY PRE-FLIGHT
# ════════════════════════════════════════════════════════════
def check_dependencies(headless=False):
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"\n{C.RED}  [!] Missing required packages:{C.RESET}")
        for pkg in missing:
            print(f"        pip install {pkg.replace('_', '-')}")
        print()
        exit_app(1)

    if not headless:
        for pkg, feature in OPTIONAL_PACKAGES.items():
            try:
                __import__(pkg)
            except ImportError:
                print(f"  {C.DIM}[~] Optional: pip install {pkg}  (enables {feature}){C.RESET}")

# ════════════════════════════════════════════════════════════
#  HARDWARE DETECTION
# ════════════════════════════════════════════════════════════
def detect_hardware():
    if _TORCH_AVAILABLE:
        try:
            if _torch.cuda.is_available():
                return "cuda", "float16"
        except Exception:
            pass
    return "cpu", "int8"

DEVICE, COMPUTE = detect_hardware()

def get_available_vram_gb():
    if _TORCH_AVAILABLE:
        try:
            if _torch.cuda.is_available():
                free, _ = _torch.cuda.mem_get_info()
                return free / (1024 ** 3)
        except Exception:
            pass
    return 0.0

def recommend_whisper_model():
    """Provides a baseline model recommendation based on local system specifications."""
    if DEVICE == "cuda":
        vram = get_available_vram_gb()
        if vram >= 10.0:
            return "5"  # large-v3
        elif vram >= 6.0:
            return "4"  # large-v3-turbo
        elif vram >= 5.0:
            return "3"  # medium
        elif vram >= 2.5:
            return "2"  # small
        return "1"      # base
    return "2"          # default small for general CPUs

def check_vram_for_model(model_size):
    required  = VRAM_REQUIREMENTS.get(model_size, 2.5)
    if DEVICE != "cuda":
        return True, ""
    available = get_available_vram_gb()
    if available < required:
        return False, (
            f"'{model_size}' needs ~{required:.1f} GB VRAM, "
            f"only {available:.1f} GB free."
        )
    return True, ""

# ════════════════════════════════════════════════════════════
#  FFMPEG SETUP
# ════════════════════════════════════════════════════════════
def _ffmpeg_dl_progress(block_count, block_size, total_size):
    downloaded = block_count * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        sys.stdout.write(f"\r    {C.DIM}Downloading... {pct:.1f}%{C.RESET}")
        sys.stdout.flush()

def setup_ffmpeg():
    local_ff = APP_DIR / "ffmpeg.exe"
    local_fp = APP_DIR / "ffprobe.exe"

    if platform.system() == "Windows" and local_ff.exists() and local_fp.exists():
        Context.ffmpeg_cmd, Context.ffprobe_cmd = str(local_ff), str(local_fp)
        return True

    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        Context.ffmpeg_cmd, Context.ffprobe_cmd = "ffmpeg", "ffprobe"
        return True

    if platform.system() == "Windows":
        print(f"{C.YELLOW}  [~] FFmpeg not found. Downloading standalone binaries (~80 MB)...{C.RESET}")
        try:
            url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
            zip_path = APP_DIR / "ffmpeg_temp.zip"
            urllib.request.urlretrieve(url, zip_path, reporthook=_ffmpeg_dl_progress)
            print()

            found_ff = found_fp = False
            with zipfile.ZipFile(zip_path, 'r') as z:
                for fn in z.namelist():
                    name = Path(fn).name
                    if name == "ffmpeg.exe" and not found_ff:
                        with open(APP_DIR / "ffmpeg.exe", 'wb') as out:
                            out.write(z.read(fn))
                        found_ff = True
                    elif name == "ffprobe.exe" and not found_fp:
                        with open(APP_DIR / "ffprobe.exe", 'wb') as out:
                            out.write(z.read(fn))
                        found_fp = True
                    if found_ff and found_fp:
                        break

            zip_path.unlink()
            if local_ff.exists() and local_fp.exists():
                Context.ffmpeg_cmd, Context.ffprobe_cmd = str(local_ff), str(local_fp)
                print(f"{C.GREEN}  [+] FFmpeg installed.{C.RESET}\n")
                return True
        except Exception as e:
            print(f"\n{C.RED}  [x] FFmpeg download failed: {e}{C.RESET}")
            return False

    else:
        print(f"\n{C.RED}  [x] FFmpeg is missing. Please install it via your package manager:{C.RESET}")
        print(f"      {C.YELLOW}Ubuntu/Debian: sudo apt install ffmpeg{C.RESET}")
        print(f"      {C.YELLOW}macOS: brew install ffmpeg{C.RESET}")

    return False

# ════════════════════════════════════════════════════════════
#  UTILITIES
# ════════════════════════════════════════════════════════════
def natural_keys(text):
    """Sorts alphabetically but groups numbers properly (e.g. Ep_2 before Ep_10)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(text))]

def fmt_time(s):
    if s <= 0:
        return "0s"
    if s < 1:
        return "<1s"
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"

def fmt_srt_ts(t):
    """Accurately convert seconds to SRT timestamp, handling carry-over math safely."""
    t = max(0.0, t)
    ms = round(t * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def get_duration(media_path):
    if not Context.ffprobe_cmd:
        return 0.0
    try:
        cmd    = [
            Context.ffprobe_cmd, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(media_path)
        ]
        result = subprocess.check_output(cmd, encoding='utf-8', errors='replace', stderr=subprocess.DEVNULL).strip()
        val    = float(result)
        return val if val > 0 else 0.0
    except Exception:
        return 0.0

def safe_remove(path):
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
            unregister_temp_file(p)
    except OSError as e:
        unregister_temp_file(path)
        Context.failed_cleanups.append(f"{path} ({e.strerror or str(e)})")

def perform_vram_gc():
    if not Context.whisper_loaded:
        return
    if _TORCH_AVAILABLE:
        try:
            if _torch.cuda.is_available():
                gc.collect()
                _torch.cuda.empty_cache()
        except Exception:
            pass

# ════════════════════════════════════════════════════════════
#  SRT HEALTH CHECK
# ════════════════════════════════════════════════════════════
def is_valid_srt(srt_path, media_duration=0.0, min_blocks=3):
    """Returns (ok: bool, reason: str)"""
    try:
        text = Path(srt_path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return False, f"Cannot read file: {e}"

    blocks = [b.strip() for b in re.split(r'\n\n+', text.strip()) if b.strip()]
    if len(blocks) < min_blocks:
        return False, (
            f"Only {len(blocks)} block(s) — minimum health threshold is {min_blocks} block(s)."
        )

    ts_pat = re.compile(
        r'(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})'
    )

    def ts_to_sec(ts):
        try:
            ts    = ts.replace(',', '.')
            parts = ts.split(':')
            if len(parts) != 3:
                return None
            h, m, r = parts
            return int(h) * 3600 + int(m) * 60 + float(r)
        except (ValueError, AttributeError):
            return None

    durations = []
    lines     = []

    for block in blocks:
        match = ts_pat.search(block)
        if match:
            t_start = ts_to_sec(match.group(1))
            t_end   = ts_to_sec(match.group(2))
            if t_start is not None and t_end is not None:
                durations.append(max(0.0, t_end - t_start))
        for ln in block.splitlines()[2:]:
            stripped = ln.strip()
            if stripped:
                lines.append(stripped.lower())

    if not durations:
        return False, "No valid timestamps found."

    avg = sum(durations) / len(durations)
    if avg > SRT_MAX_AVG_DURATION:
        return False, f"Avg block duration {avg:.1f}s > {SRT_MAX_AVG_DURATION}s — likely hallucination."
    if avg < SRT_MIN_AVG_DURATION:
        return False, f"Avg block duration {avg:.3f}s < {SRT_MIN_AVG_DURATION}s — flash hallucination."

    if lines:
        dup = 1.0 - len(set(lines)) / len(lines)
        if dup > SRT_DUP_RATIO:
            return False, f"Duplicate line ratio {dup*100:.0f}% > {SRT_DUP_RATIO*100:.0f}% — looping hallucination."

    return True, "OK"

# ════════════════════════════════════════════════════════════
#  STRUCTURAL SRT ALIGNMENT (PRESERVE TIMESTAMPS)
# ════════════════════════════════════════════════════════════
def align_translated_chunk(source_chunk_text, translated_chunk_text):
    """
    Stitches Gemini's dialogue block translations back onto the original sequence indices and timestamps.
    Returns (aligned_text, fallback_count, structural_mismatch_detected)
    """
    src_blocks = [b.strip() for b in re.split(r'\n\n+', source_chunk_text.strip()) if b.strip()]
    
    clean_tgt_dialogue = []
    for block in re.split(r'\n\n+', translated_chunk_text.strip()):
        lines = block.splitlines()
        dialogue_lines = []
        for ln in lines:
            ln_stripped = ln.strip()
            if not ln_stripped:
                continue
            # Filter out index and timeline formatting anomalies
            if ln_stripped.isdigit():
                continue
            if "-->" in ln_stripped:
                continue
            dialogue_lines.append(ln_stripped)
        if dialogue_lines:
            clean_tgt_dialogue.append("\n".join(dialogue_lines))
            
    aligned_blocks = []
    fallback_count = 0
    
    # Strict ratio-based check for block mismatches
    src_len = len(src_blocks)
    tgt_len = len(clean_tgt_dialogue)
    diff = abs(src_len - tgt_len)
    
    # Variance is flagged if it exceeds a tight 5% threshold (or more than 1 block for small chunks)
    mismatch_flag = diff > 1 or (src_len > 0 and (diff / src_len) > 0.05)

    for idx, src_block in enumerate(src_blocks):
        lines = src_block.splitlines()
        if len(lines) < 2:
            continue
        seq_num   = lines[0].strip()
        timestamp = lines[1].strip()
        
        if idx < len(clean_tgt_dialogue):
            dialogue = clean_tgt_dialogue[idx]
        else:
            # Fall back to original dialogue to preserve timing integrity
            dialogue = "\n".join(lines[2:])
            fallback_count += 1
            
        aligned_blocks.append(f"{seq_num}\n{timestamp}\n{dialogue}")
        
    return "\n\n".join(aligned_blocks), fallback_count, mismatch_flag

# ════════════════════════════════════════════════════════════
#  MUX VALIDATION
# ════════════════════════════════════════════════════════════
def verify_mux_output(output_path, hardsub=False):
    """Returns (ok: bool, reason: str)"""
    p = Path(output_path)
    if not p.exists():
        return False, "Output file does not exist."

    size = p.stat().st_size
    if size < MUXED_MIN_BYTES:
        return False, f"Output is only {size} bytes — likely a failed encode."

    if hardsub:
        return True, "OK"

    if not Context.ffprobe_cmd:
        return True, "OK (ffprobe unavailable — stream check skipped)"

    try:
        cmd    = [
            Context.ffprobe_cmd, "-v", "error",
            "-select_streams", "s",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(output_path)
        ]
        result = subprocess.check_output(cmd, encoding='utf-8', errors='replace', stderr=subprocess.DEVNULL).strip()
        if result:
            return True, "OK"
        return False, "No subtitle stream in output."
    except Exception as e:
        return False, f"ffprobe check failed: {e}"

# ════════════════════════════════════════════════════════════
#  GEMINI TRANSLATION ENGINE
# ════════════════════════════════════════════════════════════
def translate_srt_native(src_path, tgt_path, target_lang, api_key):
    """
    Returns (success: bool, msg: str, total_fallbacks: int)
    """
    if not api_key or not api_key.strip():
        return False, "No API key provided.", 0

    try:
        content = Path(src_path).read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Cannot read source SRT: {e}", 0

    blocks = [b.strip() for b in re.split(r'\n\n+', content.strip()) if b.strip()]
    if not blocks:
        return False, "Source SRT is empty.", 0

    source_count   = len(blocks)
    translated_srt = []
    url            = GEMINI_URL_TEMPLATE.format(key=api_key)
    total_chunks   = (source_count + GEMINI_CHUNK_SIZE - 1) // GEMINI_CHUNK_SIZE
    max_attempts   = 4
    total_fallbacks = 0

    for ci, i in enumerate(range(0, source_count, GEMINI_CHUNK_SIZE)):
        chunk_blocks = blocks[i : i + GEMINI_CHUNK_SIZE]
        chunk_text   = "\n\n".join(chunk_blocks)
        prompt       = (
            f"You are a professional subtitle translator. "
            f"Translate the following SRT subtitles into {target_lang}. "
            "Preserve exact sequence numbers and timestamps. "
            "Translate dialogue only. No notes, no extra text.\n\n"
            f"{chunk_text}"
        )
        payload = {
            "contents":         [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1}
        }
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )

        for attempt in range(max_attempts):
            try:
                with urllib.request.urlopen(req, timeout=GEMINI_TIMEOUT) as resp:
                    res_data  = json.loads(resp.read().decode("utf-8"))
                    candidate = res_data.get("candidates", [{}])[0]
                    text_out  = (
                        candidate
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "")
                    )
                    
                    if not text_out:
                        finish = candidate.get("finishReason", "unknown")
                        return False, f"Empty response (Reason: {finish})", 0
                    
                    # Structurally map dialogue to pristine templates
                    aligned_chunk, fallbacks, structural_mismatch = align_translated_chunk(chunk_text, text_out.strip())
                    
                    if structural_mismatch:
                        if attempt < max_attempts - 1:
                            # Apply strictly exponential retry delay: 5s, 10s, 20s...
                            time.sleep(5 * (2 ** attempt))
                            continue
                        return False, "Translation structure mismatched (differing block counts) persistently.", 0

                    # Verify if too many blocks were lost to fallbacks
                    if fallbacks > len(chunk_blocks) * 0.2:
                        if attempt < max_attempts - 1:
                            time.sleep(5 * (2 ** attempt))
                            continue
                        return False, f"Translation dropped too many blocks ({fallbacks} fell back to source).", 0

                    total_fallbacks += fallbacks
                    translated_srt.append(aligned_chunk)
                    
                    # Successful request -> clear consecutive limits
                    Context.consecutive_429s = 0
                    Context.consecutive_total_failures = 0
                    break

            except urllib.error.HTTPError as e:
                Context.consecutive_total_failures += 1
                if e.code == 429:
                    Context.consecutive_429s += 1
                    if Context.consecutive_429s >= CONSECUTIVE_429_LIMIT:
                        Context.translation_disabled = True
                        return False, "API Quota exceeded (HTTP 429) sequentially. Translation suspended.", 0
                    if attempt < max_attempts - 1:
                        time.sleep(10 * (attempt + 1))
                        continue
                    return False, "API Quota hit repeatedly.", 0
                
                # Generic HTTP errors use exponential backoff delays
                if attempt < max_attempts - 1:
                    time.sleep(5 * (2 ** attempt))
                else:
                    return False, f"HTTP {e.code} after retries.", 0
            except Exception as e:
                Context.consecutive_total_failures += 1
                if attempt < max_attempts - 1:
                    time.sleep(5 * (2 ** attempt))
                else:
                    return False, f"Network request failed: {e}", 0

        if ci < total_chunks - 1:
            time.sleep(GEMINI_INTER_CHUNK_DELAY)

    tgt_path = Path(tgt_path)
    tgt_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = tgt_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(translated_srt) + "\n")
        tmp_path.replace(tgt_path)
    except Exception as e:
        safe_remove(tmp_path)
        return False, f"Write error: {e}", 0

    return True, "Success", total_fallbacks

# ════════════════════════════════════════════════════════════
#  WHISPER LOADER
# ════════════════════════════════════════════════════════════
def load_whisper(model_size):
    with Context.whisper_lock:
        if Context.whisper_loaded:
            return

        if DEVICE == "cpu" and model_size in ["medium", "large-v3-turbo", "large-v3"]:
            print(f"  {C.YELLOW}[!] CPU Warning: '{model_size}' runs very slowly on CPU. Use a smaller model for reasonable speeds.{C.RESET}")

        ok, reason = check_vram_for_model(model_size)
        if not ok:
            print(f"  {C.YELLOW}[!] VRAM Warning: {reason} — may OOM.{C.RESET}")

        from faster_whisper import WhisperModel
        try:
            xdg_cache  = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
            hf_home    = Path(os.environ.get("HF_HOME", Path(xdg_cache) / "huggingface"))
            cache_dir  = hf_home / "hub"
            model_slug = f"Systran/faster-whisper-{model_size}"
            model_cached = any(
                model_slug.replace("/", "--") in d.name
                for d in cache_dir.iterdir()
                if d.is_dir()
            ) if cache_dir.exists() else False
        except Exception:
            model_cached = False

        if model_cached:
            print(f"  {C.DIM}[~] Loading Whisper (Model: '{model_size}' · Device: '{DEVICE}')...{C.RESET}")
        else:
            size_hint = {
                "tiny": "~150 MB", "base": "~290 MB", "small": "~970 MB",
                "medium": "~3.1 GB", "large-v3-turbo": "~3.1 GB", "large-v3": "~6.2 GB"
            }.get(model_size, "")
            print(f"  {C.YELLOW}[~] Downloading Whisper model '{model_size}' {size_hint} — first-time only...{C.RESET}")

        Context.whisper_model  = WhisperModel(model_size, device=DEVICE, compute_type=COMPUTE)
        Context.whisper_loaded = True

# ════════════════════════════════════════════════════════════
#  CORE PIPELINE
# ════════════════════════════════════════════════════════════
def process_file(media_path, args, file_index=1, total_files=1):
    media_p    = Path(media_path)
    base       = media_p.stem
    t_start    = time.time()
    temp_audio = media_p.parent / f"temp_{base}_audio.wav"
    status     = {
        "transcribed":    False,
        "translated":     False,
        "reused_srt":     False,
        "reused_all":     False,
        "mixed_language": False,
        "fallback_count": 0,
        "muxed":          False,
        "skipped":        False,
        "error":          False,
    }

    print(f"\n{C.BOLD}── [{file_index}/{total_files}] {base}{C.RESET}")

    if not media_p.exists():
        print(f"  {C.RED}[x] File no longer exists — skipping.{C.RESET}")
        status["error"] = True
        return status, 0.0

    safe_remove(temp_audio)

    srt_src  = media_p.parent / f"{base}.subs-pipeline.srt"
    srt_tgt  = media_p.parent / f"{base}.{args.tgt_ext}.srt"
    out_ext  = "mp4" if args.hardsub else "mkv"
    out_path = media_p.parent / f"muxed_{args.tgt_ext}" / f"{base}.{out_ext}"

    # Trace whether output was already completed previously
    if out_path.exists() and out_path.stat().st_size >= MUXED_MIN_BYTES:
        print(f"  {C.GREEN}[+] Output file already exists. Skipping processing.{C.RESET}")
        status["reused_all"] = True
        return status, 0.0

    duration = get_duration(media_path)
    if duration <= 0:
        print(f"  {C.YELLOW}[!] Could not determine duration — progress % unavailable.{C.RESET}")

    try:
        # ── STEP 1: AUDIO EXTRACTION ──────────────────────
        if args.transcribe and not srt_src.exists() and not srt_tgt.exists():
            print(f"  {C.CYAN}> Extracting audio...{C.RESET}")
            register_temp_file(temp_audio)
            result = subprocess.run(
                [
                    Context.ffmpeg_cmd, "-y", "-v", "error",
                    "-i", str(media_path),
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    str(temp_audio)
                ],
                capture_output=True
            )
            if result.returncode != 0 or not temp_audio.exists():
                stderr_msg = result.stderr.decode('utf-8', errors="replace").strip()
                print(f"  {C.RED}[x] Audio extraction failed.{C.RESET}")
                if stderr_msg:
                    print(f"      {C.DIM}{stderr_msg}{C.RESET}")
                print(f"  {C.YELLOW}[!] Transcription skipped due to audio failure.{C.RESET}")

        # ── STEP 2: TRANSCRIPTION ─────────────────────────
        if args.transcribe and not srt_src.exists() and not srt_tgt.exists() and temp_audio.exists():
            load_whisper(args.model)
            print(f"  {C.CYAN}> Transcribing...{C.RESET}")
            tmp_srt = srt_src.with_suffix(".tmp")
            register_temp_file(tmp_srt)
            try:
                lang_hint      = args.src_lang if args.src_lang else None
                segments, info = Context.whisper_model.transcribe(
                    str(temp_audio), beam_size=5, language=lang_hint
                )
                with open(tmp_srt, "w", encoding="utf-8") as f:
                    for idx, seg in enumerate(segments, 1):
                        f.write(
                            f"{idx}\n"
                            f"{fmt_srt_ts(seg.start)} --> {fmt_srt_ts(seg.end)}\n"
                            f"{seg.text.strip()}\n\n"
                        )
                        if duration > 0:
                            pct = min(seg.end / duration * 100, 100.0)
                            sys.stdout.write(
                                f"\r    {C.DIM}{fmt_time(seg.end)} / "
                                f"{fmt_time(duration)} ({pct:.1f}%){C.RESET}   "
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
                    f"(detected: {info.language}){C.RESET}"
                )
                status["transcribed"] = True

            except Exception as e:
                print(f"\n  {C.RED}[x] Transcription error: {e}{C.RESET}")
                safe_remove(tmp_srt)
            finally:
                safe_remove(temp_audio)

        # ── SRT HEALTH CHECK ──────────────────────────────
        srt_src_healthy = True
        if srt_src.exists() and not srt_tgt.exists():
            ok, reason = is_valid_srt(srt_src, duration, args.min_blocks)
            if not ok:
                print(f"  {C.YELLOW}[!] SRT health check failed: {reason}{C.RESET}")
                print(f"  {C.YELLOW}    Translation skipped.{C.RESET}")
                srt_src_healthy   = False
                status["skipped"] = True

        # ── STEP 3: TRANSLATION ───────────────────────────
        if (args.translate
                and not Context.translation_disabled
                and not srt_tgt.exists()
                and srt_src.exists()
                and srt_src_healthy
                and args.api_key):
            
            if Context.consecutive_total_failures >= CONSECUTIVE_TOTAL_FAIL_LIMIT:
                Context.translation_disabled = True
                print(f"  {C.RED}[!] Translation system suspended for this batch due to persistent failures.{C.RESET}")
            else:
                print(f"  {C.CYAN}> Translating -> {args.tgt_lang}...{C.RESET}")
                success, msg, fallbacks = translate_srt_native(
                    srt_src, srt_tgt, args.tgt_lang, args.api_key
                )
                if success:
                    if fallbacks > 0:
                        status["mixed_language"] = True
                        status["fallback_count"] = fallbacks
                        print(f"  {C.YELLOW}[~] Translation finished. {fallbacks} blocks fell back to the original source language.{C.RESET}")
                    else:
                        status["translated"] = True
                        print(f"  {C.GREEN}[+] Translation done.{C.RESET}")
                else:
                    print(f"  {C.RED}[x] Translation skipped/failed: {msg}{C.RESET}")

        # ── STEP 4: EMBED ─────────────────────────────────
        is_audio = media_p.suffix.lower() in AUDIO_EXTS
        target_srt = None

        if srt_tgt.exists():
            target_srt = srt_tgt
            if not status["translated"] and not status["mixed_language"]:
                status["reused_srt"] = True
        elif srt_src.exists() and srt_src_healthy:
            target_srt = srt_src
            if args.translate:
                print(f"  {C.YELLOW}[!] Using source SRT — translated SRT unavailable.{C.RESET}")

        # Create output directory safely
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"  {C.RED}[x] Failed to create output directory: {e}{C.RESET}")
            status["error"] = True
            return status, time.time() - t_start

        if out_path.exists():
            if out_path.stat().st_size < MUXED_MIN_BYTES:
                print(f"  {C.YELLOW}[!] Removing corrupt output from previous run.{C.RESET}")
                safe_remove(out_path)

        if args.embed and target_srt and not out_path.exists():
            if is_audio:
                print(f"  {C.DIM}[~] Audio-only file — muxing skipped.{C.RESET}")
            else:
                print(f"  {C.CYAN}> Muxing...{C.RESET}")

                try:
                    if args.hardsub:
                        # Collision-resistant UUID filename matching Windows ffmpeg standards
                        cwd_path   = media_p.parent
                        rel_media  = media_p.name
                        rel_out    = out_path.relative_to(cwd_path)
                        unique_id  = uuid.uuid4().hex[:8]
                        temp_srt   = f"temp_hardsub_{base}_{unique_id}.srt"
                        
                        shutil.copy(target_srt, cwd_path / temp_srt)
                        register_temp_file(cwd_path / temp_srt)

                        cmd = [
                            Context.ffmpeg_cmd, "-y", "-v", "error",
                            "-i", rel_media,
                            "-vf", f"subtitles={temp_srt}",
                            "-c:a", "copy",
                            str(rel_out)
                        ]
                        result = subprocess.run(cmd, capture_output=True, cwd=cwd_path)
                        safe_remove(cwd_path / temp_srt)

                    else:
                        cmd = [
                            Context.ffmpeg_cmd, "-y", "-v", "error",
                            "-i", str(media_path),
                            "-i", str(target_srt),
                            "-c:v", "copy", "-c:a", "copy", "-c:s", "srt",
                            "-metadata:s:s:0", f"language={args.tgt_ext}",
                            str(out_path)
                        ]
                        result = subprocess.run(cmd, capture_output=True)

                    if result.returncode != 0:
                        err = result.stderr.decode('utf-8', errors="replace").strip()
                        if err:
                            print(f"  {C.DIM}    FFmpeg: {err[:200]}{C.RESET}")

                    ok, reason = verify_mux_output(out_path, hardsub=args.hardsub)
                    if ok:
                        print(
                            f"  {C.GREEN}[+] -> "
                            f"{out_path.parent.name}\\{out_path.name}{C.RESET}"
                        )
                        status["muxed"] = True
                    else:
                        print(f"  {C.YELLOW}[!] Mux validation: {reason}{C.RESET}")

                except Exception as e:
                    print(f"  {C.RED}[x] Muxing error: {e}{C.RESET}")

    except Exception as e:
        print(f"  {C.RED}[x] Unexpected error: {e}{C.RESET}")
        status["error"] = True

    finally:
        safe_remove(temp_audio)
        perform_vram_gc()

    return status, time.time() - t_start

# ════════════════════════════════════════════════════════════
#  BATCH SUMMARY
# ════════════════════════════════════════════════════════════
def print_summary(summary, total_elapsed):
    W   = 68
    top = "╔" + "═" * W + "╗"
    mid = "╠" + "═" * W + "╣"
    bot = "╚" + "═" * W + "╝"

    print(f"\n{C.BOLD}{C.CYAN}{top}")
    title = f"  Batch Complete — {fmt_time(total_elapsed)}"
    print(f"║{title:<{W}}║")
    print(f"{mid}{C.RESET}")

    for name, st, t in summary:
        t_str = fmt_time(t).rjust(6)

        if st.get("error"):
            flag = f"{C.RED}[FAULT   ]{C.RESET}"
        elif st.get("skipped"):
            flag = f"{C.YELLOW}[SKIPPED ]{C.RESET}"
        elif st.get("reused_all"):
            flag = f"{C.GREEN}[REUSED  ]{C.RESET}"
        elif st.get("translated") and st.get("muxed"):
            flag = f"{C.GREEN}[DONE_TRN]{C.RESET}"
        elif st.get("mixed_language") and st.get("muxed"):
            # Completed with mixed-language structural fallbacks
            flag = f"{C.YELLOW}[DONE_MIX]{C.RESET}"
        elif st.get("reused_srt") and st.get("muxed"):
            flag = f"{C.GREEN}[DONE_MUX]{C.RESET}"
        elif st.get("transcribed") and (st.get("translated") or st.get("mixed_language")):
            flag = f"{C.CYAN}[TXT+TRN ]{C.RESET}"
        elif st.get("transcribed"):
            flag = f"{C.CYAN}[TXT_ONLY]{C.RESET}"
        elif st.get("translated") or st.get("reused_srt") or st.get("mixed_language"):
            flag = f"{C.CYAN}[TRN_ONLY]{C.RESET}"
        else:
            flag = f"{C.DIM}[NO-OP   ]{C.RESET}"

        flag_raw   = strip_ansi(flag)
        name_width = W - 2 - len(flag_raw) - 1 - len(t_str) - 1
        name_trunc = (name[:name_width - 1] + "~") if len(name) > name_width else name.ljust(name_width)

        print(f"║  {flag} {name_trunc} {t_str} ║")

    print(f"{C.BOLD}{C.CYAN}{bot}{C.RESET}")

# ════════════════════════════════════════════════════════════
#  DIRECTORY WATCHER
# ════════════════════════════════════════════════════════════
def run_watcher(args):
    try:
        from watchdog.observers import Observer
        from watchdog.events    import FileSystemEventHandler
    except ImportError:
        print(f"\n{C.YELLOW}  [!] watchdog not installed: pip install watchdog{C.RESET}")
        return

    in_flight    = set()
    lock         = threading.Lock()
    muxed_folder = f"muxed_{args.tgt_ext}"
    folder_root  = Path(args.folder).resolve()

    class WatchHandler(FileSystemEventHandler):
        def _dispatch(self, path):
            resolved = str(Path(path).resolve())
            if not resolved.lower().endswith(MEDIA_EXTS):
                return
            try:
                rel = Path(resolved).relative_to(folder_root)
                if rel.parts and rel.parts[0] == muxed_folder:
                    return
            except ValueError:
                return

            with lock:
                if resolved in in_flight:
                    return
                in_flight.add(resolved)

            def handle():
                try:
                    print(f"\n{C.CYAN}  [*] Detected: {Path(resolved).name}{C.RESET}")
                    time.sleep(WATCHER_SETTLE_SECS)
                    process_file(resolved, args)
                    print(f"{C.GREEN}  [*] Idle — awaiting files...{C.RESET}")
                finally:
                    with lock:
                        in_flight.discard(resolved)

            threading.Thread(target=handle, daemon=True).start()

        def on_created(self, event):
            if not event.is_directory:
                self._dispatch(event.src_path)

        def on_moved(self, event):
            if not event.is_directory:
                self._dispatch(event.dest_path)

    observer = Observer()
    observer.schedule(WatchHandler(), path=args.folder, recursive=False)
    observer.start()

    print(f"\n{C.CYAN}  [*] Watch Mode — '{Path(args.folder).name}'{C.RESET}")
    print(f"  {C.DIM}Ctrl+C to stop.{C.RESET}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# ════════════════════════════════════════════════════════════
#  INTERACTIVE WIZARD
# ════════════════════════════════════════════════════════════
def interactive_wizard(args, cfg_memory):
    print(f"\n{C.CYAN}{C.BOLD}", end="")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  Subs-pipeline                                             ║")
    print(f"╚════════════════════════════════════════════════════════════╝{C.RESET}\n")

    if Context.config_warning:
        print(f"  {C.RED}[!] CONFIG SYSTEM: {Context.config_warning}{C.RESET}\n")

    if not setup_ffmpeg():
        print(f"{C.RED}  [!] FFmpeg is required to continue.{C.RESET}")
        exit_app(1)

    # ── Step 1: Folder ────────────────────────────────────
    current_dir = os.getcwd()
    print(f"{C.BOLD}> Step 1: Media Directory{C.RESET}")
    print(f"  Current: {C.CYAN}{current_dir}{C.RESET}")
    f_in = input(
        f"  [Enter = current  |  path  |  'b' = browse]: "
    ).strip()

    if not f_in:
        args.folder = current_dir
    elif f_in.lower() == "b":
        selected    = ask_directory_gui()
        args.folder = selected if selected else current_dir
        if not selected:
            print(f"  {C.YELLOW}  Falling back to current directory.{C.RESET}")
    else:
        args.folder = str(Path(f_in).resolve())

    if not args.folder or not Path(args.folder).is_dir():
        print(f"{C.RED}  [!] Invalid directory.{C.RESET}")
        exit_app(1)
    print(f"  {C.GREEN}-> {args.folder}{C.RESET}\n")

    # Run non-destructive garbage collection sweep on folder load
    startup_garbage_collection(args.folder, skip_cleanup=args.no_cleanup)

    # ── Step 2: Saved Profiles Fast Path ──────────────────
    if cfg_memory:
        print(f"\n{C.BOLD}> Step 2: Use Saved Settings Profile?{C.RESET}")
        print(f"  Target Language:     {C.CYAN}{cfg_memory.get('tgt_lang', 'English')}{C.RESET}")
        print(f"  Subtitle Extension:  {C.CYAN}{cfg_memory.get('tgt_ext', 'en')}{C.RESET}")
        print(f"  Min Blocks Required: {C.CYAN}{cfg_memory.get('min_blocks', 3)}{C.RESET}")
        print(f"  Detected Device:     {C.CYAN}{DEVICE} ({COMPUTE}){C.RESET}")
        
        fast_path = input(f"\n  Use saved settings for {Path(args.folder).name}? [Y/n]: ").strip().lower() != "n"
        if fast_path:
            args.tgt_lang   = cfg_memory.get('tgt_lang', 'English')
            args.tgt_ext    = cfg_memory.get('tgt_ext', 'en')
            args.src_lang   = cfg_memory.get('src_lang') or None
            if args.src_lang and not args.src_lang.strip():
                args.src_lang = None
            args.api_key    = cfg_memory.get('api_key', '')
            args.translate  = bool(args.api_key)
            args.min_blocks = int(cfg_memory.get('min_blocks', 3))
            
            recommended     = recommend_whisper_model()
            args.model      = MODEL_MAP.get(recommended, "small")
            
            print(f"  {C.GREEN}-> Model Confirmed: {args.model.upper()}{C.RESET}")
            args.hardsub    = False
            args.transcribe = True
            args.embed      = True
            args.watch      = False
            print(f"  {C.GREEN}[+] Loaded settings successfully. Proceeding directly...{C.RESET}\n")
            return

    # ── Step 3: Translation settings ─────────────────────
    print(f"{C.BOLD}> Step 3: Translation Settings{C.RESET}")
    args.tgt_lang = (
        input(f"  Target Language [{cfg_memory.get('tgt_lang', 'English')}]: ").strip()
        or cfg_memory.get("tgt_lang", "English")
    )
    
    if len(args.tgt_lang) < 2 or not args.tgt_lang.replace(' ', '').isalpha():
        print(f"  {C.YELLOW}  [!] Warning: '{args.tgt_lang}' might not be a valid language name.{C.RESET}")

    args.tgt_ext  = (
        input(f"  Subtitle Extension (e.g. en, ar) [{cfg_memory.get('tgt_ext', 'en')}]: ")
        .strip().lower()
        or cfg_memory.get("tgt_ext", "en")
    )
    if len(args.tgt_ext) > 5 or " " in args.tgt_ext:
        print(f"  {C.YELLOW}  [!] Warning: '{args.tgt_ext}' looks unusual for a subtitle extension.{C.RESET}")

    # ── Step 4: Source language ───────────────────────────
    saved_src = cfg_memory.get("src_lang") or ""
    print(f"\n{C.BOLD}> Step 4: Source Language{C.RESET}")
    print(f"  {C.DIM}Blank = auto-detect  |  ISO codes: ja  en  ko  zh  ar  es ...{C.RESET}")
    src_in        = input(f"  [{saved_src or 'auto'}]: ").strip().lower()
    args.src_lang = src_in if src_in else (saved_src if saved_src else None)

    # ── Step 5: Whisper model ─────────────────────────────
    vram_avail  = get_available_vram_gb()
    vram_label  = f"{vram_avail:.1f} GB free" if DEVICE == "cuda" else "CPU mode"
    recommended = recommend_whisper_model()
    
    print(f"\n{C.BOLD}> Step 5: Transcription Model (Recommended: {MODEL_MAP[recommended].upper()}){C.RESET}")
    print(f"  {C.DIM}Device: {DEVICE}  ({vram_label}){C.RESET}")
    print(f"    {C.CYAN}[0]{C.RESET} Tiny           ~1.0 GB  Fastest")
    print(f"    {C.CYAN}[1]{C.RESET} Base           ~1.5 GB  Fast")
    print(f"    {C.CYAN}[2]{C.RESET} Small          ~2.5 GB  Recommended")
    print(f"    {C.CYAN}[3]{C.RESET} Medium         ~5.0 GB  Better accuracy")
    print(f"    {C.CYAN}[4]{C.RESET} Large-v3 Turbo ~6.0 GB  Fast + accurate")
    print(f"    {C.CYAN}[5]{C.RESET} Large-v3       ~10  GB  Best accuracy")
    
    args.model = MODEL_MAP.get(input(f"  Selection [{recommended}]: ").strip() or recommended, "small")
    print(f"  {C.GREEN}-> Selected Model: {args.model.upper()}{C.RESET}")

    # ── Step 6: API Key ───────────────────────────────────
    saved_key = cfg_memory.get("api_key", "")
    print(f"\n{C.BOLD}> Step 6: Gemini API Key{C.RESET}")
    if saved_key:
        print(f"  {C.DIM}Stored key found. Press Enter to reuse.{C.RESET}")
    args.api_key   = input("  Key: ").strip() or saved_key
    args.translate = bool(args.api_key)
    if not args.api_key:
        print(f"  {C.YELLOW}  No key — translation will be skipped.{C.RESET}")

    # ── Step 7: Output format ─────────────────────────────
    print(f"\n{C.BOLD}> Step 7: Output Format{C.RESET}")
    print(f"    {C.CYAN}[0]{C.RESET} Softsub — toggleable track  (default)")
    print(f"    {C.CYAN}[1]{C.RESET} Hardsub — burned into video")
    args.hardsub = input("  Selection [0]: ").strip() == "1"

    # ── Step 8: Subtitle Validation ────────────────────────
    print(f"\n{C.BOLD}> Step 8: Health Check Settings{C.RESET}")
    print(f"  {C.DIM}Prevents rendering bad hallucinated loops or empty subtitles.{C.RESET}")
    print(f"  {C.DIM}Set to 1 for tiny clips, 3 for standard episodes.{C.RESET}")
    saved_min = cfg_memory.get("min_blocks", 3)
    try:
        val_blocks = input(f"  Minimum valid blocks [{saved_min}]: ").strip()
        args.min_blocks = int(val_blocks) if val_blocks else int(saved_min)
    except ValueError:
        args.min_blocks = 3

    # ── Step 9: Pipeline Steps ────────────────────────────
    print(f"\n{C.BOLD}> Step 9: Pipeline Steps{C.RESET}  {C.DIM}(skip if files already exist){C.RESET}")
    args.transcribe = input("  Transcribe? [Y/n]: ").strip().lower() != "n"
    if args.translate:
        args.translate = input("  Translate?  [Y/n]: ").strip().lower() != "n"
    args.embed = input("  Mux?        [Y/n]: ").strip().lower() != "n"

    if args.hardsub and not args.embed:
        print(f"  {C.YELLOW}  Note: Hardsub selected but Mux is disabled — hardsub setting has no effect.{C.RESET}")

    # ── Step 10: Watch mode ────────────────────────────────
    args.watch = input(f"\n{C.BOLD}> Step 10: Watch Mode?{C.RESET} [y/N]: ").strip().lower() == "y"

    save_config({
        "api_key":    args.api_key,
        "tgt_lang":   args.tgt_lang,
        "tgt_ext":    args.tgt_ext,
        "src_lang":   args.src_lang or "",
        "min_blocks": args.min_blocks
    })

# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Standalone Media Transcription & Translation Pipeline"
    )
    parser.add_argument("--headless",        action="store_true")
    parser.add_argument("--folder",          type=str)
    parser.add_argument("--model",           type=str, default="small")
    parser.add_argument("--src_lang",        type=str, default=cfg.get("src_lang") or None)
    parser.add_argument("--tgt_lang",        type=str, default=cfg.get("tgt_lang", "English"))
    parser.add_argument("--tgt_ext",         type=str, default=cfg.get("tgt_ext",  "en"))
    parser.add_argument("--api_key",         type=str, default=cfg.get("api_key",  ""))
    parser.add_argument("--hardsub",         action="store_true")
    parser.add_argument("--min_blocks",      type=int, default=int(cfg.get("min_blocks", 3)))
    parser.add_argument("--skip_transcribe", action="store_true")
    parser.add_argument("--skip_translate",  action="store_true")
    parser.add_argument("--skip_embed",      action="store_true")
    parser.add_argument("--watch",           action="store_true")
    parser.add_argument("--no_cleanup",      action="store_true")
    args = parser.parse_args()

    # Reset and configure context diagnostic baselines
    Context.active_temp_files          = set()
    Context.failed_cleanups            = []
    Context.translation_disabled       = False
    Context.consecutive_429s           = 0
    Context.consecutive_total_failures = 0

    # Read cleanup requirements
    args.no_cleanup = args.no_cleanup or cfg.get("skip_cleanup", False)

    # Pre-test config system safely
    diag_status = verify_config_status()
    if diag_status is not True:
        Context.config_warning = diag_status

    if args.src_lang and not args.src_lang.strip():
        args.src_lang = None

    args.transcribe = not args.skip_transcribe
    args.translate  = not args.skip_translate and bool(args.api_key)
    args.embed      = not args.skip_embed

    check_dependencies(headless=args.headless)

    if not args.headless:
        interactive_wizard(args, cfg)
    else:
        if Context.config_warning:
            print(f"  [!] startup: {Context.config_warning}")
        if not setup_ffmpeg():
            sys.exit(1)
        if not args.folder or not Path(args.folder).is_dir():
            print(f"{C.RED}[!] --folder must be a valid directory.{C.RESET}")
            sys.exit(1)
        # Headless sweep of paths
        startup_garbage_collection(args.folder, skip_cleanup=args.no_cleanup)

    # Final visual print of config execution defaults
    if args.headless:
        print(f"  Effective Settings: Target Lang={args.tgt_lang} | Sub Ext={args.tgt_ext} | Min Blocks={args.min_blocks}")

    try:
        media_files = sorted([
            str(p) for p in Path(args.folder).iterdir()
            if p.is_file() and p.suffix.lower() in MEDIA_EXTS
        ], key=lambda x: natural_keys(Path(x).name))
    except PermissionError:
        print(f"{C.RED}[!] Permission denied accessing directory: {args.folder}{C.RESET}")
        sys.exit(1)

    summary     = []
    batch_start = time.time()

    if media_files:
        for i, file in enumerate(media_files, 1):
            try:
                status, elapsed = process_file(file, args, i, len(media_files))
                summary.append((Path(file).name, status, elapsed))
            except KeyboardInterrupt:
                print(f"\n{C.YELLOW}[!] Interrupted.{C.RESET}")
                summary.append((Path(file).name, {"error": True}, 0.0))
                break
            except Exception as e:
                print(f"{C.RED}  [x] Fault on {Path(file).name}: {e}{C.RESET}")
                summary.append((Path(file).name, {"error": True}, 0.0))

        print_summary(summary, time.time() - batch_start)
    else:
        print(f"{C.YELLOW}\n  No compatible media files found.{C.RESET}")

    if args.watch:
        run_watcher(args)
    else:
        exit_app(0)

# ════════════════════════════════════════════════════════════
#  BOOTSTRAP
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}[!] Interrupted.{C.RESET}")
        exit_app(0)
    except Exception as e:
        print(f"\n{C.RED}[x] FATAL: {e}{C.RESET}")
        import traceback
        traceback.print_exc()
        exit_app(1)
