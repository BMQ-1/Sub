# 🎬 Subs-pipeline

An intelligent, single-file media transcription, structural translation, and muxing pipeline. This application automates the process of generating accurate, multi-language subtitles locally using `faster-whisper` and translating them through the Google Gemini API, while ensuring timestamps remain synchronized.

---

## ✨ Features

- 📦 **Monolithic Single File**: The entire application is self-contained in a single, refined `.py` file for minimal setup and maximum portability.
- 🎙️ **Local High-Performance Transcription**: Driven by `faster-whisper` (utilizing CTranslate2), offering faster execution times than standard Whisper implementations.
- 🌐 **Structural SRT Translation**: Leverages the fast, affordable `gemini-3.5-flash` model for high-context natural subtitle translation.
- ⏱️ **Structural Alignment Engine**: A strict ratio-based alignment system maps translated dialogue directly onto source templates. It preserves the sequence numbers and millisecond timestamps of the original transcription, neutralizing model layout formatting anomalies.
- 🛡️ **Interactive Profile Fast-Path**: Returning users can bypass setup prompts with a single keypress, loading custom settings instantly.
- 📺 **Softsub or Hardsub Muxing**: Automatically embeds generated subtitles into MKV containers as native, selectable text tracks (softsubs) or burns them directly into MP4 containers (hardsubs).
- 🧹 **Robust Cleanup and Sweeping**: Tracks temp files and cleans up directories on crash recovery, keeping the workspace tidy.
- 🔄 **Watch Mode**: Monitors a directory continuously, processing compatible media files as soon as they appear.

---

## 🛠️ Requirements & Prerequisites

To run this pipeline from source, you will need:

1. **Python 3.8 to 3.11** (Note: `faster-whisper` might experience library compatibility issues on newer Python releases like 3.12+ depending on system platform bindings).
2. **FFmpeg & FFprobe**: 
   - **Windows**: The script will automatically attempt to download standalone binaries locally on first run if they are not in your system environment path.
   - **macOS / Linux**: Install via package manager:
     ```bash
     # Ubuntu/Debian
     sudo apt install ffmpeg
     
     # macOS (Homebrew)
     brew install ffmpeg
     ```
3. **Gemini API Key** (Required for Translation only):
   - Obtain an API key from [Google AI Studio](https://aistudio.google.com/).

---

## 🚀 Quick Setup & Installation

### Option A: Running from Source

1. Clone this repository:
   ```bash
   git clone https://github.com/BMQ-1/subs-pipeline.git
   cd subs-pipeline

---

1. Create a virtual environment and activate it:

python -m venv venv

# Windows activation:
venv\Scripts\activate
# macOS/Linux activation:
source venv/bin/activate


2. Install the dependencies:

pip install -r requirements.txt


3. Run the application:

python subs-pipeline.py

---

