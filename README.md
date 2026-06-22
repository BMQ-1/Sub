***

# 🎬 Subs-pipeline
An intelligent, monolithic media transcription, structural translation, and muxing pipeline. This application automates the generation of synchronized, multi-language subtitles locally using `faster-whisper` and translates them via the **Google Gemini 3.5 Flash** API - **OpenAI GPT-4o-mini** API - **Anthropic Claude 4.5 haiku** API - **Google Translate v1 FREE** API - **Ollama AI Models** Locally.

***

## ⚖️ Disclaimer

> [!IMPORTANT]
> **Hello everyone. Firstly i want to say that this entire repo was done by an AI - the Python, Readme, LICENSE and everything as a whole, i was only a supervisor even tho i know nothing about coding myself.**
> 
> 3 months ago i learned about yt-dlp and it was like i discovered something i've always wished for ( a tool to download videos from many major sites ESPECIALLY Youtube ).
> 
> Then less than a week ago from now i found Ani-Cli and i got a burst of passion like i never had, i could download animes in SECONDS and watch them with no lag or stutter - it was an insane experience.
> 
> BUT i ran into an issue. I speak Arabic - Animes are in Japanese and the Subtitles are mostly in English ( Which i speak but not my native language ) SO i started asking AI for ways to to be able to find a solutiom ---- i already had Whisper from months ago for something i forgot -> then learned about gemini Translation -> then about remuxing the Arabic subtitles i got. Each was a .py with a .bat file of the same name to run it.
> 
> Then the pipeline idea was born ( By the help of AI again ) - it was one file that transcribes Anything -> Translates it to anything. that means i found a gold mine, so why not share it to the public ? 
> 
> Hope i and the AI did good with this tool and whole Repo - If not then feel free to either copy the .py file and modify it for yourself or ask me to do some improvments which i HOPE i will try. Thanks.
> 
> **⚠️ DISCLAIMER** The code is visible for you all, You do NOT need to install the .exe as it might make some peopel on edge.
> 
> you can follow what the Ai said about cloning the repository - so either do so -
> 
> OR copy my .py -> put it in an AI to verify it is safe -> make a python file in the same folder you have the vid you need to work on -> add a .bat next to it ( Ask AI what to put so the .bat makes the .py work in two click -- it is no more than 3 lines to work fine ). 
> 
> NOW you have the tool FOREVOR.

---

**And finally my intention is to share the tiny bit of knowledge that i got and the tool that i came up with by the help of AI to help anyone who would reach my code. Hope you guys could build on it, learn from it, have fun with it same way i was having lots of fun and trying things i thought were impossible to do, let alone do it with no cost and with TWO clicks, Insane. And Thanks again.**

---

## ✨ Key Features

- 📦 **Monolithic Build**: The entire logic is contained within a single `subs-pipeline.py` file.
- 🎙️ **High-Speed Transcription**: Powered by `faster-whisper` for local, GPU-accelerated (or CPU optimized) audio-to-text.
- 🌐 **AI Translation**: Utilizes the latest `gemini-3.5-flash` model for context-aware natural language translation.
- ⏱️ **Timestamp Integrity**: Features a structural alignment engine that ensures translated text never drifts from the original audio timing.
- 🛡️ **Fast-Path Memory**: Remembers your settings. If you run the tool on the same folder twice, you can bypass the setup wizard with one keypress.
- 🧹 **Automatic Cleanup**: Self-cleaning temp files. It even sweeps for "orphaned" files left behind by previous crashes on startup.
- 🔄 **Watch Mode**: Can sit in the background and automatically process any new video files added to a folder.

---

## 🛠️ Prerequisites

1. **Python 3.8 - 3.11**: (3.10 is recommended for the best library stability).
2. **FFmpeg**: 
   - **Windows**: If not found, the tool will offer to download standalone binaries for you automatically.
   - **Linux/macOS**: `sudo apt install ffmpeg` or `brew install ffmpeg`.
3. **Gemini API Key**: Get one for free at [Google AI Studio](https://aistudio.google.com/).

---

## 🚀 Installation & Setup

### Running from Source
1. Clone your repository.
2. Install the required libraries:
   ```bash
   pip install faster-whisper watchdog pyinstaller
   ```
3. Run the script:
   ```bash
   python subs-pipeline.py
   ```

### Creating a Standalone `.exe` (Windows)
To turn the script into a single file you can carry on a USB drive or share with others:
1. Open your terminal in the script folder.
2. Run this command:
   ```bash
   pyinstaller --onefile --clean --name=Subs-pipeline subs-pipeline.py
   ```
3. Your executable will be in the `dist/` folder named `Subs-pipeline.exe`.

---

## 📖 Usage Guide

### 1. Interactive Wizard (Beginner Friendly) 🧙‍♂️
Just run the tool. It will ask you step-by-step for your folder path, target language, and API key. It detects your hardware and **recommends** the best Whisper model for your specific computer.

### 2. Headless CLI (Advanced/Automated) 🤖
For use in scripts or server environments:
```bash
subs-pipeline.exe --headless --folder "C:\Movies" --api_key "KEY" --tgt_lang "Arabic" --tgt_ext "ar" --model "medium"
```

---

## 📊 Status Transparency (What do the results mean?)

The summary table at the end uses specific codes so you know exactly how reliable the output is:

| Label | Meaning |
| :--- | :--- |
| **`[DONE_TRN]`** | **Perfect Success**: Transcription and Translation were both created fresh and muxed. |
| **`[DONE_MIX]`** | **Partial Success**: The translation finished, but some specific lines had to stay in the original language to prevent timing errors. |
| **`[DONE_MUX]`** | **Cached Success**: We found a translation you made previously and just muxed it into a new video for you. |
| **`[REUSED  ]`** | **No Action**: The final video file already exists. We didn't waste your API quota or time. |
| **`[SKIPPED ]`** | **Quality Control**: The audio was too quiet or the AI started "hallucinating" (looping), so we stopped to save your file from being ruined. |
| **`[FAULT   ]`** | **Error**: Something went wrong (e.g., Internet cut out or file permissions denied). |

---

## ⚙️ Advanced CLI Options

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--min_blocks` | How many subtitle lines are needed to consider a file "valid". Prevents outputting empty files. | `3` |
| `--hardsub` | Instead of a toggleable track, burn the text permanently into the video pixels. | `False` |
| `--no_cleanup` | Stop the tool from deleting old `temp_` files on startup. | `False` |
| `--src_lang` | Force a source language (e.g., `ja` for Japanese) instead of auto-detecting. | `None` |
| `--watch` | Turn on folder monitoring mode. | `False` |

---
*Built with transparency and accuracy in mind.*
