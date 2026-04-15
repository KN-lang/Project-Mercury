# 🎙️ Project Mercury: Voice-Controlled Local AI Agent

**Mem0 AI/ML & Generative AI Developer Intern Assignment**

[cite_start]Project Mercury is a fully local, voice-controlled AI agent designed to accept audio input, accurately classify user intent, execute local tool operations, and display the pipeline in a clean, interactive UI[cite: 4]. 

Built with resource efficiency and strict system safety in mind, Mercury runs entirely on local hardware without relying on external APIs for core processing.

## 🔗 Assignment Deliverables
* [cite_start]**Video Demo (YouTube Unlisted):** [Insert Link Here] - A 2-3 minute walkthrough demonstrating the UI, system flow, and multiple intents[cite: 47, 48].
* [cite_start]**Technical Article:** [Insert Link Here] - A detailed breakdown of the architecture, model selection, and engineering challenges[cite: 49, 50].

---

## ✨ Core Features & Requirements Met

* [cite_start]**Dual Audio Input:** Supports real-time microphone recording and audio file uploads (`.wav`, `.mp3`)[cite: 7, 8, 9].
* [cite_start]**Local Speech-to-Text (STT):** Utilizes a local HuggingFace Whisper model (`base.en`) for fast transcription[cite: 13].
* [cite_start]**Local Intent Classification:** Powered by a local Large Language Model via Ollama to determine actions[cite: 17].
* **Tool Execution Pipeline:** Supports the required core intents:
  1. [cite_start]Create a file[cite: 19].
  2. [cite_start]Write code to a file[cite: 20].
  3. [cite_start]Summarize text[cite: 21].
  4. [cite_start]General Chat[cite: 22].
* [cite_start]**Strict Safety Constraints:** All file creation and code generation are strictly sandboxed to a dedicated `output/` directory to prevent accidental system overwrites[cite: 26].
* [cite_start]**Interactive UI:** Built using Streamlit to clearly display transcribed text, detected intent, system action, and final output[cite: 30, 31, 32, 33, 34, 35].

## 🚀 Bonus Features Implemented
To go above and beyond the baseline requirements, Project Mercury includes:
* [cite_start]**Human-in-the-Loop Validation:** An interactive UI confirmation prompt requires manual user approval before executing any file-system operations[cite: 58].
* [cite_start]**Session Memory:** Maintains a persistent, chronological history of transcribed inputs, detected intents, and agent outputs within the session[cite: 59].
* [cite_start]**Graceful Degradation:** Safely handles unmapped intents or unintelligible audio by defaulting to polite conversational fallbacks[cite: 58].

---

## 🧠 Architecture & Hardware Optimizations (Apple M1 Workarounds)

This agent was developed and tested on an **Apple MacBook Air (M1) with 8GB of Unified Memory**. Running both an STT model and an LLM simultaneously on 8GB RAM requires aggressive resource optimization.

**Hardware Workarounds:**
1. **STT Engine (`mlx-whisper` / Whisper `base.en`):** Instead of standard PyTorch, this project uses the Whisper `base.en` model optimized for Apple Silicon (MPS backend). This ensures lightning-fast transcription using the GPU without bottlenecking the unified memory.
2. **Intent Engine (`llama3.2:1b` via Ollama):** To fit comfortably within the remaining RAM, the intent classifier uses `llama3.2:1b` (1 Billion parameters). It is incredibly lightweight (~1.3GB RAM footprint) but highly capable of accurate JSON-formatted intent extraction.

---

## 🛠️ Setup & Installation Instructions

### Prerequisites
* **Python Version:** 3.14.4 (Ensure you are using this exact runtime).
* **System Packages:** `ffmpeg` must be installed for audio processing.
* **Ollama:** Must be installed and running locally.

### Step-by-Step Installation

**1. Install System Dependencies (macOS)**
The underlying audio processing requires `ffmpeg`:
```bash
brew install ffmpeg