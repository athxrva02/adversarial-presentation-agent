# Adversarial Presentation Agent

Practice your academic or professional presentation Q&A with an AI that pushes back. Upload your slides, give your presentation out loud, and the agent asks adversarial follow-up questions — spoken back to you — then scores you at the end.

Everything runs locally. No data leaves your machine.

---

## How a session works

```
1. You provide your PDF
   └─ Agent: "I have received and processed your slides. Please begin."

2. You give your presentation (2–3 min, spoken into mic)
   └─ Agent transcribes it with Whisper

3. Agent: "Thank you. I will now ask you questions."
   └─ First question generated from your PDF + what you actually said

4. Back-and-forth Q&A  (as many turns as you like)
   └─ You speak → Whisper transcribes → LLM generates next question → agent speaks it

5. You type /end  →  session is scored
   └─ Overall score, rubric breakdown, strengths, weaknesses, top priority for next time
```

---

## Requirements

| Dependency | Why |
|---|---|
| Python 3.10+ | runtime |
| [Ollama](https://ollama.com) | local LLM |
| ffmpeg | Whisper audio decoding |

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd adversarial-presentation-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Version note:** `requirements.txt` pins `transformers==4.44.2` and `sentence-transformers==3.0.1` for compatibility with PyTorch 2.2.x. If you have a newer PyTorch you can relax these pins.

### 2. Install and start Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start the server (keep this terminal open)
ollama serve

# Pull the model (one-time, ~500 MB)
ollama pull qwen2.5:7b-instruct
```

### 3. Install ffmpeg (required by Whisper)

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

### 4. Install sounddevice (required for microphone input)

```bash
pip install sounddevice
```

---

## Running

### Standard session — your own PDF

```bash
python demo.py --pdf path/to/your_slides.pdf
```

The agent speaks all its lines using the built-in macOS `say` command (voice `Samantha`). No extra setup needed on macOS.

### Disable agent voice (text-only output)

```bash
python demo.py --pdf path/to/your_slides.pdf --no-voice
```

### Self-test — no PDF needed

Generates a built-in 3-page UBI presentation PDF and runs the full pipeline:

```bash
python demo.py --self-test
```

Use this first to verify Ollama + Whisper + ChromaDB are all working before using your real slides.

### Debug mode — see what the LLM is thinking

```bash
python demo.py --pdf slides.pdf --debug
```

Prints the classification label (weak/strong/evasion, alignment, confidence) after each of your turns.

### Persist session data across runs

By default, ChromaDB and SQLite data are written to a temp directory and cleared on restart. To keep them:

```bash
python demo.py --pdf slides.pdf --demo-dir ./my_sessions
```

---

## During a session

### Your presentation (Phase 2)

```
Agent ▸  I have received and processed your slides. Whenever you are ready,
         please go ahead and give your presentation...

  ●  Press Enter to START your presentation, then Enter again to STOP.
```

- Press **Enter** once to start recording
- Speak your full presentation (aim for 2–3 minutes)
- Press **Enter** again to stop — Whisper transcribes it automatically

### Q&A turns (Phase 4)

```
Agent ▸  What specific metric did you use to measure the 40% consumption increase,
         and how did you control for seasonal variation?

  ●  Press Enter to START your answer, then Enter again to STOP.
```

- Same pattern: **Enter** → speak → **Enter**
- Type `/end` before pressing the first Enter to end the session

### Ending early

At any "Press Enter to START" prompt, type `/end` and press Enter:

```
  ●  Press Enter to START your answer...
/end
```

---

## Troubleshooting

**`NameError: name 'nn' is not defined`** — PyTorch / transformers version mismatch.
```bash
pip install "transformers==4.44.2" "sentence-transformers==3.0.1"
```

**`Connection refused at localhost:11434`** — Ollama is not running.
```bash
ollama serve   # in a separate terminal
```

**`RuntimeError: openai-whisper is not installed`**
```bash
pip install openai-whisper
```

**Mic level too low / no speech detected** — Your microphone may be muted or not selected as the default input device. Check System Settings → Sound → Input on macOS. You can also list devices:
```bash
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

**Agent voice not working** — On macOS, `say` is built in and requires no setup. To verify:
```bash
say -v Samantha "hello"
```
To change the voice, edit `_MACOS_VOICE` in `interaction/tts.py`.

**`Failed to send telemetry event`** — Harmless ChromaDB warning, ignore it.

---

## Running the tests

```bash
# Unit tests — no Ollama, no mic, no Whisper needed
pytest tests/test_stt.py tests/test_pdf_parser.py tests/test_storage.py -v

# Full pipeline tests — real ChromaDB + sentence-transformers
pytest tests/test_pipeline_e2e.py -v

# All tests including live LLM calls (requires Ollama running)
pytest tests/ -v -m live
```

---

## Project structure

```
adversarial-presentation-agent/
│
├── demo.py               ← CLI entry point
├── session.py            ← Full conversation flow (PDF → present → Q&A → score)
│
├── interaction/
│   ├── pdf_parser.py     ← PDF → DocumentChunk pipeline
│   ├── stt.py            ← Whisper speech-to-text wrapper
│   ├── tts.py            ← macOS say / pyttsx3 text-to-speech
│   └── mic.py            ← Microphone recording (sounddevice)
│
├── memory/
│   ├── document.py       ← PDF chunk store (ChromaDB + SQLite)
│   ├── episodic.py       ← Per-session claims and session records
│   ├── semantic.py       ← Long-term promoted patterns
│   ├── common_ground.py  ← Negotiated common ground
│   ├── recency.py        ← Recency-weighted retrieval reranking
│   ├── retrieval.py      ← Unified retrieval across all stores
│   └── module.py         ← MemoryModule facade
│
├── reasoning/
│   ├── graph.py          ← LangGraph SessionRunner
│   ├── nodes/            ← classify / retrieve / generate_question / score / summarise
│   ├── prompts/          ← LLM prompt templates
│   └── state.py          ← SessionState TypedDict
│
├── storage/
│   ├── vector_store.py   ← ChromaDB wrapper
│   ├── relational_store.py ← SQLite wrapper
│   └── schemas.py        ← Pydantic data models
│
├── config.py             ← All settings (override via .env)
├── requirements.txt
└── tests/
```

---

## Configuration

All settings are in `config.py` and can be overridden with a `.env` file:

| Setting | Default | Description |
|---|---|---|
| `model_name` | `qwen2.5:7b-instruct` | Ollama model |
| `ollama_base_url` | `http://localhost:11434` | Ollama address |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-transformer |
| `max_chunk_tokens` | `256` | Max tokens per PDF chunk |
| `retrieval_top_k` | `5` | Results per memory store |
| `recency_decay_factor` | `0.85` | Decay per session age |
| `sqlite_path` | `./data/db/agent.db` | SQLite path |
| `chroma_path` | `./data/chroma` | ChromaDB path |

Example `.env`:
```env
MODEL_NAME=qwen2.5:0.5b-instruct
```

To change the agent's voice, edit `_MACOS_VOICE` in `interaction/tts.py`:
```python
_MACOS_VOICE = "Daniel"   # British English
_MACOS_VOICE = "Karen"    # Australian English
```
List all available voices: `say -v ?`
