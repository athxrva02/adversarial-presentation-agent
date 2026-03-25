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
   └─ Voice delivery is analysed in real time (pace, pauses, pitch, volume)

3. Agent: "Thank you. I will now ask you questions."
   └─ First question generated from your PDF + what you actually said

4. Back-and-forth Q&A
   └─ You speak → Whisper transcribes → LLM generates next question → agent speaks it
   └─ Questions rotate through 8 distinct attack angles to avoid repetition
   └─ Session ends automatically after 5 answers, or type /end to end early

5. Session is scored
   └─ Overall score (0–100), rubric breakdown across 9 dimensions, strengths,
      weaknesses, and top priority for next time
   └─ Vocal delivery dimension scored from real audio metrics (when mic is used)

6. Contradiction negotiation (if any contradictions were detected)
   └─ Review each conflict, accept / reject / clarify

7. Session exported to CSV
   └─ results/<timestamp>/summary.csv  — scores, rubric, voice metrics
   └─ results/<timestamp>/turns.csv    — turn-by-turn Q&A log
```

---

## Requirements

| Dependency | Why |
|---|---|
| Python 3.10+ | runtime |
| [Ollama](https://ollama.com) | local LLM |
| ffmpeg | Whisper audio decoding |
| librosa | voice delivery analysis (pace, pauses, volume) |
| praat-parselmouth | pitch analysis for vocal delivery scoring |

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd adversarial-presentation-agent
python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

> **Version note:** `requirements.txt` pins `transformers==4.44.2` and `sentence-transformers==3.0.1` for compatibility with PyTorch 2.2.x. If you have a newer PyTorch you can relax these pins.

### 2. Install and start Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download the installer from https://ollama.com/download

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

# Windows — download from https://ffmpeg.org/download.html and add to PATH
# or via winget:
winget install ffmpeg
```

### 4. Install sounddevice (required for microphone input)

```bash
pip install sounddevice

# Linux also requires PortAudio:
sudo apt install libportaudio2
```

### 5. TTS setup (optional — for voice output)

The agent can speak its questions and feedback aloud. Backend used depends on your OS:

| OS | Backend | Setup needed |
|---|---|---|
| macOS | built-in `say` | nothing |
| Windows | built-in PowerShell SAPI | nothing |
| Linux | `espeak-ng` | `sudo apt install espeak-ng` |
| Any | `pyttsx3` | `pip install pyttsx3` (fallback) |

Voice output is on by default. To disable it, pass `--no-voice`.

---

## Running

### Standard session — your own PDF

```bash
python demo.py --pdf path/to/your_slides.pdf
```

### Text-only mode (no mic, no TTS)

`--no-voice` disables both microphone input and TTS output. You type your answers instead of speaking them, and the agent's questions are printed only.

```bash
python demo.py --pdf path/to/your_slides.pdf --no-voice
```

> Note: vocal delivery analysis and the `vocal_delivery` scoring dimension are not available in text-only mode.

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

Prints the classification label (weak/strong/evasion, alignment, confidence) and the question strategy selected after each of your turns.

### Start with hybrid memory disabled

```bash
python demo.py --pdf slides.pdf --memory-disabled
```

Starts the session in document-only memory mode (see [Memory Modes](#memory-modes) below).

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
- Voice delivery metrics are collected silently in the background

### Q&A turns (Phase 4)

```
Agent ▸  What specific metric did you use to measure the 40% consumption increase,
         and how did you control for seasonal variation?

  ●  Press Enter to START your answer, then Enter to STOP. (answer 4/5)
```

- Same pattern: **Enter** → speak → **Enter**
- The prompt shows your current answer count
- The session ends **automatically** after 5 answers
- Type `/end` to end early.
- Type `/reset` to reset memory.

### Ending early

At any "Press Enter to START" prompt, type `/end` and press Enter:

```
  ●  Press Enter to START your answer...  (answer 2/5)
/end
```

If you type `/end` before answering 3 questions, you will be asked to confirm:

```
  ⚠  You have only answered 2/3 required questions.
     End the Q&A now and go to scoring? [yes / no]
     >
```

Type `yes` to proceed to scoring, or `no` to continue answering.

You can also type `/end` while the agent is **thinking** (generating its next question). It will finish processing your last answer, then end the session.

### Resetting

At any "Press Enter to START" prompt, type `/reset` and press Enter:

```
  ●  Press Enter to START your answer...  (answer 2/5)
/reset
```

You will be asked to confirm.
```
  ⚠  If you reset now, all memory of your sessions will be lost and you will start over with a fresh agent.
     End the session now and reset the agent? [yes / no]
     >
```

Type `yes` to reset the agent's entire memory, or `no` to continue with your sessions.

---

## Scoring

At the end of each session the agent produces an **overall score (0–100)** computed deterministically from a weighted rubric. Each dimension is rated 1–5 by the LLM, then mapped to the 0–100 scale (score 1 → 20, score 3 → 60, score 5 → 100).

### Rubric dimensions

| Dimension | Weight | What is measured |
|---|---|---|
| `evidence_specificity` | 18% | Concrete metrics, baselines, datasets cited |
| `logical_coherence` | 18% | Claims follow logically; no unsupported leaps or contradictions |
| `handling_adversarial_questions` | 16% | Engages with challenges; admits uncertainty appropriately |
| `depth_of_understanding` | 12% | Explains mechanisms and trade-offs, not just surface claims |
| `clarity_structure` | 10% | Answers are well-structured and directly on-topic |
| `definition_precision` | 10% | Key terms precisely defined and used consistently |
| `concession_and_qualification` | 8% | Qualifies claims appropriately; concedes valid counterpoints |
| `recovery_from_challenge` | 8% | Recovers constructively after being caught on a weak point |
| `vocal_delivery` | 8%* | Pace, pauses, pitch variation, volume — from real audio |

\* `vocal_delivery` is only scored when mic input is used. Its 8% weight is redistributed proportionally across the other dimensions in text-only mode.

---

## Voice Delivery Analysis

When you speak into the microphone, each recording is analysed with `librosa` and `praat-parselmouth` before the transcript reaches the LLM. The following metrics are extracted per turn and aggregated into a session-level summary:

| Metric | Description |
|---|---|
| `speaking_rate_wpm` | Words per minute including pauses |
| `articulation_rate_wpm` | Words per minute excluding silent intervals |
| `pause_count` | Number of pauses ≥ 0.25 s |
| `long_pause_count` | Number of pauses ≥ 1.0 s |
| `silence_ratio` | Fraction of total recording that is silent |
| `pitch_range_semitones` | Pitch variation range (higher = more expressive) |
| `volume_std_db` | Volume variation (higher = more dynamic delivery) |
| `clipping_ratio` | Fraction of samples near full-scale (indicates distortion) |

These metrics feed the `vocal_delivery` rubric dimension and are exported to `summary.csv`.

Voice analysis can be disabled in `config.py` or `.env` by setting `ENABLE_VOICE_ANALYSIS=false`.

---

## Session Export

After every session, results are automatically written to `results/<YYYY-MM-DD_HH-MM-SS>/`:

| File | Contents |
|---|---|
| `summary.csv` | Session metadata, overall score, all rubric dimension scores, strengths, weaknesses, top priority, voice metrics, negotiation outcomes |
| `turns.csv` | Turn-by-turn log: role, transcript, classification label, alignment, confidence, claim text, conflict flags |

The `results/` directory is created automatically. Each session gets its own timestamped subdirectory so runs never overwrite each other.

---

## Memory Modes

The agent maintains four types of memory, all stored locally in ChromaDB and SQLite.

### Hybrid mode (default)

All four stores are queried on every turn and fed into the LLM's context:

| Store | What it holds | When it's written |
|---|---|---|
| **Document** | Your PDF, split into labelled chunks (claim / evidence / definition / conclusion) | Once, at startup |
| **Episodic** | Every answer you gave, per session — `ClaimRecord` objects with alignment labels, plus a `SessionRecord` summary at the end | Each turn and at session end |
| **Semantic** | Long-term patterns promoted from episodic data once they appear in ≥ 2 sessions (e.g. "user consistently gives unsupported answers") | End of session, via `promote_patterns()` |
| **Common ground** | Negotiated truths from Phase 6 — contradictions you and the agent agreed to resolve | After contradiction negotiation |

With hybrid memory the agent remembers not just your slides but your personal history of answering questions about them. Contradiction detection, recency-weighted retrieval, and pattern promotion all depend on the episodic, semantic, and common-ground stores being available.

### Document-only mode (`--memory-disabled`)

Only the **document** store is queried. The LLM still has its native short-term memory, the full turn-by-turn conversation history within the current session but no data from past sessions is retrieved.

Use document-only mode when:
- You are running a first session with a new PDF and have no prior history worth retrieving
- You want to isolate one session from past performance data
- You are debugging question quality without cross-session context

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

**`ImportError: librosa` or `parselmouth` not found** — Voice analysis dependencies missing.
```bash
pip install librosa praat-parselmouth
```

**Mic level too low / no speech detected**

List available devices to find the right input:
```bash
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

- **macOS**: System Settings → Privacy & Security → Microphone → enable Terminal/iTerm
- **Windows**: Settings → Privacy → Microphone → allow access; Settings → System → Sound → Input → choose your mic
- **Linux**: Check PulseAudio/PipeWire is not muted (`pavucontrol`); ensure `libportaudio2` is installed

**Agent voice not working**

- **macOS**: `say "hello"` in Terminal — should work out of the box
- **Windows**: PowerShell SAPI is built in; if it fails, try `pip install pyttsx3`
- **Linux**: `sudo apt install espeak-ng`, then `espeak-ng "hello"` to verify

**`Failed to send telemetry event`** — Harmless ChromaDB warning, ignore it.

---

## Running the tests

```bash
# Unit tests — no Ollama, no mic, no Whisper needed
pytest tests/interaction/ tests/storage/ -v

# Memory + full pipeline tests — real ChromaDB + sentence-transformers
pytest tests/memory/ -v

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
├── export.py             ← CSV export of session results
│
├── interaction/
│   ├── pdf_parser.py     ← PDF → DocumentChunk pipeline
│   ├── stt.py            ← Whisper speech-to-text wrapper
│   ├── tts.py            ← TTS: macOS say / Windows SAPI / Linux espeak / pyttsx3
│   ├── mic.py            ← Microphone recording (sounddevice)
│   └── voice_analysis.py ← Per-turn audio metrics (pace, pitch, pauses, volume)
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
│   ├── nodes/            ← classify / retrieve / generate_question / detect_contradiction
│   │                        mediate_contradiction / negotiate / score / summarise
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
| `ollama_base_url` | `http://localhost:11434` | Ollama server address |
| `temperature` | `0.2` | LLM sampling temperature |
| `num_ctx` | `4096` | LLM context window (tokens) |
| `max_tokens` | `400` | Max tokens per LLM generation |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-transformer for memory retrieval |
| `min_questions` | `3` | Minimum answers before `/end` is accepted without confirmation |
| `max_questions` | `5` | Session ends automatically after this many answers |
| `working_memory_window` | `20` | Recent turns kept in LLM short-term context |
| `max_chunk_tokens` | `256` | Max tokens per PDF chunk |
| `chunk_overlap_tokens` | `32` | Token overlap between adjacent PDF chunks |
| `retrieval_top_k` | `5` | Results returned per memory store per turn |
| `promotion_threshold` | `2` | Sessions a pattern must appear in before semantic promotion |
| `recency_decay_factor` | `0.85` | Multiplier per session of age for episodic retrieval ranking |
| `sqlite_path` | `./data/db/agent.db` | SQLite path |
| `chroma_path` | `./data/chroma` | ChromaDB path |
| `enable_voice_analysis` | `true` | Enable per-turn audio metrics (requires librosa + parselmouth) |
| `voice_pause_min_s` | `0.25` | Minimum silence duration counted as a pause (seconds) |
| `voice_long_pause_s` | `1.0` | Silence duration counted as a long pause (seconds) |
| `voice_silence_dbfs_threshold` | `-38.0` | dBFS level below which audio is treated as silence |

Example `.env`:
```env
MODEL_NAME=qwen2.5:7b-instruct
MIN_QUESTIONS=3
MAX_QUESTIONS=5
ENABLE_VOICE_ANALYSIS=false
```
