# Adversarial Presentation Agent

A memory-aware conversational agent that helps you prepare for adversarial presentation Q&A. Fully local, GDPR-compliant.

Refer to `docs/technical-plan.md` for the full technical documentation.

## Setup

### 1. Create a virtual environment (Python 3.12)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Install and run Ollama

```bash
# Install Ollama (macOS)
brew install ollama

# Start the server
ollama serve

# Pull the model (one-time, ~4.7GB)
ollama pull qwen2.5:7b-instruct
```

### 3. Verify

```bash
# Run tests (no Ollama needed)
pytest tests/

# Run tests including live LLM calls (requires Ollama running)
pytest tests/ --live
```

## Usage

```bash
# Interactive practice session
python -m reasoning.dev.dev_run

# Scripted scenario runner
python -m reasoning.dev.dev_scenarios
```

Dev CLI commands: `/end` (end session + score), `/state` (print state), `/help`.
