"""
session.py — Full conversation flow for the Adversarial Presentation Agent.

Flow:
    1. User provides PDF  →  agent confirms receipt, asks user to present
    2. User gives 2-3 min spoken presentation  →  agent transcribes it
    3. Agent announces Q&A phase  →  asks first question (spoken + printed)
    4. Back-and-forth Q&A  (N turns, each: user speaks → agent speaks question)
    5. User types /end  →  session summarised + scored  →  results printed

Entry point:
    run_session(pdf_path, demo_dir, voice=True, debug=False)

This module is UI-independent — demo.py calls run_session() directly.
"""

from __future__ import annotations

import os
import sys
import tempfile
import logging
import threading
from datetime import datetime
from config import settings as _settings

logger = logging.getLogger(__name__)

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET   = "\033[0m";  BOLD  = "\033[1m";  DIM   = "\033[2m"
CYAN    = "\033[96m"; GREEN = "\033[92m"; RED   = "\033[91m"
YELLOW  = "\033[93m"; BLUE  = "\033[94m"; MAGENTA = "\033[95m"


# ── Pretty helpers ────────────────────────────────────────────────────────────

def _hr(char="─", width=72):
    print(DIM + char * width + RESET)

def _section(title: str):
    print(); print(BOLD + BLUE + f"▶  {title}" + RESET); _hr()

def _ok(msg: str):   print(GREEN + "  ✓  " + RESET + msg)
def _info(msg: str): print(DIM  + "     " + msg + RESET)
def _warn(msg: str): print(YELLOW + "  ⚠  " + RESET + msg)

def _agent_print(msg: str):
    """Print the agent's spoken line prominently."""
    print()
    print(BOLD + MAGENTA + "Agent ▸ " + RESET + msg)
    print()

def _user_label(source: str):
    print(BOLD + CYAN + f"You [{source}] ▸ " + RESET, end="", flush=True)


# ── TTS / mic helpers ─────────────────────────────────────────────────────────

def _speak(text: str, voice: bool) -> None:
    """Print and optionally speak text."""
    _agent_print(text)
    if voice:
        from interaction.tts import speak
        speak(text)


def _transcribe_wav(wav_path: str) -> str:
    """Transcribe a WAV file via Whisper. Returns empty string on failure."""
    from interaction.stt import transcribe
    try:
        return transcribe(wav_path)
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        return ""
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def _record_speech(prompt_line: str, label: str = "voice") -> str | None:
    """
    Record from mic.
    Returns transcript string, None to retry, or "/end" to end session.
    """
    from interaction.mic import record

    print()
    print(BOLD + CYAN + "  ● " + RESET + prompt_line)

    wav_path = record(prompt="", sample_rate=16000)

    if wav_path == "/end":
        return "/end"

    if wav_path is None:
        _warn("Mic level too low — check your microphone is not muted.")
        return None

    duration_hint = ""
    try:
        import wave as _wave
        with _wave.open(wav_path, "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
            duration_hint = f"  ({dur:.1f}s recorded)"
    except Exception:
        pass

    print(DIM + f"  Transcribing…{duration_hint}" + RESET, flush=True)
    transcript = _transcribe_wav(wav_path)

    if not transcript:
        _warn("No speech detected. Please try again.")
        return None

    _user_label(label)
    print(transcript)
    return transcript


# ── Storage setup ─────────────────────────────────────────────────────────────

def _setup_stores(demo_dir: str):
    from storage.vector_store import VectorStore
    from storage.relational_store import RelationalStore
    os.makedirs(os.path.join(demo_dir, "db"), exist_ok=True)
    vs = VectorStore(chroma_path=os.path.join(demo_dir, "chroma"))
    rs = RelationalStore(db_path=os.path.join(demo_dir, "db", "session.db"))
    return vs, rs


# ── Phase 1: PDF ingestion ────────────────────────────────────────────────────

def _ingest_pdf(pdf_path: str, vs, rs) -> int:
    from interaction.pdf_parser import ingest_pdf
    from memory.document import DocumentMemory

    chunks = ingest_pdf(pdf_path)
    dm = DocumentMemory(vector_store=vs, relational_store=rs)
    dm.store(chunks)

    type_counts: dict[str, int] = {}
    for c in chunks:
        type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1

    page_count = max((c.slide_number or 0 for c in chunks), default=0)
    _ok(f"Parsed {len(chunks)} chunks across {page_count} page(s)")
    for t, n in sorted(type_counts.items()):
        _info(f"{t:12s}: {n}")
    return len(chunks)


# ── Phase 2: Session initialisation ──────────────────────────────────────────

def _init_runner(vs, rs):
    from memory.module import MemoryModule
    from reasoning.graph import SessionRunner
    mm = MemoryModule(vector_store=vs, relational_store=rs)
    runner = SessionRunner(memory_module=mm)
    _ok(f"Session ready  (id: {BOLD}{runner.state['session_id']}{RESET})")
    return runner


# ── Phase 5: Summary display ──────────────────────────────────────────────────

def _display_summary(runner, voice: bool) -> None:
    _section("Session Results")

    rec = runner.state.get("session_summary")
    breakdown = runner.state.get("score_breakdown") or {}

    if rec is None:
        _warn("No summary produced.")
        return

    print(f"  session_id      : {rec.session_id}")
    print(f"  duration        : {rec.duration_seconds:.0f}s")
    print(f"  turns completed : {rec.claims_count}")
    print()

    score = rec.overall_score
    if score is not None:
        bar = "█" * int(score / 100 * 30) + "░" * (30 - int(score / 100 * 30))
        col = GREEN if score >= 70 else (YELLOW if score >= 40 else RED)
        print(f"  Overall Score   : {col}{BOLD}{score:.0f}/100{RESET}  {col}{bar}{RESET}")
        print()

    rubric = breakdown.get("rubric", {})
    if rubric:
        print(BOLD + "  RUBRIC" + RESET)
        for dim, val in rubric.items():
            print(f"    {dim:<35}: {val}")
        print()

    if rec.strengths:
        print(BOLD + GREEN + "  STRENGTHS" + RESET)
        for s in rec.strengths:
            print(f"    {GREEN}+{RESET}  {s}")
        print()

    if rec.weaknesses:
        print(BOLD + RED + "  WEAKNESSES" + RESET)
        for w in rec.weaknesses:
            print(f"    {RED}−{RESET}  {w}")
        print()

    notes = breakdown.get("notes", {})
    if isinstance(notes, dict):
        next_step = notes.get("most_important_next_step", "")
        if next_step:
            print(BOLD + "  TOP PRIORITY FOR NEXT SESSION" + RESET)
            print(f"    {next_step}")
            print()
            if voice:
                from interaction.tts import speak
                speak(f"My top recommendation: {next_step}")

    _hr("═")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_session(
    pdf_path: str,
    demo_dir: str,
    *,
    voice: bool = True,
    debug: bool = False,
) -> None:
    """
    Run a complete adversarial presentation practice session.

    Args:
        pdf_path:  Path to the user's presentation PDF.
        demo_dir:  Directory for ChromaDB + SQLite data.
        voice:     If True, agent speaks all its lines via TTS.
        debug:     If True, print classification labels each turn.
    """
    os.makedirs(demo_dir, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # ── Phase 1: Ingest PDF ───────────────────────────────────────────────────
    _section("Phase 1 — Processing PDF")
    print(f"  {DIM}{pdf_path}{RESET}")
    vs, rs = _setup_stores(demo_dir)
    _ingest_pdf(pdf_path, vs, rs)

    # Agent confirms receipt
    receipt_msg = (
        "I have received and processed your documents. "
        "Whenever you are ready, please go ahead and give your presentation. "
        "Press Enter to start speaking, then press Enter again when you are done."
    )
    _section("Phase 2 — Your Presentation")
    _speak(receipt_msg, voice)

    # ── Phase 2: Record the full presentation ─────────────────────────────────
    presentation_transcript = None
    while presentation_transcript is None:
        result = _record_speech(
            "Press Enter to START your presentation, then Enter again to STOP.",
            label="presentation",
        )
        if result == "/end":
            print(DIM + "Session cancelled." + RESET)
            return
        presentation_transcript = result

    _info(f"Presentation captured ({len(presentation_transcript.split())} words)")

    # ── Phase 3: Init session runner, feed presentation as turn 0 ────────────
    _section("Phase 3 — Session Initialisation")
    runner = _init_runner(vs, rs)

    # Feed the full presentation as the first user input so the agent
    # has context grounded in BOTH the PDF chunks AND what was actually said
    transition_msg = (
        "Thank you. I have listened to your presentation. "
        "I will now ask you a series of follow-up questions. "
        "Please answer each one as clearly and specifically as you can."
    )
    _section("Phase 4 — Q&A Practice")
    _speak(transition_msg, voice)

    # Generate first question from the presentation transcript
    print(DIM + "  Generating first question…" + RESET, flush=True)
    first_question = runner.handle_user_input(presentation_transcript)

    if debug:
        cls = runner.state.get("classification")
        if cls:
            print(DIM + f"  [classify] {getattr(cls,'response_class','?')} / "
                  f"{getattr(cls,'alignment','?')} / conf={getattr(cls,'confidence',0):.2f}" + RESET)

    _speak(first_question, voice)

    # ── Phase 4: Q&A loop ─────────────────────────────────────────────────────
    min_answers = _settings.min_questions   # user must answer at least this many (default 3)
    max_answers = _settings.max_questions   # session ends after this many answers (default 5)

    answered_count = 0  # number of completed answer→question exchanges

    print(DIM + f"  Q&A: answer {min_answers}–{max_answers} questions. "
          f"Commands: /end " + RESET)

    def _confirm_early_exit() -> bool:
        """Ask the user to confirm they want to end before the minimum. Returns True = end."""
        print()
        print(YELLOW + "  ⚠  " + RESET + f"You have only answered {answered_count}/{min_answers} required questions.")
        print(DIM + "     End the Q&A now and go to scoring? [yes / no]" + RESET)
        try:
            reply = input("     > ").strip().lower()
        except EOFError:
            return False
        return reply in {"yes", "y"}

    while runner.state.get("session_active", True):
        # ── Collect one answer ────────────────────────────────────────────────
        answer = None
        while answer is None:
            prompt_line = (
                f"Press Enter to START your answer, then Enter to STOP. "
                f"(answer {answered_count + 1}/{max_answers}"
                + (" — /end to finish)" if answered_count >= min_answers else ")")
            )
            result = _record_speech(prompt_line, label="answer")

            if result == "/end":
                if answered_count >= min_answers:
                    break
                else:
                    # Below minimum — ask for confirmation
                    if _confirm_early_exit():
                        break
                    else:
                        _info("Continuing Q&A.")
                        result = None
                        continue

            answer = result

        if result == "/end":
            break

        # Hard stop: max answers received — exit without generating another question
        answered_count += 1
        if answered_count >= max_answers:
            _info(f"Maximum of {max_answers} answers reached. Ending session.")
            break

        # ── Generate next question; /end typed here ends immediately ─────────
        # A background thread watches stdin. If the user types /end while the
        # LLM is running, _abort is set and we skip the question entirely.
        _abort = threading.Event()

        def _watch_for_abort(event: threading.Event) -> None:
            try:
                line = input()
                if line.strip().lower() in {"/end", "end", "quit", "exit"}:
                    event.set()
            except EOFError:
                pass

        watcher = threading.Thread(target=_watch_for_abort, args=(_abort,), daemon=True)
        watcher.start()

        print(DIM + f" Thinking…" + RESET, flush=True)

        # Run the LLM in a thread so we can abandon it if /end fires first
        _question_box: list = []

        def _run_llm() -> None:
            _question_box.append(runner.handle_user_input(answer))

        llm_thread = threading.Thread(target=_run_llm, daemon=True)
        llm_thread.start()

        # Poll: wake up frequently to check _abort; abandon as soon as it fires
        _exit_after_turn = False
        while llm_thread.is_alive():
            llm_thread.join(timeout=0.1)
            if _abort.is_set() and answered_count >= min_answers:
                _exit_after_turn = True
                break

        # Always wait for the LLM thread to fully finish before proceeding —
        # it is writing to runner.state, and end_session() must see a clean state.
        llm_thread.join()

        if _exit_after_turn:
            _info("Session ended early as requested.")
            break

        if debug:
            cls = runner.state.get("classification")
            if cls:
                print(DIM + f"  [classify] {getattr(cls,'response_class','?')} / "
                      f"{getattr(cls,'alignment','?')} / conf={getattr(cls,'confidence',0):.2f}" + RESET)

        question = _question_box[0] if _question_box else ""
        _speak(question, voice)

    # ── Phase 5: End session ──────────────────────────────────────────────────
    _section("Phase 5 — Scoring")
    print(DIM + "  Analysing session…" + RESET, flush=True)
    runner.end_session()
    _display_summary(runner, voice)
