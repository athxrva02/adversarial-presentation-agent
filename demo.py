"""
demo.py — CLI entry point for the Adversarial Presentation Agent.

Usage:
    python demo.py --pdf slides.pdf          # voice on (default)
    python demo.py --pdf slides.pdf --no-voice   # text-only, no TTS
    python demo.py --pdf slides.pdf --debug      # show classification labels
    python demo.py --self-test               # built-in demo PDF
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress ChromaDB telemetry — the capture() signature changed and it spams stderr
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

RESET = "\033[0m"; BOLD = "\033[1m"; CYAN = "\033[96m"; RED = "\033[91m"; DIM = "\033[2m"


def _banner():
    print()
    print(BOLD + CYAN + "═════════════════════════════════════════════════════════════════════" + RESET)
    print(BOLD + CYAN + "  ADVERSARIAL PRESENTATION AGENT                                     " + RESET)
    print(BOLD + CYAN + "═════════════════════════════════════════════════════════════════════" + RESET)
    print()


def _check_deps(voice: bool) -> bool:
    ok = True
    if voice:
        try:
            import sounddevice  # noqa: F401
        except ImportError:
            print(RED + "  sounddevice not installed — run: pip install sounddevice" + RESET)
            ok = False
        try:
            import numpy  # noqa: F401
        except ImportError:
            print(RED + "  numpy not installed" + RESET)
            ok = False
    try:
        import whisper  # noqa: F401
    except ImportError:
        print(RED + "  openai-whisper not installed — run: pip install openai-whisper" + RESET)
        ok = False
    try:
        import fitz  # noqa: F401
    except ImportError:
        print(RED + "  PyMuPDF not installed — run: pip install PyMuPDF" + RESET)
        ok = False
    return ok


def _make_self_test_pdf(path: str) -> None:
    import fitz
    pages = [
        ("Thesis: Universal Basic Income Reduces Poverty",
         "We argue that a Universal Basic Income of $1,000/month per adult "
         "would eliminate extreme poverty in OECD countries within 5 years. "
         "Funded through a progressive wealth tax starting at 1% on net worth "
         "above $10 million. We claim this is fiscally feasible and socially just."),
        ("Evidence",
         "According to the Finland UBI pilot (2017-2018), recipients reported "
         "higher wellbeing without reduced employment. "
         "GiveDirectly Kenya study: consumption increased 40%, effects persisted 3+ years."),
        ("Objections and Responses",
         "Objection: UBI is inflationary. Response: models show inflation stays below 2%. "
         "Objection: work disincentive. Response: pilots show negligible labour supply effects. "
         "We conclude UBI is the most cost-effective anti-poverty instrument available."),
    ]
    doc = fitz.open()
    for title, body in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 60),  title, fontsize=14, fontname="helv")
        page.insert_text((72, 100), body,  fontsize=10.5, fontname="helv")
    doc.save(path)
    doc.close()


def main():
    parser = argparse.ArgumentParser(
        description="Adversarial Presentation Agent — practice your Q&A skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pdf", help="Path to your presentation PDF")
    parser.add_argument("--no-voice", action="store_true",
                        help="Disable TTS — agent responses printed only, no speech")
    parser.add_argument("--self-test", action="store_true",
                        help="Generate a built-in demo PDF and run the full pipeline")
    parser.add_argument("--debug", action="store_true",
                        help="Print LLM classification labels after each turn")
    parser.add_argument("--memory-disabled", action="store_true",
                        help="Start with hybrid memory disabled (document memory only).")
    parser.add_argument("--demo-dir",
                        default=os.path.join(tempfile.gettempdir(), "adversarial_agent"),
                        help="Directory for session data (default: system temp)")

    args = parser.parse_args()
    voice = not args.no_voice

    _banner()

    if args.self_test:
        try:
            import fitz  # noqa: F401
        except ImportError:
            print(RED + "  PyMuPDF required: pip install PyMuPDF" + RESET)
            sys.exit(1)
        os.makedirs(args.demo_dir, exist_ok=True)
        pdf_path = os.path.join(args.demo_dir, "demo_ubi.pdf")
        _make_self_test_pdf(pdf_path)
        print(DIM + f"  Generated demo PDF: {pdf_path}" + RESET)
        args.pdf = pdf_path

    if not args.pdf:
        parser.error("--pdf is required  (or use --self-test for a built-in demo)")
    if not os.path.isfile(args.pdf):
        print(RED + f"  PDF not found: {args.pdf}" + RESET)
        sys.exit(1)
    if not _check_deps(voice=voice):
        sys.exit(1)

    from session import run_session

    while True:
        result = run_session(
            pdf_path=args.pdf,
            demo_dir=args.demo_dir,
            voice=voice,
            debug=args.debug,
            hybrid_memory=not args.memory_disabled,
        )

        if result == "RESET":
            print()
            print(DIM + "  Memory cleared." + RESET)

        break


if __name__ == "__main__":
    main()
