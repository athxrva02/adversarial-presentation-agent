#!/usr/bin/env python3
"""
Generate sample test fixtures for testing.

Creates:
    tests/fixtures/sample.wav           — 3s spoken-like WAV (sine wave)
    tests/fixtures/sample_stereo.wav    — 2s stereo WAV
    tests/fixtures/sample_silent.wav    — 1s silent WAV
    data/sample_pdfs/climate_policy.pdf — 3-page argument PDF (requires PyMuPDF)
    data/sample_pdfs/ai_ethics.pdf      — 2-page argument PDF (requires PyMuPDF)

Run from the repo root:
    python scripts/generate_fixtures.py
"""

import math
import os
import struct
import sys
import wave

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES_DIR = os.path.join(REPO_ROOT, "tests", "fixtures")
PDFS_DIR = os.path.join(REPO_ROOT, "data", "sample_pdfs")

os.makedirs(FIXTURES_DIR, exist_ok=True)
os.makedirs(PDFS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# WAV generators
# ---------------------------------------------------------------------------

def _write_wav_sine(path: str, duration_s: float, freq_hz: float, sample_rate: int = 16000,
                    channels: int = 1, amplitude: int = 20000) -> None:
    """Write a WAV file containing a sine wave (simulates speech-like audio)."""
    n_samples = int(sample_rate * duration_s)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(sample_rate)

        if channels == 1:
            samples = [
                int(amplitude * math.sin(2 * math.pi * freq_hz * i / sample_rate))
                for i in range(n_samples)
            ]
            raw = struct.pack(f"{n_samples}h", *samples)
        else:
            # Stereo: different frequencies per channel
            frames = []
            for i in range(n_samples):
                left = int(amplitude * math.sin(2 * math.pi * freq_hz * i / sample_rate))
                right = int(amplitude * math.sin(2 * math.pi * (freq_hz * 1.5) * i / sample_rate))
                frames.extend([left, right])
            raw = struct.pack(f"{len(frames)}h", *frames)

        wf.writeframes(raw)
    print(f"  Written: {path}")


def _write_wav_silent(path: str, duration_s: float = 1.0, sample_rate: int = 8000) -> None:
    """Write a silent (all-zero) WAV file."""
    n_samples = int(sample_rate * duration_s)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"{n_samples}h", *([0] * n_samples)))
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# PDF generator (requires PyMuPDF)
# ---------------------------------------------------------------------------

_PDF_CONTENTS = {
    "climate_policy.pdf": [
        # Page 1
        "Introduction: The Case for Urgent Climate Action\n\n"
        "We argue that climate change represents the most pressing challenge of our time. "
        "The scientific consensus is clear: global temperatures have risen by 1.2°C above "
        "pre-industrial levels, and without immediate action, we risk crossing irreversible "
        "tipping points. Therefore, governments must act decisively within this decade.",

        # Page 2
        "Evidence Base\n\n"
        "According to the IPCC Sixth Assessment Report, greenhouse gas emissions must be "
        "reduced by at least 43% by 2030 to limit warming to 1.5°C. "
        "Studies show that renewable energy is now the cheapest source of new electricity "
        "generation in most of the world. "
        "Figure 1 illustrates the rapid cost decline of solar photovoltaics since 2010. "
        "The evidence suggests that an energy transition is not only necessary but economically viable.",

        # Page 3
        "Policy Recommendations\n\n"
        "We conclude that a carbon tax of at least $100 per tonne CO₂ is required to "
        "internalise externalities and drive behavioural change. "
        "In summary, three pillars underpin our proposal: "
        "1) Carbon pricing through a progressive tax. "
        "2) Public investment in grid infrastructure. "
        "3) Just transition support for fossil fuel workers. "
        "Consequently, we propose a 10-year implementation roadmap beginning in 2025.",
    ],

    "ai_ethics.pdf": [
        # Page 1
        "Defining Responsible AI Development\n\n"
        "By responsible AI we mean systems designed with safety, fairness, and transparency "
        "as first-class properties, not post-hoc additions. "
        "We claim that current voluntary guidelines are insufficient to prevent harm at scale. "
        "The definition of harm in this context refers to both immediate physical risks "
        "and longer-term societal effects such as erosion of democratic discourse.",

        # Page 2
        "The Case for Mandatory Regulation\n\n"
        "According to a 2024 survey of AI researchers, 70% believe frontier AI poses "
        "moderate-to-high existential risk within 50 years. "
        "We argue that mandatory pre-deployment audits, similar to pharmaceutical trials, "
        "should be required for models above a certain capability threshold. "
        "Studies show that self-regulation in the technology sector has historically "
        "failed to prevent monopolistic behaviour and privacy violations. "
        "Therefore, binding international agreements, enforced through trade law, "
        "represent the most credible path to responsible AI governance.",
    ],
}


def _write_pdfs() -> None:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("  PyMuPDF not installed — skipping PDF generation.")
        print("  Install with: pip install PyMuPDF")
        return

    for filename, pages in _PDF_CONTENTS.items():
        path = os.path.join(PDFS_DIR, filename)
        doc = fitz.open()
        for page_text in pages:
            page = doc.new_page(width=595, height=842)  # A4
            # Write header
            title = page_text.split("\n")[0]
            body_text = "\n".join(page_text.split("\n")[2:])
            page.insert_text((72, 72), title, fontsize=14, fontname="helv")
            page.insert_text((72, 110), body_text, fontsize=11, fontname="helv")
        doc.save(path)
        doc.close()
        print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating test fixtures...")

    print("\n[1/2] Audio files:")
    _write_wav_sine(
        os.path.join(FIXTURES_DIR, "sample.wav"),
        duration_s=3.0, freq_hz=440.0, sample_rate=16000, channels=1,
    )
    _write_wav_sine(
        os.path.join(FIXTURES_DIR, "sample_stereo.wav"),
        duration_s=2.0, freq_hz=880.0, sample_rate=44100, channels=2,
    )
    _write_wav_silent(
        os.path.join(FIXTURES_DIR, "sample_silent.wav"),
        duration_s=1.0, sample_rate=8000,
    )

    print("\n[2/2] PDF files:")
    _write_pdfs()

    print("\nDone. Files are ready for testing.")
    print("\nTo run MultiModal Module tests:")
    print("  cd adversarial-presentation-agent")
    print("  pytest tests/test_integration.py -v")
    print("\nTo run only fast tests (no real DB):")
    print("  pytest tests/test_integration.py -v -m 'not slow'")
    print("\nTo run with real Whisper transcription (needs openai-whisper):")
    print("  pytest tests/test_stt.py -v  (integration test uses tests/fixtures/sample.wav)")


if __name__ == "__main__":
    main()
