from __future__ import annotations

from typing import Any
import math

import librosa
import numpy as np
import parselmouth


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def analyse_wav(
    wav_path: str,
    *,
    transcript: str = "",
    sr: int = 16000,
    pause_min_s: float = 0.25,
    long_pause_s: float = 1.0,
    silence_dbfs_threshold: float = -38.0,
) -> dict[str, Any]:
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    total_duration_s = float(librosa.get_duration(y=y, sr=sr))

    if y.size == 0 or total_duration_s <= 0:
        return {
            "total_duration_s": 0.0,
            "speech_duration_s": 0.0,
            "silence_duration_s": 0.0,
            "pause_count": 0,
            "long_pause_count": 0,
            "mean_pause_s": 0.0,
            "silence_ratio": 0.0,
            "speaking_rate_wpm": 0.0,
            "articulation_rate_wpm": 0.0,
            "volume_mean_dbfs": -80.0,
            "volume_std_db": 0.0,
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "pitch_range_semitones": 0.0,
            "clipping_ratio": 0.0,
            "word_count": 0,
        }

    hop_length = int(sr * 0.01)      # 10 ms
    frame_length = int(sr * 0.03)    # 30 ms

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]
    dbfs = 20.0 * np.log10(np.maximum(rms, 1e-8))
    voiced = dbfs > silence_dbfs_threshold

    voiced_runs = _runs(voiced)
    silence_runs = _runs(~voiced)

    frame_s = hop_length / sr
    speech_duration_s = float(np.sum(voiced) * frame_s)
    silence_duration_s = max(0.0, total_duration_s - speech_duration_s)

    pause_lengths = [
        (end - start) * frame_s
        for start, end in silence_runs
        if (end - start) * frame_s >= pause_min_s
    ]
    long_pause_count = sum(1 for p in pause_lengths if p >= long_pause_s)

    words = len([w for w in transcript.split() if w.strip()])
    speaking_rate_wpm = words / max(total_duration_s / 60.0, 1e-6)
    articulation_rate_wpm = words / max(speech_duration_s / 60.0, 1e-6)

    voiced_dbfs = dbfs[voiced] if np.any(voiced) else dbfs
    volume_mean_dbfs = float(np.mean(voiced_dbfs))
    volume_std_db = float(np.std(voiced_dbfs))

    clipping_ratio = float(np.mean(np.abs(y) >= 0.99))

    sound = parselmouth.Sound(wav_path)
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]

    if f0.size:
        pitch_mean_hz = float(np.mean(f0))
        pitch_std_hz = float(np.std(f0))
        p10 = float(np.percentile(f0, 10))
        p90 = float(np.percentile(f0, 90))
        pitch_range_semitones = 12.0 * math.log2(max(p90, 1.0) / max(p10, 1.0))
    else:
        pitch_mean_hz = 0.0
        pitch_std_hz = 0.0
        pitch_range_semitones = 0.0

    return {
        "total_duration_s": total_duration_s,
        "speech_duration_s": speech_duration_s,
        "silence_duration_s": silence_duration_s,
        "pause_count": len(pause_lengths),
        "long_pause_count": long_pause_count,
        "mean_pause_s": float(np.mean(pause_lengths)) if pause_lengths else 0.0,
        "silence_ratio": silence_duration_s / max(total_duration_s, 1e-6),
        "speaking_rate_wpm": speaking_rate_wpm,
        "articulation_rate_wpm": articulation_rate_wpm,
        "volume_mean_dbfs": volume_mean_dbfs,
        "volume_std_db": volume_std_db,
        "pitch_mean_hz": pitch_mean_hz,
        "pitch_std_hz": pitch_std_hz,
        "pitch_range_semitones": pitch_range_semitones,
        "clipping_ratio": clipping_ratio,
        "word_count": words,
    }


def aggregate_voice_metrics(turns: list[dict[str, Any]]) -> dict[str, Any]:
    if not turns:
        return {}

    total_duration = sum(float(t.get("total_duration_s", 0.0)) for t in turns)
    total_speech = sum(float(t.get("speech_duration_s", 0.0)) for t in turns)
    total_silence = sum(float(t.get("silence_duration_s", 0.0)) for t in turns)
    total_words = sum(int(t.get("word_count", 0)) for t in turns)

    def weighted_mean(key: str) -> float:
        weights = [float(t.get("total_duration_s", 0.0)) for t in turns]
        vals = [float(t.get(key, 0.0)) for t in turns]
        denom = sum(weights)
        return sum(v * w for v, w in zip(vals, weights)) / max(denom, 1e-6)

    long_pause_count = sum(int(t.get("long_pause_count", 0)) for t in turns)
    pause_count = sum(int(t.get("pause_count", 0)) for t in turns)
    total_pause_time = sum(
        float(t.get("mean_pause_s", 0.0)) * int(t.get("pause_count", 0))
        for t in turns
    )

    summary = {
        "turn_count": len(turns),
        "total_duration_s": total_duration,
        "speech_duration_s": total_speech,
        "silence_duration_s": total_silence,
        "silence_ratio": total_silence / max(total_duration, 1e-6),
        "pause_count": pause_count,
        "long_pause_count": long_pause_count,
        "speaking_rate_wpm": total_words / max(total_duration / 60.0, 1e-6),
        "articulation_rate_wpm": total_words / max(total_speech / 60.0, 1e-6),
        "mean_pause_s": total_pause_time / max(pause_count, 1),
        "volume_mean_dbfs": weighted_mean("volume_mean_dbfs"),
        "volume_std_db": weighted_mean("volume_std_db"),
        "pitch_mean_hz": weighted_mean("pitch_mean_hz"),
        "pitch_std_hz": weighted_mean("pitch_std_hz"),
        "pitch_range_semitones": weighted_mean("pitch_range_semitones"),
        "clipping_ratio": weighted_mean("clipping_ratio"),
    }
    summary["delivery_voice_score"] = compute_delivery_voice_score(summary)
    summary["delivery_feedback"] = build_delivery_feedback(summary)
    return summary


def compute_delivery_voice_score(m: dict[str, Any]) -> float:
    score = 100.0

    art = float(m.get("articulation_rate_wpm", 0.0))
    if art < 105:
        score -= min(20.0, (105 - art) * 0.4)
    elif art > 185:
        score -= min(20.0, (art - 185) * 0.4)

    silence_ratio = float(m.get("silence_ratio", 0.0))
    if silence_ratio > 0.30:
        score -= min(20.0, (silence_ratio - 0.30) * 100.0)

    long_pauses = float(m.get("long_pause_count", 0.0))
    total_minutes = max(float(m.get("total_duration_s", 0.0)) / 60.0, 1e-6)
    long_pause_rate = long_pauses / total_minutes
    if long_pause_rate > 4.0:
        score -= min(20.0, (long_pause_rate - 4.0) * 2.5)

    pitch_range = float(m.get("pitch_range_semitones", 0.0))
    if pitch_range < 2.5:
        score -= 10.0

    volume_std = float(m.get("volume_std_db", 0.0))
    if volume_std < 2.0:
        score -= 5.0

    clipping = float(m.get("clipping_ratio", 0.0))
    if clipping > 0.01:
        score -= min(15.0, clipping * 1000.0)

    return max(0.0, min(100.0, round(score, 1)))


def build_delivery_feedback(m: dict[str, Any]) -> list[str]:
    out: list[str] = []

    if float(m.get("silence_ratio", 0.0)) > 0.30:
        out.append("Too much silence between phrases; reduce hesitation and connect ideas more smoothly.")

    if float(m.get("long_pause_count", 0.0)) >= 4:
        out.append("Several long pauses were detected; rehearse transitions and key evidence points.")

    art = float(m.get("articulation_rate_wpm", 0.0))
    if art < 105:
        out.append("Speaking pace is slow; tighten phrasing and answer more directly.")
    elif art > 185:
        out.append("Speaking pace is fast; slow down slightly so arguments remain clear.")

    if float(m.get("pitch_range_semitones", 0.0)) < 2.5:
        out.append("Pitch variation is limited; add more vocal emphasis on key claims and evidence.")

    if float(m.get("volume_std_db", 0.0)) < 2.0:
        out.append("Volume is very flat; vary emphasis to make important points stand out.")

    if float(m.get("clipping_ratio", 0.0)) > 0.01:
        out.append("Audio clipping was detected; reduce mic gain or move slightly away from the microphone.")

    return out
