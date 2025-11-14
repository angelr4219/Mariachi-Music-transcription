#!/usr/bin/env python3
"""
Single-note pitch detector using autocorrelation.

Usage:
    python single_note_pitch.py input_note.wav
"""

import sys
import math
from typing import Tuple

import numpy as np
import soundfile as sf


# ---------- Utility: Hz -> MIDI & Note Name ----------
def freq_to_midi(freq_hz: float) -> int:
    """Convert frequency in Hz to nearest MIDI note number."""
    if freq_hz <= 0.0:
        return -1
    return int(round(69.0 + 12.0 * math.log2(freq_hz / 440.0)))


def midi_to_note_name(midi: int) -> str:
    """Convert MIDI note number to name like 'C4', 'A#3', etc."""
    if midi < 0 or midi > 127:
        return "Unknown"

    names = [
        "C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B"
    ]

    pitch_class = midi % 12
    octave = midi // 12 - 1  # MIDI 60 => C4

    return f"{names[pitch_class]}{octave}"


# ---------- Autocorrelation Pitch Estimator ----------
def estimate_pitch_autocorr(
    x: np.ndarray,
    sample_rate: float,
    f_min: float = 80.0,
    f_max: float = 1000.0
) -> float:
    """
    Estimate fundamental frequency using a simple (non-normalized)
    autocorrelation over a given frequency range.
    """
    if x.size == 0:
        return 0.0

    # Make sure it's 1D
    x = np.asarray(x, dtype=float).flatten()

    # Optional: normalize to [-1, 1] range for numerical stability
    max_abs = np.max(np.abs(x))
    if max_abs > 0:
        x = x / max_abs

    # Convert frequency range to lag range
    min_lag = int(sample_rate / f_max)
    max_lag = int(sample_rate / f_min)

    if max_lag >= x.size:
        max_lag = x.size - 1
    if min_lag < 1:
        min_lag = 1

    best_value = -1e30
    best_lag = -1

    # Brute-force autocorrelation (simple, not super fast, but fine for short clips)
    for lag in range(min_lag, max_lag + 1):
        # x[:-lag] and x[lag:] are aligned slices
        prod = x[:-lag] * x[lag:]
        s = float(np.sum(prod))
        if s > best_value:
            best_value = s
            best_lag = lag

    if best_lag <= 0:
        return 0.0

    freq = sample_rate / float(best_lag)
    return freq


# ---------- Load mono WAV ----------
def load_mono_wav(path: str) -> Tuple[np.ndarray, float]:
    """
    Load an audio file as mono float32.
    Returns: (samples, sample_rate)
    """
    data, sr = sf.read(path)  # data shape: (N,) or (N, channels)

    if data.ndim == 2:
        # Downmix to mono by averaging channels
        data = np.mean(data, axis=1)

    return data.astype(np.float32), float(sr)


def main():
    if len(sys.argv) < 2:
        print("Usage: python single_note_pitch.py input_note.wav")
        sys.exit(1)

    path = sys.argv[1]

    # 1) Load audio
    samples, sr = load_mono_wav(path)

    # 2) Optional: take just 1 second from the middle to avoid transient noise
    if samples.size > int(sr * 2):  # more than 2 seconds of audio
        mid = samples.size // 2
        half_win = int(sr / 2)  # 0.5s each side
        start = max(0, mid - half_win)
        end = min(samples.size, start + int(sr))
        samples = samples[start:end]

    # 3) Estimate pitch
    freq = estimate_pitch_autocorr(samples, sr)
    if freq <= 0.0:
        print("Could not estimate pitch.")
        sys.exit(1)

    # 4) Convert to MIDI/note name
    midi = freq_to_midi(freq)
    note_name = midi_to_note_name(midi)

    # 5) Print results
    print(f"Estimated frequency: {freq:.2f} Hz")
    print(f"Nearest MIDI note:   {midi}")
    print(f"Note name:           {note_name}")


if __name__ == "__main__":
    main()
