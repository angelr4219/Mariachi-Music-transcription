#!/usr/bin/env python3
"""
Convert a mono WAV file containing a single sustained note
into a one-note MIDI file.

Usage:
    python wav_to_midi.py Tones/A.wav
    python wav_to_midi.py path/to/your_note.wav --outdir Midi
"""

import os
import math
import argparse
from typing import Tuple

import numpy as np
import soundfile as sf
import mido


# ---------- Utilities: WAV loading ----------

def load_mono_wav(path: str) -> Tuple[np.ndarray, float]:
    """
    Load an audio file as mono float32.
    Returns: (samples, sample_rate)
    """
    data, sr = sf.read(path)  # data: (N,) or (N, channels)

    if data.ndim == 2:
        # Downmix to mono
        data = np.mean(data, axis=1)

    return data.astype(np.float32), float(sr)


# ---------- Utilities: Hz <-> MIDI <-> Name ----------

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
    f_min: float = 50.0,
    f_max: float = 2000.0
) -> float:
    """
    Estimate fundamental frequency using a simple (non-normalized)
    autocorrelation over a given frequency range.
    """
    if x.size == 0:
        return 0.0

    x = np.asarray(x, dtype=float).flatten()

    # Normalize
    max_abs = np.max(np.abs(x))
    if max_abs > 0:
        x = x / max_abs

    # Frequency range -> lag range
    min_lag = int(sample_rate / f_max)
    max_lag = int(sample_rate / f_min)

    if max_lag >= x.size:
        max_lag = x.size - 1
    if min_lag < 1:
        min_lag = 1

    best_value = -1e30
    best_lag = -1

    for lag in range(min_lag, max_lag + 1):
        prod = x[:-lag] * x[lag:]
        s = float(np.sum(prod))
        if s > best_value:
            best_value = s
            best_lag = lag

    if best_lag <= 0:
        return 0.0

    freq = sample_rate / float(best_lag)
    return freq


# ---------- MIDI writer ----------

def write_single_note_midi(
    midi_note: int,
    duration_s: float,
    out_path: str,
    velocity: int = 100,
    bpm: int = 120
) -> None:
    """
    Create a MIDI file with a single note starting at time 0
    and lasting duration_s seconds.
    """
    # Standard MIDI setup
    ticks_per_beat = 480
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Tempo: microseconds per beat
    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

    # Note on at time 0
    track.append(mido.Message('note_on', note=midi_note, velocity=velocity, time=0))

    # Convert duration in seconds -> ticks
    ticks = int(mido.second2tick(duration_s, ticks_per_beat, tempo))

    # Note off after 'ticks'
    track.append(mido.Message('note_off', note=midi_note, velocity=0, time=ticks))

    mid.save(out_path)


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a mono WAV file (single sustained note) into a one-note MIDI file."
    )
    parser.add_argument(
        "wav_path",
        help="Path to input WAV file."
    )
    parser.add_argument(
        "--outdir",
        default="Midi",
        help="Output directory for MIDI file (default: 'Midi')."
    )
    parser.add_argument(
        "--bpm",
        type=int,
        default=120,
        help="Tempo for the MIDI file in BPM (default: 120)."
    )

    args = parser.parse_args()

    wav_path = args.wav_path

    if not os.path.isfile(wav_path):
        print(f"Error: file not found: {wav_path}")
        return

    # 1) Load WAV
    samples, sr = load_mono_wav(wav_path)
    duration_s = len(samples) / sr

    print(f"Loaded WAV: {wav_path}")
    print(f"Sample rate: {sr} Hz, duration: {duration_s:.2f} s")

    # Optional: use a middle slice to avoid attack/decay
    if duration_s > 2.0:
        mid = len(samples) // 2
        half_win = int(sr / 2)  # 0.5s each side
        start = max(0, mid - half_win)
        end = min(len(samples), start + int(sr))
        slice_samples = samples[start:end]
    else:
        slice_samples = samples

    # 2) Estimate pitch from slice
    freq = estimate_pitch_autocorr(slice_samples, sr)

    if freq <= 0.0:
        print("Could not estimate pitch.")
        return

    midi_note = freq_to_midi(freq)
    note_name = midi_to_note_name(midi_note)

    print("\n--- Pitch estimate ---")
    print(f"Estimated frequency: {freq:.2f} Hz")
    print(f"MIDI note:          {midi_note}")
    print(f"Note name:          {note_name}")

    # 3) Write MIDI
    os.makedirs(args.outdir, exist_ok=True)

    base = os.path.splitext(os.path.basename(wav_path))[0]
    midi_out_path = os.path.join(args.outdir, f"{base}.mid")

    write_single_note_midi(midi_note, duration_s, midi_out_path, bpm=args.bpm)

    print(f"\nWrote MIDI file: {midi_out_path}")


if __name__ == "__main__":
    main()
