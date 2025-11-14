#generate_and_analyze.py

import math
from typing import Tuple
import numpy as np
import soundfile as sf
import argparse
import os
"""
Generate a test sine tone for a given note name (e.g. A, C4, F#3),
save it as a WAV file in a 'Tones' folder, then estimate its pitch
using autocorrelation.

Usage examples:
    python generate_and_analyze.py A
    python generate_and_analyze.py C4
    python generate_and_analyze.py F#3 --duration 3.0 --sr 48000
"""

# ---------- Note / MIDI / Frequency utilities ----------

def midi_to_freq(midi: int) -> float:
    """Convert MIDI note to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


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


def note_to_midi(note_str: str, default_octave: int = 4) -> int:
    """
    Convert a note string like 'A', 'A4', 'C#3', 'Bb3' to a MIDI number.

    - If no octave is given (e.g. 'A' or 'F#'), uses default_octave (4).
    - Supports sharps (#) and flats (b), e.g. 'Bb3', 'Db5'.
    """
    s = note_str.strip()
    if not s:
        raise ValueError("Empty note string")

    base = s[0].upper()
    if base not in "ABCDEFG":
        raise ValueError(f"Invalid base note: {base}")

    rest = s[1:]
    accidental = ''
    if rest and rest[0] in ['#', 'b']:
        accidental = rest[0]
        rest = rest[1:]

    octave = default_octave
    if rest:
        try:
            octave = int(rest)
        except ValueError:
            raise ValueError(f"Invalid octave in note '{note_str}'")

    name = base + accidental  # e.g. 'A', 'C#', 'Bb'

    pc_map = {
        "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4,
        "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11
    }

    # Handle flats by converting to enharmonic sharps
    if name not in pc_map and accidental == 'b':
        flat_to_sharp = {
            "DB": "C#",
            "EB": "D#",
            "GB": "F#",
            "AB": "G#",
            "BB": "A#"
        }
        name_up = base + 'B'  # e.g. 'Db' -> 'DB'
        if name_up in flat_to_sharp:
            name = flat_to_sharp[name_up]

    if name not in pc_map:
        raise ValueError(f"Unsupported note name '{note_str}' (parsed as '{name}')")

    pc = pc_map[name]
    midi = (octave + 1) * 12 + pc
    return midi


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

    # Flatten and convert to float
    x = np.asarray(x, dtype=float).flatten()

    # Optional: normalize
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

    # Brute-force autocorrelation
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


# ---------- Generate a sine tone ----------

def generate_sine_tone(
    freq_hz: float,
    duration_s: float,
    sample_rate: int = 44100,
    amplitude: float = 0.5
) -> Tuple[np.ndarray, float]:
    """
    Generate a sine wave of given frequency and duration.
    Returns (samples, sample_rate).
    """
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    x = amplitude * np.sin(2.0 * math.pi * freq_hz * t)
    return x.astype(np.float32), float(sample_rate)


def main():
    parser = argparse.ArgumentParser(
        description="Generate and analyze a sine tone for a given note."
    )
    parser.add_argument(
        "note",
        help="Note name (e.g. A, A4, C, C#3, F#, Bb3). "
             "If no octave is given, defaults to octave 4."
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=2.0,
        help="Duration in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=44100,
        help="Sample rate in Hz (default: 44100)"
    )
    parser.add_argument(
        "--amp",
        type=float,
        default=0.5,
        help="Amplitude (0.0–1.0, default: 0.5)"
    )
    parser.add_argument(
        "--outdir",
        default="Tones",
        help="Output directory for tone WAV file (default: 'Tones')"
    )

    args = parser.parse_args()

    # 1) Parse note -> MIDI -> frequency
    midi = note_to_midi(args.note, default_octave=4)
    true_freq = midi_to_freq(midi)
    note_name_canonical = midi_to_note_name(midi)

    print(f"Generating sine tone for note: {args.note} "
          f"(canonical: {note_name_canonical}, {true_freq:.2f} Hz)")
    print(f"Duration: {args.duration} s, Sample rate: {args.sr} Hz, Amp: {args.amp}")

    # 2) Generate samples
    samples, sr = generate_sine_tone(true_freq, args.duration, args.sr, args.amp)

    # 3) Save to WAV inside the Tones folder
    os.makedirs(args.outdir, exist_ok=True)

    # Make filename reasonably safe (keep # though; it's fine on macOS)
    safe_note = args.note.replace(" ", "_")
    out_path = os.path.join(args.outdir, f"{safe_note}.wav")

    sf.write(out_path, samples, int(sr))
    print(f"Saved test tone to {out_path}")

    # 4) Estimate pitch from the generated samples
    estimated_freq = estimate_pitch_autocorr(samples, sr)

    if estimated_freq <= 0.0:
        print("Could not estimate pitch.")
        return

    est_midi = freq_to_midi(estimated_freq)
    est_name = midi_to_note_name(est_midi)

    print("\n--- Analysis ---")
    print(f"True frequency:      {true_freq:.2f} Hz ({note_name_canonical}, MIDI {midi})")
    print(f"Estimated frequency: {estimated_freq:.2f} Hz")
    print(f"Nearest MIDI note:   {est_midi}")
    print(f"Estimated note name: {est_name}")


if __name__ == "__main__":
    main()