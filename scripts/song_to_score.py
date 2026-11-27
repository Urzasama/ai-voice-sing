#!/usr/bin/env python3
"""Auto-transcribe a melody from a vocal stem into a draft score JSON.

    python scripts/song_to_score.py --vocal work/vocals.wav --out scores/draft.json

This is a *draft* helper, not a finished transcriber: pYIN tracks the sung
pitch, the runs are merged into notes, and every detected note becomes one
syllable with the placeholder vowel "a".  Real lyrics have to be typed in by
hand afterwards (`--lyrics 雨爱的秘密...` prefills them when the syllable count
happens to match).  Expect noise around unvoiced consonants, octave slips on
short notes, and no rest detection inside a sustained breath.

Needs the optional analysis dependencies: librosa (+ numpy/scipy).
Verified with python 3.12 + librosa 1.0.0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(midi):
    midi = int(round(midi))
    return "%s%d" % (NOTE_NAMES[midi % 12], midi // 12 - 1)


def extract_runs(audio_path, sr=22050, hop=128, frame_length=2048,
                 fmin="E3", fmax="C6", vprob_min=0.15, median_size=13,
                 tolerance=0.7, min_note=0.11, merge_gap=0.08):
    """Return [(start_sec, dur_sec, midi)] for the melody runs in `audio_path`."""
    import librosa
    import numpy as np
    from scipy.ndimage import median_filter

    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    f0, voiced, vprob = librosa.pyin(
        y, fmin=float(librosa.note_to_hz(fmin)), fmax=float(librosa.note_to_hz(fmax)),
        sr=sr, frame_length=frame_length, hop_length=hop)
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    midi = np.full(f0.shape, np.nan, dtype=np.float64)
    ok = voiced & (vprob >= vprob_min) & np.isfinite(f0) & (f0 > 0)
    midi[ok] = 69.0 + 12.0 * np.log2(f0[ok] / 440.0)
    if ok.any():
        midi[ok] = median_filter(midi[ok], size=median_size, mode="nearest")

    runs = []
    start = None
    for i in range(len(midi)):
        cur = midi[i]
        if np.isnan(cur):
            if start is not None:
                runs.append((start, i))
                start = None
            continue
        if start is None:
            start = i
            continue
        ref = np.nanmedian(midi[start:i])
        if abs(cur - ref) > tolerance:
            runs.append((start, i))
            start = i

    if start is not None:
        runs.append((start, len(midi)))

    notes = []
    for a, b in runs:
        t0, t1 = float(times[a]), float(times[min(b, len(times) - 1)])
        dur = t1 - t0
        if dur < min_note:
            continue
        pitch = float(np.nanmedian(midi[a:b]))
        notes.append([t0, dur, pitch])

    merged = []
    for t0, dur, pitch in notes:
        if merged:
            prev = merged[-1]
            gap = t0 - (prev[0] + prev[1])
            if gap < merge_gap and abs(prev[2] - pitch) <= tolerance:
                prev[1] = t0 + dur - prev[0]
                prev[2] = (prev[2] + pitch) / 2.0
                continue
        merged.append([t0, dur, pitch])
    return merged


def to_score(runs, lyrics=None, title=None):
    """Turn detected runs into the score format used by scripts/sing.py."""
    syllables = []
    cursor = 0.0
    chars = [ch for ch in (lyrics or "") if not ch.isspace()]
    for index, (t0, dur, pitch) in enumerate(runs):
        gap = t0 - cursor
        if gap >= 0.06:
            syllables.append({"rest": round(gap, 3)})
        lyric = chars[index] if index < len(chars) else "a"
        syllables.append({"lyric": lyric, "notes": [midi_to_name(pitch)],
                          "durs": [round(max(0.05, dur), 3)]})
        cursor = t0 + dur
    score = {"syllables": syllables,
             "note": "auto-transcribed draft (pYIN); verify by hand before use"}
    if title:
        score["title"] = title
    if chars and len(chars) != len(runs):
        print("warning: %d lyrics but %d detected notes; extra notes use the "
              "placeholder vowel 'a'" % (len(chars), len(runs)))
    return score


def main(argv=None):
    parser = argparse.ArgumentParser(description="Draft score from a vocal stem (pYIN)")
    parser.add_argument("--vocal", required=True, help="vocal stem (wav/flac)")
    parser.add_argument("--out", required=True, help="output score JSON")
    parser.add_argument("--lyrics", default=None, help="optional characters, one per note")
    parser.add_argument("--title", default=None)
    parser.add_argument("--fmin", default="E3")
    parser.add_argument("--fmax", default="C6")
    parser.add_argument("--tolerance", type=float, default=0.7,
                        help="max semitone drift inside one note (default 0.7)")
    parser.add_argument("--min-note", type=float, default=0.11,
                        help="drop notes shorter than this, in seconds (default 0.11)")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.vocal):
        raise SystemExit("vocal not found: %s" % args.vocal)
    if not args.lyrics:
        print("hint: pass --lyrics <characters> to attach real text to the draft")

    runs = extract_runs(args.vocal, fmin=args.fmin, fmax=args.fmax,
                        tolerance=args.tolerance, min_note=args.min_note)
    if not runs:
        raise SystemExit("no notes detected in %s" % args.vocal)
    score = to_score(runs, args.lyrics, args.title or os.path.basename(args.vocal))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(score, handle, ensure_ascii=False, indent=2)
    total = sum(s["rest"] if "rest" in s else sum(s["durs"]) for s in score["syllables"])
    print("detected %d notes, %.2f s total -> %s" % (len(runs), total, args.out))
    for syllable in score["syllables"][:12]:
        if "rest" in syllable:
            print("  rest %.2fs" % syllable["rest"])
        else:
            print("  %s %s %.2fs" % (syllable["lyric"], syllable["notes"][0],
                                     syllable["durs"][0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
