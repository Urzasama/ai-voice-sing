#!/usr/bin/env python3
"""Synthesise a singing voice from a JSON score.

    python scripts/sing.py --score scores/yua-chorus.json --out demo/out.wav

Optionally mixes the sung vocal on top of an instrumental track:

    python scripts/sing.py --score scores/yua-chorus.json \
        --mix instrumental.wav --mix-offset 1.5 --out demo/mix.wav

The score format is documented in README.md; see scores/yua-chorus.json for a
complete example.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ds_engine import DEFAULT_MODEL_DIR, SingingVoice  # noqa: E402


def load_score(path):
    # utf-8-sig: tolerate a BOM (Notepad / PowerShell users write one by accident).
    with open(path, encoding="utf-8-sig") as handle:
        data = json.load(handle)
    syllables = data["syllables"] if isinstance(data, dict) else data
    if not syllables:
        raise SystemExit("score has no syllables: %s" % path)
    return data, syllables


def load_audio(path, sample_rate):
    """Read any audio file at `sample_rate`, mono float32, via soundfile+ffmpeg."""
    import soundfile as sf

    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        audio, sr = None, None
    if audio is not None and sr == sample_rate and audio.ndim == 1:
        return audio

    from shutil import which

    ffmpeg = which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found; cannot resample %s" % path)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", path,
             "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_f32le", tmp.name],
            check=True,
        )
        audio, _ = sf.read(tmp.name, dtype="float32")
    finally:
        os.unlink(tmp.name)
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def main(argv=None):
    parser = argparse.ArgumentParser(description="DiffSinger ONNX singing voice from a JSON score")
    parser.add_argument("--score", required=True, help="path to a score JSON file")
    parser.add_argument("--out", required=True, help="output wav path")
    parser.add_argument("--transpose", type=int, default=0, help="semitone shift (default 0)")
    parser.add_argument("--f0", dest="f0_source", default="score",
                        choices=["score", "model", "model+fix"],
                        help="F0 source: 'score' (exact, default), 'model' (PE prediction), "
                             "'model+fix' (PE prediction with octave repair)")
    parser.add_argument("--vibrato", type=float, default=0.25,
                        help="vibrato depth in semitones for --f0 score (default 0.25)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    parser.add_argument("--max-chunk", type=float, default=5.0,
                        help="max seconds of score per ONNX call (default 5.0)")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help="folder with the ONNX model files (default models/opencpop)")
    parser.add_argument("--mix", default=None, help="optional instrumental track to mix with")
    parser.add_argument("--mix-offset", type=float, default=0.0,
                        help="where the sung vocal starts inside the mix, in seconds")
    parser.add_argument("--vocal-gain", type=float, default=0.0,
                        help="vocal gain in dB when mixing (default 0)")
    parser.add_argument("--peak", type=float, default=0.95, help="peak level of the output")
    args = parser.parse_args(argv)

    import soundfile as sf

    data, syllables = load_score(args.score)
    title = data.get("title") if isinstance(data, dict) else None
    duration = sum(s["rest"] if "rest" in s else sum(s["durs"]) for s in syllables)
    print("score: %s (%d syllables, %.2f s)%s" % (
        os.path.basename(args.score), sum(1 for s in syllables if "lyric" in s), duration,
        " - " + title if title else ""))

    voice = SingingVoice(model_dir=args.model_dir, max_chunk_seconds=args.max_chunk,
                         seed=args.seed, f0_source=args.f0_source, vibrato=args.vibrato)
    audio = voice.render(syllables, transpose=args.transpose)
    audio = audio / (float(np.max(np.abs(audio))) or 1.0) * args.peak

    if args.mix:
        inst = load_audio(args.mix, voice.sr)
        if inst.size == 0:
            raise SystemExit("%s decoded to 0 samples - is the file empty or the cut "
                             "outside the source range?" % args.mix)
        offset = int(round(args.mix_offset * voice.sr))
        total = max(len(inst), offset + len(audio))
        mixed = np.zeros(total, dtype=np.float32)
        mixed[: len(inst)] += inst / (float(np.max(np.abs(inst))) or 1.0) * 0.85
        mixed[offset: offset + len(audio)] += audio * (10.0 ** (args.vocal_gain / 20.0)) * 0.55
        peak = float(np.max(np.abs(mixed))) or 1.0
        audio = mixed / peak * args.peak
        print("mixed with %s at %.2f s" % (os.path.basename(args.mix), args.mix_offset))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    sf.write(args.out, audio.astype(np.float32), voice.sr, subtype="PCM_16")
    print("wrote %s (%.2f s, %d Hz)" % (args.out, len(audio) / voice.sr, voice.sr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
