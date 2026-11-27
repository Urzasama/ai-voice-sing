#!/usr/bin/env python3
"""One-shot "make the AI sing this song" driver.

    python scripts/cover.py --song song.mp3 --start 86 --dur 16 \
        --score scores/yua-chorus.json --offset 1.0 --out work/cover.wav

Steps: cut [start, start+dur) -> separate vocals/instrumental (Demucs, or pass
ready-made stems with --vocal/--instrumental) -> render the score with the
DiffSinger OpenCpop voice -> mix the sung vocal over the instrumental.

The score, not the song, decides the words and the melody: a score you wrote or
one produced by scripts/song_to_score.py.  The `--start/--dur` pair only says
which slice of the backing track to use, so the vocal lines up with the
instrumental - not with the original singer's timing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ds_engine import DEFAULT_MODEL_DIR, SingingVoice  # noqa: E402
from separate import cut_audio, have_demucs, separate_centre, separate_demucs  # noqa: E402
from sing import load_audio, load_score  # noqa: E402


def build_stems(song, start, dur, work_dir, outdir, vocal=None, instrumental=None,
                method="auto"):
    """Return (vocal_path_or_None, instrumental_path)."""
    if instrumental:
        return vocal, instrumental

    if method == "auto":
        method = "demucs" if have_demucs() else "centre"
    print("separation: %s" % method)

    clip = os.path.join(work_dir, "clip.wav")
    cut_audio(song, clip, start, dur)
    os.makedirs(outdir, exist_ok=True)
    if method == "demucs":
        return separate_demucs(clip, outdir)
    return separate_centre(clip, outdir)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render an AI-voice cover over a song clip")
    parser.add_argument("--song", required=True, help="source audio file")
    parser.add_argument("--start", type=float, default=0.0, help="clip start, seconds")
    parser.add_argument("--dur", type=float, default=None, help="clip length, seconds")
    parser.add_argument("--score", required=True, help="score JSON with the words + melody")
    parser.add_argument("--out", required=True, help="output wav")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="where the sung vocal starts inside the clip, seconds")
    parser.add_argument("--vocal", default=None, help="pre-separated vocal stem (skips Demucs)")
    parser.add_argument("--instrumental", default=None,
                        help="pre-separated instrumental (skips Demucs)")
    parser.add_argument("--method", choices=["auto", "demucs", "centre"], default="auto")
    parser.add_argument("--work", default=None, help="directory for the separated stems")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--transpose", type=int, default=0)
    parser.add_argument("--f0", dest="f0_source", default="score",
                        choices=["score", "model", "model+fix"])
    parser.add_argument("--vocal-gain", type=float, default=0.0, help="vocal gain in dB")
    parser.add_argument("--peak", type=float, default=0.95)
    args = parser.parse_args(argv)

    if not os.path.isfile(args.song):
        raise SystemExit("song not found: %s" % args.song)
    if not os.path.isfile(args.score):
        raise SystemExit("score not found: %s" % args.score)

    work_dir = tempfile.mkdtemp(prefix="cover-")
    # Stems are scratch files unless the caller asked to keep them (--work).
    stem_dir = args.work or os.path.join(work_dir, "stems")
    try:
        vocal, instrumental = build_stems(
            args.song, args.start, args.dur, work_dir, stem_dir,
            args.vocal, args.instrumental, args.method)
        if not instrumental:
            raise SystemExit("no instrumental produced")

        _, syllables = load_score(args.score)
        voice = SingingVoice(model_dir=args.model_dir, f0_source=args.f0_source)
        audio = voice.render(syllables, transpose=args.transpose)
        audio = audio / (float(np.max(np.abs(audio))) or 1.0) * args.peak
        print("vocal: %.2f s (%d syllables)" % (len(audio) / voice.sr,
                                                sum(1 for s in syllables if "lyric" in s)))

        inst = load_audio(instrumental, voice.sr)
        if inst.size == 0:
            raise SystemExit("%s decoded to 0 samples" % instrumental)
        offset = int(round(args.offset * voice.sr))
        total = max(len(inst), offset + len(audio))
        mixed = np.zeros(total, dtype=np.float32)
        mixed[: len(inst)] += inst / (float(np.max(np.abs(inst))) or 1.0) * 0.85
        mixed[offset: offset + len(audio)] += audio * (10.0 ** (args.vocal_gain / 20.0)) * 0.55
        mixed = mixed / (float(np.max(np.abs(mixed))) or 1.0) * args.peak
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    import soundfile as sf

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    sf.write(args.out, mixed.astype(np.float32), voice.sr, subtype="PCM_16")
    print("wrote %s (%.2f s)" % (args.out, len(mixed) / voice.sr))
    if vocal is None:
        print("note: no vocal stem was used in this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
