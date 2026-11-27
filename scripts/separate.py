#!/usr/bin/env python3
"""Split a song (or a clip of it) into a vocal stem and an instrumental.

    python scripts/separate.py --input song.mp3 --start 86 --dur 16 --outdir work/

Uses Demucs `htdemucs` (needs the optional dependencies: torch + demucs, see
requirements-optional.txt).  Separation is CPU-only here and is slow: a 16 s
clip takes roughly half a minute on a modern laptop, a full 4-minute song a
few minutes.

If Demucs is not installed, `--method centre` falls back to a crude
left-minus-right centre cancellation.  That only produces an *instrumental*
with the lead vocal partially removed - it cannot recover a clean vocal stem.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile


def have_demucs():
    try:
        import demucs  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def ffmpeg_bin():
    exe = shutil.which("ffmpeg")
    if not exe:
        raise SystemExit("ffmpeg not found in PATH; install ffmpeg first")
    return exe


def cut_audio(src, dst, start=0.0, dur=None):
    """Cut [start, start+dur) out of `src` into `dst` with ffmpeg."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y"]
    if start > 0:
        cmd += ["-ss", "%.3f" % start]
    cmd += ["-i", src]
    if dur is not None:
        cmd += ["-t", "%.3f" % dur]
    cmd += ["-ac", "2", "-c:a", "pcm_s16le", dst]
    subprocess.run(cmd, check=True)
    return dst


def separate_demucs(path, outdir, model="htdemucs"):
    """Run demucs as a subprocess; returns (vocals_path, instrumental_path)."""
    work = tempfile.mkdtemp(prefix="demucs-")
    try:
        cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals",
               "-n", model, "-o", work, path]
        print("running: %s" % " ".join(cmd))
        subprocess.run(cmd, check=True)
        stem_dir = os.path.join(work, model, os.path.splitext(os.path.basename(path))[0])
        vocals = os.path.join(stem_dir, "vocals.wav")
        instrumental = os.path.join(stem_dir, "no_vocals.wav")
        if not (os.path.isfile(vocals) and os.path.isfile(instrumental)):
            raise SystemExit("demucs did not produce the expected stems in %s" % stem_dir)
        out_v = os.path.join(outdir, "vocals.wav")
        out_i = os.path.join(outdir, "instrumental.wav")
        shutil.copyfile(vocals, out_v)
        shutil.copyfile(instrumental, out_i)
        return out_v, out_i
    finally:
        shutil.rmtree(work, ignore_errors=True)


def separate_centre(path, outdir):
    """Crude centre cancellation: instrumental = L - R (vocals cannot be kept)."""
    out_i = os.path.join(outdir, "instrumental.wav")
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", "-i", path,
           "-af", "pan=mono|c0=0.5*c0-0.5*c1", "-ar", "44100", "-c:a", "pcm_s16le", out_i]
    subprocess.run(cmd, check=True)
    print("warning: centre cancellation only produced an instrumental; "
          "there is no clean vocal stem in this mode")
    return None, out_i


def main(argv=None):
    parser = argparse.ArgumentParser(description="Split a song into vocals + instrumental")
    parser.add_argument("--input", required=True, help="song file (mp3/wav/...)")
    parser.add_argument("--outdir", default="work", help="output directory (default work)")
    parser.add_argument("--start", type=float, default=0.0, help="clip start in seconds")
    parser.add_argument("--dur", type=float, default=None, help="clip length in seconds")
    parser.add_argument("--method", choices=["auto", "demucs", "centre"], default="auto",
                        help="auto = demucs if installed, else centre cancellation")
    parser.add_argument("--model", default="htdemucs", help="demucs model name")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.input):
        raise SystemExit("input not found: %s" % args.input)
    os.makedirs(args.outdir, exist_ok=True)

    method = args.method
    if method == "auto":
        method = "demucs" if have_demucs() else "centre"
        print("separation method: %s" % method)

    work_dir = tempfile.mkdtemp(prefix="sep-")
    try:
        src = args.input
        if args.start or args.dur is not None:
            src = cut_audio(args.input, os.path.join(work_dir, "clip.wav"),
                            args.start, args.dur)
        if method == "demucs":
            vocals, instrumental = separate_demucs(src, args.outdir, args.model)
        else:
            vocals, instrumental = separate_centre(src, args.outdir)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    print("vocals:       %s" % (vocals or "(not produced)"))
    print("instrumental: %s" % instrumental)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
