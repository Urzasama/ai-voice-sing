"""Timeline-accurate wrapper around the vendored DiffSinger ONNX inference code.

The upstream `sing.py` synthesises a whole score in one shot and concatenates
chunks with a fixed silence gap.  For singing on top of an existing
instrumental we need sample-accurate timing instead, so this module:

1. splits the score into chunks that the ONNX graphs can handle,
2. renders each chunk,
3. writes every chunk at its exact cumulative timestamp in one output buffer.

Only onnxruntime + numpy + pypinyin + PyYAML + six are required (no torch).
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR_ROOT = os.path.join(REPO_ROOT, "vendor", "diffsinger_onnx")
DEFAULT_MODEL_DIR = os.path.join(REPO_ROOT, "models", "opencpop")

# The OpenCpop graphs were exported for short utterances; the previous
# experiment hit failures somewhere above ten syllables, so stay well below.
DEFAULT_MAX_CHUNK_SECONDS = 5.0


def midi_to_name(midi: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    midi = int(round(midi))
    return "%s%d" % (names[midi % 12], midi // 12 - 1)


_NOTE_OFFSET = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_to_midi(name: str) -> int:
    """C#4 / Db4 / F5 -> MIDI number (same convention as librosa)."""
    name = name.strip()
    if not name or name.lower() == "rest":
        return 0
    letter, rest = name[0].upper(), name[1:]
    semitone = 0
    while rest and rest[0] in "#b":
        semitone += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    return 12 * (int(rest) + 1) + _NOTE_OFFSET[letter] + semitone


def transpose_score(syllables, semitones: int):
    """Return a copy of the score with every note shifted by `semitones`."""
    if not semitones:
        return [dict(s) for s in syllables]
    out = []
    for syl in syllables:
        if "rest" in syl:
            out.append(dict(syl))
            continue
        notes = [midi_to_name(note_to_midi(n) + semitones) for n in syl["notes"]]
        out.append({"lyric": syl["lyric"], "notes": notes, "durs": list(syl["durs"])})
    return out


class SingingVoice:
    def __init__(self, model_dir: str = None, vendor_root: str = VENDOR_ROOT,
                 max_chunk_seconds: float = DEFAULT_MAX_CHUNK_SECONDS, seed: int = 0,
                 f0_source: str = "score", vibrato: float = 0.25):
        model_dir = os.path.abspath(model_dir or DEFAULT_MODEL_DIR)
        vendor_root = os.path.abspath(vendor_root)
        self.model_dir = model_dir
        self.vendor_root = vendor_root
        self.max_chunk_seconds = float(max_chunk_seconds)
        self.seed = seed
        config_path = os.path.join(model_dir, "config.yaml")
        if not os.path.isfile(config_path):
            raise SystemExit(
                "Model not found in %s\nRun: python scripts/fetch_model.py" % model_dir
            )

        # configs and the pinyin table are loaded through CWD-relative paths, so
        # the vendor tree is the working directory while the model loads.  The
        # caller's CWD is restored afterwards (see _vendor_cwd).
        if vendor_root not in sys.path:
            sys.path.insert(0, vendor_root)

        from utils.hparams import hparams, set_hparams
        from infer import Infer

        with self._vendor_cwd():
            np.random.seed(seed)
            set_hparams(config_path)
            with contextlib.redirect_stdout(io.StringIO()):
                self._infer = Infer(hparams, model_dir=model_dir)
        self.sr = int(hparams["audio_sample_rate"])
        self.fps = float(self.sr) / float(hparams["hop_size"])
        self.f0_source = f0_source
        self.vibrato = float(vibrato)

    # ------------------------------------------------------------- vendor cwd
    @contextlib.contextmanager
    def _vendor_cwd(self):
        """Run a block with the vendored tree as CWD, then restore the old one.

        The upstream `Infer` opens `inference/svs/opencpop/cpop_pinyin2ph.txt`
        and resolves its YAML configs relative to the CWD, so the process has to
        stand inside `vendor/diffsinger_onnx` while it loads.  Keeping the chdir
        scoped means caller-relative paths (--score, --out, --mix) keep working.
        """
        previous = os.getcwd()
        os.chdir(self.vendor_root)
        try:
            yield
        finally:
            os.chdir(previous)

    # ------------------------------------------------------------------ score
    @staticmethod
    def _flatten(syllables):
        """-> list of (text, [notes], [durs]) in seconds."""
        words = []
        for syl in syllables:
            if "rest" in syl:
                words.append(("SP", ["rest"], [float(syl["rest"])]))
                continue
            lyric = str(syl["lyric"]).strip()
            notes = [str(n).strip() for n in syl["notes"]]
            durs = [float(d) for d in syl["durs"]]
            if len(notes) != len(durs):
                raise ValueError("notes/durs length mismatch for %r" % lyric)
            words.append((lyric, notes, durs))
        return words

    def _chunks(self, words):
        chunks, current, current_len = [], [], 0.0
        for word in words:
            dur = float(sum(word[2]))
            if current and current_len + dur > self.max_chunk_seconds:
                chunks.append(current)
                current, current_len = [], 0.0
            current.append(word)
            current_len += dur
        if current:
            chunks.append(current)
        return chunks

    # ------------------------------------------------------------------- f0
    def _f0_curve(self, words):
        """Build an F0 curve (Hz) for one chunk from its note schedule.

        The PE model inside the ONNX graph re-estimates F0 from the predicted
        mel spectrogram and occasionally lands an octave off on short high
        notes.  Deriving F0 from the score keeps the melody exact; a little
        portamento and vibrato are added so it does not sound like a flat synth.
        """
        duration = float(sum(sum(w[2]) for w in words))
        frames = max(1, int(round(duration * self.fps)))
        midi = np.zeros(frames, dtype=np.float32)
        t = 0.0
        prev = None
        for word in words:
            for note, dur in zip(word[1], word[2]):
                a = int(round(t * self.fps))
                t += float(dur)
                b = min(frames, int(round(t * self.fps)))
                if b <= a:
                    continue
                if note == "rest":
                    prev = None
                    continue
                value = float(note_to_midi(note))
                seg = np.full(b - a, value, dtype=np.float32)
                if prev is not None and abs(prev - value) > 1e-6:
                    ramp = min(6, (b - a) // 2)
                    if ramp > 1:
                        seg[:ramp] = np.linspace(prev, value, ramp)
                if self.vibrato > 0 and dur >= 0.30 and (b - a) > 12:
                    tt = np.arange(b - a, dtype=np.float32) / self.fps
                    depth = np.clip((tt - 0.10) / 0.15, 0.0, 1.0) * self.vibrato
                    seg = seg + depth * np.sin(2 * np.pi * 5.5 * tt)
                midi[a:b] = seg
                prev = value
        hz = np.zeros(frames, dtype=np.float32)
        voiced = midi > 0
        hz[voiced] = 440.0 * np.power(2.0, (midi[voiced] - 69.0) / 12.0)
        return hz[None, :]

    def _octave_fix(self, f0, words):
        """Correct PE octave errors against the score, keeping the natural curve.

        Returns a corrected F0 array (1, T) or None if nothing had to change.
        The PE model sometimes locks onto half the true frequency for short
        high notes; multiplying that span by 2 keeps the original glides and
        vibrato, only moving it back into the right octave.
        """
        curve = np.array(f0, dtype=np.float32).reshape(-1)
        total_frames = len(curve)
        duration = float(sum(sum(w[2]) for w in words))
        if duration <= 0 or total_frames <= 0:
            return None
        scale = total_frames / (duration * self.fps)
        fixes = 0
        t = 0.0
        for word in words:
            for note, dur in zip(word[1], word[2]):
                a = int(round(t * self.fps * scale))
                t += float(dur)
                b = min(total_frames, int(round(t * self.fps * scale)))
                if note == "rest" or b - a < 5:
                    continue
                head = a + (b - a) // 5
                tail = b - (b - a) // 5
                span = curve[head:tail]
                span = span[span > 0]
                if len(span) < 3:
                    continue
                target = 440.0 * 2.0 ** ((note_to_midi(note) - 69.0) / 12.0)
                error = 12.0 * np.log2(float(np.median(span)) / target)
                if abs(error) > 6.0:
                    shift = int(-round(error / 12.0))
                    curve[a:b] = curve[a:b] * (2.0 ** shift)
                    fixes += 1
        if not fixes:
            return None
        print("octave-corrected %d note(s)" % fixes)
        return curve[None, :]

    # --------------------------------------------------------------- synthesis
    def render(self, syllables, transpose: int = 0) -> np.ndarray:
        """Synthesise the score and return mono float32 audio at self.sr."""
        syllables = transpose_score(syllables, transpose)
        words = self._flatten(syllables)
        total = float(sum(sum(w[2]) for w in words))
        n_total = int(round(total * self.sr))
        buffer = np.zeros(n_total + self.sr // 2, dtype=np.float32)

        t0 = 0.0
        chunks = self._chunks(words)
        for index, chunk in enumerate(chunks):
            text = " ".join(w[0] for w in chunk)
            notes = " | ".join(" ".join(w[1]) for w in chunk)
            durs = " | ".join(" ".join("%.4f" % d for d in w[2]) for w in chunk)
            expected = float(sum(sum(w[2]) for w in chunk))
            with self._vendor_cwd():
                if self.f0_source == "score":
                    self._infer.f0_override = self._f0_curve(chunk)
                else:
                    self._infer.f0_override = None
                wav = self._infer.infer_once(
                    {"text": text, "notes": notes, "notes_duration": durs}
                )
                if wav is None:
                    raise RuntimeError("DiffSinger rejected chunk %d: %r" % (index, text))
                if self.f0_source == "model+fix" \
                        and getattr(self._infer, "last_f0", None) is not None:
                    fixed = self._octave_fix(self._infer.last_f0, chunk)
                    if fixed is not None:
                        wav = self._infer.run_vocoder(self._infer.last_mel, f0=fixed)[0]
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            expected_n = int(round(expected * self.sr))
            start = int(round(t0 * self.sr))
            if len(wav) >= expected_n:
                wav = wav[:expected_n]
            else:
                wav = np.pad(wav, (0, expected_n - len(wav)))
            end = min(start + len(wav), len(buffer))
            buffer[start:end] += wav[: end - start]
            t0 += expected

        return buffer[:n_total]

    def render_to_file(self, syllables, path: str, transpose: int = 0,
                       peak: float = 0.95) -> np.ndarray:
        import soundfile as sf

        audio = self.render(syllables, transpose=transpose)
        peak_value = float(np.max(np.abs(audio))) or 1.0
        audio = (audio / peak_value * peak).astype(np.float32)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        sf.write(path, audio, self.sr, subtype="PCM_16")
        return audio
