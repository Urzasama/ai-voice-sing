---
name: ai-voice-sing
description: Make an AI voice sing Chinese lyrics on a melody you specify, on CPU, and mix the result over a song's separated instrumental. Use when the user asks to make a song "sung by AI", to build a cover from an mp3, to synthesize a vocal from lyrics plus notes, or to regenerate/extend the demo audio in this repository.
metadata:
  runtime: python 3.10-3.12, CPU only for synthesis
  model: DiffSinger OpenCpop ONNX (~220 MB, downloaded, not in repo)
  verified: 2026-11-27, Windows 11, Python 3.12.14
---

# ai-voice-sing

Render a Mandarin singing voice from a lyrics+melody score with the DiffSinger ONNX
(OpenCpop) model, then optionally mix it over the original song's instrumental.

Everything below was executed and verified on the development machine.  Anything not
verified is marked **[unverified]**.

## When to use

* "Make an AI sing this chorus", "cover this song with an AI voice".
* "Turn these lyrics + notes into a vocal wav".
* You need a specific pitch/rhythm, not a TTS voice reading text: the score decides both.

Do not use it for English singing (the phoneme table is Mandarin), for voice cloning (one
fixed female voice), or when a GPU-grade studio vocal is expected.

## 0. Environment check

```bash
python --version          # 3.10-3.12
ffmpeg -version | head -1 # only for mp3 input / cutting clips
```

Work from the repository root.  If `models/opencpop/config.yaml` is missing, install first:

```bash
pip install -r requirements.txt
python scripts/fetch_model.py        # 220 MB, sha256 checked
```

`fetch_model.py` accepts `--zip <local model.zip>` when there is no network, and `--force`
to re-extract.

## 1. Get an instrumental + vocal stem (optional)

```bash
python scripts/separate.py --input song.mp3 --start 86 --dur 16 --outdir work
# -> work/vocals.wav, work/instrumental.wav   (44.1 kHz stereo, Demucs htdemucs)
```

* Needs the optional stack (`pip install -r requirements-optional.txt`, pulls torch).
* 16 s clip: ~16 s on CPU.  A full 4-minute song takes minutes.
* Without Demucs the script falls back to centre cancellation: instrumental only, no clean
  vocal stem (`--method centre` forces it).
* Skip this step entirely if you already have stems, or if you only want the a-cappella
  voice.

## 2. Get a score

Either write `score.json` by hand (recommended - see the format below), or draft it:

```bash
python scripts/song_to_score.py --vocal work/vocals.wav --out scores/draft.json \
    --lyrics 雨爱的秘密能一直延续 --title "draft"
```

The draft is **approximate**: pYIN pitch runs, one syllable per detected note, placeholder
vowel `a` for notes beyond the supplied lyrics.  Always fix lyrics/pitches by hand before
trusting it.  `scripts/song_to_score.py` needs librosa+scipy.

Score format (`scripts/sing.py` reads it):

```json
{"title": "optional",
 "syllables": [
   {"lyric": "雨", "notes": ["B4"], "durs": [0.6]},
   {"rest": 0.15},
   {"lyric": "延", "notes": ["D#5", "E4"], "durs": [0.3, 0.3]}
 ]}
```

`lyric` = one Chinese character (pinyin only matters; `a`/`la` work as filler),
`notes` = scientific pitch per note, `durs` = seconds per note (same length as `notes`),
`rest` = seconds of silence.  Rests are also the safe chunk boundaries for the model.

## 3. Sing

A cappella:

```bash
python scripts/sing.py --score scores/yua-chorus.json --out demo/out.wav
# wrote demo/out.wav (11.05 s, 24000 Hz)     ~37 s on this laptop CPU
```

Over an instrumental:

```bash
python scripts/sing.py --score scores/yua-chorus.json \
    --mix work/instrumental.wav --mix-offset 1.0 --out demo/mix.wav
# -> vocal starts 1.0 s into the instrumental; dry sum, peak-normalised
```

Whole chain in one go:

```bash
python scripts/cover.py --song song.mp3 --start 86 --dur 16 \
    --score scores/yua-chorus.json --offset 1.0 --out work/cover.wav
```

Useful flags: `--transpose N` (semitones), `--f0 score|model|model+fix` (default `score` =
exact pitch; `model` uses the model's own pitch prediction, which has livelier glides but
drops octaves on short high notes), `--vocal-gain dB`, `--max-chunk 5.0`, `--seed N`.

## 4. Inspect the result

```bash
python scripts/plot_waveform.py demo/out.wav demo/out.png --title "AI voice"
```

Programmatic check that the pitch actually followed the score (needs librosa):

```python
import librosa, numpy as np
y, sr = librosa.load("demo/out.wav", sr=24000)
f0, v, p = librosa.pyin(y, fmin=librosa.note_to_hz("E3"), fmax=librosa.note_to_hz("C6"),
                        sr=sr, frame_length=2048, hop_length=256)
midi = 69 + 12 * np.log2(f0[v & (p > 0.2)] / 440)
print("median MIDI", np.median(midi))
```

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `Model not found in models/opencpop` | run `python scripts/fetch_model.py` |
| `ModuleNotFoundError: tqdm` / `librosa` | you are running the upstream `infer.py`, not the vendored patched one; use `scripts/sing.py` |
| `ValueError: zero-size array to reduction ... maximum` when mixing | the `--mix` file decoded to 0 samples - usually a cut past the end of the source; re-cut it |
| output is 0 bytes / silent | check `--score` has non-zero `durs`, and that `--model-dir` points at `config.yaml` + the five `*.onnx` files |
| long score fails inside ONNX | lower `--max-chunk` (default 5 s); the model caps at ~10 words per call |
| vowels sound garbled | lyrics must be Chinese characters (or `a`/`la`); check pinyin homophones |
| mixing sounds rhythmically off | expected: scoring is manual, the vocal is placed at `--offset` and not beat-locked |

## Notes for maintainers

* `vendor/diffsinger_onnx/` is a patched copy of HighCWu/DiffSinger-ONNX-Infer (MIT).  The
  patches are listed in README.md; keep them minimal so future upstream diffs stay readable.
* Weights stay out of git (`models/` is ignored); `demo/` is the only place with audio and
  it must stay under 3 MB.
* `--f0 score` exists because the shipped model's pitch predictor octave-slips on short
  D#5 notes; do not remove the hook without re-measuring.
