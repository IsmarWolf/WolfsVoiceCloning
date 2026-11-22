#!/usr/bin/env python3
"""
prepare_dataset.py - turn raw voice recordings into a training dataset.

What it does
  1. Converts every input file to mono WAV at --sr (default 44100 Hz) with ffmpeg,
     removes rumble below 60 Hz, and sets a consistent loudness (linear gain only).
  2. Transcribes with faster-whisper (Whisper large-v3 by default), with word timestamps.
  3. Cuts the audio into 5-15 s clips at natural pauses / sentence ends.
  4. Writes the clips plus transcript lists for every engine we may train:
       dataset/wavs/*.wav           16-bit PCM clips (also usable directly for RVC)
       dataset/gptsovits.list        path|speaker|lang|text   (GPT-SoVITS)
       dataset/f5tts/metadata.csv    audio_file|text          (F5-TTS finetune)
       dataset/metadata.csv          everything, incl. duration and Whisper confidence
       dataset/review.csv            low-confidence clips to check by ear
       dataset/rejected/             clips that were too short/long/unclear

Language
  Taken from the file name if it contains _pt_ / _en_ (or -pt- / -en-, or ends in _pt / _en),
  otherwise from --lang, otherwise Whisper auto-detects it per file.

Usage (Windows PowerShell)
  python prepare_dataset.py --input C:\\Users\\IsmarWolf\\VoiceClone\\raw --output C:\\Users\\IsmarWolf\\VoiceClone\\dataset --speaker ismar

Requirements
  pip install faster-whisper soundfile numpy num2words
  ffmpeg on PATH.
  GPU (optional, much faster): pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".wma", ".webm", ".mp4", ".mkv"}
LANG_RE = re.compile(r"(?:^|[_\-. ])(pt|en)(?:[_\-. ]|$)", re.IGNORECASE)
SENTENCE_END = re.compile(r"[.!?…]['\")\]]*$")
SOFT_BREAK = re.compile(r"[,;:—–-]['\")\]]*$")


# --------------------------------------------------------------------------- helpers
def add_cuda_dll_dirs() -> None:
    """On Windows, make pip-installed CUDA libs (nvidia-cublas-cu12, nvidia-cudnn-cu12) loadable."""
    if os.name != "nt":
        return
    try:
        import importlib.util

        for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime"):
            spec = importlib.util.find_spec(pkg)
            if spec and spec.submodule_search_locations:
                for loc in spec.submodule_search_locations:
                    for sub in ("bin", "lib"):
                        d = Path(loc) / sub
                        if d.is_dir():
                            os.add_dll_directory(str(d))
                            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def lang_from_name(name: str) -> str | None:
    m = LANG_RE.search(Path(name).stem)
    return m.group(1).lower() if m else None


def load_normalized(path: Path, sr: int, target_rms_db: float, peak_db: float) -> np.ndarray:
    """ffmpeg -> mono float32 at sr with 60 Hz high-pass, then linear gain to target RMS (peak-limited)."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "a.wav"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
            "-ac", "1", "-ar", str(sr), "-af", "highpass=f=60", "-c:a", "pcm_f32le", str(tmp),
        ]
        subprocess.run(cmd, check=True)
        audio, _ = sf.read(tmp, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # loudness from the louder (speech) parts only, so silence doesn't skew it
    frame = int(sr * 0.05)
    n = len(audio) // frame
    if n == 0:
        return audio
    frames = audio[: n * frame].reshape(n, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    speech = rms[rms > np.percentile(rms, 40)]
    cur_rms = float(np.sqrt((speech ** 2).mean())) if len(speech) else float(rms.mean())
    peak = float(np.abs(audio).max()) + 1e-9
    gain = min(10 ** (target_rms_db / 20) / (cur_rms + 1e-9), 10 ** (peak_db / 20) / peak)
    return (audio * gain).astype(np.float32)


def clip_stats(audio: np.ndarray) -> float:
    """Return peak in dBFS (for clipping detection on the raw input)."""
    return 20 * np.log10(float(np.abs(audio).max()) + 1e-9)


_num2words = None


def spell_numbers(text: str, lang: str) -> str:
    """'em 2024, 37%' -> 'em dois mil e vinte e quatro, trinta e sete por cento'. Best effort."""
    global _num2words
    if _num2words is None:
        try:
            from num2words import num2words as n2w

            _num2words = n2w
        except ImportError:
            _num2words = False
            print("  [warn] num2words not installed; numbers stay as digits (pip install num2words)")
    if not _num2words:
        return text
    nlang = "pt_BR" if lang == "pt" else "en"
    pct = " por cento" if lang == "pt" else " percent"

    def repl(m: re.Match) -> str:
        raw, is_pct = m.group(1), m.group(2)
        try:
            if lang == "pt":
                val = raw.replace(".", "").replace(",", ".")  # 1.234,5 -> 1234.5
            else:
                val = raw.replace(",", "")  # 1,234.5 -> 1234.5
            num = float(val) if "." in val else int(val)
            return _num2words(num, lang=nlang) + (pct if is_pct else "")
        except Exception:
            return m.group(0)

    return re.sub(r"(\d[\d.,]*\d|\d)(\s?%)?", repl, text)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("|", " ")
    return text


# --------------------------------------------------------------------------- clip building
@dataclass
class Word:
    start: float
    end: float
    text: str
    prob: float


@dataclass
class Clip:
    words: list[Word] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def dur(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        return "".join(w.text for w in self.words).strip()

    @property
    def prob(self) -> float:
        return float(np.mean([w.prob for w in self.words])) if self.words else 0.0


def build_clips(words: list[Word], min_s: float, max_s: float, gap_s: float) -> list[Clip]:
    """
    Greedy: walk the words, close a clip at a good break once it is >= min_s,
    and never let it exceed max_s. Good breaks, best first:
      sentence end + pause  >  pause >= gap_s  >  comma + short pause  >  any word gap.
    """
    clips: list[Clip] = []
    i = 0
    n = len(words)
    while i < n:
        best_j, best_score = None, -1.0
        j = i
        while j < n and words[j].end - words[i].start <= max_s:
            dur = words[j].end - words[i].start
            nxt_gap = (words[j + 1].start - words[j].end) if j + 1 < n else 10.0
            w = words[j].text.strip()
            score = nxt_gap
            if SENTENCE_END.search(w):
                score += 1.0
            elif SOFT_BREAK.search(w):
                score += 0.3
            if dur >= min_s and score > best_score:
                best_j, best_score = j, score
                # a clear sentence end with a real pause is good enough - stop early
                if SENTENCE_END.search(w) and nxt_gap >= gap_s and dur >= min_s + 1.5:
                    break
            j += 1
        if best_j is None:
            # could not reach min_s within max_s (end of file or one giant word run)
            best_j = max(i, j - 1)
        clips.append(Clip(words[i : best_j + 1]))
        i = best_j + 1
    return clips


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Split + transcribe voice recordings into a TTS dataset.")
    ap.add_argument("--input", required=True, type=Path, help="folder with raw recordings")
    ap.add_argument("--output", required=True, type=Path, help="dataset folder to create")
    ap.add_argument("--speaker", default="me", help="speaker name used in lists")
    ap.add_argument("--lang", choices=["pt", "en", "auto"], default="auto",
                    help="language when the file name has no _pt_/_en_ tag")
    ap.add_argument("--model", default="large-v3", help="faster-whisper model (large-v3, large-v3-turbo, medium...)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--sr", type=int, default=44100, help="output sample rate")
    ap.add_argument("--min", dest="min_s", type=float, default=5.0)
    ap.add_argument("--max", dest="max_s", type=float, default=15.0)
    ap.add_argument("--hard-min", type=float, default=2.0, help="clips shorter than this go to rejected/")
    ap.add_argument("--min-prob", type=float, default=0.55, help="mean word prob below this -> rejected/")
    ap.add_argument("--review-prob", type=float, default=0.75, help="mean word prob below this -> review.csv")
    ap.add_argument("--no-spell-numbers", action="store_true", help="keep digits in transcripts")
    ap.add_argument("--pad-start", type=float, default=0.12)
    ap.add_argument("--pad-end", type=float, default=0.25)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found on PATH")
        return 1
    files = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    if not files:
        print(f"No audio files in {args.input}")
        return 1

    out = args.output
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            print(f"{out} is not empty. Use --overwrite to replace its contents.")
            return 1
        for sub in ("wavs", "rejected", "f5tts"):
            shutil.rmtree(out / sub, ignore_errors=True)
    wav_dir, rej_dir, f5_dir = out / "wavs", out / "rejected", out / "f5tts"
    for d in (wav_dir, rej_dir, f5_dir):
        d.mkdir(parents=True, exist_ok=True)

    add_cuda_dll_dirs()
    from faster_whisper import WhisperModel

    device = args.device
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    print(f"Loading Whisper {args.model} on {device} ({compute})...")
    try:
        model = WhisperModel(args.model, device=device, compute_type=compute)
    except Exception as e:
        if device == "cuda":
            print(f"  CUDA failed ({e}); falling back to CPU (slower).")
            device, compute = "cpu", "int8"
            model = WhisperModel(args.model, device=device, compute_type=compute)
        else:
            raise

    rows, rejected = [], []
    totals: dict[str, float] = {}
    for f in files:
        lang = lang_from_name(f.name) or (None if args.lang == "auto" else args.lang)
        print(f"\n[{f.name}] lang={lang or 'auto-detect'}")
        raw_peak = None
        try:
            info = sf.info(str(f))
            if info.format in ("WAV", "FLAC"):
                raw, _ = sf.read(str(f), dtype="float32", always_2d=True)
                raw_peak = clip_stats(raw)
                near_full = float((np.abs(raw) > 0.999).mean())
                if near_full > 1e-4:
                    print(f"  [warn] looks clipped ({near_full*100:.3f}% samples at full scale). Consider re-recording.")
        except Exception:
            pass
        audio = load_normalized(f, args.sr, target_rms_db=-20.0, peak_db=-1.0)
        if raw_peak is not None:
            print(f"  raw peak {raw_peak:.1f} dBFS, {len(audio)/args.sr/60:.1f} min")

        # Whisper wants 16 kHz; let it resample from our normalized array via a temp file
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "w.wav"
            sf.write(tmp, audio, args.sr, subtype="PCM_16")
            segments, winfo = model.transcribe(
                str(tmp), language=lang, word_timestamps=True, vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300}, beam_size=5,
                condition_on_previous_text=False,
            )
            words: list[Word] = []
            for seg in segments:
                for w in seg.words or []:
                    if w.word.strip():
                        words.append(Word(w.start, w.end, w.word, w.probability))
        lang = lang or winfo.language
        if lang not in ("pt", "en"):
            print(f"  [warn] detected language '{lang}' - treating as given; check this file")
        print(f"  {len(words)} words, language={lang}")

        clips = build_clips(words, args.min_s, args.max_s, gap_s=0.35)
        total = len(audio) / args.sr
        for k, c in enumerate(clips):
            prev_end = clips[k - 1].end if k > 0 else 0.0
            next_start = clips[k + 1].start if k + 1 < len(clips) else total
            s = max(c.start - args.pad_start, (prev_end + c.start) / 2, 0.0)
            e = min(c.end + args.pad_end, (c.end + next_start) / 2, total)
            seg_audio = audio[int(s * args.sr) : int(e * args.sr)]
            # gentle 10 ms fades to avoid clicks
            fl = min(int(0.01 * args.sr), len(seg_audio) // 4)
            if fl > 0:
                seg_audio = seg_audio.copy()
                seg_audio[:fl] *= np.linspace(0, 1, fl, dtype=np.float32)
                seg_audio[-fl:] *= np.linspace(1, 0, fl, dtype=np.float32)
            dur = len(seg_audio) / args.sr
            text = clean_text(c.text)
            if not args.no_spell_numbers:
                text_tts = clean_text(spell_numbers(text, lang))
            else:
                text_tts = text
            name = f"{f.stem}_{k:04d}.wav"
            reason = None
            if dur < args.hard_min:
                reason = "too_short"
            elif dur > args.max_s + 2.0:
                reason = "too_long"
            elif c.prob < args.min_prob:
                reason = "low_confidence"
            elif len(text) < 3:
                reason = "no_text"
            row = dict(file=name, source=f.name, lang=lang, duration=round(dur, 2),
                       prob=round(c.prob, 3), text=text_tts, text_raw=text)
            if reason:
                sf.write(rej_dir / name, seg_audio, args.sr, subtype="PCM_16")
                row["reason"] = reason
                rejected.append(row)
            else:
                sf.write(wav_dir / name, seg_audio, args.sr, subtype="PCM_16")
                rows.append(row)
                totals[lang] = totals.get(lang, 0.0) + dur
        kept = sum(1 for r in rows if r["source"] == f.name)
        print(f"  {kept} clips kept, {sum(1 for r in rejected if r['source'] == f.name)} rejected")

    # ---------------- write lists
    # GPT-SoVITS only knows zh/ja/en/ko/yue, so Portuguese lines are left out of its list
    sovits_rows = [r for r in rows if r["lang"] in ("en", "zh", "ja", "ko", "yue")]
    with open(out / "gptsovits.list", "w", encoding="utf-8", newline="\n") as fh:
        for r in sovits_rows:
            fh.write(f"{(wav_dir / r['file']).resolve()}|{args.speaker}|{r['lang']}|{r['text']}\n")
    if len(sovits_rows) < len(rows):
        print(f"  note: {len(rows) - len(sovits_rows)} non-English clips left out of gptsovits.list "
              "(GPT-SoVITS has no Portuguese)")
    # F5-TTS finetune expects lowercase-ish text; keep punctuation, lowercase it
    with open(f5_dir / "metadata.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("audio_file|text\n")
        for r in rows:
            fh.write(f"{(wav_dir / r['file']).resolve()}|{r['text'].lower()}\n")
    for lang in sorted(totals):
        with open(f5_dir / f"metadata_{lang}.csv", "w", encoding="utf-8", newline="\n") as fh:
            fh.write("audio_file|text\n")
            for r in rows:
                if r["lang"] == lang:
                    fh.write(f"{(wav_dir / r['file']).resolve()}|{r['text'].lower()}\n")
    fields = ["file", "source", "lang", "duration", "prob", "text", "text_raw"]
    with open(out / "metadata.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="|")
        w.writeheader()
        w.writerows(rows)
    with open(out / "review.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields + ["reason"], delimiter="|", extrasaction="ignore")
        w.writeheader()
        review = [dict(r, reason="check") for r in rows if r["prob"] < args.review_prob]
        w.writerows(sorted(review, key=lambda r: r["prob"]) + rejected)

    # ---------------- summary
    durs = [r["duration"] for r in rows]
    print("\n=== Summary ===")
    for lang, t in sorted(totals.items()):
        print(f"  {lang}: {t/60:.1f} min in {sum(1 for r in rows if r['lang']==lang)} clips")
    if durs:
        print(f"  clip length: min {min(durs):.1f}s, median {np.median(durs):.1f}s, max {max(durs):.1f}s")
    print(f"  rejected: {len(rejected)} (see {rej_dir})")
    print(f"  to check by ear: {sum(1 for r in rows if r['prob'] < args.review_prob)} (see review.csv)")
    print(f"\nWrote {out/'gptsovits.list'}, {f5_dir/'metadata.csv'}, {out/'metadata.csv'}")
    print("Tip: open review.csv, listen to those clips, and fix or delete lines in the lists if Whisper misheard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
