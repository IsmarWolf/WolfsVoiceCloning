#!/usr/bin/env python3
"""
narrate.py - text in, narration audio in your cloned voice out.

  python narrate.py --text "Olá! Hoje eu vou te mostrar..." --lang pt --out C:\\videos\\vo\\clip1.wav
  python narrate.py --text-file script_en.txt --lang en --out vo.mp3 --json
  echo "Some text" | python narrate.py --lang en --out vo.wav

The engine for each language is set in voice_config.toml ([routing]):
  gptsovits   GPT-SoVITS api_v2.py server (English and other zh/ja/ko/yue). Started automatically.
  chatterbox  Chatterbox Multilingual (Portuguese ok), optionally followed by your RVC model.
  f5tts       F5-TTS (your fine-tuned checkpoint), optionally followed by RVC.

Only the Python standard library is needed to run this file; each engine runs in its own
environment, set in the config. --json prints {"path":..., "duration":..., "engine":...}
so a video pipeline can read the result.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import wave
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    print("Python 3.11+ is required (tomllib).", file=sys.stderr)
    raise

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "voice_config.toml"
sys.path.insert(0, str(HERE))
import portable  # noqa: E402

portable.add_tools_to_path()


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- text
def spell_numbers(text: str, lang: str) -> str:
    try:
        from num2words import num2words
    except ImportError:
        return text
    nlang = {"pt": "pt_BR", "en": "en"}.get(lang, lang)
    pct = {"pt": " por cento", "en": " percent"}.get(lang, "%")

    def repl(m):
        raw, is_pct = m.group(1), m.group(2)
        try:
            val = raw.replace(".", "").replace(",", ".") if lang == "pt" else raw.replace(",", "")
            num = float(val) if "." in val else int(val)
            return num2words(num, lang=nlang) + (pct if is_pct else "")
        except Exception:
            return m.group(0)

    return re.sub(r"(\d[\d.,]*\d|\d)(\s?%)?", repl, text)


def split_text(text: str, max_chars: int) -> list[str]:
    """Split into sentence-based chunks of at most max_chars (keeps sentences whole when possible)."""
    text = re.sub(r"[ \t]+", " ", text.strip())
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    chunks: list[str] = []
    for p in paras:
        sentences = re.split(r"(?<=[.!?…])\s+", p)
        cur = ""
        for s in sentences:
            if len(s) > max_chars:  # very long sentence: split at commas
                parts = re.split(r"(?<=[,;:])\s+", s)
            else:
                parts = [s]
            for part in parts:
                if cur and len(cur) + 1 + len(part) > max_chars:
                    chunks.append(cur)
                    cur = part
                else:
                    cur = f"{cur} {part}".strip()
        if cur:
            chunks.append(cur)
    return chunks


# --------------------------------------------------------------------------- audio
def run(cmd: list[str], cwd: str | None = None, env: dict | None = None) -> None:
    log("  $", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    # child output goes to stderr so that stdout stays clean for --json
    r = subprocess.run(cmd, cwd=cwd, env=env, stdout=sys.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd[0]}")


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
        return float(out.stdout.strip() or 0)


def concat_and_finish(parts: list[Path], out: Path, cfg: dict, speed: float) -> None:
    """Join chunks with a short pause, apply speed, loudness-normalize, encode by extension."""
    g = cfg.get("general", {})
    ff = g.get("ffmpeg", "ffmpeg")
    pause = g.get("pause_ms", 250) / 1000.0
    sr = g.get("sample_rate", 44100)
    lufs = g.get("loudness_lufs", -16)
    with tempfile.TemporaryDirectory() as td:
        lst = Path(td) / "list.txt"
        silence = Path(td) / "sil.wav"
        run([ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             f"anullsrc=r={sr}:cl=mono", "-t", f"{pause}", str(silence)])
        norm = []
        for i, p in enumerate(parts):
            q = Path(td) / f"p{i:04d}.wav"
            run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(p), "-ac", "1", "-ar", str(sr), str(q)])
            norm.append(q)
        with open(lst, "w", encoding="utf-8") as fh:
            for i, q in enumerate(norm):
                if i:
                    fh.write(f"file '{silence.as_posix()}'\n")
                fh.write(f"file '{q.as_posix()}'\n")
        filters = []
        if abs(speed - 1.0) > 1e-3:
            filters.append(f"atempo={max(0.5, min(2.0, speed))}")
        filters.append(f"loudnorm=I={lufs}:TP=-1.5:LRA=11")
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
               "-af", ",".join(filters), "-ar", str(sr), "-ac", "1"]
        ext = out.suffix.lower()
        if ext == ".mp3":
            cmd += ["-c:a", "libmp3lame", "-b:a", "192k"]
        elif ext in (".m4a", ".aac"):
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-c:a", "pcm_s16le"]
        run(cmd + [str(out)])


# --------------------------------------------------------------------------- RVC post-pass
def rvc_convert(src: Path, dst: Path, cfg: dict) -> None:
    """Run the RVC command template from [rvc] on one file."""
    r = cfg["rvc"]
    subs = {"input": str(Path(src).resolve()), "output": str(Path(dst).resolve()), **{k: str(v) for k, v in r.items() if not isinstance(v, (list, dict))}}
    cmd = [part.format(**subs) for part in r["command"]]
    run(cmd, cwd=r.get("cwd") or None)
    if not Path(dst).resolve().exists():
        raise RuntimeError(f"RVC did not write {dst}. Check [rvc].command in the config.")


# --------------------------------------------------------------------------- engines
def port_open(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def http_get(url: str, timeout: float = 600) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def engine_gptsovits(chunks: list[str], lang: str, cfg: dict, td: Path) -> list[Path]:
    c = cfg["gptsovits"]
    host, port = c.get("host", "127.0.0.1"), int(c.get("port", 9880))
    base = f"http://{host}:{port}"
    if not port_open(host, port):
        log(f"Starting GPT-SoVITS API on {base} ...")
        cmd = [c["python"], "api_v2.py", "-a", host, "-p", str(port), "-c", c.get("config", "GPT_SoVITS/configs/tts_infer.yaml")]
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        logf = open(Path(c["root"]) / "narrate_api.log", "ab")
        subprocess.Popen(cmd, cwd=c["root"], stdout=logf, stderr=subprocess.STDOUT, creationflags=flags)
        for _ in range(240):
            if port_open(host, port):
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"GPT-SoVITS API did not start; see {Path(c['root']) / 'narrate_api.log'}")
    # load your fine-tuned weights (cheap if already loaded)
    for ep, key in (("set_gpt_weights", "gpt_weights"), ("set_sovits_weights", "sovits_weights")):
        if c.get(key):
            http_get(f"{base}/{ep}?" + urllib.parse.urlencode({"weights_path": c[key]}))
    ref_lang = c.get("ref_lang", "en")
    outs = []
    for i, text in enumerate(chunks):
        body = {
            "text": text, "text_lang": lang,
            "ref_audio_path": c["ref_audio"], "prompt_text": c.get("ref_text", ""), "prompt_lang": ref_lang,
            "aux_ref_audio_paths": c.get("aux_ref_audios", []),
            "top_k": c.get("top_k", 15), "top_p": c.get("top_p", 1.0), "temperature": c.get("temperature", 1.0),
            "text_split_method": c.get("text_split_method", "cut5"), "batch_size": 1,
            "speed_factor": 1.0, "repetition_penalty": c.get("repetition_penalty", 1.35),
            "seed": c.get("seed", -1), "media_type": "wav", "streaming_mode": False,
            "sample_steps": c.get("sample_steps", 32), "super_sampling": c.get("super_sampling", False),
        }
        req = urllib.request.Request(f"{base}/tts", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"GPT-SoVITS error: {e.read().decode('utf-8', 'replace')}") from e
        p = td / f"gsv_{i:04d}.wav"
        p.write_bytes(data)
        outs.append(p)
        log(f"  chunk {i+1}/{len(chunks)} ok")
    return outs


def engine_chatterbox(chunks: list[str], lang: str, cfg: dict, td: Path) -> list[Path]:
    c = cfg["chatterbox"]
    job = {
        "lang": lang, "ref_audio": c.get("ref_audio"), "device": c.get("device", "cuda"),
        "params": {k: c[k] for k in ("exaggeration", "cfg_weight", "temperature", "repetition_penalty") if k in c},
        "seed": c.get("seed", 0),
        "chunks": [{"text": t, "out": str(td / f"cb_{i:04d}.wav")} for i, t in enumerate(chunks)],
    }
    jf = td / "job.json"
    jf.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    run([c["python"], str(HERE / "engines" / "chatterbox_worker.py"), str(jf)])
    return [Path(ch["out"]) for ch in job["chunks"]]


def engine_f5tts(chunks: list[str], lang: str, cfg: dict, td: Path) -> list[Path]:
    c = cfg["f5tts"]
    out_dir = td / "f5"
    out_dir.mkdir()
    text = " ".join(chunks)  # F5-TTS does its own chunking; one call = one model load
    cmd = [c["cli"], "--model", c.get("model", "F5TTS_v1_Base"),
           "--ref_audio", c["ref_audio"], "--ref_text", c.get("ref_text", ""),
           "--gen_text", text, "--output_dir", str(out_dir), "--output_file", "f5.wav"]
    if c.get("ckpt_file"):
        cmd += ["--ckpt_file", c["ckpt_file"]]
    if c.get("vocab_file"):
        cmd += ["--vocab_file", c["vocab_file"]]
    if c.get("nfe_step"):
        cmd += ["--nfe_step", str(c["nfe_step"])]
    run(cmd)
    return [out_dir / "f5.wav"]


ENGINES = {"gptsovits": engine_gptsovits, "chatterbox": engine_chatterbox, "f5tts": engine_f5tts}


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Generate narration in your cloned voice.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--text", help="text to speak")
    src.add_argument("--text-file", type=Path, help="UTF-8 text file to speak")
    ap.add_argument("--lang", help="pt or en (default: [general].default_lang)")
    ap.add_argument("--out", required=True, type=Path, help="output file (.wav, .mp3, .m4a)")
    ap.add_argument("--engine", choices=list(ENGINES), help="override the engine from [routing]")
    ap.add_argument("--no-rvc", action="store_true", help="skip the RVC pass even if enabled")
    ap.add_argument("--speed", type=float, default=None, help="tempo, e.g. 1.1 = 10%% faster")
    ap.add_argument("--config", type=Path, default=Path(os.environ.get("VOICE_CONFIG", DEFAULT_CONFIG)))
    ap.add_argument("--json", action="store_true", help="print a JSON result line on stdout")
    ap.add_argument("--keep-parts", type=Path, help="also save the raw chunks to this folder")
    args = ap.parse_args()

    cfg = portable.load_config(args.config)
    g = cfg.get("general", {})
    text = args.text if args.text is not None else (
        args.text_file.read_text(encoding="utf-8") if args.text_file else sys.stdin.read())
    if not text.strip():
        log("No text given.")
        return 2
    lang = (args.lang or g.get("default_lang", "pt")).lower()
    engine = args.engine or cfg.get("routing", {}).get(lang)
    if engine not in ENGINES:
        log(f"No engine configured for language '{lang}'. Set [routing].{lang} in {args.config}.")
        return 2
    speed = args.speed if args.speed is not None else g.get("speed", 1.0)

    if g.get("spell_numbers", True):
        text = spell_numbers(text, lang)
    max_chars = cfg.get(engine, {}).get("max_chars", 220)
    chunks = split_text(text, max_chars)
    log(f"{len(chunks)} chunk(s), lang={lang}, engine={engine}")

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tds:
        td = Path(tds)
        parts = ENGINES[engine](chunks, lang, cfg, td)
        use_rvc = cfg.get(engine, {}).get("rvc", False) and not args.no_rvc
        if use_rvc:
            log("RVC pass ...")
            conv = []
            for i, p in enumerate(parts):
                q = td / f"rvc_{i:04d}.wav"
                rvc_convert(p, q, cfg)
                conv.append(q)
            parts = conv
        if args.keep_parts:
            args.keep_parts.mkdir(parents=True, exist_ok=True)
            for p in parts:
                shutil.copy(p, args.keep_parts / p.name)
        concat_and_finish(parts, args.out, cfg, speed)

    dur = wav_duration(args.out)
    result = {"path": str(args.out.resolve()), "duration": round(dur, 2) if dur else None,
              "engine": engine + ("+rvc" if use_rvc else ""), "lang": lang, "seconds": round(time.time() - t0, 1)}
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        log(f"Done: {result['path']} ({result['engine']}, {result['seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
