#!/usr/bin/env python3
"""
Persistent F5-TTS worker (Brazilian Portuguese fine-tune) for studio.py. Run with the Python of venvs/f5:
  venvs\\f5\\Scripts\\python.exe engines\\f5_server.py <project root>

Same JSON-lines protocol as chatterbox_server.py. Extra request field: "ref_text" (what is said in the
reference clip; empty = F5 transcribes it itself, which downloads a Whisper model).
Weights: firstpixel/F5-TTS-pt-br (CC-BY-NC 4.0), vocabulary from SWivid/F5-TTS (same tokenizer).
F5-TTS code is MIT; the weights are non-commercial.
"""
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

REAL_OUT = sys.stdout
sys.stdout = sys.stderr
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else Path.cwd()
DIR = ROOT / "models" / "f5_ptbr"

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

state = {"f5": None}


def send(obj):
    REAL_OUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    REAL_OUT.flush()


def log(*a):
    print("[f5]", *a, file=sys.stderr, flush=True)


def ensure_files() -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    DIR.mkdir(parents=True, exist_ok=True)
    ckpt, vocab = DIR / "model_last.pt", DIR / "vocab.txt"
    if not ckpt.exists():
        log("downloading firstpixel/F5-TTS-pt-br ...")
        p = Path(hf_hub_download("firstpixel/F5-TTS-pt-br", "pt-br/model_last.pt", local_dir=str(DIR)))
        os.replace(p, ckpt)
    if not vocab.exists():
        last = None
        for name in ("F5TTS_Base/vocab.txt", "F5TTS_v1_Base/vocab.txt"):
            try:
                p = Path(hf_hub_download("SWivid/F5-TTS", name, local_dir=str(DIR)))
                os.replace(p, vocab)
                break
            except Exception as e:  # noqa: BLE001
                last = e
        else:
            raise RuntimeError(f"vocab.txt not found in SWivid/F5-TTS: {last}")
    return ckpt, vocab


def load() -> float:
    if state["f5"] is not None:
        return 0.0
    t0 = time.time()
    from f5_tts.api import F5TTS

    ckpt, vocab = ensure_files()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["f5"] = F5TTS(model="F5TTS_Base", ckpt_file=str(ckpt), vocab_file=str(vocab), device=device)
    log(f"loaded on {device} in {time.time() - t0:.1f}s")
    return time.time() - t0


def free():
    state["f5"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate(req):
    t0 = time.time()
    load()
    f5 = state["f5"]
    outdir = Path(req["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    seed = req.get("seed")
    seed = int(seed) if seed is not None and int(seed) >= 0 else None
    files = []
    for i, text in enumerate(req["chunks"]):
        # the pt-BR checkpoint expects lower case text (numbers are spelled out earlier by the studio)
        wav, sr, _ = f5.infer(ref_file=req["ref_audio"], ref_text=req.get("ref_text") or "", gen_text=text.lower(),
                              remove_silence=False, seed=seed, speed=1.0)
        audio = np.asarray(wav, dtype=np.float32).squeeze()
        p = outdir / f"f5_{i:04d}.wav"
        sf.write(str(p), np.clip(audio, -1, 1), int(sr), subtype="PCM_16")
        files.append(str(p))
        send({"event": "progress", "id": req["id"], "done": i + 1, "total": len(req["chunks"])})
    return {"files": files, "sr": int(sr), "seconds": round(time.time() - t0, 1)}


def main():
    if "--download" in sys.argv:
        ensure_files()
        REAL_OUT.write("downloaded\n")
        return
    send({"event": "ready", "cuda": torch.cuda.is_available()})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = {}
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "ping":
                send({"done": req.get("id"), "ok": True, "loaded": "f5ptbr" if state["f5"] else None, "cuda": torch.cuda.is_available()})
            elif cmd == "load":
                send({"done": req.get("id"), "ok": True, "seconds": round(load(), 1)})
            elif cmd == "generate":
                send({"done": req["id"], "ok": True, **generate(req)})
            elif cmd == "unload":
                free()
                send({"done": req.get("id"), "ok": True})
            elif cmd == "exit":
                break
            else:
                send({"done": req.get("id"), "ok": False, "error": f"unknown command {cmd}"})
        except Exception as e:
            traceback.print_exc()
            send({"done": req.get("id"), "ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
