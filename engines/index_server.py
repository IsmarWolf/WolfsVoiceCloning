#!/usr/bin/env python3
"""
Persistent IndexTTS 2.5 worker for studio.py. Run with the Python of venvs/index:
  venvs\\index\\Scripts\\python.exe engines\\index_server.py <project root>

UNTESTED: written from the project's README (index-tts/index-tts, bilibili Model Use License).
IndexTTS 2.5 speaks Chinese, English, Japanese, Spanish and Arabic, NOT Portuguese: the studio only offers it for English.
Same JSON-lines protocol as chatterbox_server.py.
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
DIR = ROOT / "models" / "indextts25"
LANGS = {"en": "EN", "zh": "ZH", "ja": "JA", "es": "ES", "ar": "AR"}

import torch  # noqa: E402

state = {"tts": None}


def send(obj):
    REAL_OUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    REAL_OUT.flush()


def log(*a):
    print("[indextts]", *a, file=sys.stderr, flush=True)


def load() -> float:
    if state["tts"] is not None:
        return 0.0
    t0 = time.time()
    from indextts.infer_v2_5 import IndexTTS2

    state["tts"] = IndexTTS2(cfg_path=str(DIR / "config.yaml"), model_dir=str(DIR), use_bf16=torch.cuda.is_available())
    log(f"loaded in {time.time() - t0:.1f}s")
    return time.time() - t0


def free():
    state["tts"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate(req):
    t0 = time.time()
    load()
    tts = state["tts"]
    lang = LANGS.get(req.get("lang", "en"))
    if not lang:
        raise ValueError("IndexTTS 2.5 does not support this language")
    seed = req.get("seed")
    if seed is not None and int(seed) >= 0:
        torch.manual_seed(int(seed))
    outdir = Path(req["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    files = []
    for i, text in enumerate(req["chunks"]):
        p = outdir / f"ix_{i:04d}.wav"
        tts.infer(spk_audio_prompt=req["ref_audio"], text=text, lang=lang, output_path=str(p), verbose=False)
        files.append(str(p))
        send({"event": "progress", "id": req["id"], "done": i + 1, "total": len(req["chunks"])})
    return {"files": files, "sr": 22050, "seconds": round(time.time() - t0, 1)}


def main():
    if "--download" in sys.argv:
        from huggingface_hub import snapshot_download

        snapshot_download("IndexTeam/IndexTTS-2.5", local_dir=str(DIR))
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
                send({"done": req.get("id"), "ok": True, "loaded": "index25" if state["tts"] else None, "cuda": torch.cuda.is_available()})
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
