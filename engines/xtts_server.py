#!/usr/bin/env python3
"""
Persistent XTTS-v2 worker for studio.py. Run with the Python of venvs/xtts:
  venvs\\xtts\\Scripts\\python.exe engines\\xtts_server.py <project root>

Same JSON-lines protocol as chatterbox_server.py (ping / load / generate / unload / exit).
XTTS-v2 is licensed under the Coqui Public Model License (non-commercial). The studio only starts this
worker after you accepted that licence in the "Voz base" menu (it sets COQUI_TOS_AGREED=1).
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
os.environ.setdefault("COQUI_TOS_AGREED", "1" if "--accept" in sys.argv else os.environ.get("COQUI_TOS_AGREED", "0"))
os.environ.setdefault("TTS_HOME", str(ROOT / "models" / "xtts"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
state = {"tts": None}


def send(obj):
    REAL_OUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    REAL_OUT.flush()


def log(*a):
    print("[xtts]", *a, file=sys.stderr, flush=True)


def load() -> float:
    if state["tts"] is not None:
        return 0.0
    t0 = time.time()
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["tts"] = TTS(NAME).to(device)
    log(f"loaded on {device} in {time.time() - t0:.1f}s")
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
    seed = req.get("seed")
    if seed is not None and int(seed) >= 0:
        torch.manual_seed(int(seed))
    outdir = Path(req["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    lang = {"pt": "pt", "en": "en"}.get(req.get("lang", "pt"), req.get("lang", "pt"))
    files = []
    for i, text in enumerate(req["chunks"]):
        kw = {}
        if req.get("temperature") is not None:
            kw["temperature"] = float(req["temperature"])
        wav = tts.tts(text=text, speaker_wav=req["ref_audio"], language=lang, **kw)
        audio = np.asarray(wav, dtype=np.float32)
        p = outdir / f"xt_{i:04d}.wav"
        sf.write(str(p), np.clip(audio, -1, 1), 24000, subtype="PCM_16")
        files.append(str(p))
        send({"event": "progress", "id": req["id"], "done": i + 1, "total": len(req["chunks"])})
    return {"files": files, "sr": 24000, "seconds": round(time.time() - t0, 1)}


def main():
    if "--download" in sys.argv:  # used by the app's setup: fetch the model files and exit
        load()
        (ROOT / "models" / "xtts" / "downloaded.ok").write_text("ok", encoding="utf-8")  # the files live in the Hugging Face cache
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
                send({"done": req.get("id"), "ok": True, "loaded": "xtts" if state["tts"] else None, "cuda": torch.cuda.is_available()})
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
