#!/usr/bin/env python3
"""
Persistent Chatterbox worker for studio.py. Run with the Python of the Chatterbox venv:
  venvs\\chatterbox\\Scripts\\python.exe engines\\chatterbox_server.py <project root>

Protocol: one JSON object per line on stdin, one JSON object per line on stdout
(everything else the libraries print is sent to stderr).

  {"cmd": "ping"}                                   -> {"done": null, "ok": true, "loaded": "ptbr"|"v3"|null, "cuda": true}
  {"cmd": "load", "id": 1, "model": "ptbr"}          -> {"done": 1, "ok": true, "seconds": 12.3}
  {"cmd": "generate", "id": 2, "model": "ptbr", "lang": "pt", "chunks": ["..."],
   "ref_audio": "...wav", "outdir": "...", "exaggeration": 0.5, "cfg_weight": 0.5,
   "temperature": 0.8, "seed": 123}
       -> {"event": "progress", "id": 2, "done": 1, "total": 5} ... then
          {"done": 2, "ok": true, "files": [...], "sr": 24000, "seconds": 20.1}
  {"cmd": "unload"}                                  -> frees the GPU memory

Models
  v3    ResembleAI/chatterbox, multilingual v3 (23 languages, downloaded on first use)
  ptbr  ResembleAI/Chatterbox-Multilingual-pt-br, a dedicated Brazilian Portuguese fine-tune (MIT).
        Its files are collected in <root>\\models\\chatterbox_ptbr on first use (about 3.3 GB).
"""
import gc
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path

REAL_OUT = sys.stdout
sys.stdout = sys.stderr  # keep stdout for the protocol only
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
PTBR_REPO = "ResembleAI/Chatterbox-Multilingual-pt-br"
BASE_REPO = "ResembleAI/chatterbox"

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

state = {"model": None, "key": None}


def send(obj: dict) -> None:
    REAL_OUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    REAL_OUT.flush()


def log(*a) -> None:
    print("[chatterbox]", *a, file=sys.stderr, flush=True)


def ensure_ptbr() -> Path:
    """Collect the files the loader needs: pt-BR T3 + V3 S3Gen from the pt-BR repo, voice encoder + conds from the base repo."""
    from huggingface_hub import hf_hub_download

    d = ROOT / "models" / "chatterbox_ptbr"
    d.mkdir(parents=True, exist_ok=True)
    want = [  # (repo, file in repo, name the loader expects)
        (PTBR_REPO, "t3_pt_br.safetensors", "t3_pt_br.safetensors"),
        (PTBR_REPO, "s3gen_v3.pt", "s3gen.pt"),
        (PTBR_REPO, "grapheme_mtl_merged_expanded_v1.json", "grapheme_mtl_merged_expanded_v1.json"),
        (BASE_REPO, "ve.pt", "ve.pt"),
        (BASE_REPO, "conds.pt", "conds.pt"),
    ]
    for repo, name, target in want:
        if (d / target).exists():
            continue
        log(f"downloading {repo}/{name} ...")
        p = Path(hf_hub_download(repo, name, local_dir=str(d)))
        if p.name != target:
            os.replace(p, d / target)
    return d


def load_ptbr(ckpt: Path, device: str):
    """The pt-BR repo ships the V3 S3Gen without the tokenizer's mel/window buffers. Those are
    rebuilt at construction time, so tolerate exactly those keys (and nothing else) as missing."""
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    original = S3Gen.load_state_dict

    def lenient(self, state_dict, strict=True, *a, **kw):
        res = original(self, state_dict, False, *a, **kw)
        bad = [k for k in res.missing_keys if not k.endswith(("_mel_filters", ".window"))] + list(res.unexpected_keys)
        if bad:
            raise RuntimeError(f"s3gen does not match the code: {bad[:6]}")
        return res

    S3Gen.load_state_dict = lenient
    try:
        return ChatterboxMultilingualTTS.from_local(ckpt, device, t3_model="t3_pt_br.safetensors")
    finally:
        S3Gen.load_state_dict = original


def free() -> None:
    state["model"] = None
    state["key"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load(key: str) -> float:
    if state["key"] == key and state["model"] is not None:
        return 0.0
    t0 = time.time()
    free()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    if key == "ptbr":
        model = load_ptbr(ensure_ptbr(), device)
    elif key == "v3":
        model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
    else:
        raise ValueError(f"unknown model '{key}'")
    state["model"], state["key"] = model, key
    log(f"model '{key}' loaded on {device} in {time.time() - t0:.1f}s")
    return time.time() - t0


def generate(req: dict) -> dict:
    t0 = time.time()
    load(req["model"])
    model = state["model"]
    accepted = set(inspect.signature(model.generate).parameters)
    params = {}
    for k in ("exaggeration", "cfg_weight", "temperature", "repetition_penalty", "min_p", "top_p"):
        if req.get(k) is not None and k in accepted:
            params[k] = float(req[k])
    if req.get("ref_audio") and "audio_prompt_path" in accepted:
        params["audio_prompt_path"] = req["ref_audio"]
    seed = req.get("seed")
    if seed is not None and int(seed) >= 0:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    outdir = Path(req["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    chunks = req["chunks"]
    files = []
    for i, text in enumerate(chunks):
        with torch.inference_mode():
            wav = model.generate(text, language_id=req.get("lang", "pt"), **params)
        audio = wav.detach().float().cpu().numpy().squeeze()
        p = outdir / f"cb_{i:04d}.wav"
        sf.write(str(p), np.clip(audio, -1, 1), model.sr, subtype="PCM_16")
        files.append(str(p))
        send({"event": "progress", "id": req["id"], "done": i + 1, "total": len(chunks)})
    return {"files": files, "sr": int(model.sr), "seconds": round(time.time() - t0, 1)}


def main() -> None:
    if "--download" in sys.argv:  # used by the app's setup: fetch the pt-BR files and exit
        ensure_ptbr()
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
                send({"done": req.get("id"), "ok": True, "loaded": state["key"], "cuda": torch.cuda.is_available()})
            elif cmd == "load":
                send({"done": req.get("id"), "ok": True, "seconds": round(load(req["model"]), 1)})
            elif cmd == "generate":
                send({"done": req["id"], "ok": True, **generate(req)})
            elif cmd == "unload":
                free()
                send({"done": req.get("id"), "ok": True})
            elif cmd == "exit":
                break
            else:
                send({"done": req.get("id"), "ok": False, "error": f"unknown command {cmd}"})
        except Exception as e:  # keep the worker alive
            traceback.print_exc()
            send({"done": req.get("id"), "ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
