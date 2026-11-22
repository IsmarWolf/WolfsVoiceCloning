#!/usr/bin/env python3
"""
Chatterbox worker for narrate.py. Run with the Python of the Chatterbox venv:
  <chatterbox-venv>\\Scripts\\python.exe chatterbox_worker.py job.json

job.json: {"lang": "pt", "ref_audio": "...wav", "device": "cuda", "seed": 0,
           "params": {"exaggeration": 0.5, "cfg_weight": 0.5},
           "chunks": [{"text": "...", "out": "...wav"}, ...]}
Loads the model once and writes one WAV per chunk.
"""
import inspect
import json
import sys

import numpy as np
import soundfile as sf
import torch


def main(job_path: str) -> int:
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)
    device = job.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("[chatterbox] CUDA not available, using CPU (slow)", file=sys.stderr)
        device = "cpu"
    if job.get("seed") is not None:
        torch.manual_seed(int(job["seed"]))

    lang = job["lang"]
    if lang == "en" and job.get("english_model", False):
        from chatterbox.tts import ChatterboxTTS

        model = ChatterboxTTS.from_pretrained(device=device)
        gen = lambda text, **kw: model.generate(text, **kw)
    else:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        try:
            model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
        except TypeError:  # older package without the t3_model argument
            model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        gen = lambda text, **kw: model.generate(text, language_id=lang, **kw)

    # only pass parameters this version of generate() accepts
    accepted = set(inspect.signature(model.generate).parameters)
    params = {k: v for k, v in (job.get("params") or {}).items() if k in accepted}
    if job.get("ref_audio") and "audio_prompt_path" in accepted:
        params["audio_prompt_path"] = job["ref_audio"]

    for i, ch in enumerate(job["chunks"]):
        with torch.inference_mode():
            wav = gen(ch["text"], **params)
        audio = wav.detach().float().cpu().numpy().squeeze()
        sf.write(ch["out"], np.clip(audio, -1, 1), model.sr, subtype="PCM_16")
        print(f"[chatterbox] {i+1}/{len(job['chunks'])}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
