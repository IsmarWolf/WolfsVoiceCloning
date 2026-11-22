#!/usr/bin/env python3
"""
Speaker-similarity scorer for studio.py. Run with the Python of the Chatterbox venv:
  python score_voice.py <project root> <job.json>

job.json: {"reference": ["real1.wav", ...], "candidates": {"name": "file.wav", ...}}
Prints one JSON line: {"scores": {"name": 0.83, ...}, "ref_spread": 0.04}

It uses Chatterbox's own speaker encoder (models/chatterbox_ptbr/ve.pt, or the base repo's) to turn every
clip into a voice fingerprint, then reports the cosine similarity between each candidate and the average
fingerprint of the real recordings. It measures timbre likeness only; it cannot judge naturalness or
artifacts, so the app shows the numbers next to audio players instead of trusting them blindly.
"""
import json
import sys
from pathlib import Path

REAL_OUT = sys.stdout
sys.stdout = sys.stderr

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> int:
    root = Path(sys.argv[1])
    job = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    from chatterbox.models.voice_encoder import VoiceEncoder

    ve_path = root / "models" / "chatterbox_ptbr" / "ve.pt"
    if not ve_path.exists():
        from huggingface_hub import hf_hub_download

        ve_path = Path(hf_hub_download("ResembleAI/chatterbox", "ve.pt"))
    ve = VoiceEncoder()
    ve.load_state_dict(torch.load(ve_path, map_location="cpu", weights_only=True))
    ve.eval()

    import librosa

    def embed(p: str) -> np.ndarray:
        wav, _ = librosa.load(p, sr=16000)
        e = ve.embeds_from_wavs([wav], sample_rate=16000)
        return np.asarray(e).reshape(-1)

    def unit(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-9)

    refs = [unit(embed(p)) for p in job["reference"]]
    center = unit(np.mean(refs, axis=0))
    spread = float(np.std([float(r @ center) for r in refs])) if len(refs) > 1 else 0.0
    scores = {name: round(float(unit(embed(p)) @ center), 4) for name, p in job["candidates"].items()}
    REAL_OUT.write(json.dumps({"scores": scores, "ref_spread": round(spread, 4)}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
