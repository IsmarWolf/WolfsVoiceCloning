#!/usr/bin/env python3
"""
Persistent RVC worker for studio.py. Run with the Python of the RVC venv:
  RVC\\.venv\\Scripts\\python.exe engines\\rvc_server.py <RVC folder>

Same line-per-JSON protocol as chatterbox_server.py:
  {"cmd": "convert", "id": 1, "model": "ismar_e100_s4200.pth", "input": "in.wav", "output": "out.wav",
   "index": "...index" | "", "index_rate": 0.5, "protect": 0.33, "pitch": 0, "f0_method": "rmvpe",
   "rms_mix_rate": 1.0}
     -> {"done": 1, "ok": true, "seconds": 3.2}
  {"cmd": "load", "id": 2, "model": "ismar_e100_s4200.pth"}  -> loads the model ahead of time
  {"cmd": "ping"} -> {"done": null, "ok": true, "loaded": "<model>"|null}
The voice model stays loaded between requests and is only swapped when "model" changes.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

REAL_OUT = sys.stdout
sys.stdout = sys.stderr

RVC_ROOT = Path(sys.argv[1]).resolve()
os.chdir(RVC_ROOT)
sys.path.insert(0, str(RVC_ROOT))

from infer import cli as rvccli  # noqa: E402  (also sets the weight/index environment variables)

state = {"vc": None, "model": None, "config": None}


def send(obj: dict) -> None:
    REAL_OUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    REAL_OUT.flush()


def ensure_model(name: str) -> Path:
    """Load the voice model once and keep it in memory; swap only when the name changes."""
    model_path = rvccli.resolve_model(name)
    os.environ["weight_root"] = str(model_path.parent)
    if state["vc"] is None:
        from infer.vc.modules import VC

        state["config"] = rvccli.create_config()
        state["vc"] = VC(state["config"])
    if state["model"] != str(model_path):
        state["vc"].get_vc(model_path.name)
        state["model"] = str(model_path)
    return model_path


def convert(req: dict) -> dict:
    t0 = time.time()
    model_path = ensure_model(req["model"])
    vc = state["vc"]
    speaker_count, speakers = rvccli.load_model_metadata(model_path)
    speaker_id = rvccli.select_speaker(speaker_count, speakers, None)
    index_rate = float(req.get("index_rate", 0.5))
    index_path = rvccli.resolve_index(req.get("index") or None, model_path.name, speaker_id, index_rate)
    status, result = vc.vc_single(
        speaker_id, str(req["input"]), int(req.get("pitch", 0)), req.get("f0_method", "rmvpe"),
        index_path, index_rate, 0, float(req.get("rms_mix_rate", 1.0)), float(req.get("protect", 0.33)),
    )
    if not result or result[0] is None or result[1] is None:
        raise RuntimeError(f"RVC failed: {status}")
    rvccli.write_audio(Path(req["output"]), result[1], result[0], "wav")
    return {"seconds": round(time.time() - t0, 1)}


def main() -> None:
    send({"event": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = {}
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "ping":
                send({"done": req.get("id"), "ok": True, "loaded": Path(state["model"]).name if state["model"] else None})
            elif cmd == "load":
                t0 = time.time()
                ensure_model(req["model"])
                send({"done": req["id"], "ok": True, "seconds": round(time.time() - t0, 1)})
            elif cmd == "convert":
                send({"done": req["id"], "ok": True, **convert(req)})
            elif cmd == "exit":
                break
            else:
                send({"done": req.get("id"), "ok": False, "error": f"unknown command {cmd}"})
        except Exception as e:
            traceback.print_exc()
            send({"done": req.get("id"), "ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
