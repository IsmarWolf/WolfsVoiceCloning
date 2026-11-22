#!/usr/bin/env python3
"""
train_rvc.py - train an RVC v2 model from dataset\\wavs without clicking through the WebUI.

It runs the same steps as the WebUI's Train tab (preprocess -> F0 -> HuBERT features -> train -> index),
using the scripts that live in the RVC folder. Run it with the RVC venv's Python:

  RVC\\.venv\\Scripts\\python.exe train_rvc.py --exp ismar --epochs 100 --save-every 10

Every save point is written as assets\\weights\\<exp>_e<epoch>_s<step>.pth, so
compare_checkpoints.py can render the same test sentences from each one and you can pick
the earliest checkpoint that sounds like you (that is how we avoid overtraining).

Steps can be skipped (--skip preprocess f0 features) or run alone (--only train) to resume.
Re-running 'train' continues from the last checkpoint in logs\\<exp> (G_*.pth / D_*.pth).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["preprocess", "f0", "features", "train", "index"]


def run(cmd: list[str], cwd: Path, env: dict, log: Path | None = None) -> None:
    print("\n$", " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    t0 = time.time()
    if log:
        with open(log, "ab") as fh:
            r = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(cmd, cwd=cwd, env=env)
    print(f"  -> exit {r.returncode} in {time.time() - t0:.0f}s", flush=True)
    if r.returncode != 0:
        raise SystemExit(f"step failed (exit {r.returncode}); see {log or 'output above'}")


def build_filelist(rvc: Path, exp: str, sr: str, version: str, spk: int = 0) -> int:
    exp_dir = rvc / "logs" / exp
    gt = exp_dir / "0_gt_wavs"
    feat = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    f0 = exp_dir / "2a_f0"
    f0nsf = exp_dir / "2b-f0nsf"
    stems = lambda d: {p.name.split(".")[0] for p in d.iterdir()}
    names = stems(gt) & stems(feat) & stems(f0) & stems(f0nsf)
    if not names:
        raise SystemExit("No clips with all features. Did the preprocess / f0 / features steps finish?")
    fp = lambda p: str(p).replace("\\", "/")
    lines = [f"{fp(gt)}/{n}.wav|{fp(feat)}/{n}.npy|{fp(f0)}/{n}.wav.npy|{fp(f0nsf)}/{n}.wav.npy|{spk}" for n in sorted(names)]
    dim = 256 if version == "v1" else 768
    mute = rvc / "logs" / "mute"
    for _ in range(2):  # the same two "silence" lines the WebUI adds
        lines.append(f"{fp(mute)}/0_gt_wavs/mute{sr}.wav|{fp(mute)}/3_feature{dim}/mute.npy|"
                     f"{fp(mute)}/2a_f0/mute.wav.npy|{fp(mute)}/2b-f0nsf/mute.wav.npy|{spk}")
    random.shuffle(lines)
    (exp_dir / "filelist.txt").write_text("\n".join(lines), encoding="utf-8")
    return len(names)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train an RVC v2 model (WebUI steps, headless).")
    ap.add_argument("--rvc", type=Path, default=HERE / "RVC", help="RVC folder (git clone)")
    ap.add_argument("--wavs", type=Path, default=HERE / "dataset" / "wavs")
    ap.add_argument("--exp", default="ismar", help="experiment / model name")
    ap.add_argument("--sr", default="40k", choices=["32k", "40k", "48k"])
    ap.add_argument("--version", default="v2", choices=["v1", "v2"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--f0", default="rmvpe", choices=["rmvpe", "pm"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--half", default="True", choices=["True", "False"])
    ap.add_argument("--only", nargs="+", choices=STEPS)
    ap.add_argument("--skip", nargs="+", choices=STEPS, default=[])
    args = ap.parse_args()

    rvc = args.rvc.resolve()
    py = str(rvc / ".venv" / "Scripts" / "python.exe") if os.name == "nt" else str(rvc / ".venv" / "bin" / "python")
    if not Path(py).exists():
        py = sys.executable
    exp_dir = rvc / "logs" / args.exp
    exp_dir.mkdir(parents=True, exist_ok=True)
    log = exp_dir / "train_driver.log"
    env = {**os.environ, "outside_index_root": str(rvc / "assets" / "indices"), "PYTHONIOENCODING": "utf-8",
           "PYTHONUTF8": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
           "PYTHONPATH": str(rvc) + os.pathsep + os.environ.get("PYTHONPATH", ""),
           # keep train/ off sys.path, or "from train import utils" finds train/train.py instead of the package
           "PYTHONSAFEPATH": "1", "RVC_CUDA_GRAPH": "0"}
    (rvc / "assets" / "indices").mkdir(parents=True, exist_ok=True)
    (rvc / "assets" / "weights").mkdir(parents=True, exist_ok=True)
    sr_num = {"32k": 32000, "40k": 40000, "48k": 48000}[args.sr]
    fwd = lambda p: str(p).replace("\\", "/")
    todo = [s for s in (args.only or STEPS) if s not in args.skip]
    print(f"RVC: {rvc}\nexperiment: {args.exp}  sr={args.sr} {args.version}  epochs={args.epochs} save every {args.save_every}  batch={args.batch}\nsteps: {todo}")

    if "preprocess" in todo:
        n = len(list(args.wavs.glob("*.wav")))
        print(f"\n[1/5] preprocess ({n} wav files)")
        run([py, "train/preprocess.py", fwd(args.wavs.resolve()), str(sr_num), str(args.workers),
             fwd(exp_dir), "False", "3.7"], rvc, env, log)

    if "f0" in todo:
        print("\n[2/5] pitch (F0) extraction with", args.f0)
        (exp_dir / "extract_f0_feature.log").write_text("", encoding="utf-8")
        if args.f0 == "rmvpe":
            run([py, "train/dataset/extract_f0.py", "cuda", "1", "0", args.gpu, fwd(exp_dir), args.half], rvc, env, log)
        else:
            run([py, "train/dataset/extract_f0.py", "cpu", fwd(exp_dir), str(args.workers), "pm"], rvc, env, log)

    if "features" in todo:
        print("\n[3/5] HuBERT features")
        run([py, "train/dataset/extract_hubert_feature.py", f"cuda:{args.gpu}", "1", "0", args.gpu,
             fwd(exp_dir), args.version, args.half], rvc, env, log)

    if "train" in todo:
        print("\n[4/5] training")
        n = build_filelist(rvc, args.exp, args.sr, args.version)
        print(f"  filelist: {n} clips + mute")
        cfg_src = rvc / "configs" / ("v1" if args.version == "v1" or args.sr == "40k" else "v2") / f"{args.sr}.json"
        cfg = exp_dir / "config.json"
        if not cfg.exists():
            data = json.loads(cfg_src.read_text(encoding="utf-8"))
            cfg.write_text(json.dumps(data, ensure_ascii=False, indent=4, sort_keys=True) + "\n", encoding="utf-8")
        pre = "pretrained_v2" if args.version == "v2" else "pretrained"
        pg, pd = f"assets/{pre}/f0G{args.sr}.pth", f"assets/{pre}/f0D{args.sr}.pth"
        for p in (pg, pd):
            if not (rvc / p).exists():
                raise SystemExit(f"missing pretrained model {p}")
        # -l 0: keep every checkpoint, -c 0: no GPU cache, -sw 1: write a small model at every save point
        run([py, "train/train.py", "-e", args.exp, "-sr", args.sr, "-f0", "1", "-bs", str(args.batch),
             "-g", args.gpu, "-te", str(args.epochs), "-se", str(args.save_every),
             "-pg", pg, "-pd", pd, "-l", "0", "-c", "0", "-sw", "1", "-v", args.version], rvc, env, log)

    if "index" in todo:
        print("\n[5/5] index (retrieval)")
        run([py, "train/train_index.py", args.exp, args.version, str(rvc / "assets" / "indices"),
             str(min(os.cpu_count() or 4, 8)), "single"], rvc, env, log)

    w = sorted((rvc / "assets" / "weights").glob(f"{args.exp}*.pth"))
    print("\nDone. Saved models:")
    for p in w:
        print("  ", p.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
