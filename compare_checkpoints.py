#!/usr/bin/env python3
"""
compare_checkpoints.py - render the same test sentences from every saved checkpoint,
so you can pick the EARLIEST one that sounds like you (later ones are often overtrained).

  python compare_checkpoints.py gptsovits --exp ismar             # English: every GPT x SoVITS pair
  python compare_checkpoints.py rvc --exp ismar                   # Portuguese: every RVC save point
  python compare_checkpoints.py rvc --exp ismar --base-dir C:\\base   # ... using base voices you rendered yourself
  python compare_checkpoints.py f5tts --ckpt-dir C:\\...\\ckpts\\ismar_pt
  python compare_checkpoints.py rvc --exp ismar --every 2         # only every 2nd checkpoint

Output: out\\checkpoints\\<engine>\\<checkpoint>.wav  (all test sentences in one file per checkpoint)
        plus a subfolder per checkpoint with one file per sentence.
Listen in order, oldest first. Stop at the first one that sounds natural and like you.
Signs of overtraining: metallic or robotic tone, skipped or repeated words, mumbling,
breathing artefacts, or every sentence sounding identical in rhythm.
"""
from __future__ import annotations

import argparse
import copy
import re
import shutil
import sys
import tempfile
from pathlib import Path

import tomllib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import narrate  # noqa: E402


def epoch_of(p: Path) -> int:
    """Pull the epoch number out of names like ismar-e15.ckpt, ismar_e8_s400.pth, model_1200.pt."""
    m = re.search(r"[_-]e(\d+)", p.stem) or re.search(r"(\d+)(?!.*\d)", p.stem)
    return int(m.group(1)) if m else 10**9


def thin(items: list[Path], every: int) -> list[Path]:
    items = sorted(items, key=epoch_of)
    if every <= 1 or len(items) <= 2:
        return items
    kept = items[::every]
    if items[-1] not in kept:
        kept.append(items[-1])
    return kept


def load_sentences(lang: str, path: Path | None) -> list[str]:
    f = path or HERE / "test_sentences" / f"{lang}.txt"
    lines = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [narrate.spell_numbers(l, lang) for l in lines]


def save(parts: list[Path], label: str, out_root: Path, cfg: dict) -> None:
    sub = out_root / label
    sub.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(parts):
        shutil.copy(p, sub / f"{i+1:02d}.wav")
    narrate.concat_and_finish(parts, out_root / f"{label}.wav", cfg, speed=1.0)
    narrate.log(f"  -> {out_root / (label + '.wav')}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Render test sentences from every checkpoint.")
    ap.add_argument("engine", choices=["gptsovits", "rvc", "f5tts"])
    ap.add_argument("--exp", default="ismar", help="experiment name used in training")
    ap.add_argument("--lang", help="test sentence language (default: en for gptsovits, pt otherwise)")
    ap.add_argument("--sentences", type=Path, help="custom test sentence file (one per line)")
    ap.add_argument("--ckpt-dir", type=Path, help="f5tts: folder with model_*.pt checkpoints")
    ap.add_argument("--base-dir", type=Path,
                    help="rvc: folder with one pre-rendered base .wav per test sentence (sorted by name), "
                         "instead of rendering them with the engine from [routing]")
    ap.add_argument("--every", type=int, default=1, help="only every Nth checkpoint (the last is always kept)")
    ap.add_argument("--config", type=Path, default=narrate.DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=HERE / "out" / "checkpoints")
    args = ap.parse_args()

    cfg = tomllib.loads(args.config.read_text(encoding="utf-8"))
    lang = args.lang or ("en" if args.engine == "gptsovits" else "pt")
    sentences = load_sentences(lang, args.sentences)
    out_root = args.out / args.engine
    out_root.mkdir(parents=True, exist_ok=True)
    narrate.log(f"{len(sentences)} test sentences ({lang}) -> {out_root}")

    if args.engine == "gptsovits":
        root = Path(cfg["gptsovits"]["root"])
        gpts = thin(list(root.glob(f"GPT_weights*/{args.exp}-e*.ckpt")), args.every)
        sovits = thin(list(root.glob(f"SoVITS_weights*/{args.exp}_e*.pth")), args.every)
        if not gpts or not sovits:
            narrate.log(f"No checkpoints for '{args.exp}' under {root} (GPT: {len(gpts)}, SoVITS: {len(sovits)})")
            return 1
        narrate.log(f"{len(gpts)} GPT x {len(sovits)} SoVITS checkpoints = {len(gpts)*len(sovits)} renders")
        for s in sovits:
            for g in gpts:
                c = copy.deepcopy(cfg)
                c["gptsovits"].update(gpt_weights=str(g), sovits_weights=str(s), seed=42)
                label = f"sovits-e{epoch_of(s):03d}__gpt-e{epoch_of(g):03d}"
                narrate.log(f"[{label}]")
                with tempfile.TemporaryDirectory() as td:
                    save(narrate.engine_gptsovits(sentences, lang, c, Path(td)), label, out_root, c)

    elif args.engine == "rvc":
        r = cfg["rvc"]
        models = thin([p for p in (Path(r["cwd"]) / "assets" / "weights").glob(f"{args.exp}_e*.pth")], args.every)
        if not models:
            narrate.log(f"No RVC models named {args.exp}*.pth in {Path(r['cwd']) / 'assets' / 'weights'}")
            return 1
        base_engine = cfg["routing"].get(lang, "chatterbox")
        base_dir = args.base_dir or (out_root / "_base")
        base_dir.mkdir(exist_ok=True)
        base = sorted(base_dir.glob("*.wav"))
        if args.base_dir and len(base) != len(sentences):
            narrate.log(f"--base-dir has {len(base)} wavs but there are {len(sentences)} test sentences")
            return 1
        if len(base) != len(sentences):  # render the base voice once, reuse for every RVC model
            narrate.log(f"Rendering base audio with {base_engine} ...")
            with tempfile.TemporaryDirectory() as td:
                parts = narrate.ENGINES[base_engine](sentences, lang, cfg, Path(td))
                for f in base_dir.glob("*.wav"):
                    f.unlink()
                for i, p in enumerate(parts):
                    shutil.copy(p, base_dir / f"{i+1:02d}.wav")
            base = sorted(base_dir.glob("*.wav"))
        narrate.log(f"{len(models)} RVC checkpoints")
        for m in models:
            c = copy.deepcopy(cfg)
            c["rvc"]["model_name"] = m.name
            label = f"rvc-e{epoch_of(m):04d}"
            narrate.log(f"[{label}] {m.name}")
            with tempfile.TemporaryDirectory() as td:
                conv = []
                for i, b in enumerate(base):
                    q = Path(td) / f"{i+1:02d}.wav"
                    narrate.rvc_convert(b, q, c)
                    conv.append(q)
                save(conv, label, out_root, c)

    else:  # f5tts
        ckdir = args.ckpt_dir or Path(cfg["f5tts"]["ckpt_file"]).parent
        ckpts = thin([p for p in ckdir.glob("model_*.pt") if p.stem != "model_last"], args.every)
        if not ckpts:
            narrate.log(f"No model_*.pt checkpoints in {ckdir}")
            return 1
        narrate.log(f"{len(ckpts)} F5-TTS checkpoints")
        for ck in ckpts:
            c = copy.deepcopy(cfg)
            c["f5tts"]["ckpt_file"] = str(ck)
            label = f"f5-{ck.stem}"
            narrate.log(f"[{label}]")
            with tempfile.TemporaryDirectory() as td:
                parts = []
                for i, s in enumerate(sentences):  # one call per sentence keeps them separate
                    sd = Path(td) / f"s{i}"
                    sd.mkdir()
                    parts += narrate.engine_f5tts([s], lang, c, sd)
                save(parts, label, out_root, c)

    narrate.log(f"\nDone. Listen to {out_root} in name order and keep the earliest good one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
