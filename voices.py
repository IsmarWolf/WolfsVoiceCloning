#!/usr/bin/env python3
"""
voices.py - the "clone my voice" pipeline behind the studio's second tab. Standard library only.

  voices/<name>/raw/        your recordings, converted to WAV (any audio/video file works)
  voices/<name>/dataset/    clips + transcripts made by prepare_dataset.py (Whisper)
  voices/<name>/picks/      the test sentence rendered through every RVC checkpoint
  voices/<name>/voice.json  the result: reference clip, chosen checkpoint, index, the ranking

Pipeline: prepare dataset -> choose a reference clip -> train RVC (optional) -> render every checkpoint
and rank it by speaker similarity -> make it the active voice. Every stage is skipped if its output exists,
so a cancelled or failed run continues where it stopped.
"""
from __future__ import annotations

import csv
import json
import random
import re
import shutil
import subprocess
import threading
import time
import wave
from pathlib import Path

import portable as P
import setup_manager as SM

ROOT = P.ROOT


# --------------------------------------------------------------------------- files
def wav_seconds(p: Path) -> float:
    try:
        with wave.open(str(p)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


def raw_dir(name: str) -> Path:
    return P.voice_dir(name) / "raw"


def list_recordings(name: str) -> list[dict]:
    d = raw_dir(name)
    return [{"file": f.name, "seconds": round(wav_seconds(f), 1)} for f in sorted(d.glob("*.wav"))] if d.exists() else []


def create_voice(display: str) -> dict:
    name = P.slug_name(display)
    v = P.read_voice(name)
    if v:
        return v
    v = {"name": name, "display": display.strip()[:40] or name, "created": time.time(), "status": "new", "dataset": f"voices/{name}/dataset",
         "ref": None, "rvc": None, "ranking": [], "base_score": None, "note": ""}
    P.write_voice(v)
    raw_dir(name).mkdir(parents=True, exist_ok=True)
    return v


def add_recording(name: str, original: str, src: Path) -> dict:
    """Convert an uploaded audio/video file to mono 44.1 kHz WAV inside voices/<name>/raw."""
    if not P.read_voice(name):
        raise ValueError("voz não encontrada")
    d = raw_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^\w\-]+", "_", Path(original).stem)[:40] or "gravacao"
    dst = d / f"{stem}.wav"
    n = 2
    while dst.exists():
        dst = d / f"{stem}_{n}.wav"
        n += 1
    r = subprocess.run([P.ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "44100",
                        "-c:a", "pcm_s16le", str(dst)], capture_output=True, text=True)
    if r.returncode != 0 or not dst.exists():
        dst.unlink(missing_ok=True)
        raise ValueError("Não consegui ler esse arquivo como áudio.")
    return {"file": dst.name, "seconds": round(wav_seconds(dst), 1)}


def remove_recording(name: str, file: str) -> None:
    f = raw_dir(name) / file
    if f.parent == raw_dir(name) and f.is_file():
        f.unlink()


def dataset_rows(v: dict) -> list[dict]:
    meta = P.voice_dataset(v) / "metadata.csv"
    if not meta.exists():
        return []
    with open(meta, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="|"))


def pick_ref(v: dict) -> str | None:
    """A clean 5-12 s clip with a confident transcript, close to 8 s: what Chatterbox clones the voice from."""
    best, best_key = None, None
    for r in dataset_rows(v):
        try:
            dur, prob = float(r["duration"]), float(r.get("prob") or 0)
        except (KeyError, ValueError):
            continue
        if not 5.0 <= dur <= 12.0:
            continue
        key = (-round(prob, 1), abs(dur - 8.0))
        if best_key is None or key < best_key:
            best, best_key = r["file"], key
    return best


def weights_of(exp: str) -> list[Path]:
    d = P.RVC / "assets" / "weights"
    ws = [f for f in d.glob(f"{exp}_e*.pth")] if d.exists() else []
    return sorted(ws, key=lambda f: (int(re.search(r"_e(\d+)", f.name).group(1)), f.name))


def index_of(exp: str) -> str | None:
    d = P.RVC / "assets" / "indices"
    ix = sorted(d.glob(f"{exp}_added*.index"), key=lambda f: f.stat().st_mtime) if d.exists() else []
    return ix[-1].name if ix else None


def discover_legacy() -> None:
    """An install made before the app existed (dataset/ + RVC weights): register it as a voice."""
    if P.list_voices() or not (ROOT / "dataset" / "metadata.csv").exists():
        return
    ws = sorted((P.RVC / "assets" / "weights").glob("*_e*.pth")) if (P.RVC / "assets" / "weights").exists() else []
    exp = re.sub(r"_e\d+.*$", "", ws[-1].name) if ws else "voz"
    v = {"name": exp, "display": exp, "created": time.time(), "status": "ready", "dataset": "dataset", "ref": None,
         "rvc": None, "ranking": [], "base_score": None, "note": "Importada de uma instalação anterior."}
    v["ref"] = pick_ref(v)
    ws = weights_of(exp)
    if ws:
        v["rvc"] = {"model": ws[-1].name, "index": index_of(exp)}
    P.write_voice(v)
    P.write_settings(active_voice=exp)


def summary(v: dict) -> dict:
    rows = dataset_rows(v)
    return {"name": v["name"], "display": v.get("display", v["name"]), "status": v.get("status"), "ref": v.get("ref"),
            "rvc": v.get("rvc"), "ranking": v.get("ranking", []), "base_score": v.get("base_score"), "note": v.get("note", ""),
            "recordings": list_recordings(v["name"]), "dataset_minutes": round(sum(float(r.get("duration") or 0) for r in rows) / 60, 1),
            "active": P.read_settings().get("active_voice") == v["name"], "exp": v["name"]}


def activate(name: str) -> None:
    if not P.read_voice(name):
        raise ValueError("voz não encontrada")
    P.write_settings(active_voice=name)


def set_checkpoint(name: str, ckpt: str) -> None:
    v = P.read_voice(name)
    if not v or not (P.RVC / "assets" / "weights" / ckpt).is_file() or not ckpt.startswith(name + "_e"):
        raise ValueError("checkpoint inválido")
    v["rvc"] = {"model": ckpt, "index": index_of(name)}
    P.write_voice(v)


def set_rvc_off(name: str) -> None:
    v = P.read_voice(name)
    if v:
        v["rvc"] = None
        P.write_voice(v)


# --------------------------------------------------------------------------- the pipeline
class Ctx:
    """What the pipeline needs from studio.py: the two workers, the job and a stage reporter."""

    def __init__(self, job: dict, cb, rv, set_stage, narrate):
        self.job, self.cb, self.rv, self.set_stage, self.narrate = job, cb, rv, set_stage, narrate
        self.ctl = job.setdefault("ctl", {"cancel": False, "proc": None})

    def check(self) -> None:
        if self.ctl.get("cancel"):
            raise SM.Cancelled()

    def sh(self, cmd: list[str], log: Path, stage: str, cwd: Path | None = None, env: dict | None = None, parse=None) -> None:
        def on_line(t: str):
            self.job["detail"] = t[:160]
            if parse:
                parse(t)
        self.set_stage(self.job, stage)
        rc = SM.run_logged(cmd, log, self.ctl, cwd=cwd, env=env, on_line=on_line)
        if rc != 0:
            raise RuntimeError(f"Falhou (código {rc}). Veja {log.relative_to(ROOT)}")


def run_clone(ctx: Ctx, name: str, opts: dict) -> None:
    v = P.read_voice(name)
    if not v:
        raise ValueError("voz não encontrada")
    st = SM.SETUP.status()
    prof_name = st["profile"] or st["hardware"]["recommended"]
    prof = SM.PROFILES[prof_name]
    use_rvc = bool(opts.get("rvc"))
    need = {s["id"] for s in st["steps"] if not (s["needs_rvc"] and not use_rvc)}
    missing = [s["label"] for s in st["steps"] if s["id"] in need and s["state"] != "done"]
    if missing:
        raise RuntimeError("Falta instalar: " + ", ".join(missing) + ". Rode a instalação no topo da aba.")
    recs = list_recordings(name)
    total = sum(r["seconds"] for r in recs)
    if total < 60:
        raise ValueError("Envie pelo menos 1 minuto de gravação (o ideal são 20 a 30 minutos).")

    vdir = P.voice_dir(name)
    ds = P.voice_dataset(v)
    log = vdir / "logs"
    prep_py = P.venv_python(P.VENVS / "prep")
    rvc_py = P.venv_python(P.RVC / ".venv")
    cb_py = P.venv_python(P.VENVS / "chatterbox")
    v.update(status="working", note="")
    P.write_voice(v)

    stages = ["prepare", "ref"] + (["train", "pick"] if use_rvc else []) + ["final"]
    job = ctx.job
    job["stages"] = [{"id": s, "state": "todo"} for s in stages]

    def mark(sid: str, state: str):
        for s in job["stages"]:
            if s["id"] == sid:
                s["state"] = state

    # ---- 1. dataset
    ctx.check()
    mark("prepare", "running")
    if not (ds / "metadata.csv").exists():
        lang = opts.get("lang") if opts.get("lang") in ("pt", "en") else "auto"
        cmd = [str(prep_py), str(ROOT / "prepare_dataset.py"), "--input", str(raw_dir(name)), "--output", str(ds), "--speaker", name,
               "--lang", lang, "--model", prof["whisper_model"], "--device", prof["whisper_device"], "--overwrite"]
        ctx.sh(cmd, log / "prepare.log", "Transcrevendo e cortando a gravação…")
    rows = dataset_rows(v)
    minutes = sum(float(r.get("duration") or 0) for r in rows) / 60
    if minutes < 0.8:
        raise RuntimeError(f"Só {minutes:.1f} min de fala aproveitável saíram da gravação. Grave de novo em um lugar silencioso e fale perto do microfone.")
    mark("prepare", "done")

    # ---- 2. reference clip
    ctx.check()
    mark("ref", "running")
    ctx.set_stage(job, "Escolhendo o trecho de referência…")
    v["ref"] = v.get("ref") or pick_ref(v)
    if not v["ref"]:
        raise RuntimeError("Nenhum trecho de 5 a 12 s com transcrição confiável. Envie uma gravação mais longa e limpa.")
    P.write_voice(v)
    mark("ref", "done")

    if use_rvc:
        exp = name
        tr = {**prof["train"]}
        for k in ("epochs", "save_every", "batch"):
            if opts.get(k):
                tr[k] = int(opts[k])
        # ---- 3. train
        ctx.check()
        mark("train", "running")
        total_ep = tr["epochs"]
        ws = weights_of(exp)
        done_ep = max([int(re.search(r"_e(\d+)", f.name).group(1)) for f in ws], default=0)
        if done_ep < total_ep or not index_of(exp):
            stop = threading.Event()

            def poll():
                while not stop.wait(4):
                    w = weights_of(exp)
                    e = max([int(re.search(r"_e(\d+)", f.name).group(1)) for f in w], default=0)
                    job["progress"] = [min(e, total_ep), total_ep]
                    job["updated"] = time.time()
            threading.Thread(target=poll, daemon=True).start()
            try:
                cmd = [str(rvc_py), str(ROOT / "train_rvc.py"), "--rvc", str(P.RVC), "--wavs", str(ds / "wavs"), "--exp", exp,
                       "--epochs", str(total_ep), "--save-every", str(tr["save_every"]), "--batch", str(tr["batch"]),
                       "--half", tr["half"], "--device", prof["device"], "--workers", str(min(6, max(1, (P.os.cpu_count() or 2) // 2)))]
                ctx.sh(cmd, log / "train.log", "Treinando o seu timbre (RVC)…")
            finally:
                stop.set()
        mark("train", "done")
        job["progress"] = None

        # ---- 4. render every checkpoint and rank by similarity
        ctx.check()
        mark("pick", "running")
        pick(ctx, v, exp, ds, cb_py, prof)
        mark("pick", "done")

    # ---- 5. done
    mark("final", "done")
    v["status"] = "ready"
    P.write_voice(v)
    activate(name)
    ctx.set_stage(job, "Pronto")


def pick(ctx: Ctx, v: dict, exp: str, ds: Path, cb_py: Path, prof: dict) -> None:
    job, nar = ctx.job, ctx.narrate
    ws = weights_of(exp)
    if not ws:
        raise RuntimeError("O treino não gerou checkpoints. Veja voices/%s/logs/train.log" % exp)
    picks = P.voice_dir(exp) / "picks"
    shutil.rmtree(picks, ignore_errors=True)
    picks.mkdir(parents=True)
    rows = dataset_rows(v)
    n_en = sum(1 for r in rows if r.get("lang") == "en")
    lang = "en" if n_en > len(rows) / 2 else "pt"
    model = "v3" if lang == "en" else "ptbr"
    sentences = [s.strip() for s in (ROOT / "test_sentences" / f"{lang}.txt").read_text(encoding="utf-8").splitlines() if s.strip()][:3]

    ctx.set_stage(job, "Gerando a frase de teste com o Chatterbox…")
    work = picks / "_work"
    work.mkdir()
    res = ctx.cb.call({"cmd": "generate", "model": model, "lang": lang, "chunks": sentences, "ref_audio": str(ds / "wavs" / v["ref"]),
                       "outdir": str(work), "exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8, "seed": 1234})
    ctx.cb.loaded = model
    cfg = P.load_config()
    base = picks / "base_chatterbox.wav"
    nar.concat_and_finish([Path(f) for f in res["files"]], base, {"general": {**cfg.get("general", {}), "pause_ms": 250}}, 1.0)
    shutil.rmtree(work, ignore_errors=True)

    index = index_of(exp)
    index_path = str(P.RVC / "assets" / "indices" / index) if index else ""
    cands = {}
    for i, w in enumerate(ws):
        ctx.check()
        ctx.set_stage(job, f"Aplicando o checkpoint {w.name}…", i, len(ws))
        out = picks / (w.stem + ".wav")
        tmp = picks / "_rvc.wav"
        ctx.rv.call({"cmd": "convert", "model": w.name, "input": str(base), "output": str(tmp), "index": index_path,
                     "index_rate": 0.5 if index_path else 0, "protect": 0.33, "pitch": 0, "f0_method": "rmvpe"})
        ctx.rv.loaded = w.name
        nar.concat_and_finish([tmp], out, {"general": {**cfg.get("general", {}), "pause_ms": 0}}, 1.0)
        tmp.unlink(missing_ok=True)
        cands[w.name] = str(out)

    ctx.set_stage(job, "Comparando com a sua voz real…")
    clips = [r["file"] for r in rows if float(r.get("duration") or 0) >= 4]
    random.Random(7).shuffle(clips)
    job_json = picks / "_score.json"
    job_json.write_text(json.dumps({"reference": [str(ds / "wavs" / c) for c in clips[:24]],
                                    "candidates": {"__base__": str(base), **cands}}), encoding="utf-8")
    out = []
    rc = SM.run_logged([str(cb_py), "-W", "ignore", str(ROOT / "engines" / "score_voice.py"), str(ROOT), str(job_json)],
                       P.voice_dir(exp) / "logs" / "score.log", ctx.ctl, on_line=lambda t: out.append(t))
    if rc != 0 or not out or not out[-1].startswith("{"):
        raise RuntimeError("Não consegui comparar os checkpoints. Veja voices/%s/logs/score.log" % exp)
    scores = json.loads(out[-1])["scores"]
    job_json.unlink(missing_ok=True)

    ranking = []
    for w in ws:
        ranking.append({"ckpt": w.name, "epoch": int(re.search(r"_e(\d+)", w.name).group(1)), "score": scores[w.name],
                        "file": f"picks/{w.stem}.wav"})
    best = max(r["score"] for r in ranking)
    # among checkpoints that are practically as good as the best one, take the earliest: later ones tend to overtrain
    chosen = next(r for r in sorted(ranking, key=lambda r: r["epoch"]) if r["score"] >= best - 0.01)
    base_score = scores["__base__"]
    v["ranking"] = ranking
    v["base_score"] = {"score": base_score, "file": "picks/base_chatterbox.wav"}
    v["rvc"] = {"model": chosen["ckpt"], "index": index}
    v["note"] = ("O RVC não aumentou a semelhança medida em relação ao Chatterbox puro; ouça as amostras e, se preferir, desligue o RVC."
                 if best < base_score - 0.005 else "")
    P.write_voice(v)
