#!/usr/bin/env python3
"""
studio.py - a small local web app to test and use your cloned voice.

  venvs\\prep\\Scripts\\python.exe studio.py            (or double-click studio.bat)

It opens http://127.0.0.1:7870 with a page where you type text, tweak the voice
and listen. The heavy models (Chatterbox pt-BR / multilingual, your RVC model) stay loaded
in two helper processes, so after the first run a short narration takes seconds, not minutes.

Only the Python standard library is needed here; the helpers run in their own venvs
(paths come from voice_config.toml: [chatterbox].python and [rvc].python / [rvc].cwd).
Everything is served on 127.0.0.1 only. Results go to out\\studio\\.
"""
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import narrate  # noqa: E402  (text normalisation, chunking, ffmpeg finishing)
import portable  # noqa: E402  (paths, settings, active voice)
import setup_manager as SM  # noqa: E402  (installs the environments)
import tts_engines  # noqa: E402  (the 'Voz base' menu: models, licences)
import voices  # noqa: E402  (the clone pipeline)

portable.add_tools_to_path()

OUT = HERE / "out" / "studio"
REFS = OUT / "refs"
TMP = OUT / "_tmp"
LOGS = OUT / "_logs"
CONFIG_PATH = portable.CONFIG
for d in (OUT, REFS, TMP, LOGS):
    d.mkdir(parents=True, exist_ok=True)

MODELS = tts_engines.MODELS
NAME_RE = re.compile(r"^[\w.\- ()]+$")
LOCK = threading.Lock()


def load_cfg() -> dict:
    return portable.load_config(CONFIG_PATH)


def dataset_dir() -> Path:
    return portable.voice_dataset(portable.active_voice())


def clamp(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def slugify(text: str, n: int = 40) -> str:
    s = re.sub(r"[^\w]+", "-", text.lower(), flags=re.UNICODE).strip("-")
    return (s[:n].strip("-")) or "audio"


# --------------------------------------------------------------------------- helper processes
class Worker:
    """A long-lived helper process that speaks one JSON object per line."""

    def __init__(self, name: str, make_cmd, cwd=None, env=None):
        self.name, self.make_cmd, self.cwd, self.env = name, make_cmd, cwd, env or {}
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.seq = 0
        self.loaded: str | None = None
        self.error: str | None = None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        cmd = self.make_cmd()
        env = {**os.environ, **self.env, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        log = open(LOGS / f"{self.name}.log", "ab")
        self.proc = subprocess.Popen(cmd, cwd=self.cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=log, text=True, encoding="utf-8", bufsize=1, creationflags=flags)
        self.loaded = None
        while True:  # wait for the "ready" line (imports can take a while)
            line = self.proc.stdout.readline()
            if not line:
                self.proc = None
                raise RuntimeError(f"O processo '{self.name}' não iniciou. Veja out\\studio\\_logs\\{self.name}.log")
            try:
                if json.loads(line).get("event") == "ready":
                    return
            except ValueError:
                continue

    def call(self, req: dict, on_event=None) -> dict:
        with self.lock:
            if not self.alive():
                self.start()
            self.seq += 1
            req = {**req, "id": self.seq}
            self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    self.proc = None
                    self.loaded = None
                    raise RuntimeError(f"O processo '{self.name}' parou. Veja out\\studio\\_logs\\{self.name}.log")
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if msg.get("event") == "progress":
                    if on_event:
                        on_event(msg)
                elif msg.get("done") == req["id"]:
                    if not msg.get("ok"):
                        raise RuntimeError(msg.get("error", "erro desconhecido"))
                    return msg

    def kill(self) -> None:
        p = self.proc
        if p and p.poll() is None:
            p.kill()
        self.proc = None
        self.loaded = None


def make_workers() -> tuple[Worker, Worker]:
    cfg = load_cfg()
    # [chatterbox].server / [rvc].server can point to another helper script (used by the tests)
    cb = Worker("chatterbox",
                lambda: [load_cfg()["chatterbox"]["python"], "-W", "ignore",
                         load_cfg()["chatterbox"].get("server", str(HERE / "engines" / "chatterbox_server.py")), str(HERE)])
    rv = Worker("rvc",
                lambda: [load_cfg()["rvc"]["python"], "-W", "ignore",
                         load_cfg()["rvc"].get("server", str(HERE / "engines" / "rvc_server.py")), load_cfg()["rvc"]["cwd"]],
                env={"PYTHONPATH": cfg["rvc"]["cwd"]})
    return cb, rv


CB, RV = make_workers()


def _engine_worker(eid: str) -> Worker:
    def cmd():
        py = portable.venv_python(portable.VENVS / eid)
        return [str(py), "-W", "ignore", str(HERE / "engines" / f"{eid}_server.py"), str(HERE)]
    env = {}
    if eid == "xtts" and "xtts" in portable.read_settings().get("accepted_licenses", []):
        env["COQUI_TOS_AGREED"] = "1"           # set only after the CPML was accepted in the menu
    return Worker(eid, cmd, env=env)


WORKERS = {"chatterbox": CB, "xtts": _engine_worker("xtts"), "f5": _engine_worker("f5")}


def tts_worker(model: str) -> Worker:
    """The worker that serves a model; the other base-voice workers are stopped to free GPU memory."""
    eid = MODELS[model]["engine"]
    for k, w in WORKERS.items():
        if k != eid and w.alive():
            w.kill()
    w = WORKERS[eid]
    if eid == "xtts":
        w.env = {"COQUI_TOS_AGREED": "1"} if "xtts" in portable.read_settings().get("accepted_licenses", []) else {}
    return w


def model_installed(model: str) -> bool:
    eid = MODELS[model]["engine"]
    if eid == "chatterbox":
        return True if model == "v3" else ptbr_ready()
    return SM.SETUP.engine_installed(eid)


def ref_text_of(ref_id: str) -> str:
    """The transcript of a dataset reference clip (F5-TTS needs it); '' for uploaded clips."""
    kind, _, name = (ref_id or "").partition(":")
    if kind != "ds":
        return ""
    for r in list_refs():
        if r["id"] == ref_id:
            return r.get("text", "")
    return ""

# --------------------------------------------------------------------------- jobs
JOBS: dict[str, dict] = {}
QUEUE: "queue.Queue[str]" = queue.Queue()


def set_stage(job: dict, stage: str, done: int | None = None, total: int | None = None) -> None:
    job["stage"] = stage
    job["progress"] = [done, total] if total else None
    job["updated"] = time.time()


def resolve_ref(ref_id: str) -> Path:
    kind, _, name = (ref_id or "").partition(":")
    if not NAME_RE.match(name or "") or ".." in name:
        raise ValueError("referência inválida")
    p = (dataset_dir() / "wavs" / name) if kind == "ds" else (REFS / name) if kind == "up" else None
    if p is None or not p.is_file():
        raise ValueError("arquivo de referência não encontrado")
    return p


def ptbr_ready() -> bool:
    return (HERE / "models" / "chatterbox_ptbr" / "t3_pt_br.safetensors").exists()


def eq_filters(p: dict) -> list[str]:
    """ffmpeg filters for the 'profundidade' controls (all zero = no filtering)."""
    f = []
    if p["bass"] > 0:
        f.append(f"bass=g={p['bass']:.1f}:f=120:w=0.7")           # low shelf: weight in the lows
    if p["body"] != 0:
        f.append(f"equalizer=f=250:t=q:w=0.9:g={p['body']:.1f}")  # chest / body of the voice
    if p["presence"] != 0:
        f.append(f"equalizer=f=3000:t=q:w=1:g={p['presence']:.1f}")  # clarity and bite
    if p["punch"] > 0:
        f.append(f"acompressor=threshold=0.125:ratio={1 + 3 * p['punch']:.2f}:attack=15:release=150:makeup={1 + 1.5 * p['punch']:.2f}")
    return f


def run_generate(job: dict) -> None:
    p = job["params"]
    cfg = load_cfg()
    gen = dict(cfg.get("general", {}))
    gen["pause_ms"] = p["pause_ms"]
    cfg_run = {"general": gen}
    cfg_flat = {"general": dict(gen)}              # same, without the EQ: used for the A/B variant
    gen["post_filter"] = eq_filters(p)
    has_eq = bool(gen["post_filter"])
    lang = p["lang"]
    text = narrate.spell_numbers(p["text"], lang) if lang in ("pt", "en") else p["text"]
    info = MODELS[p["model"]]
    W = tts_worker(p["model"])
    if not model_installed(p["model"]) and info["engine"] != "chatterbox":
        raise ValueError(f"A voz base '{info['label']}' ainda não está instalada. Instale pelo menu Voz base.")
    chunks = narrate.split_text(text, info.get("max_chars", 250))
    if not chunks:
        raise ValueError("Escreva algum texto.")
    ref = resolve_ref(p["ref"])
    seed = p["seed"] if p["seed"] >= 0 else random.randint(0, 2**31 - 1)
    work = TMP / job["id"]
    work.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    first = W.loaded != p["model"]
    if first:
        msg = "Carregando o modelo de voz"
        if p["model"] == "ptbr" and not ptbr_ready():
            msg = "Baixando o modelo pt-BR (só na primeira vez, ~3 GB)"
        set_stage(job, msg + "…")
    else:
        set_stage(job, "Gerando a fala…", 0, len(chunks))

    def on_progress(m):
        set_stage(job, "Gerando a fala…", m["done"], m["total"])

    res = W.call({"cmd": "generate", "model": p["model"], "lang": lang, "chunks": chunks, "ref_audio": str(ref),
                  "ref_text": ref_text_of(p["ref"]) if info["engine"] == "f5" else "",
                  "outdir": str(work), "exaggeration": p["exaggeration"], "cfg_weight": p["cfg_weight"],
                  "temperature": p["temperature"], "seed": seed}, on_progress)
    W.loaded = p["model"]
    t_cb = time.time() - t_start

    set_stage(job, "Juntando as frases…")
    raw = work / "raw.wav"
    narrate.concat_and_finish([Path(f) for f in res["files"]], raw, cfg_flat, 1.0)  # EQ only on the final files

    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"{stamp}_{slugify(p['text'])}"
    ext = p["format"]
    variants = []
    base_name = tts_engines.ENGINE_LABELS[info["engine"]]
    t_rvc = 0.0
    if p["rvc"]:
        set_stage(job, "Aplicando o seu timbre (RVC)…")
        t0 = time.time()
        rvc_out = work / "rvc.wav"
        index = ""
        if p["index"]:
            index = str(Path(cfg["rvc"]["cwd"]) / "assets" / "indices" / p["index"])
        RV.call({"cmd": "convert", "model": p["rvc_model"], "input": str(raw), "output": str(rvc_out), "index": index,
                 "index_rate": p["index_rate"] if index else 0, "protect": p["protect"], "pitch": p["pitch"], "f0_method": "rmvpe"})
        RV.loaded = p["rvc_model"]
        t_rvc = time.time() - t0
        set_stage(job, "Finalizando…")
        f = OUT / f"{base}_rvc.{ext}"
        narrate.concat_and_finish([rvc_out], f, cfg_run, p["speed"])
        variants.append({"label": base_name + " + RVC (seu timbre)" + (" · com graves/corpo" if has_eq else ""), "file": f.name, "rvc": True})
        if has_eq and p["compare"]:
            f2 = OUT / f"{base}_rvc_flat.{ext}"
            narrate.concat_and_finish([rvc_out], f2, cfg_flat, p["speed"])
            variants.append({"label": base_name + " + RVC · sem ajuste de graves", "file": f2.name, "rvc": True})
    if (not p["rvc"]) or p["compare"]:
        set_stage(job, "Finalizando…")
        f = OUT / f"{base}_cb.{ext}"
        narrate.concat_and_finish([raw], f, cfg_run, p["speed"])
        variants.append({"label": "Só " + base_name, "file": f.name, "rvc": False})
    for v in variants:
        try:
            v["duration"] = round(narrate.wav_duration(OUT / v["file"]), 2)
        except Exception:
            v["duration"] = None
    meta = {"name": base, "created": time.time(), "text": p["text"], "params": {**p, "seed": seed}, "variants": variants,
            "chunks": len(chunks), "seconds": round(time.time() - t_start, 1), "seconds_voice": round(t_cb, 1),
            "seconds_rvc": round(t_rvc, 1)}
    (OUT / f"{base}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    job["result"] = meta


def run_preload(job: dict) -> None:
    p = job["params"]
    W = tts_worker(p["model"])
    set_stage(job, "Carregando a voz base…")
    if p["model"] == "ptbr" and not ptbr_ready():
        set_stage(job, "Baixando o modelo pt-BR (só na primeira vez, ~3 GB)…")
    W.call({"cmd": "load", "model": p["model"]})
    W.loaded = p["model"]
    if p.get("rvc_model"):
        set_stage(job, "Carregando o RVC…")
        RV.call({"cmd": "load", "model": p["rvc_model"]})
        RV.loaded = p["rvc_model"]
    job["result"] = {"ok": True}


def worker_loop() -> None:
    while True:
        jid = QUEUE.get()
        job = JOBS.get(jid)
        if not job or job["status"] == "cancelled":
            continue
        job["status"] = "running"
        job["started"] = time.time()
        try:
            (run_generate if job["kind"] == "generate" else run_preload)(job)
            job["status"] = "done"
            set_stage(job, "Pronto")
        except Exception as e:
            if job["status"] != "cancelled":
                traceback.print_exc()
                job["status"] = "error"
                job["error"] = str(e)
        job["finished"] = time.time()


# --------------------------------------------------------------------------- saved settings
PRESETS_PATH = HERE / "presets.json"      # git-ignored: personal
PRESET_KEYS = {"lang": str, "model": str, "ref": str, "exaggeration": float, "cfg_weight": float, "temperature": float,
               "pause_ms": float, "speed": float, "seed": float, "format": str, "rvc": bool, "compare": bool,
               "rvc_model": str, "index": str, "index_rate": float, "protect": float, "pitch": float,
               "bass": float, "body": float, "presence": float, "punch": float}


def read_presets() -> dict:
    try:
        return json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_preset(name: str, values: dict) -> None:
    name = name.strip()
    if not name or len(name) > 40:
        raise ValueError("Dê um nome de até 40 caracteres.")
    clean = {}
    for k, t in PRESET_KEYS.items():
        v = values.get(k)
        if v is None:
            continue
        if t is bool:
            clean[k] = bool(v)
        elif t is float:
            clean[k] = clamp(v, -1e9, 1e9, 0)
        else:
            clean[k] = str(v)[:200]
    allp = read_presets()
    if len(allp) >= 100 and name not in allp:
        raise ValueError("Limite de 100 configurações salvas.")
    allp[name] = {"values": clean, "saved": time.time()}
    PRESETS_PATH.write_text(json.dumps(allp, ensure_ascii=False, indent=1), encoding="utf-8")


def delete_preset(name: str) -> None:
    allp = read_presets()
    if allp.pop(name, None) is not None:
        PRESETS_PATH.write_text(json.dumps(allp, ensure_ascii=False, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- cloning jobs (long, own thread)
CLONE_Q: "queue.Queue[str]" = queue.Queue()


def clone_loop() -> None:
    while True:
        jid = CLONE_Q.get()
        job = JOBS.get(jid)
        if not job or job["status"] == "cancelled":
            continue
        job["status"] = "running"
        job["started"] = time.time()
        try:
            voices.run_clone(voices.Ctx(job, CB, RV, set_stage, narrate), job["voice"], job["params"])
            job["status"] = "done"
            set_stage(job, "Pronto")
        except SM.Cancelled:
            job["status"] = "cancelled"
            set_stage(job, "Cancelado")
        except Exception as e:  # noqa: BLE001
            if job["status"] != "cancelled":
                traceback.print_exc()
                job["status"] = "error"
                job["error"] = str(e)
        if job["status"] != "done":
            v = portable.read_voice(job["voice"])
            if v:
                v["status"] = "interrupted"
                portable.write_voice(v)
        job["finished"] = time.time()


def clean_params(raw: dict, cfg: dict) -> dict:
    lang = raw.get("lang") if raw.get("lang") in ("pt", "en") else "pt"
    model = raw.get("model") if raw.get("model") in MODELS else "ptbr"
    if lang not in MODELS[model]["langs"]:
        model = tts_engines.models_for(lang)[0]
    fmt = raw.get("format") if raw.get("format") in ("wav", "mp3") else "wav"
    text = str(raw.get("text", "")).strip()
    if len(text) > 6000:
        raise ValueError("Texto muito longo (máximo 6000 caracteres).")
    rv = cfg.get("rvc", {})
    rvc_model = str(raw.get("rvc_model") or rv.get("model_name", ""))
    if rvc_model and not NAME_RE.match(rvc_model):
        raise ValueError("modelo RVC inválido")
    index = str(raw.get("index") or "")
    if index and not NAME_RE.match(index):
        raise ValueError("índice inválido")
    return {
        "text": text, "lang": lang, "model": model, "ref": str(raw.get("ref") or ""),
        "exaggeration": clamp(raw.get("exaggeration"), 0.0, 2.0, 0.5), "cfg_weight": clamp(raw.get("cfg_weight"), 0.0, 1.0, 0.5),
        "temperature": clamp(raw.get("temperature"), 0.1, 2.0, 0.8), "seed": int(clamp(raw.get("seed"), -1, 2**31 - 1, -1)),
        "pause_ms": int(clamp(raw.get("pause_ms"), 0, 2000, 250)), "speed": clamp(raw.get("speed"), 0.7, 1.4, 1.0),
        "format": fmt, "rvc": bool(raw.get("rvc", True)) and bool(rvc_model), "compare": bool(raw.get("compare", False)),
        "rvc_model": rvc_model, "index": index, "index_rate": clamp(raw.get("index_rate"), 0.0, 1.0, 0.5),
        "protect": clamp(raw.get("protect"), 0.0, 0.5, 0.33), "pitch": int(clamp(raw.get("pitch"), -12, 12, 0)),
        "bass": clamp(raw.get("bass"), 0.0, 10.0, 0.0), "body": clamp(raw.get("body"), -6.0, 8.0, 0.0),
        "presence": clamp(raw.get("presence"), -6.0, 8.0, 0.0), "punch": clamp(raw.get("punch"), 0.0, 1.0, 0.0),
    }


# --------------------------------------------------------------------------- state
_gpu_cache = {"t": 0.0, "v": None}


def gpu_info() -> dict | None:
    if time.time() - _gpu_cache["t"] < 4:
        return _gpu_cache["v"]
    v = None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=3)
        n, used, total = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        v = {"name": n.replace("NVIDIA GeForce ", ""), "used_gb": round(int(used) / 1024, 1), "total_gb": round(int(total) / 1024, 1)}
    except Exception:
        pass
    _gpu_cache.update(t=time.time(), v=v)
    return v


def epoch_of(name: str) -> int:
    m = re.search(r"_e(\d+)", name)
    return int(m.group(1)) if m else 10**9


def list_refs() -> list[dict]:
    refs = []
    meta = dataset_dir() / "metadata.csv"
    if meta.exists():
        with open(meta, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh, delimiter="|"):
                try:
                    refs.append({"id": "ds:" + r["file"], "label": r["file"].replace(".wav", ""), "dur": float(r["duration"]),
                                 "text": r["text"], "kind": "dataset"})
                except (KeyError, ValueError):
                    continue
    for f in sorted(REFS.glob("*.wav")):
        refs.insert(0, {"id": "up:" + f.name, "label": f.stem, "dur": round(narrate.wav_duration(f), 1), "text": "(enviado por você)", "kind": "upload"})
    return refs


def list_history(limit: int = 40) -> list[dict]:
    items = []
    for f in sorted(OUT.glob("*.json"), reverse=True)[:limit]:
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
            m["variants"] = [v for v in m["variants"] if (OUT / v["file"]).exists()]
            if m["variants"]:
                items.append(m)
        except Exception:
            continue
    return items


def build_state() -> dict:
    cfg = load_cfg()
    rv, cb = cfg.get("rvc", {}), cfg.get("chatterbox", {})
    wdir = Path(rv.get("cwd", "")) / "assets" / "weights"
    idir = Path(rv.get("cwd", "")) / "assets" / "indices"
    av = portable.active_voice()
    prefix = (av["name"] + "_e") if av else ""
    weights = sorted([f.name for f in wdir.glob("*.pth") if f.name.startswith(prefix) and "_e" in f.name], key=lambda n: (epoch_of(n), n)) if wdir.exists() else []
    indices = sorted(f.name for f in idir.glob("*.index") if not av or f.name.startswith(av["name"] + "_")) if idir.exists() else []
    ref_default = Path(cb.get("ref_audio", "")).name
    return {
        "workers": {"chatterbox": {"alive": CB.alive(), "loaded": CB.loaded}, **{k: {"alive": w.alive(), "loaded": w.loaded} for k, w in WORKERS.items() if k != "chatterbox"}, "rvc": {"alive": RV.alive(), "loaded": RV.loaded}},
        "models": [{"key": k, "label": v["label"], "langs": v["langs"], "installed": model_installed(k),
                    "ready": ptbr_ready() if k == "ptbr" else None, "engine": v["engine"], "license": v["license"],
                    "commercial": v["commercial"], "pt_br": v["pt_br"], "tested": v["tested"], "note": v["note"], "size": v["size"]}
                   for k, v in MODELS.items()],
        "catalog": tts_engines.CATALOG, "accepted_licenses": portable.read_settings().get("accepted_licenses", []),
        "rvc_models": weights, "rvc_default": rv.get("model_name") or None, "indices": indices, "index_default": Path(rv.get("index_path", "")).name,
        "refs": list_refs(), "ref_default": "ds:" + ref_default if ref_default else None,
        "defaults": {"exaggeration": cb.get("exaggeration", 0.5), "cfg_weight": cb.get("cfg_weight", 0.5),
                     "index_rate": rv.get("index_rate", 0.5), "protect": rv.get("protect", 0.33),
                     "pause_ms": cfg.get("general", {}).get("pause_ms", 250), "speed": cfg.get("general", {}).get("speed", 1.0)},
        "active_voice": {"name": av["name"], "display": av.get("display", av["name"]), "rvc": bool(av.get("rvc"))} if av else None,
        "setup_ready": SM.SETUP.status()["ready"] or bool(av), "queue": QUEUE.qsize(), "history": list_history(),
    }


def public_job(job: dict) -> dict:
    keys = ("id", "kind", "status", "stage", "progress", "error", "result", "created", "started", "finished", "updated", "stages", "detail", "voice")
    d = {k: job.get(k) for k in keys}
    d["queue_position"] = None
    if job["status"] == "queued":
        ahead = [j for j in JOBS.values() if j["status"] == "queued" and j["created"] < job["created"]]
        d["queue_position"] = len(ahead) + 1
    return d


# --------------------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    server_version = "VoiceStudio/1.0"

    def log_message(self, *a):  # quiet
        pass

    # -- helpers
    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost", "[::1]")

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, status, msg):
        self._json({"error": msg}, status)

    def _body(self, limit=64 * 1024 * 1024) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > limit:
            raise ValueError("arquivo grande demais")
        return self.rfile.read(n)

    def _file(self, path: Path, download_name: str | None = None):
        if not path.is_file():
            return self._err(404, "não encontrado")
        size = path.stat().st_size
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        start, end, status = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng and (m := re.match(r"bytes=(\d*)-(\d*)", rng)):
            a, b = m.groups()
            if a == "" and b:
                start = max(0, size - int(b))
            elif a:
                start = int(a)
                end = int(b) if b else size - 1
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = fh.read(min(256 * 1024, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    # -- routes
    def do_GET(self):
        if not self._host_ok():
            return self._err(403, "host não permitido")
        u = urlparse(self.path)
        path, q = u.path, parse_qs(u.query)
        try:
            if path in ("/", "/index.html"):
                return self._file(HERE / "ui" / "index.html")
            if path == "/api/state":
                return self._json(build_state())
            if path == "/api/presets":
                return self._json({"presets": [{"name": k, "values": v["values"], "saved": v.get("saved")}
                                               for k, v in sorted(read_presets().items(), key=lambda kv: kv[0].lower())]})
            if path == "/api/setup":
                return self._json(SM.SETUP.status())
            if path == "/api/voices":
                return self._json({"voices": [voices.summary(v) for v in portable.list_voices()],
                                   "active": portable.read_settings().get("active_voice"),
                                   "jobs": [public_job(j) for j in JOBS.values() if j["kind"] == "clone"][-3:]})
            if path.startswith("/file/voice/"):
                parts = unquote(path[len("/file/voice/"):]).split("/")
                if len(parts) != 3 or parts[1] != "picks" or not all(NAME_RE.match(x) and ".." not in x for x in parts):
                    return self._err(400, "nome inválido")
                return self._file(portable.voice_dir(parts[0]) / "picks" / parts[2])
            if path == "/api/job":
                job = JOBS.get((q.get("id") or [""])[0])
                return self._json(public_job(job)) if job else self._err(404, "job não encontrado")
            if path.startswith("/file/out/"):
                name = unquote(path[len("/file/out/"):])
                if not NAME_RE.match(name) or ".." in name:
                    return self._err(400, "nome inválido")
                return self._file(OUT / name, download_name=name if q.get("dl") else None)
            if path.startswith("/file/ref/"):
                return self._file(resolve_ref(unquote(path[len("/file/ref/"):])))
            return self._err(404, "não encontrado")
        except ValueError as e:
            return self._err(400, str(e))

    def do_POST(self):
        if not self._host_ok() or self.headers.get("X-Studio") != "1":
            return self._err(403, "pedido não permitido")
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/api/generate":
                raw = json.loads(self._body() or b"{}")
                params = clean_params(raw, load_cfg())
                if not params["text"]:
                    return self._err(400, "Escreva algum texto.")
                if not params["ref"]:
                    return self._err(400, "Escolha um trecho de referência.")
                resolve_ref(params["ref"])
                jid = uuid.uuid4().hex[:10]
                JOBS[jid] = {"id": jid, "kind": "generate", "status": "queued", "stage": "Na fila…", "progress": None,
                             "params": params, "created": time.time()}
                QUEUE.put(jid)
                return self._json({"id": jid})
            if u.path == "/api/preload":
                raw = json.loads(self._body() or b"{}")
                params = clean_params({**raw, "text": "x"}, load_cfg())
                jid = uuid.uuid4().hex[:10]
                JOBS[jid] = {"id": jid, "kind": "preload", "status": "queued", "stage": "Na fila…", "progress": None,
                             "params": {"model": params["model"], "rvc_model": params["rvc_model"] if raw.get("rvc", True) else ""},
                             "created": time.time()}
                QUEUE.put(jid)
                return self._json({"id": jid})
            if u.path == "/api/cancel":
                raw = json.loads(self._body() or b"{}")
                job = JOBS.get(raw.get("id"))
                if not job:
                    return self._err(404, "job não encontrado")
                if job["status"] == "queued":
                    job["status"] = "cancelled"
                elif job["status"] == "running":
                    job["status"] = "cancelled"
                    ctl = job.get("ctl")
                    if ctl is not None:  # a clone job: stop its child process too
                        ctl["cancel"] = True
                        if ctl.get("proc"):
                            SM.kill_tree(ctl["proc"])
                    [w.kill() for w in WORKERS.values()]
                    RV.kill()
                return self._json({"ok": True})
            if u.path == "/api/presets/save":
                raw = json.loads(self._body() or b"{}")
                save_preset(str(raw.get("name", "")), raw.get("values") or {})
                return self._json({"ok": True})
            if u.path == "/api/presets/delete":
                raw = json.loads(self._body() or b"{}")
                delete_preset(str(raw.get("name", "")))
                return self._json({"ok": True})
            if u.path == "/api/setup/start":
                raw = json.loads(self._body() or b"{}")
                SM.SETUP.start(str(raw.get("profile")), bool(raw.get("with_rvc")))
                return self._json({"ok": True})
            if u.path == "/api/engine/install":
                raw = json.loads(self._body() or b"{}")
                eid = str(raw.get("engine"))
                if eid not in SM.ENGINE_STEPS:
                    return self._err(400, "Voz base desconhecida.")
                lic = [k for k, v in MODELS.items() if v["engine"] == eid and v["commercial"] != "sim"]
                if lic and not raw.get("accept"):
                    return self._err(400, "Aceite a licença para instalar.")
                acc = set(portable.read_settings().get("accepted_licenses", []))
                portable.write_settings(accepted_licenses=sorted(acc | ({eid} if lic else set())))
                SM.SETUP.start_engine(eid)
                return self._json({"ok": True})
            if u.path == "/api/setup/cancel":
                SM.SETUP.cancel()
                return self._json({"ok": True})
            if u.path == "/api/voice/create":
                raw = json.loads(self._body() or b"{}")
                display = str(raw.get("display", "")).strip()
                if not display:
                    return self._err(400, "Dê um nome para a voz.")
                return self._json(voices.summary(voices.create_voice(display)))
            if u.path == "/api/voice/upload":
                name = (q.get("voice") or [""])[0]
                if not NAME_RE.match(name) or not portable.read_voice(name):
                    return self._err(400, "voz inválida")
                orig = unquote((q.get("name") or ["gravacao"])[0])
                n = int(self.headers.get("Content-Length") or 0)
                if n > 3 * 1024**3:
                    return self._err(400, "arquivo grande demais (máximo 3 GB)")
                tmp = TMP / f"upload_{uuid.uuid4().hex[:6]}{Path(orig).suffix[:8] or '.bin'}"
                with open(tmp, "wb") as fh:
                    left = n
                    while left > 0:
                        chunk = self.rfile.read(min(1 << 20, left))
                        if not chunk:
                            break
                        fh.write(chunk)
                        left -= len(chunk)
                try:
                    return self._json(voices.add_recording(name, orig, tmp))
                finally:
                    tmp.unlink(missing_ok=True)
            if u.path == "/api/voice/remove_recording":
                raw = json.loads(self._body() or b"{}")
                if not NAME_RE.match(str(raw.get("voice", ""))) or not NAME_RE.match(str(raw.get("file", ""))):
                    return self._err(400, "nome inválido")
                voices.remove_recording(raw["voice"], raw["file"])
                return self._json({"ok": True})
            if u.path == "/api/voice/clone":
                raw = json.loads(self._body() or b"{}")
                name = str(raw.get("voice", ""))
                if not NAME_RE.match(name) or not portable.read_voice(name):
                    return self._err(400, "voz inválida")
                if any(j["kind"] == "clone" and j["status"] in ("queued", "running") for j in JOBS.values()):
                    return self._err(400, "Já existe uma clonagem em andamento.")
                opts = {"rvc": bool(raw.get("rvc")), "lang": raw.get("lang"),
                        **{k: int(raw[k]) for k in ("epochs", "save_every", "batch") if raw.get(k)}}
                jid = uuid.uuid4().hex[:10]
                JOBS[jid] = {"id": jid, "kind": "clone", "status": "queued", "stage": "Na fila…", "progress": None, "voice": name,
                             "params": opts, "created": time.time(), "ctl": {"cancel": False, "proc": None}}
                CLONE_Q.put(jid)
                return self._json({"id": jid})
            if u.path == "/api/voice/activate":
                raw = json.loads(self._body() or b"{}")
                voices.activate(str(raw.get("voice", "")))
                return self._json({"ok": True})
            if u.path == "/api/voice/checkpoint":
                raw = json.loads(self._body() or b"{}")
                if not NAME_RE.match(str(raw.get("ckpt", ""))):
                    return self._err(400, "checkpoint inválido")
                voices.set_checkpoint(str(raw.get("voice", "")), raw["ckpt"])
                return self._json({"ok": True})
            if u.path == "/api/voice/rvc_off":
                raw = json.loads(self._body() or b"{}")
                voices.set_rvc_off(str(raw.get("voice", "")))
                return self._json({"ok": True})
            if u.path == "/api/delete":
                raw = json.loads(self._body() or b"{}")
                name = str(raw.get("name", ""))
                if not NAME_RE.match(name) or ".." in name:
                    return self._err(400, "nome inválido")
                for f in OUT.glob(name + "*"):
                    if f.is_file():
                        f.unlink()
                return self._json({"ok": True})
            if u.path == "/api/upload_ref":
                data = self._body()
                stem = re.sub(r"[^\w\-]+", "_", Path(unquote((q.get("name") or ["ref"])[0])).stem)[:40] or "ref"
                src = TMP / f"upload_{uuid.uuid4().hex[:6]}{Path(unquote((q.get('name') or [''])[0])).suffix or '.bin'}"
                src.write_bytes(data)
                dst = REFS / f"{stem}.wav"
                r = subprocess.run([portable.ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "44100", "-t", "30", str(dst)],
                                   capture_output=True, text=True)
                src.unlink(missing_ok=True)
                if r.returncode != 0:
                    return self._err(400, "Não consegui ler esse áudio.")
                return self._json({"id": "up:" + dst.name})
            return self._err(404, "não encontrado")
        except ValueError as e:
            return self._err(400, str(e))
        except Exception as e:
            traceback.print_exc()
            return self._err(500, f"{type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Local web studio for your cloned voice.")
    ap.add_argument("--port", type=int, default=7870)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-warmup", action="store_true", help="do not load the models in the background at start")
    args = ap.parse_args()

    voices.discover_legacy()
    threading.Thread(target=worker_loop, daemon=True).start()
    threading.Thread(target=clone_loop, daemon=True).start()
    if not args.no_warmup and ptbr_ready() and portable.active_voice():  # models already downloaded: load them while the page opens
        cfg = load_cfg()
        jid = uuid.uuid4().hex[:10]
        JOBS[jid] = {"id": jid, "kind": "preload", "status": "queued", "stage": "Na fila…", "progress": None,
                     "params": {"model": "ptbr", "rvc_model": cfg["rvc"].get("model_name", "")}, "created": time.time()}
        QUEUE.put(jid)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Voice studio: {url}   (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        [w.kill() for w in WORKERS.values()]
        RV.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
