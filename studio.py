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

import tomllib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import narrate  # noqa: E402  (text normalisation, chunking, ffmpeg finishing)

OUT = HERE / "out" / "studio"
REFS = OUT / "refs"
TMP = OUT / "_tmp"
LOGS = OUT / "_logs"
DATASET = HERE / "dataset"
CONFIG_PATH = Path(os.environ.get("VOICE_CONFIG", HERE / "voice_config.toml"))
for d in (OUT, REFS, TMP, LOGS):
    d.mkdir(parents=True, exist_ok=True)

MODELS = {
    "ptbr": {"label": "Português do Brasil (modelo dedicado)", "lang": "pt"},
    "v3": {"label": "Multilíngue v3 (23 idiomas)", "lang": "any"},
}
NAME_RE = re.compile(r"^[\w.\- ()]+$")
LOCK = threading.Lock()


def load_cfg() -> dict:
    return tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))


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
    p = (DATASET / "wavs" / name) if kind == "ds" else (REFS / name) if kind == "up" else None
    if p is None or not p.is_file():
        raise ValueError("arquivo de referência não encontrado")
    return p


def ptbr_ready() -> bool:
    return (HERE / "models" / "chatterbox_ptbr" / "t3_pt_br.safetensors").exists()


def run_generate(job: dict) -> None:
    p = job["params"]
    cfg = load_cfg()
    gen = dict(cfg.get("general", {}))
    gen["pause_ms"] = p["pause_ms"]
    cfg_run = {"general": gen}
    lang = p["lang"]
    text = narrate.spell_numbers(p["text"], lang) if lang in ("pt", "en") else p["text"]
    chunks = narrate.split_text(text, 250)
    if not chunks:
        raise ValueError("Escreva algum texto.")
    ref = resolve_ref(p["ref"])
    seed = p["seed"] if p["seed"] >= 0 else random.randint(0, 2**31 - 1)
    work = TMP / job["id"]
    work.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    first = CB.loaded != p["model"]
    if first:
        msg = "Carregando o modelo de voz"
        if p["model"] == "ptbr" and not ptbr_ready():
            msg = "Baixando o modelo pt-BR (só na primeira vez, ~3 GB)"
        set_stage(job, msg + "…")
    else:
        set_stage(job, "Gerando a fala…", 0, len(chunks))

    def on_progress(m):
        set_stage(job, "Gerando a fala…", m["done"], m["total"])

    res = CB.call({"cmd": "generate", "model": p["model"], "lang": lang, "chunks": chunks, "ref_audio": str(ref),
                   "outdir": str(work), "exaggeration": p["exaggeration"], "cfg_weight": p["cfg_weight"],
                   "temperature": p["temperature"], "seed": seed}, on_progress)
    CB.loaded = p["model"]
    t_cb = time.time() - t_start

    set_stage(job, "Juntando as frases…")
    raw = work / "raw.wav"
    narrate.concat_and_finish([Path(f) for f in res["files"]], raw, cfg_run, 1.0)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"{stamp}_{slugify(p['text'])}"
    ext = p["format"]
    variants = []
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
        variants.append({"label": "Chatterbox + RVC (seu timbre)", "file": f.name, "rvc": True})
    if (not p["rvc"]) or p["compare"]:
        set_stage(job, "Finalizando…")
        f = OUT / f"{base}_cb.{ext}"
        narrate.concat_and_finish([raw], f, cfg_run, p["speed"])
        variants.append({"label": "Só Chatterbox", "file": f.name, "rvc": False})
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
    set_stage(job, "Carregando o Chatterbox…")
    if p["model"] == "ptbr" and not ptbr_ready():
        set_stage(job, "Baixando o modelo pt-BR (só na primeira vez, ~3 GB)…")
    CB.call({"cmd": "load", "model": p["model"]})
    CB.loaded = p["model"]
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


def clean_params(raw: dict, cfg: dict) -> dict:
    lang = raw.get("lang") if raw.get("lang") in ("pt", "en") else "pt"
    model = raw.get("model") if raw.get("model") in MODELS else "ptbr"
    if lang != "pt":
        model = "v3"
    fmt = raw.get("format") if raw.get("format") in ("wav", "mp3") else "wav"
    text = str(raw.get("text", "")).strip()
    if len(text) > 6000:
        raise ValueError("Texto muito longo (máximo 6000 caracteres).")
    rv = cfg.get("rvc", {})
    rvc_model = str(raw.get("rvc_model") or rv.get("model_name", ""))
    if not NAME_RE.match(rvc_model):
        raise ValueError("modelo RVC inválido")
    index = str(raw.get("index") or "")
    if index and not NAME_RE.match(index):
        raise ValueError("índice inválido")
    return {
        "text": text, "lang": lang, "model": model, "ref": str(raw.get("ref") or ""),
        "exaggeration": clamp(raw.get("exaggeration"), 0.0, 2.0, 0.5), "cfg_weight": clamp(raw.get("cfg_weight"), 0.0, 1.0, 0.5),
        "temperature": clamp(raw.get("temperature"), 0.1, 2.0, 0.8), "seed": int(clamp(raw.get("seed"), -1, 2**31 - 1, -1)),
        "pause_ms": int(clamp(raw.get("pause_ms"), 0, 2000, 250)), "speed": clamp(raw.get("speed"), 0.7, 1.4, 1.0),
        "format": fmt, "rvc": bool(raw.get("rvc", True)), "compare": bool(raw.get("compare", False)),
        "rvc_model": rvc_model, "index": index, "index_rate": clamp(raw.get("index_rate"), 0.0, 1.0, 0.5),
        "protect": clamp(raw.get("protect"), 0.0, 0.5, 0.33), "pitch": int(clamp(raw.get("pitch"), -12, 12, 0)),
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
    meta = DATASET / "metadata.csv"
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
    weights = sorted([f.name for f in wdir.glob("*.pth")], key=lambda n: (epoch_of(n), n)) if wdir.exists() else []
    indices = sorted(f.name for f in idir.glob("*.index")) if idir.exists() else []
    ref_default = Path(cb.get("ref_audio", "")).name
    return {
        "workers": {"chatterbox": {"alive": CB.alive(), "loaded": CB.loaded}, "rvc": {"alive": RV.alive(), "loaded": RV.loaded}},
        "models": [{"key": k, "label": v["label"], "lang": v["lang"], "ready": ptbr_ready() if k == "ptbr" else None} for k, v in MODELS.items()],
        "rvc_models": weights, "rvc_default": rv.get("model_name"), "indices": indices, "index_default": Path(rv.get("index_path", "")).name,
        "refs": list_refs(), "ref_default": "ds:" + ref_default if ref_default else None,
        "defaults": {"exaggeration": cb.get("exaggeration", 0.5), "cfg_weight": cb.get("cfg_weight", 0.5),
                     "index_rate": rv.get("index_rate", 0.5), "protect": rv.get("protect", 0.33),
                     "pause_ms": cfg.get("general", {}).get("pause_ms", 250), "speed": cfg.get("general", {}).get("speed", 1.0)},
        "gpu": gpu_info(), "queue": QUEUE.qsize(), "history": list_history(),
    }


def public_job(job: dict) -> dict:
    keys = ("id", "kind", "status", "stage", "progress", "error", "result", "created", "started", "finished", "updated")
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
                    CB.kill()
                    RV.kill()
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
                r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "44100", "-t", "30", str(dst)],
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

    threading.Thread(target=worker_loop, daemon=True).start()
    if not args.no_warmup and ptbr_ready():  # models already downloaded: load them while the page opens
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
        CB.kill()
        RV.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
