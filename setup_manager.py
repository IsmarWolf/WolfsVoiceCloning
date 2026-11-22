#!/usr/bin/env python3
"""
setup_manager.py - installs everything the studio needs, on any PC. Standard library only.

Two hardware profiles:
  cuda  NVIDIA GPU, PyTorch built for CUDA 12.8. Tested on an RTX 5060 Ti (50 series) on Windows 11.
        Other NVIDIA cards (driver 570+) should work with the same wheels but were NOT tested.
  cpu   No NVIDIA GPU. Everything runs, but slowly; RVC training is optional and takes hours.

Each step checks whether it is already done, so running the setup again only does what is missing.
Every step writes its full output to out/setup/<step>.log.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

import portable as P

OUT = P.ROOT / "out" / "setup"
OUT.mkdir(parents=True, exist_ok=True)

# Exact versions that were tested together (see 02_TRAINING_GUIDE.md).
RVC_COMMIT = "81eed5e8f68b6bed1789f682fe78cdd324495afc"          # RVC-Project/Retrieval-based-Voice-Conversion-WebUI
CHATTERBOX_COMMIT = "ab55923c27757c90cfcc407baa57dc25a7d25217"  # resemble-ai/chatterbox
RVC_ZIP = f"https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/archive/{RVC_COMMIT}.zip"
CHATTERBOX_ZIP = f"https://github.com/resemble-ai/chatterbox/archive/{CHATTERBOX_COMMIT}.zip"
PYPI = "https://pypi.org/simple"

PROFILES = {
    "cuda": {
        "label": "GPU NVIDIA (CUDA 12.8)",
        "device": "cuda",
        "tested": "Testado: RTX 5060 Ti (série 50), Windows 11. Outras placas NVIDIA com driver 570 ou mais novo devem funcionar com os mesmos pacotes, mas não foram testadas.",
        "torch_index": "https://download.pytorch.org/whl/cu128",
        "torch_chatterbox": ["torch==2.7.1", "torchaudio==2.7.1"],
        "rvc_torch": ["torch==2.7.1", "torchaudio==2.7.1"],     # then the cu128 requirements file
        "rvc_req": "requirments_cu128_py312.txt",
        "prep_extra": ["nvidia-cublas-cu12", "nvidia-cudnn-cu12==9.*"],
        "whisper_model": "large-v3", "whisper_device": "cuda",
        "train": {"epochs": 100, "save_every": 10, "batch": 12, "half": "True"},
        "default_rvc": True,
    },
    "cpu": {
        "label": "Somente CPU",
        "device": "cpu",
        "tested": "Não testado em PC real sem placa NVIDIA. Funciona, mas é lento: gerar uma frase leva minutos e treinar o RVC leva horas. Sugestão: use só o Chatterbox (sem treino de timbre).",
        "torch_index": "https://download.pytorch.org/whl/cpu",
        "torch_chatterbox": ["torch==2.7.1", "torchaudio==2.7.1"],
        "rvc_torch": [],                                          # the CPU requirements file pins torch 2.4.1
        "rvc_req": "requirments_cpu_py312.txt",
        "prep_extra": [],
        "whisper_model": "small", "whisper_device": "cpu",
        "train": {"epochs": 30, "save_every": 5, "batch": 4, "half": "False"},
        "default_rvc": False,
    },
}

CHATTERBOX_DEPS = ["numpy<2", "librosa==0.11.0", "s3tokenizer", "transformers==5.2.0", "diffusers==0.29.0", "resemble-perth",
                   "conformer==0.3.2", "safetensors==0.5.3", "spacy-pkuseg", "pykakasi==2.3.0", "pyloudnorm", "omegaconf",
                   "soundfile", "setuptools<81", "huggingface_hub"]


# --------------------------------------------------------------------------- processes
def kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if P.IS_WIN:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=15)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        proc.kill()


class Cancelled(Exception):
    pass


def run_logged(cmd: list[str], log: Path, ctl: dict, cwd: Path | None = None, env: dict | None = None,
               on_line=None) -> int:
    """Run a command, append its output to `log`, call on_line(text) for each line. ctl = {"cancel": bool, "proc": Popen}."""
    if ctl.get("cancel"):
        raise Cancelled()
    log.parent.mkdir(parents=True, exist_ok=True)
    full_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "PYTHONUNBUFFERED": "1", **(env or {})}
    kw = {"creationflags": subprocess.CREATE_NO_WINDOW} if P.IS_WIN else {"start_new_session": True}
    with open(log, "ab") as fh:
        fh.write(("\n$ " + " ".join(cmd) + "\n").encode("utf-8", "replace"))
        proc = subprocess.Popen(cmd, cwd=cwd, env=full_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw)
        ctl["proc"] = proc
        buf = b""
        while True:
            chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
            if not chunk:
                break
            fh.write(chunk)
            fh.flush()
            buf += chunk
            parts = re.split(rb"[\r\n]+", buf)
            buf = parts.pop()
            if on_line:
                for ln in parts:
                    if ln.strip():
                        on_line(ln.decode("utf-8", "replace").strip())
        proc.wait()
        ctl["proc"] = None
    if ctl.get("cancel"):
        raise Cancelled()
    return proc.returncode


def download(url: str, dest: Path, ctl: dict, on_line=None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "wolf-voice-studio"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, 0.0
        while True:
            if ctl.get("cancel"):
                raise Cancelled()
            b = r.read(1 << 20)
            if not b:
                break
            fh.write(b)
            done += len(b)
            if on_line and time.time() - last > 1:
                on_line(f"baixando {done / 1e6:.0f}" + (f" / {total / 1e6:.0f} MB" if total else " MB"))
                last = time.time()


# --------------------------------------------------------------------------- hardware
_hw = {"t": 0.0, "v": None}


def detect_hardware() -> dict:
    if _hw["v"] and time.time() - _hw["t"] < 30:
        return _hw["v"]
    _hw.update(v=_detect_hardware(), t=time.time())
    return _hw["v"]


def _detect_hardware() -> dict:
    info: dict = {"os": f"{sys.platform}", "python": sys.version.split()[0], "cpu_count": os.cpu_count(), "gpu": None,
                  "ffmpeg": P.has_ffmpeg(), "git": shutil.which("git") is not None}
    try:
        info["disk_free_gb"] = round(shutil.disk_usage(P.ROOT).free / 1e9)
    except Exception:
        info["disk_free_gb"] = None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        name, mem, drv, cc = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        info["gpu"] = {"name": name.replace("NVIDIA ", ""), "memory_gb": round(int(mem) / 1024, 1), "driver": drv, "compute": cc}
    except Exception:
        pass
    g = info["gpu"]
    try:
        driver_ok = g and float(g["driver"].split(".")[0]) >= 570
    except ValueError:
        driver_ok = False
    try:
        blackwell = g and float(g["compute"]) >= 12.0
    except ValueError:
        blackwell = False
    if g and driver_ok:
        info["recommended"] = "cuda"
        info["note"] = ("Série RTX 50 detectada: é o perfil testado." if blackwell else
                        f"Placa NVIDIA detectada ({g['name']}). O perfil de GPU usa pacotes CUDA 12.8, que não foram testados nesta placa.")
    elif g:
        info["recommended"] = "cpu"
        info["note"] = f"Placa NVIDIA detectada ({g['name']}), mas o driver {g['driver']} é anterior ao 570, que o CUDA 12.8 exige. Atualize o driver para usar a GPU."
    else:
        info["recommended"] = "cpu"
        info["note"] = "Nenhuma placa NVIDIA detectada. Placas AMD, Intel e Apple não são suportadas pela instalação automática."
    return info


# --------------------------------------------------------------------------- checks
def site_packages(venv: Path) -> Path | None:
    if P.IS_WIN:
        p = venv / "Lib" / "site-packages"
        return p if p.exists() else None
    for p in (venv / "lib").glob("python3*/site-packages"):
        return p
    return None


def has_pkg(venv: Path, *names: str) -> bool:
    sp = site_packages(venv)
    return bool(sp) and all((sp / n).exists() or (sp / (n + ".py")).exists() for n in names)


def uv_cmd() -> list[str] | None:
    u = shutil.which("uv")
    if u:
        return [u]
    py = P.venv_python(P.VENVS / "_tools")
    if py.exists() and has_pkg(P.VENVS / "_tools", "uv"):
        return [str(py), "-m", "uv"]
    return None


def chatterbox_files() -> list[Path]:
    d = P.ROOT / "models" / "chatterbox_ptbr"
    return [d / n for n in ("t3_pt_br.safetensors", "s3gen.pt", "ve.pt", "conds.pt", "grapheme_mtl_merged_expanded_v1.json")]


CHECKS = {
    "tools": lambda: uv_cmd() is not None and P.has_ffmpeg(),
    "venv_prep": lambda: has_pkg(P.VENVS / "prep", "faster_whisper", "soundfile", "num2words"),
    "rvc_src": lambda: (P.RVC / "infer" / "cli.py").exists() and (P.RVC / "train" / "train.py").exists(),
    "venv_rvc": lambda: has_pkg(P.RVC / ".venv", "torch", "faiss"),
    "rvc_models": lambda: all((P.RVC / p).exists() for p in (
        "assets/hubert_base/pytorch_model.bin", "assets/rmvpe/rmvpe.pt", "assets/pretrained_v2/f0G40k.pth",
        "assets/pretrained_v2/f0D40k.pth", "logs/mute")),
    "venv_chatterbox": lambda: has_pkg(P.VENVS / "chatterbox", "torch", "chatterbox", "perth"),
    "chatterbox_models": lambda: all(p.exists() for p in chatterbox_files()),
}

STEPS = [  # id, label, rough download size, needs RVC
    ("tools", "Ferramentas (uv e ffmpeg)", "~0,1 GB", False),
    ("venv_prep", "Preparo de dados (Whisper)", "~0,5 GB", False),
    ("rvc_src", "Código do RVC", "~0,05 GB", True),
    ("venv_rvc", "Ambiente do RVC (PyTorch)", "~5 GB", True),
    ("rvc_models", "Modelos base do RVC", "~1,5 GB", True),
    ("venv_chatterbox", "Ambiente do Chatterbox (PyTorch)", "~4 GB", False),
    ("chatterbox_models", "Modelo de voz pt-BR do Chatterbox", "~3,3 GB", False),
]


# --------------------------------------------------------------------------- the installer
class Setup:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.ctl: dict = {}
        self.states: dict[str, dict] = {}
        self.error: str | None = None
        self.running = False
        self.profile: str | None = None

    # ---- public
    def status(self) -> dict:
        s = P.read_settings()
        with_rvc = s.get("with_rvc", True)
        steps = []
        for sid, label, size, needs_rvc in STEPS:
            st = self.states.get(sid, {})
            done = bool(st.get("state") == "done" or (st.get("state") != "running" and CHECKS[sid]()))
            steps.append({"id": sid, "label": label, "size": size, "needs_rvc": needs_rvc,
                          "state": "running" if st.get("state") == "running" else "done" if done else st.get("state", "todo"),
                          "detail": st.get("detail", ""), "skipped": needs_rvc and not with_rvc and not done})
        required = [x for x in steps if not x["skipped"]]
        return {"running": self.running, "error": self.error, "steps": steps, "profile": s.get("profile"),
                "with_rvc": with_rvc, "ready": all(x["state"] == "done" for x in required),
                "profiles": {k: {"label": v["label"], "tested": v["tested"], "default_rvc": v["default_rvc"]} for k, v in PROFILES.items()},
                "hardware": detect_hardware()}

    def start(self, profile: str, with_rvc: bool) -> None:
        if profile not in PROFILES:
            raise ValueError("perfil inválido")
        with self.lock:
            if self.running:
                raise ValueError("A instalação já está em andamento.")
            self.running, self.error, self.profile = True, None, profile
            self.ctl = {"cancel": False, "proc": None}
            self.states = {}
        P.write_settings(profile=profile, with_rvc=bool(with_rvc))
        self.thread = threading.Thread(target=self._run, args=(profile, bool(with_rvc)), daemon=True)
        self.thread.start()

    def cancel(self) -> None:
        self.ctl["cancel"] = True
        proc = self.ctl.get("proc")
        if proc:
            kill_tree(proc)

    # ---- steps
    def _run(self, profile: str, with_rvc: bool) -> None:
        try:
            for sid, label, _size, needs_rvc in STEPS:
                if needs_rvc and not with_rvc:
                    continue
                if CHECKS[sid]():
                    self.states[sid] = {"state": "done", "detail": ""}
                    continue
                self.states[sid] = {"state": "running", "detail": "iniciando…"}
                getattr(self, "step_" + sid)(PROFILES[profile])
                if not CHECKS[sid]():
                    raise RuntimeError(f"O passo '{label}' terminou, mas faltam arquivos. Veja out/setup/{sid}.log")
                self.states[sid] = {"state": "done", "detail": ""}
        except Cancelled:
            self.error = "Instalação cancelada. Ao continuar, o que já foi instalado é reaproveitado."
            self._mark_failed("todo")
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            self._mark_failed("error")
        finally:
            self.running = False

    def _mark_failed(self, state: str) -> None:
        for sid, st in self.states.items():
            if st.get("state") == "running":
                self.states[sid] = {"state": state, "detail": self.error or ""}

    def _sh(self, sid: str, cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
        def on_line(t: str):
            self.states[sid]["detail"] = t[:140]
        rc = run_logged(cmd, OUT / f"{sid}.log", self.ctl, cwd=cwd, env=env, on_line=on_line)
        if rc != 0:
            raise RuntimeError(f"comando falhou (código {rc}). Veja out/setup/{sid}.log")

    def _detail(self, sid: str, t: str) -> None:
        self.states[sid]["detail"] = t

    def _make_venv(self, sid: str, venv: Path) -> Path:
        py = P.venv_python(venv)
        if py.exists():
            return py
        uv = uv_cmd()
        if not uv:
            raise RuntimeError("uv não encontrado; rode o passo de ferramentas primeiro.")
        self._sh(sid, [*uv, "venv", "--python", "3.12", "--seed", str(venv)])
        return py

    def _pip(self, sid: str, py: Path, *args: str) -> None:
        self._sh(sid, [str(py), "-m", "pip", "install", *args])

    def step_tools(self, prof: dict) -> None:
        if uv_cmd() is None or not P.has_ffmpeg():
            venv = P.VENVS / "_tools"
            if not P.venv_python(venv).exists():
                self._sh("tools", [sys.executable, "-m", "venv", str(venv)])
            py = P.venv_python(venv)
            pkgs = (["uv"] if uv_cmd() is None else []) + (["imageio-ffmpeg"] if not P.has_ffmpeg() else [])
            self._pip("tools", py, "--upgrade", "pip", *pkgs)
        if not P.has_ffmpeg():
            py = P.venv_python(P.VENVS / "_tools")
            r = subprocess.run([str(py), "-c", "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"],
                               capture_output=True, text=True)
            src = Path(r.stdout.strip().splitlines()[-1])
            P.TOOLS.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, P.TOOLS / P.exe("ffmpeg"))
        P.add_tools_to_path()

    def step_venv_prep(self, prof: dict) -> None:
        py = self._make_venv("venv_prep", P.VENVS / "prep")
        self._pip("venv_prep", py, "faster-whisper", "soundfile", "numpy", "num2words", *prof["prep_extra"])

    def _fetch_zip(self, sid: str, url: str, dest: Path, strip: bool = True) -> None:
        tmp = OUT / f"{sid}.zip"
        download(url, tmp, self.ctl, lambda t: self._detail(sid, t))
        self._detail(sid, "descompactando…")
        stage = OUT / f"{sid}_unzip"
        shutil.rmtree(stage, ignore_errors=True)
        with zipfile.ZipFile(tmp) as z:
            z.extractall(stage)
        top = next(stage.iterdir())
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(top), str(dest))
        shutil.rmtree(stage, ignore_errors=True)
        tmp.unlink(missing_ok=True)

    def step_rvc_src(self, prof: dict) -> None:
        self._fetch_zip("rvc_src", RVC_ZIP, P.RVC)

    def step_venv_rvc(self, prof: dict) -> None:
        py = self._make_venv("venv_rvc", P.RVC / ".venv")
        if prof["rvc_torch"]:
            self._pip("venv_rvc", py, *prof["rvc_torch"], "--index-url", prof["torch_index"], "--extra-index-url", PYPI)
        req = (P.RVC / prof["rvc_req"]).read_text(encoding="utf-8")
        req = req.replace("https://mirrors.pku.edu.cn/pypi/simple", PYPI)  # the file points at a Chinese mirror
        req = re.sub(r"https://mirrors\.nju\.edu\.cn/pytorch/whl/(\w+)", r"https://download.pytorch.org/whl/\1", req)
        tmp = OUT / "rvc_requirements.txt"
        tmp.write_text(req, encoding="utf-8")
        self._pip("venv_rvc", py, "-r", str(tmp))

    def step_rvc_models(self, prof: dict) -> None:
        script = OUT / "rvc_models.py"
        script.write_text('''import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
R = "lj1995/VoiceConversionWebUI"
snapshot_download(R, allow_patterns=["hubert_base/*", "pretrained_v2/f0G40k.pth", "pretrained_v2/f0D40k.pth"], local_dir="assets")
hf_hub_download(R, "rmvpe.pt", local_dir="assets/rmvpe")
z = hf_hub_download(R, "mute.zip", local_dir=".model-downloads")
zipfile.ZipFile(z).extractall("logs")
print("ok")
''', encoding="utf-8")
        self._sh("rvc_models", [str(P.venv_python(P.RVC / ".venv")), str(script)], cwd=P.RVC)

    def step_venv_chatterbox(self, prof: dict) -> None:
        py = self._make_venv("venv_chatterbox", P.VENVS / "chatterbox")
        self._pip("venv_chatterbox", py, *prof["torch_chatterbox"], "--index-url", prof["torch_index"], "--extra-index-url", PYPI)
        # Chatterbox pins torch 2.6.0, which has no RTX 50-series kernels: install it without its dependencies
        self._pip("venv_chatterbox", py, "--no-deps", CHATTERBOX_ZIP)
        self._pip("venv_chatterbox", py, *CHATTERBOX_DEPS)

    def step_chatterbox_models(self, prof: dict) -> None:
        self._sh("chatterbox_models", [str(P.venv_python(P.VENVS / "chatterbox")), "-W", "ignore",
                                       str(P.ROOT / "engines" / "chatterbox_server.py"), str(P.ROOT), "--download"])


SETUP = Setup()
