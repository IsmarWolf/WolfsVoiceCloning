#!/usr/bin/env python3
"""
portable.py - where things live, independent of the PC. Standard library only.

  ROOT                 the folder this file is in (the repo clone)
  venvs/<name>         Python environments created by the app (prep, chatterbox)
  RVC/                 the RVC source + its own .venv
  voices/<name>/       one folder per cloned voice (raw recordings, dataset, picks, voice.json)
  settings.json        what the app remembers (hardware profile, active voice)   - git-ignored
  tools/               ffmpeg downloaded by the app when the PC has none         - git-ignored

load_config() reads voice_config.toml, fills every path that is left empty and applies the
active voice, so narrate.py and studio.py work the same on any machine.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENVS = ROOT / "venvs"
RVC = ROOT / "RVC"
TOOLS = ROOT / "tools"
VOICES = ROOT / "voices"
SETTINGS = ROOT / "settings.json"
CONFIG = Path(os.environ.get("VOICE_CONFIG", ROOT / "voice_config.toml"))
IS_WIN = os.name == "nt"


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")


def exe(name: str) -> str:
    return name + (".exe" if IS_WIN else "")


# --------------------------------------------------------------------------- settings
def read_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_settings(**changes) -> dict:
    s = {**read_settings(), **changes}
    SETTINGS.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    return s


# --------------------------------------------------------------------------- ffmpeg
def ffmpeg_bin() -> str:
    local = TOOLS / exe("ffmpeg")
    if local.exists():
        return str(local)
    return shutil.which("ffmpeg") or "ffmpeg"


def has_ffmpeg() -> bool:
    return (TOOLS / exe("ffmpeg")).exists() or shutil.which("ffmpeg") is not None


def add_tools_to_path() -> None:
    """Make the downloaded ffmpeg visible to every child process."""
    if TOOLS.exists():
        os.environ["PATH"] = str(TOOLS) + os.pathsep + os.environ.get("PATH", "")


# --------------------------------------------------------------------------- voices
def slug_name(name: str) -> str:
    """Folder / RVC experiment name: lowercase ascii letters, digits and dashes."""
    import unicodedata

    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:24].strip("-") or "voz"


def voice_dir(name: str) -> Path:
    return VOICES / name


def read_voice(name: str) -> dict | None:
    try:
        return json.loads((voice_dir(name) / "voice.json").read_text(encoding="utf-8"))
    except Exception:
        return None


def write_voice(v: dict) -> None:
    d = voice_dir(v["name"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "voice.json").write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")


def list_voices() -> list[dict]:
    out = []
    if VOICES.exists():
        for d in sorted(VOICES.iterdir()):
            v = read_voice(d.name) if d.is_dir() else None
            if v:
                out.append(v)
    return out


def voice_dataset(v: dict | None) -> Path:
    """Dataset folder (with wavs/ and metadata.csv) of a voice. Old installs used ROOT/dataset."""
    if v and v.get("dataset"):
        p = Path(v["dataset"])
        return p if p.is_absolute() else ROOT / p
    return ROOT / "dataset"


def active_voice() -> dict | None:
    name = read_settings().get("active_voice")
    return read_voice(name) if name else None


# --------------------------------------------------------------------------- config
def _abs(v: str | None, default: Path) -> str:
    if not v:
        return str(default)
    p = Path(v)
    return str(p if p.is_absolute() else ROOT / p)


def load_config(path: Path | None = None) -> dict:
    cfg = tomllib.loads((path or CONFIG).read_text(encoding="utf-8"))
    s = read_settings()
    g = cfg.setdefault("general", {})
    g["ffmpeg"] = ffmpeg_bin() if g.get("ffmpeg", "ffmpeg") == "ffmpeg" else g["ffmpeg"]

    rv = cfg.setdefault("rvc", {})
    rv["cwd"] = _abs(rv.get("cwd"), RVC)
    rv["python"] = _abs(rv.get("python"), venv_python(Path(rv["cwd"]) / ".venv"))
    rv.setdefault("f0method", "rmvpe")
    rv.setdefault("index_rate", 0.5)
    rv.setdefault("protect", 0.33)
    rv.setdefault("command", ["{python}", "-W", "ignore", "-m", "infer.cli", "--model", "{model_name}", "--input", "{input}",
                              "--output", "{output}", "--index", "{index_path}", "--index-rate", "{index_rate}",
                              "--f0-method", "{f0method}", "--protect", "{protect}", "--overwrite"])

    cb = cfg.setdefault("chatterbox", {})
    cb["python"] = _abs(cb.get("python"), venv_python(VENVS / "chatterbox"))
    cb.setdefault("exaggeration", 0.5)
    cb.setdefault("cfg_weight", 0.5)
    profile = s.get("profile")
    cb["device"] = cb.get("device") or ("cpu" if profile == "cpu" else "cuda")

    # the active voice decides the model, the index and the reference clip
    v = active_voice()
    if v:
        ds = voice_dataset(v)
        if v.get("ref"):
            cb["ref_audio"] = str(ds / "wavs" / v["ref"])
        r = v.get("rvc")
        if r and r.get("model"):
            rv["model_name"] = r["model"]
            idx = r.get("index")
            rv["index_path"] = str(Path(rv["cwd"]) / "assets" / "indices" / idx) if idx else ""
            cb["rvc"] = True
        else:
            rv["model_name"] = ""
            rv["index_path"] = ""
            cb["rvc"] = False
    return cfg


if __name__ == "__main__":
    print(json.dumps(load_config(), indent=1, ensure_ascii=False))
