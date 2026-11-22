#!/usr/bin/env python3
"""
Draws the README figures (dark, hand-drawn look) from docs/figures_data.json, which holds only numbers.
  pip install matplotlib numpy
  python docs/make_figures.py          -> docs/img/*.png
The hand-drawn font is "Patrick Hand" if installed; otherwise matplotlib's xkcd fallback is used.
"""
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / "img"
OUT.mkdir(exist_ok=True)
D = json.loads((HERE / "figures_data.json").read_text(encoding="utf-8"))

BG, FG = "#0d1017", "#f4f4f5"
CORAL, TEAL, YELLOW, LILAC, GREY = "#ff6b6b", "#4ecdc4", "#ffd166", "#b8a1ff", "#8b93a7"
for f in fm.findSystemFonts():
    if "patrickhand" in f.lower().replace(" ", "").replace("-", ""):
        fm.fontManager.addfont(f)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
FONT = "Patrick Hand" if any(f.name == "Patrick Hand" for f in fm.fontManager.ttflist) else None


def style():
    ctx = plt.xkcd(scale=0.8, length=120, randomness=1.6)
    ctx.__enter__()
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
                         "axes.labelcolor": FG, "axes.edgecolor": FG, "xtick.color": FG, "ytick.color": FG,
                         "path.effects": [], "lines.linewidth": 2.4})
    if FONT:
        plt.rcParams["font.family"] = [FONT]
    return ctx


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(2)


# ----------------------------------------------------------------------------- 1. flow
def flow():
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.6); ax.axis("off")

    def box(x, y, w, h, text, color, size=14):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc="none", ec=color, lw=2.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=FG, linespacing=1.15)

    def arrow(a, b, color=FG, rad=0.0):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=2.2, color=color, connectionstyle=f"arc3,rad={rad}"))

    ax.text(0.1, 6.25, "UMA VEZ: clonar a sua voz", fontsize=17, color=YELLOW)
    row1 = [("Sua gravação\náudio ou vídeo", YELLOW), ("Whisper\ntranscreve e corta", TEAL), ("Treino do RVC\ncheckpoints salvos", CORAL),
            ("Ranking por\nsimilaridade", TEAL), ("Voz ativa\nno estúdio", YELLOW)]
    xs = [0.2, 2.8, 5.4, 8.0, 10.6]
    for (t, c), x in zip(row1, xs):
        box(x, 4.35, 2.2, 1.35, t, c)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow((a + 2.25, 5.02), (b - 0.05, 5.02))
    ax.text(0.2, 5.8, "ideal: 20 a 30 min", fontsize=12, color=GREY)
    ax.text(5.4, 5.8, "~20 s por época na RTX 5060 Ti", fontsize=12, color=GREY)
    ax.text(8.0, 5.8, "o mais cedo entre os empatados", fontsize=12, color=GREY)

    ax.text(0.1, 3.2, "SEMPRE: narrar um texto", fontsize=17, color=CORAL)
    row2 = [("Seu texto", YELLOW), ("Chatterbox\npt-BR ou multilíngue", TEAL), ("RVC\nseu timbre", CORAL), ("EQ opcional\n+ volume -16 LUFS", TEAL), ("Áudio final\nWAV ou MP3", YELLOW)]
    for (t, c), x in zip(row2, xs):
        box(x, 1.2, 2.2, 1.35, t, c)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow((a + 2.25, 1.88), (b - 0.05, 1.88))
    # the active voice feeds the narration
    arrow((10.9, 4.3), (3.9, 2.62), color=YELLOW)
    ax.text(5.6, 3.72, "trecho de referência", fontsize=12.5, color=YELLOW, rotation=0)
    arrow((11.7, 4.3), (6.5, 2.62), color=YELLOW)
    ax.text(9.4, 3.05, "checkpoint escolhido", fontsize=12.5, color=YELLOW)
    ax.text(0.2, 0.45, "Tudo roda no seu computador, em http://127.0.0.1:7870. Nada sai da sua máquina.", fontsize=13, color=GREY)
    fig.savefig(OUT / "fluxo.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- 2. checkpoints
def checkpoints():
    c = D["checkpoints"]
    eps = sorted(int(k) for k in c["ckpt"])
    vals = {e: np.array(c["ckpt"][str(e)]) for e in eps}
    mean = np.array([vals[e].mean() for e in eps])
    fig, ax = plt.subplots(figsize=(11, 5.6))
    clean(ax)
    for e in eps:
        ax.plot([e] * len(vals[e]), vals[e], "o", color=TEAL, alpha=0.35, ms=6)
    ax.plot(eps, mean, color=CORAL, lw=3.2, label="média de 5 frases")
    ax.plot([], [], "o", color=TEAL, alpha=0.5, label="cada frase")
    real = np.mean(c["real_holdout"])
    ax.axhline(real, color=FG, ls=":", lw=2)
    ax.text(101, real + 0.004, "gravações reais (média)", ha="right", fontsize=13, color=FG)
    ax.axhline(c["base_voice_without_rvc"], color=GREY, ls=":", lw=2)
    ax.text(101, c["base_voice_without_rvc"] + 0.006, "voz base sem RVC", ha="right", fontsize=13, color=GREY)
    best = int(np.argmax(mean))
    ax.annotate(f"melhor média: época {eps[best]}\n({mean[best]:.3f})", xy=(eps[best], mean[best]), xytext=(eps[best] + 6, 0.925),
                fontsize=14, color=FG, arrowprops=dict(arrowstyle="->", lw=2, color=FG))
    ax.annotate(f"época 100: {mean[-1]:.3f}\n(a escolhida por ouvido)", xy=(100, mean[-1]), xytext=(70, 0.79),
                fontsize=14, color=FG, arrowprops=dict(arrowstyle="->", lw=2, color=FG))
    ax.set_xlim(0, 104); ax.set_ylim(0.68, 1.0)
    ax.set_xlabel("épocas de treino do RVC", fontsize=15); ax.set_ylabel("semelhança com a sua voz (0 a 1)", fontsize=15)
    ax.set_xticks(eps)
    ax.tick_params(labelsize=13)
    ax.legend(loc="lower left", frameon=False, fontsize=13, bbox_to_anchor=(0.0, 0.12))
    ax.set_title("Mais épocas não deixam a voz mais parecida", fontsize=18, color=YELLOW, pad=14)
    fig.savefig(OUT / "checkpoints.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- 3. presets
def presets():
    bars = D["presets"]["bars"]
    cols = [YELLOW, TEAL, CORAL, LILAC]
    real_sim = np.mean(D["checkpoints"]["real_holdout"]) * 100
    real_low = np.mean(D["checkpoints"]["low_real"])
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    clean(ax)
    w, step = 0.2, 0.23
    groups = [("semelhança com a sua voz\n(×100)", [b["similarity"] * 100 for b in bars], real_sim, 0.0),
              ("energia abaixo de 250 Hz\n(% do espectro)", [b["low_share"] for b in bars], real_low, 1.35)]
    for name, vals, ref, x0 in groups:
        for i, v in enumerate(vals):
            ax.bar(x0 + i * step, v, w, color=cols[i], ec=FG, lw=1.8)
            ax.text(x0 + i * step, v - 3.4, f"{v:.1f}", ha="center", fontsize=13, color=BG)
        ax.plot([x0 - 0.2, x0 + 3 * step + 0.2], [ref, ref], color=FG, ls=":", lw=2.6)
        ax.text(x0 + 3 * step + 0.2, ref + 2, "gravações reais", ha="right", fontsize=12.5, color=FG)
        ax.text(x0 + 1.5 * step, -6, name, ha="center", fontsize=14.5, va="top")
    ax.set_xticks([]); ax.set_ylim(0, 108); ax.set_xlim(-0.35, 2.3)
    ax.set_ylabel("%", fontsize=15, rotation=0, labelpad=14); ax.tick_params(labelsize=13)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec=FG) for c in cols]
    ax.legend(handles, [b["label"].replace("\n", " ") for b in bars], frameon=False, fontsize=13, loc="lower center",
              ncol=4, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=1.4)
    fig.savefig(OUT / "presets.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- 4. profiles
def profiles():
    fig, ax = plt.subplots(figsize=(13, 4.3))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.3); ax.axis("off")
    items = [("GPU NVIDIA série RTX 50", "CUDA 12.8", "TESTADO\nde ponta a ponta", TEAL),
             ("Outras GPUs NVIDIA", "driver 570 ou mais novo", "mesmos pacotes\nNÃO testado", YELLOW),
             ("Somente CPU", "sem placa NVIDIA", "instalador pronto\nNÃO testado, lento", YELLOW),
             ("AMD, Intel, Apple", "GPU", "sem suporte na\ninstalação automática", CORAL)]
    for i, (a, b, c, col) in enumerate(items):
        x = 0.2 + i * 3.2
        ax.add_patch(FancyBboxPatch((x, 0.5), 2.9, 3.0, boxstyle="round,pad=0.02,rounding_size=0.15", fc="none", ec=col, lw=2.8))
        ax.text(x + 1.45, 2.95, a, ha="center", fontsize=15, color=FG)
        ax.text(x + 1.45, 2.4, b, ha="center", fontsize=12.5, color=GREY)
        ax.text(x + 1.45, 1.3, c, ha="center", va="center", fontsize=15, color=col, linespacing=1.2)
    ax.text(0.2, 3.95, "O que foi (e o que não foi) testado", fontsize=18, color=YELLOW)
    fig.savefig(OUT / "perfis.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ctx = style()
    flow(); checkpoints(); presets(); profiles()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
