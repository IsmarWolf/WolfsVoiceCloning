# Training guide: your voice clone on an RTX 5060 Ti (16 GB, Windows)

> **You no longer have to follow this guide by hand.** `studio.bat` opens the app, and its **Clonar voz** tab installs the environments, prepares your recordings, trains RVC and picks the checkpoint (see README.md for the hardware profiles and what was tested). This guide stays as the reference for what the app does, and for the optional GPT-SoVITS / F5-TTS routes, which the app does not set up.

**Plan in one line:** record once, prepare once, then train **GPT-SoVITS v2Pro** for English and **RVC** for Portuguese.
Portuguese speech comes from **Chatterbox Multilingual** (zero-shot, using your reference clip) and is passed through your RVC model.
**F5-TTS** is the backup: fine-tune it on the same clips and compare by ear.

Everything goes in one folder, which is also the git repo: `C:\Users\IsmarWolf\Projetos\WolfsVoiceCloning\`

```
WolfsVoiceCloning\
  raw\            your recordings (Treino_pt_01.wav, ...)       (git-ignored)
  dataset\        output of prepare_dataset.py                  (git-ignored)
  RVC\            git clone of the RVC repo, with its own .venv  (git-ignored)
  venvs\          prep, chatterbox, f5tts                       (git-ignored)
  GPT-SoVITS\     only if you record English                    (git-ignored)
  prepare_dataset.py  train_rvc.py  compare_checkpoints.py  narrate.py  voice_config.toml  engines\
```

> **RTX 50-series rule:** every environment needs **PyTorch built for CUDA 12.8 or newer** (`cu128`).
> Older builds fail with "no kernel image is available" or "sm_120 is not compatible".
> Check with: `python -c "import torch;print(torch.__version__, torch.cuda.get_arch_list())"`. The list must include `sm_120`.

---

## 0. Data prep (both languages)

```powershell
cd C:\Users\IsmarWolf\Projetos\WolfsVoiceCloning
python -m venv venvs\prep
venvs\prep\Scripts\pip install faster-whisper soundfile numpy num2words nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
venvs\prep\Scripts\python prepare_dataset.py --input raw --output dataset --speaker ismar
```

- About 30 minutes of audio takes roughly 3–6 minutes on the GPU. If CUDA fails, it falls back to CPU, which is slower but still works.
- **Check `dataset\review.csv`.** Listen to the low-confidence clips and fix the text in `gptsovits.list` / `f5tts\metadata.csv`, or delete the line. Ten minutes spent here gives a noticeably better model.
- Outputs: `wavs\` (for RVC), `gptsovits.list` (English lines only), `f5tts\metadata_pt.csv` and `metadata_en.csv`.
- **Pick reference clips now.** Choose one clean **4–9 s** clip per language that sounds like your normal narration (not a question, not a laugh). Put its path and exact text in `voice_config.toml`.

---

## 1. GPT-SoVITS v2Pro: English (MIT)

### Install (about 10 GB)
```powershell
# Miniconda first if you don't have it: https://docs.conda.io/en/latest/miniconda.html
cd C:\Users\IsmarWolf\Projetos\WolfsVoiceCloning
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
conda create -n GPTSoVits python=3.11 -y
conda activate GPTSoVits
.\install.ps1 -Device CU128 -Source HF
```
- Don't use the prebuilt Windows `.7z` package. It ships an older PyTorch without 50-series support.
- `install.ps1` downloads the pretrained models (v2Pro included) and installs the `cu128` PyTorch.

### Train
```powershell
conda activate GPTSoVits
python webui.py en_US
```
In the browser (http://127.0.0.1:9874):
1. Top of the page, **version: `v2Pro`**. Use `v2ProPlus` if VRAM allows. It's a bit better and still fits in 16 GB.
2. Tab **1-GPT-SoVITS-TTS → 1A Dataset formatting**
   - Experiment name: `ismar`
   - Text labelling file: `C:\Users\IsmarWolf\Projetos\WolfsVoiceCloning\dataset\gptsovits.list`
   - Audio folder: leave **empty** (the list has full paths)
   - Click **One-click formatting** and wait for "done".
3. **1B Fine-tuning**
   - **Untick "save only the latest checkpoint"** and **tick "save a small final model to the weights folder at each save point"**. Otherwise you can't go back to an earlier epoch.
   - **SoVITS training:** batch size **8**, total epochs **8**, save every **2** epochs. Click start.
   - **GPT training:** batch size **8**, total epochs **15**, save every **5** epochs. Leave DPO off. Click start.
   - Roughly 10–25 min each for 10–15 min of English audio. These are the project's moderate defaults. Don't raise them unless every checkpoint still sounds unlike you.
4. **Pick checkpoints by ear (see [Avoiding overtraining](#avoiding-overtraining)):**
   ```powershell
   python compare_checkpoints.py gptsovits --exp ismar
   ```
   This renders the same 5 test sentences from every SoVITS × GPT pair into `out\checkpoints\gptsovits\`.

Then copy the chosen weight paths into `[gptsovits]` in `voice_config.toml`. `narrate.py` starts `api_v2.py` itself when needed.

---

## 2. RVC: your timbre, for Portuguese (MIT)

RVC doesn't need transcripts, so **all** your clips go in. More data gives better timbre.
The current RVC repo targets **Python 3.12 + PyTorch 2.7.1 cu128**, which is exactly what an RTX 50-series card needs.
(Older guides mention `infer-web.py`, `tools/infer_cli.py` and Python 3.10. Those no longer exist.)

### Install (about 8 GB, inside the project folder)
```powershell
cd C:\Users\IsmarWolf\Projetos\WolfsVoiceCloning
git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git RVC
cd RVC
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python -m pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
# the requirements file points at a Chinese mirror; use the official index instead
(Get-Content requirments_cu128_py312.txt) -replace 'https://mirrors.pku.edu.cn/pypi/simple','https://pypi.org/simple' | Set-Content $env:TEMP\rvc_req.txt
.venv\Scripts\python -m pip install -r $env:TEMP\rvc_req.txt
```
Then download the base models (hubert, rmvpe, the v2 40k pretrained pair, the silence samples) from `lj1995/VoiceConversionWebUI` into `RVC\assets` and `RVC\logs\mute` as the RVC README says.

### Train (one command, no clicking)
```powershell
cd C:\Users\IsmarWolf\Projetos\WolfsVoiceCloning
RVC\.venv\Scripts\python.exe train_rvc.py --exp ismar --epochs 100 --save-every 10
```
`train_rvc.py` runs the same steps as the WebUI's Train tab: slice, pitch (rmvpe), HuBERT features, train, retrieval index.
- Sample rate 40k, v2, batch 12. Every save point is kept as `RVC\assets\weights\ismar_e<epoch>_s<step>.pth`.
- About 20 s per epoch on an RTX 5060 Ti with ~33 min of audio, so ~35 min for 100 epochs.
- The index is written to `RVC\assets\indices\`.
- Resume or redo one step: `--only train`, or `--skip preprocess f0 features`.
- With ~30 min of clean audio RVC usually sounds right around epoch 40–80. Past that it tends to get harsh and lose the natural breath.

### Pick the checkpoint
```powershell
python compare_checkpoints.py rvc --exp ismar
```
This renders the Portuguese test sentences once with the base voice, then passes them through each RVC checkpoint (see "Avoiding overtraining" below).

### Convert a file with RVC by hand
```powershell
cd RVC
.venv\Scripts\python -m infer.cli --model ismar_e60_s2520.pth --input C:\path\in.wav --output C:\path\out.wav --index assets\indices\<your .index> --index-rate 0.5
```
Use absolute paths for `--input` and `--output`: the CLI changes into the RVC folder first.

---

## 3. Chatterbox Multilingual: Portuguese speech (MIT)

No training is needed. It clones the voice from one reference clip, and RVC adds your exact timbre on top.
Tuning knobs in `voice_config.toml`: `exaggeration` (energy) and `cfg_weight` (pacing).

This is the install that worked on the RTX 5060 Ti (about 4 GB plus ~3 GB of model weights on first run):
```powershell
cd C:\Users\IsmarWolf\Projetos\WolfsVoiceCloning
py -3.12 -m venv venvs\chatterbox
venvs\chatterbox\Scripts\python -m pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
# chatterbox pins torch 2.6.0, which has no RTX 50-series kernels, so install it without its dependencies...
venvs\chatterbox\Scripts\python -m pip install --no-deps git+https://github.com/resemble-ai/chatterbox.git
# ...and add the rest by hand
venvs\chatterbox\Scripts\python -m pip install "numpy<2" librosa==0.11.0 s3tokenizer transformers==5.2.0 diffusers==0.29.0 resemble-perth conformer==0.3.2 safetensors==0.5.3 spacy-pkuseg pykakasi==2.3.0 pyloudnorm omegaconf soundfile "setuptools<81"
```
- Use the GitHub version, not PyPI's `chatterbox-tts` 0.1.7: only the GitHub one has the newer `t3_model="v3"` multilingual model.
- `setuptools<81` matters: without it, `resemble-perth` fails to load its watermarker (`'NoneType' object is not callable`).
- pip will warn that chatterbox wants torch 2.6.0 and gradio. Ignore both.
- First generation downloads the model weights from Hugging Face.

---

## 4. F5-TTS: backup to compare (code MIT, **weights CC-BY-NC**)

Fine-tune it on your Portuguese clips, starting from a pt-BR checkpoint such as `firstpixel/F5-TTS-pt-br`:
```powershell
py -3.11 -m venv venvs\f5tts
venvs\f5tts\Scripts\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
venvs\f5tts\Scripts\pip install f5-tts
venvs\f5tts\Scripts\f5-tts_finetune-gradio
```
In the Gradio UI: create a project, point it at `dataset\f5tts\metadata_pt.csv` (pipe-separated, absolute paths, header `audio_file|text`), and run **Prepare**.
On **Train**, choose "finetune", set the pretrained checkpoint to the pt-BR model, and use learning rate `1e-5`, frame batch `~3000` for 16 GB, **30 epochs max**, and a save every few hundred updates (keep the last 10 or more). Turn off EMA for early checkpoints.
Compare them with `python compare_checkpoints.py f5tts --ckpt-dir <its ckpts folder>`.
Then set `[routing] pt = "f5tts"` (or run `narrate.py --engine f5tts`) to compare against Chatterbox + RVC.
I'll fill in the exact clicks when we get there, because the UI changes between versions.

---

## Avoiding overtraining

The rule: **train a moderate amount, save often, and keep the earliest checkpoint that sounds like you.**

| Engine | Total epochs | Save every | Usually good around |
|---|---|---|---|
| GPT-SoVITS SoVITS | 8 | 2 | 4–8 |
| GPT-SoVITS GPT | 15 | 5 | 10–15 |
| RVC | 100 | 10 | 40–80 |
| F5-TTS fine-tune | ≤30 | few hundred updates | early-middle |

`compare_checkpoints.py` renders the same fixed test sentences (`test_sentences\pt.txt` and `en.txt`) from every checkpoint. Each checkpoint gets one file with all the sentences, named so they sort oldest first. Listen in that order, with headphones, at the same volume.

- **Undertrained:** sounds like a generic voice, or only somewhat like you. Keep going to later checkpoints.
- **Right:** sounds like you, words are clear, and rhythm varies naturally. **Stop here and use this one**, even if later ones exist.
- **Overtrained:** metallic or harsh tone, words skipped or repeated, mumbling, odd breaths, or every sentence with the same sing-song rhythm. Go back one or two checkpoints.

If two checkpoints sound the same, take the earlier one. It generalises better to new scripts.
Use `--every 2` to listen to fewer checkpoints at first, then fill in around the best one.

---

## 5. Generate narration

```powershell
cd C:\Users\IsmarWolf\Projetos\WolfsVoiceCloning
python narrate.py --lang pt --text "Você sabia que 90% das pessoas fazem isso errado?" --out C:\Users\IsmarWolf\Videos\vo\hook.wav
python narrate.py --lang en --text-file script_en.txt --out vo_en.mp3 --json
python narrate.py --lang pt --engine f5tts --text "..." --out compare_f5.wav      # A/B test
```
`--json` prints `{"path": ..., "duration": ..., "engine": ...}` so a video pipeline (or Claude) can read the result.
The output is loudness-normalised to −16 LUFS and has a 250 ms gap between sentences. Both are set in `[general]`.

---

## Licences: what you can and can't do

| Part | Licence | Monetised Instagram OK? |
|---|---|---|
| GPT-SoVITS code + pretrained models | MIT | ✅ |
| RVC WebUI | MIT | ✅ |
| Chatterbox (code + weights) | MIT | ✅ Every output carries an inaudible Perth watermark that marks it as AI-generated. That's fine for narration, but don't try to remove it. |
| faster-whisper / Whisper | MIT | ✅ (only used for data prep) |
| F5-TTS weights (and pt-BR fine-tunes built on them) | **CC-BY-NC 4.0** | ❌ Not for monetised or sponsored content. Fine for testing and personal use. |
| XTTS-v2 (not used) | Coqui Public Model Licence, non-commercial. Coqui shut down, so no commercial licence can be bought. | ❌ |

It's your own voice, so consent isn't an issue. Instagram and TikTok ask you to label realistic AI-generated audio, so tick the "AI info" toggle when you post.
