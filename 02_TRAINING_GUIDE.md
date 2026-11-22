# Training guide: your voice clone on an RTX 5060 Ti (16 GB, Windows)

**Plan in one line:** record once, prepare once, then train **GPT-SoVITS v2Pro** for English and **RVC** for Portuguese.
Portuguese speech comes from **Chatterbox Multilingual** (zero-shot, using your reference clip) and is passed through your RVC model.
**F5-TTS** is the backup: fine-tune it on the same clips and compare by ear.

Everything goes in one folder: `C:\Users\IsmarWolf\VoiceClone\`

```
VoiceClone\
  raw\            your recordings (session1_pt_01.wav, ...)
  dataset\        output of prepare_dataset.py
  pipeline\       prepare_dataset.py, narrate.py, voice_config.toml, engines\
  GPT-SoVITS\     git clone
  RVC\            git clone
  venvs\          chatterbox, f5tts, prep
```

> **RTX 50-series rule:** every environment needs **PyTorch built for CUDA 12.8 or newer** (`cu128`).
> Older builds fail with "no kernel image is available" or "sm_120 is not compatible".
> Check with: `python -c "import torch;print(torch.__version__, torch.cuda.get_arch_list())"`. The list must include `sm_120`.

---

## 0. Data prep (both languages)

```powershell
cd C:\Users\IsmarWolf\VoiceClone
python -m venv venvs\prep
venvs\prep\Scripts\pip install faster-whisper soundfile numpy num2words nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
venvs\prep\Scripts\python pipeline\prepare_dataset.py --input raw --output dataset --speaker ismar
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
cd C:\Users\IsmarWolf\VoiceClone
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
   - Text labelling file: `C:\Users\IsmarWolf\VoiceClone\dataset\gptsovits.list`
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

RVC doesn't need transcripts, so **all** your clips (PT + EN) go in. More data gives better timbre.

### Install
```powershell
cd C:\Users\IsmarWolf\VoiceClone
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git RVC
cd RVC
py -3.10 -m venv .venv          # RVC is happiest on Python 3.10 (install it from python.org if `py -3.10` is missing)
.venv\Scripts\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r requirements.txt
# then download the base models (hubert, rmvpe, pretrained_v2) as the RVC README says (tools/download_models.py)
```
If `requirements.txt` tries to replace torch with an older build, re-run the `cu128` torch line afterwards.
The RVC repo has a 50-series install thread: <https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/2745>.

### Train
`.venv\Scripts\python infer-web.py` → **Train** tab:
- Experiment name `ismar`, target SR **40k**, pitch guidance **yes**, version **v2**
- Training folder: `C:\Users\IsmarWolf\VoiceClone\dataset\wavs`
- f0 method **rmvpe_gpu**, batch size **12**, total epochs **100**, save every **10**
- "Save only the latest ckpt": **No**. "Save a small final model to the weights folder at each save point": **Yes**. That gives you `ismar_e10_….pth`, `ismar_e20_….pth` and so on in `assets\weights`.
- Click "One-click training". It also builds the `.index` file.
- With ~30 min of clean audio, RVC usually sounds right somewhere around epoch 40–80. Past that it tends to get harsh and lose the natural breath. Only go beyond 100 if e100 still clearly sounds unlike you.
- After Chatterbox is installed (step 3), compare every save point:
  ```powershell
  python compare_checkpoints.py rvc --exp ismar
  ```
  This renders the Portuguese test sentences once with Chatterbox, then passes them through each RVC checkpoint.

You know RVC already. The only differences from your earlier runs are the cu128 torch and the clean dataset from step 0.

---

## 3. Chatterbox Multilingual: Portuguese speech (MIT)

```powershell
cd C:\Users\IsmarWolf\VoiceClone
py -3.11 -m venv venvs\chatterbox
venvs\chatterbox\Scripts\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
venvs\chatterbox\Scripts\pip install chatterbox-tts --no-deps
venvs\chatterbox\Scripts\pip install "transformers==5.2.0" "diffusers==0.29.0" "librosa==0.11.0" soundfile safetensors conformer s3tokenizer "resemble-perth @ git+https://github.com/resemble-ai/Perth.git@master" pykakasi spacy-pkuseg
```
Why `--no-deps`: Chatterbox pins `torch==2.6.0`, which doesn't run on RTX 50-series. We keep the cu128 torch and install the rest by hand.
If an import error names a missing package, `pip install` it. The full list is in Chatterbox's `pyproject.toml`.

No training is needed. It clones from the reference clip, and RVC adds your exact timbre on top.
Tuning knobs in `voice_config.toml`: `exaggeration` (energy) and `cfg_weight` (pacing).
Resemble also published a Brazilian-Portuguese fine-tune (`ResembleAI/Chatterbox-Multilingual-pt-br`) that we can try if the accent from the multilingual model sounds off.

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
cd C:\Users\IsmarWolf\VoiceClone\pipeline
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
