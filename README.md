# Voice clone pipeline

1. `01_RECORDING_CHECKLIST.md`: how to record your ~30 minutes.
2. `prepare_dataset.py`: raw recordings → 5–15 s clips + Whisper transcripts + lists for GPT-SoVITS, F5-TTS and RVC.
3. `02_TRAINING_GUIDE.md`: install and train on the RTX 5060 Ti (GPT-SoVITS v2Pro for English, Chatterbox + RVC for Portuguese, F5-TTS as the backup). Also covers licences.
4. `train_rvc.py`: trains the RVC model headless (slice, pitch, features, train, index).
5. `compare_checkpoints.py` + `test_sentences/`: renders the same test sentences from every saved checkpoint, so you pick the earliest good one and don't overtrain.
6. `narrate.py` + `voice_config.toml` + `engines/`: text in, narration file out. Use `--json` for pipelines.
7. `studio.py` + `ui/` (`studio.bat` / `studio.sh`): the app. Open http://127.0.0.1:7870. Two tabs:
   - **Estúdio**: type text, tune Chatterbox (pt-BR model) and RVC, compare with/without RVC, keep a history. Models stay loaded between generations.
   - **Clonar voz**: does everything from this guide by itself: detects your hardware, installs the environments and models, takes your recordings (any audio/video file), transcribes and cuts them, trains RVC (optional), ranks the checkpoints by similarity to your real voice and activates the result.
8. `portable.py`, `setup_manager.py`, `voices.py`: the code behind the second tab. Nothing in the repo depends on a fixed path.

## Start from a fresh clone (any PC)

```
git clone https://github.com/IsmarWolf/WolfsVoiceCloning
cd WolfsVoiceCloning
studio.bat            # Windows (needs Python 3.11+ on PATH; the app installs everything else)
./studio.sh           # Linux
```
Then open the **Clonar voz** tab, pick a profile and press *Instalar*.

| Profile | For | Status |
|---|---|---|
| `cuda` GPU NVIDIA (CUDA 12.8) | RTX 50-series (tested), other NVIDIA cards with driver 570+ | Tested end to end on an RTX 5060 Ti, Windows 11. Other NVIDIA cards use the same wheels but were **not tested**. |
| `cpu` | PCs without an NVIDIA GPU | The installer follows the vendor's CPU requirements, but it was **not tested on a real CPU-only PC**. Expect minutes per sentence and hours of RVC training; the app suggests Chatterbox only (no RVC). |

AMD, Intel and Apple GPUs are not supported by the automatic setup.

Quick use once trained:
```
python narrate.py --lang pt --text "Olá, pessoal!" --out vo.wav --json
```
