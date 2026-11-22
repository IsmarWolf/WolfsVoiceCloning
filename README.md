# Voice clone pipeline

1. `01_RECORDING_CHECKLIST.md`: how to record your ~30 minutes.
2. `prepare_dataset.py`: raw recordings → 5–15 s clips + Whisper transcripts + lists for GPT-SoVITS, F5-TTS and RVC.
3. `02_TRAINING_GUIDE.md`: install and train on the RTX 5060 Ti (GPT-SoVITS v2Pro for English, Chatterbox + RVC for Portuguese, F5-TTS as the backup). Also covers licences.
4. `narrate.py` + `voice_config.toml` + `engines/`: text in, narration file out. Use `--json` for pipelines.

Quick use once trained:
```
python narrate.py --lang pt --text "Olá, pessoal!" --out vo.wav --json
```
