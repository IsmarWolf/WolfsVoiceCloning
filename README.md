# Voice clone pipeline

1. `01_RECORDING_CHECKLIST.md`: how to record your ~30 minutes.
2. `prepare_dataset.py`: raw recordings → 5–15 s clips + Whisper transcripts + lists for GPT-SoVITS, F5-TTS and RVC.
3. `02_TRAINING_GUIDE.md`: install and train on the RTX 5060 Ti (GPT-SoVITS v2Pro for English, Chatterbox + RVC for Portuguese, F5-TTS as the backup). Also covers licences.
4. `train_rvc.py`: trains the RVC model headless (slice, pitch, features, train, index).
5. `compare_checkpoints.py` + `test_sentences/`: renders the same test sentences from every saved checkpoint, so you pick the earliest good one and don't overtrain.
6. `narrate.py` + `voice_config.toml` + `engines/`: text in, narration file out. Use `--json` for pipelines.
7. `studio.py` + `ui/` (`studio.bat`): visual interface at http://127.0.0.1:7870 to type text, tune Chatterbox (pt-BR model) and RVC, compare with/without RVC, and keep a history. Models stay loaded between generations.

Quick use once trained:
```
python narrate.py --lang pt --text "Olá, pessoal!" --out vo.wav --json
```
