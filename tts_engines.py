#!/usr/bin/env python3
"""
tts_engines.py - the list of base voices (TTS engines) the studio can use. Standard library only.

Two lists:
  MODELS   what the "Voz base" menu can select. Each one runs in its own venv (venvs/<engine>) and speaks
           the same JSON-lines protocol as engines/chatterbox_server.py. The RVC pass sits on top of any of them.
  CATALOG  engines that were checked but are NOT offered, with the reason (licence or no Portuguese).

The licence and language facts were read from each project's own LICENSE and README (see 02_TRAINING_GUIDE.md).
"tested" is only "sim" for what was run end to end on the author's PC; everything else says so.
Nothing here is legal advice: read the licence before you monetise.
"""
from __future__ import annotations

# commercial: "sim" | "limitado" | "não"
MODELS = {
    "ptbr": {
        "engine": "chatterbox", "label": "Chatterbox · português do Brasil (modelo dedicado)", "langs": ["pt"],
        "license": "MIT", "commercial": "sim", "pt_br": "Sim, modelo treinado para pt-BR",
        "tested": "sim", "max_chars": 250, "size": "~3,3 GB",
        "note": "Voz base padrão.",
    },
    "v3": {
        "engine": "chatterbox", "label": "Chatterbox · multilíngue v3 (23 idiomas)", "langs": ["pt", "en"],
        "license": "MIT", "commercial": "sim", "pt_br": "Sim, entre os 23 idiomas",
        "tested": "sim", "max_chars": 250, "size": "baixa na 1ª vez",
        "note": "É o que o inglês usa.",
    },
    "xtts": {
        "engine": "xtts", "label": "XTTS-v2 (Coqui)", "langs": ["pt", "en"],
        "license": "Coqui Public Model License (CPML)", "commercial": "não",
        "pt_br": "Sim, português (17 idiomas)", "tested": "tecnico",
        "max_chars": 190, "size": "~2 GB",
        "note": "A licença não permite uso comercial e a Coqui encerrou as atividades (não dá para comprar outra). Ok para uso pessoal.",
    },
    "f5ptbr": {
        "engine": "f5", "label": "F5-TTS · pt-BR (ajuste da comunidade)", "langs": ["pt"],
        "license": "CC-BY-NC 4.0 (pesos); código MIT", "commercial": "não",
        "pt_br": "Sim, só português do Brasil", "tested": "tecnico",
        "max_chars": 180, "size": "~3 GB",
        "note": "Os pesos base do F5-TTS são não comerciais (dados Emilia) e o ajuste pt-BR mantém CC-BY-NC. Precisa da transcrição do trecho de referência; o app usa a da gravação enviada, quando existe. Só pt.",
    },
}

ENGINE_LABELS = {"chatterbox": "Chatterbox", "xtts": "XTTS-v2", "f5": "F5-TTS"}

CATALOG = [
    {"label": "Fish Audio S2-Pro (4B)", "license": "Fish Audio Research License (não comercial)",
     "commercial": "não", "pt_br": "Cerca de 100 idiomas, português incluído (pt-BR não verificado)",
     "why": "Candidato: cabe numa placa de 16 GB e a licença serve para uso pessoal. Ainda não integrado: precisa do repositório fish-speech e do servidor próprio, e a instalação no Windows não foi validada. O Drama 3 (35B) não cabe em 16 GB."},
    {"label": "Higgs Audio v3 (4B)", "license": "Boson Higgs Audio v3 Research and Non-Commercial License",
     "commercial": "não", "pt_br": "Cita mais de 100 idiomas (pt-BR não verificado)",
     "why": "Candidato: licença serve para uso pessoal. Ainda não integrado: o servidor recomendado (SGLang-Omni) é voltado a Linux e não foi validado no Windows. O v2 é focado em en, zh, de e ko."},
    {"label": "IndexTTS 2 / 2.5", "license": "bilibili Model Use License",
     "commercial": "sim", "pt_br": "Não: chinês, inglês, japonês, espanhol e árabe",
     "why": "Não fala português."},
    {"label": "CosyVoice 3 (Fun-CosyVoice3)", "license": "Apache 2.0 (código)",
     "commercial": "sim", "pt_br": "Não na lista oficial (zh, en, ja, ko, de, es, fr, it, ru)",
     "why": "Português não está entre os idiomas suportados. Não foi validado se o modo entre-idiomas serve."},
]

ENGINE_PY_NAMES = {"xtts": "xtts", "f5": "f5"}


def model_info(key: str) -> dict | None:
    return MODELS.get(key)


def models_for(lang: str) -> list[str]:
    return [k for k, v in MODELS.items() if lang in v["langs"]]
