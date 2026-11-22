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
    "index25": {
        "engine": "index", "label": "IndexTTS 2.5 (bilibili)", "langs": ["en"],
        "license": "bilibili Model Use License", "commercial": "sim",
        "pt_br": "Não fala português (zh, en, ja, es, ar): só funciona para inglês no estúdio", "tested": "não",
        "max_chars": 200, "size": "~12 GB",
        "note": "Instalador escrito a partir do README do projeto, NUNCA executado. Pede Python 3.11 e PyTorch 2.8; se falhar, o log fica em out/setup. Licença permite uso comercial abaixo de 100 milhões de usuários ativos ou RMB 1 bilhão de faturamento.",
    },
}

ENGINE_LABELS = {"chatterbox": "Chatterbox", "xtts": "XTTS-v2", "f5": "F5-TTS", "index": "IndexTTS 2.5"}

CATALOG = [
    {"label": "Higgs Audio v3 TTS (4B, Boson AI)", "license": "Boson Higgs TTS 3 Research and Non-Commercial License, com Creator Use Grant",
     "commercial": "limitado", "pt_br": "Sim: português está entre os 85 idiomas de nível mais alto",
     "status": "Sem instalação automática (Linux)",
     "why": "Cabe na placa de 16 GB (4B em bf16). A licença libera criadores de conteúdo, inclusive monetizado, de graça, desde que credite 'Boson AI Higgs Audio' no áudio ou na descrição; proíbe clonar voz sem consentimento. O servidor oficial (SGLang-Omni ou vLLM-Omni) só tem receita para Linux/Docker, então o app não instala no Windows. Via WSL2 ou Docker, dá para subir o servidor e usar o endpoint /v1/audio/speech (ainda não ligado ao estúdio)."},
    {"label": "Fish Audio S2-Pro (4B)", "license": "Fish Audio Research License (não comercial)",
     "commercial": "não", "pt_br": "Cerca de 100 idiomas, português incluído (pt-BR não verificado)",
     "status": "Sem instalação automática (24 GB, Linux/WSL)",
     "why": "A documentação oficial pede pelo menos 24 GB de GPU e Linux ou WSL. Sua placa tem 16 GB, então não é viável de forma suportada. O Drama 3 (35B) pede 48 GB."},
    {"label": "CosyVoice 3 (Fun-CosyVoice3)", "license": "Apache 2.0 (código)",
     "commercial": "sim", "pt_br": "Não: zh, en, ja, ko, de, es, fr, it, ru",
     "status": "Sem instalação automática (sem português, PyTorch antigo)",
     "why": "O repositório fixa PyTorch 2.3.1, que não tem kernels para as placas RTX 50, e a instalação oficial é Conda no Linux. Também não lista português."},
]

ENGINE_PY_NAMES = {"xtts": "xtts", "f5": "f5"}


def model_info(key: str) -> dict | None:
    return MODELS.get(key)


def models_for(lang: str) -> list[str]:
    return [k for k, v in MODELS.items() if lang in v["langs"]]
