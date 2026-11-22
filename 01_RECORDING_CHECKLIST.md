# Recording checklist (about 30 minutes of clean audio)

The model copies everything it hears, including room echo, hum and how tired you sound.
Clean, consistent input matters more than extra minutes.

## Gear
- [ ] **Mic:** a USB condenser or dynamic mic (e.g. Fifine, Samson Q2U, AT2020USB) beats any headset or laptop mic. Use the same mic for every session.
- [ ] **Distance:** about 15–20 cm (a hand's width plus a bit). Speak slightly off-axis or use a pop filter, so "p" and "b" sounds don't hit the mic.
- [ ] **Format:** WAV, 48 kHz (44.1 kHz is fine), 24-bit or 16-bit, **mono**. No MP3 or OGG.
- [ ] **Level:** peaks around −12 to −6 dBFS. It must never clip (hit 0 dB). If it clips, re-record. Clipping can't be fixed.
- [ ] **Turn off every effect:** Windows "audio enhancements", noise suppression (NVIDIA Broadcast, Discord, Krisp), auto-gain, compressor, EQ. The data prep script does the processing itself.

## Room
- [ ] A small room with soft things in it: a closet with clothes, a bedroom with the bed, curtains and a rug. Avoid bare walls, bathrooms and kitchens.
- [ ] Quick test: clap once. If you hear a ring or tail, add blankets or cushions near the mic.
- [ ] No fan, AC, fridge hum, PC fans near the mic, traffic or pets. Record when the house is quiet.
- [ ] Record about 5 seconds of pure silence at the start of each session. It's useful for checking the noise floor.

## Style (record the way you want the output to sound)
- [ ] **Read narration scripts like the videos you'll make:** Instagram-style hooks, explanations, lists, a few questions and exclamations. The clone copies your energy, so if the videos are upbeat, read upbeat.
- [ ] Stay consistent. Use one style and one energy level across sessions. Don't mix whispering, shouting and character voices.
- [ ] Cover variety in content: numbers, dates, brand names, English loanwords, short and long sentences, questions.
- [ ] Natural pauses between sentences (about 0.5–1 s). That's where the script cuts the clips.
- [ ] If you stumble, pause, then re-read the **whole sentence**. Mistakes that stay in are fine if the transcript matches, but the script trusts Whisper, so clear re-reads help.
- [ ] Drink water, and avoid recording when you have a cold or have just woken up. Split it into 2–3 sessions of 10–15 minutes rather than one tired 30-minute take.

## Clips and length
- [ ] You can record long takes (2–10 minutes per file). The data prep script cuts them into **5–15 s clips** at pauses and transcribes them.
- [ ] Target **30 minutes of speech after trimming**, which is about 150–300 clips.
- [ ] Name files like `session1_pt_01.wav`, `session2_en_01.wav`. The `_pt_` or `_en_` part tells the script the language.

## Language
- [ ] **Record in the language(s) you'll publish in.** Accent and prosody come from the training data.
- [ ] **Both languages:** about 20 min Portuguese + 10–15 min English (or split to match how often you'll post in each). Keep each file in one language. Don't switch mid-file.
- [ ] **One language:** all 30 min in that language.
- [ ] For English, read in your natural accent unless you want the clone to sound different.

## Before you send it to training
- [ ] Listen to 3–4 random spots with headphones: no echo, no hum, no clipping, no mouth clicks every few words.
- [ ] Put all raw WAVs in one folder (e.g. `C:\Users\IsmarWolf\VoiceClone\raw\`). The prep script takes it from there.

## Script ideas (pick ~60–80 short paragraphs)
- Your own past captions and video scripts. This is the best source, because it matches what you'll generate.
- News or Wikipedia paragraphs in your topic area, read in your narrator voice.
- Lists ("3 coisas que ninguém te conta sobre...", "5 tips to...") and numbers ("em 2024, 37% das pessoas...").
