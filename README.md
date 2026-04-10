# MoodSynth

MoodSynth takes a plain-English mood description and turns it into live ambient audio.

## What it does

- User types a mood like `calm rainy night`
- Local LLM converts the text into synthesis parameters as JSON
- NumPy synth engine generates audio
- sounddevice plays the sound live
- Different moods produce different sound textures

## Tech stack

- Python
- ollama
- numpy
- sounddevice

## How it works

Text mood input is sent to a local Ollama model.  
The model returns JSON with:

- `base_freq`
- `tempo`
- `waveform`
- `reverb`
- `amplitude`
- `harmonics`
- `tremolo`

Those parameters are passed into a pure-NumPy synthesis engine, which creates an audio waveform and plays it through the speakers.

## Added features

I improved the prompt so the LLM returns more reliable JSON-only output.

I also added:

- `noise` waveform
- `tremolo` effect

These make moods like stormy, glitchy, eerie, and forest-like sound more distinct.

## Run

```bash
ollama serve
ollama pull qwen2.5:3b
python day11_starter.py
