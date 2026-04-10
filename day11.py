import subprocess
import json
import numpy as np
import sounddevice as sd
import sys
import re
import time

def check_ollama():
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            print("ERROR: ollama is not running.")
            print("Fix: open another terminal and run: ollama serve")
            sys.exit(1)
        if "qwen2.5" not in result.stdout.lower():
            print("ERROR: qwen2.5:3b not pulled.")
            print("Fix: ollama pull qwen2.5:3b")
            sys.exit(1)
        return True
    except FileNotFoundError:
        print("ERROR: ollama not installed.")
        print("Get it from https://ollama.com")
        sys.exit(1)

check_ollama()

MODEL = "qwen2.5:3b"
SAMPLE_RATE = 44100

SYNTH_PROMPT_TEMPLATE = """
You are a synthesizer parameter generator.

Convert the user's mood into ONE valid JSON object.
Return ONLY raw JSON.
Do not explain.
Do not add markdown.
Do not add code fences.
Do not add any text before or after the JSON.

Mood: "{mood}"

Required JSON schema:
{{
  "base_freq": integer from 80 to 800,
  "tempo": number from 0.3 to 3.0,
  "waveform": one of ["sine", "square", "triangle", "sawtooth", "noise"],
  "reverb": number from 0.0 to 1.0,
  "amplitude": number from 0.05 to 0.35,
  "harmonics": integer from 1 to 5,
  "tremolo": number from 0.0 to 1.0
}}

Mapping hints:
- calm, peaceful, sleepy, rainy, soft -> lower base_freq, slower tempo, sine or triangle, more reverb
- tense, thriller, horror, suspense -> square or sawtooth, medium tempo, darker lower-mid frequencies
- energetic, bright, excited -> sawtooth, faster tempo, moderate reverb
- forest, wind, storm, ocean, static, glitch -> noise can be used
- more intense moods can use higher harmonics and more tremolo

Example valid output:
{"base_freq":220,"tempo":0.8,"waveform":"sine","reverb":0.6,"amplitude":0.18,"harmonics":2,"tremolo":0.2}
"""

def get_default_params():
    return {
        "base_freq": 220,
        "tempo": 1.0,
        "waveform": "sine",
        "reverb": 0.3,
        "amplitude": 0.2,
        "harmonics": 1,
        "tremolo": 0.1,
    }

def extract_json(text):
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start:end + 1]

def validate_params(params):
    safe = {
        "base_freq": int(np.clip(params.get("base_freq", 220), 80, 800)),
        "tempo": float(np.clip(params.get("tempo", 1.0), 0.3, 3.0)),
        "waveform": str(params.get("waveform", "sine")).lower().strip(),
        "reverb": float(np.clip(params.get("reverb", 0.3), 0.0, 1.0)),
        "amplitude": float(np.clip(params.get("amplitude", 0.2), 0.05, 0.35)),
        "harmonics": int(np.clip(params.get("harmonics", 1), 1, 5)),
        "tremolo": float(np.clip(params.get("tremolo", 0.1), 0.0, 1.0)),
    }
    if safe["waveform"] not in WAVEFORMS:
        safe["waveform"] = "sine"
    return safe

def get_params_from_mood(mood):
    prompt = SYNTH_PROMPT_TEMPLATE.format(mood=mood)
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True,
            text=True,
            timeout=30
        )
        raw = result.stdout.strip()
    except Exception as e:
        print(f"LLM error: {e}")
        return get_default_params()

    json_str = extract_json(raw)
    if not json_str:
        print(f"Could not find JSON. Raw output:\n{raw[:300]}")
        return get_default_params()

    try:
        params = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        print(f"Raw JSON candidate: {json_str[:300]}")
        return get_default_params()

    return validate_params(params)

def gen_sine(freq, duration, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)

def gen_square(freq, duration, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sign(np.sin(2 * np.pi * freq * t))

def gen_triangle(freq, duration, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1

def gen_sawtooth(freq, duration, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return 2 * (t * freq - np.floor(t * freq + 0.5))

def gen_noise(freq, duration, sample_rate=SAMPLE_RATE):
    samples = int(sample_rate * duration)
    white = np.random.uniform(-1.0, 1.0, samples)
    kernel_size = max(8, int(sample_rate / max(freq, 80)))
    kernel = np.ones(kernel_size) / kernel_size
    smooth = np.convolve(white, kernel, mode="same")
    max_val = np.max(np.abs(smooth))
    if max_val > 0:
        smooth = smooth / max_val
    return smooth

WAVEFORMS = {
    "sine": gen_sine,
    "square": gen_square,
    "triangle": gen_triangle,
    "sawtooth": gen_sawtooth,
    "noise": gen_noise,
}

def apply_reverb(signal, depth):
    if depth <= 0:
        return signal
    output = signal.copy()
    delays_ms = [29, 47, 73, 109]
    decays = [0.6, 0.5, 0.4, 0.3]
    for delay_ms, decay in zip(delays_ms, decays):
        delay_samples = int(SAMPLE_RATE * delay_ms / 1000)
        if delay_samples >= len(signal):
            continue
        delayed = np.zeros_like(signal)
        delayed[delay_samples:] = signal[:-delay_samples] * decay * depth
        output += delayed
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val
    return output

def add_harmonics(signal, base_freq, num_harmonics, waveform_func, duration):
    if num_harmonics <= 1 or waveform_func == gen_noise:
        return signal
    output = signal.copy()
    for h in range(2, num_harmonics + 1):
        harmonic_freq = base_freq * h
        if harmonic_freq > 8000:
            break
        harmonic_wave = waveform_func(harmonic_freq, duration) / (h * 1.5)
        output += harmonic_wave
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val
    return output

def apply_tremolo(signal, rate_hz, depth):
    if depth <= 0:
        return signal
    t = np.linspace(0, len(signal) / SAMPLE_RATE, len(signal), endpoint=False)
    mod = (1.0 - depth) + depth * (0.5 + 0.5 * np.sin(2 * np.pi * rate_hz * t))
    return signal * mod

def apply_envelope(signal, attack=0.05, release=0.4):
    attack_samples = int(attack * SAMPLE_RATE)
    release_samples = int(release * SAMPLE_RATE)
    envelope = np.ones_like(signal)
    if 0 < attack_samples < len(signal):
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    if 0 < release_samples < len(signal):
        envelope[-release_samples:] = np.linspace(1, 0, release_samples)
    return signal * envelope

def synthesize(params, duration=8.0):
    waveform_func = WAVEFORMS.get(params["waveform"], gen_sine)
    base_freq = params["base_freq"]
    tempo_freq = params["tempo"]

    signal = waveform_func(base_freq, duration)
    signal = add_harmonics(signal, base_freq, params["harmonics"], waveform_func, duration)

    t = np.linspace(0, duration, len(signal), endpoint=False)
    tempo_lfo = 0.65 + 0.35 * np.sin(2 * np.pi * tempo_freq * t)
    signal = signal * tempo_lfo

    tremolo_rate = max(0.5, tempo_freq * 2.0)
    signal = apply_tremolo(signal, tremolo_rate, params["tremolo"])
    signal = apply_reverb(signal, params["reverb"])
    signal = signal * params["amplitude"]
    signal = apply_envelope(signal)

    peak = np.max(np.abs(signal))
    if peak > 0.95:
        signal = signal * (0.95 / peak)

    signal = np.clip(signal, -1.0, 1.0)
    return signal.astype(np.float32)

def show_params(params):
    print()
    print(" ┌─ Synth Parameters ─────────────────")
    print(f" │ Waveform: {params['waveform']}")
    print(f" │ Frequency: {params['base_freq']} Hz")
    print(f" │ Tempo: {params['tempo']:.2f} Hz")
    print(f" │ Reverb: {'█' * int(params['reverb'] * 10):<10} {params['reverb']:.2f}")
    print(f" │ Amplitude: {'█' * int(params['amplitude'] * 25):<10} {params['amplitude']:.2f}")
    print(f" │ Harmonics: {params['harmonics']}")
    print(f" │ Tremolo: {'█' * int(params['tremolo'] * 10):<10} {params['tremolo']:.2f}")
    print(" └────────────────────────────────────")
    print()

def main():
    print()
    print("=" * 50)
    print(" MoodSynth — AI-driven ambient generator")
    print(f" Model: {MODEL}")
    print("=" * 50)
    print()
    print(" Type a mood and hear the result.")
    print(" Examples:")
    print(" calm rainy night")
    print(" tense thriller scene")
    print(" peaceful forest morning")
    print(" glitchy alien transmission")
    print()
    print(" Type 'quit' to exit.")
    print()

    while True:
        try:
            mood = input("Mood > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not mood:
            continue

        if mood.lower() in ("quit", "exit", "q"):
            break

        print("Asking the model...", end="", flush=True)
        start = time.time()
        params = get_params_from_mood(mood)
        elapsed = time.time() - start
        print("\r" + " " * 40 + "\r", end="")
        print(f"LLM responded in {elapsed:.1f}s")
        show_params(params)

        print("Synthesizing audio...")
        audio = synthesize(params, duration=8.0)

        print("Playing...")
        try:
            sd.play(audio, samplerate=SAMPLE_RATE)
            sd.wait()
            print("Done")
        except Exception as e:
            print(f"Audio playback failed: {e}")
            print("Check sounddevice.default.device")

    print("\nMoodSynth ended.")

if __name__ == "__main__":
    main()