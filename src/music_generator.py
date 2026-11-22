"""Polyphonic demo music generator so the gesture music player works out of the box."""

from __future__ import annotations

import wave
from pathlib import Path
import numpy as np

SAMPLE_RATE = 44100


def _midi_to_freq(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _envelope(length: int, attack: float = 0.02, release: float = 0.25) -> np.ndarray:
    env = np.ones(length, dtype=np.float32)
    atk_samples = min(length, max(1, int(attack * SAMPLE_RATE)))
    rel_samples = min(length, max(1, int(release * SAMPLE_RATE)))
    env[:atk_samples] = np.linspace(0.0, 1.0, atk_samples, dtype=np.float32)
    env[-rel_samples:] *= np.linspace(1.0, 0.0, rel_samples, dtype=np.float32)
    return env


def _synth_pad(freq: float, duration: float, warm: bool = True) -> np.ndarray:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False, dtype=np.float32)
    sig = 0.55 * np.sin(2.0 * np.pi * freq * t)
    sig += 0.25 * np.sin(2.0 * np.pi * (freq * 1.003) * t)
    if warm:
        sig += 0.12 * np.sin(2.0 * np.pi * (freq * 2.0) * t)
        sig += 0.06 * np.sin(2.0 * np.pi * (freq * 0.5) * t)
    return sig * _envelope(len(t), attack=0.06, release=min(0.4, duration * 0.45))


def _synth_pluck(freq: float, duration: float) -> np.ndarray:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False, dtype=np.float32)
    decay = np.exp(-5.5 * t / max(duration, 0.1))
    sig = (
        0.60 * np.sin(2.0 * np.pi * freq * t)
        + 0.25 * np.sin(2.0 * np.pi * freq * 2.0 * t) * np.exp(-8.0 * t)
        + 0.10 * np.sin(2.0 * np.pi * freq * 3.0 * t) * np.exp(-12.0 * t)
    )
    return sig * decay * _envelope(len(t), attack=0.008, release=0.08)


def _synth_kick(duration: float = 0.22) -> np.ndarray:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False, dtype=np.float32)
    freq_sweep = 48.0 + 95.0 * np.exp(-26.0 * t)
    phase = 2.0 * np.pi * np.cumsum(freq_sweep) / SAMPLE_RATE
    return 0.75 * np.sin(phase) * np.exp(-11.0 * t)


def _synth_hat(duration: float = 0.06) -> np.ndarray:
    rng = np.random.default_rng(42)
    n = int(SAMPLE_RATE * duration)
    noise = rng.standard_normal(n).astype(np.float32)
    # Simple high-pass character by first difference
    hp = np.diff(noise, prepend=0.0)
    t = np.linspace(0, duration, n, endpoint=False, dtype=np.float32)
    return 0.18 * hp * np.exp(-48.0 * t)


def _place(mix: np.ndarray, sample: np.ndarray, start_sec: float, gain: float = 1.0) -> None:
    idx = int(start_sec * SAMPLE_RATE)
    if idx >= len(mix):
        return
    end = min(len(mix), idx + len(sample))
    mix[idx:end] += gain * sample[: end - idx]


def _write_stereo_wav(path: Path, mono_or_stereo: np.ndarray) -> None:
    peak = float(np.max(np.abs(mono_or_stereo))) or 1.0
    normalized = np.clip(mono_or_stereo / peak * 0.85, -1.0, 1.0)
    pcm = (normalized * 32767.0).astype(np.int16)
    if pcm.ndim == 1:
        pcm = np.column_stack((pcm, pcm))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def _build_lofi_chill() -> np.ndarray:
    bpm = 84.0
    beat = 60.0 / bpm
    bars = 8
    total_sec = bars * 4 * beat
    mix = np.zeros(int(total_sec * SAMPLE_RATE), dtype=np.float32)

    chords = [
        [60, 64, 67, 71],  # Cmaj7
        [57, 60, 64, 67],  # Am7
        [65, 69, 72, 76],  # Fmaj7
        [62, 65, 69, 72],  # Dm7
    ]
    melody = [76, 79, 74, 72, 71, 72, 74, 76]
    kick = _synth_kick()
    hat = _synth_hat()

    for bar in range(bars):
        bar_t = bar * 4 * beat
        chord = chords[bar % len(chords)]
        for note in chord:
            _place(mix, _synth_pad(_midi_to_freq(note), 4 * beat), bar_t, 0.22)
        # Bass
        _place(mix, _synth_pad(_midi_to_freq(chord[0] - 24), 2 * beat, warm=False), bar_t, 0.36)
        _place(mix, _synth_pad(_midi_to_freq(chord[0] - 24), 1.8 * beat, warm=False), bar_t + 2.5 * beat, 0.30)
        # Gentle beat
        _place(mix, kick, bar_t, 0.65)
        _place(mix, kick, bar_t + 2.5 * beat, 0.55)
        for step in range(8):
            _place(mix, hat, bar_t + step * 0.5 * beat, 0.45 if step % 2 == 0 else 0.25)
        # Pluck melody
        for step in range(4):
            m_note = melody[(bar * 2 + step) % len(melody)]
            _place(mix, _synth_pluck(_midi_to_freq(m_note), beat * 1.1), bar_t + step * beat + 0.25 * beat, 0.35)
    return mix


def _build_neon_synth() -> np.ndarray:
    bpm = 112.0
    beat = 60.0 / bpm
    bars = 8
    total_sec = bars * 4 * beat
    mix = np.zeros(int(total_sec * SAMPLE_RATE), dtype=np.float32)

    roots = [45, 41, 48, 43]  # Am, F, C, G
    arp_offsets = [0, 7, 12, 15, 12, 7, 19, 12]
    kick = _synth_kick()
    hat = _synth_hat()

    for bar in range(bars):
        bar_t = bar * 4 * beat
        root = roots[bar % len(roots)]
        for interval in (12, 15, 19):
            _place(mix, _synth_pad(_midi_to_freq(root + interval), 4 * beat), bar_t, 0.18)
        for step in range(8):
            arp_note = root + 12 + arp_offsets[step]
            _place(mix, _synth_pluck(_midi_to_freq(arp_note), 0.45 * beat), bar_t + step * 0.5 * beat, 0.32)
            _place(mix, _synth_pluck(_midi_to_freq(root), 0.42 * beat), bar_t + step * 0.5 * beat, 0.28)
        for b in range(4):
            _place(mix, kick, bar_t + b * beat, 0.72)
            _place(mix, hat, bar_t + (b + 0.5) * beat, 0.65)
    return mix


def _build_ambient_zen() -> np.ndarray:
    total_sec = 22.0
    mix = np.zeros(int(total_sec * SAMPLE_RATE), dtype=np.float32)
    progression = [
        (0.0, 5.5, [50, 57, 62, 66, 69]),   # Dmaj9
        (5.5, 5.5, [47, 54, 59, 62, 66]),   # Bm7
        (11.0, 5.5, [43, 50, 55, 59, 62]),  # Gmaj7
        (16.5, 5.5, [45, 52, 57, 61, 64]),  # A6
    ]
    bells = [74, 78, 81, 86, 81, 78, 76, 74, 71, 74, 78, 81]
    for start, dur, notes in progression:
        for n in notes:
            _place(mix, _synth_pad(_midi_to_freq(n), dur + 0.6), start, 0.18)
    for i, note in enumerate(bells):
        _place(mix, _synth_pluck(_midi_to_freq(note), 2.2), i * 1.75 + 0.4, 0.34)
    return mix


def _build_upbeat_groove() -> np.ndarray:
    bpm = 120.0
    beat = 60.0 / bpm
    bars = 8
    total_sec = bars * 4 * beat
    mix = np.zeros(int(total_sec * SAMPLE_RATE), dtype=np.float32)

    chords = [
        [60, 64, 67, 72],
        [62, 65, 69, 74],
        [64, 67, 71, 76],
        [67, 71, 74, 79],
    ]
    kick = _synth_kick()
    hat = _synth_hat()

    for bar in range(bars):
        bar_t = bar * 4 * beat
        chord = chords[bar % len(chords)]
        for b in range(4):
            _place(mix, kick, bar_t + b * beat, 0.70)
            _place(mix, hat, bar_t + (b + 0.5) * beat, 0.55)
            # Bouncy chord stabs
            stab_offset = 0.0 if b % 2 == 0 else 0.5 * beat
            for n in chord:
                _place(mix, _synth_pluck(_midi_to_freq(n), 0.55 * beat), bar_t + b * beat + stab_offset, 0.22)
        # Bassline
        for step in range(4):
            _place(mix, _synth_pluck(_midi_to_freq(chord[0] - 24), 0.75 * beat), bar_t + step * beat, 0.42)
    return mix


def ensure_demo_music(music_dir: Path) -> list[Path]:
    """Generate 4 built-in demo music tracks if the music directory is empty."""
    music_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        p for p in music_dir.iterdir()
        if p.suffix.lower() in {".wav", ".mp3", ".ogg", ".flac"}
    )
    if existing:
        return existing

    generators = [
        ("01_LoFi_Chill_Focus.wav", _build_lofi_chill),
        ("02_Neon_Synth_Wave.wav", _build_neon_synth),
        ("03_Ambient_Zen_Pad.wav", _build_ambient_zen),
        ("04_Upbeat_Gesture_Groove.wav", _build_upbeat_groove),
    ]
    created: list[Path] = []
    for filename, builder in generators:
        target = music_dir / filename
        _write_stereo_wav(target, builder())
        created.append(target)
    return created
