"""Gesture-controllable Music Player powered by pygame.mixer."""

from __future__ import annotations

import json
import shutil
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pygame

from .config import DEMO_FILENAMES, MUSIC_DIR, PLAYLIST_FILE

SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac"}


@dataclass
class TrackInfo:
    path: Path
    title: str
    duration_sec: float
    is_demo: bool = False
    spectrum_frames: Optional[np.ndarray] = None  # shape: (num_frames, 16) normalized 0..1


@dataclass
class PlayerState:
    tracks: List[TrackInfo]
    current_index: int
    marked_index: Optional[int]
    current_title: str
    is_playing: bool
    is_paused: bool
    volume: float  # 0.0 .. 1.0
    is_muted: bool
    position_sec: float
    duration_sec: float
    audio_ready: bool
    spectrum: List[float]


class MusicPlayer:
    """Thread-safe music player with playlist management, ducking, and spectrum visualization."""

    def __init__(
        self,
        on_track_change: Optional[Callable[[TrackInfo], None]] = None,
        music_dir: Path = MUSIC_DIR,
        playlist_file: Path = PLAYLIST_FILE,
    ) -> None:
        self._lock = threading.RLock()
        self._on_track_change = on_track_change
        self.music_dir = Path(music_dir)
        self.playlist_file = Path(playlist_file)
        self.audio_ready = False
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
            self.audio_ready = True
        except Exception:
            self.audio_ready = False

        self.tracks: List[TrackInfo] = []
        self.current_index: int = 0
        self.marked_index: Optional[int] = None
        self._marked_path_str: Optional[str] = None
        self.is_playing: bool = False
        self.is_paused: bool = False
        self.volume: float = 0.72
        self.is_muted: bool = False
        self._duck_factor: float = 1.0
        self._play_offset_sec: float = 0.0

        self.refresh_library()
        self._apply_volume()

    def _cleanup_demo_files(self) -> None:
        if not self.music_dir.exists():
            return
        for name in DEMO_FILENAMES:
            demo_path = self.music_dir / name
            if demo_path.exists():
                try:
                    demo_path.unlink()
                except Exception:
                    pass

    def _load_saved_playlist_data(self) -> tuple[List[Path], Optional[str]]:
        if not self.playlist_file.exists():
            return [], None
        try:
            data = json.loads(self.playlist_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                raw_tracks = data.get("tracks", [])
                marked = data.get("marked")
            else:
                raw_tracks = data
                marked = None
            paths = []
            for raw in raw_tracks:
                p = Path(raw)
                if (
                    p.is_file()
                    and p.suffix.lower() in SUPPORTED_AUDIO_EXTS
                    and p.name not in DEMO_FILENAMES
                ):
                    paths.append(p)
            return paths, str(marked) if marked else None
        except Exception:
            return [], None

    def _save_playlist_paths(self) -> None:
        try:
            self.playlist_file.parent.mkdir(parents=True, exist_ok=True)
            marked_str = None
            if self.marked_index is not None and 0 <= self.marked_index < len(self.tracks):
                marked_str = str(self.tracks[self.marked_index].path)
            payload = {
                "tracks": [str(t.path) for t in self.tracks],
                "marked": marked_str,
            }
            self.playlist_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def refresh_library(self) -> None:
        with self._lock:
            self.music_dir.mkdir(parents=True, exist_ok=True)
            self._cleanup_demo_files()

            seen = set()
            candidates: List[Path] = []
            for p in sorted(self.music_dir.iterdir()):
                if (
                    p.is_file()
                    and p.suffix.lower() in SUPPORTED_AUDIO_EXTS
                    and p.name not in DEMO_FILENAMES
                ):
                    rp = p.resolve()
                    if rp not in seen:
                        seen.add(rp)
                        candidates.append(p)

            saved_paths, saved_marked = self._load_saved_playlist_data()
            for p in saved_paths:
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    candidates.append(p)

            self.tracks = [self._build_track_info(p) for p in candidates]
            if self.current_index >= len(self.tracks):
                self.current_index = 0

            self.marked_index = None
            target_marked = saved_marked or self._marked_path_str
            if target_marked:
                try:
                    m_res = Path(target_marked).resolve()
                    for idx, tr in enumerate(self.tracks):
                        if tr.path.resolve() == m_res:
                            self.marked_index = idx
                            break
                except Exception:
                    pass

    def _build_track_info(self, path: Path) -> TrackInfo:
        clean_title = path.stem.replace("_", " ")
        is_demo = path.stem[:3] in {"01_", "02_", "03_", "04_"}
        duration = 0.0
        spectrum_frames = None

        if path.suffix.lower() == ".wav":
            try:
                with wave.open(str(path), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate() or 44100
                    channels = wf.getnchannels() or 1
                    sampwidth = wf.getsampwidth()
                    duration = frames / float(rate)
                    if sampwidth == 2 and frames > 0:
                        raw = np.frombuffer(wf.readframes(frames), dtype=np.int16).astype(np.float32)
                        if channels > 1:
                            raw = raw.reshape(-1, channels).mean(axis=1)
                        raw /= 32768.0
                        spectrum_frames = self._compute_spectrum_frames(raw, rate, fps=20)
            except Exception:
                duration = 20.0
        elif self.audio_ready:
            try:
                snd = pygame.mixer.Sound(str(path))
                duration = float(snd.get_length())
            except Exception:
                duration = 0.0

        return TrackInfo(
            path=path,
            title=clean_title,
            duration_sec=duration,
            is_demo=is_demo,
            spectrum_frames=spectrum_frames,
        )

    @staticmethod
    def _compute_spectrum_frames(mono: np.ndarray, rate: int, fps: int = 20) -> np.ndarray:
        hop = max(1, rate // fps)
        n_frames = max(1, len(mono) // hop)
        bands = 16
        out = np.zeros((n_frames, bands), dtype=np.float32)
        window = np.hanning(min(hop, 1024))
        wlen = len(window)
        freqs = np.fft.rfftfreq(wlen, d=1.0 / rate)
        band_edges = np.geomspace(50.0, min(8000.0, rate * 0.45), bands + 1)

        for i in range(n_frames):
            start = i * hop
            chunk = mono[start : start + wlen]
            if len(chunk) < wlen:
                padded = np.zeros(wlen, dtype=np.float32)
                padded[: len(chunk)] = chunk
                chunk = padded
            mag = np.abs(np.fft.rfft(chunk * window))
            for b in range(bands):
                mask = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])
                if np.any(mask):
                    out[i, b] = float(np.mean(mag[mask]))

        peak = float(np.percentile(out, 98)) if out.size else 1.0
        if peak > 1e-6:
            out = np.clip(out / peak, 0.0, 1.0)
        return out

    def add_files(self, file_paths: List[str], copy_to_library: bool = False) -> int:
        added = 0
        with self._lock:
            for raw in file_paths:
                src = Path(raw)
                if not src.is_file() or src.suffix.lower() not in SUPPORTED_AUDIO_EXTS:
                    continue
                target = src
                if copy_to_library:
                    self.music_dir.mkdir(parents=True, exist_ok=True)
                    dest = self.music_dir / src.name
                    if dest != src:
                        shutil.copy2(src, dest)
                    target = dest
                if all(t.path.resolve() != target.resolve() for t in self.tracks):
                    self.tracks.append(self._build_track_info(target))
                    added += 1
            if added:
                self._save_playlist_paths()
        return added

    def remove_track(self, index: Optional[int] = None) -> Optional[str]:
        """Remove track at index (or current track) from playlist and delete if inside assets/music."""
        with self._lock:
            if not self.tracks:
                return None
            idx = self.current_index if index is None else index
            if idx < 0 or idx >= len(self.tracks):
                return None
            target = self.tracks[idx]
            was_playing_target = self.is_playing and idx == self.current_index
            if idx == self.current_index:
                self.stop()
            removed = self.tracks.pop(idx)
            if self.current_index >= len(self.tracks):
                self.current_index = max(0, len(self.tracks) - 1)
            elif idx < self.current_index:
                self.current_index -= 1

            # Update marked_index safely
            if self.marked_index is not None:
                if idx == self.marked_index:
                    self.marked_index = None
                    self._marked_path_str = None
                elif idx < self.marked_index:
                    self.marked_index -= 1

            # Delete file if stored inside local assets/music folder
            try:
                if removed.path.parent.resolve() == self.music_dir.resolve() and removed.path.exists():
                    removed.path.unlink()
            except Exception:
                pass
            self._save_playlist_paths()
            if was_playing_target and self.tracks:
                self.play(self.current_index)
            return removed.title

    def set_marked_track(self, index: int) -> Optional[TrackInfo]:
        """Mark (or unmark if already marked) a track as the glasses-triggered song."""
        with self._lock:
            if not (0 <= index < len(self.tracks)):
                return None
            if self.marked_index == index:
                self.marked_index = None
                self._marked_path_str = None
                self._save_playlist_paths()
                return None
            self.marked_index = index
            self._marked_path_str = str(self.tracks[index].path)
            self._save_playlist_paths()
            return self.tracks[index]

    def play_marked_track(self) -> Optional[TrackInfo]:
        """Play the starred/marked song (or first track if none marked)."""
        with self._lock:
            if not self.tracks:
                return None
            target_idx = self.marked_index if (self.marked_index is not None and self.marked_index < len(self.tracks)) else 0
            return self.play(target_idx)

    def clear_playlist(self) -> None:
        with self._lock:
            self.stop()
            for tr in list(self.tracks):
                try:
                    if tr.path.parent.resolve() == self.music_dir.resolve() and tr.path.exists():
                        tr.path.unlink()
                except Exception:
                    pass
            self.tracks.clear()
            self.current_index = 0
            self.marked_index = None
            self._marked_path_str = None
            self._save_playlist_paths()

    def load_folder(self, folder_path: str) -> int:
        folder = Path(folder_path)
        if not folder.is_dir():
            return 0
        files = sorted(
            str(p) for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS
        )
        return self.add_files(files, copy_to_library=False)

    def _apply_volume(self) -> None:
        if not self.audio_ready:
            return
        effective = 0.0 if self.is_muted else (self.volume * self._duck_factor)
        pygame.mixer.music.set_volume(float(np.clip(effective, 0.0, 1.0)))

    def set_ducking(self, active: bool) -> None:
        with self._lock:
            self._duck_factor = 0.22 if active else 1.0
            self._apply_volume()

    def play(self, index: Optional[int] = None) -> Optional[TrackInfo]:
        with self._lock:
            if not self.tracks or not self.audio_ready:
                return None
            if index is not None:
                self.current_index = index % len(self.tracks)
            track = self.tracks[self.current_index]
            try:
                pygame.mixer.music.load(str(track.path))
                pygame.mixer.music.play()
                self.is_playing = True
                self.is_paused = False
                self._play_offset_sec = 0.0
                self._apply_volume()
                if self._on_track_change:
                    self._on_track_change(track)
                return track
            except Exception:
                self.is_playing = False
                return None

    def pause(self) -> None:
        with self._lock:
            if self.audio_ready and self.is_playing and not self.is_paused:
                pygame.mixer.music.pause()
                self.is_paused = True

    def resume(self) -> None:
        with self._lock:
            if not self.audio_ready:
                return
            if self.is_playing and self.is_paused:
                pygame.mixer.music.unpause()
                self.is_paused = False
            elif not self.is_playing:
                self.play(self.current_index)

    def toggle_play_pause(self) -> str:
        with self._lock:
            if not self.is_playing:
                track = self.play(self.current_index)
                return f"Memutar {track.title}" if track else "Tidak ada lagu"
            if self.is_paused:
                self.resume()
                return "Musik dilanjutkan"
            self.pause()
            return "Musik dijeda"

    def stop(self) -> None:
        with self._lock:
            if self.audio_ready:
                pygame.mixer.music.stop()
                try:
                    pygame.mixer.music.unload()
                except Exception:
                    pass
            self.is_playing = False
            self.is_paused = False

    def next_track(self) -> Optional[TrackInfo]:
        with self._lock:
            if not self.tracks:
                return None
            return self.play((self.current_index + 1) % len(self.tracks))

    def prev_track(self) -> Optional[TrackInfo]:
        with self._lock:
            if not self.tracks:
                return None
            return self.play((self.current_index - 1) % len(self.tracks))

    def set_volume(self, value: float) -> float:
        with self._lock:
            self.volume = float(np.clip(value, 0.0, 1.0))
            if self.volume > 0.01 and self.is_muted:
                self.is_muted = False
            self._apply_volume()
            return self.volume

    def adjust_volume(self, delta: float) -> float:
        return self.set_volume(self.volume + delta)

    def toggle_mute(self) -> bool:
        with self._lock:
            self.is_muted = not self.is_muted
            self._apply_volume()
            return self.is_muted

    def tick(self) -> None:
        """Poll mixer state to auto-advance playlist when track finishes."""
        with self._lock:
            if self.audio_ready and self.is_playing and not self.is_paused:
                if not pygame.mixer.music.get_busy():
                    self.next_track()

    def get_state(self) -> PlayerState:
        with self._lock:
            if not self.tracks:
                return PlayerState(
                    tracks=[],
                    current_index=0,
                    marked_index=self.marked_index,
                    current_title="Playlist kosong — Tambah Lagu Anda",
                    is_playing=False,
                    is_paused=False,
                    volume=self.volume,
                    is_muted=self.is_muted,
                    position_sec=0.0,
                    duration_sec=0.0,
                    audio_ready=self.audio_ready,
                    spectrum=[0.05] * 16,
                )

            track = self.tracks[self.current_index]
            pos = 0.0
            if self.audio_ready and self.is_playing:
                ms = pygame.mixer.music.get_pos()
                if ms >= 0:
                    pos = self._play_offset_sec + (ms / 1000.0)

            spectrum = [0.06] * 16
            if self.is_playing and not self.is_paused:
                if track.spectrum_frames is not None and len(track.spectrum_frames) > 0:
                    frame_idx = int(pos * 20.0) % len(track.spectrum_frames)
                    vals = track.spectrum_frames[frame_idx]
                    spectrum = [float(max(0.06, v * (0.2 if self.is_muted else max(0.3, self.volume)))) for v in vals]
                else:
                    # Smooth procedural fallback visualizer for MP3/OGG without PCM pre-scan
                    spectrum = [
                        float(max(0.08, 0.45 + 0.45 * np.sin(pos * 6.0 + b * 0.65) * np.cos(pos * 2.3 + b)))
                        for b in range(16)
                    ]

            return PlayerState(
                tracks=list(self.tracks),
                current_index=self.current_index,
                marked_index=self.marked_index,
                current_title=track.title,
                is_playing=self.is_playing,
                is_paused=self.is_paused,
                volume=self.volume,
                is_muted=self.is_muted,
                position_sec=pos,
                duration_sec=track.duration_sec,
                audio_ready=self.audio_ready,
                spectrum=spectrum,
            )
