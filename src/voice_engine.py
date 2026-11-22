"""Complete Voice-to-Speech Engine: SANTO (Jarvis UK Male Neural TTS) + STT + Extensible Keywords."""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import edge_tts
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from gtts import gTTS

from .config import SANTO_KEYWORDS_FILE, TTS_CACHE_DIR

try:
    import pythoncom
except ImportError:
    pythoncom = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

JARVIS_VOICE = "en-GB-RyanNeural"
JARVIS_RATE = "-3%"
JARVIS_PITCH = "-6Hz"

DEFAULT_SANTO_CONFIG: Dict[str, Any] = {
    "assistant_name": "Santo",
    "owner_name": "Boss Budi",
    "jarvis_voice": JARVIS_VOICE,
    "jarvis_rate": JARVIS_RATE,
    "jarvis_pitch": JARVIS_PITCH,
    "rules": [
        {
            "id": "wake_santo",
            "patterns": [
                "santo",
                "hey santo",
                "hello santo",
                "hi santo",
                "halo santo",
                "ok santo",
                "okay santo",
                "santo wake up",
            ],
            "match_type": "exact",
            "response": "Yes, sir.",
            "action": "SANTO_WAKE",
        },
        {
            "id": "who_am_i",
            "patterns": [
                "who am i",
                "santo who am i",
                "who am i santo",
                "tell me who am i",
                "do you know me",
                "siapa saya",
                "hu em ai",
            ],
            "match_type": "contains",
            "response": "Boss Budi, is there anything I can help you with?",
            "action": "CUSTOM_REPLY",
        },
    ],
}


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _strip_wake_prefix(norm_text: str) -> str:
    for prefix in (
        "hey santo ",
        "hello santo ",
        "hi santo ",
        "halo santo ",
        "ok santo ",
        "okay santo ",
        "santo ",
    ):
        if norm_text.startswith(prefix):
            return norm_text[len(prefix) :].strip()
    return norm_text


class SantoKeywordStore:
    """Persistent, hot-reloading JSON keyword & response registry for SANTO."""

    def __init__(self, path: Path = SANTO_KEYWORDS_FILE) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._last_mtime: float = 0.0
        self._data: Dict[str, Any] = {}
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        with self._lock:
            if not self.path.exists():
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._data = json.loads(json.dumps(DEFAULT_SANTO_CONFIG))
                self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
                self._last_mtime = self.path.stat().st_mtime
                return

            try:
                mtime = self.path.stat().st_mtime
                if mtime != self._last_mtime or not self._data:
                    loaded = json.loads(self.path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict) and isinstance(loaded.get("rules"), list):
                        self._data = loaded
                        self._last_mtime = mtime
            except Exception:
                if not self._data:
                    self._data = json.loads(json.dumps(DEFAULT_SANTO_CONFIG))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        self._last_mtime = self.path.stat().st_mtime

    def get_voice_settings(self) -> Tuple[str, str, str]:
        self._ensure_loaded()
        with self._lock:
            return (
                str(self._data.get("jarvis_voice", JARVIS_VOICE)),
                str(self._data.get("jarvis_rate", JARVIS_RATE)),
                str(self._data.get("jarvis_pitch", JARVIS_PITCH)),
            )

    def list_rules(self) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        with self._lock:
            return list(self._data.get("rules", []))

    def add_rule(
        self,
        patterns: List[str],
        response: str,
        match_type: str = "contains",
        rule_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._ensure_loaded()
        norm_patterns = [_normalize_text(p) for p in patterns if _normalize_text(p)]
        if not norm_patterns:
            raise ValueError("Kata kunci tidak boleh kosong")
        with self._lock:
            new_rule = {
                "id": rule_id or f"custom_{uuid.uuid4().hex[:6]}",
                "patterns": norm_patterns,
                "match_type": match_type,
                "response": response.strip(),
                "action": "CUSTOM_REPLY",
            }
            self._data.setdefault("rules", []).append(new_rule)
            self._save()
            return new_rule

    def remove_rule(self, rule_id: str) -> bool:
        self._ensure_loaded()
        with self._lock:
            rules = self._data.get("rules", [])
            kept = [r for r in rules if r.get("id") != rule_id]
            if len(kept) == len(rules):
                return False
            self._data["rules"] = kept
            self._save()
            return True

    def match_non_wake(self, norm_text: str) -> Optional[Tuple[str, str]]:
        self._ensure_loaded()
        stripped = _strip_wake_prefix(norm_text)
        candidates = (norm_text, stripped) if stripped != norm_text else (norm_text,)
        with self._lock:
            for rule in self._data.get("rules", []):
                if rule.get("id") == "wake_santo" or rule.get("action") == "SANTO_WAKE":
                    continue
                mtype = rule.get("match_type", "contains")
                patterns = [_normalize_text(p) for p in rule.get("patterns", [])]
                resp = str(rule.get("response", "")).strip()
                act = str(rule.get("action") or "CUSTOM_REPLY")
                for cand in candidates:
                    if not cand:
                        continue
                    for pat in patterns:
                        if not pat:
                            continue
                        hit = (cand == pat) if mtype == "exact" else (pat in cand)
                        if hit:
                            return act, resp
        return None

    def match_wake(self, norm_text: str) -> Optional[Tuple[str, str]]:
        self._ensure_loaded()
        with self._lock:
            for rule in self._data.get("rules", []):
                if rule.get("id") == "wake_santo" or rule.get("action") == "SANTO_WAKE":
                    patterns = [_normalize_text(p) for p in rule.get("patterns", [])]
                    if norm_text in patterns:
                        return "SANTO_WAKE", str(rule.get("response", "Yes, sir."))
        return None


@dataclass
class VoiceTranscriptEvent:
    text: str
    language: str
    command_action: Optional[str] = None
    command_arg: Optional[str] = None
    reply_text: Optional[str] = None


@dataclass
class VoiceStatus:
    listening: bool = False
    continuous: bool = False
    speaking: bool = False
    mic_level: float = 0.0
    status_text: str = "Siap"


def list_microphones() -> List[Tuple[int, str]]:
    """Return list of (device_index, device_name) for available input devices."""
    result: List[Tuple[int, str]] = [(-1, "Auto (Default Mic)")]
    try:
        devices = sd.query_devices()
        seen_names = set()
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                name = str(dev.get("name", f"Mic {idx}"))
                if "sound mapper" in name.lower() or "primary sound" in name.lower():
                    continue
                if name not in seen_names:
                    seen_names.add(name)
                    result.append((idx, name))
    except Exception:
        pass
    return result


def _match_builtin_command(raw: str, low: str) -> Tuple[Optional[str], Optional[str]]:
    # Direct speak command: "katakan ...", "say ..."
    for prefix in ("katakan ", "ucapkan ", "bacakan ", "baca ", "say ", "speak "):
        if low.startswith(prefix) and len(raw) > len(prefix):
            return "SPEAK_CUSTOM", raw[len(prefix) :].strip()

    # Music commands
    if any(k in low for k in ("lagu selanjutnya", "lagu berikutnya", "berikutnya", "next track", "next song", "next")):
        return "NEXT_TRACK", None
    if any(k in low for k in ("lagu sebelumnya", "sebelumnya", "kembali", "previous", "prev track", "back")):
        return "PREV_TRACK", None
    if any(k in low for k in ("stop musik", "stop music", "hentikan musik", "matikan musik", "berhenti")):
        return "STOP_MUSIC", None
    if any(k in low for k in ("pause", "jeda", "tahan musik")):
        return "PAUSE_MUSIC", None
    if any(k in low for k in ("putar musik", "putar lagu", "mainkan musik", "lanjutkan musik", "play musik", "play music", "play")):
        return "PLAY_MUSIC", None
    if any(k in low for k in ("volume naik", "besarkan suara", "keraskan suara", "naikkan volume", "volume up", "louder")):
        return "VOLUME_UP", None
    if any(k in low for k in ("volume turun", "kecilkan suara", "pelankan suara", "turunkan volume", "volume down", "quieter")):
        return "VOLUME_DOWN", None
    if any(k in low for k in ("unmute", "nyalakan suara")):
        return "UNMUTE", None
    if any(k in low for k in ("mute", "senyap", "bisukan")):
        return "MUTE", None
    if any(k in low for k in ("lagu apa", "judul lagu", "sedang main", "now playing", "what song")):
        return "ANNOUNCE_TRACK", None

    # Camera PTZ commands
    if any(k in low for k in ("ikuti wajah", "auto face", "tracking wajah", "fokus wajah", "follow face")):
        return "PTZ_MODE", "Auto Face"
    if any(k in low for k in ("ikuti tangan", "auto hand", "tracking tangan", "fokus tangan", "follow hand")):
        return "PTZ_MODE", "Auto Hand"
    if any(k in low for k in ("hybrid", "otomatis campuran")):
        return "PTZ_MODE", "Auto Hybrid"
    if any(k in low for k in ("mode manual", "manual kamera", "manual mode")):
        return "PTZ_MODE", "Manual PTZ"
    if any(k in low for k in ("reset kamera", "kamera tengah", "center camera", "reset zoom", "reset camera")):
        return "PTZ_RESET", None
    if any(k in low for k in ("zoom in", "perbesar kamera", "perbesar", "dekat")):
        return "PTZ_ZOOM_IN", None
    if any(k in low for k in ("zoom out", "perkecil kamera", "perkecil", "jauh")):
        return "PTZ_ZOOM_OUT", None
    if any(k in low for k in ("geser kiri", "kamera kiri", "ke kiri", "pan left")):
        return "PTZ_PAN_LEFT", None
    if any(k in low for k in ("geser kanan", "kamera kanan", "ke kanan", "pan right")):
        return "PTZ_PAN_RIGHT", None
    if any(k in low for k in ("geser atas", "kamera atas", "ke atas", "pan up")):
        return "PTZ_PAN_UP", None
    if any(k in low for k in ("geser bawah", "kamera bawah", "ke bawah", "pan down")):
        return "PTZ_PAN_DOWN", None

    # Overlay & Eye Filter commands
    if any(k in low for k in ("overlay muka saja", "hanya muka", "muka saja", "face overlay only")):
        return "OVERLAY_MODE", "Muka Saja"
    if any(k in low for k in ("overlay tangan saja", "hanya tangan", "tangan saja", "hand overlay only")):
        return "OVERLAY_MODE", "Tangan Saja"
    if any(k in low for k in ("overlay semua", "muka dan tangan", "nyalakan overlay", "show all overlay")):
        return "OVERLAY_MODE", "Muka & Tangan"
    if any(k in low for k in ("matikan overlay", "hapus overlay", "overlay off", "hide overlay")):
        return "OVERLAY_MODE", "Off"
    if any(k in low for k in ("cyber glitch", "glitch keras")):
        return "EYE_FILTER_MODE", "Cyber Glitch"
    if any(k in low for k in ("glitch mata", "efek mata glitch", "mata glitch", "error mata", "sensor mata", "eye glitch")):
        return "EYE_FILTER_MODE", "Glitch Error"
    if any(k in low for k in ("blur mata", "sensor blur", "mata blur", "blur eyes")):
        return "EYE_FILTER_MODE", "Blur Halus"
    if any(k in low for k in ("matikan efek mata", "mata normal", "hapus efek mata", "normal eyes")):
        return "EYE_FILTER_MODE", "Off"

    return None, None


def parse_voice_command(
    text: str,
    keyword_store: Optional[SantoKeywordStore] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Map spoken text to SANTO custom keyword rules or built-in system commands."""
    raw = text.strip()
    norm = _normalize_text(raw)
    if not norm:
        return None, None

    # 1. Check extensible custom rules (like "who am i" -> "Boss Budi, is there anything I can help you with?")
    if keyword_store is not None:
        custom_hit = keyword_store.match_non_wake(norm)
        if custom_hit is not None:
            return custom_hit

    # 2. Check built-in commands (either direct or prefixed with "Santo, ...")
    stripped = _strip_wake_prefix(norm)
    act, arg = _match_builtin_command(raw, stripped)
    if act is not None:
        return act, arg

    # 3. Check wake word ("santo", "hey santo", etc.) -> "Yes, sir."
    if keyword_store is not None:
        wake_hit = keyword_store.match_wake(norm)
        if wake_hit is not None:
            return wake_hit
    elif norm in {"santo", "hey santo", "hello santo", "hi santo", "halo santo", "ok santo", "okay santo"}:
        return "SANTO_WAKE", "Yes, sir."

    return None, None


class VoiceEngine:
    """Handles Microphone STT (sounddevice + SpeechRecognition) and SANTO Jarvis TTS."""

    SAMPLE_RATE = 16000

    def __init__(
        self,
        on_transcript: Optional[Callable[[VoiceTranscriptEvent], None]] = None,
        on_status: Optional[Callable[[VoiceStatus], None]] = None,
        on_speaking_change: Optional[Callable[[bool], None]] = None,
        keyword_store: Optional[SantoKeywordStore] = None,
    ) -> None:
        self._on_transcript = on_transcript
        self._on_status = on_status
        self._on_speaking_change = on_speaking_change
        self.keywords = keyword_store or SantoKeywordStore()

        self.language: str = "en-US"
        self.tts_mode: str = "Santo Jarvis (UK Neural)"
        self.mic_device_index: Optional[int] = None
        self.echo_mode: bool = False
        self.voice_feedback_enabled: bool = True

        self._status = VoiceStatus()
        self._recognizer = sr.Recognizer()
        self._stop_event = threading.Event()
        self._continuous_enabled = False
        self._single_listen_event = threading.Event()
        self._cancel_record_event = threading.Event()
        self._mic_dirty_event = threading.Event()
        self._adaptive_noise_floor: float = 0.006

        self._tts_queue: queue.Queue[Optional[Tuple[str, str]]] = queue.Queue()
        self._tts_speaking_lock = threading.Lock()
        self._pyttsx3_engine = None
        self._pyttsx3_broken = False

        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()

        self._stt_thread = threading.Thread(target=self._stt_worker, daemon=True)
        self._stt_thread.start()

        threading.Thread(target=self._prewarm_santo_cache, daemon=True).start()

    def _emit_status(self, **updates) -> None:
        for k, v in updates.items():
            setattr(self._status, k, v)
        if self._on_status:
            self._on_status(
                VoiceStatus(
                    listening=self._status.listening,
                    continuous=self._status.continuous,
                    speaking=self._status.speaking,
                    mic_level=self._status.mic_level,
                    status_text=self._status.status_text,
                )
            )

    def set_language(self, lang_code: str) -> None:
        self.language = lang_code

    def set_tts_mode(self, mode: str) -> None:
        self.tts_mode = mode

    def set_mic_device(self, device_index: Optional[int]) -> None:
        self.mic_device_index = None if device_index is not None and device_index < 0 else device_index
        self._mic_dirty_event.set()

    def trigger_single_listen(self) -> None:
        if self._status.listening:
            self._cancel_record_event.set()
        else:
            self._cancel_record_event.clear()
            self._single_listen_event.set()

    def set_continuous_listening(self, enabled: bool) -> None:
        self._continuous_enabled = enabled
        if not enabled:
            self._cancel_record_event.set()
        else:
            self._cancel_record_event.clear()
        self._emit_status(continuous=enabled, status_text="SANTO Auto-Listen Active" if enabled else "Siap")

    def speak(self, text: str, force: bool = False) -> None:
        clean = text.strip()
        if not clean:
            return
        if not force and not self.voice_feedback_enabled:
            return
        while self._tts_queue.qsize() > 2:
            try:
                self._tts_queue.get_nowait()
            except queue.Empty:
                break
        lang = "id" if self.language.startswith("id") else "en"
        self._tts_queue.put((clean, lang))

    def shutdown(self) -> None:
        self._stop_event.set()
        self._continuous_enabled = False
        self._cancel_record_event.set()
        self._tts_queue.put(None)

    # ------------------------------------------------------------------
    # TTS Worker & SANTO Jarvis Neural Cache
    # ------------------------------------------------------------------
    def _prewarm_santo_cache(self) -> None:
        """Quietly cache frequent SANTO phrases so wake responses feel instantaneous."""
        try:
            for rule in self.keywords.list_rules():
                resp = str(rule.get("response", "")).strip()
                if resp:
                    self._synthesize_edge_tts_cached(resp)
        except Exception:
            pass

    def _synthesize_edge_tts_cached(self, text: str) -> Optional[Path]:
        voice, rate, pitch = self.keywords.get_voice_settings()
        TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(f"{voice}|{rate}|{pitch}|{text}".encode("utf-8")).hexdigest()[:20]
        cached_path = TTS_CACHE_DIR / f"{key}.mp3"
        if cached_path.exists() and cached_path.stat().st_size > 1000:
            return cached_path

        tmp_path = TTS_CACHE_DIR / f"{key}.{uuid.uuid4().hex[:6]}.tmp"
        try:
            async def _gen():
                comm = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
                await comm.save(str(tmp_path))

            asyncio.run(_gen())
            if tmp_path.exists() and tmp_path.stat().st_size > 500:
                tmp_path.replace(cached_path)
                return cached_path
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
        return None

    @staticmethod
    def _play_mp3_blocking(mp3_path: Path) -> bool:
        alias = f"tts_{uuid.uuid4().hex[:8]}"
        try:
            winmm = ctypes.windll.winmm
            open_cmd = f'open "{mp3_path}" type mpegvideo alias {alias}'
            err = winmm.mciSendStringW(open_cmd, None, 0, None)
            if err != 0:
                return False
            winmm.mciSendStringW(f"play {alias} wait", None, 0, None)
            winmm.mciSendStringW(f"close {alias}", None, 0, None)
            return True
        except Exception:
            try:
                ctypes.windll.winmm.mciSendStringW(f"close {alias}", None, 0, None)
            except Exception:
                pass
            return False

    def _tts_worker(self) -> None:
        while not self._stop_event.is_set():
            item = self._tts_queue.get()
            if item is None:
                break
            text, lang = item
            try:
                if self._on_speaking_change:
                    self._on_speaking_change(True)
                self._emit_status(speaking=True, status_text=f"SANTO Speaking: {text[:30]}...")
                with self._tts_speaking_lock:
                    self._speak_blocking(text, lang)
            finally:
                self._emit_status(
                    speaking=False,
                    status_text="SANTO Auto-Listen Active" if self._continuous_enabled else "Siap",
                )
                if self._on_speaking_change:
                    self._on_speaking_change(False)

    def _speak_blocking(self, text: str, lang: str) -> None:
        if self.tts_mode.startswith("Offline"):
            if self._speak_pyttsx3(text):
                return
            self._speak_powershell_sapi(text)
            return

        if self.tts_mode.startswith("Santo"):
            if self._speak_edge_jarvis(text):
                return
            if self._speak_gtts(text, "en", tld="co.uk"):
                return
            if self._speak_pyttsx3(text):
                return
            self._speak_powershell_sapi(text)
            return

        # Standard gTTS -> fallback
        if self._speak_gtts(text, lang):
            return
        if self._speak_pyttsx3(text):
            return
        self._speak_powershell_sapi(text)

    def _speak_edge_jarvis(self, text: str) -> bool:
        mp3_path = self._synthesize_edge_tts_cached(text)
        if mp3_path is None:
            return False
        return self._play_mp3_blocking(mp3_path)

    def _speak_gtts(self, text: str, lang: str, tld: str = "com") -> bool:
        tmp_path: Optional[Path] = None
        try:
            tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
            fd, raw_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            tmp_path = Path(raw_path)
            tts.save(str(tmp_path))
            return self._play_mp3_blocking(tmp_path)
        except Exception:
            return False
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    def _speak_pyttsx3(self, text: str) -> bool:
        if pyttsx3 is None or self._pyttsx3_broken:
            return False
        try:
            if pythoncom is not None:
                pythoncom.CoInitialize()
            if self._pyttsx3_engine is None:
                self._pyttsx3_engine = pyttsx3.init()
                self._pyttsx3_engine.setProperty("rate", 158)
                voices = self._pyttsx3_engine.getProperty("voices") or []
                for v in voices:
                    v_info = f"{v.name} {getattr(v, 'id', '')}".lower()
                    if any(m in v_info for m in ("george", "ryan", "daniel", "david", "mark", "male")):
                        self._pyttsx3_engine.setProperty("voice", v.id)
                        break
            self._pyttsx3_engine.say(text)
            self._pyttsx3_engine.runAndWait()
            return True
        except Exception:
            self._pyttsx3_broken = True
            self._pyttsx3_engine = None
            return False

    @staticmethod
    def _speak_powershell_sapi(text: str) -> bool:
        try:
            safe_text = text.replace("'", "''")
            cmd = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Male); "
                "$s.Rate = -1; "
                f"$s.Speak('{safe_text}')"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", cmd],
                timeout=12,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # STT Worker (sounddevice + VAD + SpeechRecognition)
    # ------------------------------------------------------------------
    def _stt_worker(self) -> None:
        while not self._stop_event.is_set():
            if not self._continuous_enabled and not self._single_listen_event.is_set():
                time.sleep(0.08)
                continue

            if self._status.speaking:
                time.sleep(0.12)
                continue

            is_single = self._single_listen_event.is_set()
            self._single_listen_event.clear()
            self._cancel_record_event.clear()

            pcm_bytes = self._record_utterance(single_shot=is_single)
            if not pcm_bytes:
                continue

            self._emit_status(listening=False, mic_level=0.0, status_text="Memproses suara...")
            audio_data = sr.AudioData(pcm_bytes, self.SAMPLE_RATE, 2)
            try:
                transcript = self._recognizer.recognize_google(audio_data, language=self.language)
                transcript = transcript.strip()
                if transcript:
                    cmd_action, cmd_arg = parse_voice_command(transcript, self.keywords)
                    event = VoiceTranscriptEvent(
                        text=transcript,
                        language=self.language,
                        command_action=cmd_action,
                        command_arg=cmd_arg,
                    )
                    if self._on_transcript:
                        self._on_transcript(event)
                    self._emit_status(status_text=f"Terdengar: {transcript[:36]}")
            except sr.UnknownValueError:
                if is_single:
                    self._emit_status(status_text="Suara tidak terdengar jelas, coba lagi")
            except sr.RequestError:
                self._emit_status(status_text="STT butuh koneksi internet (Google Speech)")
                time.sleep(1.0)
            except Exception:
                self._emit_status(status_text="Gagal mengenali suara")

    def _record_utterance(self, single_shot: bool) -> Optional[bytes]:
        block_size = 1600  # 100 ms chunks at 16kHz
        captured_chunks: List[np.ndarray] = []
        pre_buffer: List[np.ndarray] = []

        self._mic_dirty_event.clear()
        self._emit_status(
            listening=single_shot,
            status_text="Mendengarkan..." if single_shot else "SANTO Standby (Silakan Bicara)...",
        )

        speech_started = False
        silence_blocks = 0
        max_utterance_blocks = 100  # max 10 seconds once speech begins
        single_idle_limit = 70      # max 7 seconds waiting in single-shot mode
        idle_blocks = 0
        min_speech_blocks = 2
        speech_blocks = 0
        echo_guard_blocks = 0

        try:
            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=block_size,
                device=self.mic_device_index,
            ) as stream:
                while not self._stop_event.is_set() and not self._cancel_record_event.is_set():
                    if self._mic_dirty_event.is_set():
                        break
                    if not single_shot and not self._continuous_enabled:
                        break

                    chunk, _ = stream.read(block_size)
                    mono = chunk[:, 0]

                    # Ignore audio while TTS speaks (+0.3s tail) so SANTO never hears himself
                    if self._status.speaking:
                        echo_guard_blocks = 3
                        if not speech_started:
                            pre_buffer.clear()
                            self._emit_status(mic_level=0.0)
                        continue
                    if echo_guard_blocks > 0:
                        echo_guard_blocks -= 1
                        pre_buffer.clear()
                        continue

                    rms = float(np.sqrt(np.mean((mono.astype(np.float32) / 32768.0) ** 2)))
                    self._emit_status(mic_level=min(1.0, rms * 14.0))

                    threshold = max(0.0075, self._adaptive_noise_floor * 1.95)
                    if rms > threshold:
                        if not speech_started:
                            speech_started = True
                            captured_chunks.extend(pre_buffer[-5:])
                            self._emit_status(listening=True, status_text="Merekam ucapan Anda...")
                        captured_chunks.append(mono.copy())
                        speech_blocks += 1
                        silence_blocks = 0
                        if speech_blocks >= max_utterance_blocks:
                            break
                    else:
                        if speech_started:
                            captured_chunks.append(mono.copy())
                            silence_blocks += 1
                            if silence_blocks >= 8:  # ~0.8s pause finishes utterance
                                break
                        else:
                            # Smoothly adapt ambient noise floor only on quiet background frames
                            if rms < self._adaptive_noise_floor * 2.2:
                                self._adaptive_noise_floor = float(
                                    np.clip(0.90 * self._adaptive_noise_floor + 0.10 * rms, 0.0025, 0.022)
                                )
                            pre_buffer.append(mono.copy())
                            if len(pre_buffer) > 5:
                                pre_buffer.pop(0)
                            if single_shot:
                                idle_blocks += 1
                                if idle_blocks >= single_idle_limit:
                                    break

        except Exception as exc:
            self._emit_status(listening=False, mic_level=0.0, status_text=f"Error Mikrofon: {exc}")
            time.sleep(0.8)
            return None
        finally:
            self._emit_status(listening=False, mic_level=0.0)

        if not speech_started or speech_blocks < min_speech_blocks or not captured_chunks:
            return None

        joined = np.concatenate(captured_chunks).astype(np.int16)
        return joined.tobytes()
