"""Configuration constants and gesture/voice mappings for Smart Vision Studio."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = ASSETS_DIR / "models"
MUSIC_DIR = ASSETS_DIR / "music"
PLAYLIST_FILE = ASSETS_DIR / "playlist.json"
SANTO_KEYWORDS_FILE = ASSETS_DIR / "santo_keywords.json"
TTS_CACHE_DIR = ASSETS_DIR / "tts_cache"

DEMO_FILENAMES = {
    "01_LoFi_Chill_Focus.wav",
    "02_Neon_Synth_Wave.wav",
    "03_Ambient_Zen_Pad.wav",
    "04_Upbeat_Gesture_Groove.wav",
}

# MediaPipe official Task model URLs (used if local model files are not yet cached)
HAND_MODEL_FILENAME = "hand_landmarker.task"
FACE_MODEL_FILENAME = "face_landmarker.task"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Camera & PTZ settings
DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
DEFAULT_FPS = 30
PTZ_MIN_ZOOM = 1.0
PTZ_MAX_ZOOM = 3.0
PTZ_SMOOTHING = 0.14  # Lower = smoother camera movement

# Tracking modes
TRACKING_MODES = [
    "Auto Face",
    "Auto Hand",
    "Auto Hybrid",
    "Manual PTZ",
]

# Overlay & Eye Filter modes
OVERLAY_MODES = [
    "Muka & Tangan",
    "Muka Saja",
    "Tangan Saja",
    "Off",
]

EYE_FILTER_MODES = [
    "Off",
    "Glitch Error",
    "Blur Halus",
    "Cyber Glitch",
]

SCREEN_GLITCH_MODES = [
    "Auto (Kacamata)",
    "Selalu Aktif",
    "Off",
]

# Hand gesture -> Music action mapping
GESTURE_LABELS_ID = {
    "OPEN_PALM": "Telapak Terbuka (5 Jari)",
    "FIST": "Kepalan Tangan (0 Jari)",
    "VICTORY": "Dua Jari / Peace (2 Jari)",
    "THREE_FINGERS": "Tiga Jari (3 Jari)",
    "POINT_UP": "Telunjuk ke Atas (1 Jari)",
    "THUMBS_UP": "Jempol ke Atas",
    "THUMBS_DOWN": "Jempol ke Bawah",
    "PINCH": "Cubit / Pinch Volume",
    "ROCK": "Rock / Shaka",
    "SWIPE_RIGHT": "Geser Kanan (Swipe)",
    "SWIPE_LEFT": "Geser Kiri (Swipe)",
    "NONE": "Tidak Ada Gesture",
}

GESTURE_MUSIC_ACTIONS = {
    "OPEN_PALM": ("PLAY_PAUSE", "Play / Pause Musik"),
    "FIST": ("STOP_PAUSE", "Pause / Hentikan Musik"),
    "VICTORY": ("NEXT_TRACK", "Lagu Berikutnya"),
    "THREE_FINGERS": ("PREV_TRACK", "Lagu Sebelumnya"),
    "POINT_UP": ("VOLUME_UP", "Volume Naik (+10%)"),
    "THUMBS_DOWN": ("VOLUME_DOWN", "Volume Turun (-10%)"),
    "THUMBS_UP": ("ANNOUNCE_TRACK", "Umumkan Lagu (Suara)"),
    "ROCK": ("MUTE_TOGGLE", "Mute / Unmute"),
    "SWIPE_RIGHT": ("NEXT_TRACK", "Lagu Berikutnya (Swipe)"),
    "SWIPE_LEFT": ("PREV_TRACK", "Lagu Sebelumnya (Swipe)"),
}

# Hold duration (seconds) required to trigger discrete hand gestures cleanly
GESTURE_HOLD_SECONDS = 0.55
GESTURE_COOLDOWN_SECONDS = 1.4

# UI Theme Palette (Black Hacker / Cyber Terminal)
COLORS = {
    "bg_root": "#000000",
    "bg_panel": "#050A06",
    "bg_card": "#09120B",
    "bg_elevated": "#102115",
    "accent": "#00FF41",
    "accent_hover": "#00C832",
    "emerald": "#00FF41",
    "cyan": "#00F0FF",
    "amber": "#FACC15",
    "rose": "#FF003C",
    "violet": "#00F0FF",
    "text_primary": "#D8FFE0",
    "text_secondary": "#5DBB75",
    "border": "#14532D",
    "border_bright": "#00FF41",
}
