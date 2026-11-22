"""Modern CustomTkinter Desktop GUI for Smart Vision Studio."""

from __future__ import annotations

import queue
import random
import threading
import time
import tkinter as tk
from tkinter import filedialog
from typing import Dict, List, Optional

import customtkinter as ctk
from PIL import Image, ImageTk

from .config import (
    COLORS,
    EYE_FILTER_MODES,
    GESTURE_MUSIC_ACTIONS,
    OVERLAY_MODES,
    SCREEN_GLITCH_MODES,
    TRACKING_MODES,
)
from .music_player import MusicPlayer, TrackInfo
from .ptz_camera import PTZController
from .vision_engine import VisionEngine, VisionFrameResult
from .voice_engine import VoiceEngine, VoiceStatus, VoiceTranscriptEvent, list_microphones


def _fmt_time(seconds: float) -> str:
    sec = max(0, int(seconds))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _mono(size: int, bold: bool = False) -> ctk.CTkFont:
    return ctk.CTkFont(family="Consolas", size=size, weight="bold" if bold else "normal")


class SmartVisionStudioApp(ctk.CTk):
    """Main desktop application window."""

    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("[ROOT@VISION-OS]:~# SMART_VISION_STUDIO // PTZ • GESTURE • VOICE • AUDIO")
        self.geometry("1380x840")
        self.minsize(1160, 720)
        self.configure(fg_color=COLORS["bg_root"])

        # Core engines
        self.ptz = PTZController()
        self.music = MusicPlayer(on_track_change=self._on_track_changed)
        self.vision = VisionEngine(self.ptz)

        self._ui_events: "queue.Queue[tuple]" = queue.Queue()
        self.voice = VoiceEngine(
            on_transcript=lambda ev: self._ui_events.put(("transcript", ev)),
            on_status=lambda st: self._ui_events.put(("voice_status", st)),
            on_speaking_change=lambda speaking: self.music.set_ducking(speaking),
        )

        # Runtime flags
        self._running = True
        self._gesture_music_enabled = True
        self._face_voice_reaction = False
        self._last_face_react_ts = 0.0
        self._last_face_react_key = ""
        self._glasses_auto_play = True
        self._glasses_pause_on_remove = False
        self._glasses_currently_on = False
        self._camera_only_mode = False
        self._ui_glitch_enabled = True
        self._ui_glitch_burst_until = 0.0
        self._next_ambient_glitch_ts = 0.0
        self._glitch_active_frame = False
        self._glitch_panels: List[ctk.CTkFrame] = []

        # Latest vision frame holder (single-slot overwrite prevents lag)
        self._frame_lock = threading.Lock()
        self._latest_vision: Optional[VisionFrameResult] = None
        self._photo_ref: Optional[ImageTk.PhotoImage] = None
        self._canvas_img_id: Optional[int] = None
        self._playlist_buttons: List[ctk.CTkButton] = []
        self._star_buttons: List[ctk.CTkButton] = []
        self._playlist_rows: List[ctk.CTkBaseClass] = []
        self._last_playlist_count: int = -1
        self._last_active_idx: int = -1
        self._last_marked_idx: Optional[int] = -999
        self._updating_vol_slider = False

        # Build UI
        self._build_layout()

        # Open default camera asynchronously so UI appears immediately
        threading.Thread(target=self._init_camera_async, daemon=True).start()

        # Start Vision worker thread
        self._vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self._vision_thread.start()

        # Bind keyboard shortcuts & window close
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Greet user in SANTO Jarvis style after startup
        self.after(900, lambda: self.voice.speak("System online. Santo at your service, sir.", force=False))
        self.after(25, self._ui_tick)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=13)
        self.grid_columnconfigure(1, weight=8)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_left_vision_column()
        self._build_right_audio_voice_column()

    def _build_header(self) -> None:
        self.header = ctk.CTkFrame(
            self,
            fg_color=COLORS["bg_panel"],
            border_width=1,
            border_color=COLORS["border"],
            corner_radius=6,
            height=56,
        )
        self.header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 6))
        self.header.grid_columnconfigure(1, weight=1)
        self._glitch_panels.append(self.header)

        title_box = ctk.CTkFrame(self.header, fg_color="transparent")
        title_box.grid(row=0, column=0, sticky="w", padx=12, pady=8)
        self.lbl_header_prompt = ctk.CTkLabel(
            title_box,
            text="[ROOT@VISION-OS]:~#",
            font=_mono(16, bold=True),
            text_color=COLORS["accent"],
        )
        self.lbl_header_prompt.pack(side="left")
        self.lbl_header_sub = ctk.CTkLabel(
            title_box,
            text="  PTZ_CAM // HAND_FACE // GLASSES_AUDIO // SANTO_AI",
            font=_mono(12),
            text_color=COLORS["text_secondary"],
        )
        self.lbl_header_sub.pack(side="left")

        right_pills = ctk.CTkFrame(self.header, fg_color="transparent")
        right_pills.grid(row=0, column=1, sticky="e", padx=12, pady=8)

        self.lbl_fps = ctk.CTkLabel(
            right_pills,
            text="FPS: --",
            font=_mono(12, bold=True),
            text_color=COLORS["accent"],
            fg_color=COLORS["bg_card"],
            corner_radius=4,
            padx=10,
            pady=4,
        )
        self.lbl_fps.pack(side="left", padx=6)

        self.btn_cam_only = ctk.CTkButton(
            right_pills,
            text="⛶ Cam Only",
            width=88,
            height=26,
            font=_mono(10, bold=True),
            fg_color=COLORS["bg_card"],
            border_width=1,
            border_color=COLORS["cyan"],
            hover_color=COLORS["bg_elevated"],
            text_color=COLORS["cyan"],
            command=self._toggle_camera_only_mode,
        )
        self.btn_cam_only.pack(side="left", padx=6)

        self.sw_ui_glitch = ctk.CTkSwitch(
            right_pills,
            text="GLITCH_UI",
            font=_mono(11, bold=True),
            text_color=COLORS["rose"],
            command=self._toggle_ui_glitch,
            progress_color=COLORS["rose"],
            button_color=COLORS["rose"],
        )
        self.sw_ui_glitch.select()
        self.sw_ui_glitch.pack(side="left", padx=6)

        self.sw_gesture_music = ctk.CTkSwitch(
            right_pills,
            text="GESTURE_AUDIO",
            font=_mono(11, bold=True),
            text_color=COLORS["text_primary"],
            command=self._toggle_gesture_music,
            progress_color=COLORS["accent"],
            button_color="#00FF41",
        )
        self.sw_gesture_music.select()
        self.sw_gesture_music.pack(side="left", padx=8)

        self.sw_voice_feedback = ctk.CTkSwitch(
            right_pills,
            text="TTS_VOICE",
            font=_mono(11, bold=True),
            text_color=COLORS["text_primary"],
            command=self._toggle_voice_feedback,
            progress_color=COLORS["cyan"],
            button_color=COLORS["cyan"],
        )
        self.sw_voice_feedback.select()
        self.sw_voice_feedback.pack(side="left", padx=6)

    def _build_left_vision_column(self) -> None:
        self.left_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.left_panel.grid(row=1, column=0, sticky="nsew", padx=(10, 6), pady=(0, 10))
        self.left_panel.grid_rowconfigure(0, weight=5)
        self.left_panel.grid_rowconfigure(2, weight=2)
        self.left_panel.grid_columnconfigure(0, weight=1)

        # 1. Camera viewport container
        self.cam_card = ctk.CTkFrame(
            self.left_panel,
            fg_color=COLORS["bg_panel"],
            border_width=1,
            border_color=COLORS["border"],
            corner_radius=6,
        )
        self.cam_card.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        self.cam_card.grid_rowconfigure(0, weight=1)
        self.cam_card.grid_columnconfigure(0, weight=1)
        self._glitch_panels.append(self.cam_card)

        self.video_canvas = tk.Canvas(
            self.cam_card,
            bg="#010302",
            highlightthickness=1,
            highlightbackground="#0D3B1E",
            bd=0,
        )
        self.video_canvas.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.video_canvas.bind("<Double-Button-1>", lambda _e: self._toggle_camera_only_mode())

        self.btn_exit_cam_only = ctk.CTkButton(
            self.cam_card,
            text="◧ Tampilkan Panel (F / Esc)",
            width=180,
            height=28,
            font=_mono(11, bold=True),
            fg_color="#050A06",
            border_width=1,
            border_color=COLORS["accent"],
            hover_color="#0D3B1E",
            text_color=COLORS["accent"],
            command=self._toggle_camera_only_mode,
        )

        # 2. Camera & PTZ Control Bar (PTZ + Overlay/Eye Filter + Screen Glitch)
        self.ptz_bar = ctk.CTkFrame(
            self.left_panel,
            fg_color=COLORS["bg_panel"],
            border_width=1,
            border_color=COLORS["border"],
            corner_radius=6,
        )
        self.ptz_bar.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.ptz_bar.grid_columnconfigure(1, weight=1)
        self._glitch_panels.append(self.ptz_bar)

        # Row 0 Left: Camera source + Mirror
        cam_ctrl = ctk.CTkFrame(self.ptz_bar, fg_color="transparent")
        cam_ctrl.grid(row=0, column=0, sticky="w", padx=10, pady=(8, 4))

        self.cam_combo = ctk.CTkComboBox(
            cam_ctrl,
            values=["Kamera 0", "Kamera 1", "Kamera 2"],
            width=112,
            font=_mono(11),
            fg_color=COLORS["bg_card"],
            border_color=COLORS["border"],
            button_color=COLORS["bg_elevated"],
            text_color=COLORS["text_primary"],
            command=self._on_camera_select,
        )
        self.cam_combo.set("Kamera 0")
        self.cam_combo.pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            cam_ctrl,
            text="↻ CAM",
            width=70,
            font=_mono(11, bold=True),
            fg_color=COLORS["bg_card"],
            border_width=1,
            border_color=COLORS["border"],
            hover_color=COLORS["bg_elevated"],
            text_color=COLORS["accent"],
            command=self._restart_camera,
        ).pack(side="left", padx=(0, 8))

        self.sw_mirror = ctk.CTkSwitch(
            cam_ctrl,
            text="Mirror",
            width=70,
            font=_mono(11),
            progress_color=COLORS["accent"],
            command=self._toggle_mirror,
        )
        self.sw_mirror.select()
        self.sw_mirror.pack(side="left", padx=4)

        # Row 0 Right: Tracking Mode + Manual PTZ buttons
        ptz_ctrl = ctk.CTkFrame(self.ptz_bar, fg_color="transparent")
        ptz_ctrl.grid(row=0, column=1, sticky="e", padx=10, pady=(8, 4))

        ctk.CTkLabel(
            ptz_ctrl,
            text="PTZ_MODE:",
            font=_mono(11, bold=True),
            text_color=COLORS["text_secondary"],
        ).pack(side="left", padx=(0, 6))

        self.ptz_mode_combo = ctk.CTkComboBox(
            ptz_ctrl,
            values=TRACKING_MODES,
            width=126,
            font=_mono(11),
            fg_color=COLORS["bg_card"],
            border_color=COLORS["border"],
            button_color=COLORS["bg_elevated"],
            text_color=COLORS["text_primary"],
            command=self._on_ptz_mode_change,
        )
        self.ptz_mode_combo.set("Auto Face")
        self.ptz_mode_combo.pack(side="left", padx=(0, 6))

        self.sw_auto_zoom = ctk.CTkSwitch(
            ptz_ctrl,
            text="AutoZoom",
            width=85,
            font=_mono(11),
            progress_color=COLORS["accent"],
            command=self._toggle_auto_zoom,
        )
        self.sw_auto_zoom.select()
        self.sw_auto_zoom.pack(side="left", padx=(0, 8))

        # Manual PTZ pad
        btn_specs = [
            ("◀", lambda: self._manual_nudge(-0.08, 0.0)),
            ("▲", lambda: self._manual_nudge(0.0, -0.08)),
            ("▼", lambda: self._manual_nudge(0.0, 0.08)),
            ("▶", lambda: self._manual_nudge(0.08, 0.0)),
            ("+", lambda: self.ptz.adjust_zoom(0.22)),
            ("-", lambda: self.ptz.adjust_zoom(-0.22)),
            ("RST", self._reset_ptz),
        ]
        for label, cb in btn_specs:
            w = 46 if label == "RST" else 28
            ctk.CTkButton(
                ptz_ctrl,
                text=label,
                width=w,
                height=26,
                font=_mono(11, bold=True),
                fg_color=COLORS["bg_card"],
                border_width=1,
                border_color=COLORS["border"],
                hover_color=COLORS["bg_elevated"],
                text_color=COLORS["accent"],
                command=cb,
            ).pack(side="left", padx=2)

        # Row 1: Selective Overlay (Muka & Tangan / Muka Saja / Tangan Saja / Off) + Eye Glitch Filter
        fx_row = ctk.CTkFrame(self.ptz_bar, fg_color="transparent")
        fx_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(2, 8))
        fx_row.grid_columnconfigure(1, weight=1)

        ov_box = ctk.CTkFrame(fx_row, fg_color="transparent")
        ov_box.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            ov_box,
            text="OVERLAY:",
            font=_mono(11, bold=True),
            text_color=COLORS["accent"],
        ).pack(side="left", padx=(0, 6))
        self.seg_overlay = ctk.CTkSegmentedButton(
            ov_box,
            values=OVERLAY_MODES,
            font=_mono(11, bold=True),
            fg_color=COLORS["bg_card"],
            unselected_color=COLORS["bg_card"],
            unselected_hover_color=COLORS["bg_elevated"],
            selected_color="#0D3B1E",
            selected_hover_color="#14532D",
            text_color=COLORS["accent"],
            command=self._on_overlay_mode_change,
        )
        self.seg_overlay.set("Muka & Tangan")
        self.seg_overlay.pack(side="left")

        eye_box = ctk.CTkFrame(fx_row, fg_color="transparent")
        eye_box.grid(row=0, column=1, sticky="e")
        ctk.CTkLabel(
            eye_box,
            text="EYE_FX:",
            font=_mono(11, bold=True),
            text_color=COLORS["rose"],
        ).pack(side="left", padx=(12, 6))
        self.seg_eye_filter = ctk.CTkSegmentedButton(
            eye_box,
            values=EYE_FILTER_MODES,
            font=_mono(11, bold=True),
            fg_color=COLORS["bg_card"],
            unselected_color=COLORS["bg_card"],
            unselected_hover_color="#230B12",
            selected_color="#7F1D1D",
            selected_hover_color="#991B1B",
            text_color="#FFE4E6",
            command=self._on_eye_filter_change,
        )
        self.seg_eye_filter.set("Glitch Error")
        self.vision.set_eye_filter_mode("Glitch Error")
        self.seg_eye_filter.pack(side="left")

        # Row 2: Full-Screen Glitch Mode + Tether Glitch Box + Glasses Auto-Play Controls
        gl_row = ctk.CTkFrame(self.ptz_bar, fg_color="transparent")
        gl_row.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 8))
        gl_row.grid_columnconfigure(1, weight=1)

        gl_left = ctk.CTkFrame(gl_row, fg_color="transparent")
        gl_left.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            gl_left,
            text="GLITCH_CAM:",
            font=_mono(11, bold=True),
            text_color=COLORS["amber"],
        ).pack(side="left", padx=(0, 6))
        self.seg_screen_glitch = ctk.CTkSegmentedButton(
            gl_left,
            values=SCREEN_GLITCH_MODES,
            font=_mono(10, bold=True),
            fg_color=COLORS["bg_card"],
            unselected_color=COLORS["bg_card"],
            unselected_hover_color=COLORS["bg_elevated"],
            selected_color="#422006",
            selected_hover_color="#713F12",
            text_color=COLORS["amber"],
            command=self._on_screen_glitch_change,
        )
        self.seg_screen_glitch.set("Auto (Kacamata)")
        self.seg_screen_glitch.pack(side="left")

        gl_right = ctk.CTkFrame(gl_row, fg_color="transparent")
        gl_right.grid(row=0, column=1, sticky="e")

        self.sw_tether_box = ctk.CTkSwitch(
            gl_right,
            text="Tether Box",
            font=_mono(10, bold=True),
            text_color=COLORS["cyan"],
            progress_color=COLORS["cyan"],
            command=self._toggle_tether_box,
        )
        self.sw_tether_box.select()
        self.sw_tether_box.pack(side="left", padx=(0, 8))

        self.sw_glasses_play = ctk.CTkSwitch(
            gl_right,
            text="Kacamata ★",
            font=_mono(10, bold=True),
            text_color=COLORS["amber"],
            progress_color=COLORS["amber"],
            command=self._toggle_glasses_autoplay,
        )
        self.sw_glasses_play.select()
        self.sw_glasses_play.pack(side="left", padx=(0, 8))

        self.sw_glasses_pause = ctk.CTkSwitch(
            gl_right,
            text="Pause Lepas",
            font=_mono(10),
            text_color=COLORS["text_secondary"],
            progress_color=COLORS["rose"],
            command=self._toggle_glasses_pause,
        )
        self.sw_glasses_pause.pack(side="left")

        # 3. Bottom Telemetry Row: Hand Card + Face Card
        self.telem_row = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.telem_row.grid(row=2, column=0, sticky="nsew")
        self.telem_row.grid_columnconfigure(0, weight=1)
        self.telem_row.grid_columnconfigure(1, weight=1)
        self.telem_row.grid_rowconfigure(0, weight=1)

        # Hand Card
        hand_card = ctk.CTkFrame(
            self.telem_row,
            fg_color=COLORS["bg_panel"],
            border_width=1,
            border_color=COLORS["border"],
            corner_radius=6,
        )
        hand_card.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self._glitch_panels.append(hand_card)

        ctk.CTkLabel(
            hand_card,
            text="[HAND_TRACKER // GESTURE_CTRL]",
            font=_mono(11, bold=True),
            text_color=COLORS["accent"],
        ).pack(anchor="w", padx=12, pady=(8, 4))

        self.lbl_hand_gesture = ctk.CTkLabel(
            hand_card,
            text="Gesture: Tidak Ada Gesture",
            font=_mono(14, bold=True),
            text_color=COLORS["text_primary"],
        )
        self.lbl_hand_gesture.pack(anchor="w", padx=12)

        self.lbl_hand_meta = ctk.CTkLabel(
            hand_card,
            text="Jari: 0 | Tangan: - | Aksi: -",
            font=_mono(11),
            text_color=COLORS["text_secondary"],
        )
        self.lbl_hand_meta.pack(anchor="w", padx=12, pady=(2, 6))

        self.hand_hold_bar = ctk.CTkProgressBar(
            hand_card,
            height=7,
            fg_color=COLORS["bg_card"],
            progress_color=COLORS["accent"],
        )
        self.hand_hold_bar.set(0.0)
        self.hand_hold_bar.pack(fill="x", padx=12, pady=(0, 8))

        guide_text = (
            "✋ 5 Jari: Play/Pause | ✊ Kepal: Stop/Jeda | ✌️ 2 Jari/Swipe ➡: Next\n"
            "🤟 3 Jari/Swipe ⬅: Prev | ☝️ 1 Jari: Vol+ | 👎 Jempol Bawah: Vol-\n"
            "👌 Pinch + Naik/Turun: Slide Vol | 👍 Jempol Atas: Info | 🤘 Rock: Mute"
        )
        ctk.CTkLabel(
            hand_card,
            text=guide_text,
            font=_mono(10),
            text_color=COLORS["text_secondary"],
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 8))

        # Face & Glasses Card
        face_card = ctk.CTkFrame(
            self.telem_row,
            fg_color=COLORS["bg_panel"],
            border_width=1,
            border_color=COLORS["border"],
            corner_radius=6,
        )
        face_card.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self._glitch_panels.append(face_card)

        face_hdr = ctk.CTkFrame(face_card, fg_color="transparent")
        face_hdr.pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkLabel(
            face_hdr,
            text="[FACE_SCANNER // GLASSES_TRIGGER]",
            font=_mono(11, bold=True),
            text_color=COLORS["cyan"],
        ).pack(side="left")

        self.sw_face_voice = ctk.CTkSwitch(
            face_hdr,
            text="TTS Reaksi",
            font=_mono(10),
            width=40,
            progress_color=COLORS["cyan"],
            command=self._toggle_face_reaction,
        )
        self.sw_face_voice.pack(side="right")

        self.lbl_face_expr = ctk.CTkLabel(
            face_card,
            text="Ekspresi: Menunggu Wajah...",
            font=_mono(13, bold=True),
            text_color=COLORS["text_primary"],
        )
        self.lbl_face_expr.pack(anchor="w", padx=12)

        self.lbl_face_pose = ctk.CTkLabel(
            face_card,
            text="Posisi Kepala: Lurus ke Depan | Kedipan: 0",
            font=_mono(11),
            text_color=COLORS["amber"],
        )
        self.lbl_face_pose.pack(anchor="w", padx=12, pady=(1, 2))

        self.lbl_face_angles = ctk.CTkLabel(
            face_card,
            text="Yaw: 0.0° | Pitch: 0.0° | Roll: 0.0° | Senyum: 0% | Mulut: 0%",
            font=_mono(10),
            text_color=COLORS["text_secondary"],
        )
        self.lbl_face_angles.pack(anchor="w", padx=12, pady=(0, 4))

        # Glasses Detection Status + Sensitivity + Calibrate Row
        g_row = ctk.CTkFrame(face_card, fg_color=COLORS["bg_card"], corner_radius=4)
        g_row.pack(fill="x", padx=12, pady=(0, 8))

        self.lbl_glasses_status = ctk.CTkLabel(
            g_row,
            text="KACAMATA: STANDBY (0%)",
            font=_mono(11, bold=True),
            text_color=COLORS["text_secondary"],
        )
        self.lbl_glasses_status.pack(side="left", padx=8, pady=5)

        ctk.CTkButton(
            g_row,
            text="Kalibrasi",
            width=72,
            height=22,
            font=_mono(10, bold=True),
            fg_color=COLORS["bg_panel"],
            border_width=1,
            border_color=COLORS["cyan"],
            hover_color=COLORS["bg_elevated"],
            text_color=COLORS["cyan"],
            command=self._calibrate_no_glasses,
        ).pack(side="right", padx=6, pady=4)

        self.glasses_sens_slider = ctk.CTkSlider(
            g_row,
            from_=0.0,
            to=1.0,
            width=84,
            number_of_steps=20,
            fg_color=COLORS["bg_panel"],
            progress_color=COLORS["amber"],
            button_color=COLORS["amber"],
            command=self._on_glasses_sens_change,
        )
        self.glasses_sens_slider.set(0.52)
        self.glasses_sens_slider.pack(side="right", padx=4)

        ctk.CTkLabel(
            g_row,
            text="Sens:",
            font=_mono(10),
            text_color=COLORS["text_secondary"],
        ).pack(side="right")

    def _build_right_audio_voice_column(self) -> None:
        self.right_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.right_panel.grid(row=1, column=1, sticky="nsew", padx=(4, 10), pady=(0, 10))
        self.right_panel.grid_rowconfigure(0, weight=5)
        self.right_panel.grid_rowconfigure(1, weight=6)
        self.right_panel.grid_columnconfigure(0, weight=1)

        # --------------------------------------------------------------
        # 1. Music Player Panel
        # --------------------------------------------------------------
        music_card = ctk.CTkFrame(
            self.right_panel,
            fg_color=COLORS["bg_panel"],
            border_width=1,
            border_color=COLORS["border"],
            corner_radius=6,
        )
        music_card.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        music_card.grid_columnconfigure(0, weight=1)
        music_card.grid_rowconfigure(5, weight=1)
        self._glitch_panels.append(music_card)

        m_hdr = ctk.CTkFrame(music_card, fg_color="transparent")
        m_hdr.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 2))
        ctk.CTkLabel(
            m_hdr,
            text="[AUDIO_PLAYER // GESTURE_SYNC]",
            font=_mono(12, bold=True),
            text_color=COLORS["accent"],
        ).pack(side="left")

        ctk.CTkButton(
            m_hdr,
            text="Clear",
            width=52,
            height=24,
            font=_mono(10, bold=True),
            fg_color=COLORS["bg_card"],
            border_width=1,
            border_color="#7F1D1D",
            hover_color="#3B0D16",
            text_color=COLORS["rose"],
            command=self._clear_music_playlist,
        ).pack(side="right", padx=(4, 0))
        ctk.CTkButton(
            m_hdr,
            text="📁 Folder",
            width=72,
            height=24,
            font=_mono(10, bold=True),
            fg_color=COLORS["bg_card"],
            border_width=1,
            border_color=COLORS["border"],
            hover_color=COLORS["bg_elevated"],
            text_color=COLORS["text_primary"],
            command=self._add_music_folder,
        ).pack(side="right", padx=4)
        ctk.CTkButton(
            m_hdr,
            text="+ Lagu",
            width=66,
            height=24,
            font=_mono(10, bold=True),
            fg_color="#0D3B1E",
            border_width=1,
            border_color=COLORS["accent"],
            hover_color="#14532D",
            text_color=COLORS["accent"],
            command=self._add_music_files,
        ).pack(side="right")

        self.lbl_track_title = ctk.CTkLabel(
            music_card,
            text="Playlist kosong — Tambah Lagu Anda",
            font=_mono(14, bold=True),
            text_color=COLORS["text_primary"],
            anchor="w",
        )
        self.lbl_track_title.grid(row=1, column=0, sticky="ew", padx=12, pady=(2, 0))

        self.lbl_track_time = ctk.CTkLabel(
            music_card,
            text="Status: Berhenti  •  00:00 / 00:00",
            font=_mono(11),
            text_color=COLORS["text_secondary"],
            anchor="w",
        )
        self.lbl_track_time.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 2))

        self.lbl_marked_track = ctk.CTkLabel(
            music_card,
            text="★ TARGET KACAMATA: Belum ditandai (klik ☆ pada lagu)",
            font=_mono(10, bold=True),
            text_color=COLORS["amber"],
            anchor="w",
        )
        self.lbl_marked_track.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 4))

        # Spectrum visualizer canvas
        self.spec_canvas = tk.Canvas(
            music_card,
            height=42,
            bg="#010302",
            highlightthickness=1,
            highlightbackground="#0D3B1E",
            bd=0,
        )
        self.spec_canvas.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 6))

        # Playlist container
        self.playlist_frame = ctk.CTkScrollableFrame(
            music_card,
            fg_color=COLORS["bg_card"],
            border_width=1,
            border_color=COLORS["border"],
            corner_radius=4,
            height=86,
        )
        self.playlist_frame.grid(row=5, column=0, sticky="nsew", padx=12, pady=(0, 6))

        # Transport & volume controls
        ctrl_row = ctk.CTkFrame(music_card, fg_color="transparent")
        ctrl_row.grid(row=6, column=0, sticky="ew", padx=12, pady=(0, 10))
        ctrl_row.grid_columnconfigure(5, weight=1)

        btn_style = dict(
            width=36,
            height=30,
            font=_mono(12, bold=True),
            fg_color=COLORS["bg_card"],
            border_width=1,
            border_color=COLORS["border"],
            hover_color=COLORS["bg_elevated"],
            text_color=COLORS["accent"],
        )
        ctk.CTkButton(ctrl_row, text="⏮", command=self._music_prev, **btn_style).grid(row=0, column=0, padx=2)
        self.btn_play_pause = ctk.CTkButton(
            ctrl_row,
            text="▶ PLAY",
            width=82,
            height=30,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            text_color="#001A07",
            font=_mono(12, bold=True),
            command=self._music_play_pause,
        )
        self.btn_play_pause.grid(row=0, column=1, padx=4)
        ctk.CTkButton(ctrl_row, text="⏹", command=self.music.stop, **btn_style).grid(row=0, column=2, padx=2)
        ctk.CTkButton(ctrl_row, text="⏭", command=self._music_next, **btn_style).grid(row=0, column=3, padx=2)
        self.btn_mute = ctk.CTkButton(ctrl_row, text="🔊", command=self._music_mute, **btn_style)
        self.btn_mute.grid(row=0, column=4, padx=(6, 4))

        self.vol_slider = ctk.CTkSlider(
            ctrl_row,
            from_=0.0,
            to=1.0,
            number_of_steps=50,
            command=self._on_volume_slider,
            fg_color=COLORS["bg_card"],
            progress_color=COLORS["accent"],
            button_color=COLORS["accent"],
        )
        self.vol_slider.set(self.music.volume)
        self.vol_slider.grid(row=0, column=5, sticky="ew", padx=(2, 4))

        self.lbl_vol = ctk.CTkLabel(
            ctrl_row, text="72%", width=38, font=_mono(11, bold=True), text_color=COLORS["accent"]
        )
        self.lbl_vol.grid(row=0, column=6)

        # --------------------------------------------------------------
        # 2. Voice-to-Speech (STT + TTS) Panel
        # --------------------------------------------------------------
        voice_card = ctk.CTkFrame(
            self.right_panel,
            fg_color=COLORS["bg_panel"],
            border_width=1,
            border_color=COLORS["border"],
            corner_radius=6,
        )
        voice_card.grid(row=1, column=0, sticky="nsew")
        voice_card.grid_columnconfigure(0, weight=1)
        voice_card.grid_rowconfigure(4, weight=1)
        self._glitch_panels.append(voice_card)

        v_hdr = ctk.CTkFrame(voice_card, fg_color="transparent")
        v_hdr.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        ctk.CTkLabel(
            v_hdr,
            text="[SANTO_AI // JARVIS_VOICE_SYS]",
            font=_mono(11, bold=True),
            text_color=COLORS["accent"],
        ).pack(side="left")

        ctk.CTkButton(
            v_hdr,
            text="+ Keyword",
            width=74,
            height=24,
            font=_mono(10, bold=True),
            fg_color="#0D3B1E",
            border_width=1,
            border_color=COLORS["accent"],
            hover_color="#14532D",
            text_color=COLORS["accent"],
            command=self._open_keyword_manager,
        ).pack(side="right", padx=(6, 0))

        self.lang_combo = ctk.CTkComboBox(
            v_hdr,
            values=["en-US (Santo / Jarvis)", "id-ID (Indonesia)"],
            width=154,
            height=24,
            font=_mono(10),
            fg_color=COLORS["bg_card"],
            border_color=COLORS["border"],
            button_color=COLORS["bg_elevated"],
            text_color=COLORS["text_primary"],
            command=self._on_lang_change,
        )
        self.lang_combo.set("en-US (Santo / Jarvis)")
        self.lang_combo.pack(side="right")

        # Mic & TTS engine selector row
        dev_row = ctk.CTkFrame(voice_card, fg_color="transparent")
        dev_row.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 6))
        dev_row.grid_columnconfigure(0, weight=1)
        dev_row.grid_columnconfigure(1, weight=1)

        self._mic_devices = list_microphones()
        mic_labels = [name[:26] if idx < 0 else f"{idx}: {name[:24]}" for idx, name in self._mic_devices]
        self._mic_map: Dict[str, int] = {lbl: idx for lbl, (idx, _) in zip(mic_labels, self._mic_devices)}
        self.mic_combo = ctk.CTkComboBox(
            dev_row,
            values=mic_labels,
            height=26,
            font=_mono(10),
            fg_color=COLORS["bg_card"],
            border_color=COLORS["border"],
            button_color=COLORS["bg_elevated"],
            text_color=COLORS["text_primary"],
            command=self._on_mic_change,
        )
        self.mic_combo.set(mic_labels[0])
        self.mic_combo.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.tts_combo = ctk.CTkComboBox(
            dev_row,
            values=[
                "Santo Jarvis (UK Neural)",
                "Auto (gTTS -> Offline)",
                "Offline Cepat (SAPI5)",
            ],
            height=26,
            font=_mono(10),
            fg_color=COLORS["bg_card"],
            border_color=COLORS["border"],
            button_color=COLORS["bg_elevated"],
            text_color=COLORS["text_primary"],
            command=lambda val: self.voice.set_tts_mode(val),
        )
        self.tts_combo.set("Santo Jarvis (UK Neural)")
        self.tts_combo.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        # Voice control buttons + switches
        v_ctrl = ctk.CTkFrame(voice_card, fg_color="transparent")
        v_ctrl.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 4))

        self.btn_listen_once = ctk.CTkButton(
            v_ctrl,
            text="🎙️ REKAM SUARA",
            width=128,
            height=30,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            text_color="#001A07",
            font=_mono(11, bold=True),
            command=self.voice.trigger_single_listen,
        )
        self.btn_listen_once.pack(side="left", padx=(0, 8))

        self.sw_continuous = ctk.CTkSwitch(
            v_ctrl,
            text="Auto Listen",
            font=_mono(11),
            progress_color=COLORS["accent"],
            command=self._toggle_continuous_listen,
        )
        self.sw_continuous.select()
        self.sw_continuous.pack(side="left", padx=4)
        self.voice.set_continuous_listening(True)

        self.sw_echo = ctk.CTkSwitch(
            v_ctrl,
            text="Echo TTS",
            font=_mono(11),
            progress_color=COLORS["cyan"],
            command=self._toggle_echo_mode,
        )
        self.sw_echo.pack(side="right", padx=2)

        # Mic level bar + status label
        mic_row = ctk.CTkFrame(voice_card, fg_color="transparent")
        mic_row.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 6))
        mic_row.grid_columnconfigure(1, weight=1)

        self.lbl_voice_status = ctk.CTkLabel(
            mic_row,
            text="STATUS: SIAP",
            font=_mono(11),
            text_color=COLORS["text_secondary"],
            width=185,
            anchor="w",
        )
        self.lbl_voice_status.grid(row=0, column=0, sticky="w", padx=(0, 8))

        self.mic_bar = ctk.CTkProgressBar(
            mic_row,
            height=7,
            fg_color=COLORS["bg_card"],
            progress_color=COLORS["accent"],
        )
        self.mic_bar.set(0.0)
        self.mic_bar.grid(row=0, column=1, sticky="ew")

        # Transcript history log
        self.voice_log = ctk.CTkTextbox(
            voice_card,
            fg_color="#020503",
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["accent"],
            font=_mono(11),
            corner_radius=4,
            wrap="word",
        )
        self.voice_log.grid(row=4, column=0, sticky="nsew", padx=12, pady=(0, 6))
        self.voice_log.insert(
            "end",
            "[SANTO_AI] >> JARVIS-STYLE VOICE ASSISTANT ONLINE\n"
            "• Wake    : Ucapkan 'Santo' -> 'Yes, sir.'\n"
            "• Identity: Ucapkan 'Who am I' -> 'Boss Budi, is there anything I can help you with?'\n"
            "• Custom  : Klik '+ Keyword' atau edit assets/santo_keywords.json\n"
            "• Control : 'Play music', 'Pause', 'Next', 'Zoom in', 'Follow face'\n\n",
        )
        self.voice_log.configure(state="disabled")

        # Direct Text-to-Speech input box
        tts_row = ctk.CTkFrame(voice_card, fg_color="transparent")
        tts_row.grid(row=5, column=0, sticky="ew", padx=12, pady=(0, 10))
        tts_row.grid_columnconfigure(0, weight=1)

        self.tts_entry = ctk.CTkEntry(
            tts_row,
            placeholder_text="> ketik perintah/teks untuk diucapkan (TTS)...",
            height=30,
            font=_mono(11),
            fg_color="#020503",
            border_color=COLORS["border"],
            text_color=COLORS["accent"],
            placeholder_text_color="#3A7A4E",
        )
        self.tts_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.tts_entry.bind("<Return>", lambda _e: self._speak_manual_text())

        ctk.CTkButton(
            tts_row,
            text="🔊 SPEAK",
            width=92,
            height=30,
            fg_color="#0D3B1E",
            border_width=1,
            border_color=COLORS["accent"],
            hover_color="#14532D",
            text_color=COLORS["accent"],
            font=_mono(11, bold=True),
            command=self._speak_manual_text,
        ).grid(row=0, column=1)

    # ------------------------------------------------------------------
    # Camera & Vision Thread
    # ------------------------------------------------------------------
    def _init_camera_async(self) -> None:
        self.vision.open_camera(0)

    def _vision_loop(self) -> None:
        while self._running:
            vol = self.music.volume
            result = self.vision.process_next_frame(current_music_volume=vol)
            with self._frame_lock:
                self._latest_vision = result
            # Small sleep if camera failed so CPU stays low
            if not result.camera_ok:
                time.sleep(0.12)
            else:
                time.sleep(0.003)

    # ------------------------------------------------------------------
    # Periodic UI Refresh & Dispatch
    # ------------------------------------------------------------------
    def _ui_tick(self) -> None:
        if not self._running:
            return

        self.music.tick()

        # 1. Drain voice / async events
        while not self._ui_events.empty():
            try:
                kind, payload = self._ui_events.get_nowait()
                if kind == "transcript":
                    self._handle_voice_transcript(payload)
                elif kind == "voice_status":
                    self._handle_voice_status(payload)
            except queue.Empty:
                break

        # 2. Render latest camera frame & update hand/face telemetry
        vision_res: Optional[VisionFrameResult] = None
        with self._frame_lock:
            if self._latest_vision is not None:
                vision_res = self._latest_vision
                self._latest_vision = None

        if vision_res is not None:
            self._render_video_frame(vision_res.frame_rgb)
            self.lbl_fps.configure(
                text=f"FPS: {vision_res.fps:.1f}" if vision_res.camera_ok else vision_res.camera_message
            )
            self._process_hand_telemetry(vision_res)
            self._process_face_telemetry(vision_res)

        # 3. Update music player UI & spectrum
        self._refresh_music_ui()

        # 4. Animate hacker UI glitch effects
        self._update_ui_glitch_fx(time.monotonic())

        self.after(25, self._ui_tick)

    def _update_ui_glitch_fx(self, now: float) -> None:
        if not self._ui_glitch_enabled:
            if self._glitch_active_frame:
                self._glitch_active_frame = False
                self._reset_ui_glitch_visuals()
            return

        in_burst = now < self._ui_glitch_burst_until
        always_screen = self.vision.screen_glitch_mode == "Selalu Aktif"
        glasses_on = self._glasses_currently_on

        if now >= self._next_ambient_glitch_ts:
            # Schedule next subtle ambient glitch flicker
            gap = 0.08 if in_burst else (0.45 if (always_screen or glasses_on) else 2.4)
            self._next_ambient_glitch_ts = now + gap + random.uniform(0.0, 0.25 if in_burst else 0.7)
            do_glitch = in_burst or always_screen or glasses_on or random.random() < 0.35
            if do_glitch:
                self._glitch_active_frame = True
                prompts = (
                    "[R00T@V1S10N-0S]:~#",
                    "[SYS_GL1TCH::0x4F]#",
                    "[ROOT@VISION-OS]:!#",
                    "[0xDEAD_EYE::OVERRIDE]",
                )
                border_col = random.choice((COLORS["rose"], COLORS["cyan"], COLORS["accent"]))
                self.lbl_header_prompt.configure(
                    text=random.choice(prompts),
                    text_color=border_col,
                )
                for p in self._glitch_panels:
                    p.configure(border_color=border_col)
                if in_burst:
                    self.lbl_fps.configure(text_color=COLORS["rose"])
                return

        if self._glitch_active_frame and now > (self._next_ambient_glitch_ts - 0.14):
            self._glitch_active_frame = False
            self._reset_ui_glitch_visuals()

    def _reset_ui_glitch_visuals(self) -> None:
        self.lbl_header_prompt.configure(text="[ROOT@VISION-OS]:~#", text_color=COLORS["accent"])
        self.lbl_fps.configure(text_color=COLORS["accent"])
        for p in self._glitch_panels:
            p.configure(border_color=COLORS["border"])

    def _render_video_frame(self, frame_rgb) -> None:
        cw = max(320, self.video_canvas.winfo_width())
        ch = max(200, self.video_canvas.winfo_height())
        fh, fw = frame_rgb.shape[:2]

        scale = min(cw / fw, ch / fh)
        nw = max(1, int(fw * scale))
        nh = max(1, int(fh * scale))

        pil_img = Image.fromarray(frame_rgb).resize((nw, nh), Image.Resampling.BILINEAR)
        photo = ImageTk.PhotoImage(image=pil_img)
        self._photo_ref = photo

        cx, cy = cw // 2, ch // 2
        if self._canvas_img_id is None:
            self._canvas_img_id = self.video_canvas.create_image(cx, cy, image=photo, anchor="center")
        else:
            self.video_canvas.coords(self._canvas_img_id, cx, cy)
            self.video_canvas.itemconfigure(self._canvas_img_id, image=photo)

    def _process_hand_telemetry(self, res: VisionFrameResult) -> None:
        h = res.hand
        self.hand_hold_bar.set(h.hold_progress)
        if not h.detected:
            self.lbl_hand_gesture.configure(text="Gesture: Tidak Ada Tangan")
            return

        self.lbl_hand_gesture.configure(text=f"Gesture: {h.gesture_label}")

        # Continuous Pinch volume control
        if self._gesture_music_enabled and h.pinch_active and h.pinch_volume is not None:
            self.music.set_volume(h.pinch_volume)

        # Discrete triggered gesture
        action_desc = self._last_triggered_action_text()
        if h.triggered_gesture and self._gesture_music_enabled:
            action_desc = self._execute_gesture_music_action(h.triggered_gesture)

        self.lbl_hand_meta.configure(
            text=f"Jari Terbuka: {h.finger_count}  |  Tangan: {h.handedness}  |  Aksi: {action_desc}"
        )

    def _last_triggered_action_text(self) -> str:
        return getattr(self, "_last_action_str", "-")

    def _execute_gesture_music_action(self, gesture_code: str) -> str:
        mapping = GESTURE_MUSIC_ACTIONS.get(gesture_code)
        if not mapping:
            return "-"
        action, label = mapping
        self._last_action_str = label

        if action == "PLAY_PAUSE":
            was_playing = self.music.is_playing and not self.music.is_paused
            self.music.toggle_play_pause()
            self.voice.speak("Music paused, sir." if was_playing else "Playing music, sir.")
        elif action == "STOP_PAUSE":
            self.music.pause()
            self.voice.speak("Music paused, sir.")
        elif action == "NEXT_TRACK":
            tr = self.music.next_track()
            if tr:
                self.voice.speak("Next track, sir.")
        elif action == "PREV_TRACK":
            tr = self.music.prev_track()
            if tr:
                self.voice.speak("Previous track, sir.")
        elif action == "VOLUME_UP":
            v = self.music.adjust_volume(0.10)
            self.voice.speak(f"Volume {int(v * 100)} percent, sir.")
        elif action == "VOLUME_DOWN":
            v = self.music.adjust_volume(-0.10)
            self.voice.speak(f"Volume {int(v * 100)} percent, sir.")
        elif action == "MUTE_TOGGLE":
            muted = self.music.toggle_mute()
            self.voice.speak("Audio muted, sir." if muted else "Audio restored, sir.")
        elif action == "ANNOUNCE_TRACK":
            st = self.music.get_state()
            short_name = st.current_title[:24]
            self.voice.speak(f"Playing {short_name}, sir.", force=True)

        return label

    def _process_face_telemetry(self, res: VisionFrameResult) -> None:
        f = res.face
        if not f.detected:
            self._glasses_currently_on = False
            self.lbl_face_expr.configure(text="Ekspresi: Wajah Tidak Terlihat")
            self.lbl_face_pose.configure(text=f"Posisi Kepala: Standby | Kedipan: {f.blink_count}")
            self.lbl_glasses_status.configure(text="KACAMATA: STANDBY", text_color=COLORS["text_secondary"])
            return

        self.lbl_face_expr.configure(text=f"Ekspresi: {f.expression}")
        pose_extra = ""
        if f.head_gesture == "NOD":
            pose_extra = "  [ANGGUKAN]"
        elif f.head_gesture == "SHAKE":
            pose_extra = "  [GELENGAN]"

        self.lbl_face_pose.configure(
            text=f"Posisi Kepala: {f.head_pose_label} | Kedipan: {f.blink_count}{pose_extra}"
        )
        self.lbl_face_angles.configure(
            text=(
                f"Yaw: {f.yaw:+.1f}° | Pitch: {f.pitch:+.1f}° | Roll: {f.roll:+.1f}° | "
                f"Senyum: {int(f.smile_score * 100)}% | Mulut: {int(f.mouth_open_score * 100)}%"
            )
        )

        # Glasses telemetry & trigger marked song
        self._glasses_currently_on = f.glasses_detected
        pct = int(f.glasses_score * 100)
        if f.glasses_detected:
            self.lbl_glasses_status.configure(
                text=f"KACAMATA: TERDETEKSI ({pct}%)",
                text_color=COLORS["amber"],
            )
        else:
            self.lbl_glasses_status.configure(
                text=f"KACAMATA: TIDAK ({pct}%)",
                text_color=COLORS["text_secondary"],
            )

        if f.glasses_just_worn:
            self._ui_glitch_burst_until = time.monotonic() + 1.8
            if self._glasses_auto_play:
                tr = self.music.play_marked_track()
                if tr:
                    self._append_voice_log(f"[GLASSES_ON] >> Memutar lagu bertanda: {tr.title}")
                    self.voice.speak("Glasses detected. Playing marked track, sir.")
                else:
                    self._append_voice_log("[GLASSES_ON] >> Kacamata terdeteksi, tetapi playlist masih kosong.")
                    self.voice.speak("Glasses detected, sir. Please add your music first.")
        elif f.glasses_just_removed and self._glasses_pause_on_remove:
            self.music.pause()
            self._append_voice_log("[GLASSES_OFF] >> Kacamata dilepas, musik dijeda.")

        # Optional voice reaction to facial expressions / head gestures
        if self._face_voice_reaction:
            now = time.monotonic()
            if now - self._last_face_react_ts > 3.5:
                speak_msg = ""
                key = ""
                if f.head_gesture == "NOD":
                    key = "NOD"
                    speak_msg = "Anda mengangguk."
                elif f.head_gesture == "SHAKE":
                    key = "SHAKE"
                    speak_msg = "Anda menggeleng."
                elif f.expression in ("Senyum Lebar", "Mulut Terbuka / Terkejut", "Kedip Mata Kiri", "Kedip Mata Kanan"):
                    key = f.expression
                    speak_msg = f"Ekspresi {f.expression.lower()}."
                if speak_msg and key != self._last_face_react_key:
                    self._last_face_react_ts = now
                    self._last_face_react_key = key
                    self.voice.speak(speak_msg)

    def _refresh_music_ui(self) -> None:
        st = self.music.get_state()
        status_str = "IDLE"
        if st.is_playing:
            status_str = "PAUSED" if st.is_paused else "PLAYING"

        self.lbl_track_title.configure(text=st.current_title)
        self.lbl_track_time.configure(
            text=f"[{status_str}]  •  {_fmt_time(st.position_sec)} / {_fmt_time(st.duration_sec)}"
        )

        if st.marked_index is not None and 0 <= st.marked_index < len(st.tracks):
            marked_name = st.tracks[st.marked_index].title
            self.lbl_marked_track.configure(
                text=f"★ TARGET KACAMATA: [{st.marked_index + 1:02d}] {marked_name}",
                text_color=COLORS["amber"],
            )
        elif st.tracks:
            self.lbl_marked_track.configure(
                text=f"☆ TARGET KACAMATA: Default [{st.tracks[0].title}] (klik ☆ untuk ubah)",
                text_color=COLORS["text_secondary"],
            )
        else:
            self.lbl_marked_track.configure(
                text="★ TARGET KACAMATA: Tambahkan lagu lalu klik ☆ untuk menandai",
                text_color=COLORS["text_secondary"],
            )

        if st.is_playing and not st.is_paused:
            self.btn_play_pause.configure(
                text="⏸ PAUSE", fg_color=COLORS["amber"], hover_color="#EAB308", text_color="#000000"
            )
        else:
            self.btn_play_pause.configure(
                text="▶ PLAY", fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"], text_color="#001A07"
            )

        self.btn_mute.configure(text="🔇" if st.is_muted else "🔊")
        self._updating_vol_slider = True
        self.vol_slider.set(st.volume)
        self._updating_vol_slider = False
        self.lbl_vol.configure(text=f"{int(st.volume * 100)}%")

        # Rebuild playlist rows only when track count changes, highlight active/marked track
        if len(st.tracks) != self._last_playlist_count:
            self._rebuild_playlist(st.tracks, st.current_index, st.marked_index)
        elif st.current_index != self._last_active_idx or st.marked_index != self._last_marked_idx:
            self._highlight_active_track(st.current_index, st.marked_index)

        # Draw 16-band spectrum bars
        self._draw_spectrum(st.spectrum, st.is_playing and not st.is_paused)

    def _rebuild_playlist(
        self,
        tracks: List[TrackInfo],
        active_idx: int,
        marked_idx: Optional[int],
    ) -> None:
        for row in self._playlist_rows:
            row.destroy()
        self._playlist_rows.clear()
        self._playlist_buttons.clear()
        self._star_buttons.clear()

        if not tracks:
            empty_lbl = ctk.CTkLabel(
                self.playlist_frame,
                text="[EMPTY_PLAYLIST] >> Klik '+ Lagu' atau '📁 Folder' untuk menambahkan lagu Anda, lalu klik ☆ untuk menandai lagu kacamata.",
                font=_mono(11),
                text_color=COLORS["text_secondary"],
                wraplength=360,
                justify="left",
            )
            empty_lbl.pack(anchor="w", padx=8, pady=12)
            self._playlist_rows.append(empty_lbl)
        else:
            for i, tr in enumerate(tracks):
                dur = _fmt_time(tr.duration_sec) if tr.duration_sec > 0 else "--:--"
                row = ctk.CTkFrame(self.playlist_frame, fg_color="transparent")
                row.pack(fill="x", padx=2, pady=1)
                row.grid_columnconfigure(1, weight=1)

                is_marked = i == marked_idx
                star_btn = ctk.CTkButton(
                    row,
                    text="★" if is_marked else "☆",
                    width=26,
                    height=24,
                    font=_mono(12, bold=True),
                    fg_color="#2D1E03" if is_marked else "transparent",
                    hover_color="#422006",
                    text_color=COLORS["amber"] if is_marked else COLORS["text_secondary"],
                    command=lambda idx=i: self._toggle_mark_track(idx),
                )
                star_btn.grid(row=0, column=0, padx=(0, 3))

                btn = ctk.CTkButton(
                    row,
                    text=f"[{i + 1:02d}] {tr.title} ({dur})",
                    anchor="w",
                    height=26,
                    font=_mono(11, bold=(i == active_idx)),
                    fg_color=COLORS["bg_elevated"] if i == active_idx else "transparent",
                    hover_color=COLORS["bg_elevated"],
                    text_color=COLORS["accent"] if i == active_idx else COLORS["text_primary"],
                    command=lambda idx=i: self.music.play(idx),
                )
                btn.grid(row=0, column=1, sticky="ew", padx=(0, 4))

                del_btn = ctk.CTkButton(
                    row,
                    text="✕",
                    width=24,
                    height=24,
                    font=_mono(10, bold=True),
                    fg_color="transparent",
                    hover_color="#3B0D16",
                    text_color=COLORS["rose"],
                    command=lambda idx=i: self._remove_music_track(idx),
                )
                del_btn.grid(row=0, column=2)

                self._playlist_rows.append(row)
                self._star_buttons.append(star_btn)
                self._playlist_buttons.append(btn)

        self._last_playlist_count = len(tracks)
        self._last_active_idx = active_idx
        self._last_marked_idx = marked_idx

    def _highlight_active_track(self, active_idx: int, marked_idx: Optional[int]) -> None:
        for i, btn in enumerate(self._playlist_buttons):
            is_act = i == active_idx
            btn.configure(
                font=_mono(11, bold=is_act),
                fg_color=COLORS["bg_elevated"] if is_act else "transparent",
                text_color=COLORS["accent"] if is_act else COLORS["text_primary"],
            )
        for i, sbtn in enumerate(self._star_buttons):
            is_mk = i == marked_idx
            sbtn.configure(
                text="★" if is_mk else "☆",
                fg_color="#2D1E03" if is_mk else "transparent",
                text_color=COLORS["amber"] if is_mk else COLORS["text_secondary"],
            )
        self._last_active_idx = active_idx
        self._last_marked_idx = marked_idx

    def _draw_spectrum(self, spectrum: List[float], active: bool) -> None:
        c = self.spec_canvas
        w = max(120, c.winfo_width())
        h = max(36, c.winfo_height())
        c.delete("all")
        n = len(spectrum)
        gap = 4
        bar_w = max(4, (w - (n + 1) * gap) / n)

        for i, val in enumerate(spectrum):
            x1 = gap + i * (bar_w + gap)
            x2 = x1 + bar_w
            bar_h = max(4, val * (h - 8))
            y1 = h - 4 - bar_h
            y2 = h - 4
            if not active:
                color = "#0D3B1E"
            elif val > 0.78:
                color = "#FF003C"
            elif val > 0.55:
                color = "#00F0FF"
            else:
                color = "#00FF41"
            c.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    # ------------------------------------------------------------------
    # Voice Handlers
    # ------------------------------------------------------------------
    def _handle_voice_status(self, st: VoiceStatus) -> None:
        self.lbl_voice_status.configure(text=f"STATUS: {st.status_text.upper()}")
        self.mic_bar.set(st.mic_level)
        if st.listening:
            self.btn_listen_once.configure(text="⏹ STOP REKAM", fg_color=COLORS["rose"], text_color="#FFFFFF")
        else:
            self.btn_listen_once.configure(text="🎙️ REKAM SUARA", fg_color=COLORS["accent"], text_color="#001A07")

    def _handle_voice_transcript(self, ev: VoiceTranscriptEvent) -> None:
        reply = ""
        act = ev.command_action
        arg = ev.command_arg

        if act == "SANTO_WAKE":
            reply = arg or "Yes, sir."
            self.voice.speak(reply, force=True)
        elif act == "CUSTOM_REPLY" and arg:
            reply = arg
            self.voice.speak(reply, force=True)
        elif act == "SPEAK_CUSTOM" and arg:
            reply = arg
            self.voice.speak(arg, force=True)
        elif act == "PLAY_MUSIC":
            tr = self.music.play()
            reply = f"Playing {tr.title}" if tr else "Playlist is empty, sir."
            self.voice.speak("Yes, sir. Playing music." if tr else "Playlist is empty, sir.")
        elif act == "PAUSE_MUSIC":
            self.music.pause()
            reply = "Music paused, sir."
            self.voice.speak(reply)
        elif act == "STOP_MUSIC":
            self.music.stop()
            reply = "Music stopped, sir."
            self.voice.speak(reply)
        elif act == "NEXT_TRACK":
            tr = self.music.next_track()
            reply = f"Next track: {tr.title}" if tr else "Playlist is empty, sir."
            self.voice.speak("Next track, sir." if tr else "Playlist is empty, sir.")
        elif act == "PREV_TRACK":
            tr = self.music.prev_track()
            reply = f"Previous track: {tr.title}" if tr else "Playlist is empty, sir."
            self.voice.speak("Previous track, sir." if tr else "Playlist is empty, sir.")
        elif act == "VOLUME_UP":
            v = self.music.adjust_volume(0.15)
            reply = f"Volume increased to {int(v * 100)} percent, sir."
            self.voice.speak(reply)
        elif act == "VOLUME_DOWN":
            v = self.music.adjust_volume(-0.15)
            reply = f"Volume decreased to {int(v * 100)} percent, sir."
            self.voice.speak(reply)
        elif act == "MUTE":
            self.music.is_muted = False
            self.music.toggle_mute()
            reply = "Audio muted, sir."
            self.voice.speak(reply)
        elif act == "UNMUTE":
            self.music.is_muted = True
            self.music.toggle_mute()
            reply = "Audio restored, sir."
            self.voice.speak(reply)
        elif act == "ANNOUNCE_TRACK":
            st = self.music.get_state()
            reply = f"Now playing {st.current_title}."
            self.voice.speak(f"Playing {st.current_title[:24]}, sir.", force=True)
        elif act == "PTZ_MODE" and arg:
            self.ptz.set_mode(arg)
            self.ptz_mode_combo.set(arg)
            reply = f"Camera tracking set to {arg}, sir."
            self.voice.speak(reply)
        elif act == "PTZ_RESET":
            self.ptz.reset()
            reply = "Camera centered, sir."
            self.voice.speak(reply)
        elif act == "PTZ_ZOOM_IN":
            self.ptz.adjust_zoom(0.35)
            reply = "Zooming in, sir."
            self.voice.speak(reply)
        elif act == "PTZ_ZOOM_OUT":
            self.ptz.adjust_zoom(-0.35)
            reply = "Zooming out, sir."
            self.voice.speak(reply)
        elif act == "PTZ_PAN_LEFT":
            self._manual_nudge(-0.12, 0.0)
            reply = "Panning camera left, sir."
            self.voice.speak(reply)
        elif act == "PTZ_PAN_RIGHT":
            self._manual_nudge(0.12, 0.0)
            reply = "Panning camera right, sir."
            self.voice.speak(reply)
        elif act == "PTZ_PAN_UP":
            self._manual_nudge(0.0, -0.12)
            reply = "Tilting camera up, sir."
            self.voice.speak(reply)
        elif act == "PTZ_PAN_DOWN":
            self._manual_nudge(0.0, 0.12)
            reply = "Tilting camera down, sir."
            self.voice.speak(reply)
        elif act == "OVERLAY_MODE" and arg:
            self.seg_overlay.set(arg)
            self.vision.set_overlay_mode(arg)
            reply = f"Overlay mode set to {arg}, sir."
            self.voice.speak(reply)
        elif act == "EYE_FILTER_MODE" and arg:
            self.seg_eye_filter.set(arg)
            self.vision.set_eye_filter_mode(arg)
            reply = f"Eye filter set to {arg}, sir."
            self.voice.speak(reply)
        elif self.voice.echo_mode:
            reply = f"[Echo] {ev.text}"
            self.voice.speak(ev.text, force=True)

        self._append_voice_log(f"[BOSS_BUDI] >> {ev.text}")
        if reply:
            self._append_voice_log(f"[SANTO_AI]  << {reply}")

    def _append_voice_log(self, line: str) -> None:
        self.voice_log.configure(state="normal")
        self.voice_log.insert("end", line + "\n")
        self.voice_log.see("end")
        self.voice_log.configure(state="disabled")

    # ------------------------------------------------------------------
    # Widget Callbacks
    # ------------------------------------------------------------------
    def _on_track_changed(self, track: TrackInfo) -> None:
        pass

    def _toggle_gesture_music(self) -> None:
        self._gesture_music_enabled = bool(self.sw_gesture_music.get())

    def _toggle_voice_feedback(self) -> None:
        self.voice.voice_feedback_enabled = bool(self.sw_voice_feedback.get())

    def _toggle_face_reaction(self) -> None:
        self._face_voice_reaction = bool(self.sw_face_voice.get())

    def _on_camera_select(self, choice: str) -> None:
        try:
            idx = int(choice.split()[-1])
        except Exception:
            idx = 0
        threading.Thread(target=self.vision.open_camera, args=(idx,), daemon=True).start()

    def _restart_camera(self) -> None:
        self._on_camera_select(self.cam_combo.get())

    def _toggle_mirror(self) -> None:
        self.vision.mirror_view = bool(self.sw_mirror.get())

    def _on_overlay_mode_change(self, mode: str) -> None:
        self.vision.set_overlay_mode(mode)

    def _on_eye_filter_change(self, mode: str) -> None:
        self.vision.set_eye_filter_mode(mode)

    def _on_screen_glitch_change(self, mode: str) -> None:
        self.vision.set_screen_glitch_mode(mode)

    def _toggle_ui_glitch(self) -> None:
        self._ui_glitch_enabled = bool(self.sw_ui_glitch.get())
        if not self._ui_glitch_enabled:
            self._reset_ui_glitch_visuals()

    def _toggle_tether_box(self) -> None:
        self.vision.set_tether_glitch_enabled(bool(self.sw_tether_box.get()))

    def _toggle_camera_only_mode(self) -> None:
        self._camera_only_mode = not self._camera_only_mode
        if self._camera_only_mode:
            self.header.grid_remove()
            self.right_panel.grid_remove()
            self.ptz_bar.grid_remove()
            self.telem_row.grid_remove()
            self.grid_columnconfigure(1, weight=0)
            self.left_panel.grid_rowconfigure(2, weight=0)
            self.left_panel.grid_configure(columnspan=2, padx=8, pady=8)
            self.cam_card.grid_configure(pady=0)
            self.btn_exit_cam_only.place(relx=0.985, rely=0.025, anchor="ne")
            self.btn_exit_cam_only.lift()
        else:
            self.btn_exit_cam_only.place_forget()
            self.header.grid()
            self.grid_columnconfigure(1, weight=8)
            self.left_panel.grid_rowconfigure(2, weight=2)
            self.left_panel.grid_configure(columnspan=1, padx=(10, 6), pady=(0, 10))
            self.cam_card.grid_configure(pady=(0, 8))
            self.ptz_bar.grid()
            self.telem_row.grid()
            self.right_panel.grid()

    def _toggle_glasses_autoplay(self) -> None:
        self._glasses_auto_play = bool(self.sw_glasses_play.get())

    def _toggle_glasses_pause(self) -> None:
        self._glasses_pause_on_remove = bool(self.sw_glasses_pause.get())

    def _on_glasses_sens_change(self, value: float) -> None:
        self.vision.set_glasses_sensitivity(float(value))

    def _calibrate_no_glasses(self) -> None:
        base = self.vision.calibrate_no_glasses()
        self._append_voice_log(f"[CALIB] >> Baseline wajah tanpa kacamata disimpan ({int(base * 100)}%).")

    def _toggle_mark_track(self, index: int) -> None:
        marked = self.music.set_marked_track(index)
        self._last_marked_idx = -999
        if marked:
            self._append_voice_log(f"[STAR] >> Lagu ditandai untuk kacamata: {marked.title}")
        else:
            self._append_voice_log("[STAR] >> Tanda lagu kacamata dihapus.")

    def _on_ptz_mode_change(self, mode: str) -> None:
        self.ptz.set_mode(mode)

    def _toggle_auto_zoom(self) -> None:
        self.ptz.set_auto_zoom(bool(self.sw_auto_zoom.get()))

    def _manual_nudge(self, dx: float, dy: float) -> None:
        self.ptz.nudge(dx, dy)
        self.ptz_mode_combo.set("Manual PTZ")

    def _reset_ptz(self) -> None:
        self.ptz.reset()

    def _music_play_pause(self) -> None:
        self.music.toggle_play_pause()

    def _music_prev(self) -> None:
        self.music.prev_track()

    def _music_next(self) -> None:
        self.music.next_track()

    def _music_mute(self) -> None:
        self.music.toggle_mute()

    def _on_volume_slider(self, value: float) -> None:
        if not self._updating_vol_slider:
            self.music.set_volume(float(value))

    def _add_music_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Pilih File Musik (MP3 / WAV / OGG / FLAC)",
            filetypes=[("Audio Files", "*.mp3 *.wav *.ogg *.flac"), ("All Files", "*.*")],
        )
        if paths:
            added = self.music.add_files(list(paths))
            if added:
                self._last_playlist_count = -1
                self._append_voice_log(f"[AUDIO] >> Ditambahkan {added} lagu ke playlist.")

    def _add_music_folder(self) -> None:
        folder = filedialog.askdirectory(title="Pilih Folder Musik")
        if folder:
            added = self.music.load_folder(folder)
            if added:
                self._last_playlist_count = -1
                self._append_voice_log(f"[AUDIO] >> Dimuat {added} lagu dari folder.")

    def _remove_music_track(self, index: int) -> None:
        removed = self.music.remove_track(index)
        if removed:
            self._last_playlist_count = -1
            self._append_voice_log(f"[AUDIO] >> Dihapus dari playlist: {removed}")

    def _clear_music_playlist(self) -> None:
        self.music.clear_playlist()
        self._last_playlist_count = -1
        self._append_voice_log("[AUDIO] >> Playlist dikosongkan.")

    def _on_lang_change(self, choice: str) -> None:
        code = "id-ID" if choice.startswith("id-ID") else "en-US"
        self.voice.set_language(code)

    def _on_mic_change(self, choice: str) -> None:
        idx = self._mic_map.get(choice, -1)
        self.voice.set_mic_device(idx)

    def _toggle_continuous_listen(self) -> None:
        self.voice.set_continuous_listening(bool(self.sw_continuous.get()))

    def _toggle_echo_mode(self) -> None:
        self.voice.echo_mode = bool(self.sw_echo.get())

    def _speak_manual_text(self) -> None:
        txt = self.tts_entry.get().strip()
        if not txt:
            return
        self.tts_entry.delete(0, "end")
        self._append_voice_log(f"[SANTO_AI]  << {txt}")
        self.voice.speak(txt, force=True)

    def _open_keyword_manager(self) -> None:
        win = ctk.CTkToplevel(self)
        win.title("[SANTO_KEYWORDS] // Kelola Kata Kunci Suara")
        win.geometry("540x430")
        win.configure(fg_color=COLORS["bg_root"])
        win.transient(self)
        win.grab_set()

        ctk.CTkLabel(
            win,
            text="[SANTO_KEYWORD_REGISTRY] :: assets/santo_keywords.json",
            font=_mono(12, bold=True),
            text_color=COLORS["accent"],
        ).pack(anchor="w", padx=14, pady=(12, 6))

        list_frame = ctk.CTkScrollableFrame(
            win,
            fg_color=COLORS["bg_panel"],
            border_width=1,
            border_color=COLORS["border"],
            corner_radius=4,
            height=210,
        )
        list_frame.pack(fill="both", expand=True, padx=14, pady=(0, 10))

        def refresh_rows():
            for child in list_frame.winfo_children():
                child.destroy()
            for r in self.voice.keywords.list_rules():
                row = ctk.CTkFrame(list_frame, fg_color=COLORS["bg_card"], corner_radius=4)
                row.pack(fill="x", padx=4, pady=3)
                row.grid_columnconfigure(0, weight=1)
                pats = ", ".join(r.get("patterns", []))
                resp = r.get("response", "")
                ctk.CTkLabel(
                    row,
                    text=f"[{pats}]  ->  \"{resp}\"",
                    font=_mono(10),
                    text_color=COLORS["text_primary"],
                    wraplength=415,
                    justify="left",
                    anchor="w",
                ).grid(row=0, column=0, sticky="w", padx=8, pady=6)
                rid = str(r.get("id", ""))
                if rid not in ("wake_santo", "who_am_i"):
                    ctk.CTkButton(
                        row,
                        text="✕",
                        width=26,
                        height=24,
                        font=_mono(10, bold=True),
                        fg_color="transparent",
                        hover_color="#3B0D16",
                        text_color=COLORS["rose"],
                        command=lambda rule_id=rid: (self.voice.keywords.remove_rule(rule_id), refresh_rows()),
                    ).grid(row=0, column=1, padx=6)

        refresh_rows()

        form = ctk.CTkFrame(win, fg_color=COLORS["bg_panel"], border_width=1, border_color=COLORS["border"])
        form.pack(fill="x", padx=14, pady=(0, 12))
        form.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(form, text="Kata Kunci:", font=_mono(11), text_color=COLORS["text_secondary"]).grid(
            row=0, column=0, padx=8, pady=(8, 4), sticky="w"
        )
        ent_pat = ctk.CTkEntry(
            form,
            placeholder_text="misal: system status, check status",
            font=_mono(11),
            fg_color="#020503",
            border_color=COLORS["border"],
            text_color=COLORS["accent"],
        )
        ent_pat.grid(row=0, column=1, padx=(0, 8), pady=(8, 4), sticky="ew")

        ctk.CTkLabel(form, text="Jawaban:", font=_mono(11), text_color=COLORS["text_secondary"]).grid(
            row=1, column=0, padx=8, pady=(4, 8), sticky="w"
        )
        ent_resp = ctk.CTkEntry(
            form,
            placeholder_text="misal: All systems online, Boss Budi.",
            font=_mono(11),
            fg_color="#020503",
            border_color=COLORS["border"],
            text_color=COLORS["accent"],
        )
        ent_resp.grid(row=1, column=1, padx=(0, 8), pady=(4, 8), sticky="ew")

        def save_rule():
            raw_pats = [p.strip() for p in ent_pat.get().split(",") if p.strip()]
            resp = ent_resp.get().strip()
            if not raw_pats or not resp:
                return
            self.voice.keywords.add_rule(raw_pats, resp)
            ent_pat.delete(0, "end")
            ent_resp.delete(0, "end")
            refresh_rows()
            self._append_voice_log(f"[SANTO_CFG] >> Keyword baru ditambahkan: {raw_pats} -> '{resp}'")

        ctk.CTkButton(
            form,
            text="+ Simpan Keyword",
            font=_mono(11, bold=True),
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            text_color="#001A07",
            command=save_rule,
        ).grid(row=2, column=0, columnspan=2, padx=8, pady=(0, 8), sticky="ew")

    def _bind_shortcuts(self) -> None:
        def guard(fn):
            def wrapper(event):
                if self.focus_get() == self.tts_entry._entry:
                    return None
                fn()
                return "break"
            return wrapper

        self.bind("<Left>", guard(lambda: self._manual_nudge(-0.08, 0.0)))
        self.bind("<Right>", guard(lambda: self._manual_nudge(0.08, 0.0)))
        self.bind("<Up>", guard(lambda: self._manual_nudge(0.0, -0.08)))
        self.bind("<Down>", guard(lambda: self._manual_nudge(0.0, 0.08)))
        self.bind("<plus>", guard(lambda: self.ptz.adjust_zoom(0.22)))
        self.bind("<equal>", guard(lambda: self.ptz.adjust_zoom(0.22)))
        self.bind("<minus>", guard(lambda: self.ptz.adjust_zoom(-0.22)))
        self.bind("<r>", guard(self._reset_ptz))
        self.bind("<f>", guard(self._toggle_camera_only_mode))
        self.bind("<F>", guard(self._toggle_camera_only_mode))
        self.bind("<Escape>", guard(lambda: self._toggle_camera_only_mode() if self._camera_only_mode else None))
        self.bind("<space>", guard(self._music_play_pause))

    def _on_close(self) -> None:
        self._running = False
        self.voice.shutdown()
        self.music.stop()
        if hasattr(self, "_vision_thread") and self._vision_thread.is_alive():
            self._vision_thread.join(timeout=1.5)
        self.vision.release()
        self.destroy()
