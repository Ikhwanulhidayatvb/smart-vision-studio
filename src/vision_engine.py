"""OpenCV + MediaPipe Vision Engine for Hand Gestures, Face Movements, and PTZ HUD."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import requests
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .config import (
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    FACE_MODEL_FILENAME,
    FACE_MODEL_URL,
    GESTURE_COOLDOWN_SECONDS,
    GESTURE_HOLD_SECONDS,
    GESTURE_LABELS_ID,
    HAND_MODEL_FILENAME,
    HAND_MODEL_URL,
    MODELS_DIR,
)
from .ptz_camera import PTZController, PTZSnapshot


@dataclass
class HandTelemetry:
    detected: bool = False
    hand_count: int = 0
    handedness: str = "-"
    gesture_code: str = "NONE"
    gesture_label: str = "Tidak Ada Gesture"
    finger_count: int = 0
    hold_progress: float = 0.0  # 0.0 .. 1.0
    pinch_active: bool = False
    pinch_volume: Optional[float] = None  # 0.0 .. 1.0 when pinch volume active
    triggered_gesture: Optional[str] = None
    tether_active: bool = False
    tether_tension: float = 0.0
    box_norm: Optional[Tuple[float, float, float, float]] = None


@dataclass
class FaceTelemetry:
    detected: bool = False
    expression: str = "Netral"
    head_pose_label: str = "Lurus ke Depan"
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    smile_score: float = 0.0
    mouth_open_score: float = 0.0
    brow_up_score: float = 0.0
    left_blink: bool = False
    right_blink: bool = False
    blink_count: int = 0
    head_gesture: Optional[str] = None  # "NOD" or "SHAKE"
    glasses_detected: bool = False
    glasses_score: float = 0.0
    glasses_just_worn: bool = False
    glasses_just_removed: bool = False
    box_norm: Optional[Tuple[float, float, float, float]] = None


@dataclass
class VisionFrameResult:
    frame_rgb: np.ndarray
    fps: float
    camera_ok: bool
    camera_message: str
    hand: HandTelemetry = field(default_factory=HandTelemetry)
    face: FaceTelemetry = field(default_factory=FaceTelemetry)
    ptz: Optional[PTZSnapshot] = None


def ensure_model_file(filename: str, url: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    if path.exists() and path.stat().st_size > 100_000:
        return path
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def _dist2d(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


class VisionEngine:
    """Real-time Hand + Face + PTZ Camera processor."""

    def __init__(self, ptz: PTZController) -> None:
        self.ptz = ptz
        self.camera_index: int = 0
        self.cap: Optional[cv2.VideoCapture] = None
        self._cap_lock = threading.Lock()
        self.mirror_view: bool = True
        self.overlay_mode: str = "Muka & Tangan"
        self.eye_filter_mode: str = "Off"
        self.screen_glitch_mode: str = "Auto (Kacamata)"
        self.tether_glitch_enabled: bool = True
        self.show_minimap: bool = True

        # Cyber Tether & Glitch Visual Box state
        self._tether_mode: Optional[str] = None  # "SINGLE" or "DUAL"
        self._tether_armed: bool = False
        self._tether_active: bool = False
        self._tether_pt_a: Tuple[float, float] = (0.5, 0.5)
        self._tether_pt_b: Tuple[float, float] = (0.5, 0.5)
        self._tether_prev_center: Optional[Tuple[float, float]] = None
        self._tether_vel: float = 0.0
        self._tether_tension: float = 0.0
        self._tether_last_seen_ts: float = 0.0

        self._glitch_rng = np.random.default_rng()
        self._glitch_tick_ts: float = 0.0
        self._glitch_seed: int = 1
        self._screen_glitch_tick_ts: float = 0.0
        self._screen_glitch_seed: int = 7
        self._screen_glitch_burst_until: float = 0.0

        # Glasses detection state
        self.glasses_threshold: float = 0.46
        self._glasses_baseline: Optional[float] = None
        self._glasses_ema: float = 0.0
        self._glasses_raw_ema: float = 0.0
        self._glasses_state: bool = False
        self._glasses_cand_state: bool = False
        self._glasses_cand_since: float = 0.0

        self._left_eye_indices = sorted(
            {c.start for c in mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE}
            | {c.end for c in mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE}
        )
        self._right_eye_indices = sorted(
            {c.start for c in mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE}
            | {c.end for c in mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE}
        )

        self._hand_landmarker: Optional[mp_vision.HandLandmarker] = None
        self._face_landmarker: Optional[mp_vision.FaceLandmarker] = None
        self._models_error: str = ""
        self._last_ts_ms: int = 0

        # FPS tracking
        self._fps_times: Deque[float] = deque(maxlen=25)

        # Gesture hold/debounce/swipe state
        self._pending_gesture: str = "NONE"
        self._pending_start_ts: float = 0.0
        self._last_trigger_ts: float = 0.0
        self._last_triggered_label: str = ""
        self._palm_x_history: Deque[Tuple[float, float, float]] = deque(maxlen=14)
        self._pinch_base_y: Optional[float] = None
        self._pinch_base_vol: float = 0.72

        # Face blink & head nod/shake history
        self._prev_both_blink: bool = False
        self._blink_count: int = 0
        self._pitch_history: Deque[Tuple[float, float]] = deque(maxlen=18)
        self._yaw_history: Deque[Tuple[float, float]] = deque(maxlen=18)
        self._last_head_gesture_ts: float = 0.0

        self._init_models()

    def _init_models(self) -> None:
        try:
            hand_path = ensure_model_file(HAND_MODEL_FILENAME, HAND_MODEL_URL)
            face_path = ensure_model_file(FACE_MODEL_FILENAME, FACE_MODEL_URL)

            hand_opts = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(hand_path)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.55,
                min_hand_presence_confidence=0.55,
                min_tracking_confidence=0.55,
            )
            face_opts = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(face_path)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.55,
                min_face_presence_confidence=0.55,
                min_tracking_confidence=0.55,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self._hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_opts)
            self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_opts)
            self._models_error = ""
        except Exception as exc:
            self._models_error = f"Model MediaPipe gagal dimuat: {exc}"

    def open_camera(self, index: int = 0) -> Tuple[bool, str]:
        with self._cap_lock:
            self._close_camera_locked()
            self.camera_index = index
            # Try DirectShow first on Windows for fast startup & 720p negotiation
            for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAMERA_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ok, _ = cap.read()
                    if ok:
                        self.cap = cap
                        return True, f"Kamera {index} aktif"
                    cap.release()
            return False, f"Kamera {index} tidak ditemukan / sedang digunakan aplikasi lain"

    def _close_camera_locked(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def close_camera(self) -> None:
        with self._cap_lock:
            self._close_camera_locked()

    def release(self) -> None:
        self.close_camera()
        if self._hand_landmarker is not None:
            try:
                self._hand_landmarker.close()
            except Exception:
                pass
        if self._face_landmarker is not None:
            try:
                self._face_landmarker.close()
            except Exception:
                pass

    def set_pinch_volume_anchor(self, current_volume: float) -> None:
        self._pinch_base_vol = current_volume

    def set_overlay_mode(self, mode: str) -> None:
        self.overlay_mode = mode

    def set_eye_filter_mode(self, mode: str) -> None:
        self.eye_filter_mode = mode

    def set_screen_glitch_mode(self, mode: str) -> None:
        self.screen_glitch_mode = mode

    def set_tether_glitch_enabled(self, enabled: bool) -> None:
        self.tether_glitch_enabled = enabled
        if not enabled:
            self._tether_armed = False
            self._tether_active = False
            self._tether_mode = None

    def set_glasses_sensitivity(self, sensitivity: float) -> None:
        """Map sensitivity [0.0..1.0] to threshold (higher sensitivity = easier trigger)."""
        sens = float(np.clip(sensitivity, 0.0, 1.0))
        self.glasses_threshold = 0.68 - sens * 0.42  # 0.5 -> ~0.47

    def calibrate_no_glasses(self) -> float:
        """Snapshot current raw face score as baseline without glasses."""
        self._glasses_baseline = self._glasses_raw_ema
        self._glasses_state = False
        return self._glasses_baseline

    def _next_timestamp_ms(self) -> int:
        now_ms = int(time.monotonic() * 1000)
        if now_ms <= self._last_ts_ms:
            now_ms = self._last_ts_ms + 1
        self._last_ts_ms = now_ms
        return now_ms

    @staticmethod
    def list_available_cameras(max_index: int = 4) -> List[int]:
        found: List[int] = []
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(idx)
                cap.release()
        return found or [0]

    def process_next_frame(self, current_music_volume: float = 0.72) -> VisionFrameResult:
        now = time.monotonic()
        self._fps_times.append(now)
        fps = 0.0
        if len(self._fps_times) >= 2:
            span = self._fps_times[-1] - self._fps_times[0]
            if span > 0:
                fps = (len(self._fps_times) - 1) / span

        with self._cap_lock:
            if self.cap is None or not self.cap.isOpened():
                blank = self._render_status_frame(
                    "Kamera belum aktif atau tidak terdeteksi.\n"
                    "Klik 'Restart Kamera' atau ganti Index Kamera."
                )
                return VisionFrameResult(
                    frame_rgb=blank,
                    fps=fps,
                    camera_ok=False,
                    camera_message="Kamera tidak aktif",
                )

            ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            blank = self._render_status_frame("Gagal membaca frame dari kamera...")
            return VisionFrameResult(
                frame_rgb=blank,
                fps=fps,
                camera_ok=False,
                camera_message="Gagal baca frame",
            )

        if self.mirror_view:
            frame_bgr = cv2.flip(frame_bgr, 1)

        if self._models_error:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return VisionFrameResult(
                frame_rgb=rgb,
                fps=fps,
                camera_ok=True,
                camera_message=self._models_error,
            )

        frame_rgb_raw = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb_raw)
        ts_ms = self._next_timestamp_ms()

        hand_result = self._hand_landmarker.detect_for_video(mp_image, ts_ms) if self._hand_landmarker else None
        face_result = self._face_landmarker.detect_for_video(mp_image, ts_ms) if self._face_landmarker else None

        self._update_tether_state(hand_result, now)
        hand_tel, primary_hand_lms = self._analyze_hands(hand_result, now, current_music_volume)
        hand_tel.tether_active = self._tether_active
        hand_tel.tether_tension = self._tether_tension
        face_tel = self._analyze_face(face_result, now)

        # Detect glasses on clean raw frame BEFORE PTZ / eye filter
        if face_result and face_result.face_landmarks:
            self._detect_glasses(frame_bgr, face_result.face_landmarks[0], face_tel, now)

        # Update Digital PTZ Camera tracking from full sensor coordinates
        self.ptz.update_tracking(face_tel.box_norm, hand_tel.box_norm)
        ptz_bgr, ptz_snap = self.ptz.apply(frame_bgr)

        # Apply Eye Blur / System Error Glitch Filter independently of wireframe overlay
        if self.eye_filter_mode != "Off" and face_result and face_result.face_landmarks:
            self._apply_eye_filter(
                ptz_bgr,
                face_result.face_landmarks[0],
                ptz_snap.crop_norm,
                now,
            )

        # Apply Hand-Driven Elastic Tether & Glitch Visual Box
        if self.tether_glitch_enabled and (self._tether_armed or self._tether_active):
            self._draw_tether_and_glitch_box(ptz_bgr, ptz_snap.crop_norm, now)

        # Draw selective overlays (Muka & Tangan / Muka Saja / Tangan Saja / Off)
        show_face = self.overlay_mode in ("Muka & Tangan", "Muka Saja")
        show_hand = self.overlay_mode in ("Muka & Tangan", "Tangan Saja")

        if show_face and face_result and face_result.face_landmarks:
            self._draw_face_overlay(
                ptz_bgr,
                face_result.face_landmarks[0],
                ptz_snap.crop_norm,
                face_tel,
                skip_eyes=(self.eye_filter_mode != "Off"),
            )
        if show_hand and hand_result and hand_result.hand_landmarks:
            for idx, h_lms in enumerate(hand_result.hand_landmarks):
                self._draw_hand_overlay(
                    ptz_bgr,
                    h_lms,
                    ptz_snap.crop_norm,
                    hand_tel if idx == 0 else None,
                )

        if self.show_minimap:
            self._draw_ptz_hud(ptz_bgr, ptz_snap, face_tel.box_norm, hand_tel.box_norm)

        # Apply full-frame cyber glitch display when active or triggered by glasses
        self._apply_screen_glitch(ptz_bgr, face_tel, now)

        out_rgb = cv2.cvtColor(ptz_bgr, cv2.COLOR_BGR2RGB)
        return VisionFrameResult(
            frame_rgb=out_rgb,
            fps=fps,
            camera_ok=True,
            camera_message="Kamera aktif",
            hand=hand_tel,
            face=face_tel,
            ptz=ptz_snap,
        )

    @staticmethod
    def _render_status_frame(message: str) -> np.ndarray:
        canvas = np.zeros((540, 960, 3), dtype=np.uint8)
        canvas[:, :] = (2, 7, 3)
        y = 240
        for line in message.split("\n"):
            cv2.putText(
                canvas,
                line,
                (70, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (65, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 42
        return canvas

    def _analyze_hands(
        self,
        result,
        now: float,
        current_music_volume: float,
    ) -> Tuple[HandTelemetry, Optional[list]]:
        tel = HandTelemetry()
        if not result or not result.hand_landmarks:
            self._pending_gesture = "NONE"
            self._pinch_base_y = None
            self._palm_x_history.clear()
            return tel, None

        tel.detected = True
        tel.hand_count = len(result.hand_landmarks)
        lms = result.hand_landmarks[0]
        if result.handedness and result.handedness[0]:
            tel.handedness = result.handedness[0][0].category_name

        xs = [p.x for p in lms]
        ys = [p.y for p in lms]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        tel.box_norm = (
            (min_x + max_x) * 0.5,
            (min_y + max_y) * 0.5,
            max(0.05, max_x - min_x),
            max(0.05, max_y - min_y),
        )

        gesture_code, finger_count, pinch_dist_ratio = self._classify_hand_gesture(lms)
        tel.finger_count = finger_count
        tel.gesture_code = gesture_code

        # Check horizontal swipe gesture using palm center (landmark 9)
        palm_x, palm_y = lms[9].x, lms[9].y
        if finger_count >= 4:
            self._palm_x_history.append((now, palm_x, palm_y))
        else:
            self._palm_x_history.clear()

        swipe_gesture: Optional[str] = None
        if len(self._palm_x_history) >= 6:
            old_t, old_x, old_y = self._palm_x_history[0]
            dt = now - old_t
            dx = palm_x - old_x
            dy = abs(palm_y - old_y)
            if 0.08 < dt < 0.42 and abs(dx) > 0.24 and dy < 0.14:
                swipe_gesture = "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"
                self._palm_x_history.clear()

        if swipe_gesture and (now - self._last_trigger_ts) > GESTURE_COOLDOWN_SECONDS:
            tel.gesture_code = swipe_gesture
            tel.gesture_label = GESTURE_LABELS_ID.get(swipe_gesture, swipe_gesture)
            tel.triggered_gesture = swipe_gesture
            tel.hold_progress = 1.0
            self._last_trigger_ts = now
            self._pending_gesture = "NONE"
            return tel, lms

        # Continuous Pinch volume control (OK / Thumb-Index pinch while other fingers free)
        if gesture_code == "PINCH":
            tel.pinch_active = True
            tel.gesture_label = GESTURE_LABELS_ID["PINCH"]
            if self._pinch_base_y is None:
                self._pinch_base_y = palm_y
                self._pinch_base_vol = current_music_volume
            # Moving hand up decreases y -> increases volume
            delta_vol = (self._pinch_base_y - palm_y) * 2.1
            tel.pinch_volume = float(np.clip(self._pinch_base_vol + delta_vol, 0.0, 1.0))
            self._pending_gesture = "NONE"
            return tel, lms
        else:
            self._pinch_base_y = None

        tel.gesture_label = GESTURE_LABELS_ID.get(gesture_code, gesture_code)

        # Hold-to-trigger discrete gestures (suppressed while stretching Cyber Tether Box)
        if gesture_code == "NONE" or self._tether_active:
            self._pending_gesture = "NONE"
            tel.hold_progress = 0.0
            if self._tether_active:
                tel.gesture_label = f"Cyber Tether ({int(self._tether_tension * 100)}%)"
        else:
            if gesture_code != self._pending_gesture:
                self._pending_gesture = gesture_code
                self._pending_start_ts = now

            elapsed = now - self._pending_start_ts
            in_cooldown = (now - self._last_trigger_ts) < GESTURE_COOLDOWN_SECONDS
            if in_cooldown:
                tel.hold_progress = 0.0
            else:
                tel.hold_progress = float(min(1.0, elapsed / GESTURE_HOLD_SECONDS))
                if elapsed >= GESTURE_HOLD_SECONDS:
                    tel.triggered_gesture = gesture_code
                    self._last_trigger_ts = now
                    self._pending_start_ts = now + 999.0  # require fresh hold or cooldown

        return tel, lms

    @staticmethod
    def _hand_pinch_geometry(lms) -> Tuple[Tuple[float, float], Tuple[float, float], float, bool, bool]:
        wrist = lms[0]
        palm_scale = max(0.04, _dist2d(wrist, lms[9]))
        thumb_pt = (lms[4].x, lms[4].y)
        index_pt = (lms[8].x, lms[8].y)
        pinch_ratio = _dist2d(lms[4], lms[8]) / palm_scale

        # Check ring & pinky extended (open palm release vs tether pose)
        rng_ext = _dist2d(wrist, lms[16]) > _dist2d(wrist, lms[14]) * 1.08
        pnk_ext = _dist2d(wrist, lms[20]) > _dist2d(wrist, lms[18]) * 1.08
        mid_ext = _dist2d(wrist, lms[12]) > _dist2d(wrist, lms[10]) * 1.08
        idx_ext = _dist2d(wrist, lms[8]) > _dist2d(wrist, lms[6]) * 1.08
        is_open_palm = idx_ext and mid_ext and rng_ext and pnk_ext
        other_curled = (not rng_ext) and (not pnk_ext)
        return thumb_pt, index_pt, pinch_ratio, is_open_palm, other_curled

    def _update_tether_state(self, hand_result, now: float) -> None:
        if not self.tether_glitch_enabled:
            self._tether_armed = False
            self._tether_active = False
            self._tether_mode = None
            return

        if not hand_result or not hand_result.hand_landmarks:
            if (now - self._tether_last_seen_ts) > 0.30:
                self._tether_armed = False
                self._tether_active = False
                self._tether_mode = None
                self._tether_prev_center = None
                self._tether_tension = 0.0
            return

        self._tether_last_seen_ts = now
        hands = hand_result.hand_landmarks

        # 2-Hand Tether: pinch thumb+index on both hands (or bring both hands together) then pull apart
        if len(hands) >= 2:
            t0, i0, r0, open0, _ = self._hand_pinch_geometry(hands[0])
            t1, i1, r1, open1, _ = self._hand_pinch_geometry(hands[1])
            mid0 = ((t0[0] + i0[0]) * 0.5, (t0[1] + i0[1]) * 0.5)
            mid1 = ((t1[0] + i1[0]) * 0.5, (t1[1] + i1[1]) * 0.5)
            span = math.hypot(mid1[0] - mid0[0], mid1[1] - mid0[1])

            if (r0 < 0.48 and r1 < 0.48) or (span < 0.14 and not open0 and not open1):
                self._tether_mode = "DUAL"
                self._tether_armed = True
                self._tether_active = span >= 0.09
                self._tether_pt_a, self._tether_pt_b = mid0, mid1
            elif self._tether_mode == "DUAL" and (self._tether_armed or self._tether_active):
                if open0 and open1:
                    self._tether_armed = False
                    self._tether_active = False
                    self._tether_mode = None
                    self._tether_tension = 0.0
                else:
                    self._tether_active = span >= 0.08
                    self._tether_pt_a, self._tether_pt_b = mid0, mid1
            if self._tether_mode == "DUAL":
                span_cur = math.hypot(self._tether_pt_b[0] - self._tether_pt_a[0], self._tether_pt_b[1] - self._tether_pt_a[1])
                self._tether_tension = float(np.clip((span_cur - 0.06) / 0.52, 0.0, 1.0))
                self._update_tether_velocity()
                return

        # 1-Hand Tether: touch thumb & index together (other fingers curled), then pull them apart
        t0, i0, r0, open0, other_curled = self._hand_pinch_geometry(hands[0])
        if open0:
            self._tether_armed = False
            self._tether_active = False
            self._tether_mode = None
            self._tether_prev_center = None
            self._tether_tension = 0.0
            return

        if r0 < 0.38 and other_curled:
            self._tether_mode = "SINGLE"
            self._tether_armed = True
            self._tether_active = False
            self._tether_pt_a, self._tether_pt_b = t0, i0
            self._tether_tension = 0.05
        elif self._tether_mode == "SINGLE" and (self._tether_armed or self._tether_active):
            if other_curled or r0 >= 0.40:
                self._tether_pt_a, self._tether_pt_b = t0, i0
                if r0 >= 0.42:
                    self._tether_active = True
                    self._tether_tension = float(np.clip((r0 - 0.38) / 1.05, 0.0, 1.0))
                else:
                    self._tether_active = False
                    self._tether_tension = 0.05
            else:
                self._tether_armed = False
                self._tether_active = False
                self._tether_mode = None
                self._tether_tension = 0.0

        self._update_tether_velocity()

    def _update_tether_velocity(self) -> None:
        cx = (self._tether_pt_a[0] + self._tether_pt_b[0]) * 0.5
        cy = (self._tether_pt_a[1] + self._tether_pt_b[1]) * 0.5
        if self._tether_prev_center is not None:
            dv = math.hypot(cx - self._tether_prev_center[0], cy - self._tether_prev_center[1])
            self._tether_vel = float(np.clip(0.35 * min(1.0, dv * 14.0) + 0.65 * self._tether_vel, 0.0, 1.0))
        self._tether_prev_center = (cx, cy)

    def _draw_tether_and_glitch_box(
        self,
        frame: np.ndarray,
        crop_norm: Tuple[float, float, float, float],
        now: float,
    ) -> None:
        h, w = frame.shape[:2]
        ax, ay = self._to_ptz_px(self._tether_pt_a[0], self._tether_pt_a[1], crop_norm, w, h)
        bx, by = self._to_ptz_px(self._tether_pt_b[0], self._tether_pt_b[1], crop_norm, w, h)

        # Anchor point sparks
        cv2.circle(frame, (ax, ay), 9, (255, 240, 0), 2, cv2.LINE_AA)
        cv2.circle(frame, (ax, ay), 4, (65, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (bx, by), 9, (255, 240, 0), 2, cv2.LINE_AA)
        cv2.circle(frame, (bx, by), 4, (65, 255, 0), -1, cv2.LINE_AA)

        if not self._tether_active:
            mx, my = (ax + bx) // 2, (ay + by) // 2
            cv2.putText(
                frame,
                "[LOCKED :: TARIK UNTUK GLITCH]",
                (max(10, mx - 110), max(24, my - 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 240, 0),
                1,
                cv2.LINE_AA,
            )
            return

        dist_px = math.hypot(bx - ax, by - ay)
        tension = self._tether_tension
        energy = float(np.clip(0.55 * tension + 0.45 * self._tether_vel, 0.15, 1.0))

        # 1. Compute Interactive Glitch Visual Box spanning pulled anchor points
        pad_x = int(max(36, dist_px * 0.24))
        pad_y = int(max(30, dist_px * 0.24))
        x1 = max(2, min(ax, bx) - pad_x)
        y1 = max(2, min(ay, by) - pad_y)
        x2 = min(w - 2, max(ax, bx) + pad_x)
        y2 = min(h - 2, max(ay, by) + pad_y)
        bw, bh = x2 - x1, y2 - y1

        if bw >= 24 and bh >= 24:
            roi = frame[y1:y2, x1:x2].copy()
            # Pixelate + RGB Chromatic Split driven by tension & hand motion velocity
            pix = max(3, int(3 + energy * 6))
            sw, sh = max(4, bw // pix), max(4, bh // pix)
            mosaic = cv2.resize(
                cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR),
                (bw, bh),
                interpolation=cv2.INTER_NEAREST,
            )
            roi = cv2.addWeighted(roi, 0.45, mosaic, 0.55, 0)

            rng = np.random.default_rng(int(now * 28) % 99991)
            rgb_shift = max(2, int(3 + energy * 10))
            b_ch, g_ch, r_ch = cv2.split(roi)
            r_ch = np.roll(r_ch, -rgb_shift, axis=1)
            b_ch = np.roll(b_ch, rgb_shift, axis=1)
            glitched = cv2.merge((b_ch, g_ch, r_ch))

            # Horizontal glitch slice tears inside the box
            n_slices = int(4 + energy * 8)
            for _ in range(n_slices):
                sy = int(rng.integers(0, max(1, bh - 4)))
                slice_h = int(rng.integers(2, max(3, min(14, bh // 4))))
                sy2 = min(bh, sy + slice_h)
                dx = int(rng.integers(-int(10 + energy * 24), int(11 + energy * 24)))
                glitched[sy:sy2] = np.roll(glitched[sy:sy2], dx, axis=1)
                if rng.random() < 0.38:
                    if rng.random() < 0.5:
                        glitched[sy:sy2, :, 1] = np.clip(glitched[sy:sy2, :, 1].astype(np.int16) + 90, 0, 255).astype(np.uint8)
                    else:
                        glitched[sy:sy2, :, 2] = np.clip(glitched[sy:sy2, :, 2].astype(np.int16) + 110, 0, 255).astype(np.uint8)

            # Corrupted cyber blocks + CRT scanlines
            glitched[::2] = (glitched[::2].astype(np.float32) * 0.78).astype(np.uint8)
            for _ in range(int(2 + energy * 4)):
                bx_i = int(rng.integers(0, max(1, bw - 14)))
                by_i = int(rng.integers(0, max(1, bh - 8)))
                bw_i = int(rng.integers(8, max(9, min(32, bw // 3))))
                bh_i = int(rng.integers(4, max(5, min(14, bh // 4))))
                tint = (65, 255, 0) if rng.random() < 0.45 else ((45, 35, 255) if rng.random() < 0.5 else (255, 235, 40))
                glitched[by_i : min(bh, by_i + bh_i), bx_i : min(bw, bx_i + bw_i)] = (
                    glitched[by_i : min(bh, by_i + bh_i), bx_i : min(bw, bx_i + bw_i)].astype(np.float32) * 0.4
                    + np.array(tint, dtype=np.float32) * 0.6
                ).astype(np.uint8)

            frame[y1:y2, x1:x2] = glitched

            # Chromatic ghost wireframe box + tactical corner brackets
            jx, jy = int(rng.integers(-4, 5)), int(rng.integers(-3, 4))
            cv2.rectangle(frame, (x1 + jx, y1 + jy), (x2 + jx, y2 + jy), (45, 35, 255), 1)
            cv2.rectangle(frame, (x1 - jx, y1 - jy), (x2 - jx, y2 - jy), (255, 235, 40), 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (65, 255, 0), 1)
            cl = min(18, bw // 4, bh // 4)
            for (px, py), (dx_c, dy_c) in [
                ((x1, y1), (1, 1)),
                ((x2, y1), (-1, 1)),
                ((x1, y2), (1, -1)),
                ((x2, y2), (-1, -1)),
            ]:
                cv2.line(frame, (px, py), (px + dx_c * cl, py), (65, 255, 0), 2, cv2.LINE_AA)
                cv2.line(frame, (px, py), (px, py + dy_c * cl), (65, 255, 0), 2, cv2.LINE_AA)

            tag = f"[GLITCH_BOX // TENSION:{int(tension * 100)}% // {bw}x{bh}]"
            cv2.putText(
                frame,
                tag,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (65, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # 2. Draw Elastic Tension Rope ("tali yang ditarik") connecting anchor A <-> B
        steps = 18
        ux = (bx - ax) / max(1.0, dist_px)
        uy = (by - ay) / max(1.0, dist_px)
        nx_p, ny_p = -uy, ux  # perpendicular normal vector
        sag_amp = (1.0 - tension) * min(22.0, dist_px * 0.14)
        vib_amp = 2.0 + energy * 7.5

        rope_pts = []
        echo_pts = []
        for s in range(steps + 1):
            t = s / float(steps)
            px = ax + (bx - ax) * t
            py = ay + (by - ay) * t + math.sin(math.pi * t) * sag_amp
            wave = math.sin(t * math.pi * 5.0 + now * 38.0) * vib_amp * math.sin(math.pi * t)
            rope_pts.append([int(px + nx_p * wave * 0.4), int(py + ny_p * wave * 0.4)])
            echo_pts.append([int(px - nx_p * wave), int(py - ny_p * wave)])

        rope_arr = np.array(rope_pts, dtype=np.int32).reshape((-1, 1, 2))
        echo_arr = np.array(echo_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [echo_arr], False, (45, 35, 255), 1, cv2.LINE_AA)
        cv2.polylines(frame, [rope_arr], False, (255, 240, 0), 4, cv2.LINE_AA)
        cv2.polylines(frame, [rope_arr], False, (210, 255, 220), 2, cv2.LINE_AA)

    @staticmethod
    def _classify_hand_gesture(lms) -> Tuple[str, int, float]:
        wrist = lms[0]
        index_mcp = lms[5]
        middle_mcp = lms[9]
        pinky_mcp = lms[17]
        palm_scale = max(0.04, _dist2d(wrist, middle_mcp))

        # Four fingers: tip further from wrist & MCP than PIP joint
        finger_indices = [
            (5, 6, 8),    # Index
            (9, 10, 12),  # Middle
            (13, 14, 16), # Ring
            (17, 18, 20), # Pinky
        ]
        extended = []
        for mcp_i, pip_i, tip_i in finger_indices:
            d_tip = _dist2d(wrist, lms[tip_i])
            d_pip = _dist2d(wrist, lms[pip_i])
            d_mcp_tip = _dist2d(lms[mcp_i], lms[tip_i])
            d_mcp_pip = _dist2d(lms[mcp_i], lms[pip_i])
            extended.append(d_tip > d_pip * 1.08 and d_mcp_tip > d_mcp_pip * 1.12)

        idx_ext, mid_ext, rng_ext, pnk_ext = extended

        # Thumb extension: tip further from pinky MCP and index MCP than thumb IP
        thumb_tip = lms[4]
        thumb_ip = lms[3]
        thumb_mcp = lms[2]
        thumb_ext = (
            _dist2d(thumb_tip, pinky_mcp) > _dist2d(thumb_ip, pinky_mcp) * 1.12
            and _dist2d(thumb_tip, index_mcp) > palm_scale * 0.45
        )

        finger_count = int(thumb_ext) + sum(extended)
        pinch_dist = _dist2d(thumb_tip, lms[8])
        pinch_ratio = pinch_dist / palm_scale

        # Pinch / OK gesture: thumb & index tips touching while middle/ring/pinky mostly open
        if pinch_ratio < 0.36 and (mid_ext or rng_ext or pnk_ext):
            return "PINCH", finger_count, pinch_ratio

        # 5 fingers open
        if thumb_ext and idx_ext and mid_ext and rng_ext and pnk_ext:
            return "OPEN_PALM", 5, pinch_ratio

        # 4 fingers open without thumb also counts comfortably as open palm
        if idx_ext and mid_ext and rng_ext and pnk_ext:
            return "OPEN_PALM", 4, pinch_ratio

        # Thumbs up / down: 4 fingers curled, thumb clearly pointing vertically
        if not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
            dy_thumb = thumb_tip.y - thumb_mcp.y
            if _dist2d(thumb_tip, index_mcp) > palm_scale * 0.48:
                if dy_thumb < -palm_scale * 0.34 and thumb_tip.y < wrist.y:
                    return "THUMBS_UP", 1, pinch_ratio
                if dy_thumb > palm_scale * 0.34 and thumb_tip.y > wrist.y - palm_scale * 0.1:
                    return "THUMBS_DOWN", 1, pinch_ratio
            return "FIST", 0, pinch_ratio

        # 1 finger pointing up
        if idx_ext and not mid_ext and not rng_ext and not pnk_ext:
            if lms[8].y < lms[6].y:
                return "POINT_UP", 1, pinch_ratio

        # Victory / Peace (2 fingers)
        if idx_ext and mid_ext and not rng_ext and not pnk_ext:
            return "VICTORY", 2, pinch_ratio

        # 3 fingers
        if idx_ext and mid_ext and rng_ext and not pnk_ext:
            return "THREE_FINGERS", 3, pinch_ratio

        # Rock / Horns (index + pinky)
        if idx_ext and not mid_ext and not rng_ext and pnk_ext:
            return "ROCK", finger_count, pinch_ratio

        return "NONE", finger_count, pinch_ratio

    def _analyze_face(self, result, now: float) -> FaceTelemetry:
        tel = FaceTelemetry(blink_count=self._blink_count)
        if not result or not result.face_landmarks:
            return tel

        tel.detected = True
        lms = result.face_landmarks[0]
        xs = [p.x for p in lms]
        ys = [p.y for p in lms]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        tel.box_norm = (
            (min_x + max_x) * 0.5,
            (min_y + max_y) * 0.5,
            max(0.08, max_x - min_x),
            max(0.08, max_y - min_y),
        )

        # Extract blendshapes
        bs: Dict[str, float] = {}
        if result.face_blendshapes and result.face_blendshapes[0]:
            for cat in result.face_blendshapes[0]:
                bs[cat.category_name] = float(cat.score)

        smile = (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) * 0.5
        jaw_open = bs.get("jawOpen", 0.0)
        brow_up = max(
            bs.get("browInnerUp", 0.0),
            (bs.get("browOuterUpLeft", 0.0) + bs.get("browOuterUpRight", 0.0)) * 0.5,
        )
        left_blink = bs.get("eyeBlinkLeft", 0.0) > 0.52
        right_blink = bs.get("eyeBlinkRight", 0.0) > 0.52

        tel.smile_score = smile
        tel.mouth_open_score = jaw_open
        tel.brow_up_score = brow_up
        tel.left_blink = left_blink
        tel.right_blink = right_blink

        both_blink = left_blink and right_blink
        if both_blink and not self._prev_both_blink:
            self._blink_count += 1
        self._prev_both_blink = both_blink
        tel.blink_count = self._blink_count

        # Expression priority classification
        if jaw_open > 0.42:
            tel.expression = "Mulut Terbuka / Terkejut"
        elif smile > 0.58:
            tel.expression = "Senyum Lebar"
        elif smile > 0.34:
            tel.expression = "Tersenyum"
        elif left_blink and not right_blink:
            tel.expression = "Kedip Mata Kiri"
        elif right_blink and not left_blink:
            tel.expression = "Kedip Mata Kanan"
        elif both_blink:
            tel.expression = "Mata Berkedip"
        elif brow_up > 0.45:
            tel.expression = "Alis Terangkat"
        else:
            tel.expression = "Netral"

        # Stable 3D-landmark Head Pose (immune to Euler sign flips across mirror/unmirror)
        nose = lms[1]
        forehead = lms[10]
        chin = lms[152]
        left_cheek = lms[234]
        right_cheek = lms[454]
        left_eye_pt = lms[33]
        right_eye_pt = lms[263]

        face_w = max(0.05, abs(right_cheek.x - left_cheek.x))
        face_h = max(0.05, abs(chin.y - forehead.y))
        cheek_mid_x = (left_cheek.x + right_cheek.x) * 0.5
        eye_mid_y = (left_eye_pt.y + right_eye_pt.y) * 0.5

        # Yaw: positive = turning toward right side of display, negative = left side
        tel.yaw = ((nose.x - cheek_mid_x) / (face_w * 0.5)) * 45.0

        # Pitch: combine 2D nose vertical shift and 3D forehead-vs-chin depth
        nose_ratio = (nose.y - eye_mid_y) / face_h
        pitch_2d = (0.25 - nose_ratio) * 120.0
        pitch_3d = ((forehead.z - chin.z) / face_h) * 95.0
        tel.pitch = 0.55 * pitch_2d + 0.45 * pitch_3d

        # Roll: eye-line tilt angle
        dx_eye = right_eye_pt.x - left_eye_pt.x
        dy_eye = right_eye_pt.y - left_eye_pt.y
        tel.roll = math.degrees(math.atan2(dy_eye, max(1e-4, dx_eye)))

        pose_parts: List[str] = []
        if tel.yaw > 16:
            pose_parts.append("Menoleh Kanan")
        elif tel.yaw < -16:
            pose_parts.append("Menoleh Kiri")

        if tel.pitch > 14:
            pose_parts.append("Mendongak")
        elif tel.pitch < -14:
            pose_parts.append("Menunduk")

        if tel.roll > 15:
            pose_parts.append("Miring Kanan")
        elif tel.roll < -15:
            pose_parts.append("Miring Kiri")

        tel.head_pose_label = " + ".join(pose_parts) if pose_parts else "Lurus ke Depan"

        # Detect Nod / Shake over short window
        self._pitch_history.append((now, tel.pitch))
        self._yaw_history.append((now, tel.yaw))
        if now - self._last_head_gesture_ts > 1.8 and len(self._pitch_history) >= 10:
            recent_pitches = [p for t, p in self._pitch_history if now - t < 0.85]
            recent_yaws = [y for t, y in self._yaw_history if now - t < 0.85]
            if recent_pitches and (max(recent_pitches) - min(recent_pitches)) > 20.0:
                tel.head_gesture = "NOD"
                self._last_head_gesture_ts = now
                self._pitch_history.clear()
            elif recent_yaws and (max(recent_yaws) - min(recent_yaws)) > 26.0:
                tel.head_gesture = "SHAKE"
                self._last_head_gesture_ts = now
                self._yaw_history.clear()

        return tel

    def _detect_glasses(
        self,
        frame_bgr: np.ndarray,
        lms,
        tel: FaceTelemetry,
        now: float,
    ) -> None:
        """Detect eyeglasses/sunglasses via normalized bridge/rim/temple edge contrast & brightness."""
        if abs(tel.yaw) > 28.0 or abs(tel.roll) > 28.0:
            tel.glasses_detected = self._glasses_state
            tel.glasses_score = self._glasses_ema
            return

        fh, fw = frame_bgr.shape[:2]
        xs = [p.x for p in lms]
        ys = [p.y for p in lms]
        fx1 = max(0, int((min(xs) - 0.04) * fw))
        fy1 = max(0, int((min(ys) - 0.04) * fh))
        fx2 = min(fw, int((max(xs) + 0.04) * fw))
        fy2 = min(fh, int((max(ys) + 0.04) * fh))
        bw, bh = fx2 - fx1, fy2 - fy1
        if bw < 40 or bh < 40:
            tel.glasses_detected = self._glasses_state
            tel.glasses_score = self._glasses_ema
            return

        face_crop = frame_bgr[fy1:fy2, fx1:fx2]
        norm_w = 180
        norm_h = max(60, int(round(bh * (norm_w / float(bw)))))
        face_resized = cv2.resize(face_crop, (norm_w, norm_h), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY), (3, 3), 0)

        gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        grad = 0.62 * gy + 0.38 * gx

        def to_patch_pt(lm) -> Tuple[float, float]:
            px = ((lm.x * fw) - fx1) * (norm_w / float(bw))
            py = ((lm.y * fh) - fy1) * (norm_h / float(bh))
            return px, py

        pts_a = [to_patch_pt(lms[i]) for i in self._left_eye_indices]
        pts_b = [to_patch_pt(lms[i]) for i in self._right_eye_indices]
        if np.mean([p[0] for p in pts_a]) > np.mean([p[0] for p in pts_b]):
            pts_a, pts_b = pts_b, pts_a

        outer_a = min(p[0] for p in pts_a)
        inner_a = max(p[0] for p in pts_a)
        inner_b = min(p[0] for p in pts_b)
        outer_b = max(p[0] for p in pts_b)
        mid_ya = float(np.mean([p[1] for p in pts_a]))
        mid_yb = float(np.mean([p[1] for p in pts_b]))
        bot_ya = max(p[1] for p in pts_a)
        bot_yb = max(p[1] for p in pts_b)

        eye_span = max(24.0, outer_b - outer_a)
        inner_span = max(8.0, inner_b - inner_a)
        bridge_cy = 0.5 * (mid_ya + mid_yb)

        def roi_stats(rx1: float, ry1: float, rx2: float, ry2: float) -> Tuple[float, float]:
            ix1 = max(0, min(norm_w - 2, int(round(rx1))))
            iy1 = max(0, min(norm_h - 2, int(round(ry1))))
            ix2 = max(ix1 + 2, min(norm_w, int(round(rx2))))
            iy2 = max(iy1 + 2, min(norm_h, int(round(ry2))))
            g_sub = grad[iy1:iy2, ix1:ix2]
            gr_sub = gray[iy1:iy2, ix1:ix2]
            if g_sub.size == 0:
                return 0.0, 128.0
            return float(np.percentile(g_sub, 85)), float(np.mean(gr_sub))

        # 1. Nose bridge between inner eye corners
        b_grad, b_bright = roi_stats(
            inner_a + 0.10 * inner_span,
            bridge_cy - 0.18 * inner_span,
            inner_b - 0.10 * inner_span,
            bridge_cy + 0.24 * inner_span,
        )

        # 2. Under-eye lower rim zones
        c_xa = 0.5 * (outer_a + inner_a)
        c_xb = 0.5 * (inner_b + outer_b)
        half_eye_w = max(6.0, 0.36 * (inner_a - outer_a))
        ra_grad, ra_bright = roi_stats(c_xa - half_eye_w, bot_ya + 0.05 * eye_span, c_xa + half_eye_w, bot_ya + 0.22 * eye_span)
        rb_grad, rb_bright = roi_stats(c_xb - half_eye_w, bot_yb + 0.05 * eye_span, c_xb + half_eye_w, bot_yb + 0.22 * eye_span)
        rim_grad = 0.5 * (ra_grad + rb_grad)
        rim_bright = 0.5 * (ra_bright + rb_bright)

        # 3. Outer temple hinges
        ta_grad, _ = roi_stats(outer_a - 0.15 * eye_span, mid_ya - 0.10 * eye_span, outer_a - 0.01 * eye_span, mid_ya + 0.10 * eye_span)
        tb_grad, _ = roi_stats(outer_b + 0.01 * eye_span, mid_yb - 0.10 * eye_span, outer_b + 0.15 * eye_span, mid_yb + 0.10 * eye_span)
        temple_grad = 0.5 * (ta_grad + tb_grad)

        # 4. Clean cheek skin reference
        ca_grad, ca_bright = roi_stats(c_xa - half_eye_w, bot_ya + 0.30 * eye_span, c_xa + half_eye_w, bot_ya + 0.46 * eye_span)
        cb_grad, cb_bright = roi_stats(c_xb - half_eye_w, bot_yb + 0.30 * eye_span, c_xb + half_eye_w, bot_yb + 0.46 * eye_span)
        skin_grad = max(12.0, 0.5 * (ca_grad + cb_grad))
        skin_bright = 0.5 * (ca_bright + cb_bright)

        bridge_ratio = b_grad / skin_grad
        rim_ratio = rim_grad / skin_grad
        temple_ratio = temple_grad / skin_grad

        # Also score darker frame/sunglass intensity drop vs cheek skin
        dark_drop = max(0.0, skin_bright - min(b_bright, rim_bright))
        dark_bonus = min(0.35, dark_drop / 110.0)

        edge_combo = 0.48 * bridge_ratio + 0.32 * rim_ratio + 0.20 * temple_ratio
        raw_score = float(np.clip((edge_combo - 1.10) / 1.90 + dark_bonus, 0.0, 1.0))

        self._glasses_raw_ema = 0.28 * raw_score + 0.72 * self._glasses_raw_ema
        if self._glasses_baseline is not None:
            calibrated = float(np.clip(0.22 + (self._glasses_raw_ema - self._glasses_baseline) * 1.45, 0.0, 1.0))
        else:
            calibrated = self._glasses_raw_ema

        self._glasses_ema = calibrated
        tel.glasses_score = calibrated

        # Hysteresis + temporal debounce
        want_on = calibrated >= self.glasses_threshold
        want_off = calibrated < (self.glasses_threshold - 0.10)
        cand = self._glasses_state
        if not self._glasses_state and want_on:
            cand = True
        elif self._glasses_state and want_off:
            cand = False

        if cand != self._glasses_state:
            if cand != self._glasses_cand_state:
                self._glasses_cand_state = cand
                self._glasses_cand_since = now
            else:
                req_hold = 0.45 if cand else 0.75
                if (now - self._glasses_cand_since) >= req_hold:
                    self._glasses_state = cand
                    if cand:
                        tel.glasses_just_worn = True
                        self._screen_glitch_burst_until = now + 1.8
                    else:
                        tel.glasses_just_removed = True
        else:
            self._glasses_cand_state = self._glasses_state

        tel.glasses_detected = self._glasses_state

    def _apply_screen_glitch(
        self,
        frame: np.ndarray,
        face_tel: FaceTelemetry,
        now: float,
    ) -> None:
        """Apply cyber-hacker full-frame glitch effect when active or triggered by glasses."""
        if self.screen_glitch_mode == "Off":
            return
        in_burst = now < self._screen_glitch_burst_until
        active = self.screen_glitch_mode == "Selalu Aktif" or (
            self.screen_glitch_mode == "Auto (Kacamata)" and (face_tel.glasses_detected or in_burst)
        )
        if not active:
            return

        h, w = frame.shape[:2]
        if h < 40 or w < 60:
            return

        refresh = 0.055 if in_burst else 0.095
        if now - self._screen_glitch_tick_ts > refresh:
            self._screen_glitch_tick_ts = now
            self._screen_glitch_seed = int(self._glitch_rng.integers(1, 100000))
        rng = np.random.default_rng(self._screen_glitch_seed)

        # 1. Subtle CRT scanline texture
        frame[::3] = (frame[::3].astype(np.float32) * (0.80 if in_burst else 0.87)).astype(np.uint8)

        # 2. RGB Chromatic Aberration channel shift
        shift = int(rng.integers(4, 9 if in_burst else 6))
        b_ch, g_ch, r_ch = cv2.split(frame)
        r_ch = np.roll(r_ch, -shift, axis=1)
        b_ch = np.roll(b_ch, shift, axis=1)
        cv2.merge((b_ch, g_ch, r_ch), dst=frame)

        # 3. Horizontal glitch slice tearing
        n_slices = int(rng.integers(4, 9 if in_burst else 6))
        for _ in range(n_slices):
            sy = int(rng.integers(0, max(1, h - 12)))
            sh = int(rng.integers(3, 14 if in_burst else 9))
            sy2 = min(h, sy + sh)
            dx = int(rng.integers(-28 if in_burst else -16, 29 if in_burst else 17))
            frame[sy:sy2] = np.roll(frame[sy:sy2], dx, axis=1)
            if rng.random() < 0.35:
                # Matrix green or crimson error scan flash
                if rng.random() < 0.55:
                    frame[sy:sy2, :, 1] = np.clip(frame[sy:sy2, :, 1].astype(np.int16) + 70, 0, 255).astype(np.uint8)
                else:
                    frame[sy:sy2, :, 2] = np.clip(frame[sy:sy2, :, 2].astype(np.int16) + 95, 0, 255).astype(np.uint8)

        # 4. Cyber glitch status banner
        jx = int(rng.integers(-3, 4))
        jy = int(rng.integers(-2, 3))
        banner = "[GLASSES_ON // AUDIO_OVERRIDE :: GLITCH]" if face_tel.glasses_detected else "[SYS_GLITCH // CYBER_FEED_ACTIVE]"
        bx, by = max(14, 18 + jx), max(28, h - 22 + jy)
        cv2.putText(frame, banner, (bx + 2, by), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (45, 35, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, banner, (bx - 2, by), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 235, 40), 2, cv2.LINE_AA)
        cv2.putText(frame, banner, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (65, 255, 0), 1, cv2.LINE_AA)

    @staticmethod
    def _to_ptz_px(
        nx: float,
        ny: float,
        crop_norm: Tuple[float, float, float, float],
        w: int,
        h: int,
    ) -> Tuple[int, int]:
        cx1, cy1, cx2, cy2 = crop_norm
        cw = max(1e-4, cx2 - cx1)
        ch = max(1e-4, cy2 - cy1)
        px = int(round(((nx - cx1) / cw) * w))
        py = int(round(((ny - cy1) / ch) * h))
        return px, py

    def _draw_hand_overlay(
        self,
        frame: np.ndarray,
        lms,
        crop_norm: Tuple[float, float, float, float],
        tel: Optional[HandTelemetry],
    ) -> None:
        h, w = frame.shape[:2]
        pts = [self._to_ptz_px(p.x, p.y, crop_norm, w, h) for p in lms]

        for conn in mp_vision.HandLandmarksConnections.HAND_CONNECTIONS:
            p1 = pts[conn.start]
            p2 = pts[conn.end]
            cv2.line(frame, p1, p2, (65, 255, 0), 2, cv2.LINE_AA)

        for idx, pt in enumerate(pts):
            is_tip = idx in (4, 8, 12, 16, 20)
            color = (255, 240, 0) if is_tip else (180, 255, 190)
            radius = 6 if is_tip else 4
            cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, radius + 1, (2, 8, 4), 1, cv2.LINE_AA)

        if tel is not None:
            wrist_pt = pts[0]
            cx = int(np.clip(wrist_pt[0], 45, w - 45))
            cy = int(np.clip(min(p[1] for p in pts) - 24, 34, h - 30))

            # Draw gesture hold ring
            if tel.hold_progress > 0.02:
                cv2.circle(frame, (cx, cy), 18, (12, 34, 18), 4, cv2.LINE_AA)
                end_angle = int(360 * tel.hold_progress)
                cv2.ellipse(frame, (cx, cy), (18, 18), -90, 0, end_angle, (65, 255, 0), 4, cv2.LINE_AA)

            badge = f"[{tel.gesture_label}]"
            if tel.pinch_active and tel.pinch_volume is not None:
                badge = f"[PINCH_VOL::{int(tel.pinch_volume * 100)}%]"
            cv2.putText(
                frame,
                badge,
                (max(12, cx - 80), max(22, cy - 26)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (65, 255, 0),
                2,
                cv2.LINE_AA,
            )

    def _apply_eye_filter(
        self,
        frame: np.ndarray,
        lms,
        crop_norm: Tuple[float, float, float, float],
        now: float,
    ) -> None:
        h, w = frame.shape[:2]
        left_pts = np.array(
            [self._to_ptz_px(lms[i].x, lms[i].y, crop_norm, w, h) for i in self._left_eye_indices],
            dtype=np.float32,
        )
        right_pts = np.array(
            [self._to_ptz_px(lms[i].x, lms[i].y, crop_norm, w, h) for i in self._right_eye_indices],
            dtype=np.float32,
        )
        if len(left_pts) == 0 or len(right_pts) == 0:
            return

        l_center = left_pts.mean(axis=0)
        r_center = right_pts.mean(axis=0)
        eye_vec = r_center - l_center
        eye_dist = float(np.linalg.norm(eye_vec))
        if eye_dist < 14.0:
            return

        u = eye_vec / eye_dist
        v = np.array([-u[1], u[0]], dtype=np.float32)
        center = (l_center + r_center) * 0.5

        half_w = max(24.0, eye_dist * 0.88)
        half_h = max(13.0, eye_dist * 0.28)

        corners = np.array(
            [
                center - u * half_w - v * half_h,
                center + u * half_w - v * half_h,
                center + u * half_w + v * half_h,
                center - u * half_w + v * half_h,
            ],
            dtype=np.int32,
        )

        # Expand slightly for RGB chromatic shift & glitch tearing margin
        pad = 12 if self.eye_filter_mode != "Blur Halus" else 2
        x1 = max(0, int(np.min(corners[:, 0])) - pad)
        y1 = max(0, int(np.min(corners[:, 1])) - pad)
        x2 = min(w, int(np.max(corners[:, 0])) + pad)
        y2 = min(h, int(np.max(corners[:, 1])) + pad)
        rw, rh = x2 - x1, y2 - y1
        if rw < 10 or rh < 8:
            return

        roi = frame[y1:y2, x1:x2].copy()
        local_poly = corners - np.array([x1, y1], dtype=np.int32)
        mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillPoly(mask, [local_poly], 255)

        # 1. Strong Blur + Pixelation Base
        k_val = max(15, int(half_h * 1.4) | 1)
        blurred = cv2.GaussianBlur(roi, (k_val, k_val), 0)

        pix_scale = 7 if self.eye_filter_mode == "Cyber Glitch" else 5
        small_w = max(4, rw // pix_scale)
        small_h = max(3, rh // pix_scale)
        pixelated = cv2.resize(
            cv2.resize(blurred, (small_w, small_h), interpolation=cv2.INTER_LINEAR),
            (rw, rh),
            interpolation=cv2.INTER_NEAREST,
        )
        effect = cv2.addWeighted(blurred, 0.52, pixelated, 0.48, 0)

        if self.eye_filter_mode == "Blur Halus":
            # Smooth frosted glass privacy blur
            mask_f = (cv2.GaussianBlur(mask, (7, 7), 0).astype(np.float32) / 255.0)[..., None]
            blended = (effect.astype(np.float32) * mask_f + roi.astype(np.float32) * (1.0 - mask_f)).astype(np.uint8)
            frame[y1:y2, x1:x2] = blended
            cv2.polylines(frame, [corners], True, (248, 189, 56), 1, cv2.LINE_AA)
            return

        # 2. System Error Glitch Effects ("Glitch Error" & "Cyber Glitch")
        if now - self._glitch_tick_ts > (0.065 if self.eye_filter_mode == "Cyber Glitch" else 0.085):
            self._glitch_tick_ts = now
            self._glitch_seed = int(self._glitch_rng.integers(1, 100000))
        rng = np.random.default_rng(self._glitch_seed)

        is_heavy = self.eye_filter_mode == "Cyber Glitch"
        rgb_shift = int(rng.integers(4, 11 if is_heavy else 8))
        b_ch, g_ch, r_ch = cv2.split(effect)
        r_ch = np.roll(r_ch, -rgb_shift, axis=1)
        b_ch = np.roll(b_ch, rgb_shift, axis=1)
        if is_heavy:
            g_ch = np.roll(g_ch, int(rng.integers(-2, 3)), axis=0)
        glitched = cv2.merge((b_ch, g_ch, r_ch))

        # Horizontal glitch slice tearing
        n_slices = int(rng.integers(6, 12 if is_heavy else 9))
        for _ in range(n_slices):
            sy = int(rng.integers(0, max(1, rh - 3)))
            sh = int(rng.integers(2, max(3, min(10, rh // 4))))
            sy2 = min(rh, sy + sh)
            dx = int(rng.integers(-20 if is_heavy else -14, 21 if is_heavy else 15))
            glitched[sy:sy2] = np.roll(glitched[sy:sy2], dx, axis=1)

            # Occasional crimson error or cyber cyan scan bar
            roll_pick = float(rng.random())
            if roll_pick < 0.24:
                glitched[sy:sy2, :, 2] = np.clip(glitched[sy:sy2, :, 2].astype(np.int16) + 120, 0, 255).astype(np.uint8)
                glitched[sy:sy2, :, 0] = (glitched[sy:sy2, :, 0] * 0.45).astype(np.uint8)
            elif roll_pick < 0.42:
                glitched[sy:sy2, :, 0] = np.clip(glitched[sy:sy2, :, 0].astype(np.int16) + 125, 0, 255).astype(np.uint8)
                glitched[sy:sy2, :, 1] = np.clip(glitched[sy:sy2, :, 1].astype(np.int16) + 95, 0, 255).astype(np.uint8)

        # CRT scanlines across the visor
        glitched[::2] = (glitched[::2].astype(np.float32) * 0.72).astype(np.uint8)

        # Corrupted digital micro-blocks
        for _ in range(4 if is_heavy else 2):
            bx = int(rng.integers(0, max(1, rw - 16)))
            by = int(rng.integers(0, max(1, rh - 6)))
            bw = int(rng.integers(8, max(9, min(26, rw // 3))))
            bh = int(rng.integers(3, max(4, min(8, rh // 3))))
            bx2, by2 = min(rw, bx + bw), min(rh, by + bh)
            tint = (45, 35, 255) if rng.random() < 0.55 else (255, 235, 40)
            glitched[by:by2, bx:bx2] = (
                glitched[by:by2, bx:bx2].astype(np.float32) * 0.45 + np.array(tint, dtype=np.float32) * 0.55
            ).astype(np.uint8)

        # Glitch mask itself with slight jagged edge offsets
        glitch_mask = mask.copy()
        for _ in range(4):
            my = int(rng.integers(0, max(1, rh - 3)))
            mh = int(rng.integers(2, max(3, min(7, rh // 4))))
            mdx = int(rng.integers(-10, 11))
            glitch_mask[my : min(rh, my + mh)] = np.roll(glitch_mask[my : min(rh, my + mh)], mdx, axis=1)

        mask_bool = glitch_mask > 127
        out_roi = roi
        out_roi[mask_bool] = glitched[mask_bool]
        frame[y1:y2, x1:x2] = out_roi

        # Draw crisp system-error frame & detached echo artifacts
        jitter_x = int(rng.integers(-3, 4))
        jitter_y = int(rng.integers(-2, 3))
        echo_poly = corners + np.array([jitter_x, jitter_y], dtype=np.int32)
        cv2.polylines(frame, [echo_poly], True, (45, 35, 255), 1, cv2.LINE_AA)
        cv2.polylines(frame, [corners], True, (255, 235, 40), 1, cv2.LINE_AA)

        err_labels = ("SYS.ERR // 0x404", "EYE_MASK // FAULT", "0xDEAD_SIGNAL", "ERR::BLUR_LOCK")
        err_tag = err_labels[int(rng.integers(0, len(err_labels)))]
        tx = int(np.clip(corners[0, 0] + jitter_x, 8, max(8, w - 165)))
        ty = int(np.clip(corners[0, 1] - 7 + jitter_y, 18, h - 8))
        cv2.putText(frame, err_tag, (tx + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (45, 35, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, err_tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 235, 40), 1, cv2.LINE_AA)

    def _draw_face_overlay(
        self,
        frame: np.ndarray,
        lms,
        crop_norm: Tuple[float, float, float, float],
        tel: FaceTelemetry,
        skip_eyes: bool = False,
    ) -> None:
        h, w = frame.shape[:2]
        pts = [self._to_ptz_px(p.x, p.y, crop_norm, w, h) for p in lms]

        contours = [
            (mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL, (65, 255, 0), 1),
            (mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS, (255, 240, 0), 1),
            (mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW, (0, 220, 255), 2),
            (mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW, (0, 220, 255), 2),
        ]
        if not skip_eyes:
            contours.extend(
                [
                    (mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE, (255, 240, 0), 1),
                    (mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE, (255, 240, 0), 1),
                ]
            )
        for conns, color, thickness in contours:
            for c in conns:
                cv2.line(frame, pts[c.start], pts[c.end], color, thickness, cv2.LINE_AA)

        # Face bounding corner reticle + expression tag
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = max(8, min(xs) - 10), min(w - 8, max(xs) + 10)
        y1, y2 = max(8, min(ys) - 14), min(h - 8, max(ys) + 10)
        corner = 18
        col = (65, 255, 0)
        for (px, py), (dx, dy) in [
            ((x1, y1), (1, 1)),
            ((x2, y1), (-1, 1)),
            ((x1, y2), (1, -1)),
            ((x2, y2), (-1, -1)),
        ]:
            cv2.line(frame, (px, py), (px + dx * corner, py), col, 2, cv2.LINE_AA)
            cv2.line(frame, (px, py), (px, py + dy * corner), col, 2, cv2.LINE_AA)

        tag = f"[FACE_SCAN] {tel.expression} | {tel.head_pose_label}"
        cv2.putText(
            frame,
            tag,
            (x1, max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (65, 255, 0),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_ptz_hud(
        frame: np.ndarray,
        snap: PTZSnapshot,
        face_box: Optional[Tuple[float, float, float, float]],
        hand_box: Optional[Tuple[float, float, float, float]],
    ) -> None:
        h, w = frame.shape[:2]

        # Tactical hacker viewfinder outer corner brackets
        margin = 10
        clen = 22
        bcol = (36, 145, 55)
        for (px, py), (dx, dy) in [
            ((margin, margin), (1, 1)),
            ((w - margin, margin), (-1, 1)),
            ((margin, h - margin), (1, -1)),
            ((w - margin, h - margin), (-1, -1)),
        ]:
            cv2.line(frame, (px, py), (px + dx * clen, py), bcol, 1, cv2.LINE_AA)
            cv2.line(frame, (px, py), (px, py + dy * clen), bcol, 1, cv2.LINE_AA)

        # Top-left live PTZ terminal pill
        status_color = (65, 255, 0) if snap.target_locked else (0, 220, 255)
        cv2.rectangle(frame, (16, 16), (390, 52), (2, 7, 3), -1)
        cv2.rectangle(frame, (16, 16), (390, 52), status_color, 1)
        cv2.circle(frame, (32, 34), 5, status_color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"CAM::PTZ [{snap.mode}] | {snap.zoom:.2f}x | {snap.target_label}",
            (46, 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 255, 195),
            1,
            cv2.LINE_AA,
        )

        # Bottom-right PTZ sensor radar minimap showing full camera view vs zoomed window
        mw, mh = 154, 88
        mx1, my1 = w - mw - 16, h - mh - 16
        mx2, my2 = mx1 + mw, my1 + mh
        overlay = frame.copy()
        cv2.rectangle(overlay, (mx1, my1), (mx2, my2), (2, 7, 3), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (29, 105, 45), 1)
        cv2.line(frame, (mx1 + mw // 2, my1), (mx1 + mw // 2, my2), (16, 56, 25), 1)
        cv2.line(frame, (mx1, my1 + mh // 2), (mx2, my1 + mh // 2), (16, 56, 25), 1)

        cx1, cy1, cx2, cy2 = snap.crop_norm
        rx1 = int(mx1 + cx1 * mw)
        ry1 = int(my1 + cy1 * mh)
        rx2 = int(mx1 + cx2 * mw)
        ry2 = int(my1 + cy2 * mh)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (65, 255, 0), 2)

        if face_box is not None:
            fx, fy = int(mx1 + face_box[0] * mw), int(my1 + face_box[1] * mh)
            cv2.circle(frame, (fx, fy), 4, (255, 240, 0), -1)
        if hand_box is not None:
            hx, hy = int(mx1 + hand_box[0] * mw), int(my1 + hand_box[1] * mh)
            cv2.circle(frame, (hx, hy), 4, (65, 255, 0), -1)
