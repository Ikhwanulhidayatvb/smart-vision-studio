"""Digital Pan-Tilt-Zoom (PTZ) & Auto-Tracking Camera Engine."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import PTZ_MAX_ZOOM, PTZ_MIN_ZOOM, PTZ_SMOOTHING


@dataclass
class PTZSnapshot:
    mode: str
    zoom: float
    pan_x: float
    pan_y: float
    crop_norm: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in [0, 1]
    target_locked: bool
    target_label: str


class PTZController:
    """Smooth digital Pan-Tilt-Zoom controller with Auto Face/Hand tracking."""

    def __init__(self) -> None:
        self.mode: str = "Auto Face"
        self.auto_zoom: bool = True

        self.target_zoom: float = 1.0
        self.target_pan_x: float = 0.5
        self.target_pan_y: float = 0.5

        self.current_zoom: float = 1.0
        self.current_pan_x: float = 0.5
        self.current_pan_y: float = 0.5

        self.manual_zoom_bias: float = 0.0
        self.last_target_seen_ts: float = 0.0
        self.target_locked: bool = False
        self.target_label: str = "Standby"

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        if mode == "Manual PTZ":
            self.target_locked = False
            self.target_label = "Manual Control"

    def set_auto_zoom(self, enabled: bool) -> None:
        self.auto_zoom = enabled

    def reset(self) -> None:
        self.target_zoom = 1.0
        self.target_pan_x = 0.5
        self.target_pan_y = 0.5
        self.manual_zoom_bias = 0.0
        self.target_label = "Center Reset"

    def nudge(self, dx: float = 0.0, dy: float = 0.0) -> None:
        """Move virtual camera pan/tilt manually (switches to Manual PTZ if needed)."""
        self.mode = "Manual PTZ"
        step_scale = 1.0 / max(self.current_zoom, 1.0)
        self.target_pan_x = float(np.clip(self.target_pan_x + dx * step_scale, 0.05, 0.95))
        self.target_pan_y = float(np.clip(self.target_pan_y + dy * step_scale, 0.05, 0.95))
        self.target_label = "Manual Pan/Tilt"

    def adjust_zoom(self, delta: float) -> None:
        if self.mode == "Manual PTZ" or not self.auto_zoom:
            self.target_zoom = float(np.clip(self.target_zoom + delta, PTZ_MIN_ZOOM, PTZ_MAX_ZOOM))
        else:
            self.manual_zoom_bias = float(np.clip(self.manual_zoom_bias + delta, -0.8, 1.5))

    def set_zoom(self, zoom: float) -> None:
        self.target_zoom = float(np.clip(zoom, PTZ_MIN_ZOOM, PTZ_MAX_ZOOM))

    def update_tracking(
        self,
        face_box: Optional[Tuple[float, float, float, float]],
        hand_box: Optional[Tuple[float, float, float, float]],
    ) -> None:
        """Update PTZ targets based on detected face/hand bounding boxes (cx, cy, w, h) normalized [0, 1]."""
        now = time.monotonic()
        chosen_box: Optional[Tuple[float, float, float, float]] = None
        label = "Searching..."

        if self.mode == "Auto Face":
            if face_box is not None:
                chosen_box = face_box
                label = "Lock: Wajah"
        elif self.mode == "Auto Hand":
            if hand_box is not None:
                chosen_box = hand_box
                label = "Lock: Tangan"
        elif self.mode == "Auto Hybrid":
            if hand_box is not None:
                chosen_box = hand_box
                label = "Lock: Tangan (Hybrid)"
            elif face_box is not None:
                chosen_box = face_box
                label = "Lock: Wajah (Hybrid)"
        elif self.mode == "Manual PTZ":
            self.target_locked = False
            self.target_label = "Manual PTZ"
            self._smooth_step()
            return

        if chosen_box is not None:
            cx, cy, bw, bh = chosen_box
            self.last_target_seen_ts = now
            self.target_locked = True
            self.target_label = label

            # Deadzone avoids micro-jitter when subject holds still
            if abs(cx - self.target_pan_x) > 0.03:
                self.target_pan_x = float(np.clip(cx, 0.1, 0.9))
            if abs(cy - self.target_pan_y) > 0.03:
                self.target_pan_y = float(np.clip(cy, 0.1, 0.9))

            if self.auto_zoom:
                desired_subject_size = 0.36 if "Wajah" in label else 0.42
                subject_span = max(bw, bh, 0.12)
                ideal_zoom = (desired_subject_size / subject_span) + self.manual_zoom_bias
                self.target_zoom = float(np.clip(ideal_zoom, 1.12, PTZ_MAX_ZOOM))
        else:
            self.target_locked = False
            if now - self.last_target_seen_ts > 1.8:
                # Ease back to full frame if target left the view
                self.target_pan_x = 0.5
                self.target_pan_y = 0.5
                if self.auto_zoom:
                    self.target_zoom = 1.0
                self.target_label = "Wide Standby"

        self._smooth_step()

    def _smooth_step(self) -> None:
        alpha = PTZ_SMOOTHING
        self.current_pan_x += (self.target_pan_x - self.current_pan_x) * alpha
        self.current_pan_y += (self.target_pan_y - self.current_pan_y) * alpha
        self.current_zoom += (self.target_zoom - self.current_zoom) * alpha
        self.current_zoom = float(np.clip(self.current_zoom, PTZ_MIN_ZOOM, PTZ_MAX_ZOOM))

    def apply(self, frame: np.ndarray) -> Tuple[np.ndarray, PTZSnapshot]:
        """Apply smoothed digital Pan-Tilt-Zoom crop & resize to frame."""
        h, w = frame.shape[:2]
        z = max(1.0, self.current_zoom)

        if z <= 1.01:
            crop_norm = (0.0, 0.0, 1.0, 1.0)
            out = frame.copy()
        else:
            crop_w = max(64, int(round(w / z)))
            crop_h = max(48, int(round(h / z)))

            cx_px = int(round(self.current_pan_x * w))
            cy_px = int(round(self.current_pan_y * h))

            x1 = max(0, min(w - crop_w, cx_px - crop_w // 2))
            y1 = max(0, min(h - crop_h, cy_px - crop_h // 2))
            x2 = x1 + crop_w
            y2 = y1 + crop_h

            cropped = frame[y1:y2, x1:x2]
            out = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            crop_norm = (x1 / w, y1 / h, x2 / w, y2 / h)

        snapshot = PTZSnapshot(
            mode=self.mode,
            zoom=self.current_zoom,
            pan_x=self.current_pan_x,
            pan_y=self.current_pan_y,
            crop_norm=crop_norm,
            target_locked=self.target_locked,
            target_label=self.target_label,
        )
        return out, snapshot
