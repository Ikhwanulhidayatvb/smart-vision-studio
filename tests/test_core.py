"""Core unit & integration tests for Smart Vision Studio."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

# Ensure writable matplotlib config dir before importing mediapipe
_mpl_dir = Path(tempfile.gettempdir()) / "smart_vision_test_mpl"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

from src.config import EYE_FILTER_MODES, MUSIC_DIR, OVERLAY_MODES
from src.music_generator import ensure_demo_music
from src.music_player import MusicPlayer
from src.ptz_camera import PTZController
from src.vision_engine import FaceTelemetry, VisionEngine
from src.voice_engine import SantoKeywordStore, parse_voice_command


def _pt(x: float, y: float, z: float = 0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def _make_hand_landmarks(
    thumb_ext: bool = False,
    thumb_dir: str = "side",
    idx_ext: bool = False,
    mid_ext: bool = False,
    rng_ext: bool = False,
    pnk_ext: bool = False,
    pinch: bool = False,
):
    lms = [_pt(0.5, 0.78) for _ in range(21)]
    # Wrist & MCPs
    lms[0] = _pt(0.5, 0.80)
    lms[1] = _pt(0.45, 0.73)
    lms[2] = _pt(0.42, 0.67)
    lms[5] = _pt(0.44, 0.60)
    lms[9] = _pt(0.50, 0.59)
    lms[13] = _pt(0.56, 0.60)
    lms[17] = _pt(0.61, 0.63)

    # Thumb
    if thumb_ext:
        if thumb_dir == "up":
            lms[3] = _pt(0.40, 0.57)
            lms[4] = _pt(0.39, 0.45)
        elif thumb_dir == "down":
            lms[0] = _pt(0.5, 0.45)
            lms[2] = _pt(0.42, 0.50)
            lms[5] = _pt(0.44, 0.55)
            lms[9] = _pt(0.50, 0.56)
            lms[13] = _pt(0.56, 0.55)
            lms[17] = _pt(0.61, 0.52)
            lms[3] = _pt(0.40, 0.63)
            lms[4] = _pt(0.39, 0.74)
        else:
            lms[3] = _pt(0.34, 0.65)
            lms[4] = _pt(0.28, 0.61)
    else:
        lms[3] = _pt(0.48, 0.65)
        lms[4] = _pt(0.51, 0.64)

    # Fingers: (mcp, pip, dip, tip, x, extended)
    specs = [
        (5, 6, 7, 8, 0.44, idx_ext),
        (9, 10, 11, 12, 0.50, mid_ext),
        (13, 14, 15, 16, 0.56, rng_ext),
        (17, 18, 19, 20, 0.61, pnk_ext),
    ]
    for mcp_i, pip_i, dip_i, tip_i, x, ext in specs:
        base_y = lms[mcp_i].y
        if ext:
            lms[pip_i] = _pt(x, base_y - 0.08)
            lms[dip_i] = _pt(x, base_y - 0.14)
            lms[tip_i] = _pt(x, base_y - 0.20)
        else:
            lms[pip_i] = _pt(x, base_y - 0.04)
            lms[dip_i] = _pt(x, base_y + 0.01)
            lms[tip_i] = _pt(x, base_y + 0.03)

    if pinch:
        lms[4] = _pt(0.41, 0.50)
        lms[8] = _pt(0.42, 0.51)

    return lms


def test_demo_music_and_player():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        music_dir = tmp_path / "music"
        external_dir = tmp_path / "user_songs"
        playlist_file = tmp_path / "playlist.json"

        # Verify MusicPlayer starts clean without auto-creating demo files
        player = MusicPlayer(music_dir=music_dir, playlist_file=playlist_file)
        assert len(player.get_state().tracks) == 0

        # Generate sample WAVs in external folder and load them as user songs
        files = ensure_demo_music(external_dir)
        assert len(files) >= 4
        user_wav1 = external_dir / "My_Custom_Song_A.wav"
        user_wav2 = external_dir / "My_Custom_Song_B.wav"
        files[0].rename(user_wav1)
        files[1].rename(user_wav2)

        added = player.add_files([str(user_wav1), str(user_wav2)])
        assert added == 2
        assert len(player.get_state().tracks) == 2

        # Mark second track for glasses auto-play and verify persistence
        marked = player.set_marked_track(1)
        assert marked is not None and marked.title == "My Custom Song B"

        # Persistence across restart
        player2 = MusicPlayer(music_dir=music_dir, playlist_file=playlist_file)
        assert len(player2.get_state().tracks) == 2
        assert player2.get_state().marked_index == 1

        state = player2.get_state()
        assert len(state.spectrum) == 16
        vol = player2.set_volume(0.5)
        assert abs(vol - 0.5) < 1e-6
        player2.set_ducking(True)
        player2.set_ducking(False)
        assert player2.toggle_mute() is True
        assert player2.toggle_mute() is False

        tr_marked = player2.play_marked_track()
        if player2.audio_ready:
            assert tr_marked is not None and tr_marked.title == "My Custom Song B"
            assert player2.get_state().current_index == 1
            player2.stop()

        tr = player2.play(0)
        if player2.audio_ready:
            assert tr is not None
            assert player2.get_state().is_playing is True
            player2.pause()
            assert player2.get_state().is_paused is True
            player2.next_track()
            assert player2.get_state().current_index == 1
            player2.prev_track()
            assert player2.get_state().current_index == 0
            player2.stop()

        removed = player2.remove_track(0)
        assert removed == "My Custom Song A"
        assert len(player2.get_state().tracks) == 1
        player2.clear_playlist()
        assert len(player2.get_state().tracks) == 0


def test_ptz_controller():
    ptz = PTZController()
    ptz.set_mode("Auto Face")
    for _ in range(20):
        ptz.update_tracking((0.7, 0.4, 0.2, 0.2), None)
    assert ptz.target_locked is True
    assert ptz.current_pan_x > 0.55
    assert ptz.current_zoom > 1.05

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    out, snap = ptz.apply(frame)
    assert out.shape == frame.shape
    assert snap.zoom == ptz.current_zoom

    ptz.nudge(-0.1, 0.05)
    assert ptz.mode == "Manual PTZ"
    ptz.reset()
    assert ptz.target_pan_x == 0.5 and ptz.target_pan_y == 0.5 and ptz.target_zoom == 1.0


def test_gesture_classification_and_vision_models():
    ptz = PTZController()
    engine = VisionEngine(ptz)
    assert engine._models_error == ""

    assert engine._classify_hand_gesture(_make_hand_landmarks(True, "side", True, True, True, True))[0] == "OPEN_PALM"
    assert engine._classify_hand_gesture(_make_hand_landmarks(False, "side", False, False, False, False))[0] == "FIST"
    assert engine._classify_hand_gesture(_make_hand_landmarks(False, "side", True, False, False, False))[0] == "POINT_UP"
    assert engine._classify_hand_gesture(_make_hand_landmarks(False, "side", True, True, False, False))[0] == "VICTORY"
    assert engine._classify_hand_gesture(_make_hand_landmarks(False, "side", True, True, True, False))[0] == "THREE_FINGERS"
    assert engine._classify_hand_gesture(_make_hand_landmarks(False, "side", True, False, False, True))[0] == "ROCK"
    assert engine._classify_hand_gesture(_make_hand_landmarks(True, "up", False, False, False, False))[0] == "THUMBS_UP"
    assert engine._classify_hand_gesture(_make_hand_landmarks(True, "down", False, False, False, False))[0] == "THUMBS_DOWN"
    assert engine._classify_hand_gesture(_make_hand_landmarks(False, "side", False, True, True, True, pinch=True))[0] == "PINCH"
    engine.release()


def test_voice_command_parser():
    with tempfile.TemporaryDirectory() as tmp:
        store = SantoKeywordStore(path=Path(tmp) / "santo_keywords.json")
        assert parse_voice_command("Santo", store) == ("SANTO_WAKE", "Yes, sir.")
        assert parse_voice_command("Hey Santo!", store) == ("SANTO_WAKE", "Yes, sir.")
        assert parse_voice_command("Who am I?", store) == (
            "CUSTOM_REPLY",
            "Boss Budi, is there anything I can help you with?",
        )
        assert parse_voice_command("Santo, who am I?", store) == (
            "CUSTOM_REPLY",
            "Boss Budi, is there anything I can help you with?",
        )
        assert parse_voice_command("Santo, play music", store) == ("PLAY_MUSIC", None)

        # Future extensible keyword rule
        added = store.add_rule(["system status", "check status"], "All systems are online, Boss Budi.")
        assert parse_voice_command("Santo system status", store) == (
            "CUSTOM_REPLY",
            "All systems are online, Boss Budi.",
        )
        assert store.remove_rule(added["id"]) is True

    assert parse_voice_command("Putar musik sekarang") == ("PLAY_MUSIC", None)
    assert parse_voice_command("Lagu selanjutnya") == ("NEXT_TRACK", None)
    assert parse_voice_command("Lagu sebelumnya") == ("PREV_TRACK", None)
    assert parse_voice_command("Besarkan suara") == ("VOLUME_UP", None)
    assert parse_voice_command("Ikuti wajah saya") == ("PTZ_MODE", "Auto Face")
    assert parse_voice_command("Geser kiri") == ("PTZ_PAN_LEFT", None)
    assert parse_voice_command("Zoom in") == ("PTZ_ZOOM_IN", None)
    assert parse_voice_command("Reset kamera") == ("PTZ_RESET", None)
    assert parse_voice_command("Katakan selamat pagi dunia") == ("SPEAK_CUSTOM", "selamat pagi dunia")
    assert parse_voice_command("Overlay muka saja") == ("OVERLAY_MODE", "Muka Saja")
    assert parse_voice_command("Overlay tangan saja") == ("OVERLAY_MODE", "Tangan Saja")
    assert parse_voice_command("Efek mata glitch") == ("EYE_FILTER_MODE", "Glitch Error")
    assert parse_voice_command("Matikan efek mata") == ("EYE_FILTER_MODE", "Off")


def test_eye_glitch_and_overlay_modes():
    ptz = PTZController()
    engine = VisionEngine(ptz)

    # Create synthetic 478 face landmarks with left & right eye regions
    face_lms = [_pt(0.5, 0.5) for _ in range(478)]
    for idx in engine._left_eye_indices:
        face_lms[idx] = _pt(0.41, 0.44)
    for idx in engine._right_eye_indices:
        face_lms[idx] = _pt(0.59, 0.45)

    base_frame = np.full((360, 640, 3), 120, dtype=np.uint8)
    base_frame[140:180, 220:420] = (180, 90, 60)

    for mode in EYE_FILTER_MODES:
        engine.set_eye_filter_mode(mode)
        frame = base_frame.copy()
        if mode != "Off":
            engine._apply_eye_filter(frame, face_lms, (0.0, 0.0, 1.0, 1.0), 10.0)
            assert not np.array_equal(frame, base_frame), f"Eye filter {mode} did not modify frame"

    for ov in OVERLAY_MODES:
        engine.set_overlay_mode(ov)
        frame = base_frame.copy()
        if ov in ("Muka & Tangan", "Muka Saja"):
            engine._draw_face_overlay(frame, face_lms, (0.0, 0.0, 1.0, 1.0), FaceTelemetry(), skip_eyes=True)
            assert not np.array_equal(frame, base_frame)

    engine.release()


def test_glasses_detection_and_screen_glitch():
    ptz = PTZController()
    engine = VisionEngine(ptz)

    # Realistic face landmarks in 640x360 frame
    lms = [_pt(0.5, 0.5) for _ in range(478)]
    lms[10] = _pt(0.5, 0.22)
    lms[152] = _pt(0.5, 0.78)
    lms[234] = _pt(0.32, 0.50)
    lms[454] = _pt(0.68, 0.50)

    for i, idx in enumerate(engine._right_eye_indices):  # viewer-left eye cluster
        ang = (2 * np.pi * i) / len(engine._right_eye_indices)
        lms[idx] = _pt(0.42 + 0.035 * np.cos(ang), 0.44 + 0.015 * np.sin(ang))
    for i, idx in enumerate(engine._left_eye_indices):   # viewer-right eye cluster
        ang = (2 * np.pi * i) / len(engine._left_eye_indices)
        lms[idx] = _pt(0.58 + 0.035 * np.cos(ang), 0.44 + 0.015 * np.sin(ang))

    # 1. Bare face (smooth skin tone) -> should NOT trigger glasses
    bare_frame = np.full((360, 640, 3), (150, 175, 200), dtype=np.uint8)
    t = 1.0
    for step in range(10):
        tel = FaceTelemetry()
        engine._detect_glasses(bare_frame, lms, tel, t + step * 0.1)
    assert tel.glasses_detected is False
    assert tel.glasses_score < 0.40

    # 2. Face wearing dark glasses frames across bridge + rims + temples
    glasses_frame = bare_frame.copy()
    # Left & right frame boxes + bridge bar + temple arms
    cv2.rectangle(glasses_frame, (242, 146), (296, 178), (20, 20, 20), 3)
    cv2.rectangle(glasses_frame, (344, 146), (398, 178), (20, 20, 20), 3)
    cv2.line(glasses_frame, (296, 158), (344, 158), (18, 18, 18), 4)
    cv2.line(glasses_frame, (220, 156), (242, 158), (18, 18, 18), 3)
    cv2.line(glasses_frame, (398, 158), (420, 156), (18, 18, 18), 3)

    worn_triggered = False
    for step in range(18):
        tel = FaceTelemetry()
        engine._detect_glasses(glasses_frame, lms, tel, 5.0 + step * 0.1)
        if tel.glasses_just_worn:
            worn_triggered = True
    assert worn_triggered is True
    assert tel.glasses_detected is True
    assert tel.glasses_score >= 0.5

    # 3. Screen glitch modifies frame when glasses are detected
    before = glasses_frame.copy()
    engine._apply_screen_glitch(glasses_frame, tel, 7.0)
    assert not np.array_equal(glasses_frame, before)

    engine.release()


def test_cyber_tether_and_glitch_box():
    ptz = PTZController()
    engine = VisionEngine(ptz)

    # 1. Single-hand pinch -> pull thumb & index apart -> active tether & glitch box
    lms = _make_hand_landmarks(False, "side", False, False, False, False, pinch=True)
    res_armed = SimpleNamespace(hand_landmarks=[lms])
    engine._update_tether_state(res_armed, 1.0)
    assert engine._tether_armed is True
    assert engine._tether_active is False

    # Pull thumb (4) and index (8) apart while ring & pinky stay curled
    lms_pull = _make_hand_landmarks(False, "side", True, False, False, False, pinch=False)
    lms_pull[4] = _pt(0.28, 0.56)
    lms_pull[8] = _pt(0.62, 0.42)
    engine._update_tether_state(SimpleNamespace(hand_landmarks=[lms_pull]), 1.1)
    assert engine._tether_active is True
    assert engine._tether_tension > 0.2

    base = np.full((360, 640, 3), 90, dtype=np.uint8)
    drawn = base.copy()
    engine._draw_tether_and_glitch_box(drawn, (0.0, 0.0, 1.0, 1.0), 1.1)
    assert not np.array_equal(drawn, base)

    # 2. Dual-hand pinch -> pull both hands apart
    h_left = _make_hand_landmarks(False, "side", False, False, False, False, pinch=True)
    h_right = _make_hand_landmarks(False, "side", False, False, False, False, pinch=True)
    for p in h_left:
        p.x -= 0.18
    for p in h_right:
        p.x += 0.18
    engine._update_tether_state(SimpleNamespace(hand_landmarks=[h_left, h_right]), 2.0)
    assert engine._tether_mode == "DUAL"
    assert engine._tether_active is True
    drawn2 = base.copy()
    engine._draw_tether_and_glitch_box(drawn2, (0.0, 0.0, 1.0, 1.0), 2.1)
    assert not np.array_equal(drawn2, base)

    engine.release()


if __name__ == "__main__":
    test_demo_music_and_player()
    test_ptz_controller()
    test_gesture_classification_and_vision_models()
    test_voice_command_parser()
    test_eye_glitch_and_overlay_modes()
    test_glasses_detection_and_screen_glitch()
    test_cyber_tether_and_glitch_box()
    print("ALL CORE TESTS PASSED!")
