"""
CAMERA MOUSE - Desktop Controller
Control your real mouse cursor with hand gestures

Gesture Controls (total fingers across both hands):
  1 finger  =  Move mouse cursor
  2 fingers =  Left click (one-shot, return to 1 finger to click again)
  3 fingers =  Right click (one-shot, return to 1 finger to click again)
  4 fingers =  Scroll (move hand up/down to scroll)
  5 fingers =  Copy (Ctrl+C)
  6 fingers =  Paste (Ctrl+V)
  7 fingers =  Take screenshot
  10 fingers = Show help overlay

Press 'Q' on the camera window to quit.
Press 'R' to reset cursor to center.
"""

import cv2
import numpy as np
import pyautogui
import time
import os
import sys
import ctypes
import urllib.request
import threading
import json
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from functools import partial

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─── Configuration ───────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
WINDOW_NAME = "Camera Mouse"
WEB_PORT = 5678

# Default settings (all adjustable via web UI)
DEFAULT_SETTINGS = {
    'smoothing': 0.5,
    'sensitivity_x': 1.6,
    'sensitivity_y': 1.6,
    'max_cursor_speed': 120,
    'scroll_sensitivity': 12,
    'mode_switch_delay': 0.15,
    'boundary_pad': 0.08,
    'thumb_margin': 0.04,
    'thumb_dist_threshold': 0.08,
    'acceleration_enabled': False,
    'accel_min_multiplier': 0.5,
    'accel_max_multiplier': 2.5,
    'accel_threshold_low': 0.008,
    'accel_threshold_high': 0.06,
}

# Cooldowns for one-shot actions (seconds)
COPY_COOLDOWN = 1.0
PASTE_COOLDOWN = 1.0
SCREENSHOT_COOLDOWN = 2.0

# Model
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Settings HTML file path
SETTINGS_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.html")

# PyAutoGUI
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0
pyautogui.MINIMUM_DURATION = 0

# ─── Colors (BGR) ────────────────────────────────────────
C_MOVE      = (255, 229, 0)       # Cyan
C_CLICK     = (102, 68, 255)      # Red
C_RCLICK    = (0, 170, 255)       # Orange (right click)
C_SCROLL    = (0, 255, 200)       # Teal (scroll)
C_COPY      = (200, 255, 100)     # Lime
C_PASTE     = (255, 180, 50)      # Light blue
C_SCREENSHOT= (200, 100, 255)     # Purple
C_HELP      = (255, 255, 255)     # White
C_PAUSED    = (120, 120, 120)     # Gray
C_IDLE      = (80, 80, 80)        # Dark gray
C_BG        = (18, 18, 26)
C_WHITE     = (255, 255, 255)
C_GREEN     = (100, 255, 100)
C_RED       = (100, 100, 255)
C_CALIB     = (0, 200, 255)       # Calibration color


def download_model():
    """Download hand landmarker model if not present."""
    if os.path.exists(MODEL_PATH):
        return
    print(f"  Downloading hand tracking model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"  Model downloaded! ({os.path.getsize(MODEL_PATH) / 1024 / 1024:.1f} MB)")
    except Exception as e:
        print(f"  ERROR downloading model: {e}")
        print(f"  Download manually: {MODEL_URL}")
        print(f"  Save to: {MODEL_PATH}")
        sys.exit(1)


# ─── Web Server for Settings UI ─────────────────────────
class SettingsHandler(BaseHTTPRequestHandler):
    """HTTP handler for the settings web UI."""

    def __init__(self, camera_mouse_app, *args, **kwargs):
        self.app = camera_mouse_app
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        pass  # Suppress noisy server logs

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get('Content-Length', 0))
        return self.rfile.read(length)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            # Serve settings.html
            try:
                with open(SETTINGS_HTML, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self.send_error(404, "settings.html not found")

        elif self.path == '/api/settings':
            self._send_json(dict(self.app.settings))

        elif self.path == '/api/status':
            status = {
                'fps': self.app.fps,
                'fingers': self.app.total_fingers,
                'mode': self.app.current_mode,
                'cursor_x': int(self.app.smooth_x),
                'cursor_y': int(self.app.smooth_y),
                'hand_detected': self.app.current_hand_detected,
                'calibrating': self.app.calibrating,
                'calib_bounds': None,
                'calib_active': self.app.calib_active_bounds,
                'hand_x_raw': self.app.current_hand_x,
                'hand_y_raw': self.app.current_hand_y,
            }
            if self.app.calibrating:
                status['calib_bounds'] = {
                    'min_x': self.app.calib_min_x,
                    'max_x': self.app.calib_max_x,
                    'min_y': self.app.calib_min_y,
                    'max_y': self.app.calib_max_y,
                }
            self._send_json(status)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/api/settings':
            try:
                body = self._read_body()
                new_settings = json.loads(body)
                for k, v in new_settings.items():
                    if k in self.app.settings:
                        expected_type = type(DEFAULT_SETTINGS[k])
                        if expected_type == bool:
                            self.app.settings[k] = bool(v)
                        elif expected_type == float:
                            self.app.settings[k] = float(v)
                        elif expected_type == int:
                            self.app.settings[k] = int(v)
                self._send_json(dict(self.app.settings))
            except Exception as e:
                self._send_json({'error': str(e)}, 400)

        elif self.path == '/api/calibration/start':
            self.app.calibrating = True
            self.app.calib_min_x = 1.0
            self.app.calib_max_x = 0.0
            self.app.calib_min_y = 1.0
            self.app.calib_max_y = 0.0
            print("  [Calibration] Started — move hand to all corners")
            self._send_json({'status': 'started'})

        elif self.path == '/api/calibration/stop':
            self.app.calibrating = False
            range_x = self.app.calib_max_x - self.app.calib_min_x
            range_y = self.app.calib_max_y - self.app.calib_min_y
            if range_x > 0.05 and range_y > 0.05:
                # Add small padding to calibrated bounds
                pad = 0.02
                self.app.calib_active_bounds = {
                    'min_x': max(0, self.app.calib_min_x - pad),
                    'max_x': min(1, self.app.calib_max_x + pad),
                    'min_y': max(0, self.app.calib_min_y - pad),
                    'max_y': min(1, self.app.calib_max_y + pad),
                }
                print(f"  [Calibration] Saved bounds: X[{self.app.calib_active_bounds['min_x']:.2f}-{self.app.calib_active_bounds['max_x']:.2f}] Y[{self.app.calib_active_bounds['min_y']:.2f}-{self.app.calib_active_bounds['max_y']:.2f}]")
            else:
                print("  [Calibration] Range too small, not applied")
            self._send_json({'status': 'stopped', 'bounds': self.app.calib_active_bounds})

        elif self.path == '/api/calibration/reset':
            self.app.calibrating = False
            self.app.calib_active_bounds = None
            print("  [Calibration] Reset to default boundaries")
            self._send_json({'status': 'reset'})

        else:
            self.send_error(404)


class CameraMouse:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()

        download_model()

        # Create hand landmarker (detect BOTH hands for 6+ finger gestures)
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,  # Both hands for 6-10 finger gestures
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)

        # Dynamic settings (adjustable via web UI)
        self.settings = dict(DEFAULT_SETTINGS)

        # Cursor state
        self.smooth_x = self.screen_w / 2
        self.smooth_y = self.screen_h / 2

        # Mode stabilization
        self.prev_finger_count = 0
        self.stable_finger_count = 0
        self.finger_change_time = 0

        # Cursor acceleration state
        self.prev_hand_x = None
        self.prev_hand_y = None

        # Calibration state
        self.calibrating = False
        self.calib_min_x = 1.0
        self.calib_max_x = 0.0
        self.calib_min_y = 1.0
        self.calib_max_y = 0.0
        self.calib_active_bounds = None

        # State flags
        self.click_performed = False    # True after 2-finger left click, resets on 1 finger
        self.rclick_performed = False   # True after 3-finger right click, resets on 1 finger
        self.copy_performed = False     # One-shot flag for copy
        self.paste_performed = False    # One-shot flag for paste
        self.screenshot_performed = False
        self.show_help = False

        # Scroll state
        self.scroll_anchor_y = None     # Y position when scroll mode started
        self.last_scroll_y = None       # Last Y position for scroll delta

        # Status for web UI
        self.current_mode = "WAITING..."
        self.current_hand_detected = False
        self.current_hand_x = 0.5
        self.current_hand_y = 0.5

        # Counters
        self.click_count = 0
        self.rclick_count = 0
        self.scroll_count = 0
        self.copy_count = 0
        self.paste_count = 0
        self.screenshot_count = 0
        self.total_fingers = 0
        self.fps = 0
        self.frame_count = 0
        self.fps_time = time.time()
        self.frame_timestamp = 0

        # Screenshot dir
        self.screenshot_dir = os.path.join(
            os.path.expanduser("~"), "Desktop", "Camera Mouse Screenshots"
        )

        # Action feedback (flashes text on screen)
        self.action_text = ""
        self.action_color = C_WHITE
        self.action_time = 0

    def start_web_server(self):
        """Start the settings web UI server in a background thread."""
        handler = partial(SettingsHandler, self)
        try:
            server = HTTPServer(('127.0.0.1', WEB_PORT), handler)
            server.daemon_threads = True
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{WEB_PORT}"
            print(f"  Settings UI: {url}")
            webbrowser.open(url)
        except OSError as e:
            print(f"  WARNING: Could not start settings server on port {WEB_PORT}: {e}")
            print(f"  The camera will still work, but web settings won't be available.")

    def flash_action(self, text, color):
        """Show a temporary action indicator."""
        self.action_text = text
        self.action_color = color
        self.action_time = time.time()

    def count_fingers_one_hand(self, landmarks, handedness_label):
        """Count extended fingers for a single hand."""
        count = 0

        # Thumb - use stricter detection to avoid false positives
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_mcp = landmarks[5]

        thumb_dist_x = abs(thumb_tip.x - index_mcp.x)
        thumb_dist_y = abs(thumb_tip.y - index_mcp.y)
        thumb_dist = (thumb_dist_x ** 2 + thumb_dist_y ** 2) ** 0.5

        t_margin = self.settings['thumb_margin']
        t_dist = self.settings['thumb_dist_threshold']
        if thumb_dist > t_dist:
            if handedness_label == "Right":
                if thumb_tip.x < thumb_ip.x - t_margin:
                    count += 1
            else:
                if thumb_tip.x > thumb_ip.x + t_margin:
                    count += 1

        # Index (8 vs 6), Middle (12 vs 10), Ring (16 vs 14), Pinky (20 vs 18)
        finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
        for tip, pip_joint in finger_pairs:
            if landmarks[tip].y < landmarks[pip_joint].y:
                count += 1

        return count

    def get_stable_fingers(self, raw_count, now):
        """Stabilize finger count to prevent mode flickering."""
        if raw_count != self.prev_finger_count:
            self.prev_finger_count = raw_count
            self.finger_change_time = now
            return self.stable_finger_count

        if now - self.finger_change_time >= self.settings['mode_switch_delay']:
            self.stable_finger_count = raw_count

        return self.stable_finger_count

    def take_screenshot(self):
        """Take screenshot and save to Desktop."""
        os.makedirs(self.screenshot_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.screenshot_dir, f"screenshot_{timestamp}.png")
        pyautogui.screenshot().save(filepath)
        self.screenshot_count += 1
        print(f"  Screenshot saved: {filepath}")

    def draw_hand(self, frame, landmarks_list, color):
        """Draw hand skeleton on frame."""
        h, w = frame.shape[:2]
        connections = [
            (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
        ]
        points = [(int((1 - lm.x) * w), int(lm.y * h)) for lm in landmarks_list]

        conn_color = tuple(c // 2 for c in color)
        for s, e in connections:
            if s < len(points) and e < len(points):
                cv2.line(frame, points[s], points[e], conn_color, 2, cv2.LINE_AA)

        fingertips = [4, 8, 12, 16, 20]
        for i, pt in enumerate(points):
            if i in fingertips:
                cv2.circle(frame, pt, 6, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 8, color, 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 3, C_WHITE, -1, cv2.LINE_AA)

        # Highlight index fingertip
        if len(points) > 8:
            cv2.circle(frame, points[8], 10, C_GREEN, 2, cv2.LINE_AA)

    def draw_help_overlay(self, frame):
        """Draw the full help/options overlay when 10 fingers shown."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 18), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Title
        cv2.putText(frame, "CAMERA MOUSE - GESTURE GUIDE", (w // 2 - 190, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_WHITE, 2, cv2.LINE_AA)

        # Gesture list
        gestures = [
            ("1 Finger", "Move Cursor", C_MOVE),
            ("2 Fingers", "Left Click", C_CLICK),
            ("3 Fingers", "Right Click", C_RCLICK),
            ("4 Fingers", "Scroll Up/Down", C_SCROLL),
            ("5 Fingers", "Copy (Ctrl+C)", C_COPY),
            ("6 Fingers", "Paste (Ctrl+V)", C_PASTE),
            ("7 Fingers", "Take Screenshot", C_SCREENSHOT),
            ("10 Fingers", "Show This Help", C_HELP),
        ]

        start_y = 80
        for i, (fingers, action, color) in enumerate(gestures):
            y = start_y + i * 38

            # Colored dot
            cv2.circle(frame, (40, y + 4), 8, color, -1, cv2.LINE_AA)

            # Finger count
            cv2.putText(frame, fingers, (60, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            # Arrow
            cv2.putText(frame, "->", (200, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

            # Action
            cv2.putText(frame, action, (240, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 210), 1, cv2.LINE_AA)

        # Footer
        cv2.putText(frame, "Show fewer fingers to dismiss", (w // 2 - 130, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1, cv2.LINE_AA)

        return frame

    def draw_ui(self, frame, mode_name, mode_color, finger_count, hand_count):
        """Draw the HUD overlay."""
        h, w = frame.shape[:2]
        now = time.time()

        # ── Top bar ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 54), C_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        cv2.putText(frame, "CAMERA MOUSE", (14, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{self.fps} FPS", (w - 80, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

        # Mode + finger count
        cv2.putText(frame, mode_name, (14, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2, cv2.LINE_AA)

        finger_text = f"{finger_count}F"
        if hand_count == 2:
            finger_text += " (2 hands)"
        cv2.putText(frame, finger_text, (w - 110, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        # Hand dot
        dot_color = C_GREEN if hand_count > 0 else C_RED
        cv2.circle(frame, (w - 24, 18), 6, dot_color, -1)

        # ── Calibration indicator ──
        if self.calibrating:
            cv2.putText(frame, "CALIBRATING", (w // 2 - 55, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_CALIB, 2, cv2.LINE_AA)
        elif self.calib_active_bounds:
            cv2.putText(frame, "CALIBRATED", (w // 2 - 50, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_GREEN, 1, cv2.LINE_AA)

        # ── Acceleration indicator ──
        if self.settings['acceleration_enabled']:
            cv2.putText(frame, "ACCEL", (w // 2 - 50, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1, cv2.LINE_AA)

        # ── Bottom bar ──
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 70), (w, h), C_BG, -1)
        cv2.addWeighted(overlay2, 0.85, frame, 0.15, 0, frame)

        y1 = h - 48
        cv2.putText(frame, f"L-Click:{self.click_count}", (10, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, f"R-Click:{self.rclick_count}", (120, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Scrolls:{self.scroll_count}", (240, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Copy:{self.copy_count} Paste:{self.paste_count}", (360, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)

        y2 = h - 22
        cx, cy = int(self.smooth_x), int(self.smooth_y)
        cv2.putText(frame, f"Cursor: ({cx}, {cy})", (10, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Screenshots: {self.screenshot_count}", (220, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)

        # ── Mini gesture guide (right side) ──
        guide_x = w - 155
        mini_guide = [
            ("1F:Move", C_MOVE), ("2F:L-Click", C_CLICK), ("3F:R-Click", C_RCLICK),
            ("4F:Scroll", C_SCROLL), ("5F:Copy", C_COPY), ("6F:Paste", C_PASTE),
            ("7F:Screenshot", C_SCREENSHOT), ("10F:Help", C_HELP),
        ]
        for i, (text, col) in enumerate(mini_guide):
            y = h - 68 + i * 13
            # Dim unless active
            is_active = (
                (finger_count == 1 and "1F" in text) or
                (finger_count == 2 and "2F" in text) or
                (finger_count == 3 and "3F" in text) or
                (finger_count == 4 and "4F" in text) or
                (finger_count == 5 and "5F" in text) or
                (finger_count == 6 and "6F" in text) or
                (finger_count == 7 and "7F" in text) or
                (finger_count >= 10 and "10F" in text)
            )
            draw_col = col if is_active else tuple(c // 4 for c in col)
            thick = 2 if is_active else 1
            cv2.putText(frame, text, (guide_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, draw_col, thick, cv2.LINE_AA)
            if is_active:
                cv2.circle(frame, (guide_x - 6, y - 3), 3, col, -1)

        # ── Color bar at top ──
        bar_color = C_CALIB if self.calibrating else mode_color
        cv2.rectangle(frame, (0, 0), (w, 3), bar_color, -1)

        # ── Action flash ──
        if now - self.action_time < 0.8:
            alpha = max(0, 1.0 - (now - self.action_time) / 0.8)
            act_color = tuple(int(c * alpha) for c in self.action_color)
            cv2.putText(frame, self.action_text, (w // 2 - len(self.action_text) * 7, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, act_color, 2, cv2.LINE_AA)

        return frame

    def release_all_holds(self):
        """Release any mouse holds and reset scroll state."""
        self.scroll_anchor_y = None
        self.last_scroll_y = None

    def move_cursor_to_hand(self, ix, iy):
        """Map hand position to screen cursor with smoothing, acceleration, and calibration."""
        s = self.settings

        # ── Step 1: Calculate cursor acceleration multiplier ──
        accel = 1.0
        if s['acceleration_enabled'] and self.prev_hand_x is not None:
            hand_dx = ix - self.prev_hand_x
            hand_dy = iy - self.prev_hand_y
            hand_speed = (hand_dx ** 2 + hand_dy ** 2) ** 0.5

            low = s['accel_threshold_low']
            high = s['accel_threshold_high']
            min_mult = s['accel_min_multiplier']
            max_mult = s['accel_max_multiplier']

            if hand_speed <= low:
                accel = min_mult       # Precision mode
            elif hand_speed >= high:
                accel = max_mult       # Fast mode
            else:
                # Smooth interpolation between min and max
                t = (hand_speed - low) / (high - low)
                accel = min_mult + t * (max_mult - min_mult)

        self.prev_hand_x = ix
        self.prev_hand_y = iy

        # ── Step 2: Map hand position to screen coordinates ──
        if self.calib_active_bounds:
            # Use calibrated boundaries
            bounds = self.calib_active_bounds
            range_x = max(bounds['max_x'] - bounds['min_x'], 0.01)
            range_y = max(bounds['max_y'] - bounds['min_y'], 0.01)
            mapped_x = 1.0 - (ix - bounds['min_x']) / range_x
            mapped_y = (iy - bounds['min_y']) / range_y
        else:
            # Use default boundary padding
            pad = s['boundary_pad']
            mapped_x = (1.0 - ix - pad) / (1 - 2 * pad)
            mapped_y = (iy - pad) / (1 - 2 * pad)

        # Apply sensitivity with acceleration
        eff_sens_x = s['sensitivity_x'] * accel
        eff_sens_y = s['sensitivity_y'] * accel
        mapped_x = 0.5 + (mapped_x - 0.5) * eff_sens_x
        mapped_y = 0.5 + (mapped_y - 0.5) * eff_sens_y

        target_x = max(0, min(self.screen_w - 1, mapped_x * self.screen_w))
        target_y = max(0, min(self.screen_h - 1, mapped_y * self.screen_h))

        # ── Step 3: Apply smoothing ──
        smoothing = s['smoothing']
        new_x = self.smooth_x + (target_x - self.smooth_x) * smoothing
        new_y = self.smooth_y + (target_y - self.smooth_y) * smoothing

        # ── Step 4: Clamp speed to prevent sudden jumps ──
        max_speed = s['max_cursor_speed']
        dx = new_x - self.smooth_x
        dy = new_y - self.smooth_y
        dist = (dx ** 2 + dy ** 2) ** 0.5
        if dist > max_speed:
            scale = max_speed / dist
            new_x = self.smooth_x + dx * scale
            new_y = self.smooth_y + dy * scale

        self.smooth_x = new_x
        self.smooth_y = new_y

        pyautogui.moveTo(int(self.smooth_x), int(self.smooth_y), _pause=False)

    def run(self):
        """Main loop."""
        print("\n" + "=" * 56)
        print("  CAMERA MOUSE - Desktop Controller")
        print("=" * 56)
        print(f"  Screen: {self.screen_w} x {self.screen_h}")
        print("  Press 'Q' to quit | 'R' to reset cursor")
        print("=" * 56)
        print()
        print("  Gestures:")
        print("    1F = Move    2F = L-Click  3F = R-Click")
        print("    4F = Scroll  5F = Copy     6F = Paste")
        print("    7F = Screenshot            10F = Help")
        print()

        # Start the settings web server
        self.start_web_server()

        print("  Starting camera...")
        print()

        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not cap.isOpened():
            print("  ERROR: Could not open camera!")
            input("  Press Enter to exit...")
            return

        print("  Camera ready! Show your hand.\n")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 480, 360)

        try:
            hwnd = ctypes.windll.user32.FindWindowW(None, WINDOW_NAME)
            if hwnd:
                ctypes.windll.user32.SetWindowPos(hwnd, -1, 50, 50, 480, 400, 0x0001)
        except Exception:
            pass

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)  # Mirror for display
                h, w = frame.shape[:2]
                now = time.time()

                # FPS
                self.frame_count += 1
                if now - self.fps_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.fps_time = now

                # Detect hands (un-mirror for detection)
                detect_frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                self.frame_timestamp += 33
                results = self.hand_landmarker.detect_for_video(mp_image, self.frame_timestamp)

                hand_count = 0
                total_fingers = 0
                primary_ix = 0.5
                primary_iy = 0.5
                mode_name = "WAITING..."
                mode_color = C_IDLE

                if results.hand_landmarks and len(results.hand_landmarks) > 0:
                    hand_count = len(results.hand_landmarks)

                    # Count fingers across ALL hands
                    for i, landmarks in enumerate(results.hand_landmarks):
                        label = "Right"
                        if results.handedness and i < len(results.handedness):
                            label = results.handedness[i][0].category_name

                        total_fingers += self.count_fingers_one_hand(landmarks, label)

                    # Use first hand's index finger for cursor position
                    primary_ix = results.hand_landmarks[0][8].x
                    primary_iy = results.hand_landmarks[0][8].y

                    # Draw all hands
                    for i, landmarks in enumerate(results.hand_landmarks):
                        self.draw_hand(frame, landmarks, C_MOVE)

                # Update status for web UI
                self.current_hand_detected = hand_count > 0
                self.current_hand_x = primary_ix
                self.current_hand_y = primary_iy

                # Record calibration data
                if self.calibrating and hand_count > 0:
                    self.calib_min_x = min(self.calib_min_x, primary_ix)
                    self.calib_max_x = max(self.calib_max_x, primary_ix)
                    self.calib_min_y = min(self.calib_min_y, primary_iy)
                    self.calib_max_y = max(self.calib_max_y, primary_iy)

                # Stabilize finger count
                fingers = self.get_stable_fingers(total_fingers, now)
                self.total_fingers = fingers

                # ═══════════════════════════════════════════
                # GESTURE STATE MACHINE
                # ═══════════════════════════════════════════

                if hand_count == 0:
                    # No hand - release everything
                    self.release_all_holds()
                    self.click_performed = False
                    self.rclick_performed = False
                    self.copy_performed = False
                    self.paste_performed = False
                    self.screenshot_performed = False
                    self.show_help = False
                    # Reset acceleration tracking
                    self.prev_hand_x = None
                    self.prev_hand_y = None
                    mode_name = "WAITING..."
                    mode_color = C_IDLE

                elif fingers >= 10:
                    # ═══ 10 FINGERS: HELP ═══
                    self.release_all_holds()
                    self.show_help = True
                    mode_name = "HELP"
                    mode_color = C_HELP

                elif fingers == 7 or fingers == 8 or fingers == 9:
                    # ═══ 7 FINGERS: SCREENSHOT ═══
                    self.release_all_holds()
                    mode_name = "SCREENSHOT"
                    mode_color = C_SCREENSHOT
                    self.show_help = False

                    if not self.screenshot_performed:
                        self.screenshot_performed = True
                        self.take_screenshot()
                        self.flash_action("SCREENSHOT!", C_SCREENSHOT)

                elif fingers == 6:
                    # ═══ 6 FINGERS: PASTE ═══
                    self.release_all_holds()
                    mode_name = "PASTE"
                    mode_color = C_PASTE
                    self.show_help = False

                    if not self.paste_performed:
                        self.paste_performed = True
                        self.paste_count += 1
                        pyautogui.hotkey('ctrl', 'v', _pause=False)
                        self.flash_action("PASTED!", C_PASTE)
                        print(f"  Paste #{self.paste_count}")

                elif fingers == 5:
                    # ═══ 5 FINGERS: COPY ═══
                    self.release_all_holds()
                    mode_name = "COPY"
                    mode_color = C_COPY
                    self.show_help = False

                    if not self.copy_performed:
                        self.copy_performed = True
                        self.copy_count += 1
                        pyautogui.hotkey('ctrl', 'c', _pause=False)
                        self.flash_action("COPIED!", C_COPY)
                        print(f"  Copy #{self.copy_count}")

                elif fingers == 4:
                    # ═══ 4 FINGERS: SCROLL ═══
                    mode_name = "SCROLL"
                    mode_color = C_SCROLL
                    self.show_help = False

                    if self.scroll_anchor_y is None:
                        # Just entered scroll mode
                        self.scroll_anchor_y = primary_iy
                        self.last_scroll_y = primary_iy
                        self.scroll_count += 1
                        self.flash_action("SCROLL MODE", C_SCROLL)
                        print(f"  Scroll #{self.scroll_count} started")
                    else:
                        # Calculate scroll delta from hand movement
                        delta_y = primary_iy - self.last_scroll_y
                        scroll_sens = self.settings['scroll_sensitivity']
                        scroll_amount = int(delta_y * scroll_sens * 100)
                        if abs(scroll_amount) > 0:
                            # Negative scroll_amount = scroll up (hand moves up)
                            pyautogui.scroll(-scroll_amount, _pause=False)
                            self.last_scroll_y = primary_iy

                elif fingers == 3:
                    # ═══ 3 FINGERS: RIGHT CLICK (one-shot) ═══
                    mode_name = "R-CLICK"
                    mode_color = C_RCLICK
                    self.show_help = False
                    self.scroll_anchor_y = None
                    self.last_scroll_y = None

                    # Right click only once per 1->3 transition
                    if not self.rclick_performed:
                        self.rclick_performed = True
                        self.rclick_count += 1
                        pyautogui.rightClick(_pause=False)
                        self.flash_action("RIGHT CLICK!", C_RCLICK)
                        print(f"  Right Click #{self.rclick_count} at ({int(self.smooth_x)}, {int(self.smooth_y)})")

                    # Cursor does NOT move in right-click mode (frozen)

                elif fingers == 2:
                    # ═══ 2 FINGERS: LEFT CLICK (one-shot) ═══
                    mode_name = "L-CLICK"
                    mode_color = C_CLICK
                    self.show_help = False
                    self.scroll_anchor_y = None
                    self.last_scroll_y = None

                    # Left click only once per 1->2 transition
                    if not self.click_performed:
                        self.click_performed = True
                        self.click_count += 1
                        pyautogui.click(_pause=False)
                        self.flash_action("LEFT CLICK!", C_CLICK)
                        print(f"  Left Click #{self.click_count} at ({int(self.smooth_x)}, {int(self.smooth_y)})")

                    # Cursor does NOT move in click mode (frozen)

                elif fingers == 1:
                    # ═══ 1 FINGER: MOVE CURSOR ═══
                    mode_name = "MOVE"
                    mode_color = C_MOVE
                    self.show_help = False

                    # Release any holds and reset scroll
                    self.release_all_holds()

                    # Reset one-shot flags so they can trigger again
                    self.click_performed = False
                    self.rclick_performed = False
                    self.copy_performed = False
                    self.paste_performed = False
                    self.screenshot_performed = False

                    # Move cursor
                    self.move_cursor_to_hand(primary_ix, primary_iy)

                else:
                    # 0 fingers (fist) or other
                    mode_name = "IDLE"
                    mode_color = C_IDLE
                    self.release_all_holds()
                    self.show_help = False
                    # Reset acceleration tracking on fist
                    self.prev_hand_x = None
                    self.prev_hand_y = None

                # Update current mode for web UI
                self.current_mode = mode_name

                # ═══ Draw UI ═══
                if self.show_help:
                    frame = self.draw_help_overlay(frame)
                else:
                    frame = self.draw_ui(frame, mode_name, mode_color, fingers, hand_count)

                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\n  Quitting...")
                    break
                elif key == ord('r') or key == ord('R'):
                    self.smooth_x = self.screen_w / 2
                    self.smooth_y = self.screen_h / 2
                    pyautogui.moveTo(int(self.smooth_x), int(self.smooth_y))
                    print("  Cursor reset to center")

        except KeyboardInterrupt:
            print("\n  Interrupted.")
        finally:
            self.release_all_holds()
            cap.release()
            cv2.destroyAllWindows()
            self.hand_landmarker.close()

            print("\n" + "=" * 56)
            print("  Session Summary")
            print("=" * 56)
            print(f"  L-Clicks:    {self.click_count}")
            print(f"  R-Clicks:    {self.rclick_count}")
            print(f"  Scrolls:     {self.scroll_count}")
            print(f"  Copies:      {self.copy_count}")
            print(f"  Pastes:      {self.paste_count}")
            print(f"  Screenshots: {self.screenshot_count}")
            if self.screenshot_count > 0:
                print(f"  Saved to:    {self.screenshot_dir}")
            print("=" * 56 + "\n")


if __name__ == "__main__":
    app = CameraMouse()
    app.run()
