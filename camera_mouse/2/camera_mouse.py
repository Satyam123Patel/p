"""
CAMERA MOUSE - Desktop Controller
Control your real mouse cursor with hand gestures

Gesture Controls:
  1 finger  =  Move mouse cursor across entire screen
  2 fingers =  Click at current cursor position
  3 fingers =  Drag (hold & move items/folders)
  4 fingers =  Take a screenshot (saved to Desktop)
  5 fingers =  Shut down / pause tracking

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
from datetime import datetime

# MediaPipe Tasks API (new API - works with mediapipe 0.10.35+)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─── Configuration ───────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
WINDOW_NAME = "Camera Mouse"

# Cursor smoothing (0-1, lower = smoother but laggier)
SMOOTHING = 0.25

# Sensitivity multiplier for cursor movement
SENSITIVITY_X = 1.6
SENSITIVITY_Y = 1.6

# Cooldowns (seconds)
CLICK_COOLDOWN = 0.5
SCREENSHOT_COOLDOWN = 2.0
MODE_SWITCH_DELAY = 0.18  # Prevents flickering between modes

# Boundary padding (fraction of frame to ignore at edges)
BOUNDARY_PAD = 0.08

# Model file
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# ─── PyAutoGUI Safety ────────────────────────────────────
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0
pyautogui.MINIMUM_DURATION = 0

# ─── Colors (BGR) ────────────────────────────────────────
COLOR_MOVE = (255, 229, 0)
COLOR_CLICK = (102, 68, 255)
COLOR_DRAG = (0, 170, 255)
COLOR_SCREENSHOT = (200, 100, 255)
COLOR_PAUSED = (120, 120, 120)
COLOR_IDLE = (80, 80, 80)
COLOR_BG = (18, 18, 26)
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (100, 255, 100)
COLOR_RED = (100, 100, 255)


def download_model():
    """Download the hand landmarker model if not present."""
    if os.path.exists(MODEL_PATH):
        return
    print(f"  Downloading hand tracking model...")
    print(f"  From: {MODEL_URL}")
    print(f"  To:   {MODEL_PATH}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"  Model downloaded successfully! ({os.path.getsize(MODEL_PATH) / 1024 / 1024:.1f} MB)")
    except Exception as e:
        print(f"  ERROR downloading model: {e}")
        print(f"  Please download manually from:")
        print(f"  {MODEL_URL}")
        print(f"  and save to: {MODEL_PATH}")
        sys.exit(1)


class CameraMouse:
    def __init__(self):
        # Screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()

        # Download model if needed
        download_model()

        # Create hand landmarker using Tasks API
        base_options = mp_python.BaseOptions(
            model_asset_path=MODEL_PATH
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)

        # State
        self.smooth_x = self.screen_w / 2
        self.smooth_y = self.screen_h / 2
        self.prev_mode = "idle"
        self.stable_mode = "idle"
        self.mode_change_time = 0
        self.last_click_time = 0
        self.last_screenshot_time = 0
        self.is_dragging = False
        self.is_paused = False
        self.click_count = 0
        self.drag_count = 0
        self.screenshot_count = 0
        self.finger_count = 0
        self.fps = 0
        self.frame_count = 0
        self.fps_time = time.time()
        self.frame_timestamp = 0

        # Screenshot save directory
        self.screenshot_dir = os.path.join(
            os.path.expanduser("~"), "Desktop", "Camera Mouse Screenshots"
        )

    def count_fingers(self, landmarks, handedness_label):
        """Count extended fingers from hand landmarks."""
        count = 0

        # Thumb: compare tip (4) x to IP (3) x
        # handedness_label is "Left" or "Right" as detected (before mirror)
        if handedness_label == "Right":
            if landmarks[4].x < landmarks[3].x:
                count += 1
        else:
            if landmarks[4].x > landmarks[3].x:
                count += 1

        # Index: tip (8) above PIP (6) → lower y = higher
        if landmarks[8].y < landmarks[6].y:
            count += 1

        # Middle: tip (12) above PIP (10)
        if landmarks[12].y < landmarks[10].y:
            count += 1

        # Ring: tip (16) above PIP (14)
        if landmarks[16].y < landmarks[14].y:
            count += 1

        # Pinky: tip (20) above PIP (18)
        if landmarks[20].y < landmarks[18].y:
            count += 1

        return count

    def get_mode(self, fingers, now):
        """Determine gesture mode from finger count with stabilization."""
        if fingers == 1:
            new_mode = "move"
        elif fingers == 2:
            new_mode = "click"
        elif fingers == 3:
            new_mode = "drag"
        elif fingers == 4:
            new_mode = "screenshot"
        elif fingers >= 5:
            new_mode = "paused"
        else:
            new_mode = "idle"

        # Stabilization
        if new_mode != self.prev_mode:
            self.prev_mode = new_mode
            self.mode_change_time = now
            return self.stable_mode

        if now - self.mode_change_time >= MODE_SWITCH_DELAY:
            self.stable_mode = new_mode

        return self.stable_mode

    def take_screenshot(self):
        """Take a screenshot and save to Desktop."""
        os.makedirs(self.screenshot_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(self.screenshot_dir, filename)

        screenshot = pyautogui.screenshot()
        screenshot.save(filepath)
        self.screenshot_count += 1
        print(f"  Screenshot saved: {filepath}")
        return filepath

    def draw_hand(self, frame, landmarks_list, mode):
        """Draw hand landmarks on the frame."""
        h, w = frame.shape[:2]

        mode_colors = {
            "move": COLOR_MOVE,
            "click": COLOR_CLICK,
            "drag": COLOR_DRAG,
            "screenshot": COLOR_SCREENSHOT,
            "paused": COLOR_PAUSED,
        }
        color = mode_colors.get(mode, (200, 200, 200))

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        ]

        # Convert landmarks to pixel coords (mirrored)
        points = []
        for lm in landmarks_list:
            px = int((1 - lm.x) * w)  # Mirror x
            py = int(lm.y * h)
            points.append((px, py))

        # Draw connections
        conn_color = tuple(c // 2 for c in color)
        for start, end in connections:
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], conn_color, 2, cv2.LINE_AA)

        # Draw points
        fingertips = [4, 8, 12, 16, 20]
        for i, pt in enumerate(points):
            if i in fingertips:
                cv2.circle(frame, pt, 6, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 8, color, 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 3, COLOR_WHITE, -1, cv2.LINE_AA)

        # Extra highlight on index fingertip
        if len(points) > 8:
            cv2.circle(frame, points[8], 10, COLOR_GREEN, 2, cv2.LINE_AA)

    def draw_ui(self, frame, mode, finger_count, hand_found):
        """Draw the HUD overlay on the camera frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        mode_colors = {
            "move": COLOR_MOVE, "click": COLOR_CLICK, "drag": COLOR_DRAG,
            "screenshot": COLOR_SCREENSHOT, "paused": COLOR_PAUSED, "idle": COLOR_IDLE,
        }
        mode_labels = {
            "move": "MOVE CURSOR", "click": "CLICK", "drag": "DRAG",
            "screenshot": "SCREENSHOT", "paused": "PAUSED", "idle": "WAITING...",
        }
        mode_icons = {
            "move": "1 Finger", "click": "2 Fingers", "drag": "3 Fingers",
            "screenshot": "4 Fingers", "paused": "5 Fingers", "idle": "No Hand",
        }

        color = mode_colors.get(mode, COLOR_IDLE)

        # ── Top bar ──
        cv2.rectangle(overlay, (0, 0), (w, 54), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        cv2.putText(frame, "CAMERA MOUSE", (14, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)

        cv2.putText(frame, f"{self.fps} FPS", (w - 80, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

        label = mode_labels.get(mode, "IDLE")
        icon = mode_icons.get(mode, "")
        cv2.putText(frame, label, (14, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"  {icon}", (14 + len(label) * 14, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        # Hand status dot
        dot_color = COLOR_GREEN if hand_found else COLOR_RED
        cv2.circle(frame, (w - 24, 42), 6, dot_color, -1)
        cv2.circle(frame, (w - 24, 42), 8, dot_color, 1)

        # ── Bottom bar ──
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 88), (w, h), COLOR_BG, -1)
        cv2.addWeighted(overlay2, 0.85, frame, 0.15, 0, frame)

        stats_y = h - 62
        cv2.putText(frame, f"Clicks: {self.click_count}", (14, stats_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Drags: {self.drag_count}", (140, stats_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Screenshots: {self.screenshot_count}", (260, stats_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

        fingers_y = h - 30
        cv2.putText(frame, f"Fingers: {finger_count}", (14, fingers_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        cx, cy = int(self.smooth_x), int(self.smooth_y)
        cv2.putText(frame, f"Cursor: ({cx}, {cy})", (180, fingers_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

        # Gesture guide (right side)
        guide_x = w - 200
        guide_entries = [
            ("1F: Move", COLOR_MOVE, mode == "move"),
            ("2F: Click", COLOR_CLICK, mode == "click"),
            ("3F: Drag", COLOR_DRAG, mode == "drag"),
            ("4F: Screenshot", COLOR_SCREENSHOT, mode == "screenshot"),
            ("5F: Pause", COLOR_PAUSED, mode == "paused"),
        ]
        for i, (text, col, active) in enumerate(guide_entries):
            y = h - 80 + i * 16
            thickness = 2 if active else 1
            alpha = col if active else tuple(c // 3 for c in col)
            cv2.putText(frame, text, (guide_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, alpha, thickness, cv2.LINE_AA)
            if active:
                cv2.circle(frame, (guide_x - 8, y - 4), 3, col, -1)

        # Paused overlay
        if mode == "paused":
            overlay3 = frame.copy()
            cv2.rectangle(overlay3, (0, 0), (w, h), (10, 10, 15), -1)
            cv2.addWeighted(overlay3, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, "PAUSED", (w // 2 - 80, h // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_PAUSED, 3, cv2.LINE_AA)
            cv2.putText(frame, "Show fewer fingers to resume", (w // 2 - 140, h // 2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)
        elif mode == "drag" and self.is_dragging:
            cv2.putText(frame, "DRAGGING...", (w // 2 - 60, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_DRAG, 2, cv2.LINE_AA)

        # Mode color bar at top
        cv2.rectangle(frame, (0, 0), (w, 3), color, -1)

        return frame

    def run(self):
        """Main application loop."""
        print("\n" + "=" * 56)
        print("  CAMERA MOUSE - Desktop Controller")
        print("=" * 56)
        print(f"  Screen: {self.screen_w} x {self.screen_h}")
        print("  Press 'Q' to quit | 'R' to reset cursor")
        print("=" * 56)
        print("\n  Starting camera...\n")

        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not cap.isOpened():
            print("  ERROR: Could not open camera!")
            print("  Make sure no other app is using the camera.")
            input("  Press Enter to exit...")
            return

        print("  Camera started! Show your hand to begin.\n")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 480, 360)

        # Try to keep window always on top (Windows)
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
                    print("  ERROR: Failed to read from camera.")
                    break

                # Mirror the frame for display
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                now = time.time()

                # FPS
                self.frame_count += 1
                if now - self.fps_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.fps_time = now

                # Convert to RGB for MediaPipe (need un-mirrored for detection)
                detect_frame = cv2.flip(frame, 1)  # Un-mirror for detection
                rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)

                # Create MediaPipe Image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # Detect hands
                self.frame_timestamp += 33  # ~30fps timestamps in ms
                results = self.hand_landmarker.detect_for_video(mp_image, self.frame_timestamp)

                hand_found = False
                current_mode = "idle"

                if results.hand_landmarks and len(results.hand_landmarks) > 0:
                    hand_found = True
                    landmarks = results.hand_landmarks[0]  # List of NormalizedLandmark

                    # Get handedness
                    hand_label = "Right"
                    if results.handedness and len(results.handedness) > 0:
                        hand_label = results.handedness[0][0].category_name

                    # Count fingers
                    fingers = self.count_fingers(landmarks, hand_label)
                    self.finger_count = fingers

                    # Get stable mode
                    current_mode = self.get_mode(fingers, now)

                    # Get index finger tip position (landmark 8)
                    ix = landmarks[8].x  # 0-1 (un-mirrored)
                    iy = landmarks[8].y  # 0-1

                    # ═══ MODE: MOVE (1 finger) ═══
                    if current_mode == "move":
                        if self.is_dragging:
                            pyautogui.mouseUp()
                            self.is_dragging = False
                            print("  Drag released")

                        self.is_paused = False

                        # Map hand position to screen (mirror x since camera is mirrored)
                        mapped_x = (1.0 - ix - BOUNDARY_PAD) / (1 - 2 * BOUNDARY_PAD)
                        mapped_y = (iy - BOUNDARY_PAD) / (1 - 2 * BOUNDARY_PAD)

                        mapped_x = 0.5 + (mapped_x - 0.5) * SENSITIVITY_X
                        mapped_y = 0.5 + (mapped_y - 0.5) * SENSITIVITY_Y

                        target_x = max(0, min(self.screen_w - 1, mapped_x * self.screen_w))
                        target_y = max(0, min(self.screen_h - 1, mapped_y * self.screen_h))

                        self.smooth_x += (target_x - self.smooth_x) * SMOOTHING
                        self.smooth_y += (target_y - self.smooth_y) * SMOOTHING

                        pyautogui.moveTo(int(self.smooth_x), int(self.smooth_y), _pause=False)

                    # ═══ MODE: CLICK (2 fingers) ═══
                    elif current_mode == "click":
                        if self.is_dragging:
                            pyautogui.mouseUp()
                            self.is_dragging = False

                        self.is_paused = False

                        if now - self.last_click_time > CLICK_COOLDOWN:
                            self.last_click_time = now
                            self.click_count += 1
                            pyautogui.click(_pause=False)
                            print(f"  Click #{self.click_count} at ({int(self.smooth_x)}, {int(self.smooth_y)})")

                    # ═══ MODE: DRAG (3 fingers) ═══
                    elif current_mode == "drag":
                        self.is_paused = False

                        if not self.is_dragging:
                            self.is_dragging = True
                            self.drag_count += 1
                            pyautogui.mouseDown(_pause=False)
                            print(f"  Drag #{self.drag_count} started")

                        mapped_x = (1.0 - ix - BOUNDARY_PAD) / (1 - 2 * BOUNDARY_PAD)
                        mapped_y = (iy - BOUNDARY_PAD) / (1 - 2 * BOUNDARY_PAD)
                        mapped_x = 0.5 + (mapped_x - 0.5) * SENSITIVITY_X
                        mapped_y = 0.5 + (mapped_y - 0.5) * SENSITIVITY_Y

                        target_x = max(0, min(self.screen_w - 1, mapped_x * self.screen_w))
                        target_y = max(0, min(self.screen_h - 1, mapped_y * self.screen_h))

                        self.smooth_x += (target_x - self.smooth_x) * SMOOTHING
                        self.smooth_y += (target_y - self.smooth_y) * SMOOTHING

                        pyautogui.moveTo(int(self.smooth_x), int(self.smooth_y), _pause=False)

                    # ═══ MODE: SCREENSHOT (4 fingers) ═══
                    elif current_mode == "screenshot":
                        if self.is_dragging:
                            pyautogui.mouseUp()
                            self.is_dragging = False

                        self.is_paused = False

                        if now - self.last_screenshot_time > SCREENSHOT_COOLDOWN:
                            self.last_screenshot_time = now
                            self.take_screenshot()

                    # ═══ MODE: PAUSED (5 fingers) ═══
                    elif current_mode == "paused":
                        if self.is_dragging:
                            pyautogui.mouseUp()
                            self.is_dragging = False

                        if not self.is_paused:
                            self.is_paused = True
                            print("  Tracking paused (show fewer fingers to resume)")

                    # Draw hand landmarks on the mirrored display frame
                    self.draw_hand(frame, landmarks, current_mode)

                else:
                    self.finger_count = 0
                    self.stable_mode = "idle"
                    self.prev_mode = "idle"

                    if self.is_dragging:
                        pyautogui.mouseUp()
                        self.is_dragging = False
                        print("  Drag released (hand lost)")

                # Draw UI overlay
                frame = self.draw_ui(frame, current_mode, self.finger_count, hand_found)

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
            print("\n  Interrupted by user.")
        finally:
            if self.is_dragging:
                pyautogui.mouseUp()
            cap.release()
            cv2.destroyAllWindows()
            self.hand_landmarker.close()

            print("\n" + "=" * 56)
            print("  Session Summary")
            print("=" * 56)
            print(f"  Clicks:      {self.click_count}")
            print(f"  Drags:       {self.drag_count}")
            print(f"  Screenshots: {self.screenshot_count}")
            if self.screenshot_count > 0:
                print(f"  Saved to:    {self.screenshot_dir}")
            print("=" * 56)
            print("  Goodbye!\n")


if __name__ == "__main__":
    app = CameraMouse()
    app.run()
