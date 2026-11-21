#!/usr/bin/env python3
"""AirDesk: face-locked air-gesture control for macOS.

Camera I/O, the face lock (FaceGate), macOS actions, and the main loop.
Gesture classification itself lives in gestures.py (pure, unit tested).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# mediapipe imports matplotlib; keep its cache inside the project folder.
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".cache" / "matplotlib"))
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gestures as g  # noqa: E402

WINDOW = "AirDesk"
DEFAULT_CONFIG_PATH = HERE / "config.json"
DEFAULT_CONFIG = {
    "face_photo": "",
    "camera_index": 0,
    "frame_width": 960,
    "frame_height": 540,
    "face_check_every_n_frames": 10,
    "face_grace_seconds": 4.0,
    "face_tolerance": 0.5,
    "face_scale": 0.5,
    "hand_model_complexity": 1,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.6,
    "cursor_smoothing": 0.35,
    "cursor_margin": 0.15,
    "cooldown_seconds": 0.9,
    "hold_seconds": {"pinch": 0.08, "fist": 0.5, "open_palm": 0.5},
    "swipe_distance": 0.2,
    "swipe_window_seconds": 0.4,
    "swipe_return_block_seconds": 1.2,
    "swap_swipe_directions": False,
    "preview_width": 640,
    "tuning": {},
}

# gesture -> action name; swipes are keyed by (pose, direction)
POSE_ACTIONS = {
    g.PINCH_INDEX: "left_click",
    g.PINCH_MIDDLE: "right_click",
    g.FIST: "close_window",
    g.PALM: "minimize_window",
}
SWIPE_ACTIONS = {
    (g.TWO, "left"): "next_app",
    (g.TWO, "right"): "prev_app",
    (g.THREE, "left"): "space_right",
    (g.THREE, "right"): "space_left",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_config(path: Path) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path.is_file():
        with open(path) as fh:
            user = json.load(fh)
        for key, value in user.items():
            if key.startswith("_"):
                continue
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg


# --------------------------------------------------------------------------- face lock


class FaceLockError(RuntimeError):
    pass


class FaceGate:
    """Unlocks gestures only while the owner's face is in view.

    Every N frames a downscaled copy of the frame is handed to a worker thread
    that runs face_recognition against the reference encoding. A match opens a
    grace window; seeing only other faces closes it immediately.
    """

    def __init__(self, photo_path: str, check_every: int = 10, grace_seconds: float = 4.0,
                 tolerance: float = 0.5, scale: float = 0.5):
        import cv2
        import face_recognition
        import numpy as np

        self._cv2, self._fr, self._np = cv2, face_recognition, np
        if not photo_path:
            raise FaceLockError(
                "No face photo configured. Set \"face_photo\" in config.json or pass --face-photo PATH.")
        path = Path(photo_path).expanduser()
        if not path.is_file():
            raise FaceLockError(f"Face photo not found: {path}")
        try:
            image = face_recognition.load_image_file(str(path))
        except Exception as exc:  # unreadable / unsupported format (e.g. HEIC)
            raise FaceLockError(f"Could not read face photo {path}: {exc}. Use a JPEG or PNG.") from exc
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest > 1000:
            f = 1000 / longest
            image = cv2.resize(image, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        locations = face_recognition.face_locations(image, number_of_times_to_upsample=1, model="hog")
        if not locations:
            raise FaceLockError(
                f"No face found in {path}. Use a clear, well-lit, front-facing photo of your face.")
        largest = max(locations, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))
        self.reference = face_recognition.face_encodings(image, [largest])[0]
        self.photo_path = path
        self.faces_in_photo = len(locations)

        self.check_every = max(1, int(check_every))
        self.tolerance = tolerance
        self.scale = scale
        self.timer = g.GraceTimer(grace_seconds)
        self.status = "waiting for first check"
        self.last_distance: float | None = None
        self.checks = 0
        self.last_check_ms = 0.0

        self._slot = None
        self._cond = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._worker, name="FaceGate", daemon=True)
        self._thread.start()

    def submit(self, frame_bgr, frame_index: int) -> None:
        if frame_index % self.check_every:
            return
        small = self._cv2.resize(frame_bgr, None, fx=self.scale, fy=self.scale, interpolation=self._cv2.INTER_AREA)
        rgb = self._np.ascontiguousarray(small[:, :, ::-1])
        with self._cond:
            self._slot = rgb
            self._cond.notify()

    def check(self, rgb, now: float | None = None) -> str:
        """Run one recognition pass on an RGB image and update the lock state."""
        t0 = time.perf_counter()
        now = time.monotonic() if now is None else now
        locations = self._fr.face_locations(rgb, number_of_times_to_upsample=1, model="hog")
        if not locations:
            self.status, self.last_distance = "no face", None
        else:
            encodings = self._fr.face_encodings(rgb, locations)
            best = float(min(self._fr.face_distance(encodings, self.reference)))
            self.last_distance = best
            if best <= self.tolerance:
                self.timer.match(now)
                self.status = f"owner {best:.2f}"
            else:
                self.timer.revoke()
                self.status = f"stranger {best:.2f}"
        self.checks += 1
        self.last_check_ms = (time.perf_counter() - t0) * 1000
        return self.status

    def _worker(self) -> None:
        while True:
            with self._cond:
                while self._slot is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                rgb, self._slot = self._slot, None
            try:
                self.check(rgb)
            except Exception as exc:  # never let a bad frame kill the lock thread
                self.timer.revoke()
                self.status = f"error: {exc}"

    def unlocked(self, now: float) -> bool:
        return self.timer.is_open(now)

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._thread.join(timeout=2)


# --------------------------------------------------------------------------- macOS actions


class Actions:
    """Clicks via pyautogui; key combos via Quartz with explicit modifier flags,
    which macOS needs for Cmd+Tab and the Mission Control Ctrl+Arrow shortcuts."""

    KEYCODES = {"tab": 48, "left": 123, "right": 124, "w": 13, "m": 46}
    MODIFIERS = {"cmd": (55, 0x100000), "shift": (56, 0x20000), "ctrl": (59, 0x40000)}
    ARROW_FLAGS = 0x800000 | 0x200000  # SecondaryFn | NumericPad, as a real arrow key sends

    DESCRIPTIONS = {
        "left_click": "left click",
        "right_click": "right click",
        "next_app": "Cmd+Tab (switch app)",
        "prev_app": "Cmd+Shift+Tab (switch app back)",
        "space_right": "Ctrl+Right (next desktop space)",
        "space_left": "Ctrl+Left (previous desktop space)",
        "close_window": "Cmd+W (close window)",
        "minimize_window": "Cmd+M (minimize window)",
    }

    def __init__(self, dry_run: bool):
        import pyautogui

        pyautogui.FAIL_SAFE = True
        pyautogui.PAUSE = 0  # default 0.1 s after every call would cap the loop at 10 fps
        self.pg = pyautogui
        self.dry_run = dry_run
        self.screen = tuple(pyautogui.size())
        self._quartz = None
        if not dry_run:
            import Quartz

            self._quartz = Quartz
            self._src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
        self._last_move_log = 0.0
        self._last_move = None

    def fail_safe_check(self) -> None:
        self.pg.failSafeCheck()

    def move(self, x: int, y: int) -> None:
        if self.dry_run:
            now = time.monotonic()
            if (x, y) != self._last_move and now - self._last_move_log > 0.5:
                log(f"[dry-run] move cursor -> ({x}, {y})")
                self._last_move_log = now
                self._last_move = (x, y)
            return
        self.pg.moveTo(x, y, _pause=False)

    def perform(self, name: str) -> str:
        desc = self.DESCRIPTIONS[name]
        if self.dry_run:
            log(f"[dry-run] {name}: {desc}")
            return desc
        log(f"action {name}: {desc}")
        if name == "left_click":
            self.pg.click(_pause=False)
        elif name == "right_click":
            self.pg.rightClick(_pause=False)
        elif name == "next_app":
            self._combo("tab", ("cmd",))
        elif name == "prev_app":
            self._combo("tab", ("cmd", "shift"))
        elif name == "space_right":
            self._combo("right", ("ctrl",))
        elif name == "space_left":
            self._combo("left", ("ctrl",))
        elif name == "close_window":
            self._combo("w", ("cmd",))
        elif name == "minimize_window":
            self._combo("m", ("cmd",))
        return desc

    def _post(self, code: int, down: bool, flags: int) -> None:
        q = self._quartz
        ev = q.CGEventCreateKeyboardEvent(self._src, code, down)
        q.CGEventSetFlags(ev, flags)
        q.CGEventPost(q.kCGHIDEventTap, ev)
        time.sleep(0.02)

    def _combo(self, key: str, mods: tuple) -> None:
        flags = 0
        for m in mods:
            code, mask = self.MODIFIERS[m]
            flags |= mask
            self._post(code, True, flags)
        key_flags = flags | (self.ARROW_FLAGS if key in ("left", "right") else 0)
        code = self.KEYCODES[key]
        self._post(code, True, key_flags)
        self._post(code, False, key_flags)
        for m in reversed(mods):
            code, mask = self.MODIFIERS[m]
            flags &= ~mask
            self._post(code, False, flags)


# --------------------------------------------------------------------------- camera


CAMERA_DENIED_HELP = (
    "Camera access is off for the app you launched AirDesk from (e.g. Terminal).\n"
    "  1. Open System Settings > Privacy & Security > Camera\n"
    "     (shortcut: open 'x-apple.systempreferences:com.apple.preference.security?Privacy_Camera').\n"
    "  2. Turn on your terminal app.\n"
    "  3. Quit the terminal app completely (Cmd+Q), reopen it, and run AirDesk again."
)

AV_NOT_DETERMINED, AV_RESTRICTED, AV_DENIED, AV_AUTHORIZED = 0, 1, 2, 3


def ensure_camera_permission(timeout: float = 120.0) -> bool:
    """Ask macOS for camera access and wait for the user's answer.

    OpenCV only fires the permission request and fails straight away; the
    process then exits, which cancels the prompt, so the request never sticks.
    """
    try:
        import AVFoundation as AVF
        from Foundation import NSDate, NSRunLoop
    except ImportError:
        return True  # no bindings: let OpenCV try on its own

    media = AVF.AVMediaTypeVideo

    def status() -> int:
        return int(AVF.AVCaptureDevice.authorizationStatusForMediaType_(media))

    current = status()
    if current == AV_AUTHORIZED:
        return True
    if current == AV_NOT_DETERMINED:
        log("macOS is asking for camera access for your terminal app. Click OK/Allow in the dialog "
            f"(waiting up to {int(timeout)} s)...")
        answer = {}
        done = threading.Event()

        def handler(granted):
            answer["granted"] = bool(granted)
            done.set()

        AVF.AVCaptureDevice.requestAccessForMediaType_completionHandler_(media, handler)
        deadline = time.monotonic() + timeout
        while not done.is_set() and status() == AV_NOT_DETERMINED and time.monotonic() < deadline:
            NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.25))
        current = status()
        if answer.get("granted") or current == AV_AUTHORIZED:
            log("camera access granted")
            return True
        if current == AV_NOT_DETERMINED:
            print("No answer to the camera prompt. Run AirDesk again and click Allow, or:", file=sys.stderr)
            print(CAMERA_DENIED_HELP, file=sys.stderr)
            return False
    if current == AV_RESTRICTED:
        print("Camera access is restricted on this Mac (Screen Time or a device profile).", file=sys.stderr)
        return False
    print(CAMERA_DENIED_HELP, file=sys.stderr)
    return False


def open_camera(index: int, width: int, height: int):
    import cv2

    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(30):  # the first frames can be empty while the camera warms up
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
        time.sleep(0.05)
    cap.release()
    return None


class FileSource:
    """Stand-in for the webcam (testing): loops a video file or repeats a still image at ~30 fps."""

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, path: str, width: int):
        import cv2

        self._cv2 = cv2
        self.path = Path(path).expanduser()
        self._image = None
        self._cap = None
        if self.path.suffix.lower() in self.IMAGE_EXTS:
            img = cv2.imread(str(self.path))
            if img is None:
                raise FileNotFoundError(f"cannot read image {self.path}")
            f = width / img.shape[1]
            self._image = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        else:
            self._cap = cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise FileNotFoundError(f"cannot open video {self.path}")
        self._next = time.monotonic()

    def read(self):
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next = max(self._next, time.monotonic()) + 1 / 30
        if self._image is not None:
            return True, self._image.copy()
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return ok, frame

    def release(self):
        if self._cap is not None:
            self._cap.release()


CAMERA_HELP = (
    "Could not read from camera {index}. Check that:\n"
    "  1. The app you run this from (Terminal, iTerm, VS Code, ...) has Camera access:\n"
    "     System Settings > Privacy & Security > Camera.\n"
    "  2. No other app is holding the camera.\n"
    "  3. The index is right: run `python airdesk.py --list-cameras` and pass --camera N."
)


def list_cameras(max_index: int = 4) -> None:
    import cv2

    if not ensure_camera_permission():
        return

    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        ok, frame = (cap.read() if cap.isOpened() else (False, None))
        if ok and frame is not None:
            print(f"camera {i}: OK {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"camera {i}: not available")
        cap.release()


# --------------------------------------------------------------------------- preview


def draw_preview(cv2, frame, lm_proto, mp_draw, mp_hands, status: str, ok: bool, info: list[str],
                 flash: str | None, width: int):
    if lm_proto is not None:
        mp_draw.draw_landmarks(frame, lm_proto, mp_hands.HAND_CONNECTIONS)
    h, w = frame.shape[:2]
    if width and w != width:
        frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
    color = (0, 200, 0) if ok else (0, 0, 255)
    cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    y = h - 10
    for line in reversed(info):
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y -= 18
    if flash:
        cv2.putText(frame, flash, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------------- main


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AirDesk: face-locked air-gesture control for macOS.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to config.json")
    p.add_argument("--face-photo", help="reference photo of your face (overrides config.json)")
    p.add_argument("--camera", type=int, help="camera index (overrides config.json)")
    p.add_argument("--dry-run", action="store_true", help="print actions instead of performing them")
    p.add_argument("--no-face-lock", action="store_true", help="TESTING ONLY: accept gestures from anyone")
    p.add_argument("--list-cameras", action="store_true", help="probe camera indices and exit")
    p.add_argument("--check-face", action="store_true",
                   help="load the face photo, then report live match distances for ~5 s and exit")
    p.add_argument("--max-seconds", type=float, default=0, help="quit automatically after N seconds")
    p.add_argument("--save-preview", help="write the last preview frame to this image file on exit")
    p.add_argument("--video", help="TESTING: read frames from a video or image file instead of the webcam")
    return p.parse_args(argv)


def build_gate(cfg: dict) -> FaceGate:
    gate = FaceGate(
        cfg["face_photo"],
        check_every=cfg["face_check_every_n_frames"],
        grace_seconds=cfg["face_grace_seconds"],
        tolerance=cfg["face_tolerance"],
        scale=cfg["face_scale"],
    )
    log(f"face lock: loaded {gate.photo_path} ({gate.faces_in_photo} face(s) found, using the largest)")
    return gate


def check_face(cfg: dict) -> int:
    try:
        gate = build_gate(cfg)
    except FaceLockError as exc:
        print(f"FACE LOCK ERROR: {exc}", file=sys.stderr)
        return 2
    if not ensure_camera_permission():
        gate.close()
        return 3
    cap = open_camera(cfg["camera_index"], cfg["frame_width"], cfg["frame_height"])
    if cap is None:
        print(CAMERA_HELP.format(index=cfg["camera_index"]), file=sys.stderr)
        gate.close()
        return 3
    import cv2
    import numpy as np

    log(f"tolerance {gate.tolerance} (lower distance = more similar); look at the camera...")
    end = time.monotonic() + 5
    while time.monotonic() < end:
        ok, frame = cap.read()
        if not ok:
            continue
        small = cv2.resize(frame, None, fx=gate.scale, fy=gate.scale, interpolation=cv2.INTER_AREA)
        status = gate.check(np.ascontiguousarray(small[:, :, ::-1]))
        log(f"  {status:<16} ({gate.last_check_ms:.0f} ms)")
    cap.release()
    gate.close()
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(Path(args.config).expanduser())
    if args.face_photo:
        cfg["face_photo"] = args.face_photo
    if args.camera is not None:
        cfg["camera_index"] = args.camera

    if args.list_cameras:
        list_cameras()
        return 0
    if args.check_face:
        return check_face(cfg)

    gate = None
    if args.no_face_lock:
        log("WARNING: face lock disabled (--no-face-lock). Anyone in view can control this Mac.")
    else:
        try:
            gate = build_gate(cfg)
        except FaceLockError as exc:
            print(f"FACE LOCK ERROR: {exc}", file=sys.stderr)
            return 2

    import cv2
    import mediapipe as mp
    import pyautogui

    if args.video:
        cap = FileSource(args.video, cfg["frame_width"])
        log(f"TEST SOURCE: reading frames from {cap.path} instead of the webcam")
    elif not ensure_camera_permission():
        if gate:
            gate.close()
        return 3
    else:
        cap = open_camera(cfg["camera_index"], cfg["frame_width"], cfg["frame_height"])
    if cap is None:
        print(CAMERA_HELP.format(index=cfg["camera_index"]), file=sys.stderr)
        if gate:
            gate.close()
        return 3

    actions = Actions(dry_run=args.dry_run)
    tuning = g.Tuning(**cfg["tuning"])
    hold = cfg["hold_seconds"]
    rules = {
        g.PINCH_INDEX: g.HoldRule(3, hold["pinch"]),
        g.PINCH_MIDDLE: g.HoldRule(3, hold["pinch"]),
        g.FIST: g.HoldRule(5, hold["fist"]),
        g.PALM: g.HoldRule(5, hold["open_palm"]),
    }
    stabilizer = g.Stabilizer(rules, cooldown=cfg["cooldown_seconds"])
    swipes = g.SwipeTracker(
        min_distance=cfg["swipe_distance"],
        window=cfg["swipe_window_seconds"],
        cooldown=cfg["cooldown_seconds"],
        return_block=cfg["swipe_return_block_seconds"],
    )
    cursor = g.CursorMapper(*actions.screen, alpha=cfg["cursor_smoothing"], margin=cfg["cursor_margin"])
    swipe_actions = dict(SWIPE_ACTIONS)
    if cfg["swap_swipe_directions"]:
        swipe_actions = {(p, "right" if d == "left" else "left"): a for (p, d), a in SWIPE_ACTIONS.items()}

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=cfg["hand_model_complexity"],
        min_detection_confidence=cfg["min_detection_confidence"],
        min_tracking_confidence=cfg["min_tracking_confidence"],
    )

    mode = "DRY RUN" if args.dry_run else "LIVE"
    log(f"AirDesk running ({mode}), camera {cfg['camera_index']}, screen {actions.screen[0]}x{actions.screen[1]}. "
        "Press q in the preview window to quit.")

    started = time.monotonic()
    frame_index = 0
    fps, fps_t, fps_n = 0.0, started, 0
    flash, flash_until = None, 0.0
    preview = None
    was_unlocked = None
    exit_code = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                if time.monotonic() - started > 5 and frame_index == 0:
                    print(CAMERA_HELP.format(index=cfg["camera_index"]), file=sys.stderr)
                    exit_code = 3
                    break
                continue
            now = time.monotonic()
            frame_index += 1
            if gate:
                gate.submit(frame, frame_index)  # unmirrored, to match the reference photo
            unlocked = gate is None or gate.unlocked(now)
            if gate and unlocked != was_unlocked:
                log("face lock: UNLOCKED" if unlocked else "face lock: LOCKED")
                was_unlocked = unlocked

            mirrored = cv2.flip(frame, 1)
            h, w = mirrored.shape[:2]
            result = hands.process(cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB))
            lm_proto, lm, label = None, None, g.NONE
            if result.multi_hand_landmarks:
                lm_proto = result.multi_hand_landmarks[0]
                lm = [(p.x, p.y, p.z) for p in lm_proto.landmark]
                label = g.classify(lm, tuning, aspect=w / h)

            if not unlocked:
                stabilizer.reset()
                swipes.reset()
                cursor.reset()
            else:
                if label == g.POINT and not g.near_pinch(lm, tuning, aspect=w / h):
                    tip = lm[g.INDEX_TIP]
                    actions.move(*cursor.update(tip[0], tip[1]))
                fired = stabilizer.update(label, now)
                if fired in POSE_ACTIONS:
                    flash, flash_until = actions.perform(POSE_ACTIONS[fired]), now + 1.2
                sw = swipes.update(label, g.palm_x(lm) if lm else None, now)
                if sw in swipe_actions:
                    flash, flash_until = actions.perform(swipe_actions[sw]), now + 1.2
            actions.fail_safe_check()

            fps_n += 1
            if now - fps_t >= 1.0:
                fps, fps_t, fps_n = fps_n / (now - fps_t), now, 0
            if not unlocked:
                status, status_ok = "LOCKED", False
            else:
                status, status_ok = g.DISPLAY_NAMES[label], True
            info = [f"{mode}  {fps:4.1f} fps  hand: {label}"]
            info.append("face lock: OFF (testing)" if gate is None else f"face: {gate.status}")
            preview = draw_preview(cv2, mirrored, lm_proto, mp_draw, mp_hands, status, status_ok, info,
                                   flash if now < flash_until else None, cfg["preview_width"])
            cv2.imshow(WINDOW, preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if args.max_seconds and now - started >= args.max_seconds:
                break
    except pyautogui.FailSafeException:
        log("FAIL-SAFE: cursor reached a screen corner. AirDesk stopped.")
        exit_code = 1
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.monotonic() - started
        log(f"stopped after {frame_index} frames in {elapsed:.1f} s ({frame_index / max(elapsed, 1e-6):.1f} fps avg)")
        if args.save_preview and preview is not None:
            cv2.imwrite(str(Path(args.save_preview).expanduser()), preview)
            log(f"saved preview frame to {args.save_preview}")
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
        if gate:
            gate.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
