"""Pure gesture logic for AirDesk.

No cv2 / mediapipe / pyautogui imports here, so everything in this module can be
unit tested with synthetic landmarks and fake timestamps.

Landmarks follow the MediaPipe Hands layout: 21 points, each (x, y) or (x, y, z)
in normalized image coordinates (0..1, y grows downward).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
}

NONE = "none"
POINT = "point"
PINCH_INDEX = "pinch_index"
PINCH_MIDDLE = "pinch_middle"
TWO = "two_fingers"
THREE = "three_fingers"
FIST = "fist"
PALM = "open_palm"

LABELS = (NONE, POINT, PINCH_INDEX, PINCH_MIDDLE, TWO, THREE, FIST, PALM)

DISPLAY_NAMES = {
    NONE: "no gesture",
    POINT: "POINT - move cursor",
    PINCH_INDEX: "PINCH - left click",
    PINCH_MIDDLE: "MIDDLE PINCH - right click",
    TWO: "TWO FINGERS - swipe to switch app",
    THREE: "THREE FINGERS - swipe to switch space",
    FIST: "FIST - close window",
    PALM: "OPEN PALM - minimize",
}


@dataclass
class Tuning:
    """Classification knobs. All distances are in units of hand size
    (wrist to middle-finger knuckle), so they work near or far from the camera."""

    # A finger is extended when wrist->tip is this many times wrist->PIP.
    extended_ratio: float = 1.12
    # Thumb tip to finger tip distance (hand sizes) that counts as a pinch.
    pinch_ratio: float = 0.30
    # In a fist the fingertips and the tucked thumb sit on the palm. A pinch only
    # counts when both pinching tips are at least this far from the palm center.
    pinch_min_reach: float = 0.45
    # While pointing, freeze the cursor once thumb and index are this close
    # (multiple of pinch_ratio), so starting a click does not drag the cursor.
    pre_pinch_freeze: float = 1.6
    # Hands smaller than this (normalized frame units) are ignored as noise.
    min_hand_size: float = 0.03


DEFAULT_TUNING = Tuning()


def _xy(p, aspect: float) -> tuple[float, float]:
    return (p[0] * aspect, p[1])


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _points(landmarks: Sequence, aspect: float) -> list[tuple[float, float]]:
    if len(landmarks) != 21:
        raise ValueError(f"expected 21 landmarks, got {len(landmarks)}")
    return [_xy(p, aspect) for p in landmarks]


def hand_size(landmarks: Sequence, aspect: float = 1.0) -> float:
    pts = _points(landmarks, aspect)
    return _dist(pts[WRIST], pts[MIDDLE_MCP])


def palm_center(landmarks: Sequence, aspect: float = 1.0) -> tuple[float, float]:
    """Palm center in the same (aspect-corrected) space used for distances."""
    pts = _points(landmarks, aspect)
    ids = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
    return (sum(pts[i][0] for i in ids) / len(ids), sum(pts[i][1] for i in ids) / len(ids))


def palm_x(landmarks: Sequence) -> float:
    """Horizontal palm position as a fraction of frame width (for swipes)."""
    ids = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
    return sum(landmarks[i][0] for i in ids) / len(ids)


@dataclass
class HandState:
    size: float
    extended: dict
    thumb_out: bool
    pinch_index: float  # thumb-index tip distance in hand sizes
    pinch_middle: float
    reach_index: float  # min distance of the pinching tips from palm center
    reach_middle: float


def analyze(landmarks: Sequence, tuning: Tuning = DEFAULT_TUNING, aspect: float = 1.0) -> HandState:
    pts = _points(landmarks, aspect)
    size = _dist(pts[WRIST], pts[MIDDLE_MCP])
    if size <= 1e-9:
        size = 1e-9
    wrist = pts[WRIST]
    extended = {
        name: _dist(wrist, pts[tip]) > tuning.extended_ratio * _dist(wrist, pts[pip])
        for name, (_mcp, pip, tip) in FINGERS.items()
    }
    thumb_out = _dist(pts[THUMB_TIP], pts[PINKY_MCP]) > _dist(pts[THUMB_IP], pts[PINKY_MCP])
    ids = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
    pc = (sum(pts[i][0] for i in ids) / 5, sum(pts[i][1] for i in ids) / 5)
    thumb_reach = _dist(pts[THUMB_TIP], pc) / size
    return HandState(
        size=size,
        extended=extended,
        thumb_out=thumb_out,
        pinch_index=_dist(pts[THUMB_TIP], pts[INDEX_TIP]) / size,
        pinch_middle=_dist(pts[THUMB_TIP], pts[MIDDLE_TIP]) / size,
        reach_index=min(thumb_reach, _dist(pts[INDEX_TIP], pc) / size),
        reach_middle=min(thumb_reach, _dist(pts[MIDDLE_TIP], pc) / size),
    )


def classify(landmarks: Optional[Sequence], tuning: Tuning = DEFAULT_TUNING, aspect: float = 1.0) -> str:
    """Return one gesture label for a single hand.

    aspect = frame width / height. MediaPipe normalizes x and y separately, so
    x must be stretched by the aspect ratio before measuring distances.
    """
    if landmarks is None:
        return NONE
    s = analyze(landmarks, tuning, aspect)
    if s.size < tuning.min_hand_size:
        return NONE

    candidates = []
    if s.pinch_index < tuning.pinch_ratio and s.reach_index >= tuning.pinch_min_reach:
        candidates.append((s.pinch_index, PINCH_INDEX))
    if s.pinch_middle < tuning.pinch_ratio and s.reach_middle >= tuning.pinch_min_reach:
        candidates.append((s.pinch_middle, PINCH_MIDDLE))
    if candidates:
        return min(candidates)[1]

    e = s.extended
    up = (e["index"], e["middle"], e["ring"], e["pinky"])
    if up == (False, False, False, False):
        return FIST
    if up == (True, True, True, True):
        return PALM if s.thumb_out else NONE
    if up == (True, True, True, False):
        return THREE
    if up == (True, True, False, False):
        return TWO
    if up == (True, False, False, False):
        return POINT
    return NONE


def near_pinch(landmarks: Sequence, tuning: Tuning = DEFAULT_TUNING, aspect: float = 1.0) -> bool:
    """True when thumb and index are closing in on a pinch (cursor should freeze)."""
    s = analyze(landmarks, tuning, aspect)
    return s.pinch_index < tuning.pinch_ratio * tuning.pre_pinch_freeze


@dataclass
class HoldRule:
    frames: int
    seconds: float


DEFAULT_HOLD_RULES = {
    PINCH_INDEX: HoldRule(frames=3, seconds=0.08),
    PINCH_MIDDLE: HoldRule(frames=3, seconds=0.08),
    FIST: HoldRule(frames=5, seconds=0.5),
    PALM: HoldRule(frames=5, seconds=0.5),
}


class Stabilizer:
    """Turns noisy per-frame labels into one-shot actions.

    A pose fires once after it has been held for its HoldRule. It cannot fire
    again until the pose is released, and no pose can fire within `cooldown`
    seconds of the previous one. Up to `grace_frames` stray frames in the middle
    of a hold (tracking flicker) are tolerated without resetting the hold.
    """

    def __init__(self, rules: Optional[dict] = None, cooldown: float = 0.9, grace_frames: int = 2):
        self.rules = dict(DEFAULT_HOLD_RULES if rules is None else rules)
        self.cooldown = cooldown
        self.grace_frames = grace_frames
        self._last_fire = -math.inf
        self.reset()

    def reset(self) -> None:
        self._cand: Optional[str] = None
        self._start = 0.0
        self._count = 0
        self._miss = 0
        self._fired = False

    def update(self, label: str, now: float) -> Optional[str]:
        if label == self._cand:
            self._count += 1
            self._miss = 0
        else:
            if self._cand is not None and self._miss < self.grace_frames:
                self._miss += 1
                return None
            self._cand = label if label in self.rules else None
            self._start = now
            self._count = 1
            self._miss = 0
            self._fired = False
        if self._cand is None or self._fired:
            return None
        rule = self.rules[self._cand]
        if (
            self._count >= rule.frames
            and now - self._start >= rule.seconds
            and now - self._last_fire >= self.cooldown
        ):
            self._fired = True
            self._last_fire = now
            return self._cand
        return None


class SwipeTracker:
    """Detects a fast horizontal swipe while a swipe pose (two/three fingers) is held.

    Fires when the palm moves at least `min_distance` (fraction of frame width)
    within `window` seconds. Slow drift never covers that distance inside the
    window, so it does not fire. After a swipe there is a `cooldown`, and swipes
    in the opposite direction are blocked for `return_block` seconds so moving
    the hand back to center does not undo the action.
    """

    def __init__(
        self,
        poses=(TWO, THREE),
        min_distance: float = 0.2,
        window: float = 0.4,
        cooldown: float = 0.9,
        return_block: float = 1.2,
        grace_frames: int = 2,
    ):
        self.poses = tuple(poses)
        self.min_distance = min_distance
        self.window = window
        self.cooldown = cooldown
        self.return_block = return_block
        self.grace_frames = grace_frames
        self._last_fire = -math.inf
        self._last_dir: Optional[str] = None
        self.reset()

    def reset(self) -> None:
        self._hist: deque = deque()
        self._pose: Optional[str] = None
        self._miss = 0

    def update(self, label: str, x: Optional[float], now: float) -> Optional[tuple[str, str]]:
        """Returns (pose, "left"|"right") when a swipe fires, else None."""
        if label not in self.poses or x is None:
            self._miss += 1
            if self._miss > self.grace_frames:
                self.reset()
            return None
        self._miss = 0
        if label != self._pose:
            self._hist.clear()
            self._pose = label
        self._hist.append((now, x))
        while self._hist and now - self._hist[0][0] > self.window:
            self._hist.popleft()
        if now - self._last_fire < self.cooldown:
            return None
        dx = x - self._hist[0][1]
        if abs(dx) < self.min_distance:
            return None
        direction = "right" if dx > 0 else "left"
        self._hist.clear()
        self._hist.append((now, x))
        if (
            self._last_dir is not None
            and direction != self._last_dir
            and now - self._last_fire < self.return_block
        ):
            return None
        self._last_fire = now
        self._last_dir = direction
        return (label, direction)


class CursorMapper:
    """Maps a fingertip in (already mirrored) frame coordinates to screen pixels
    with exponential-moving-average smoothing.

    Only the central part of the frame (inside `margin`) maps to the full screen,
    so the far edges are reachable without the hand leaving the camera view.
    The result stays `edge_inset` pixels away from the screen edges so the cursor
    never lands exactly on a corner and trips pyautogui's fail-safe by itself.
    """

    def __init__(self, screen_w: int, screen_h: int, alpha: float = 0.35, margin: float = 0.15, edge_inset: int = 2):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.alpha = alpha
        self.margin = margin
        self.edge_inset = edge_inset
        self.reset()

    def reset(self) -> None:
        self._pos: Optional[tuple[float, float]] = None

    def _map(self, v: float) -> float:
        span = 1.0 - 2 * self.margin
        return min(1.0, max(0.0, (v - self.margin) / span))

    def update(self, x: float, y: float) -> tuple[int, int]:
        tx = self._map(x) * (self.screen_w - 1)
        ty = self._map(y) * (self.screen_h - 1)
        if self._pos is None:
            self._pos = (tx, ty)
        else:
            px, py = self._pos
            self._pos = (px + self.alpha * (tx - px), py + self.alpha * (ty - py))
        lo = self.edge_inset
        cx = int(round(min(self.screen_w - 1 - lo, max(lo, self._pos[0]))))
        cy = int(round(min(self.screen_h - 1 - lo, max(lo, self._pos[1]))))
        return cx, cy


class GraceTimer:
    """Face-lock window: a face match opens it for `grace` seconds.

    Seeing only a different face closes it immediately, so a stranger cannot
    inherit the grace period after the owner steps away.
    """

    def __init__(self, grace: float = 4.0):
        self.grace = grace
        self._last: Optional[float] = None

    def match(self, now: float) -> None:
        self._last = now

    def revoke(self) -> None:
        self._last = None

    def is_open(self, now: float) -> bool:
        return self._last is not None and now - self._last <= self.grace
