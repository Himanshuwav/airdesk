#!/usr/bin/env python3
"""Synthetic-landmark tests for gestures.py. Run: python3 test_gestures.py"""
import math
import unittest

import gestures as g

# Hand model in "hand units": wrist at the origin, wrist->middle knuckle = 1.0,
# y grows downward like image coordinates, palm facing the camera.
MCP = {"index": (-0.30, -0.95), "middle": (0.0, -1.0), "ring": (0.25, -0.92), "pinky": (0.45, -0.80)}
SEG = {"index": (0.42, 0.25, 0.22), "middle": (0.47, 0.28, 0.24), "ring": (0.43, 0.26, 0.22), "pinky": (0.33, 0.20, 0.19)}
SPLAY = {"index": -0.12, "middle": 0.0, "ring": 0.10, "pinky": 0.22}
THUMB_OUT = [(-0.30, -0.25), (-0.55, -0.45), (-0.72, -0.65), (-0.85, -0.82)]
# Thumb folded across the curled fingers, as in a real fist.
THUMB_TUCKED = [(-0.30, -0.25), (-0.40, -0.42), (-0.35, -0.70), (-0.05, -0.85)]


def straight(name):
    mx, my = MCP[name]
    a = SPLAY[name]
    dx, dy = math.sin(a), -math.cos(a)
    pts, x, y = [], mx, my
    for seg in SEG[name]:
        x, y = x + dx * seg, y + dy * seg
        pts.append((x, y))
    return [(mx, my)] + pts


def curled(name):
    # Seen from the front, a curled finger's PIP sticks up a little and the
    # tip folds back down onto the palm below the knuckle.
    mx, my = MCP[name]
    pip = (mx, my - 0.15)
    dip = (mx, pip[1] + 0.12)
    tip = (mx, dip[1] + 0.16)
    return [(mx, my), pip, dip, tip]


def build(fingers, thumb):
    pts = [(0.0, 0.0)] + list(thumb)
    for name in ("index", "middle", "ring", "pinky"):
        f = fingers[name]
        pts += f if isinstance(f, list) else (straight(name) if f else curled(name))
    assert len(pts) == 21
    return pts


def place(hand, scale=0.15, cx=0.5, cy=0.75, angle_deg=0.0, aspect=1.0, mirror=False):
    """Hand units -> normalized MediaPipe coordinates (x divided by aspect)."""
    a = math.radians(angle_deg)
    out = []
    for x, y in hand:
        if mirror:
            x = -x
        rx = x * math.cos(a) - y * math.sin(a)
        ry = x * math.sin(a) + y * math.cos(a)
        out.append(((cx + scale * rx) / aspect if aspect != 1.0 else cx + scale * rx, cy + scale * ry, 0.0))
    return out


def F(i, m, r, p):
    return {"index": i, "middle": m, "ring": r, "pinky": p}


POINT_HAND = build(F(True, False, False, False), THUMB_TUCKED)
TWO_HAND = build(F(True, True, False, False), THUMB_TUCKED)
THREE_HAND = build(F(True, True, True, False), THUMB_TUCKED)
PALM_HAND = build(F(True, True, True, True), THUMB_OUT)
FIST_HAND = build(F(False, False, False, False), THUMB_TUCKED)

PINCH_THUMB = [(-0.30, -0.25), (-0.50, -0.50), (-0.58, -0.80), (-0.60, -1.05)]
PINCH_INDEX_FINGER = [(-0.30, -0.95), (-0.42, -1.30), (-0.56, -1.28), (-0.61, -1.08)]
PINCH_HAND = build(F(PINCH_INDEX_FINGER, False, False, False), PINCH_THUMB)
OK_SIGN_HAND = build(F(PINCH_INDEX_FINGER, True, True, True), PINCH_THUMB)

MID_PINCH_THUMB = [(-0.30, -0.25), (-0.45, -0.50), (-0.38, -0.88), (-0.24, -1.22)]
MID_PINCH_FINGER = [(0.0, -1.0), (-0.02, -1.45), (-0.14, -1.42), (-0.22, -1.25)]
MID_PINCH_HAND = build(F(True, MID_PINCH_FINGER, False, False), MID_PINCH_THUMB)

ALL = {
    g.POINT: POINT_HAND,
    g.TWO: TWO_HAND,
    g.THREE: THREE_HAND,
    g.PALM: PALM_HAND,
    g.FIST: FIST_HAND,
    g.PINCH_INDEX: PINCH_HAND,
    g.PINCH_MIDDLE: MID_PINCH_HAND,
}


class TestClassify(unittest.TestCase):
    def test_every_gesture(self):
        for label, hand in ALL.items():
            with self.subTest(label=label):
                self.assertEqual(g.classify(place(hand)), label)

    def test_ok_sign_is_left_click(self):
        self.assertEqual(g.classify(place(OK_SIGN_HAND)), g.PINCH_INDEX)

    def test_scale_invariant_near_and_far(self):
        for scale in (0.05, 0.10, 0.25, 0.40):
            for label, hand in ALL.items():
                with self.subTest(scale=scale, label=label):
                    lm = place(hand, scale=scale, cy=0.5 + scale)
                    self.assertEqual(g.classify(lm), label)

    def test_rotation_and_position(self):
        for angle in (-30, -15, 15, 30):
            for cx in (0.3, 0.5, 0.7):
                for label, hand in ALL.items():
                    with self.subTest(angle=angle, cx=cx, label=label):
                        self.assertEqual(g.classify(place(hand, cx=cx, angle_deg=angle)), label)

    def test_widescreen_aspect_correction(self):
        aspect = 16 / 9
        for label, hand in ALL.items():
            with self.subTest(label=label):
                lm = place(hand, cx=0.9, aspect=aspect)
                self.assertEqual(g.classify(lm, aspect=aspect), label)

    def test_mirrored_hand(self):
        for label, hand in ALL.items():
            with self.subTest(label=label):
                self.assertEqual(g.classify(place(hand, mirror=True)), label)

    def test_open_hand_with_tucked_thumb_is_not_palm(self):
        hand = build(F(True, True, True, True), THUMB_TUCKED)
        self.assertEqual(g.classify(place(hand)), g.NONE)

    def test_odd_combinations_are_none(self):
        hand = build(F(False, True, False, False), THUMB_TUCKED)  # middle finger only
        self.assertEqual(g.classify(place(hand)), g.NONE)
        hand = build(F(True, False, False, True), THUMB_TUCKED)  # "rock" sign
        self.assertEqual(g.classify(place(hand)), g.NONE)

    def test_tiny_or_missing_hand_is_none(self):
        self.assertEqual(g.classify(None), g.NONE)
        self.assertEqual(g.classify(place(FIST_HAND, scale=0.01)), g.NONE)

    def test_wrong_landmark_count_raises(self):
        with self.assertRaises(ValueError):
            g.classify([(0, 0, 0)] * 20)


class TestPinchVsFist(unittest.TestCase):
    def test_fist_thumb_is_close_to_index_tip_but_is_not_a_pinch(self):
        lm = place(FIST_HAND)
        s = g.analyze(lm)
        # The tucked thumb really is within pinch distance of the curled index
        # tip; only the palm-reach guard keeps this from being a click.
        self.assertLess(s.pinch_index, g.DEFAULT_TUNING.pinch_ratio)
        self.assertLess(s.reach_index, g.DEFAULT_TUNING.pinch_min_reach)
        self.assertEqual(g.classify(lm), g.FIST)

    def test_fist_with_thumb_on_index_tip_worst_case(self):
        tip = curled("index")[3]
        thumb = THUMB_TUCKED[:3] + [tip]
        hand = build(F(False, False, False, False), thumb)
        for scale in (0.06, 0.15, 0.3):
            for angle in (-25, 0, 25):
                with self.subTest(scale=scale, angle=angle):
                    self.assertEqual(g.classify(place(hand, scale=scale, angle_deg=angle)), g.FIST)

    def test_fist_with_thumb_on_middle_tip_is_not_right_click(self):
        tip = curled("middle")[3]
        hand = build(F(False, False, False, False), THUMB_TUCKED[:3] + [tip])
        self.assertEqual(g.classify(place(hand)), g.FIST)

    def test_tucked_thumb_poses_never_pinch(self):
        for label in (g.POINT, g.TWO, g.THREE, g.FIST):
            with self.subTest(label=label):
                self.assertNotIn(g.classify(place(ALL[label])), (g.PINCH_INDEX, g.PINCH_MIDDLE))

    def test_pinch_threshold_is_respected(self):
        # Open the pinch gradually: it must stop being a pinch past pinch_ratio.
        for gap, expect_pinch in ((0.05, True), (0.2, True), (0.45, False)):
            thumb = PINCH_THUMB[:3] + [(PINCH_INDEX_FINGER[3][0] + gap, PINCH_INDEX_FINGER[3][1] + 0.0)]
            hand = build(F(PINCH_INDEX_FINGER, False, False, False), thumb)
            label = g.classify(place(hand))
            with self.subTest(gap=gap):
                self.assertEqual(label == g.PINCH_INDEX, expect_pinch, label)

    def test_near_pinch_freezes_cursor_only_when_closing(self):
        self.assertFalse(g.near_pinch(place(POINT_HAND)))
        self.assertTrue(g.near_pinch(place(PINCH_HAND)))


def feed(stab, label, t0, seconds, fps=30):
    fired = []
    n = int(round(seconds * fps))
    for i in range(n):
        t = t0 + i / fps
        r = stab.update(label, t)
        if r:
            fired.append((r, t))
    return fired, t0 + n / fps


class TestStabilizer(unittest.TestCase):
    def test_fist_fires_once_after_half_second(self):
        s = g.Stabilizer()
        fired, _ = feed(s, g.FIST, 0.0, 3.0)
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0][0], g.FIST)
        self.assertGreaterEqual(fired[0][1], 0.5)
        self.assertLess(fired[0][1], 0.6)

    def test_short_fist_does_not_fire(self):
        s = g.Stabilizer()
        fired, t = feed(s, g.FIST, 0.0, 0.3)
        fired2, _ = feed(s, g.NONE, t, 0.5)
        self.assertEqual(fired + fired2, [])

    def test_palm_fires_once(self):
        s = g.Stabilizer()
        fired, _ = feed(s, g.PALM, 0.0, 2.0)
        self.assertEqual([f[0] for f in fired], [g.PALM])

    def test_pinch_fires_quickly(self):
        s = g.Stabilizer()
        fired, _ = feed(s, g.PINCH_INDEX, 0.0, 0.2)
        self.assertEqual([f[0] for f in fired], [g.PINCH_INDEX])
        self.assertLess(fired[0][1], 0.12)

    def test_point_and_swipe_poses_never_fire(self):
        s = g.Stabilizer()
        for label in (g.POINT, g.TWO, g.THREE, g.NONE):
            fired, _ = feed(s, label, 0.0, 2.0)
            self.assertEqual(fired, [], label)

    def test_cooldown_blocks_immediate_refire(self):
        s = g.Stabilizer()
        fired1, t = feed(s, g.PINCH_INDEX, 0.0, 0.15)
        _, t = feed(s, g.NONE, t, 0.15)
        fired2, t = feed(s, g.PINCH_INDEX, t, 0.15)  # still inside the 0.9 s cooldown
        self.assertEqual(len(fired1), 1)
        self.assertEqual(fired2, [])
        _, t = feed(s, g.NONE, t, 0.8)
        fired3, _ = feed(s, g.PINCH_INDEX, t, 0.15)
        self.assertEqual(len(fired3), 1)

    def test_pose_held_through_cooldown_fires_when_cooldown_ends(self):
        s = g.Stabilizer()
        fired1, t = feed(s, g.PINCH_INDEX, 0.0, 0.15)
        fired2, _ = feed(s, g.FIST, t, 1.5)
        self.assertEqual(len(fired1), 1)
        self.assertEqual([f[0] for f in fired2], [g.FIST])
        self.assertGreaterEqual(fired2[0][1] - fired1[0][1], 0.9)

    def test_flicker_tolerated_but_real_release_resets(self):
        s = g.Stabilizer()
        t = 0.0
        _, t = feed(s, g.FIST, t, 0.3)
        self.assertIsNone(s.update(g.NONE, t))  # one-frame tracking glitch
        fired, _ = feed(s, g.FIST, t + 1 / 30, 0.3)
        self.assertEqual(len(fired), 1)

        s = g.Stabilizer()
        _, t = feed(s, g.FIST, 0.0, 0.3)
        _, t = feed(s, g.NONE, t, 0.2)  # real release
        fired, _ = feed(s, g.FIST, t, 0.3)  # only 0.3 s into a new hold
        self.assertEqual(fired, [])

    def test_reset_clears_hold(self):
        s = g.Stabilizer()
        _, t = feed(s, g.FIST, 0.0, 0.4)
        s.reset()
        fired, _ = feed(s, g.FIST, t, 0.3)
        self.assertEqual(fired, [])


def swipe(tracker, label, x0, x1, t0, seconds, fps=30):
    fired = []
    n = max(1, int(round(seconds * fps)))
    for i in range(n + 1):
        t = t0 + i / fps
        x = x0 + (x1 - x0) * i / n
        r = tracker.update(label, x, t)
        if r:
            fired.append(r)
    return fired, t0 + (n + 1) / fps


class TestSwipeTracker(unittest.TestCase):
    def test_fast_two_finger_swipe_right(self):
        st = g.SwipeTracker()
        fired, _ = swipe(st, g.TWO, 0.3, 0.6, 0.0, 0.25)
        self.assertEqual(fired, [(g.TWO, "right")])

    def test_fast_three_finger_swipe_left(self):
        st = g.SwipeTracker()
        fired, _ = swipe(st, g.THREE, 0.7, 0.4, 0.0, 0.25)
        self.assertEqual(fired, [(g.THREE, "left")])

    def test_slow_drift_does_not_fire(self):
        st = g.SwipeTracker()
        fired, _ = swipe(st, g.TWO, 0.1, 0.9, 0.0, 3.0)  # 0.8 of the frame, but slowly
        self.assertEqual(fired, [])

    def test_fast_but_short_move_does_not_fire(self):
        st = g.SwipeTracker()
        fired, _ = swipe(st, g.TWO, 0.5, 0.6, 0.0, 0.1)
        self.assertEqual(fired, [])

    def test_fast_move_without_swipe_pose_does_not_fire(self):
        for label in (g.POINT, g.PALM, g.FIST, g.NONE):
            st = g.SwipeTracker()
            fired, _ = swipe(st, label, 0.2, 0.8, 0.0, 0.2)
            self.assertEqual(fired, [], label)

    def test_one_swipe_fires_once(self):
        st = g.SwipeTracker()
        fired, _ = swipe(st, g.TWO, 0.1, 0.9, 0.0, 0.35)  # long fast swipe
        self.assertEqual(fired, [(g.TWO, "right")])

    def test_return_stroke_is_blocked(self):
        st = g.SwipeTracker()
        f1, t = swipe(st, g.TWO, 0.3, 0.6, 0.0, 0.2)
        _, t = swipe(st, g.TWO, 0.6, 0.6, t, 0.7)
        f2, t = swipe(st, g.TWO, 0.6, 0.3, t, 0.2)  # bringing the hand back
        self.assertEqual(f1, [(g.TWO, "right")])
        self.assertEqual(f2, [])
        _, t = swipe(st, g.TWO, 0.3, 0.3, t, 0.6)
        f3, _ = swipe(st, g.TWO, 0.3, 0.05, t, 0.2)  # deliberate left swipe, after the block
        self.assertEqual(f3, [(g.TWO, "left")])

    def test_cooldown_between_same_direction_swipes(self):
        st = g.SwipeTracker()
        f1, t = swipe(st, g.TWO, 0.2, 0.5, 0.0, 0.2)
        f2, t = swipe(st, g.TWO, 0.5, 0.8, t, 0.2)  # too soon
        self.assertEqual(len(f1), 1)
        self.assertEqual(f2, [])
        _, t = swipe(st, g.TWO, 0.2, 0.2, t, 0.8)
        f3, _ = swipe(st, g.TWO, 0.2, 0.5, t, 0.2)
        self.assertEqual(f3, [(g.TWO, "right")])

    def test_pose_change_resets_history(self):
        st = g.SwipeTracker()
        f1, t = swipe(st, g.TWO, 0.3, 0.45, 0.0, 0.1)
        f2, _ = swipe(st, g.THREE, 0.45, 0.55, t, 0.1)
        self.assertEqual(f1 + f2, [])

    def test_brief_tracking_loss_mid_swipe_is_tolerated(self):
        st = g.SwipeTracker()
        self.assertIsNone(st.update(g.TWO, 0.30, 0.00))
        self.assertIsNone(st.update(g.TWO, 0.38, 0.05))
        self.assertIsNone(st.update(g.NONE, None, 0.10))  # motion blur
        self.assertEqual(st.update(g.TWO, 0.55, 0.15), (g.TWO, "right"))


class TestCursorMapper(unittest.TestCase):
    def test_center_maps_to_center(self):
        c = g.CursorMapper(1440, 900)
        x, y = c.update(0.5, 0.5)
        self.assertAlmostEqual(x, 720, delta=1)
        self.assertAlmostEqual(y, 450, delta=1)

    def test_smoothing_converges_without_jumping(self):
        c = g.CursorMapper(1000, 1000, alpha=0.35)
        c.update(0.5, 0.5)
        x1, _ = c.update(0.8, 0.5)
        self.assertGreater(x1, 500)
        self.assertLess(x1, 900)  # did not jump straight to the target
        for _ in range(40):
            x, _ = c.update(0.8, 0.5)
        self.assertAlmostEqual(x, (0.8 - 0.15) / 0.7 * 999, delta=2)

    def test_never_lands_on_a_corner(self):
        c = g.CursorMapper(1440, 900)
        for _ in range(50):
            x, y = c.update(0.0, 0.0)
        self.assertEqual((x, y), (2, 2))
        c.reset()
        for _ in range(50):
            x, y = c.update(1.0, 1.0)
        self.assertEqual((x, y), (1437, 897))


class TestGraceTimer(unittest.TestCase):
    def test_grace_window(self):
        gt = g.GraceTimer(4.0)
        self.assertFalse(gt.is_open(0.0))
        gt.match(10.0)
        self.assertTrue(gt.is_open(13.9))
        self.assertFalse(gt.is_open(14.1))

    def test_stranger_revokes_immediately(self):
        gt = g.GraceTimer(4.0)
        gt.match(10.0)
        gt.revoke()
        self.assertFalse(gt.is_open(10.5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
