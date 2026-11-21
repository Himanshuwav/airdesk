# AirDesk: Gesture Control for macOS

Control your Mac with hand gestures through the webcam. A face lock makes sure only you can drive it: the app continuously verifies the person on camera against your reference photo and ignores gestures from anyone else.

Macs have no Face ID API, so face recognition runs locally on webcam frames with our own model. No cloud calls, no extra hardware. Iron Man garage style.

## What it does

- Point with your index finger to move the cursor
- Pinch to click
- Three-finger swipe to switch apps / desktop spaces
- Fist to close a window
- Open-palm flick to minimize
- Face lock: gestures only work while your face is recognized, and it re-locks when you step away

## Status

Built and tested on a MacBook Air (November 2026):

- 40/40 gesture unit tests pass (classification, pinch vs fist, stabilization, swipes, cursor mapping, face-lock grace timer)
- Face match verified on webcam stills: the owner scored 0.37-0.45 and unlocked, a stranger scored 0.80-0.84 and stayed locked (0.50 threshold, face_recognition distance metric)
- Runs on the built-in laptop webcam, no extra hardware

## Setup (macOS, Apple Silicon)

1. Python 3.12
2. Run ./setup.sh (creates .venv and installs dependencies)
3. Set face_photo in config.json to a clear photo of you containing one face, or pass --face-photo /path/to/photo.jpg
4. Grant Camera permission to your terminal, and Accessibility for live cursor and keyboard control
5. Try .venv/bin/python airdesk.py --dry-run first

## How it works

- Hands: MediaPipe hand-landmark tracking on webcam frames (OpenCV capture)
- Face lock: face_recognition (dlib embeddings) compares the live face to your reference photo, all locally
- Actions: PyAutoGUI and Quartz for cursor, clicks and keyboard shortcuts
- The CLI supports dry runs, camera listing, a standalone face check and saving a preview

## Roadmap

- Tighten gesture accuracy and cursor feel
- Custom gesture mapping (user-defined gesture to action)
- Package as a proper Mac app (menu bar, launch at login)
- Docs and a demo video

## Known limits

- One hand at a time; decent lighting helps both hands and face
- Face lock is photo matching, not liveness: a held-up photo could pass. Fine for keeping strangers from driving your Mac, not for high security
- Needs Camera and Accessibility permissions; the first run may require reopening the terminal
