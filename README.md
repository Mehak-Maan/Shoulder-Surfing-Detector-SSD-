# Shoulder Surfing Detector

**GitHub Repository:** [https://github.com/Mehak-Maan/Shoulder-Surfing-Detector-SSD-](https://github.com/Mehak-Maan/Shoulder-Surfing-Detector-SSD-)

Real-time, **fully offline**, **CPU-only** protection against people looking at your screen from behind.
A webcam watches for additional faces; escalation is toast → blur overlay → OS lock.

* Python 3.11 · MediaPipe Tasks face detection · OpenCV · no GPU · no network calls at runtime
* Windows primary; Linux and macOS via fallbacks

## 1. Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

Do not install `opencv-python` yourself; MediaPipe already brings OpenCV.
Tkinter (blur overlay) ships with python.org Python on Windows/macOS; on Debian/Ubuntu: `sudo apt install python3-tk`.

**Model file (one-time download, then offline forever).** Put the BlazeFace short-range model at
`models/blaze_face_short_range.tflite` (name and URL are in the MediaPipe "Face Detector" docs). The app
refuses to start with a clear message if the file is missing; it never downloads anything itself.

## 2. Choose a mode

| Mode | Command | What you get |
|------|---------|--------------|
| **A. GUI (debug)** | `python main.py --mode gui` | OpenCV window: green box = owner, red = intruder, state, FPS, debug toggles. Actions are **dry-run** (logged only) by default. |
| **B. Headless (service)** | `python main.py --mode headless` | No window. ~8 FPS, tray icon (pause/resume/quit), rotating log, single-instance guard, real actions **on** by default. |

Override actions in either mode: `--actions off|dry-run|on`.

## 3. Run

```bash
python main.py --mode gui                          # try it, nothing is really locked
python main.py --mode gui --actions on             # real blur/lock while watching the window
python main.py --mode headless                     # the everyday service
python main.py --mode headless --no-tray --log-level DEBUG
python main.py --mode gui --source video --video clip.mp4 --realtime
python main.py --benchmark 30 --bench-fps 8        # FPS / latency / CPU%
python main.py --report --source video --video clip.mp4 --labels labels.json
```

`labels.json`: `{"intruder_windows": [[12.0, 40.0], [70.0, 95.0]]}` (seconds). The report prints
false alarms/hour, alert latency, FPS and CPU%. For synthetic or static videos set `health.enabled: false`
(identical frames look like a frozen camera).

Exit codes: `0` ok · `1` too many consecutive errors · `2` bad config/arguments · `3` already running.

### GUI keys
`q` quit · `d` debug boxes · `t` tracks/IDs · `s` state history · `space` pause.

## 4. Configuration (`config.yaml`)

Every key is documented in the file; unknown keys, wrong types and out-of-range values are rejected at start.
Important ones:

| Key | Default | Meaning |
|-----|---------|---------|
| `state.debounce_s` | 0.7 | an intruder must be seen this long before SUSPECT escalates |
| `state.clear_s` | 2.0 | hysteresis before stepping down |
| `actions.camera_fault_policy` | `blur` | what a covered/unplugged camera triggers: `warn`, `blur`, `lock` |
| `health.*` | on | covered/dark and frozen-stream detection |
| `privacy.evidence_enabled` | `false` | opt-in encrypted snapshots on ALERT/LOCKED |
| `runtime.headless_fps` | 8 | headless processing rate |
| `runtime.log_file` | `~/.shoulder_surfing_detector/ssd.log` | rotating log (1 MB × 3) |

## 4b. Owner enrollment and gaze filter (Phase 4)

**Gaze filter** (`gaze:` section, on by default): a face only counts as a watcher when it is turned towards
the screen. Head yaw comes from the six BlazeFace keypoints, is smoothed per track and uses hysteresis
(`detector.max_yaw_deg` is the limit, `gaze.hysteresis_deg` the way back), so a face hovering around the limit does
not flicker. `gaze.use_pitch: true` also ignores faces looking far up or down (coarse, off by default).
This is head pose, not eye tracking.

**Owner enrollment** (optional, `identity:` section): without it the owner is "the largest face", which causes a false
alarm when you return after more than 10 s and misses a stranger who takes your seat. With it the owner is whoever
matches your stored face embeddings.

```bash
python main.py --enroll            # alone in view, face the screen, slowly turn your head a little (~10 s)
# set identity.enabled: true in config.yaml
python main.py --mode gui          # debug view (d) shows "sim 0.93" next to each face
python main.py --forget-owner      # delete the profile
```

* Stored: embedding vectors only, encrypted with `privacy.key_file` (Fernet, mode 0600). No photo is ever written.
  An embedding is still biometric data: protect the profile and key like a password.
* New faces are `verifying` for about 1 s (never intruders before they were checked); faces narrower than
  `identity.min_embed_px` cannot be verified and are never the owner.
* If `identity.enabled` is true but the profile is missing, unreadable or made by another embedder, the program
  refuses to start (exit 2) instead of silently running without identity.
* **Embedder quality.** The built-in `lbp` embedder needs no model file but is a weak texture descriptor, sensitive to
  lighting and glasses, and can confuse look-alikes or accept a photo. For better identity, set `embedder: onnx` and
  `onnx_model:` to an ArcFace/MobileFaceNet-style 112x112 ONNX model (runs on CPU through OpenCV). Calibrate with the
  GUI similarity read-out: owner should sit clearly above `threshold`, a colleague clearly below; adjust
  `threshold_margin`/`threshold_floor` if not.

## 5. Privacy

* No frames are written to disk by default. Frames live in RAM only, the latest one only.
* No network code is executed at runtime.
* Optional evidence (`privacy.evidence_enabled: true`): JPEG encrypted with Fernet (AES-128-CBC + HMAC), key file
  created with mode `0600`, at most `evidence_max_files` kept. Decrypt on purpose with
  `python main.py --decrypt-evidence file.jpg.enc --out file.jpg`.
  Whoever can read the key file can read the snapshots; keep both on an encrypted disk.

## 6. Autostart (headless)

**Windows** (Task Scheduler, runs at logon, no console window):

```bat
schtasks /Create /SC ONLOGON /TN "ShoulderSurfingDetector" /RL LIMITED /TR "\"C:\path\.venv\Scripts\pythonw.exe\" \"C:\path\main.py\" --mode headless"
```
Remove: `schtasks /Delete /TN "ShoulderSurfingDetector" /F`. It must run in your **interactive** session,
otherwise it cannot lock your desktop.

**Linux (systemd user service)** `~/.config/systemd/user/ssd.service`:

```ini
[Unit]
Description=Shoulder Surfing Detector
[Service]
ExecStart=/path/.venv/bin/python /path/main.py --mode headless --no-tray
Restart=on-failure
[Install]
WantedBy=graphical-session.target
```
`systemctl --user enable --now ssd.service` (use `Restart=on-failure`, not `always`: exit code 3 means "already running").

**macOS (launchd)** `~/Library/LaunchAgents/com.user.ssd.plist` with `ProgramArguments` =
`/path/.venv/bin/python /path/main.py --mode headless`, `RunAtLoad` = true, `KeepAlive` = `{SuccessfulExit: false}`;
then `launchctl load ~/Library/LaunchAgents/com.user.ssd.plist`. Grant the terminal/python Camera permission once.

## 7. Tests

```bash
pytest -q                       # synthetic sequences, no camera or model needed
mypy --strict .                 # (sandbox verification used --python-version 3.13)
ruff check .
```

## 8. Manual test checklist

Run `python main.py --mode gui --actions on` first (keep the window visible), then repeat in headless.

- [ ] **Pass-by**: someone walks behind you in < 1 s → no warning (debounce).
- [ ] **Second person**: someone stands/sits 1–2 m behind for > 1 s → SUSPECT → WARNING toast → blur after the configured delay; leaving for 2 s steps down to RECOVERY then SECURE.
- [ ] **Side angle**: stand 45° off-axis at ~1.5 m and ~3 m. Note the distance at which detection starts; 3 m is the weak spot.
- [ ] **Low light**: dim the room. Detection quality drops; if the picture goes near-black you get a camera fault (policy `blur`).
- [ ] **Covered camera**: tape over the lens → after ~2 s state jumps per `camera_fault_policy`; uncover → recovery.
- [ ] **Unplugged camera**: same, banner `CAMERA disconnected`, automatic reconnect on replug.
- [ ] **Owner leaves, stranger sits**: without enrollment the stranger is adopted as the owner after `owner.reacquire_s`; with enrollment you must get a WARNING.
- [ ] **Owner returns after > 10 s** (enrolled): no alarm. Repeat in a different light and with/without glasses; check the `sim` value stays above the threshold.
- [ ] **Colleague looks away**: a second person turned away from the screen is ignored (`not_facing`), turning towards it escalates.
- [ ] **Enrollment**: `--enroll` with a second person in view asks for you to be alone; `--forget-owner` removes the profile.
- [ ] **Pause/resume** from the tray; **Quit**; start a second copy → refused (exit 3).
- [ ] Headless `Ctrl+C` / `kill -TERM` exits cleanly and the overlay disappears.

## 9. Known limitations (honest list)

* The short-range model is built for ≤ 2 m; far-field tiling helps at 2–3 m but costs CPU and still misses small or turned heads.
* Without enrollment the owner is "the largest face": if the owner leaves and returns within 10 s you may get a false alarm; a stranger who sits down after the owner left for > 10 s becomes the owner. Enrollment fixes this, but the built-in LBP embedder is weak (see 4b); the ONNX embedder path is untested here because no model file was available.
* A genuinely dark room is indistinguishable from a covered lens and fails closed.
* The gaze filter is head pose only: eyes can look at the screen with the head turned, and a coarse yaw estimate fails on very small faces (they are then assumed to face the screen).
* Pause keeps the camera open (the camera LED stays on).
* Notifications, blur overlay and screen lock use OS facilities (PowerShell toast, Tk, `LockWorkStation`, `loginctl`/`xdg-screensaver`, `pmset`) and depend on your desktop environment. Wayland screenshots for the blur may be blocked.
* A camera-based detector cannot see someone looking at a reflection, a second camera or a phone zoom from afar.
