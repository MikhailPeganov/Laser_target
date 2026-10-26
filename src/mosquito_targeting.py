"""
Mosquito Laser Targeting System
================================
Detect -> Track -> Predict -> Aim -> Fire, with a hardware abstraction layer
so the same pipeline runs against a simulated world or real hardware.

Pipeline
--------
1. FrameSource   : SimCamera (synthetic frames) | RealCamera (cv2.VideoCapture)
2. Detector      : background subtraction + blob filtering by mosquito-sized area
3. KalmanTracker : constant-velocity Kalman filter, predicts ahead to cancel latency
4. Aimer         : pixel coords -> pan/tilt angles via pinhole camera model
5. Gimbal        : SimGimbal (slew-rate limited) | ServoGimbal (serial to Arduino)
6. Laser         : SimLaser | GPIOLaser
7. SafetyInterlock: arming, target-size gate, large-object (person) inhibit,
                    pulse length cap, cooldown, no-fire zones

To go real: set MODE = "real", wire up ServoGimbal / GPIOLaser / RealCamera,
run calibrate_camera() once, and keep every interlock enabled.
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODE = "sim"           # "sim" or "real"
np.random.seed(3)


# =====================================================================
# Configuration
# =====================================================================
@dataclass
class Config:
    frame_w: int = 640
    frame_h: int = 480
    fps: float = 60.0
    focal_px: float = 700.0          # camera focal length in pixels (from calibration)
    min_target_area: int = 4         # px^2 - mosquito lower bound
    max_target_area: int = 60        # px^2 - mosquito upper bound
    person_area: int = 4000          # px^2 - anything this big inhibits firing
    lock_tolerance_deg: float = 0.25 # aim error required to fire
    system_latency_s: float = 0.030  # camera+processing+servo delay to predict over
    pulse_s: float = 0.010           # laser pulse length (short!)
    cooldown_s: float = 0.050        # min time between pulses
    max_pan_deg: float = 30.0        # mechanical / no-fire limits
    max_tilt_deg: float = 20.0
    gimbal_slew_deg_s: float = 400.0 # how fast the gimbal can move
    gimbal_lag: float = 0.35         # first-order response per step (sim)

CFG = Config()


# =====================================================================
# 1. Frame sources
# =====================================================================
class SimCamera:
    """Synthetic camera: noisy background + a mosquito with erratic flight.
    Occasionally a large object (a 'person') enters frame to test interlocks."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pos = np.array([cfg.frame_w * 0.3, cfg.frame_h * 0.5])
        self.vel = np.array([120.0, -40.0])          # px/s
        self.dt = 1.0 / cfg.fps
        self.person_frames = set(range(220, 260))    # frames where a person is visible
        self.frame_idx = 0
        self.true_path = []

    def _step_target(self):
        # random acceleration = erratic mosquito flight
        accel = np.random.normal(0, 900, 2)
        self.vel = np.clip(self.vel + accel * self.dt, -350, 350)
        self.pos += self.vel * self.dt
        # bounce off edges
        for i, lim in enumerate((self.cfg.frame_w, self.cfg.frame_h)):
            if self.pos[i] < 20 or self.pos[i] > lim - 20:
                self.vel[i] *= -1
                self.pos[i] = np.clip(self.pos[i], 20, lim - 20)

    def read(self):
        self._step_target()
        frame = np.full((self.cfg.frame_h, self.cfg.frame_w), 200, np.uint8)
        frame = frame + np.random.normal(0, 6, frame.shape).astype(np.int16)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        x, y = int(self.pos[0]), int(self.pos[1])
        cv2.circle(frame, (x, y), 2, 40, -1)                 # the mosquito (~13 px^2)

        if self.frame_idx in self.person_frames:            # large intruding object
            cv2.rectangle(frame, (400, 100), (520, 400), 60, -1)

        self.true_path.append(self.pos.copy())
        self.frame_idx += 1
        return True, frame


class RealCamera:
    """Live camera via OpenCV. Use a high-fps global-shutter camera for real use."""

    def __init__(self, cfg: Config, device=0):
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_h)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)

    def read(self):
        ok, frame = self.cap.read()
        if ok and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return ok, frame


# =====================================================================
# 2. Detector  (identical code for sim and real frames)
# =====================================================================
class Detector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.bg = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=25, detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, frame):
        """Returns (targets, big_objects). targets = list of (cx, cy, area)."""
        mask = self.bg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        targets, big = [], []
        for c in contours:
            area = cv2.contourArea(c)
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            if self.cfg.min_target_area <= area <= self.cfg.max_target_area:
                targets.append((cx, cy, area))
            elif area >= self.cfg.person_area:
                big.append((cx, cy, area))
        return targets, big


# =====================================================================
# 3. Kalman tracker (constant velocity) with predict-ahead
# =====================================================================
class KalmanTracker:
    def __init__(self, dt):
        self.dt = dt
        self.x = None                                # state [px, py, vx, vy]
        self.P = np.eye(4) * 500
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], float)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], float)
        q = 800.0                                     # process noise (erratic flight)
        self.Q = np.diag([0.25 * dt**4, 0.25 * dt**4, dt**2, dt**2]) * q
        self.R = np.eye(2) * 2.0                      # measurement noise (px^2)
        self.misses = 0

    def update(self, meas):
        if meas is None:
            if self.x is not None:
                self.x = self.F @ self.x
                self.P = self.F @ self.P @ self.F.T + self.Q
                self.misses += 1
                if self.misses > 15:                  # lost track
                    self.x = None
            return
        z = np.array(meas, float)
        if self.x is None:
            self.x = np.array([z[0], z[1], 0, 0], float)
            self.misses = 0
            return
        # predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # update
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.misses = 0

    def predict_ahead(self, seconds):
        """Where will the target be after `seconds`? Cancels system latency."""
        if self.x is None:
            return None
        return self.x[:2] + self.x[2:] * seconds

    @property
    def tracking(self):
        return self.x is not None


# =====================================================================
# 4. Aimer: pixel -> pan/tilt angles (pinhole model, laser co-located w/ camera)
# =====================================================================
class Aimer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cx, self.cy = cfg.frame_w / 2, cfg.frame_h / 2
        # For real hardware replace with calibration offsets between laser and camera axes
        self.pan_offset_deg = 0.0
        self.tilt_offset_deg = 0.0

    def pixel_to_angles(self, px, py):
        pan = math.degrees(math.atan2(px - self.cx, self.cfg.focal_px)) + self.pan_offset_deg
        tilt = -math.degrees(math.atan2(py - self.cy, self.cfg.focal_px)) + self.tilt_offset_deg
        return pan, tilt


# =====================================================================
# 5. Gimbal (actuator) — sim and real share an interface
# =====================================================================
class SimGimbal:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pan = 0.0
        self.tilt = 0.0
        self.dt = 1.0 / cfg.fps

    def command(self, pan, tilt):
        pan = float(np.clip(pan, -self.cfg.max_pan_deg, self.cfg.max_pan_deg))
        tilt = float(np.clip(tilt, -self.cfg.max_tilt_deg, self.cfg.max_tilt_deg))
        max_step = self.cfg.gimbal_slew_deg_s * self.dt
        for attr, target in (("pan", pan), ("tilt", tilt)):
            cur = getattr(self, attr)
            step = (target - cur) * self.cfg.gimbal_lag       # first-order lag
            step = float(np.clip(step, -max_step, max_step))  # slew-rate limit
            setattr(self, attr, cur + step)

    def position(self):
        return self.pan, self.tilt


class ServoGimbal:
    """Real 2-axis servo/galvo gimbal driven by an Arduino over serial.
    Arduino sketch should parse lines like 'P<pan>,T<tilt>\\n' in degrees."""

    def __init__(self, cfg: Config, port="/dev/ttyUSB0", baud=115200):
        import serial                           # pip install pyserial
        self.cfg = cfg
        self.ser = serial.Serial(port, baud, timeout=0.01)
        self.pan = self.tilt = 0.0

    def command(self, pan, tilt):
        self.pan = float(np.clip(pan, -self.cfg.max_pan_deg, self.cfg.max_pan_deg))
        self.tilt = float(np.clip(tilt, -self.cfg.max_tilt_deg, self.cfg.max_tilt_deg))
        self.ser.write(f"P{self.pan:.3f},T{self.tilt:.3f}\n".encode())

    def position(self):
        return self.pan, self.tilt         # or read back encoder values if available


# =====================================================================
# 6. Laser — sim and real share an interface
# =====================================================================
class SimLaser:
    def __init__(self):
        self.pulses = []

    def pulse(self, seconds, t):
        self.pulses.append(t)


class GPIOLaser:
    """Real laser driver on a GPIO pin (e.g. Raspberry Pi) through a MOSFET/driver board.
    NEVER wire the laser to be on by default; pin LOW must mean OFF."""

    def __init__(self, pin=18):
        import RPi.GPIO as GPIO                 # only on the Pi
        self.GPIO = GPIO
        self.pin = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        self.pulses = []

    def pulse(self, seconds, t):
        self.GPIO.output(self.pin, self.GPIO.HIGH)
        time.sleep(seconds)
        self.GPIO.output(self.pin, self.GPIO.LOW)
        self.pulses.append(t)

    def shutdown(self):
        self.GPIO.output(self.pin, self.GPIO.LOW)
        self.GPIO.cleanup()


# =====================================================================
# 7. Safety interlock — every fire request passes through here
# =====================================================================
class SafetyInterlock:
    INHIBIT_HOLD_S = 3.0     # once a large object is seen, hold fire this long

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.armed = False
        self.last_fire_t = -1e9
        self.inhibit_until = -1e9
        self.block_reasons = []

    def arm(self):
        self.armed = True

    def disarm(self):
        self.armed = False

    def may_fire(self, t, target, big_objects, pan, tilt, aim_err_deg):
        reasons = []
        if not self.armed:
            reasons.append("disarmed")
        if big_objects:
            self.inhibit_until = t + self.INHIBIT_HOLD_S  # latch: stationary person
        if t < self.inhibit_until:                         # stays inhibited after
            reasons.append("large object in frame")        # background model absorbs it
        if target is None:
            reasons.append("no valid target")
        if aim_err_deg > self.cfg.lock_tolerance_deg:
            reasons.append("not locked")
        if abs(pan) >= self.cfg.max_pan_deg or abs(tilt) >= self.cfg.max_tilt_deg:
            reasons.append("outside fire envelope")
        if t - self.last_fire_t < self.cfg.cooldown_s:
            reasons.append("cooldown")
        if reasons:
            self.block_reasons.extend(reasons)
            return False
        self.last_fire_t = t
        return True


# =====================================================================
# Main control loop
# =====================================================================
def build_hardware(mode, cfg):
    if mode == "real":
        return RealCamera(cfg), ServoGimbal(cfg), GPIOLaser()
    return SimCamera(cfg), SimGimbal(cfg), SimLaser()


def run(cfg: Config, n_frames=600):
    cam, gimbal, laser = build_hardware(MODE, cfg)
    det = Detector(cfg)
    trk = KalmanTracker(1.0 / cfg.fps)
    aim = Aimer(cfg)
    safe = SafetyInterlock(cfg)
    safe.arm()

    log = {k: [] for k in ("t", "meas", "pred", "cmd", "pos", "err", "fired", "inhibit")}
    dt = 1.0 / cfg.fps

    for i in range(n_frames):
        t = i * dt
        ok, frame = cam.read()
        if not ok:
            break

        targets, big = det.detect(frame)
        meas = None
        if targets:
            # nearest target to current track (or first if no track)
            if trk.tracking:
                meas = min(targets, key=lambda m: (m[0]-trk.x[0])**2 + (m[1]-trk.x[1])**2)[:2]
            else:
                meas = targets[0][:2]
        trk.update(meas)

        pred = trk.predict_ahead(cfg.system_latency_s)
        fired = False
        if pred is not None:
            pan_cmd, tilt_cmd = aim.pixel_to_angles(*pred)
            gimbal.command(pan_cmd, tilt_cmd)
            pan, tilt = gimbal.position()
            # aim error = angle between where laser points and predicted target
            err = math.hypot(pan - pan_cmd, tilt - tilt_cmd)
            if safe.may_fire(t, pred, big, pan, tilt, err):
                laser.pulse(cfg.pulse_s, t)
                fired = True
        else:
            pan_cmd = tilt_cmd = float("nan")
            pan, tilt = gimbal.position()
            err = float("nan")

        log["t"].append(t)
        log["meas"].append(meas if meas else (np.nan, np.nan))
        log["pred"].append(tuple(pred) if pred is not None else (np.nan, np.nan))
        log["cmd"].append((pan_cmd, tilt_cmd))
        log["pos"].append((pan, tilt))
        log["err"].append(err)
        log["fired"].append(fired)
        log["inhibit"].append(bool(big) or t < safe.inhibit_until)

    return cam, laser, safe, log


# =====================================================================
# Report + plots
# =====================================================================
def report(cam, laser, safe, log, cfg):
    t = np.array(log["t"])
    err = np.array(log["err"])
    fired = np.array(log["fired"])
    inhibit = np.array(log["inhibit"])
    meas = np.array(log["meas"], float)
    pred = np.array(log["pred"], float)
    truth = np.array(cam.true_path)
    tracked = ~np.isnan(pred[:, 0])

    # how far the predicted aim point was from the true target position
    pred_err_px = np.hypot(pred[:, 0] - truth[:len(pred), 0], pred[:, 1] - truth[:len(pred), 1])

    print("=" * 62)
    print("MOSQUITO LASER TARGETING — SIMULATION REPORT")
    print("=" * 62)
    print(f"Frames processed        : {len(t)}  ({t[-1]:.2f} s at {cfg.fps:.0f} fps)")
    print(f"Frames with detection   : {(~np.isnan(meas[:,0])).sum()}")
    print(f"Frames with active track: {tracked.sum()}")
    print(f"Mean prediction error   : {np.nanmean(pred_err_px):.2f} px "
          f"(median {np.nanmedian(pred_err_px):.2f} px)")
    print(f"Mean aim error (locked) : {np.nanmean(err[fired]):.3f} deg" if fired.any() else "No shots")
    print(f"Laser pulses fired      : {fired.sum()}")
    print(f"Frames inhibited (person in frame): {inhibit.sum()}  -> shots during inhibit: {(fired & inhibit).sum()}")
    from collections import Counter
    print("Top interlock blocks    :", dict(Counter(safe.block_reasons).most_common(4)))
    print("=" * 62)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) target truth vs measurement vs predicted aim point
    ax = axes[0, 0]
    ax.plot(truth[:, 0], truth[:, 1], color="black", lw=1, label="True mosquito path")
    ax.scatter(meas[:, 0], meas[:, 1], s=6, color="steelblue", label="Detections")
    ax.plot(pred[:, 0], pred[:, 1], color="red", lw=0.8, alpha=0.8, label="Predicted aim point")
    ax.scatter(pred[fired, 0], pred[fired, 1], marker="x", s=30, color="green", label="Laser pulse")
    ax.set_xlim(0, cfg.frame_w); ax.set_ylim(cfg.frame_h, 0)
    ax.set_title("Camera view: track and fire events")
    ax.set_xlabel("px"); ax.set_ylabel("px"); ax.legend(fontsize=8)

    # (b) prediction error over time, with inhibit windows
    ax = axes[0, 1]
    ax.plot(t, pred_err_px, color="darkorange", lw=0.9, label="Prediction error (px)")
    ax.fill_between(t, 0, np.nanmax(pred_err_px), where=inhibit, color="red", alpha=0.15,
                    label="Person in frame -> fire inhibited")
    ax.set_title("Predicted aim vs true target position")
    ax.set_xlabel("time (s)"); ax.set_ylabel("px"); ax.legend(fontsize=8)

    # (c) gimbal command vs actual position
    ax = axes[1, 0]
    cmd = np.array(log["cmd"]); pos = np.array(log["pos"])
    ax.plot(t, cmd[:, 0], color="red", lw=0.8, label="Pan command")
    ax.plot(t, pos[:, 0], color="black", lw=0.8, label="Pan actual")
    ax.plot(t, cmd[:, 1], color="blue", lw=0.8, label="Tilt command")
    ax.plot(t, pos[:, 1], color="gray", lw=0.8, label="Tilt actual")
    ax.set_title("Gimbal tracking response (slew-rate limited)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("degrees"); ax.legend(fontsize=8)

    # (d) aim error vs lock tolerance and pulses
    ax = axes[1, 1]
    ax.plot(t, err, color="purple", lw=0.8, label="Aim error (deg)")
    ax.axhline(cfg.lock_tolerance_deg, color="black", ls="--", label="Lock tolerance")
    ax.scatter(t[fired], err[fired], marker="x", color="green", s=25, label="Pulse fired")
    ax.set_ylim(0, max(1.0, np.nanmax(err) * 1.1))
    ax.set_title("Lock acquisition and firing")
    ax.set_xlabel("time (s)"); ax.set_ylabel("degrees"); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("/home/claude/mc_options/mosquito_targeting.png", dpi=130)
    print("\nChart saved to mosquito_targeting.png")


if __name__ == "__main__":
    cam, laser, safe, log = run(CFG)
    report(cam, laser, safe, log, CFG)
