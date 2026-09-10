# Laser_target
# Mosquito Laser Targeting System

A detect → track → predict → aim → fire pipeline for identifying and neutralising
mosquitoes with a laser, built with a hardware abstraction layer so the exact
same control code runs against a simulation or real camera/gimbal/laser hardware.


## Why this project

This started as an exploration of real-time computer vision and control systems —
the same category of problem behind autonomous targeting, drone tracking, and
sensor-fusion systems more broadly. The mosquito use case (inspired by Intellectual
Ventures' Photonic Fence research) is a good self-contained way to implement the
full pipeline end to end: detection, state estimation, latency-compensated
prediction, actuator control, and safety-critical interlocks.

## How it works

1. **Detect** — OpenCV background subtraction (`MOG2`) isolates moving blobs;
   contour area filters them into mosquito-sized targets vs. large objects
   (e.g. a person), which are treated differently downstream.
2. **Track** — a constant-velocity Kalman filter smooths noisy detections into
   a position + velocity estimate, coasting through brief missed detections.
3. **Predict ahead** — the filter projects the target's position forward by the
   pipeline's own latency (camera + processing + servo lag), so the system aims
   where the target *will be*, not where it *was*.
4. **Aim** — predicted pixel coordinates convert to pan/tilt angles via a
   pinhole camera model.
5. **Actuate** — a slew-rate-limited gimbal model (or a real servo/galvo rig
   over serial) moves toward the commanded angle.
6. **Fire — gated by a safety interlock** — a pulse only fires when aim error
   is below a lock tolerance *and* every interlock clears: armed state, no
   large object (e.g. a person) in frame within a 3-second hold window, inside
   the mechanical firing envelope, and past the pulse cooldown.

## Results (simulation)

| Metric | Value |
|---|---|
| Mean prediction error | ~15.6 px |
| Mean aim error when firing | 0.17° |
| Pulses fired / 10s run | 13 |
| Shots fired while a person was in frame | 0 / 0 |

## Running it

```bash
pip install -r requirements.txt
python src/mosquito_targeting.py
```

Runs in simulation mode by default (`MODE = "sim"` in `config.py`) — a synthetic
camera generates an erratically flying target and, at intervals, a large
"person" object to test the safety interlock. Output is a console report plus
`docs/mosquito_targeting.png`.

## Going from simulation to real hardware

The hardware layer is already abstracted behind matching interfaces:

| Component | Simulation | Real hardware |
|---|---|---|
| Camera | `SimCamera` (synthetic frames) | `RealCamera` (`cv2.VideoCapture`) |
| Gimbal | `SimGimbal` (slew-limited model) | `ServoGimbal` (serial → Arduino, `P<pan>,T<tilt>`) |
| Laser | `SimLaser` (logs pulses) | `GPIOLaser` (Raspberry Pi GPIO → MOSFET driver) |

To go real: set `MODE = "real"` in `config.py`, wire up the hardware, run a
one-time camera/laser offset calibration (see `Aimer.pan_offset_deg` /
`tilt_offset_deg`), and keep every safety interlock enabled.

## Known limitations

- Person-detection currently relies on blob size from background subtraction,
  not a real person detector — a production system should swap this for an
  actual model (e.g. a lightweight YOLO detector) so a stationary or partially
  occluded person can't be silently absorbed into the background model.
- The pinhole aiming model assumes the laser is co-located with the camera;
  real rigs need a one-time calibration to correct for the physical offset
  between the two.
- 60 fps is workable at close range but marginal for fast, distant targets —
  higher frame-rate global-shutter cameras would meaningfully improve
  prediction accuracy.
- **Laser safety is a real, non-optional constraint.** Anything with enough
  power to affect an insect is well outside Class 1 eye-safe limits. Beam
  containment, an enclosed test volume, and never firing toward reflective
  surfaces are hardware-level requirements, not software features — this
  repo does not attempt to solve that part.

## Stack

Python, OpenCV, NumPy, Matplotlib. Kalman filtering implemented from first
principles (no `filterpy` dependency) to keep the state estimation logic
fully visible.
