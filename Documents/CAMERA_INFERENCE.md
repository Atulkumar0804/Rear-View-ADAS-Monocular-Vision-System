# Camera Inference — Technical Reference

**File:** `inference/camera_inference.py`  
**Purpose:** Real-time ADAS perception from a live camera feed (USB webcam or Intel RealSense D455).

---

## Overview

`camera_inference.py` runs the full rear-view ADAS pipeline on a live video stream. Every frame is processed through detection, tracking, depth estimation, lane analysis, and safety scoring, then rendered with a professional HUD overlay.

```
Live Camera / Video File
        │
        ▼
 ┌──────────────────┐
 │ YOLO11n Detection │  ← frame-synchronous
 └────────┬─────────┘
          │
          ▼
 ┌──────────────────────┐
 │ Rider–Vehicle Merge  │  ← fuses person+vehicle detections
 └────────┬─────────────┘
          │
          ▼
 ┌──────────────────┐
 │   ByteTracker    │  ← persistent multi-object tracking
 └────────┬─────────┘
          │
          ▼
 ┌─────────────────────────────┐
 │ UVH-26 Classifier (cached)  │  ← every 5 frames per track
 └────────┬────────────────────┘
          │
          ▼
 ┌──────────────────────────────┐
 │ Classical Depth Estimation   │  ← every frame (fast)
 │  • Ground-plane projection   │
 │  • Size-based (UVH-26 dims)  │
 │  + Kalman smoothing          │
 └────────┬─────────────────────┘
          │
          ▼
 ┌─────────────────────────────────┐
 │ Lane-Aware Safety Assessment    │  ← TTC, MTTC, PET, DRAC
 │ + Rider Action Recommendation   │
 └────────┬────────────────────────┘
          │
          ▼
   HUD Annotated Frame → Display / Save
```

---

## Command-Line Usage

```bash
# Basic: default USB camera (index 0)
python inference/camera_inference.py

# Specify camera index
python inference/camera_inference.py --camera 1

# Process a video file instead of live camera
python inference/camera_inference.py --camera /path/to/video.mp4

# Intel RealSense D455 camera
python inference/camera_inference.py --realsense

# Save output to file
python inference/camera_inference.py --camera 0 --save output.mp4

# Run without display (headless / server)
python inference/camera_inference.py --camera 0 --no-display --save output.mp4

# Select GPU profile
python inference/camera_inference.py --profile a6000_full
python inference/camera_inference.py --profile jetson_nano_restricted
python inference/camera_inference.py --profile jetson_nano_power_save

# CPU-only mode
python inference/camera_inference.py --device cpu

# Rear-camera mode flag (enables rear-view specific tuning)
python inference/camera_inference.py --camera 0 --rear-camera

# Enable hybrid depth estimation
python inference/camera_inference.py --camera 0 --hybrid-depth
```

### All Arguments

| Argument | Default | Description |
|---|---|---|
| `--camera` | `0` | Camera index (int) or video file path (str) |
| `--width` | `1280` | Capture width in pixels |
| `--height` | `720` | Capture height in pixels |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--save` | `None` | Output video path to save annotated result |
| `--no-display` | off | Disable OpenCV window (headless mode) |
| `--realsense` | off | Use Intel RealSense D455 instead of USB camera |
| `--force-usb` | off | Force USB camera even if RealSense available |
| `--profile` | `a6000_full` | GPU compute profile |
| `--rear-camera` | off | Enable rear-camera ADAS mode |
| `--hybrid-depth` | off | Enable hybrid ML+classical depth |
| `-v` / `--verbose` | off | Verbose logging |

---

## Key Classes and Functions

### `KalmanFilter1D`
Smooths per-track depth estimates over time.

```python
kf = KalmanFilter1D(process_variance=0.01, measurement_variance=0.1)
smoothed = kf.update(raw_depth_m)
```

| Parameter | Default | Effect |
|---|---|---|
| `process_variance` | `0.01` | How fast we expect depth to change |
| `measurement_variance` | `0.1` | Trust in raw measurements (higher = smoother but laggier) |

---

### `DynamicHorizonEstimator`
Detects road horizon from lane-line vanishing points and applies EMA temporal smoothing. Corrects ground-plane depth errors caused by camera pitch from suspension.

```python
estimator = DynamicHorizonEstimator(
    frame_width=1920, frame_height=1080,
    ema_alpha=0.15,      # smoothing speed (0.05–0.5)
    fallback_ratio=0.55  # fraction of frame height used as fallback horizon
)
y_horizon, confidence = estimator.update(frame)
```

**Method:** Canny edge detection → Hough line detection → left/right line grouping → vanishing point intersection → EMA smoothed.

**Rate:** Recomputed every **10 frames** (configurable via `_horizon_interval`).

---

### `LaneDetector`
Assigns each detected bounding box to LEFT / CENTER / RIGHT lane by horizontal position. Accounts for rear-camera perspective flip (camera-left = rider-right).

```python
ld = LaneDetector(frame_width=1920, frame_height=1080)
info = ld.detect_lane([x1, y1, x2, y2])
# Returns: {'lane': 'CENTER', 'confidence': 0.87, 'spans_multiple_lanes': False, ...}
```

Lane assignment:
- `x_center < frame_width/3` → **RIGHT** lane (camera-left = rider-right)
- `frame_width/3 ≤ x_center < 2×frame_width/3` → **CENTER** lane
- `x_center ≥ 2×frame_width/3` → **LEFT** lane

---

### `RiderActionRecommendation`
Maps safety level + lane position to a human-readable rider instruction.

```python
rar = RiderActionRecommendation()
action = rar.get_rider_action(
    safety_level='CRITICAL',   # CRITICAL / WARNING / CAUTION / SAFE
    lane_info={'lane': 'CENTER'},
    distance_m=8.5,
    speed_kmh=60.0,
    relative_speed_kmh=15.0,
    motion='approaching',
    ego_speed_kmh=45.0
)
# Returns: {'action': 'EMERGENCY_BRAKE', 'urgency': 'CRITICAL',
#           'rider_instruction': 'Apply strong brakes immediately!', ...}
```

| Safety Level | Same Lane | Action |
|---|---|---|
| CRITICAL + approaching | Yes | `EMERGENCY_BRAKE` |
| CRITICAL | Yes | `STRONG_DECELERATE` |
| WARNING | Yes | `DECELERATE` |
| CAUTION | Yes | `MONITOR` |
| SAFE | Yes | `MAINTAIN_SPEED` |
| Any | Adjacent | `BE_AWARE` |

---

### `RearViewSafetyAssessment`
Computes four surrogate safety measures (SSMs) and determines risk level.

```python
rsa = RearViewSafetyAssessment(ego_vehicle_speed=0.0)

ttc  = rsa.calculate_ttc(distance_m, ego_speed_ms, rear_speed_ms)
mttc = rsa.calculate_mttc(distance_m, ego_speed_ms, rear_speed_ms, ego_a, rear_a)
pet  = rsa.calculate_pet(distance_m, ego_speed_ms, rear_speed_ms)
drac = rsa.calculate_drac(distance_m, ego_speed_ms, rear_speed_ms)

result = rsa.assess_risk_level(ttc, mttc, pet, drac, distance_m,
                                ego_speed_ms, rear_speed_ms, lane_info)
```

**SSM Thresholds:**

| Metric | Critical | Warning |
|---|---|---|
| TTC | < 1.0 s | < 1.5 s |
| DRAC | > 3.35 m/s² | > 2.0 m/s² |
| Distance | < 10 m | < 15 m |
| PET | < 1.0 s | — |

Risk is **CRITICAL** when ≥ 2 indicators are critical simultaneously.

---

### `RearSideUseCaseValidator`
Validates individual detections and the overall rear-view scenario.

```python
validator = RearSideUseCaseValidator(frame_width=1920, frame_height=1080)

# Per-detection
v = validator.is_valid_rear_detection(bbox, bbox_height, distance_m)
# {'is_valid': True, 'confidence': 0.83, ...}

# Scenario level
scenario = validator.validate_rear_scenario(detections, ego_speed_ms)
# {'scenario_type': 'approaching_vehicle', 'threat_level': 'high', ...}
```

Validation checks: horizontal FOV margins, vertical sky exclusion, minimum bbox size (20 px), distance in range (0.5–30 m).

---

### `CameraVehicleDetector` (Main Class)
Orchestrates the entire pipeline.

```python
detector = CameraVehicleDetector(
    device='cuda',
    zoedepth_interval=30,  # ML depth refresh every N frames
    correction_alpha=0.3,  # EMA weight for depth correction factor
    fps=30
)
detections = detector.detect_frame(frame)
assessments, scenario = detector.validate_and_assess_rear_scenario(detections, ego_speed_kmh=0.0)
annotated = detector.draw_detections(frame, detections, fps=fps)
```

**Classifier caching:** The UVH-26 classifier runs once per track every `_cls_interval=5` frames and caches the result. This reduces per-frame classification cost from ~15.4 ms to ~1.8 ms amortized.

**Horizon caching:** Horizon recomputed every `_horizon_interval=10` frames.

---

## Camera Auto-Discovery

If the specified camera index fails, the script automatically scans indices 0–9 and uses the first available camera:

```
🔍 Searching for available cameras...
✅ Found and opened camera 2 (1280x720)
```

Use `v4l2-ctl --list-devices` on Linux to see available devices.

---

## Intel RealSense D455 Setup

```bash
# Install RealSense SDK
pip install pyrealsense2

# Run with RealSense
python inference/camera_inference.py --realsense
```

If RealSense fails, the script automatically falls back to a USB camera.

Troubleshooting:
- Use a **USB 3.0** port (or powered USB hub)
- Remove any lens cover
- Run `rs-enumerate-devices` to confirm detection

---

## HUD Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ REAR VIEW ADAS                    CLEAR REAR | THREAT: NONE     │ FPS: 54.1 │
│ Advanced Driver Assistance System                     DET: 3    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  T01  Sedan  87%                T03  Bus  94%                   │
│  ┌──────────┐                   ┌──────────────┐                │
│  │          │                   │              │                │
│  └──────────┘                   └──────────────┘                │
│  12.3m  STABLE                  8.1m  APPROACHING  42.0km/h    │
│  [WARNING]  TTC 1.42s           [CRITICAL]  TTC 0.81s           │
│  >> Slow down gradually         >> Apply strong brakes          │
│                                                                  │
└──────────────────────────────────────────── SAFETY LEVEL ───────┘
                                              ■ CRITICAL
                                              ■ WARNING
                                              ■ CAUTION
                                              ■ SAFE
```

**Bounding box colors:** RED = CRITICAL, ORANGE = WARNING, CYAN = CAUTION, GREEN = SAFE.

---

## Performance

| Platform | FPS | Notes |
|---|---|---|
| RTX A6000 | 54.1 | Lightweight mode |
| Jetson Orin NX | 27.0 | Lightweight mode |
| Jetson Nano | 12.1 | TRT INT8 → 21.4 |
| Raspberry Pi 5 | 16–20 | Reduced resolution |

Press `q` or `ESC` to stop. Session statistics are printed on exit.