# Video Inference — Technical Reference

**File:** `inference/video_inference.py`  
**Purpose:** Offline ADAS processing of a pre-recorded video file. Reads input → runs full perception pipeline → writes annotated output video and optional CSV telemetry log.

---

## Overview

`video_inference.py` is the batch-processing counterpart to `camera_inference.py`. It adds:
- **Hybrid depth fusion** (classical geometric cues + async ML depth via DA2/MiDaS)
- **Multi-factor depth confidence** (pixel-height, aspect ratio, occlusion)
- **Learnable EMA correction factor** that adapts per-track and per-class
- **CSV telemetry logger** for offline analysis of all safety metrics

```
Input Video File
      │
      ▼
┌─────────────────────┐
│  YOLO11n Detection  │  ← frame-synchronous
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Rider–Vehicle Merge │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────┐
│     ByteTracker      │
└──────────┬───────────┘
           │
           ▼
┌───────────────────────────┐
│  UVH-26 Classifier        │  ← cached per track (every 5 frames)
└──────────┬────────────────┘
           │
           ┌─────────────────────────────────────────────┐
           │                                             │
           ▼                                             ▼
┌────────────────────────┐             ┌────────────────────────────┐
│  Classical Depth       │             │  Async ML Depth (DA2/MIDAS)│
│  • Ground-plane proj.  │             │  • Runs every N frames     │
│  • Size-based ranging  │             │  • Provides scale anchor   │
│  • Motion parallax     │             └────────────┬───────────────┘
│  • Multi-factor conf.  │                          │
│  • Kalman smoothing    │                          │ correction factor (EMA)
└──────────┬─────────────┘                          │
           │◄──────────────────── fused depth ──────┘
           ▼
┌─────────────────────────────────┐
│  Lane-Aware Safety Assessment   │
│  (TTC, MTTC, PET, DRAC)        │
│  + Rider Action Recommendation  │
└──────────┬──────────────────────┘
           │
           ├──► Annotated Output Video
           └──► CSV Telemetry Log (optional)
```

---

## Command-Line Usage

```bash
# Basic: process video, write output
python inference/video_inference.py --input video.mp4 --output result.mp4

# Write CSV telemetry log alongside video
python inference/video_inference.py --input video.mp4 --output result.mp4 --log detections.csv

# CPU-only mode
python inference/video_inference.py --input video.mp4 --output result.mp4 --device cpu

# Set classical depth weight (0.0 = full ML, 1.0 = full classical)
python inference/video_inference.py --input video.mp4 --output result.mp4 --classical-weight 0.75

# Set ML depth correction frequency (frames between ML depth refreshes)
python inference/video_inference.py --input video.mp4 --output result.mp4 --depth-interval 15

# Set ego vehicle speed for SSM calculations
python inference/video_inference.py --input video.mp4 --output result.mp4 --ego-speed 40.0

# Full example with all options
python inference/video_inference.py \
    --input rear_video.mp4 \
    --output annotated.mp4 \
    --log metrics.csv \
    --device cuda \
    --depth-interval 15 \
    --classical-weight 0.80 \
    --ego-speed 0.0
```

### All Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Input video file path |
| `--output` | required | Output annotated video path |
| `--log` | `None` | CSV log file path for telemetry |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--depth-interval` | `30` | Frames between async ML depth refreshes |
| `--classical-weight` | `0.80` | Weight of classical depth in hybrid fusion (0–1) |
| `--ego-speed` | `0.0` | Ego vehicle speed in km/h for SSM calculations |

---

## Key Classes and Functions

### `KalmanFilter1D`
Same as in `camera_inference.py`. Provides per-track depth smoothing.

---

### `DynamicHorizonEstimator`
Identical API to `camera_inference.py`. In video mode, runs every `_horizon_interval=15` frames (slightly less frequent than camera mode's 10 frames, since video playback is deterministic).

Also exposes a fallback `_estimate_horizon_from_edges()` method that looks for peak edge activity in the upper-middle region of the frame when lane-line vanishing point detection fails.

---

### `LaneDetector`
Identical to `camera_inference.py`. Divides frame width into three equal zones and maps camera-left/center/right to rider-right/center/left respectively.

---

### `RiderActionRecommendation`
Extended version relative to `camera_inference.py`. Adjacent-lane vehicles further distinguish between close approaching (`< 15 m`) vs far approaching, returning a `BE_AWARE` or `MONITOR` action accordingly.

---

### `RearViewSafetyAssessment`
Same thresholds as `camera_inference.py`, with one addition: `calculate_tet()` (Time Exposed TTC) is also available for batch analysis:

```python
tet = rsa.calculate_tet(distance_m, ego_speed_ms, rear_speed_ms, ttc_threshold=1.5)
# Returns: seconds spent below ttc_threshold in this frame
```

---

### `RearSideUseCaseValidator`
Identical to `camera_inference.py`, with an additional `'total_vehicles_detected'` field in the scenario output and a `'vehicles_monitored'` scenario type for cases where vehicles are visible but not critical.

---

### `DetectionLogger`
Writes per-frame detection + safety data to a CSV file for offline analysis.

```python
logger = DetectionLogger("detections.csv")
logger.log_detections(
    frame_number=42,
    detections=detections,
    timestamp_s=1.4,
    safety_assessments=assessments,
    scenario_validation=scenario
)
logger.flush()   # periodically
logger.close()   # at end
```

**CSV columns:**

| Column | Description |
|---|---|
| `frame_number` | Frame index |
| `track_id` | ByteTracker persistent ID |
| `vehicle_class` | Fine-grained class (UVH-26) |
| `confidence` | Classifier confidence |
| `distance_m` | Estimated distance (m) |
| `speed_kmh` | Estimated approach speed (km/h) |
| `motion_state` | `approaching` / `stable` / `receding` |
| `bbox_x1/y1/x2/y2` | Bounding box pixels |
| `classical_depth` | Pure geometric depth (m) |
| `ml_depth` | ML model depth sample (m) |
| `safety_level` | `CRITICAL` / `WARNING` / `CAUTION` / `SAFE` / `INFO` |
| `alert_type` | `collision_imminent` / `distance_warning` / etc. |
| `ttc_s` | Time to Collision (s) |
| `mttc_s` | Modified TTC (s) |
| `pet_s` | Post Encroachment Time (s) |
| `drac_ms2` | Deceleration Rate to Avoid Collision (m/s²) |
| `scenario_type` | Scenario classification for the frame |

---

### `VideoDetector` (Main Class)

The video-mode equivalent of `CameraVehicleDetector`. Key differences:

```python
detector = VideoDetector(
    device='cuda',
    zoedepth_interval=30,      # ML depth refresh interval (frames)
    correction_alpha=0.3,      # Initial EMA alpha for correction factor
    learnable_alpha=True,      # Adapt alpha based on relative error
    alpha_lr=0.05,             # Learning rate for alpha updates
    classical_depth_weight=0.80,  # Blend weight for classical vs corrected depth
    fps=30
)
```

#### Hybrid Depth Pipeline

**Classical estimation** (every frame):

```python
z_ground, c_ground = detector._estimate_ground_plane_depth(bbox, frame_h, y_horizon)
z_size,   c_size   = detector._estimate_size_depth(bbox_h, class_name)
z_motion, c_motion = detector._estimate_motion_depth(track_id, bbox, ts, frame_shape)

# Weighted fusion
Z_geo = 0.55*z_ground + 0.30*z_size + 0.15*z_motion  (normalized weights)
```

**ML correction** (async, every `zoedepth_interval` frames):

```python
Z_final = classical_weight * Z_classical + (1 - classical_weight) * Z_classical * C_correction
```

where `C_correction` is an EMA-smoothed ratio `Z_ML / Z_classical` clamped to [0.75, 1.35].

**Learnable alpha:** When `learnable_alpha=True`, the EMA smoothing rate adapts based on relative error between geometric and ML depths. High disagreement → faster adaptation (larger alpha).

#### Multi-Factor Size Confidence

```python
conf = detector.calculate_size_confidence_multifactor(bbox, class_name, frame_shape)
```

Three factors:
1. **Pixel height** — `< 30 px` → 0.20, `30–100 px` → ramping, `100–300 px` → 0.95, `> 300 px` → 0.85
2. **Aspect ratio** — penalizes boxes with unusual width/height ratios vs expected vehicle geometry
3. **Frame position** — penalizes boxes cut off at frame edges (likely partially occluded)

#### Motion Parallax Depth

```python
z_motion, conf = detector._estimate_motion_depth(track_id, bbox, timestamp_s, frame_shape)
```

Uses inter-frame centroid displacement as a proxy for parallax. Blends with a scale-anchored estimate from the previous frame's depth:

```
Z = 0.65 × Z_parallax + 0.35 × (Z_prev × bbox_h_prev / bbox_h_current)
```

Confidence is proportional to centroid displacement (small displacement → unreliable).

---

## Async ML Depth Backends

`load_depth_with_fallback()` tries backends in priority order:

1. `da2_kitti_metric.onnx` (ONNX, fastest)
2. `da2_kitti_metric.pt` (PyTorch)
3. `midas_kitti_metric.onnx` (ONNX fallback)
4. Generic `trt_fp16`, `onnx`, `pytorch` backends

```python
depth_model = load_depth_with_fallback(device='cuda', interval=30)
```

If no backend loads, the system falls back to classical-only depth with a warning.

---

## Depth Sampling from ML Map

```python
z_ml, conf, method = detector._sample_ml_depth_for_bbox(depth_map, bbox)
```

Samples from a **ground-contact strip** (bottom 5% of the bounding box height), taking the 20th percentile of valid depth values. Falls back to the center-bottom pixel if fewer than 10 valid samples exist.

---

## Configuration Constants

```python
FOCAL_LENGTH        = 1000    # pixels (approximate rear camera focal length)
MOUNTING_HEIGHT_M   = 1.1     # camera mounting height (meters)
DEPTH_CLIP_MIN_M    = 0.5     # minimum reported depth
DEPTH_CLIP_MAX_M    = 25.0    # maximum reported depth

# SSM thresholds
TTC_CRITICAL     = 1.0   # seconds
TTC_WARNING      = 1.5   # seconds
DRAC_CRITICAL    = 3.35  # m/s²
DRAC_WARNING     = 2.0   # m/s²
DISTANCE_CRITICAL = 10.0 # meters
DISTANCE_WARNING  = 15.0 # meters
```

**Vehicle physical dimensions** (used for size-based depth) are defined in `VEHICLE_DIMENSIONS` as `(height_m, width_m, height_uncertainty_m, typical_rear_aspect_ratio)` for all 18 UVH-26 categories.

---

## Performance Notes

| Setting | Effect |
|---|---|
| `--depth-interval 30` | ML runs at 1 Hz (30 fps) — default |
| `--depth-interval 15` | ML runs at 2 Hz — slightly better accuracy |
| `--depth-interval 5` | ML runs at 6 Hz — highest accuracy, lower FPS |
| `--classical-weight 1.0` | Pure classical depth, fastest |
| `--classical-weight 0.0` | Pure ML-corrected depth, most accurate |

Classifier caching (`_cls_interval=5`) reduces classification overhead by ~88% with less than 0.4% accuracy drop.
