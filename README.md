# Rear-View ADAS — Monocular Vision System

Real-time rear-side rider assistance for two-wheelers using a single rear-mounted RGB camera. Detects, classifies, and tracks surrounding vehicles; estimates metric depth; assesses collision risk via surrogate safety measures; and generates lane-aware rider recommendations — all at 27–54 FPS on embedded hardware.

---

## Features

- **YOLOv11n detection** + **UVH-26 fine-grained vehicle classifier** (94.2% Top-1 accuracy, 15 Indian traffic categories)
- **Hybrid monocular depth estimation** — ground-plane projection + size-based ranging + motion parallax, periodically corrected by Depth Anything V2 / MiDaS (MAE 1.08 m overall)
- **ByteTrack multi-object tracking** with IoU fallback (61.7% fewer ID switches vs IoU-only)
- **Dynamic horizon estimation** — adapts to camera suspension pitch in real time
- **Lane-aware surrogate safety measures** — TTC, MTTC, PET, DRAC conditioned on lane assignment
- **Rider action recommendations** — emergency braking, deceleration, monitoring
- **Multi-platform support** — RTX A6000, Jetson Orin NX, Jetson Nano, Raspberry Pi 5
- **Intel RealSense D455** support for metric ground-truth depth

---

## Performance

| Platform | FPS | Depth MAE | Power |
|---|---|---|---|
| RTX A6000 | 54.1 | 1.08 m | 48 W |
| Jetson Orin NX | 27.0 | 1.08 m | 6.2 W |
| Jetson Nano (TRT INT8) | 21.4 | 1.18 m | 2.7 W |
| Raspberry Pi 5 | 16–20 | 1.32 m | 5.0 W |

Safety alert TPR: **93.3%** | FPR: **6.7%** (lane-aware filtering reduces FPR 2.6× vs proximity-only)

---

## Repository Structure

```
CNN/
├── inference/                    # Core inference scripts
│   ├── camera_inference.py       # Live camera / real-time ADAS
│   ├── video_inference.py        # Offline video processing + CSV logging
│   ├── byte_tracker.py           # ByteTrack multi-object tracker
│   ├── jetson_depth_lite.py      # Async depth backend (DA2 / MiDaS)
│   ├── gpu_config.py             # GPU profile manager
│   ├── model_optimizer.py        # TensorRT / quantization helpers
│   ├── web_server.py             # Flask web interface
│   └── templates/index.html      # Web UI
│
├── scripts/                      # Training and calibration utilities
│   ├── zoedepth_loader.py        # ZoeDepth model loader
│   ├── train_depth_kitti.py      # Depth model fine-tuning
│   ├── calibrate_camera.py       # Camera intrinsic calibration
│   ├── download_zoedepth.py      # Download ZoeDepth weights
│   └── ...
│
├── models/                       # Model weights (NOT in git — download separately)
│   ├── classifier/weights/best.pt        # UVH-26 fine-grained classifier
│   ├── depth_lite/                       # Lightweight KITTI-metric depth models
│   │   ├── da2_kitti_metric.onnx
│   │   └── midas_kitti_metric.onnx
│   ├── depth_anything_v2/                # Depth Anything V2 base weights
│   └── depth_anything_v2_finetuned/      # Fine-tuned DA2 weights
│
├── Documents/                    # Technical documentation
│   ├── CAMERA_INFERENCE.md       # camera_inference.py reference
│   ├── VIDEO_INFERENCE.md        # video_inference.py reference
│   └── MODEL_TRAINING_EVALUATION.md
│
├── requirements.txt              # Python dependencies
├── main.sh                       # Convenience launcher
├── Dockerfile                    # Docker deployment
└── docker-compose.web.yml        # Docker Compose for web interface
```

> **Note:** `dataset/`, `testing_data/`, model weight files (`*.pt`, `*.onnx`, `*.safetensors`), and personal documents are excluded from git via `.gitignore`.

---

## Prerequisites

- Python 3.9 or 3.10
- CUDA 11.8+ (for GPU inference; CPU mode also supported)
- PyTorch 2.1+
- OpenCV 4.8+
- 8 GB RAM minimum (16 GB recommended for full pipeline)
- For Jetson: JetPack 5.1+, CUDA Toolkit pre-installed

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Atulkumar0804/Rear-View-ADAS.git
cd Rear-View-ADAS/CNN
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs: `ultralytics`, `torch`, `torchvision`, `opencv-python`, `numpy`, `transformers`, `accelerate`, `flask`, and optional `pyrealsense2`.

### 4. Download model weights

Model weights are not stored in git due to size. Download them using the provided scripts or manually place them in the `models/` directory.

#### Option A — Automatic download scripts

```bash
# Download ZoeDepth / DA2 base weights
python scripts/download_zoedepth.py

# Download KITTI depth dataset (for retraining only)
# bash scripts/download_kitti_depth.sh
```

#### Option B — Manual placement

Download the following files and place them at the exact paths shown:

| File | Path in repo | Source |
|---|---|---|
| `yolo11n.pt` | `yolo11n.pt` (root) | Auto-downloaded by Ultralytics on first run |
| `best.pt` (UVH-26 classifier) | `models/classifier/weights/best.pt` | See GitHub Releases |
| `da2_kitti_metric.onnx` | `models/depth_lite/da2_kitti_metric.onnx` | See GitHub Releases |
| `midas_kitti_metric.onnx` | `models/depth_lite/midas_kitti_metric.onnx` | See GitHub Releases |
| DA2 base model | `models/depth_anything_v2/` | HuggingFace: `depth-anything/Depth-Anything-V2-Small-hf` |

> **YOLO weights** are downloaded automatically the first time you run any inference script if `yolo11n.pt` is not found.

#### Option C — From GitHub Releases

Check the [Releases page](https://github.com/Atulkumar0804/Rear-View-ADAS/releases) for pre-packaged model archives.

### 5. Verify setup

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

---

## Running — Live Camera (`camera_inference.py`)

```bash
# Default USB camera (index 0)
python inference/camera_inference.py

# Specific camera index
python inference/camera_inference.py --camera 1

# Intel RealSense D455
python inference/camera_inference.py --realsense

# Save output video
python inference/camera_inference.py --camera 0 --save output.mp4

# Headless (no display window)
python inference/camera_inference.py --camera 0 --no-display --save output.mp4

# Jetson power-save profile
python inference/camera_inference.py --profile jetson_nano_power_save
```

Press `q` or `ESC` to stop. Session FPS stats are printed on exit.

See [Documents/CAMERA_INFERENCE.md](Documents/CAMERA_INFERENCE.md) for full argument reference and class documentation.

---

## Running — Video File (`video_inference.py`)

```bash
# Basic: process video, save result
python inference/video_inference.py --input video.mp4 --output result.mp4

# With CSV telemetry log
python inference/video_inference.py --input video.mp4 --output result.mp4 --log metrics.csv

# Set ego vehicle speed (km/h) for accurate SSM calculations
python inference/video_inference.py --input video.mp4 --output result.mp4 --ego-speed 40.0

# Tune depth pipeline
python inference/video_inference.py \
    --input video.mp4 \
    --output result.mp4 \
    --depth-interval 15 \
    --classical-weight 0.75
```

See [Documents/VIDEO_INFERENCE.md](Documents/VIDEO_INFERENCE.md) for full argument reference, hybrid depth pipeline details, and CSV schema.

---

## Using the Main Launcher

```bash
bash main.sh
```

`main.sh` provides an interactive menu to select GPU profile and inference mode (camera or video).

---

## Docker Deployment

```bash
# Build and run with Docker
docker build -t rear-adas .
docker run --gpus all -p 5000:5000 rear-adas

# With Docker Compose (web interface)
docker-compose -f docker-compose.web.yml up
```

The web interface streams the annotated ADAS output to any browser on `http://<host-ip>:5000`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ADAS_DEVICE` | `cuda` | Override inference device |
| `ADAS_PROFILE` | `a6000_full` | GPU profile for optimization |
| `ADAS_LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Documentation

| Document | Description |
|---|---|
| [Documents/CAMERA_INFERENCE.md](Documents/CAMERA_INFERENCE.md) | Full reference for `camera_inference.py` — all classes, arguments, HUD layout |
| [Documents/VIDEO_INFERENCE.md](Documents/VIDEO_INFERENCE.md) | Full reference for `video_inference.py` — hybrid depth, CSV schema, tuning |
| [Documents/MODEL_TRAINING_EVALUATION.md](Documents/MODEL_TRAINING_EVALUATION.md) | UVH-26 and depth model training details |
| [inference/GPU_CONFIG_GUIDE.md](inference/GPU_CONFIG_GUIDE.md) | GPU profile selection and TensorRT setup |

---

## Scripts Reference

| Script | Purpose |
|---|---|
| `scripts/calibrate_camera.py` | Checkerboard camera calibration (intrinsics) |
| `scripts/calibrate_distance_interactive.py` | Interactive distance calibration with known objects |
| `scripts/train_depth_kitti.py` | Fine-tune DA2 depth model on KITTI |
| `scripts/train_depth_da2_kitti.py` | Alternative DA2 training script |
| `scripts/download_zoedepth.py` | Download ZoeDepth model weights |
| `scripts/export_jetson.py` | Export models to TensorRT for Jetson |
| `scripts/DEPTH_ACCURACY_EVALUATION.py` | Evaluate depth MAE vs ground truth |
| `scripts/zoedepth_loader.py` | ZoeDepth / DA2 model loading utility |

---

## Troubleshooting

**No CUDA device found:**
```bash
python -c "import torch; print(torch.cuda.device_count())"
# If 0, install CUDA-enabled PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Camera not found:**
```bash
v4l2-ctl --list-devices        # Linux
# Try different camera index: --camera 1, --camera 2
```

**YOLO model not found:**
```bash
# yolo11n.pt auto-downloads on first run. If it fails, download manually:
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

**Classifier weights missing:**
```
FileNotFoundError: models/classifier/weights/best.pt
```
Download `best.pt` from the GitHub Releases page and place at `models/classifier/weights/best.pt`.

**ByteTracker import error:**
```
⚠️  ByteTracker not available, will fall back to IoU tracking
```
This is non-fatal. The system continues with IoU-based tracking (slightly lower ID stability).

**Low FPS on Jetson:**
```bash
# Switch to power-save profile
python inference/camera_inference.py --profile jetson_nano_restricted
# Or enable TensorRT (see inference/GPU_CONFIG_GUIDE.md)
python scripts/export_jetson.py
```

---

## License

This project is released for research and educational use. See `LICENSE` for details.

---

## Citation

If you use this system in your research, please cite:

```bibtex
@misc{singh2025rearviewadas,
  title  = {Real-Time Rear-Side Rider Assistance System for Two-Wheelers
             Using Hybrid Monocular Depth Estimation},
  author = {Singh, Atul Kumar},
  year   = {2025},
  url    = {https://github.com/Atulkumar0804/Rear-View-ADAS}
}
```
