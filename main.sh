#!/bin/bash
#
# Main CNN Launcher - Easy access to all features with GPU Profile Selection
#

CNN_DIR="/home/atul/Desktop/atul/rear_view_adas_monocular/CNN"

# Prefer currently activated venv; fallback to project-local .venv.
if [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -x "$CNN_DIR/.venv/bin/python" ]; then
    PYTHON="$CNN_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

echo "Using Python: $PYTHON"
export PYTHONPATH="$CNN_DIR:$PYTHONPATH"

clear
echo "================================================================"
echo "  REAR-VIEW ADAS - MAIN LAUNCHER"
echo "================================================================"
echo ""

# ── GPU Profile Selection ────────────────────────────────────────────────────
echo "Select GPU Profile:"
echo ""
echo "  1. RTX A6000        (Full Performance)"
echo "  2. Jetson Nano      (Restricted - 8 GB memory)"
echo "  3. Jetson Nano      (Power Save - 7 W)"
echo ""
read -p "Enter GPU profile [1-3, default: 1]: " gpu_choice
gpu_choice=${gpu_choice:-1}

case $gpu_choice in
    1)
        GPU_PROFILE="a6000_full"
        echo "Selected: RTX A6000 Full Performance"
        ;;
    2)
        GPU_PROFILE="jetson_nano_restricted"
        echo "Selected: Jetson Nano Restricted (8 GB)"
        ;;
    3)
        GPU_PROFILE="jetson_nano_power_save"
        echo "Selected: Jetson Nano Power Save"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "================================================================"
echo "Select mode:"
echo ""
echo "  1. Camera Detection   (Real-time inference on live camera)"
echo "  2. Video Processing   (Offline inference on a video file)"
echo "  3. Train Models       (Fine-tune classifier or depth model)"
echo "  4. Exit"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in

    # ── MODE 1: Camera Detection ─────────────────────────────────────────────
    1)
        echo ""
        echo "Camera Detection  |  GPU Profile: $GPU_PROFILE"
        echo ""
        echo "  1. USB / V4L2 Camera"
        echo "  2. Test Video (use a local video file as input)"
        echo ""
        read -p "Select camera source [1-2, default: 1]: " cam_choice
        cam_choice=${cam_choice:-1}

        case $cam_choice in
            1)
                cam_id=4
                echo ""
                echo "Starting camera inference on camera $cam_id ..."
                cd "$CNN_DIR"
                $PYTHON inference/camera_inference.py \
                    --profile "$GPU_PROFILE" \
                    --camera "$cam_id" \
                    --save "detection_output_camera.mp4" \
                    -v
                ;;
            2)
                read -p "Enter video file path: " video_path
                if [ ! -f "$video_path" ]; then
                    echo "File not found: $video_path"
                    exit 1
                fi
                echo ""
                echo "Starting camera_inference on: $video_path ..."
                cd "$CNN_DIR"
                $PYTHON inference/camera_inference.py \
                    --profile "$GPU_PROFILE" \
                    --camera "$video_path" \
                    --save "${video_path%.*}_camera_result.mp4" \
                    -v
                ;;
            *)
                echo "Invalid choice. Exiting."
                exit 1
                ;;
        esac
        ;;

    # ── MODE 2: Video Processing ─────────────────────────────────────────────
    2)
        echo ""
        read -p "Enter input video path: " video_path
        if [ ! -f "$video_path" ]; then
            echo "File not found: $video_path"
            exit 1
        fi

        output="${video_path%.*}_detected.mp4"
        echo ""
        echo "Video Processing  |  GPU Profile: $GPU_PROFILE"
        echo "  Input : $video_path"
        echo "  Output: $output"
        echo ""
        echo "Starting video_inference ..."
        echo ""
        cd "$CNN_DIR"
        $PYTHON inference/video_inference.py \
            --input  "$video_path" \
            --output "$output"
        ;;

    # ── MODE 3: Train Models ──────────────────────────────────────────────────
    3)
        echo ""
        echo "Starting training ..."
        echo ""
        cd "$CNN_DIR/training"
        $PYTHON train_classifier.py
        ;;

    # ── MODE 4: Exit ──────────────────────────────────────────────────────────
    4)
        echo "Goodbye!"
        exit 0
        ;;

    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
