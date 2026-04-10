# Workspace Overview

## Summary

- **Total projects**: 50
- **Trainable projects**: 17
- **Projects using model registry**: 17
- **Dataset configs**: 17
  - Auto-download capable: 9
  - Manual download required: 8

## Projects by Task Category

### Detection (11)

| # | Project | Framework | Trainable |
|---|---------|-----------|-----------|
| 3 | Face Detection (YOLO) | Ultralytics YOLO | Yes |
| 12 | Real-Time Object Detection (YOLO) | Ultralytics YOLO | Yes |
| 13 | Sudoku Solver (YOLO-enhanced) | Ultralytics YOLO | Yes |
| 16 | Car Detection (YOLO) | Ultralytics YOLO | Yes |
| 18 | Ball Tracking (YOLO) | Ultralytics YOLO | Yes |
| 24 | Custom Object Detection (YOLO) | Ultralytics YOLO | Yes |
| 37 | Number Plate Detection | Ultralytics YOLO | Yes |
| 46 | Face Detection – Haar→YOLO | Ultralytics YOLO | Yes |
| 47 | Face Mask Detection (YOLO) | Ultralytics YOLO | Yes |
| 48 | Face Attributes (YOLO + Torch) | Ultralytics YOLO | Yes |
| 49 | Text Detection (YOLO) | Ultralytics YOLO | Yes |

### Pose Estimation (6)

| # | Project | Framework | Trainable |
|---|---------|-----------|-----------|
| 4 | Facial Landmarks (YOLO-Pose) | Ultralytics YOLO-Pose | Yes |
| 5 | Finger Counter (YOLO-Pose) | Ultralytics YOLO-Pose | — |
| 6 | Hand Tracking – Volume Control (YOLO-Pose) | Ultralytics YOLO-Pose | Yes |
| 10 | Pose Detector (YOLO-Pose) | Ultralytics YOLO-Pose | Yes |
| 17 | Blink Detection (YOLO-Pose) | Ultralytics YOLO-Pose | Yes |
| 21 | Volume Controller (YOLO-Pose) | Ultralytics YOLO-Pose | Yes |

### Segmentation (1)

| # | Project | Framework | Trainable |
|---|---------|-----------|-----------|
| 41 | Image Segmentation (YOLO-Seg) | Ultralytics YOLO-Seg | Yes |

### Utility / OpenCV (32)

| # | Project | Framework | Trainable |
|---|---------|-----------|-----------|
| 1 | Angle Detector (v2) | OpenCV | — |
| 2 | Document Scanner (v2) | OpenCV | — |
| 7 | Object Size Detector (v2) | OpenCV | — |
| 8 | OMR Evaluator (v2) | OpenCV | — |
| 9 | Real-Time Painter (v2) | OpenCV | — |
| 11 | QR Reader (v2) | OpenCV | — |
| 14 | Warp Perspective (v2) | OpenCV | — |
| 15 | Image Cartoonifier (v2) | OpenCV | — |
| 19 | Grayscale Converter (v2) | OpenCV | — |
| 20 | Image Finder (v2) | OpenCV | — |
| 22 | Color Picker (v2) | OpenCV | — |
| 23 | Crop & Resize (v2) | OpenCV | — |
| 25 | Object Measurement (v2) | OpenCV | — |
| 26 | Color Detection (v2) | OpenCV | — |
| 27 | Shape Detection (v2) | OpenCV | — |
| 28 | Watermarking (v2) | OpenCV | — |
| 29 | Virtual Pen (v2) | OpenCV | — |
| 30 | Contrast Enhancement Color (v2) | OpenCV | — |
| 31 | Contrast Enhancement Gray (v2) | OpenCV | — |
| 32 | Coin Lines (v2) | OpenCV | — |
| 33 | Image Blurring (v2) | OpenCV | — |
| 34 | Motion Blur (v2) | OpenCV | — |
| 35 | Image Sharpening (v2) | OpenCV | — |
| 36 | Thresholding Techniques (v2) | OpenCV | — |
| 38 | Pencil Sketch (v2) | OpenCV | — |
| 39 | Noise Removal (v2) | OpenCV | — |
| 40 | Non-Photorealistic Rendering (v2) | OpenCV | — |
| 42 | Image Resizing (v2) | OpenCV | — |
| 43 | Cartoon Effect (v2) | OpenCV | — |
| 44 | Image Joining (v2) | OpenCV | — |
| 45 | Click Detection (v2) | OpenCV | — |
| 50 | Video Reverse (v2) | OpenCV | — |

## Dataset Status

| Project Key | Method | Auto-Download |
|-------------|--------|---------------|
| ball_tracking | roboflow | Yes |
| blink_detection | manual | No |
| car_detection | manual | No |
| custom_object_detection | ultralytics | Yes |
| face_attributes | manual | No |
| face_detection | http | Yes |
| face_detection_haar | http | Yes |
| face_mask_detection | kaggle | Yes |
| facial_landmarks | manual | No |
| finger_counter | manual | No |
| hand_tracking | manual | No |
| image_segmentation | manual | No |
| object_detection | http | Yes |
| pose_detector | http | Yes |
| sudoku_solver | http | Yes |
| text_detection | http | Yes |
| volume_controller | manual | No |

## Commands Index

| Task | Command |
|------|---------|
| Smoke test | `python scripts/smoke_3b3.py` |
| CI sanity (5 checks) | `python scripts/ci_sanity.py --verbose` |
| Performance benchmark | `python benchmarks/run_all.py` |
| Accuracy evaluation | `python -m benchmarks.evaluate_accuracy` |
| Download all datasets | `python -m utils.data_downloader --all` |
| Download one dataset | `python -m utils.data_downloader --project <key>` |
| List registered models | `python -m models.registry list` |
| Run a project | `python -m core.runner <name> --source 0 --import-all` |
| List all projects | `python -m core.runner --list --import-all` |

