# PHASE 0 — FULL REPOSITORY AUDIT REPORT

> **Generated**: Forensic-level audit of `Computer-Vision-Projects` repository  
> **Mode**: STRICT READ-ONLY — zero modifications to existing files  
> **Scope**: 50 project directories + root files

---

## 1. GLOBAL SUMMARY

| Metric | Value |
|---|---|
| Total Projects | 50 |
| Total Python Scripts (.py) | ~38 |
| Total Jupyter Notebooks (.ipynb) | ~20 |
| Total Model/Weight Files | 8 (2× .caffemodel, 1× .h5, 2× .dat, 2× .xml cascade, 18× haarcascade XMLs) |
| Total Dataset Assets (images/video) | ~900+ files (853 in Project 47 alone) |
| Dominant Framework | OpenCV (cv2) — used in 100% of projects |
| Secondary Frameworks | Mediapipe (4), Keras/TensorFlow (3), dlib (2), Caffe via cv2.dnn (2) |
| GPU/CUDA Usage | **NONE** — all pipelines are CPU-only |
| Requirements.txt Present | **NO** — zero projects have dependency manifests |
| Evaluation Metrics / Test Suites | **NONE** |
| CI/CD Configuration | **NONE** |
| Duplicate Projects | 2 confirmed pairs (6↔21, 14↔45) |
| Critical Bugs (code will crash) | 6 projects |

### Framework Distribution

| Classification | Count | Projects |
|---|---|---|
| `classical_opencv` | 38 | 1, 2, 7, 8, 9, 11, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 49, 50 |
| `mixed_opencv_mediapipe` | 4 | 5, 6, 10, 21 |
| `mixed_opencv_dlib` | 2 | 4, 17 |
| `caffe_based (cv2.dnn)` | 2 | 3, 12 |
| `deep_learning_keras` | 3 | 13, 47, 48 |
| `classical_opencv + pyzbar` | 1 | 11 |

### GPU / Compute Readiness

- **PyTorch**: Not used anywhere in the repository.
- **TensorFlow/Keras GPU**: Project 48 uses `tensorflow` + `keras`, but contains **no explicit GPU configuration** (`tf.config`, `tf.device`, `CUDA_VISIBLE_DEVICES`, etc.). Training ran on default device (likely CPU).
- **Project 47**: Uses standalone `keras` (not `tf.keras`) with VGG19 transfer learning — no GPU pinning.
- **Project 13**: Uses standalone `keras` for digit classification — no GPU config.
- **Mediapipe projects** (5, 6, 10, 21): CPU inference only.
- **Caffe DNN projects** (3, 12): `cv2.dnn` defaults to CPU backend.
- **Verdict**: The entire repository is CPU-bound. No project is GPU-ready without modification.

---

## 2. PROJECT-BY-PROJECT BREAKDOWN

---

### Project 1 — Real Time Angle Detector
**Path**: `CV Project 1 -  Real Time Angle Detector/`

| Attribute | Value |
|---|---|
| Files | `AngleDetector.py`, `readme.txt`, `test.jpg` |
| Libraries | `cv2`, `math` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `test.jpg` (included) |
| Entry Point | `AngleDetector.py` |

**Technical Notes**:
- Mouse-click-based angle measurement on a static image.
- Uses relative path `path = 'test.jpg'` — portable.
- No external dependencies beyond OpenCV.

**Issues**: None critical.

---

### Project 2 — Real Time Document Scanner
**Path**: `CV Project 2 - Real Time Document Scanner-fine/`

| Attribute | Value |
|---|---|
| Files | `docScanner.py`, `how2run.txt`, `helpFunctions/transform.py`, `helpFunctions/__init__.py`, 5× JPG |
| Libraries | `cv2`, `numpy`, `argparse`, `imutils`, `skimage.filters` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | 5 sample document images (included) |
| Entry Point | `docScanner.py` (argparse: `--image`) |

**Technical Notes**:
- Modular design with `helpFunctions/transform.py` (four_point_transform).
- Uses `threshold_local` from scikit-image for adaptive thresholding.
- Proper argparse CLI interface.

**Issues**: None critical.

---

### Project 3 — Real Time Face Detector Image
**Path**: `CV Project 3 - Real Time Face detector Image/`

| Attribute | Value |
|---|---|
| Files | `detect_faces.py`, `detect_faces_video.py`, `HOW2RUN.txt`, `iron_chic.jpg`, `res10_300x300_ssd_iter_140000.caffemodel` |
| Libraries | `cv2`, `numpy`, `argparse`, `imutils` |
| Framework | `caffe_based` (cv2.dnn.readNetFromCaffe) |
| Model Files | `res10_300x300_ssd_iter_140000.caffemodel` (**present**) |
| Dataset | `iron_chic.jpg` (included) |
| Entry Point | `detect_faces.py` / `detect_faces_video.py` |

**Technical Notes**:
- SSD face detection using Caffe model via `cv2.dnn`.
- Two scripts: static image and live video.

**Issues**:
- ⚠️ **MISSING FILE**: `deploy.prototxt.txt` referenced in HOW2RUN.txt and code is **NOT present** — both scripts will crash on `cv2.dnn.readNetFromCaffe()`.

---

### Project 4 — Facial Landmarking
**Path**: `CV Project 4 - Facial Landmarking/`

| Attribute | Value |
|---|---|
| Files | `facial_landmarking.py`, `HOW2RUN.txt`, `images/example.py`, 4× example JPGs, `shape_predictor_68_face_landmarks.dat` |
| Libraries | `cv2`, `numpy`, `argparse`, `imutils`, `dlib` |
| Framework | `classical_opencv + dlib` |
| Model Files | `shape_predictor_68_face_landmarks.dat` (**present**, duplicated in Project 17) |
| Dataset | 4 example images (included) |
| Entry Point | `facial_landmarking.py` (argparse: `--shape-predictor`, `--image`) |

**Technical Notes**:
- 68-point facial landmark detection using dlib's shape predictor.
- Proper argparse CLI interface.
- `images/example.py` is a throwaway test script (reads/blurs image) — not part of main pipeline.

**Issues**: 
- `images/example.py` is dead code.

---

### Project 5 — Finger Counter
**Path**: `CV Project 5 - fingerCounter/`

| Attribute | Value |
|---|---|
| Files | `fingerCount.py`, `HandDetectionModule.py`, `HOW2RUN.txt`, `FingerImages/` (6× JPG) |
| Libraries | `cv2`, `mediapipe`, `time`, `os` |
| Framework | `mixed_opencv_mediapipe` |
| Model Files | None (Mediapipe bundles models internally) |
| Dataset | `FingerImages/` — 6 finger overlay images (included) |
| Entry Point | `fingerCount.py` |

**Technical Notes**:
- Uses Mediapipe Hands for finger detection.
- `HandDetectionModule.py` is a reusable module with class `handDetector` (also has `__main__` block).

**Issues**:
- 🔴 **HARDCODED ABSOLUTE PATH**: `fingerCount.py` contains `r"C:\Users\Ashu.ASHUTOSH\PycharmProjects\fingerCounter\FingerImages"` — will fail on any other machine.

---

### Project 6 — Live HTM (Hand Tracking Module)
**Path**: `CV Project 6 - Live HTM/`

| Attribute | Value |
|---|---|
| Files | `HandTrackingModule.py`, `VolumeControl.py`, `.html` files, `index.html` |
| Libraries | `cv2`, `mediapipe`, `time`, `numpy`, `math`, `pycaw`, `comtypes` |
| Framework | `mixed_opencv_mediapipe` + `pycaw` |
| Model Files | None |
| Dataset | None (webcam) |
| Entry Point | `VolumeControl.py` |

**Technical Notes**:
- Hand-gesture-based volume controller using Mediapipe + pycaw.
- Contains `.html` exports (notebook conversion artifacts, non-functional).

**Issues**:
- 🔴 **EXACT DUPLICATE** of Project 21 (`CV Projects 21 - VolumeController/`).
- ⚠️ **Windows-only**: `pycaw` + `comtypes` are Windows audio APIs.
- No HOW2RUN.txt file.

---

### Project 7 — Real Time Object Size Detector
**Path**: `CV Project 7 - Real Time Object Size detector/`

| Attribute | Value |
|---|---|
| Files | `object_size.py`, `HOW2RUN.txt`, `images/` (3× PNG), `example_02.png` |
| Libraries | `cv2`, `numpy`, `argparse`, `imutils`, `scipy.spatial.distance` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | Example images (included) |
| Entry Point | `object_size.py` (argparse: `--image`, `--width`) |

**Technical Notes**:
- Measures real-world object sizes from images using a reference object width.
- Proper argparse CLI interface.

**Issues**: None critical.

---

### Project 8 — OMR Evaluator
**Path**: `CV Project 8 - OMR Evaluator/`

| Attribute | Value |
|---|---|
| Files | `OMRevaluator.py`, `HOW2RUN.txt`, `images/` (6× test images), 2× JPG |
| Libraries | `cv2`, `numpy`, `argparse`, `imutils` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | OMR sheet images (included) |
| Entry Point | `OMRevaluator.py` (argparse: `--image`) |

**Technical Notes**:
- Hardcoded answer key: `ANSWER_KEY = {0:1, 1:4, 2:0, 3:3, 4:1}`.
- Perspective-transforms OMR sheet, then checks bubbles via contour analysis.

**Issues**:
- Hardcoded 5-question answer key — not configurable.

---

### Project 9 — Real Time Painter of Camera Screen
**Path**: `CV Project 9 - Real Time Painter of Camera Screen/`

| Attribute | Value |
|---|---|
| Files | `Paint.py` |
| Libraries | `cv2`, `numpy`, `collections.deque` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | None (webcam) |
| Entry Point | `Paint.py` |

**Technical Notes**:
- Tracks blue-colored object via HSV filtering, draws on canvas.
- Uses deque for tracking trail.

**Issues**:
- No HOW2RUN.txt or README file.

---

### Project 10 — Live PoseDetector
**Path**: `CV Project 10 - Live PoseDetector/`

| Attribute | Value |
|---|---|
| Files | `main.py`, `poseDetector-I.py`, `HOW2RUN.txt` |
| Libraries | `cv2`, `mediapipe` |
| Framework | `mixed_opencv_mediapipe` |
| Model Files | None (Mediapipe bundles models) |
| Dataset | None (webcam) |
| Entry Point | `poseDetector-I.py` |

**Technical Notes**:
- Uses `mp.solutions.holistic` for full-body pose estimation.

**Issues**:
- ⚠️ **MISLEADING FILE**: `main.py` is a **PyCharm auto-generated boilerplate** (`def print_hi(name)`) — NOT the actual entry point.

---

### Project 11 — Live QR-Reader
**Path**: `CV Project 11 - Live QR-Reader/`

| Attribute | Value |
|---|---|
| Files | `QR_Reader.py`, `HOW2RUN.txt` |
| Libraries | `cv2`, `numpy`, `pyzbar` |
| Framework | `classical_opencv + pyzbar` |
| Model Files | None |
| Dataset | None |
| Entry Point | `QR_Reader.py` |

**Technical Notes**:
- Reads QR/barcodes from webcam feed using pyzbar.
- Logs decoded data to `myDataFile.txt`.

**Issues**:
- ⚠️ **MISSING FILE**: References `myDataFile.txt` for authorized data matching — file is not present.

---

### Project 12 — Real Time Object Detection
**Path**: `CV Project 12 - Real Time Object Detection-fine/`

| Attribute | Value |
|---|---|
| Files | `real_time_object_detection.py`, `HOW2RUN.txt`, `MobileNetSSD_deploy.caffemodel`, `MobileNetSSD_deploy.prototxt.txt`, 1× JPG |
| Libraries | `cv2`, `numpy`, `argparse`, `imutils` |
| Framework | `caffe_based` (cv2.dnn.readNetFromCaffe + MobileNetSSD) |
| Model Files | `MobileNetSSD_deploy.caffemodel` + `MobileNetSSD_deploy.prototxt.txt` (**both present**) |
| Dataset | Sample image (included) |
| Entry Point | `real_time_object_detection.py` (argparse: `--prototxt`, `--model`, `--confidence`) |

**Technical Notes**:
- Detects 20 PASCAL VOC classes in real-time using MobileNet-SSD.
- Proper argparse CLI with configurable confidence threshold.
- Well-structured code.

**Issues**: None critical.

---

### Project 13 — Real Time Sudoku Solver
**Path**: `CV Project 13 - Real Time Sudoku Solver/`

| Attribute | Value |
|---|---|
| Files | `main.py`, `Operator.py`, `Sudoku.py`, `SudokuSolver.py`, `digit_model.h5`, `HOW2RUN.txt`, `Test.gif` |
| Libraries | `cv2`, `numpy`, `keras` |
| Framework | `deep_learning_keras` |
| Model Files | `digit_model.h5` (**present**) |
| Dataset | `Test.gif` (demo) |
| Entry Point | `Sudoku.py` (**NOT main.py**) |

**Technical Notes**:
- Multi-file architecture: Sudoku.py (main pipeline), SudokuSolver.py (backtracking solver), digit_model.h5 (CNN digit classifier).
- Uses `from keras.models import load_model` (standalone keras, not `tf.keras`).

**Issues**:
- ⚠️ **MISLEADING FILE**: `main.py` is **PyCharm boilerplate** — real entry is `Sudoku.py`.
- ⚠️ **DEAD CODE**: `Operator.py` is a verbatim copy of Python's built-in `operator` module — completely unnecessary.
- 🔴 **DEPRECATED API**: Uses `classifier.predict_classes()` — removed in TensorFlow 2.6+.
- ⚠️ **LEGACY IMPORT**: `from keras.models import load_model` should be `from tensorflow.keras.models import load_model`.

---

### Project 14 — Click-detect on Image
**Path**: `CV Project 14 - click-detect on image/`

| Attribute | Value |
|---|---|
| Files | `Warp.py`, `how2run.txt`, `cards.jpg` |
| Libraries | `cv2`, `numpy` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `cards.jpg` (included) |
| Entry Point | `Warp.py` |

**Technical Notes**:
- Mouse-click perspective warp on card images.

**Issues**:
- 🔴 **BUG**: Final line uses `cv.destroyAllWindows()` but OpenCV is imported as `cv2` — **will crash** with `NameError`.
- ⚠️ **NEAR-DUPLICATE** of Project 45.

---

### Project 15 — Live Image Cartoonifier
**Path**: `CV Project 15 - Live Image Cartoonifier/`

| Attribute | Value |
|---|---|
| Files | `cartoon.py`, `how2run.txt`, `v.jpeg` |
| Libraries | `cv2`, `numpy`, `skimage` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `v.jpeg` (included) |
| Entry Point | `cartoon.py` |

**Issues**:
- 🔴 **BUG**: Indentation error — `cv2.imshow("Cartoon",cartoon)` appears to have tab/space mixing inside the while loop — may cause `IndentationError`.

---

### Project 16 — Live Car Detection
**Path**: `CV Project 16 -Live Car-Detection/`

| Attribute | Value |
|---|---|
| Files | `Vehicles_detection.py`, `how to run.txt`, `carx.xml`, 3× video files (mp4/avi) |
| Libraries | `cv2`, `argparse` |
| Framework | `classical_opencv` (Haarcascade) |
| Model Files | `carx.xml` (cascade classifier, **present**) |
| Dataset | 3 video files (included) |
| Entry Point | `Vehicles_detection.py` |

**Issues**:
- Uses deprecated Haarcascade approach for vehicle detection.

---

### Project 17 — Blink Detection
**Path**: `CV Project 17 - Blink Detection/`

| Attribute | Value |
|---|---|
| Files | `blink_detector.py`, `Howtorun.txt`, `shape_predictor_68_face_landmarks.dat` |
| Libraries | `cv2`, `numpy`, `argparse`, `imutils`, `dlib`, `scipy.spatial.distance` |
| Framework | `classical_opencv + dlib` |
| Model Files | `shape_predictor_68_face_landmarks.dat` (**present**, duplicate of Project 4) |
| Dataset | None (webcam) |
| Entry Point | `blink_detector.py` (argparse: `--shape-predictor`) |

**Technical Notes**:
- Eye Aspect Ratio (EAR) algorithm for real-time blink detection.
- Well-structured code with proper argparse.

**Issues**:
- ⚠️ **DUPLICATE MODEL**: `shape_predictor_68_face_landmarks.dat` (~100MB) is also in Project 4 — wasted disk space.

---

### Project 18 — Live Ball Tracking
**Path**: `CV Project 18 - Live Ball Tracking/`

| Attribute | Value |
|---|---|
| Files | `ballTracking.py`, `ball_tracking_example.mp4`, `How to run.txt` |
| Libraries | `cv2`, `numpy`, `argparse`, `imutils`, `collections.deque` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `ball_tracking_example.mp4` (included) |
| Entry Point | `ballTracking.py` |

**Issues**:
- 🔴 **CORRUPTED CODE**: Lines ~35-38 contain garbled text mixed into Python code (`collections i`, `imutils.video`, random fragments) — **script will not parse**.

---

### Project 19 — GrayScaleConverter
**Path**: `CV Projects 19 - GrayScaleConverter/`

| Attribute | Value |
|---|---|
| Files | `GrayscaleConverter.py`, `HOW2RUN.txt` |
| Libraries | `cv2`, `numpy` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | None (webcam) |
| Entry Point | `GrayscaleConverter.py` |

**Technical Notes**:
- HSV trackbar-based color space converter. Despite the name, it's actually a **color detection** tool, not just grayscale conversion.

**Issues**:
- Misleading project name.

---

### Project 20 — Image Finder
**Path**: `CV Projects 20 - image_finder/`

| Attribute | Value |
|---|---|
| Files | `finder.py`, `images/waldo.jpg`, `images/WaldoBeach.jpg` |
| Libraries | `cv2`, `numpy` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | Waldo images (included) |
| Entry Point | `finder.py` |

**Technical Notes**:
- Template matching (`cv2.matchTemplate`) to find Waldo in a scene.

**Issues**: None critical.

---

### Project 21 — VolumeController
**Path**: `CV Projects 21 - VolumeController/`

| Attribute | Value |
|---|---|
| Files | `VolumeControl.py`, `HandTrackingModule.py`, `HOW2RUN.txt`, `.html` files, `index.html` |
| Libraries | `cv2`, `mediapipe`, `time`, `numpy`, `math`, `pycaw`, `comtypes` |
| Framework | `mixed_opencv_mediapipe` + `pycaw` |
| Model Files | None |
| Dataset | None (webcam) |
| Entry Point | `VolumeControl.py` |

**Issues**:
- 🔴 **EXACT DUPLICATE** of Project 6 (`CV Project 6 - Live HTM/`). Identical source files.
- ⚠️ **Windows-only**: `pycaw` + `comtypes`.

---

### Project 22 — Live Color Picker
**Path**: `CV Projects 22 - Live Color Picker/`

| Attribute | Value |
|---|---|
| Files | `ColorPicker.py`, `How 2 Run.txt`, `test.png` |
| Libraries | `cv2`, `numpy` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `test.png` (included) |
| Entry Point | `ColorPicker.py` |

**Technical Notes**:
- HSV color picker with trackbars.
- Contains `stackImages()` utility function (duplicated in Projects 27, 44).

**Issues**: None critical.

---

### Project 23 — Crop Resize Image using OpenCV
**Path**: `CV Projects 23 - Crop Resize Image using OpenCV/`

| Attribute | Value |
|---|---|
| Files | `Crop_Resize_Images.py`, `How 2 Run.txt`, `road.jpg` |
| Libraries | `cv2` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `road.jpg` (included) |
| Entry Point | `Crop_Resize_Images.py` |

**Issues**: None critical. Simple tutorial-level project.

---

### Project 24 — Custom Object Detection
**Path**: `CV Projects 24 - Custom Object Detection/`

| Attribute | Value |
|---|---|
| Files | `createData.py`, `objectDetectoin.py` (typo), `How 2 Run.txt`, `haarcascades/` (18× XML) |
| Libraries | `cv2`, `os`, `time` |
| Framework | `classical_opencv` (Haarcascade) |
| Model Files | 18 haarcascade XML classifiers (**present**) |
| Dataset | None (webcam capture) |
| Entry Point | `objectDetectoin.py` |

**Technical Notes**:
- `createData.py` captures training data from webcam.
- `objectDetectoin.py` uses CascadeClassifier with runtime trackbar controls.

**Issues**:
- ⚠️ **FILENAME TYPO**: `objectDetectoin.py` (should be `objectDetection.py`).
- ⚠️ **HARDCODED PATH**: `createData.py` references `'data/images'` which doesn't exist in the repo.

---

### Project 25 — Real Time Object Measurement
**Path**: `CV Projects 25 - Real Time Object Measurement/`

| Attribute | Value |
|---|---|
| Files | `ObjectMeasurement.py`, `utlis.py`, `How 2 Run.txt`, `ReadMe.md`, `1.jpg` |
| Libraries | `cv2`, `numpy` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `1.jpg` (included) |
| Entry Point | `ObjectMeasurement.py` |

**Technical Notes**:
- Modular design: `utlis.py` contains `getContours`, `reorder`, `warpImg`, `findDis` helpers.
- Uses A4 paper as size reference for real-world measurements.

**Issues**:
- ⚠️ `ReadMe.md` is **empty** (whitespace only).

---

### Project 26 — Real Time Color Detection
**Path**: `CV Projects 26 - Real Time Color Detection/`

| Attribute | Value |
|---|---|
| Files | `LiveHSVAdjustor.py`, `How 2 Run.txt` |
| Libraries | `cv2`, `numpy` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | None (webcam) |
| Entry Point | `LiveHSVAdjustor.py` |

**Issues**:
- 🔴 **BUG**: Code contains `cv2.cvtColor(img, cv2.COLOR_BGR2Result)` — `COLOR_BGR2Result` is **not a valid OpenCV constant** and will raise `AttributeError`.

---

### Project 27 — Real Time Shape Detection
**Path**: `CV Projects 27 - Real Time Shape Detection/`

| Attribute | Value |
|---|---|
| Files | `RealTime_Shape_Detection_Contours.py`, `How 2 Run.txt` |
| Libraries | `cv2`, `numpy` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | None (webcam) |
| Entry Point | `RealTime_Shape_Detection_Contours.py` |

**Technical Notes**:
- Contour-based shape detection with trackbar parameters.
- Contains `stackImages()` utility (also in Projects 22, 44).

**Issues**: None critical.

---

### Project 28 — Water Marking on Image using OpenCV
**Path**: `CV Projects 28 - Water Marking on Image using OpenCV/`

| Attribute | Value |
|---|---|
| Files | `watermarking on images using OpenCV.ipynb`, `How 2 Run.txt`, `imugi.jpg`, `watermark.png` |
| Libraries | `cv2`, `numpy`, `matplotlib` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `imugi.jpg`, `watermark.png` (included) |
| Entry Point | Notebook |

**Issues**: None critical.

---

### Project 29 — Live Virtual Pen
**Path**: `CV Projects 29 - Live Virtual Pen/`

| Attribute | Value |
|---|---|
| Files | `VPen.ipynb`, `finding HSV values.ipynb`, `How 2 Run.txt`, `hsv_value.npy`, demo video + screenshots |
| Libraries | `cv2`, `numpy`, `time` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `hsv_value.npy` (HSV calibration data, included) |
| Entry Point | `VPen.ipynb` |

**Issues**:
- 🔴 **BUG**: `VPen.ipynb` uses `cv2.findcnt()` which **does not exist** — should be `cv2.findContours()`. Will crash at runtime.

---

### Project 30 — Contrast Enhancing of Color Images
**Path**: `CV projects 30 - Contrast enhancing of color images/`

| Attribute | Value |
|---|---|
| Files | `OpenCV (Contrast enhancing of color images).ipynb`, `How 2 Run.txt`, `yoaimo.png` |
| Libraries | `cv2`, `numpy`, `matplotlib` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `yoaimo.png` (included) |
| Entry Point | Notebook |

**Issues**: None critical.

---

### Project 31 — Contrast Enhancing of Gray Scale Image
**Path**: `CV Projects 31 - contrast enhancing of gray scale image using opencv/`

| Attribute | Value |
|---|---|
| Files | `Enhancing contrast of gray scale image (OpenCV).ipynb`, `How 2 Run.txt`, `yoaimo.png` |
| Libraries | `cv2`, `numpy` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `yoaimo.png` (included) |
| Entry Point | Notebook |

**Issues**: None critical.

---

### Project 32 — Draw Vertical Lines of Coin
**Path**: `CV Projects 32 - Draw vertical lines of coin/`

| Attribute | Value |
|---|---|
| Files | `Drawing vertical lines on coin.ipynb`, `draw vertical lines on coin1.ipynb`, `How 2 Run.txt`, `coins.jpg` |
| Libraries | `cv2`, `numpy`, `matplotlib` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `coins.jpg` (included) |
| Entry Point | Either notebook |

**Issues**:
- ⚠️ **DUPLICATE NOTEBOOKS**: Two nearly identical notebooks; second one has execution errors in saved output.

---

### Project 33 — Image Blurring using OpenCV
**Path**: `CV Projects 33 - Image Bluring using opencv/`

| Attribute | Value |
|---|---|
| Files | `image blurring (OpenCV).ipynb`, `How 2 Run.txt`, `yoaimo.png` |
| Libraries | `cv2`, `numpy`, `matplotlib` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `yoaimo.png` (included) |
| Entry Point | Notebook |

**Issues**: None critical. Folder name has typo ("Bluring" → "Blurring").

---

### Project 34 — Live Motion Blurring
**Path**: `CV Projects 34 - Live Motion Blurring/`

| Attribute | Value |
|---|---|
| Files | `motion blurring effect.ipynb`, `How 2 Run.txt`, `yoaimo.png` |
| Libraries | `cv2`, `numpy` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `yoaimo.png` (included) |
| Entry Point | Notebook |

**Issues**: None critical.

---

### Project 35 — Sharpening of Images using OpenCV
**Path**: `CV Projects 35 - Sharpning of images using OpenCV/`

| Attribute | Value |
|---|---|
| Files | `sharpening of images using opencv.ipynb`, `How 2 Run.txt`, `yoaimo.png` |
| Libraries | `cv2`, `numpy` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `yoaimo.png` (included) |
| Entry Point | Notebook |

**Issues**: Folder name has typo ("Sharpning" → "Sharpening").

---

### Project 36 — Thresholding Techniques
**Path**: `CV Projects 36 - Thresholding Techiques/`

| Attribute | Value |
|---|---|
| Files | `Thresholding techniques (OpenCV).ipynb`, `How 2 Run.txt` |
| Libraries | `cv2`, `numpy`, `matplotlib` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | **NONE** — no sample image included |
| Entry Point | Notebook |

**Issues**:
- ⚠️ **MISSING DATASET**: No sample image — notebook will fail unless user provides their own.
- Folder name has typo ("Techiques" → "Techniques").

---

### Project 37 — Number Plate Detection
**Path**: `CV Projects 37 - Number plate detection/`

| Attribute | Value |
|---|---|
| Files | `number plate detection.ipynb`, `How 2 Run.txt`, `carrs.jpg`, `cars.jpg` |
| Libraries | `cv2`, `imutils` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | 2 car images (included) |
| Entry Point | Notebook |

**Issues**: None critical.

---

### Project 38 — Pencil Drawing Effect
**Path**: `CV Projects 38 - Pencil drawing effect/`

| Attribute | Value |
|---|---|
| Files | `Pencil drawing effect (openCV).ipynb`, `How 2 Run.txt`, `elonobama.png` |
| Libraries | `cv2`, `numpy`, `matplotlib` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `elonobama.png` (included) |
| Entry Point | Notebook |

**Issues**: None critical.

---

### Project 39 — Noise Removing (MISLABELED as "Pencil drawing effect")
**Path**: `CV Projects 39 - Pencil drawing effect/`

| Attribute | Value |
|---|---|
| Files | `Noise Removing (OpenCV).ipynb`, `How 2 Run.txt`, `image.jpg` |
| Libraries | `cv2`, `numpy`, `matplotlib` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `image.jpg` (included) |
| Entry Point | Notebook |

**Issues**:
- 🔴 **MISLABELED FOLDER**: Folder is named "Pencil drawing effect" but actual content is "Noise Removing (OpenCV)". Conflicts with Project 38 which actually IS pencil drawing.

---

### Project 40 — Non-photorealistic Rendering
**Path**: `CV Projects 40 - Non-photorealistic rendering/`

| Attribute | Value |
|---|---|
| Files | `Non-Photorealistic Rendering (openCV).ipynb`, `How 2 Run.txt`, 3× sample images |
| Libraries | `cv2`, `numpy`, `matplotlib` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | 3 sample images (included) |
| Entry Point | Notebook |

**Issues**: None critical.

---

### Project 41 — Image Segmentation
**Path**: `CV Projects 41 - Image Segmentation/`

| Attribute | Value |
|---|---|
| Files | `Image segmentation (openCV).ipynb`, `How 2 Run.txt`, `sundarpichai.jpg` |
| Libraries | `cv2`, `numpy` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `sundarpichai.jpg` (included) |
| Entry Point | Notebook |

**Technical Notes**:
- Single large code cell (~80 lines). Watershed-based segmentation.

**Issues**: None critical.

---

### Project 42 — Image Resizing using OpenCV
**Path**: `CV Projects 42 - Image resizing using opencv/`

| Attribute | Value |
|---|---|
| Files | `image resizing (OpenCV).ipynb`, `How 2 Run.txt`, `sukuna.png` |
| Libraries | `cv2`, `numpy` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `sukuna.png` (included) |
| Entry Point | Notebook |

**Issues**: None critical.

---

### Project 43 — Funny Cartoonizing Images using OpenCV
**Path**: `CV Projects 43 - Funny Cartoonizing Images using openCV/`

| Attribute | Value |
|---|---|
| Files | `cartooning an image (OpenCV).ipynb`, `How 2 Run.txt`, `kanaki.jpg` |
| Libraries | `cv2`, `numpy`, `matplotlib` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `kanaki.jpg` (included) |
| Entry Point | Notebook |

**Issues**: None critical.

---

### Project 44 — Joining Multiple Images to Display
**Path**: `CV Projects 44 - Joining Multiple Images to Display/`

| Attribute | Value |
|---|---|
| Files | `Joining_Multiple_Images_To_Display.py`, `How 2 Run.txt` |
| Libraries | `cv2`, `numpy` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | None (webcam) |
| Entry Point | `Joining_Multiple_Images_To_Display.py` |

**Technical Notes**:
- Contains `stackImages()` utility function (duplicated across Projects 22, 27).

**Issues**: None critical.

---

### Project 45 — Detecting Clicks on Images
**Path**: `CV Projects 45 -  Detecting clicks on images/`

| Attribute | Value |
|---|---|
| Files | `Detecting_Clicks_On_Images.py`, `How 2 Run.txt`, `cards.jpg` |
| Libraries | `cv2`, `numpy` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `cards.jpg` (included) |
| Entry Point | `Detecting_Clicks_On_Images.py` |

**Issues**:
- ⚠️ **NEAR-DUPLICATE** of Project 14 — very similar mouse-click perspective warp on the same card image.

---

### Project 46 — Face Detection Second Approach
**Path**: `CV Projects 46 - Face Detection Second Approach/`

| Attribute | Value |
|---|---|
| Files | `face detection (OpenCV).ipynb`, `How 2 Run.txt`, `group.jpg`, `haarcascade_frontalface_default.xml` |
| Libraries | `cv2`, `matplotlib` |
| Framework | `classical_opencv` (Haarcascade) |
| Model Files | `haarcascade_frontalface_default.xml` (**present**) |
| Dataset | `group.jpg` (included) |
| Entry Point | Notebook |

**Issues**:
- ⚠️ Notebook has execution errors preserved in saved output.

---

### Project 47 — Face Mask Detection
**Path**: `CV Projects 47 -  Face Mask Detection/`

| Attribute | Value |
|---|---|
| Files | `mask-detection (openCV).ipynb`, `How 2 Run.txt`, `annotations/` (853× XML), `images/` (853× PNG) |
| Libraries | `pandas`, `numpy`, `seaborn`, `sklearn`, `matplotlib`, `keras` (VGG19), `cv2`, `scipy` |
| Framework | `deep_learning_keras` (Transfer learning with VGG19) |
| Model Files | None saved (training notebook) |
| Dataset | 853 images + 853 XML annotations (**present locally**) |
| Entry Point | Notebook |

**Technical Notes**:
- 23-cell comprehensive ML pipeline: data loading → EDA → VGG19 transfer learning → training (50 epochs).
- Uses standalone `keras` imports (not `tf.keras`).

**Issues**:
- 🔴 **BROKEN DATASET PATHS**: Code references Kaggle paths (`../input/face-mask-12k-images-dataset/`, `../input/human-faces/`) — **will not work locally** despite having 853 local images.
- ⚠️ **LEGACY IMPORT**: Uses `from keras.applications import VGG19` instead of `from tensorflow.keras.applications`.
- No saved model output — training results not preserved.

---

### Project 48 — Face, Gender & Ethnicity Recognizer Model
**Path**: `CV Projects 48 - Face, Gender & Ethincity recognizer model/`

| Attribute | Value |
|---|---|
| Files | `keras-tuner-multi-output-cnn.ipynb`, `How 2 Run.txt`, `face,age & ethincity.zip`, `Link to the Dataset.txt` |
| Libraries | `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `keras_tuner`, `tensorflow`, `keras` |
| Framework | `deep_learning_keras` (Multi-output CNN + Keras Tuner hyperparameter optimization) |
| Model Files | None saved (training notebook) |
| Dataset | `face,age & ethincity.zip` (**present**, likely contains `age_gender.csv`) |
| Entry Point | Notebook |

**Technical Notes**:
- Most complex notebook in the repository (53 cells).
- Multi-output model: predicts gender (binary), ethnicity (multiclass), age (regression).
- Uses Keras Tuner for hyperparameter search.
- Uses proper `tensorflow.keras` imports (unlike Projects 13/47).
- Uses `ReduceLROnPlateau`, `EarlyStopping` callbacks.
- Uses `ImageDataGenerator` for data augmentation.

**Issues**:
- ⚠️ **EMPTY FILE**: `Link to the Dataset.txt` is completely empty — no dataset link provided.
- ⚠️ **HARDCODED PATH**: Keras Tuner references `'..\kaggle\working\'` directory.
- ⚠️ **INLINE PIP**: Uses `!pip install -U keras-tuner` inside notebook.
- ⚠️ Folder name has typo ("Ethincity" → "Ethnicity").
- Training notebook has execution errors in output (OOM / path errors).

---

### Project 49 — Real Time TextDetection
**Path**: `CV Projects 49 - Real Time TextDetection/`

| Attribute | Value |
|---|---|
| Files | `TextSimple.py`, `TextMoreExamples.py`, `How 2 Run.txt`, `ReadMe.md`, `1.png`, `oem.PNG`, `psm.PNG` |
| Libraries | `cv2`, `pytesseract`, `numpy`, `PIL` |
| Framework | `classical_opencv + pytesseract` |
| Model Files | None (requires external Tesseract-OCR installation) |
| Dataset | `1.png` (included) |
| Entry Point | `TextSimple.py` / `TextMoreExamples.py` |

**Technical Notes**:
- OCR-based text detection using Tesseract.
- `ReadMe.md` provides Tesseract installation links.

**Issues**:
- 🔴 **HARDCODED ABSOLUTE PATH**: `pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Nayanika Singh\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'` — **will fail on any other machine**.
- Requires external Tesseract-OCR installation (not documented in dependencies).

---

### Project 50 — Reversing Video
**Path**: `CV Projects 50 - Reversing video using openCV/`

| Attribute | Value |
|---|---|
| Files | `reversing video (openCV).ipynb`, `How 2 Run.txt`, `BMW.mp4`, 301× `frame*.jpg` files |
| Libraries | `cv2` |
| Framework | `classical_opencv` |
| Model Files | None |
| Dataset | `BMW.mp4` (included) |
| Entry Point | Notebook |

**Issues**:
- 🔴 **301 FRAME FILES COMMITTED**: 301 extracted `frame*.jpg` files committed to the repository — should be in `.gitignore`. Bloats repo significantly.

---

## 3. CRITICAL FINDINGS

### 3.1 — Code-Breaking Bugs (Will Crash at Runtime)

| # | Project | File | Bug |
|---|---|---|---|
| 1 | 3 | `detect_faces.py` | Missing `deploy.prototxt.txt` file — `cv2.dnn.readNetFromCaffe()` will fail |
| 2 | 14 | `Warp.py` | Uses `cv.destroyAllWindows()` but imported as `cv2` → `NameError` |
| 3 | 15 | `cartoon.py` | Indentation error (tab/space mixing) → `IndentationError` |
| 4 | 18 | `ballTracking.py` | Corrupted/garbled code at lines ~35-38 → `SyntaxError` |
| 5 | 26 | `LiveHSVAdjustor.py` | `cv2.COLOR_BGR2Result` is not a valid constant → `AttributeError` |
| 6 | 29 | `VPen.ipynb` | `cv2.findcnt()` does not exist → `AttributeError` (should be `cv2.findContours`) |

### 3.2 — Hardcoded Absolute Paths (Portability Failures)

| # | Project | File | Path |
|---|---|---|---|
| 1 | 5 | `fingerCount.py` | `C:\Users\Ashu.ASHUTOSH\PycharmProjects\fingerCounter\FingerImages` |
| 2 | 49 | `TextSimple.py` | `C:\Users\Nayanika Singh\AppData\Local\Programs\Tesseract-OCR\tesseract.exe` |
| 3 | 48 | notebook | `..\kaggle\working\` (Keras Tuner directory) |
| 4 | 47 | notebook | `../input/face-mask-12k-images-dataset/` (Kaggle paths) |
| 5 | 24 | `createData.py` | `'data/images'` (directory doesn't exist) |

### 3.3 — Duplicate / Redundant Content

| Type | Items | Impact |
|---|---|---|
| **Exact duplicate projects** | Projects 6 ↔ 21 (VolumeController) | Identical source files in two folders |
| **Near-duplicate projects** | Projects 14 ↔ 45 (click-detect / warp) | ~90% similar code, same `cards.jpg` |
| **Duplicate model files** | `shape_predictor_68_face_landmarks.dat` in Projects 4 & 17 | ~100MB × 2 wasted |
| **Duplicate utility code** | `stackImages()` in Projects 22, 27, 44 | Copy-pasted utility function |
| **Mislabeled folder** | Project 39 says "Pencil drawing" but contains "Noise Removing" | Conflicts with Project 38 |
| **Repo bloat** | 301 frame JPGs committed in Project 50 | Should be gitignored |

### 3.4 — Deprecated APIs & Legacy Code

| # | Project | Issue |
|---|---|---|
| 1 | 13 | `classifier.predict_classes()` removed in TensorFlow 2.6+ |
| 2 | 13 | Standalone `from keras.models import load_model` (should be `tf.keras`) |
| 3 | 47 | Standalone `from keras.applications import VGG19` (should be `tf.keras`) |
| 4 | 13 | `Operator.py` is a copy of Python's built-in `operator` module |
| 5 | 16, 24, 46 | Haarcascade-based detection is considered legacy approach |

### 3.5 — Missing Files & Documentation

| # | Project | Missing Item |
|---|---|---|
| 1 | 3 | `deploy.prototxt.txt` (required for model loading) |
| 2 | 11 | `myDataFile.txt` (referenced in code) |
| 3 | 36 | No sample image (notebook requires one) |
| 4 | 48 | `Link to the Dataset.txt` is empty |
| 5 | 25 | `ReadMe.md` is empty |
| 6 | 9 | No HOW2RUN.txt or README |
| 7 | 6 | No HOW2RUN.txt |
| 8 | ALL | **No `requirements.txt`** in any project |
| 9 | 10, 13 | `main.py` files are PyCharm boilerplate, not actual entry points |

### 3.6 — Technical Debt Summary

| Category | Count | Severity |
|---|---|---|
| Crash-inducing bugs | 6 | 🔴 Critical |
| Hardcoded absolute paths | 5 | 🔴 Critical |
| Missing required files | 4 | 🔴 High |
| Exact duplicate projects | 1 pair | ⚠️ Medium |
| Near-duplicate projects | 1 pair | ⚠️ Medium |
| Deprecated APIs | 5 instances | ⚠️ Medium |
| Missing documentation | 4 projects | ⚠️ Low |
| Filename/folder typos | 5 instances | ⚠️ Low |
| No dependency manifests | 50/50 projects | 🔴 High |
| No tests/evaluation | 50/50 projects | ⚠️ Medium |
| No CI/CD | entire repo | ⚠️ Medium |
| Repo bloat (committed artifacts) | 1 project (301 files) | ⚠️ Medium |

### 3.7 — Folder Naming Inconsistencies

The repository has no consistent naming convention:
- `CV Project 1` through `CV Project 18` (singular "Project", spaces)
- `CV Projects 19` onward (plural "Projects")
- `CV projects 30` (lowercase)
- Missing Project 5 as separate folder (bundled as `CV Project 5 - fingerCounter/`)
- Varied separator styles: `-`, `—`, spaces
- Multiple typos: "Bluring", "Sharpning", "Techiques", "Ethincity", "objectDetectoin"

### 3.8 — GPU/Compute Readiness Assessment

| Criterion | Status |
|---|---|
| PyTorch used | ❌ No |
| CUDA/GPU pinning | ❌ No |
| `tf.config.experimental` GPU config | ❌ No |
| `CUDA_VISIBLE_DEVICES` | ❌ No |
| Mixed precision training | ❌ No |
| Batch size configuration | Only in Projects 47, 48 (hardcoded) |
| Model checkpointing | Only Project 48 (via Keras Tuner) |
| Data pipeline optimization | ❌ No (`tf.data` not used anywhere) |

**Verdict**: Zero projects are GPU-ready. The 3 deep learning projects (13, 47, 48) would benefit from GPU acceleration but require explicit device configuration to leverage it.

---

## 4. DATASET DETECTION SUMMARY

| Project | Dataset Type | Included? | Notes |
|---|---|---|---|
| 1 | Static image | ✅ | `test.jpg` |
| 2 | Document images | ✅ | 5 JPGs |
| 3 | Face image | ✅ | `iron_chic.jpg` |
| 4 | Face images | ✅ | 4 examples |
| 5 | Finger overlay images | ✅ | 6 JPGs in `FingerImages/` |
| 6-10 | Webcam only | N/A | Real-time capture |
| 11 | None | ❌ | Missing `myDataFile.txt` |
| 12 | Sample image | ✅ | 1 JPG |
| 13 | Demo GIF | ✅ | `Test.gif` |
| 14, 45 | Card image | ✅ | `cards.jpg` |
| 16 | Video files | ✅ | 3 videos (mp4/avi) |
| 17 | Webcam only | N/A | |
| 18 | Video | ✅ | `ball_tracking_example.mp4` |
| 20 | Template images | ✅ | Waldo images |
| 22-23 | Sample images | ✅ | |
| 25 | Reference image | ✅ | `1.jpg` |
| 28-35, 38-43 | Sample images | ✅ | Various (mostly `yoaimo.png`) |
| 36 | None | ❌ | No image provided |
| 37 | Car images | ✅ | 2 JPGs |
| 46 | Group photo | ✅ | `group.jpg` + cascade XML |
| 47 | Face mask dataset | ✅ (local) | 853 images + 853 XML annotations; but code paths point to Kaggle |
| 48 | Face dataset | ✅ (zipped) | `face,age & ethincity.zip`; dataset link file empty |
| 49 | Sample image | ✅ | `1.png` |
| 50 | Video + frames | ✅ | `BMW.mp4` + 301 extracted frames |

---

## 5. MODEL / WEIGHT FILE INVENTORY

| Project | File | Type | Size | Status |
|---|---|---|---|---|
| 3 | `res10_300x300_ssd_iter_140000.caffemodel` | Caffe weights | ~10MB | ✅ Present, but missing prototxt |
| 4 | `shape_predictor_68_face_landmarks.dat` | dlib shape predictor | ~100MB | ✅ Present |
| 12 | `MobileNetSSD_deploy.caffemodel` | Caffe weights | ~23MB | ✅ Present |
| 12 | `MobileNetSSD_deploy.prototxt.txt` | Caffe architecture | ~28KB | ✅ Present |
| 13 | `digit_model.h5` | Keras CNN | ~1MB | ✅ Present |
| 16 | `carx.xml` | Haarcascade | ~1MB | ✅ Present |
| 17 | `shape_predictor_68_face_landmarks.dat` | dlib shape predictor | ~100MB | ✅ Present (duplicate of P4) |
| 24 | `haarcascades/` (18 XMLs) | Haarcascade collection | ~25MB total | ✅ Present |
| 46 | `haarcascade_frontalface_default.xml` | Haarcascade | ~1MB | ✅ Present |

---

## 6. DEPENDENCY MAP (Reconstructed — No requirements.txt Exists)

### Core Dependencies (used across multiple projects)
```
opencv-python (cv2)          — ALL 50 projects
numpy                        — ~40 projects
imutils                      — Projects 2, 3, 4, 7, 8, 17, 18, 37
matplotlib                   — Projects 28, 30, 32, 38, 40, 43, 46
argparse                     — Projects 2, 3, 4, 7, 8, 12, 16, 17, 18
```

### Specialized Dependencies
```
mediapipe                    — Projects 5, 6, 10, 21
dlib                         — Projects 4, 17
pyzbar                       — Project 11
pytesseract                  — Project 49
pycaw + comtypes             — Projects 6, 21 (Windows-only)
scikit-image (skimage)       — Projects 2, 15
scipy                        — Projects 7, 17, 47
```

### Deep Learning Dependencies
```
keras (standalone)           — Projects 13, 47
tensorflow                   — Project 48
keras-tuner                  — Project 48
pandas                       — Projects 47, 48
seaborn                      — Projects 47, 48
scikit-learn (sklearn)       — Projects 47, 48
Pillow (PIL)                 — Project 49
```

---

*END OF PHASE 0 AUDIT — ZERO FILES MODIFIED*
