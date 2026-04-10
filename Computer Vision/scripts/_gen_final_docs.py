#!/usr/bin/env python
"""
FINAL DOCS GENERATOR — Code-as-Truth README Generator
======================================================
Generates:
  1) Per-project README.md for all 50 project folders
  2) Root README.md
  3) WORKSPACE_OVERVIEW.md

Every claim is derived from actual code inspection done beforehand.
Run:  python scripts/_gen_final_docs.py [--force]
"""

from __future__ import annotations
import argparse, sys, textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ============================================================================
# PROJECT METADATA — verified from reading every modern.py and train.py
# ============================================================================

PROJECTS = [
    # (num, folder_name, register_key, display_name, category, framework,
    #  uses_resolve, has_train, load_summary, predict_summary, visualize_summary,
    #  docstring_short, dataset_config_key_or_None, files_extra)
    dict(
        num=1,
        folder="CV Project 1 -  Real Time Angle Detector",
        reg_key="angle_detector_v2",
        display="Angle Detector (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="Initializes empty points list (no model needed).",
        predict_summary="Converts to grayscale, applies Gaussian blur + Canny edge detection, finds lines via HoughLinesP.",
        visualize_summary="Draws detected lines on frame, computes angle between first two lines using atan2, displays angle text.",
        short_desc="Detects edges and lines in a live camera feed and computes the angle between the first two detected lines using OpenCV Hough line transform.",
        dataset_key=None,
        extra_files=["AngleDetector.py"],
    ),
    dict(
        num=2,
        folder="CV Project 2 - Real Time Document Scanner-fine",
        reg_key="doc_scanner_v2",
        display="Document Scanner (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="No-op (no model to load).",
        predict_summary="Grayscale + blur + Canny edges, finds contours, searches for 4-point quadrilateral (document contour).",
        visualize_summary="Draws detected document contour, applies 4-point perspective warp, shows warped thumbnail inset on frame.",
        short_desc="Scans documents in real-time by detecting rectangular contours and applying a 4-point perspective warp to produce a top-down view.",
        dataset_key=None,
        extra_files=["docScanner.py", "helpFunctions/"],
    ),
    dict(
        num=3,
        folder="CV Project 3 - Real Time Face detector Image",
        reg_key="face_detection_v2",
        display="Face Detection (YOLO)",
        category="detection",
        framework="Ultralytics YOLO",
        uses_resolve=True,
        has_train=True,
        load_summary='Resolves weights via `models.registry.resolve("face_detection", "detect")`, falls back to `yolo26n.pt`, loads YOLO model via `utils.yolo.load_yolo()`.',
        predict_summary="Runs YOLO detection on frame with confidence threshold 0.5.",
        visualize_summary="Calls `output[0].plot()` to draw YOLO bounding boxes.",
        short_desc="Detects faces in images and live camera feeds using YOLO26 object detection with model-registry weight resolution.",
        dataset_key="face_detection",
        extra_files=["detect_faces.py", "detect_faces_video.py", "train.py"],
    ),
    dict(
        num=4,
        folder="CV Project 4 - Facial Landmarking",
        reg_key="facial_landmarks_v2",
        display="Facial Landmarks (YOLO-Pose)",
        category="pose",
        framework="Ultralytics YOLO-Pose",
        uses_resolve=True,
        has_train=True,
        load_summary='Resolves weights via `models.registry.resolve("facial_landmarks", "pose")`, falls back to `yolo26n-pose.pt`, loads via `utils.yolo.load_yolo_pose()`.',
        predict_summary="Runs YOLO-Pose inference on frame with confidence 0.5.",
        visualize_summary="Calls `output[0].plot()`, then highlights face keypoints (nose, eyes, ears) with colored circles and labels.",
        short_desc="Detects facial landmarks using YOLO-Pose keypoint estimation, highlighting nose, eyes, and ears on detected faces.",
        dataset_key="facial_landmarks",
        extra_files=["facial_landmarking.py", "train.py"],
    ),
    dict(
        num=5,
        folder="CV Projects 5 - fingerCounter",
        reg_key="finger_counter_v2",
        display="Finger Counter (YOLO-Pose)",
        category="pose",
        framework="Ultralytics YOLO-Pose",
        uses_resolve=True,
        has_train=False,
        load_summary='Resolves weights via `models.registry.resolve("finger_counter", "pose")`, falls back to `yolo26n-pose.pt`.',
        predict_summary="Runs YOLO-Pose inference on frame with confidence 0.5.",
        visualize_summary="Calls `output[0].plot()`, highlights wrist keypoints (COCO indices 9, 10) with red circles and labels.",
        short_desc="Counts visible fingers using YOLO-Pose body keypoints for wrist detection, replacing legacy MediaPipe hand tracking. No training pipeline — uses pretrained pose model.",
        dataset_key=None,
        extra_files=["fingerCount.py", "HandDetectionModule.py", "FingerImages/"],
    ),
    dict(
        num=6,
        folder="CV Project 6 - Live HTM",
        reg_key="hand_tracking_v2",
        display="Hand Tracking – Volume Control (YOLO-Pose)",
        category="pose",
        framework="Ultralytics YOLO-Pose",
        uses_resolve=True,
        has_train=True,
        load_summary='Resolves weights via `models.registry.resolve("hand_tracking", "pose")`, falls back to `yolo26n-pose.pt`.',
        predict_summary="Runs YOLO-Pose inference on frame with confidence 0.5.",
        visualize_summary="Calls `output[0].plot()`, highlights wrist keypoints (indices 9, 10) with magenta circles.",
        short_desc="Tracks hand positions using YOLO-Pose for gesture-based volume control, replacing legacy MediaPipe hand tracking.",
        dataset_key="hand_tracking",
        extra_files=["HandTrackingModule.py", "VolumeControl.py", "train.py"],
    ),
    dict(
        num=7,
        folder="CV Project 7 - Real Time Object Size detector",
        reg_key="object_size_v2",
        display="Object Size Detector (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="No-op (no model to load).",
        predict_summary="Grayscale + blur + Canny + dilate/erode, finds external contours, computes minAreaRect, calculates dimensions in cm using PIXELS_PER_CM calibration.",
        visualize_summary="Draws rotated bounding boxes, labels each object with width x height in cm, shows object count.",
        short_desc="Measures real-world object dimensions in centimeters from a camera feed using contour detection and a pixel-per-cm calibration constant.",
        dataset_key=None,
        extra_files=["object_size.py"],
    ),
    dict(
        num=8,
        folder="CV Project 8 - OMR Evaluator",
        reg_key="omr_evaluator_v2",
        display="OMR Evaluator (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="No-op (no model to load).",
        predict_summary="Edge detection + contour-based document detection, adaptive thresholding for bubble detection, computes fill ratio for each circular bubble contour.",
        visualize_summary="Draws document contour, highlights filled bubbles (fill_ratio > 0.5) with red rectangles, shows filled-bubble count.",
        short_desc="Evaluates Optical Mark Recognition (OMR) sheets by detecting filled bubbles via contour analysis and fill-ratio thresholding.",
        dataset_key=None,
        extra_files=["OMRevaluator.py"],
    ),
    dict(
        num=9,
        folder="CV Project 9 - Real Time Painter of Camera Screen",
        reg_key="painter_v2",
        display="Real-Time Painter (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="Initializes a `deque` of canvas points (max 1024).",
        predict_summary="HSV color tracking (blue marker range), finds largest contour, tracks center point via minEnclosingCircle.",
        visualize_summary="Flips frame horizontally, draws connected lines between tracked points in red, shows green circle at current tracked center.",
        short_desc="Lets users paint on the camera feed in real-time by tracking a blue-colored marker using HSV color detection.",
        dataset_key=None,
        extra_files=["Paint.py"],
    ),
    dict(
        num=10,
        folder="CV Project 10 - Live PoseDetector",
        reg_key="pose_detector_v2",
        display="Pose Detector (YOLO-Pose)",
        category="pose",
        framework="Ultralytics YOLO-Pose",
        uses_resolve=True,
        has_train=True,
        load_summary='Resolves weights via `models.registry.resolve("pose_detection", "pose")`, falls back to `yolo26n-pose.pt`.',
        predict_summary="Runs YOLO-Pose inference on frame with confidence 0.5.",
        visualize_summary="Calls `output[0].plot()`, shows person count text at bottom of frame.",
        short_desc="Detects human body poses in real-time using YOLO-Pose keypoint estimation with person counting.",
        dataset_key="pose_detector",
        extra_files=["main.py", "poseDetector-I.py", "train.py"],
    ),
    dict(
        num=11,
        folder="CV Project 11 - Live QR-Reader",
        reg_key="qr_reader_v2",
        display="QR Reader (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="Creates a `cv2.QRCodeDetector()` instance.",
        predict_summary="Calls `detectAndDecode()` on frame to find and decode QR codes.",
        visualize_summary="Draws green lines around QR bounding box, shows decoded QR data text.",
        short_desc="Reads QR codes in real-time from a camera feed using OpenCV's built-in QR code detector.",
        dataset_key=None,
        extra_files=["QR_Reader.py"],
    ),
    dict(
        num=12,
        folder="CV Project 12 - Real Time Object Detection-fine",
        reg_key="object_detection_v2",
        display="Real-Time Object Detection (YOLO)",
        category="detection",
        framework="Ultralytics YOLO",
        uses_resolve=True,
        has_train=True,
        load_summary='Resolves weights via `models.registry.resolve("object_detection", "detect")`, falls back to `yolo26n.pt`.',
        predict_summary="Runs YOLO detection on frame with confidence 0.25.",
        visualize_summary="Returns `output[0].plot()` — standard YOLO annotated frame.",
        short_desc="Performs real-time object detection on camera or video using YOLO26, replacing the legacy MobileNet-SSD Caffe model.",
        dataset_key="object_detection",
        extra_files=["real_time_object_detection.py", "MobileNetSSD_deploy.prototxt.txt", "train.py"],
    ),
    dict(
        num=13,
        folder="CV Project 13 - Real Time Sudoku Solver",
        reg_key="sudoku_solver_v2",
        display="Sudoku Solver (YOLO-enhanced)",
        category="detection",
        framework="Ultralytics YOLO",
        uses_resolve=True,
        has_train=True,
        load_summary='Resolves weights via `models.registry.resolve("sudoku_solver", "detect")`, falls back to `yolo26n.pt`.',
        predict_summary="Runs YOLO detection on frame (conf=0.4). Highlights `book`/`cell_phone` detections as potential sudoku grids.",
        visualize_summary="Draws bounding boxes; grids (book/cell_phone) in green, other objects in orange. Shows status text: digit OCR requires a custom model.",
        short_desc="Runs YOLO26 detection to locate potential sudoku grids (book/cell_phone objects). Does not solve puzzles — digit OCR requires a custom-trained model. Separate `train.py` trains a ResNet-18 digit classifier on MNIST via torchvision.",
        dataset_key="sudoku_solver",
        extra_files=["main.py", "Operator.py", "Sudoku.py", "SudokuSolver.py", "train.py"],
    ),
    dict(
        num=14,
        folder="CV Project 14 - click-detect on image",
        reg_key="warp_perspective_v2",
        display="Warp Perspective (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="No-op (no model to load).",
        predict_summary="Grayscale + blur + Canny edges, finds largest 4-sided contour (quadrilateral), applies 4-point perspective transform.",
        visualize_summary="Draws detected quadrilateral contour in green, shows status text.",
        short_desc="Detects and warps quadrilateral regions in images using OpenCV contour detection and perspective transforms.",
        dataset_key=None,
        extra_files=["Warp.py", "cards.jpg"],
    ),
    dict(
        num=15,
        folder="CV Project 15 - Live Image Cartoonifier",
        reg_key="cartoonifier_v2",
        display="Image Cartoonifier (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="No-op (no model to load).",
        predict_summary="Grayscale + median blur + adaptive threshold for edges, 7 rounds of bilateral filter for smooth color, combined via bitwise_and.",
        visualize_summary="Returns cartoon image with text overlay.",
        short_desc="Applies a cartoon effect to live camera frames using bilateral filtering and adaptive thresholding for edge detection.",
        dataset_key=None,
        extra_files=["cartoon.py"],
    ),
    dict(
        num=16,
        folder="CV Project 16 -Live Car-Detection",
        reg_key="car_detection_v2",
        display="Car Detection (YOLO)",
        category="detection",
        framework="Ultralytics YOLO",
        uses_resolve=True,
        has_train=True,
        load_summary='Resolves weights via `models.registry.resolve("car_detection", "detect")`, falls back to `yolo26n.pt`.',
        predict_summary="Runs YOLO detection filtered to vehicle COCO classes (car=2, bus=5, truck=7, motorcycle=3) with confidence 0.4.",
        visualize_summary="Returns `output[0].plot()` — standard YOLO annotated frame.",
        short_desc="Detects vehicles (cars, buses, trucks, motorcycles) in live video using YOLO26 with COCO class filtering.",
        dataset_key="car_detection",
        extra_files=["Vehicles_detection.py", "train.py"],
    ),
    dict(
        num=17,
        folder="CV Project 17 - Blink Detection",
        reg_key="blink_detection_v2",
        display="Blink Detection (YOLO-Pose)",
        category="pose",
        framework="Ultralytics YOLO-Pose",
        uses_resolve=True,
        has_train=True,
        load_summary='Resolves weights via `models.registry.resolve("blink_detection", "pose")`, falls back to `yolo26n-pose.pt`.',
        predict_summary="Runs YOLO-Pose inference on frame with confidence 0.5.",
        visualize_summary="Calls `output[0].plot()`, adds text noting face-keypoint model needed for blink EAR (Eye Aspect Ratio).",
        short_desc="Detects eye blinks using YOLO-Pose keypoints. Requires a fine-tuned face-keypoint model for accurate Eye Aspect Ratio computation.",
        dataset_key="blink_detection",
        extra_files=["blink_detector.py", "train.py"],
    ),
    dict(
        num=18,
        folder="CV Project 18 - Live Ball Tracking",
        reg_key="ball_tracking_v2",
        display="Ball Tracking (YOLO)",
        category="detection",
        framework="Ultralytics YOLO",
        uses_resolve=True,
        has_train=True,
        load_summary='Resolves weights via `models.registry.resolve("ball_tracking", "detect")`, falls back to `yolo26n.pt`.',
        predict_summary="Runs YOLO detection filtered to COCO class 32 (sports ball) with confidence 0.3.",
        visualize_summary="Calls `output[0].plot()`, shows ball count text at bottom of frame.",
        short_desc="Tracks sports balls in live video using YOLO26 detection filtered to COCO sports-ball class.",
        dataset_key="ball_tracking",
        extra_files=["ballTracking.py", "train.py"],
    ),
    dict(
        num=19,
        folder="CV Projects 19 - GrayScaleConverter",
        reg_key="grayscale_converter_v2",
        display="Grayscale Converter (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="No-op (no model to load).",
        predict_summary="Converts frame to grayscale via `cv2.cvtColor(BGR2GRAY)`.",
        visualize_summary="Converts grayscale back to BGR for display, adds text label.",
        short_desc="Converts live camera frames to grayscale using OpenCV color space conversion.",
        dataset_key=None,
        extra_files=["GrayscaleConverter.py"],
    ),
    dict(
        num=20,
        folder="CV Projects 20 - image_finder",
        reg_key="image_finder_v2",
        display="Image Finder (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="Initializes `self.template = None` (must call `set_template()` before use).",
        predict_summary="Runs `cv2.matchTemplate` (TM_CCOEFF_NORMED) between frame and template, returns match locations above threshold 0.8.",
        visualize_summary="Draws green rectangles at each match location, shows match count.",
        short_desc="Finds occurrences of a template image within a larger image or camera frame using OpenCV template matching.",
        dataset_key=None,
        extra_files=["finder.py", "images/"],
    ),
    dict(
        num=21,
        folder="CV Projects 21 - VolumeController",
        reg_key="volume_controller_v2",
        display="Volume Controller (YOLO-Pose)",
        category="pose",
        framework="Ultralytics YOLO-Pose",
        uses_resolve=True,
        has_train=True,
        load_summary='Resolves weights via `models.registry.resolve("volume_controller", "pose")`, falls back to `yolo26n-pose.pt`.',
        predict_summary="Runs YOLO-Pose inference on frame with confidence 0.5.",
        visualize_summary="Calls `output[0].plot()`, highlights wrist keypoints (indices 9, 10) with green/red circles.",
        short_desc="Controls system volume via hand gesture distance using YOLO-Pose wrist keypoint detection.",
        dataset_key="volume_controller",
        extra_files=["HandTrackingModule.py", "VolumeControl.py", "train.py"],
    ),
    dict(
        num=22,
        folder="CV Projects 22 - Live Color Picker",
        reg_key="color_picker_v2",
        display="Color Picker (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="Initializes `self.last_color = (0, 0, 0)`.",
        predict_summary="Samples center 10×10 pixel ROI, computes average BGR and HSV values.",
        visualize_summary="Draws crosshair at center, renders colored info panel with BGR/HSV values.",
        short_desc="Picks and displays the BGR and HSV color values from the center of a live camera feed.",
        dataset_key=None,
        extra_files=["ColorPicker.py"],
    ),
    dict(
        num=23,
        folder="CV Projects 23 - Crop Resize Image using OpenCV",
        reg_key="crop_resize_v2",
        display="Crop & Resize (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="No-op (no model to load).",
        predict_summary="Center-crops image to 50%, then resizes by SCALE_FACTOR (0.5) using INTER_AREA.",
        visualize_summary="Draws green rectangle showing crop region, shows original and crop dimensions text.",
        short_desc="Demonstrates center-cropping and resizing of images using OpenCV with various interpolation modes.",
        dataset_key=None,
        extra_files=["Crop_Resize_Images.py"],
    ),
    dict(
        num=24,
        folder="CV Projects 24 - Custom Object Detection",
        reg_key="custom_object_detection_v2",
        display="Custom Object Detection (YOLO)",
        category="detection",
        framework="Ultralytics YOLO",
        uses_resolve=True,
        has_train=True,
        load_summary='Resolves weights via `models.registry.resolve("custom_object_detection", "detect")`, falls back to `yolo26n.pt`.',
        predict_summary="Runs YOLO detection on frame with confidence 0.3.",
        visualize_summary="Returns `output[0].plot()` — standard YOLO annotated frame.",
        short_desc="Custom object detection using YOLO26, replacing legacy Haar cascade approach. Trainable on any YOLO-format dataset.",
        dataset_key="custom_object_detection",
        extra_files=["objectDetectoin.py", "createData.py", "haarcascades/", "train.py"],
    ),
    dict(
        num=25,
        folder="CV Projects 25 - Real Time Object Measurement",
        reg_key="object_measurement_v2",
        display="Object Measurement (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="No-op (no model to load).",
        predict_summary="Grayscale + blur + Canny + dilate/erode, finds external contours, computes minAreaRect, calculates real-world dimensions using PIXELS_PER_METRIC.",
        visualize_summary="Draws rotated bounding boxes with center dots, labels width × height, shows object count.",
        short_desc="Measures real-world object dimensions from a camera feed using contour detection and a pixel-per-metric calibration constant.",
        dataset_key=None,
        extra_files=["ObjectMeasurement.py", "utlis.py"],
    ),
    dict(
        num=26,
        folder="CV Projects 26 - Real Time Color Detection",
        reg_key="color_detection_v2",
        display="Color Detection (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="No-op (no model to load).",
        predict_summary="HSV conversion, dual-range inRange mask for red detection, morphological open/close, finds contours with area > 500.",
        visualize_summary="Draws red rectangles around detected red regions, labels them, shows detection count.",
        short_desc="Detects red-colored regions in a live camera feed using HSV color space and morphological filtering.",
        dataset_key=None,
        extra_files=["LiveHSVAdjustor.py"],
    ),
    dict(
        num=27,
        folder="CV Projects 27 - Real Time Shape Detection",
        reg_key="shape_detection_v2",
        display="Shape Detection (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="No-op (no model to load).",
        predict_summary="Grayscale + blur + adaptive threshold, finds contours, classifies shapes by polygon vertex count (triangle, square, rectangle, pentagon, hexagon, circle).",
        visualize_summary="Draws contours in green, labels each shape name at centroid, shows total shape count.",
        short_desc="Classifies geometric shapes (triangle, square, circle, etc.) in real-time using contour analysis and polygon approximation.",
        dataset_key=None,
        extra_files=["RealTime_Shape_Detection_Contours.py"],
    ),
    dict(
        num=28,
        folder="CV Projects 28 - Water Marking on Image using OpenCV",
        reg_key="watermarking_v2",
        display="Watermarking (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="No-op (no model to load).",
        predict_summary='Creates text watermark overlay ("CV Projects v2") at bottom-right, blends via `addWeighted` (alpha=0.7).',
        visualize_summary="Returns the watermarked image directly.",
        short_desc="Applies a semi-transparent text watermark to images using OpenCV alpha blending.",
        dataset_key=None,
        extra_files=["watermarking on images using OpenCV.ipynb"],
    ),
    dict(
        num=29,
        folder="CV Projects 29 - Live Virtual Pen",
        reg_key="virtual_pen_v2",
        display="Virtual Pen (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="Initializes a `deque` of points (max 2048).",
        predict_summary="HSV color tracking (green marker range), erode/dilate morphology, finds largest contour, tracks center via minEnclosingCircle.",
        visualize_summary="Draws connected red lines between tracked points, shows green circle at current center.",
        short_desc="Draws on the camera feed in real-time by tracking a green-colored marker using HSV color detection.",
        dataset_key=None,
        extra_files=["finding HSV values.ipynb", "VPen.ipynb"],
    ),
    dict(
        num=30,
        folder="CV projects 30 - Contrast enhancing of color images",
        reg_key="contrast_color_v2",
        display="Contrast Enhancement Color (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="Creates a CLAHE instance with clipLimit=3.0, tileGridSize=(8, 8).",
        predict_summary="Converts to LAB color space, applies CLAHE to L channel, converts back to BGR.",
        visualize_summary="Side-by-side split view — original on left, CLAHE enhanced on right, separated by a green line.",
        short_desc="Enhances color image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space.",
        dataset_key=None,
        extra_files=["OpenCV ( Contrast enhancing of color images ).ipynb"],
    ),
    dict(
        num=31,
        folder="CV Projects 31 - contrast enhancing of gray scale image using opencv",
        reg_key="contrast_gray_v2",
        display="Contrast Enhancement Gray (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="Creates a CLAHE instance with clipLimit=2.0, tileGridSize=(8, 8).",
        predict_summary="Converts to grayscale, applies both histogram equalization and CLAHE.",
        visualize_summary="3-image horizontal strip: Original, Hist EQ, CLAHE — each labeled.",
        short_desc="Compares histogram equalization and CLAHE for grayscale contrast enhancement.",
        dataset_key=None,
        extra_files=["Enhancing contrast of gray scale image (OpenCV).ipynb"],
    ),
    dict(
        num=32,
        folder="CV Projects 32 - Draw vertical lines of coin",
        reg_key="coin_lines_v2",
        display="Coin Lines (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="No-op (no model to load).",
        predict_summary="Grayscale + median blur, detects circles via HoughCircles (dp=1.2, minDist=50).",
        visualize_summary="Draws detected circles in green, center dots in red, vertical blue lines through each coin center, shows coin count.",
        short_desc="Detects coins in images using Hough Circle Transform and draws vertical reference lines through their centers.",
        dataset_key=None,
        extra_files=["Drawing vertical lines on coin.ipynb"],
    ),
    dict(
        num=33,
        folder="CV Projects 33 - Image Bluring using opencv",
        reg_key="image_blurring_v2",
        display="Image Blurring (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="No-op (no model to load).",
        predict_summary="Applies 4 blur methods: GaussianBlur, medianBlur, bilateralFilter, box blur (`cv2.blur`) with kernel size 15.",
        visualize_summary="2×2 grid showing Gaussian, Median, Bilateral, and Box blur results — each labeled.",
        short_desc="Demonstrates and compares four OpenCV blurring methods: Gaussian, Median, Bilateral, and Box blur.",
        dataset_key=None,
        extra_files=["image blurring (OpenCV ).ipynb"],
    ),
    dict(
        num=34,
        folder="CV Projects 34 - Live Motion Blurring",
        reg_key="motion_blur_v2",
        display="Motion Blur (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="Creates horizontal and vertical motion blur kernels (size 30).",
        predict_summary="Applies horizontal and vertical motion blur via `cv2.filter2D` with custom kernels.",
        visualize_summary="3-image horizontal strip: Original, H-Motion, V-Motion — each labeled.",
        short_desc="Applies directional (horizontal and vertical) motion blur effects using custom convolution kernels.",
        dataset_key=None,
        extra_files=["motion blurring effect .ipynb"],
    ),
    dict(
        num=35,
        folder="CV Projects 35 - Sharpning of images using OpenCV",
        reg_key="image_sharpening_v2",
        display="Image Sharpening (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="Creates Laplacian sharpen kernel and edge enhance kernel.",
        predict_summary="Applies 3 sharpening methods: Laplacian kernel, edge-enhance kernel, unsharp mask (`addWeighted` with Gaussian blur).",
        visualize_summary="2×2 grid: Original, Laplacian Sharp, Edge Enhance, Unsharp Mask — each labeled.",
        short_desc="Compares three image sharpening techniques: Laplacian kernel, edge enhancement kernel, and unsharp masking.",
        dataset_key=None,
        extra_files=["sharpening of images using opencv.ipynb"],
    ),
    dict(
        num=36,
        folder="CV Projects 36 - Thresholding Techiques",
        reg_key="thresholding_v2",
        display="Thresholding Techniques (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="No-op (no model to load).",
        predict_summary="Applies 4 thresholding methods: Binary, Binary Inverse, Otsu, Adaptive Gaussian.",
        visualize_summary="2×2 grid: Binary, Binary Inv, Otsu, Adaptive — each labeled.",
        short_desc="Demonstrates and compares four OpenCV thresholding techniques on live camera frames.",
        dataset_key=None,
        extra_files=["Thresholding techniques (OpenCV).ipynb"],
    ),
    dict(
        num=37,
        folder="CV Projects 37 - Number plate detection",
        reg_key=None,
        display="Number Plate Detection",
        category="detection",
        framework="Ultralytics YOLO",
        uses_resolve=False,
        has_train=True,
        load_summary="modern.py is empty (no wrapper class defined).",
        predict_summary="N/A (empty modern.py).",
        visualize_summary="N/A (empty modern.py).",
        short_desc="Number/license plate detection project with a YOLO26 training pipeline. The modern.py wrapper is not yet implemented.",
        dataset_key=None,
        extra_files=["number plate detection.ipynb", "train.py"],
    ),
    dict(
        num=38,
        folder="CV Projects 38 - Pencil drawing effect",
        reg_key="pencil_sketch_v2",
        display="Pencil Sketch (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="No-op (no model to load).",
        predict_summary="Calls `cv2.pencilSketch()` to produce gray and color pencil sketch variants.",
        visualize_summary="3-image horizontal strip: Original, Pencil B&W, Pencil Color — each labeled.",
        short_desc="Applies pencil sketch effects (grayscale and color) using OpenCV's built-in `pencilSketch()` function.",
        dataset_key=None,
        extra_files=["Pencil drawing effect (openCV).ipynb"],
    ),
    dict(
        num=39,
        folder="CV Projects 39 - Pencil drawing effect",
        reg_key="noise_removal_v2",
        display="Noise Removal (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="No-op (no model to load).",
        predict_summary="Applies 3 denoising methods: `fastNlMeansDenoisingColored`, GaussianBlur, medianBlur.",
        visualize_summary="2×2 grid: Original, NLMeans, Gaussian, Median — each labeled.",
        short_desc="Compares three noise removal techniques: Non-Local Means, Gaussian, and Median denoising. (Note: folder is named 'Pencil drawing effect' but contains noise removal code.)",
        dataset_key=None,
        extra_files=["Noise Removing (OpenCV).ipynb"],
    ),
    dict(
        num=40,
        folder="CV Projects 40 - Non-photorealistic rendering",
        reg_key="npr_rendering_v2",
        display="Non-Photorealistic Rendering (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="No-op (no model to load).",
        predict_summary="Applies 3 NPR effects: `cv2.stylization`, `cv2.edgePreservingFilter`, `cv2.detailEnhance`.",
        visualize_summary="2×2 grid: Original, Stylized, Edge Preserved, Detail Enhanced — each labeled.",
        short_desc="Demonstrates OpenCV non-photorealistic rendering effects: stylization, edge-preserving filter, and detail enhancement.",
        dataset_key=None,
        extra_files=["Non-Photorealistic Rendering (openCV).ipynb"],
    ),
    dict(
        num=41,
        folder="CV Projects 41 - Image Segmentation",
        reg_key="image_segmentation_v2",
        display="Image Segmentation (YOLO-Seg)",
        category="segmentation",
        framework="Ultralytics YOLO-Seg",
        uses_resolve=True,
        has_train=True,
        load_summary='Resolves weights via `models.registry.resolve("image_segmentation", "seg")`, falls back to `yolo26n-seg.pt`.',
        predict_summary="Runs YOLO-Seg inference on frame with confidence 0.4.",
        visualize_summary="Calls `output[0].plot()` for segmentation masks, shows segment count text.",
        short_desc="Performs instance segmentation on live camera frames using YOLO26-Seg. Training supports both YOLO-Seg and DeepLabV3 backends (see `train.py deeplab`).",
        dataset_key="image_segmentation",
        extra_files=["Image segmentation (openCV).ipynb", "train.py"],
    ),
    dict(
        num=42,
        folder="CV Projects 42 - Image resizing using opencv",
        reg_key="image_resizing_v2",
        display="Image Resizing (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="No-op (no model to load).",
        predict_summary="Downscales to 25%, then upscales back using 4 interpolation methods: NEAREST, LINEAR, CUBIC, LANCZOS4.",
        visualize_summary="2×2 grid: Nearest, Linear, Cubic, Lanczos4 — each labeled.",
        short_desc="Compares four OpenCV interpolation methods for image resizing: Nearest, Bilinear, Bicubic, and Lanczos4.",
        dataset_key=None,
        extra_files=["image resizing (OpenCV).ipynb"],
    ),
    dict(
        num=43,
        folder="CV Projects 43 - Funny Cartoonizing Images using openCV",
        reg_key="cartoon_effect_v2",
        display="Cartoon Effect (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="No-op (no model to load).",
        predict_summary="Grayscale + median blur + adaptive threshold for edges, 7 rounds of bilateral filter, combines via bitwise_and.",
        visualize_summary="Side-by-side: Original on left, Cartoon on right — each labeled.",
        short_desc="Creates cartoon-style images by combining edge detection with bilateral filtering for smooth color regions.",
        dataset_key=None,
        extra_files=["cartooning an image (OpenCV).ipynb"],
    ),
    dict(
        num=44,
        folder="CV Projects 44 - Joining Multiple Images to Display",
        reg_key="image_joining_v2",
        display="Image Joining (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="No-op (no model to load).",
        predict_summary="Creates 4 variants of frame: original, grayscale, Canny edges, Gaussian blur.",
        visualize_summary="2×2 grid: Original, Gray, Edges, Blurred — each labeled.",
        short_desc="Demonstrates how to join multiple processed images into a single display grid using OpenCV.",
        dataset_key=None,
        extra_files=["Joining_Multiple_Images_To_Display.py"],
    ),
    dict(
        num=45,
        folder="CV Projects 45 -  Detecting clicks on images",
        reg_key="click_detection_v2",
        display="Click Detection (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="Initializes empty `click_points` list.",
        predict_summary="Returns current click_points state (detection handled by cv2 mouse callbacks via `add_click()`).",
        visualize_summary="Draws red dots at each click point, labels with coordinates, shows total click count.",
        short_desc="Demonstrates OpenCV mouse callback events by detecting and displaying click coordinates on images.",
        dataset_key=None,
        extra_files=["Detecting_Clicks_On_Images.py"],
    ),
    dict(
        num=46,
        folder="CV Projects 46 - Face Detection Second Approach",
        reg_key="face_detection_haar_v2",
        display="Face Detection – Haar→YOLO",
        category="detection",
        framework="Ultralytics YOLO",
        uses_resolve=True,
        has_train=True,
        load_summary='Resolves weights via `models.registry.resolve("face_detection_haar", "detect")`, falls back to `yolo26n.pt`.',
        predict_summary="Runs YOLO detection on frame with confidence 0.5.",
        visualize_summary="Returns `output[0].plot()` — standard YOLO annotated frame.",
        short_desc="Face detection modernized from Haar cascades to YOLO26, with a training pipeline for custom face datasets.",
        dataset_key="face_detection_haar",
        extra_files=["face detection ( OpenCV).ipynb", "haarcascade_frontalface_default.xml", "train.py"],
    ),
    dict(
        num=47,
        folder="CV Projects 47 -  Face Mask Detection",
        reg_key="face_mask_detection_v2",
        display="Face Mask Detection (YOLO)",
        category="detection",
        framework="Ultralytics YOLO",
        uses_resolve=True,
        has_train=True,
        load_summary='Resolves weights via `models.registry.resolve("face_mask_detection", "detect")`, falls back to `yolo26n.pt`.',
        predict_summary="Runs YOLO detection filtered to class 0 (person) with confidence 0.5.",
        visualize_summary="Calls `output[0].plot()`, adds text noting fine-tuning needed for mask/no-mask classification.",
        short_desc="Detects faces with/without masks using YOLO26. Training pipeline includes VOC-to-YOLO conversion for the in-repo annotated dataset.",
        dataset_key="face_mask_detection",
        extra_files=["annotations/", "images/", "mask-detection (openCV).ipynb", "train.py"],
    ),
    dict(
        num=48,
        folder="CV Projects 48 - Face, Gender & Ethincity recognizer model",
        reg_key="face_attributes_v2",
        display="Face Attributes (YOLO + Torch)",
        category="detection",
        framework="Ultralytics YOLO",
        uses_resolve=True,
        has_train=True,
        load_summary='Resolves weights via `models.registry.resolve("face_attributes", "detect")`, falls back to `yolo26n.pt`.',
        predict_summary="Runs YOLO detection filtered to class 0 (person) with confidence 0.5.",
        visualize_summary="Calls `output[0].plot()`, adds text noting fine-tuned age/gender model needed.",
        short_desc="Detects faces and predicts age, gender, and ethnicity. Training pipeline uses ResNet-18 multi-output classification on UTKFace via torchvision.",
        dataset_key="face_attributes",
        extra_files=["keras-tuner-multi-output-cnn.ipynb", "train.py"],
    ),
    dict(
        num=49,
        folder="CV Projects 49- Real Time TextDetection",
        reg_key="text_detection_v2",
        display="Text Detection (YOLO)",
        category="detection",
        framework="Ultralytics YOLO",
        uses_resolve=True,
        has_train=True,
        load_summary='Resolves weights via `models.registry.resolve("text_detection", "detect")`, falls back to `yolo26n.pt`.',
        predict_summary="Runs YOLO detection on frame with confidence 0.3.",
        visualize_summary="Calls `output[0].plot()`, adds text noting PaddleOCR or fine-tuned YOLO recommended for actual text detection.",
        short_desc="Detects text regions in images using YOLO26, with a training pipeline for custom text-region datasets.",
        dataset_key="text_detection",
        extra_files=["TextSimple.py", "TextMoreExamples.py", "train.py"],
    ),
    dict(
        num=50,
        folder="CV Projects 50 - Reversing video using opencv",
        reg_key="video_reverse_v2",
        display="Video Reverse (v2)",
        category="utility",
        framework="OpenCV",
        uses_resolve=False,
        has_train=False,
        load_summary="Initializes frame buffer `deque` (max 60 frames), sets recording mode.",
        predict_summary="Records frames into buffer until full (60), then plays back in reverse order, cycles back to recording.",
        visualize_summary='Shows "Recording..." with red dot during recording, shows reverse playback frame during playback.',
        short_desc="Records short video segments and plays them back in reverse in real-time using a frame buffer.",
        dataset_key=None,
        extra_files=["reversing video (openCV).ipynb"],
    ),
]

# ============================================================================
# Dataset configs — which are auto-download vs manual
# ============================================================================
DATASET_CONFIGS = {
    "ball_tracking": ("roboflow", True),
    "blink_detection": ("manual", False),
    "car_detection": ("manual", False),
    "custom_object_detection": ("ultralytics", True),
    "face_attributes": ("manual", False),
    "face_detection": ("http", True),
    "face_detection_haar": ("http", True),
    "face_mask_detection": ("kaggle", True),
    "facial_landmarks": ("manual", False),
    "finger_counter": ("manual", False),
    "hand_tracking": ("manual", False),
    "image_segmentation": ("manual", False),
    "object_detection": ("http", True),
    "pose_detector": ("http", True),
    "sudoku_solver": ("http", True),
    "text_detection": ("http", True),
    "volume_controller": ("manual", False),
}


# ============================================================================
# GENERATORS
# ============================================================================

def _gen_project_readme(p: dict) -> str:
    """Generate README.md content for a single project."""
    num = p["num"]
    folder = p["folder"]
    display = p["display"]
    category = p["category"]
    framework = p["framework"]
    has_train = p["has_train"]
    uses_resolve = p["uses_resolve"]
    reg_key = p["reg_key"]
    desc = p["short_desc"]
    dataset_key = p["dataset_key"]

    badge_fw = f"![{framework}](https://img.shields.io/badge/Framework-{framework.replace(' ', '_')}-blue)"
    cat_label = category.replace("_", " ").title()
    badge_cat = f"![{cat_label}](https://img.shields.io/badge/Task-{cat_label.replace(' ', '_')}-green)"
    badges = f"{badge_fw} {badge_cat}"
    if has_train:
        badges += " ![Trainable](https://img.shields.io/badge/Trainable-yes-orange)"

    lines = []
    lines.append(f"# P{num}: {display}")
    lines.append("")
    lines.append(badges)
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(desc)
    lines.append("")

    # Entry points
    lines.append("## Entry Points")
    lines.append("")
    lines.append("### Run / Inference")
    lines.append("")
    if reg_key:
        lines.append("**Via the unified runner:**")
        lines.append("")
        lines.append("```bash")
        lines.append(f"python -m core.runner {reg_key} --source 0 --import-all")
        lines.append("```")
        lines.append("")
        lines.append("**Programmatic usage:**")
        lines.append("")
        lines.append("```python")
        lines.append("from core.runner import run_camera")
        lines.append(f'run_camera("{reg_key}", source=0)')
        lines.append("```")
    else:
        lines.append("This project's `modern.py` is not yet implemented. Use the legacy scripts directly.")
    lines.append("")

    if has_train:
        lines.append("### Train")
        lines.append("")
        if dataset_key and dataset_key in DATASET_CONFIGS:
            lines.append("```bash")
            lines.append(f'cd "{folder}"')
            lines.append("python train.py --data path/to/data.yaml")
            lines.append("```")
        else:
            lines.append("```bash")
            lines.append(f'cd "{folder}"')
            lines.append("python train.py --data path/to/data.yaml")
            lines.append("```")
        lines.append("")
        lines.append("Training registers the resulting model version in the model registry")
        lines.append("(`models/metadata.json`) and auto-promotes based on the primary metric.")
        lines.append("")

    # Model resolution
    if uses_resolve:
        lines.append("## Model Resolution")
        lines.append("")
        lines.append(p["load_summary"])
        lines.append("")
        lines.append("The model registry (`models/registry.py`) resolves weights in this order:")
        lines.append("")
        lines.append("1. **Trained model** — `models/<project>/<version>/best.pt` if registered and file exists")
        lines.append("2. **YOLO26 pretrained fallback** — auto-downloaded by Ultralytics on first use")
        lines.append("")

    # Dataset
    if dataset_key and dataset_key in DATASET_CONFIGS:
        method, auto = DATASET_CONFIGS[dataset_key]
        lines.append("## Dataset")
        lines.append("")
        lines.append(f"Configuration: `configs/datasets/{dataset_key}.yaml`")
        lines.append("")
        if auto:
            lines.append(f"Download method: **{method}** (auto-download enabled)")
            lines.append("")
            lines.append("```bash")
            lines.append(f"python -m utils.data_downloader --project {dataset_key}")
            lines.append("```")
        else:
            lines.append(f"Download method: **{method}** (manual download required — see URL in config)")
        lines.append("")
        lines.append("Expected layout after download:")
        lines.append("")
        lines.append("```")
        lines.append(f"data/{dataset_key}/")
        lines.append("  data.yaml")
        lines.append("  train/images/")
        lines.append("  valid/images/")
        lines.append("```")
        lines.append("")

    # Pipeline
    lines.append("## Processing Pipeline")
    lines.append("")
    if uses_resolve:
        # Avoid duplicating Model Resolution; just reference it
        lines.append("- **Load**: See [Model Resolution](#model-resolution) above.")
    else:
        lines.append(f"- **Load**: {p['load_summary']}")
    lines.append(f"- **Predict**: {p['predict_summary']}")
    lines.append(f"- **Visualize**: {p['visualize_summary']}")
    lines.append("")

    # Outputs
    is_empty = reg_key is None  # empty modern.py (e.g. P37)
    lines.append("## Outputs")
    lines.append("")
    if is_empty:
        lines.append("- **modern.py is empty** — no runner output.")
        if has_train:
            lines.append("- Training: `runs/detect/train/weights/best.pt` (registered in model registry)")
    else:
        lines.append("- OpenCV display window showing annotated frames in real-time")
        if has_train:
            lines.append("- Training: `runs/detect/train/weights/best.pt` (registered in model registry)")
        lines.append("- Press `q` to quit the camera loop")
    lines.append("")

    # Troubleshooting
    lines.append("## Troubleshooting")
    lines.append("")
    lines.append("| Issue | Solution |")
    lines.append("|-------|----------|")
    if not is_empty:
        lines.append("| `Cannot open video source` | Check webcam index — try `--source 1` |")
    if uses_resolve:
        lines.append("| Model downloads on first run | Normal — Ultralytics auto-downloads YOLO26 pretrained weights |")
    if dataset_key:
        lines.append(f"| Dataset not found | Run `python -m utils.data_downloader --project {dataset_key}` |")
    if has_train:
        lines.append("| Training OOM | Reduce `--batch` or use a smaller model (`yolo26n.pt`) |")
    lines.append("")

    # Testing
    lines.append("## Testing")
    lines.append("")
    lines.append("No project-level test suite. Use workspace-level smoke tests:")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/smoke_3b3.py")
    lines.append("```")

    return "\n".join(lines) + "\n"


def _gen_root_readme() -> str:
    """Generate root README.md."""
    # Count categories
    detect_count = sum(1 for p in PROJECTS if p["category"] == "detection")
    pose_count = sum(1 for p in PROJECTS if p["category"] == "pose")
    seg_count = sum(1 for p in PROJECTS if p["category"] == "segmentation")
    cls_count = sum(1 for p in PROJECTS if p["category"] == "classification")
    util_count = sum(1 for p in PROJECTS if p["category"] == "utility")
    train_count = sum(1 for p in PROJECTS if p["has_train"])
    resolve_count = sum(1 for p in PROJECTS if p["uses_resolve"])

    lines = []
    lines.append("# Computer Vision Projects")
    lines.append("")
    lines.append("A collection of **50 independent computer vision projects** sharing a unified")
    lines.append("platform for model management, training, benchmarking, and dataset handling.")
    lines.append("")
    lines.append("Each project wraps its core logic in a `modern.py` file that subclasses")
    lines.append("`core.base.CVProject` and registers with `core.registry`. This enables")
    lines.append("a single runner (`core.runner`) to load and execute any project by name.")
    lines.append("")
    lines.append("## Tech Stack")
    lines.append("")
    lines.append("| Component | Version |")
    lines.append("|-----------|---------|")
    lines.append("| Python | ≥ 3.13 |")
    lines.append("| OpenCV | ≥ 4.10 (4.12 recommended) |")
    lines.append("| PyTorch | ≥ 2.10 (CUDA 13.0) |")
    lines.append("| Ultralytics | ≥ 8.4 (YOLO26) |")
    lines.append("")
    lines.append("See [requirements.txt](requirements.txt) for the full dependency list")
    lines.append("and [requirements-lock.txt](requirements-lock.txt) for pinned versions.")
    lines.append("")

    # Architecture diagram
    lines.append("## Architecture")
    lines.append("")
    lines.append("```mermaid")
    lines.append("graph TD")
    lines.append("    A[core.runner CLI] -->|--import-all| B[core.registry]")
    lines.append("    B -->|discover| C[50 x modern.py<br>CVProject subclasses]")
    lines.append("    C -->|load| D{models.registry.resolve}")
    lines.append("    D -->|trained model exists| E[models/project/version/best.pt]")
    lines.append("    D -->|no trained model| F[YOLO26 pretrained fallback]")
    lines.append("    C -->|predict + visualize| G[OpenCV display / output]")
    lines.append("```")
    lines.append("")

    # Training flow diagram
    lines.append("## Training Flow")
    lines.append("")
    lines.append("```mermaid")
    lines.append("graph LR")
    lines.append("    A[configs/datasets/*.yaml] --> B[utils.data_downloader]")
    lines.append("    B -->|auto-download or manual| C[data/project_key/]")
    lines.append("    C --> D[train/*.py or project/train.py]")
    lines.append("    D --> E[models.registry.register]")
    lines.append("    E --> F[models/project/version/best.pt]")
    lines.append("```")
    lines.append("")

    # Accuracy evaluation flow
    lines.append("## Accuracy Evaluation Flow")
    lines.append("")
    lines.append("```mermaid")
    lines.append("graph LR")
    lines.append("    A[configs/datasets/*.yaml] --> B[ensure_dataset<br>NO-SKIP policy]")
    lines.append("    B --> C[models.registry.resolve]")
    lines.append("    C --> D[benchmarks.evaluate_accuracy]")
    lines.append("    D --> E[CSV + JSON results]")
    lines.append("    D -.->|--write-registry| F[models/metadata.json]")
    lines.append("```")
    lines.append("")

    # Quick start
    lines.append("## Quick Start")
    lines.append("")
    lines.append("### 1. Environment Setup")
    lines.append("")
    lines.append("**Windows (PowerShell):**")
    lines.append("")
    lines.append("```powershell")
    lines.append(".\\scripts\\setup_env.ps1")
    lines.append("```")
    lines.append("")
    lines.append("**Linux / macOS:**")
    lines.append("")
    lines.append("```bash")
    lines.append("bash scripts/setup_env.sh")
    lines.append("```")
    lines.append("")
    lines.append("**Manual (GPU):**")
    lines.append("")
    lines.append("```bash")
    lines.append("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130")
    lines.append("pip install -r requirements.txt")
    lines.append("```")
    lines.append("")

    lines.append("### 2. Smoke Test")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/smoke_3b3.py")
    lines.append("```")
    lines.append("")
    lines.append("Runs 9 structural checks (registry imports, YOLO26 defaults, AST parse, etc.).")
    lines.append("")

    lines.append("### 3. Run a Project")
    lines.append("")
    lines.append("```bash")
    lines.append("# Run any project by its registered name")
    lines.append("python -m core.runner face_detection_v2 --source 0 --import-all")
    lines.append("")
    lines.append("# List all registered projects")
    lines.append("python -m core.runner --list --import-all")
    lines.append("```")
    lines.append("")

    lines.append("### 4. Train a Project")
    lines.append("")
    lines.append("```bash")
    lines.append("# Download dataset first")
    lines.append("python -m utils.data_downloader --project face_mask_detection")
    lines.append("")
    lines.append("# Train (from project folder)")
    lines.append('cd "CV Projects 47 -  Face Mask Detection"')
    lines.append("python train.py --data ../data/face_mask_detection/data.yaml")
    lines.append("```")
    lines.append("")

    lines.append("### 5. Performance Benchmark")
    lines.append("")
    lines.append("```bash")
    lines.append("python benchmarks/run_all.py")
    lines.append("```")
    lines.append("")
    lines.append("Measures per-project load time, average latency, FPS, and memory usage.")
    lines.append("Results written to `benchmarks/results.csv`.")
    lines.append("")

    lines.append("### 6. Accuracy Evaluation")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m benchmarks.evaluate_accuracy")
    lines.append("python -m benchmarks.evaluate_accuracy --write-registry  # save metrics back")
    lines.append("```")
    lines.append("")
    lines.append("Evaluates model accuracy for all trainable projects. Uses the **no-skip policy**:")
    lines.append("missing datasets trigger auto-download attempts; failures report `download_failed`")
    lines.append("instead of silently skipping.")
    lines.append("")

    # Repo structure
    lines.append("## Repository Structure")
    lines.append("")
    lines.append("```")
    lines.append("Computer-Vision-Projects/")
    lines.append("├── core/                    # Unified runner engine")
    lines.append("│   ├── base.py              # CVProject abstract base class")
    lines.append("│   ├── registry.py          # PROJECT_REGISTRY + @register decorator")
    lines.append("│   └── runner.py            # run(), run_camera(), CLI")
    lines.append("├── models/                  # Model registry + trained weights")
    lines.append("│   ├── registry.py          # ModelRegistry, resolve(), YOLO26_DEFAULTS")
    lines.append("│   └── metadata.json        # Version tracking (auto-generated)")
    lines.append("├── train/                   # Shared training pipelines")
    lines.append("│   ├── train_detection.py   # YOLO detection + VOC→YOLO converter")
    lines.append("│   ├── train_classification.py  # torchvision transfer learning")
    lines.append("│   └── train_segmentation.py    # YOLO-Seg + DeepLabV3")
    lines.append("├── benchmarks/              # Performance & accuracy evaluation")
    lines.append("│   ├── run_all.py           # Latency / FPS / memory benchmark")
    lines.append("│   └── evaluate_accuracy.py # Accuracy eval with no-skip policy")
    lines.append("├── configs/datasets/        # 17 dataset YAML configs")
    lines.append("├── utils/                   # Shared utilities")
    lines.append("│   ├── yolo.py              # YOLO loader (cached)")
    lines.append("│   ├── data_downloader.py   # Central dataset download engine")
    lines.append("│   ├── data_resolver.py     # Asset resolution (project → data → models)")
    lines.append("│   ├── datasets.py          # Dataset registry + auto-download")
    lines.append("│   └── paths.py             # PathResolver")
    lines.append("├── scripts/                 # Setup, CI, and utilities")
    lines.append("│   ├── setup_env.ps1        # Windows environment setup")
    lines.append("│   ├── setup_env.sh         # Linux/macOS environment setup")
    lines.append("│   ├── smoke_3b3.py         # 9-check smoke test")
    lines.append("│   ├── ci_sanity.py         # 5-check CI validation")
    lines.append("│   └── check_large_files.py # Pre-commit size guard (50 MB)")
    lines.append("├── .githooks/               # Pre-commit hooks (size guard + smoke)")
    lines.append("├── CV Project */            # 50 project folders")
    lines.append("│   ├── modern.py            # CVProject wrapper (standardized)")
    lines.append("│   ├── train.py             # Per-project training (if trainable)")
    lines.append("│   └── README.md            # Per-project documentation")
    lines.append("├── requirements.txt         # Dependency ranges")
    lines.append("├── requirements-lock.txt    # Pinned versions")
    lines.append("└── WORKSPACE_OVERVIEW.md    # Project catalogue + stats")
    lines.append("```")
    lines.append("")

    # Dataset policy
    lines.append("## Dataset Policy")
    lines.append("")
    lines.append("All dataset files are **gitignored** (see `.gitignore`). Dataset metadata lives in")
    lines.append("`configs/datasets/*.yaml` (17 configs). To download datasets:")
    lines.append("")
    lines.append("```bash")
    lines.append("# Download all enabled datasets")
    lines.append("python -m utils.data_downloader --all")
    lines.append("")
    lines.append("# Download a specific dataset")
    lines.append("python -m utils.data_downloader --project face_mask_detection")
    lines.append("")
    lines.append("# Dry-run (show what would be downloaded)")
    lines.append("python -m utils.data_downloader --all --dry-run")
    lines.append("```")
    lines.append("")

    # Large file policy
    lines.append("## Large File Policy")
    lines.append("")
    lines.append("Files > 50 MB are blocked by the pre-commit hook (`.githooks/pre-commit`).")
    lines.append("Model weights (`.pt`, `.pth`, `.h5`, etc.) are gitignored by default.")
    lines.append("Configure the hook path with:")
    lines.append("")
    lines.append("```bash")
    lines.append("git config core.hooksPath .githooks")
    lines.append("```")
    lines.append("")

    # Model registry
    lines.append("## Model Registry")
    lines.append("")
    lines.append("The model registry (`models/registry.py`) tracks trained model versions:")
    lines.append("")
    lines.append("```bash")
    lines.append("# List registered models")
    lines.append("python -m models.registry list")
    lines.append("")
    lines.append("# Show active model for a project")
    lines.append("python -m models.registry active face_mask_detection")
    lines.append("")
    lines.append("# Resolve weights (API)")
    lines.append("from models.registry import resolve")
    lines.append('weights, version, is_default = resolve("face_mask_detection", "detect")')
    lines.append("```")
    lines.append("")

    # Project index table
    lines.append("## Project Index")
    lines.append("")
    # Build category summary, omitting zero-count categories
    _cat_parts = []
    for _label, _count in [("detection", detect_count), ("pose", pose_count),
                            ("segmentation", seg_count), ("classification", cls_count),
                            ("utility", util_count)]:
        if _count > 0:
            _cat_parts.append(f"{_count} {_label}")
    _cat_summary = ", ".join(_cat_parts)
    lines.append(f"**{len(PROJECTS)} projects** — {_cat_summary}")
    lines.append(f"— {train_count} trainable, {resolve_count} using model registry")
    lines.append("")
    lines.append("| # | Project | Category | Framework | Trainable |")
    lines.append("|---|---------|----------|-----------|-----------|")
    for p in PROJECTS:
        num = p["num"]
        train_mark = "Yes" if p["has_train"] else "—"
        cat = p["category"].replace("_", " ").title()
        fw = p["framework"]
        # Make folder link
        folder_safe = p["folder"].replace(" ", "%20")
        link = f'[{p["display"]}]({folder_safe}/)'
        lines.append(f"| {num} | {link} | {cat} | {fw} | {train_mark} |")
    lines.append("")

    return "\n".join(lines) + "\n"


def _gen_workspace_overview() -> str:
    """Generate WORKSPACE_OVERVIEW.md."""
    detect = [p for p in PROJECTS if p["category"] == "detection"]
    pose = [p for p in PROJECTS if p["category"] == "pose"]
    seg = [p for p in PROJECTS if p["category"] == "segmentation"]
    cls = [p for p in PROJECTS if p["category"] == "classification"]
    util = [p for p in PROJECTS if p["category"] == "utility"]

    auto_dl = sum(1 for _, (_, a) in DATASET_CONFIGS.items() if a)
    manual_dl = sum(1 for _, (_, a) in DATASET_CONFIGS.items() if not a)

    lines = []
    lines.append("# Workspace Overview")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total projects**: {len(PROJECTS)}")
    lines.append(f"- **Trainable projects**: {sum(1 for p in PROJECTS if p['has_train'])}")
    lines.append(f"- **Projects using model registry**: {sum(1 for p in PROJECTS if p['uses_resolve'])}")
    lines.append(f"- **Dataset configs**: {len(DATASET_CONFIGS)}")
    lines.append(f"  - Auto-download capable: {auto_dl}")
    lines.append(f"  - Manual download required: {manual_dl}")
    lines.append("")

    lines.append("## Projects by Task Category")
    lines.append("")

    for label, group in [("Detection", detect), ("Pose Estimation", pose),
                          ("Segmentation", seg), ("Classification", cls),
                          ("Utility / OpenCV", util)]:
        if not group:
            continue
        lines.append(f"### {label} ({len(group)})")
        lines.append("")
        lines.append("| # | Project | Framework | Trainable |")
        lines.append("|---|---------|-----------|-----------|")
        for p in group:
            train_mark = "Yes" if p["has_train"] else "—"
            lines.append(f"| {p['num']} | {p['display']} | {p['framework']} | {train_mark} |")
        lines.append("")

    lines.append("## Dataset Status")
    lines.append("")
    lines.append("| Project Key | Method | Auto-Download |")
    lines.append("|-------------|--------|---------------|")
    for key in sorted(DATASET_CONFIGS.keys()):
        method, auto = DATASET_CONFIGS[key]
        auto_str = "Yes" if auto else "No"
        lines.append(f"| {key} | {method} | {auto_str} |")
    lines.append("")

    lines.append("## Commands Index")
    lines.append("")
    lines.append("| Task | Command |")
    lines.append("|------|---------|")
    lines.append("| Smoke test | `python scripts/smoke_3b3.py` |")
    lines.append("| CI sanity (5 checks) | `python scripts/ci_sanity.py --verbose` |")
    lines.append("| Performance benchmark | `python benchmarks/run_all.py` |")
    lines.append("| Accuracy evaluation | `python -m benchmarks.evaluate_accuracy` |")
    lines.append("| Download all datasets | `python -m utils.data_downloader --all` |")
    lines.append("| Download one dataset | `python -m utils.data_downloader --project <key>` |")
    lines.append("| List registered models | `python -m models.registry list` |")
    lines.append("| Run a project | `python -m core.runner <name> --source 0 --import-all` |")
    lines.append("| List all projects | `python -m core.runner --list --import-all` |")
    lines.append("")

    return "\n".join(lines) + "\n"


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate final documentation")
    parser.add_argument("--force", action="store_true", help="Overwrite existing READMEs")
    args = parser.parse_args()

    # 1. Per-project READMEs
    written = 0
    skipped = 0
    for p in PROJECTS:
        folder = ROOT / p["folder"]
        if not folder.is_dir():
            print(f"  [SKIP] Folder not found: {p['folder']}")
            skipped += 1
            continue
        readme_path = folder / "README.md"
        if readme_path.exists() and not args.force:
            print(f"  [SKIP] Already exists: {readme_path.relative_to(ROOT)}")
            skipped += 1
            continue
        content = _gen_project_readme(p)
        readme_path.write_text(content, encoding="utf-8")
        written += 1

    print(f"\nPer-project READMEs: {written} written, {skipped} skipped")

    # 2. Root README.md
    root_readme = ROOT / "README.md"
    content = _gen_root_readme()
    root_readme.write_text(content, encoding="utf-8")
    print(f"Root README.md written ({len(content)} bytes)")

    # 3. WORKSPACE_OVERVIEW.md
    overview_path = ROOT / "WORKSPACE_OVERVIEW.md"
    content = _gen_workspace_overview()
    overview_path.write_text(content, encoding="utf-8")
    print(f"WORKSPACE_OVERVIEW.md written ({len(content)} bytes)")

    print("\nDone!")


if __name__ == "__main__":
    main()
