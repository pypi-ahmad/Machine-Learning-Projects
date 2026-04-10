#!/usr/bin/env python
"""
Generate README.md for each of the 50 CV project folders.

Each README includes:
  - Project title & description
  - Mermaid pipeline diagram
  - Quick start instructions
  - Tech stack
  - File listing
  - Links to shared infrastructure

Run:
    python scripts/_gen_project_readmes.py
    python scripts/_gen_project_readmes.py --force   # overwrite existing
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Project metadata ────────────────────────────────────────────────────────

PROJECTS = [
    {
        "num": 1,
        "folder": "CV Project 1 -  Real Time Angle Detector",
        "title": "Real-Time Angle Detector",
        "desc": "Detect and measure angles in real time using OpenCV contour analysis and trigonometry.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Webcam / Image",
        "output": "Annotated frame with angle overlay",
        "steps": ["Capture frame", "Edge detection (Canny)", "Find contours", "Compute angle with atan2", "Draw angle annotation"],
    },
    {
        "num": 2,
        "folder": "CV Project 2 - Real Time Document Scanner-fine",
        "title": "Real-Time Document Scanner",
        "desc": "Automatically detect document edges and apply perspective warp to produce a scanned document effect.",
        "type": "opencv",
        "libs": ["OpenCV", "scikit-image"],
        "input": "Webcam / Image",
        "output": "Warped top-down document view",
        "steps": ["Capture frame", "Edge detection", "Find largest contour", "Order corner points", "Perspective warp"],
    },
    {
        "num": 3,
        "folder": "CV Project 3 - Real Time Face detector Image",
        "title": "Real-Time Face Detector",
        "desc": "Detect faces in images and video using YOLO26 object detection model.",
        "type": "yolo",
        "task": "detect",
        "libs": ["Ultralytics (YOLO26)", "OpenCV"],
        "input": "Webcam / Image",
        "output": "Bounding boxes around detected faces",
        "steps": ["Capture frame", "YOLO26 inference", "Draw bounding boxes", "Display confidence scores"],
    },
    {
        "num": 4,
        "folder": "CV Project 4 - Facial Landmarking",
        "title": "Facial Landmarking",
        "desc": "Detect facial landmarks (68-point) using YOLO26 pose estimation with dlib integration.",
        "type": "yolo",
        "task": "pose",
        "libs": ["Ultralytics (YOLO26)", "dlib", "OpenCV"],
        "input": "Webcam / Image",
        "output": "Face with landmark points overlay",
        "steps": ["Capture frame", "YOLO26 pose inference", "Extract keypoints", "Draw landmark points", "Display annotated frame"],
    },
    {
        "num": 5,
        "folder": "CV Projects 5 - fingerCounter",
        "title": "Finger Counter",
        "desc": "Count raised fingers using YOLO26 hand pose estimation and MediaPipe.",
        "type": "yolo",
        "task": "pose",
        "libs": ["Ultralytics (YOLO26)", "MediaPipe", "OpenCV"],
        "input": "Webcam",
        "output": "Finger count overlay on frame",
        "steps": ["Capture frame", "Hand detection", "Landmark extraction", "Finger state analysis", "Display count"],
    },
    {
        "num": 6,
        "folder": "CV Project 6 - Live HTM",
        "title": "Live Hand Tracking + Volume Control",
        "desc": "Control system volume by tracking hand distance between thumb and index finger using YOLO26 pose.",
        "type": "yolo",
        "task": "pose",
        "libs": ["Ultralytics (YOLO26)", "MediaPipe", "pycaw", "OpenCV"],
        "input": "Webcam",
        "output": "Volume bar + hand tracking overlay",
        "steps": ["Capture frame", "Hand pose estimation", "Measure finger distance", "Map to volume level", "Set system volume"],
    },
    {
        "num": 7,
        "folder": "CV Project 7 - Real Time Object Size detector",
        "title": "Real-Time Object Size Detector",
        "desc": "Measure real-world object dimensions using a reference object and contour analysis.",
        "type": "opencv",
        "libs": ["OpenCV", "scipy"],
        "input": "Image with reference object",
        "output": "Annotated image with measurements",
        "steps": ["Load image", "Edge detection", "Find contours", "Order points", "Compute pixel-per-metric ratio", "Measure objects"],
    },
    {
        "num": 8,
        "folder": "CV Project 8 - OMR Evaluator",
        "title": "OMR (Optical Mark Recognition) Evaluator",
        "desc": "Automatically grade bubble-sheet answer forms using contour detection and thresholding.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "OMR sheet image",
        "output": "Graded sheet with correct/incorrect marks",
        "steps": ["Load image", "Detect sheet edges", "Perspective warp", "Find bubble contours", "Threshold & evaluate marks", "Compare with answer key"],
    },
    {
        "num": 9,
        "folder": "CV Project 9 - Real Time Painter of Camera Screen",
        "title": "Real-Time Painter",
        "desc": "Draw on the screen in real time using a colored object tracked by the webcam.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Webcam + colored object",
        "output": "Paint strokes overlaid on video",
        "steps": ["Capture frame", "HSV color masking", "Find contour centroid", "Track position", "Draw on canvas"],
    },
    {
        "num": 10,
        "folder": "CV Project 10 - Live PoseDetector",
        "title": "Live Pose Detector",
        "desc": "Detect human body pose in real time using YOLO26 pose estimation model.",
        "type": "yolo",
        "task": "pose",
        "libs": ["Ultralytics (YOLO26)", "MediaPipe", "OpenCV"],
        "input": "Webcam",
        "output": "Skeleton overlay on body",
        "steps": ["Capture frame", "YOLO26 pose inference", "Extract keypoints", "Draw skeleton connections", "Display FPS"],
    },
    {
        "num": 11,
        "folder": "CV Project 11 - Live QR-Reader",
        "title": "Live QR Code Reader",
        "desc": "Scan and decode QR codes in real time from webcam feed.",
        "type": "opencv",
        "libs": ["OpenCV", "pyzbar"],
        "input": "Webcam",
        "output": "Decoded QR data overlay",
        "steps": ["Capture frame", "Convert to grayscale", "Detect QR codes (pyzbar)", "Decode data", "Draw bounding polygon + text"],
    },
    {
        "num": 12,
        "folder": "CV Project 12 - Real Time Object Detection-fine",
        "title": "Real-Time Object Detection",
        "desc": "Detect 80+ object classes in real time using YOLO26 detection model.",
        "type": "yolo",
        "task": "detect",
        "libs": ["Ultralytics (YOLO26)", "OpenCV"],
        "input": "Webcam / Video",
        "output": "Bounding boxes with class labels",
        "steps": ["Capture frame", "YOLO26 inference", "NMS filtering", "Draw boxes + labels", "Display FPS"],
    },
    {
        "num": 13,
        "folder": "CV Project 13 - Real Time Sudoku Solver",
        "title": "Real-Time Sudoku Solver",
        "desc": "Detect a Sudoku grid from camera, recognize digits with YOLO26, and solve the puzzle in real time.",
        "type": "yolo",
        "task": "cls",
        "libs": ["Ultralytics (YOLO26)", "OpenCV"],
        "input": "Webcam / Image of Sudoku",
        "output": "Solved Sudoku overlaid on original",
        "steps": ["Capture frame", "Detect grid contour", "Warp perspective", "Extract cells", "Classify digits (YOLO26)", "Solve with backtracking", "Overlay solution"],
    },
    {
        "num": 14,
        "folder": "CV Project 14 - click-detect on image",
        "title": "Click Detection on Image",
        "desc": "Detect mouse clicks on image regions and identify perspective-warped areas.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Image + mouse clicks",
        "output": "Warped region view",
        "steps": ["Load image", "Register mouse callback", "Detect clicked region", "Apply perspective warp", "Display warped view"],
    },
    {
        "num": 15,
        "folder": "CV Project 15 - Live Image Cartoonifier",
        "title": "Live Image Cartoonifier",
        "desc": "Apply cartoon effect to live webcam feed using bilateral filtering and edge detection.",
        "type": "opencv",
        "libs": ["OpenCV", "scikit-image"],
        "input": "Webcam",
        "output": "Cartoon-style video",
        "steps": ["Capture frame", "Edge detection", "Bilateral filter (smoothing)", "Combine edges with filtered image", "Display cartoon effect"],
    },
    {
        "num": 16,
        "folder": "CV Project 16 -Live Car-Detection",
        "title": "Live Car Detection",
        "desc": "Detect vehicles in video using YOLO26 object detection.",
        "type": "yolo",
        "task": "detect",
        "libs": ["Ultralytics (YOLO26)", "OpenCV"],
        "input": "Video file",
        "output": "Annotated video with car bounding boxes",
        "steps": ["Load video", "YOLO26 inference per frame", "Filter car class", "Draw bounding boxes", "Display annotated video"],
    },
    {
        "num": 17,
        "folder": "CV Project 17 - Blink Detection",
        "title": "Blink Detection",
        "desc": "Detect eye blinks in real time using YOLO26 pose estimation and eye aspect ratio.",
        "type": "yolo",
        "task": "pose",
        "libs": ["Ultralytics (YOLO26)", "dlib", "OpenCV"],
        "input": "Webcam",
        "output": "Blink count overlay",
        "steps": ["Capture frame", "Face/eye landmark detection", "Compute Eye Aspect Ratio (EAR)", "Threshold for blink", "Increment counter"],
    },
    {
        "num": 18,
        "folder": "CV Project 18 - Live Ball Tracking",
        "title": "Live Ball Tracking",
        "desc": "Track a colored ball in real time using YOLO26 detection.",
        "type": "yolo",
        "task": "detect",
        "libs": ["Ultralytics (YOLO26)", "OpenCV"],
        "input": "Webcam",
        "output": "Trail overlay following ball",
        "steps": ["Capture frame", "YOLO26 detection", "Track ball centroid", "Draw trail", "Display tracking"],
    },
    {
        "num": 19,
        "folder": "CV Projects 19 - GrayScaleConverter",
        "title": "Grayscale Converter",
        "desc": "Convert color images to grayscale using OpenCV color space conversion.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Webcam / Image",
        "output": "Grayscale image",
        "steps": ["Load/capture image", "Convert BGR→GRAY", "Display result"],
    },
    {
        "num": 20,
        "folder": "CV Projects 20 - image_finder",
        "title": "Image Finder (Template Matching)",
        "desc": "Find a template image within a larger image using template matching.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Template + scene image",
        "output": "Detected region highlighted",
        "steps": ["Load template and scene", "Template matching", "Find best match location", "Draw rectangle", "Display result"],
    },
    {
        "num": 21,
        "folder": "CV Projects 21 - VolumeController",
        "title": "Volume Controller",
        "desc": "Control system volume using hand gesture recognition via YOLO26 pose and MediaPipe.",
        "type": "yolo",
        "task": "pose",
        "libs": ["Ultralytics (YOLO26)", "MediaPipe", "pycaw", "OpenCV"],
        "input": "Webcam",
        "output": "Volume bar + gesture overlay",
        "steps": ["Capture frame", "Hand pose detection", "Measure thumb-index distance", "Map to volume", "Set system volume"],
    },
    {
        "num": 22,
        "folder": "CV Projects 22 - Live Color Picker",
        "title": "Live Color Picker",
        "desc": "Pick HSV color values from live webcam feed using trackbars.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Webcam",
        "output": "HSV mask preview + values",
        "steps": ["Capture frame", "Create HSV trackbars", "Convert to HSV", "Apply mask", "Display mask + values"],
    },
    {
        "num": 23,
        "folder": "CV Projects 23 - Crop Resize Image using OpenCV",
        "title": "Crop & Resize Image",
        "desc": "Crop and resize images using OpenCV array slicing and resize functions.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Image",
        "output": "Cropped and resized images",
        "steps": ["Load image", "Define ROI", "Crop via slicing", "Resize with interpolation", "Display results"],
    },
    {
        "num": 24,
        "folder": "CV Projects 24 - Custom Object Detection",
        "title": "Custom Object Detection",
        "desc": "Train and run custom object detection using YOLO26 with user-defined classes.",
        "type": "yolo",
        "task": "detect",
        "libs": ["Ultralytics (YOLO26)", "OpenCV"],
        "input": "Custom dataset / Webcam",
        "output": "Detected custom objects",
        "steps": ["Prepare dataset", "Train YOLO26 model", "Load trained weights", "Run inference", "Draw detections"],
    },
    {
        "num": 25,
        "folder": "CV Projects 25 - Real Time Object Measurement",
        "title": "Real-Time Object Measurement",
        "desc": "Measure object dimensions in real time using contour analysis and a reference object.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Webcam / Image",
        "output": "Measurements overlaid on objects",
        "steps": ["Capture frame", "Edge detection", "Find contours", "Compute pixel-per-metric", "Annotate dimensions"],
    },
    {
        "num": 26,
        "folder": "CV Projects 26 - Real Time Color Detection",
        "title": "Real-Time Color Detection",
        "desc": "Detect and track specific colors in real time using HSV masking.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Webcam",
        "output": "Highlighted color regions",
        "steps": ["Capture frame", "Convert to HSV", "Define color range", "Apply mask", "Find and draw contours"],
    },
    {
        "num": 27,
        "folder": "CV Projects 27 - Real Time Shape Detection",
        "title": "Real-Time Shape Detection",
        "desc": "Detect geometric shapes (triangle, rectangle, circle, etc.) using contour approximation.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Webcam",
        "output": "Shape labels on detected regions",
        "steps": ["Capture frame", "Threshold", "Find contours", "Approximate polygon", "Classify by vertex count", "Label shape"],
    },
    {
        "num": 28,
        "folder": "CV Projects 28 - Water Marking on Image using OpenCV",
        "title": "Watermarking on Image",
        "desc": "Apply text or image watermarks to photos using OpenCV blending.",
        "type": "opencv",
        "libs": ["OpenCV", "matplotlib"],
        "input": "Image + watermark",
        "output": "Watermarked image",
        "steps": ["Load image and watermark", "Resize watermark", "Create ROI", "Blend with addWeighted", "Display result"],
    },
    {
        "num": 29,
        "folder": "CV Projects 29 - Live Virtual Pen",
        "title": "Live Virtual Pen",
        "desc": "Draw on screen by tracking a colored pen/object in real time.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Webcam + colored object",
        "output": "Drawing overlay on video",
        "steps": ["Capture frame", "HSV color detection", "Track object centroid", "Draw line segments", "Overlay on video"],
    },
    {
        "num": 30,
        "folder": "CV projects 30 - Contrast enhancing of color images",
        "title": "Contrast Enhancing (Color Images)",
        "desc": "Enhance contrast of color images using histogram equalization and CLAHE.",
        "type": "opencv",
        "libs": ["OpenCV", "matplotlib"],
        "input": "Color image",
        "output": "Enhanced image + histogram comparison",
        "steps": ["Load image", "Convert to LAB/YCrCb", "Apply CLAHE to luminance", "Merge channels", "Display comparison"],
    },
    {
        "num": 31,
        "folder": "CV Projects 31 - contrast enhancing of gray scale image using opencv",
        "title": "Contrast Enhancing (Grayscale)",
        "desc": "Enhance contrast of grayscale images using histogram equalization.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Grayscale image",
        "output": "Enhanced image + histogram",
        "steps": ["Load image", "Convert to grayscale", "Apply equalizeHist", "Display before/after"],
    },
    {
        "num": 32,
        "folder": "CV Projects 32 - Draw vertical lines of coin",
        "title": "Draw Vertical Lines on Coin",
        "desc": "Detect coins using Hough Circle Transform and draw vertical cross-section lines.",
        "type": "opencv",
        "libs": ["OpenCV", "matplotlib"],
        "input": "Coin image",
        "output": "Annotated coin image",
        "steps": ["Load image", "Convert to grayscale", "Detect circles (HoughCircles)", "Draw vertical lines", "Display result"],
    },
    {
        "num": 33,
        "folder": "CV Projects 33 - Image Bluring using opencv",
        "title": "Image Blurring",
        "desc": "Apply various blur techniques: average, Gaussian, median, and bilateral.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Image",
        "output": "Blurred images (4 methods)",
        "steps": ["Load image", "Apply blur / GaussianBlur / medianBlur / bilateralFilter", "Compare results"],
    },
    {
        "num": 34,
        "folder": "CV Projects 34 - Live Motion Blurring",
        "title": "Live Motion Blurring",
        "desc": "Apply directional motion blur effect to live webcam feed.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Webcam / Image",
        "output": "Motion-blurred video/image",
        "steps": ["Capture frame", "Create motion kernel", "Apply filter2D", "Display result"],
    },
    {
        "num": 35,
        "folder": "CV Projects 35 - Sharpning of images using OpenCV",
        "title": "Image Sharpening",
        "desc": "Sharpen images using unsharp masking and custom convolution kernels.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Image",
        "output": "Sharpened image",
        "steps": ["Load image", "Create sharpening kernel", "Apply filter2D", "Display sharpened result"],
    },
    {
        "num": 36,
        "folder": "CV Projects 36 - Thresholding Techiques",
        "title": "Thresholding Techniques",
        "desc": "Demonstrate binary, adaptive, and Otsu thresholding methods.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Image",
        "output": "Multiple thresholded images",
        "steps": ["Load image", "Convert to grayscale", "Apply threshold / adaptiveThreshold / Otsu", "Compare results"],
    },
    {
        "num": 37,
        "folder": "CV Projects 37 - Number plate detection",
        "title": "Number Plate Detection",
        "desc": "Detect and localize vehicle number plates using YOLO26.",
        "type": "yolo",
        "task": "detect",
        "libs": ["Ultralytics (YOLO26)", "OpenCV", "imutils"],
        "input": "Image / Video",
        "output": "Detected plates with bounding boxes",
        "steps": ["Load image", "YOLO26 inference", "Localize plate region", "Crop plate", "Display results"],
    },
    {
        "num": 38,
        "folder": "CV Projects 38 - Pencil drawing effect",
        "title": "Pencil Drawing Effect",
        "desc": "Convert images to pencil sketch effect using Gaussian blur and divide blend.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Image",
        "output": "Pencil sketch image",
        "steps": ["Load image", "Convert to grayscale", "Invert", "Gaussian blur", "Divide blend", "Display sketch"],
    },
    {
        "num": 39,
        "folder": "CV Projects 39 - Pencil drawing effect",
        "title": "Noise Removing",
        "desc": "Remove noise from images using OpenCV denoising filters.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Noisy image",
        "output": "Denoised image",
        "steps": ["Load image", "Add noise (optional)", "Apply fastNlMeansDenoisingColored", "Compare before/after"],
    },
    {
        "num": 40,
        "folder": "CV Projects 40 - Non-photorealistic rendering",
        "title": "Non-Photorealistic Rendering",
        "desc": "Apply artistic rendering effects (oil painting, stylization) using OpenCV NPR functions.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Image",
        "output": "Stylized image",
        "steps": ["Load image", "Apply stylization / edgePreservingFilter / detailEnhance", "Display artistic results"],
    },
    {
        "num": 41,
        "folder": "CV Projects 41 - Image Segmentation",
        "title": "Image Segmentation",
        "desc": "Segment images using YOLO26 segmentation model for pixel-level object masks.",
        "type": "yolo",
        "task": "seg",
        "libs": ["Ultralytics (YOLO26)", "OpenCV"],
        "input": "Image / Webcam",
        "output": "Segmentation masks overlaid on image",
        "steps": ["Load image", "YOLO26 segmentation inference", "Extract masks", "Overlay colored masks", "Display result"],
    },
    {
        "num": 42,
        "folder": "CV Projects 42 - Image resizing using opencv",
        "title": "Image Resizing",
        "desc": "Resize images using different interpolation methods (NEAREST, LINEAR, CUBIC, etc.).",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Image",
        "output": "Resized images",
        "steps": ["Load image", "Apply cv2.resize with various interpolation flags", "Compare quality vs speed"],
    },
    {
        "num": 43,
        "folder": "CV Projects 43 - Funny Cartoonizing Images using openCV",
        "title": "Funny Image Cartoonizer",
        "desc": "Apply cartoon effect using bilateral filtering and adaptive thresholding.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Image",
        "output": "Cartoonized image",
        "steps": ["Load image", "Edge mask (adaptive threshold)", "Bilateral filter", "Combine edge mask + filtered", "Display cartoon"],
    },
    {
        "num": 44,
        "folder": "CV Projects 44 - Joining Multiple Images to Display",
        "title": "Joining Multiple Images",
        "desc": "Combine multiple images into a single display grid using numpy and OpenCV.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Multiple images",
        "output": "Composite grid image",
        "steps": ["Load images", "Resize to uniform size", "Stack horizontally/vertically", "Display grid"],
    },
    {
        "num": 45,
        "folder": "CV Projects 45 -  Detecting clicks on images",
        "title": "Detecting Clicks on Images",
        "desc": "Detect and handle mouse click events on images using OpenCV event callbacks.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Image + mouse events",
        "output": "Click coordinates + region info",
        "steps": ["Load image", "Register mouse callback", "Capture click events", "Display coordinates", "Handle regions"],
    },
    {
        "num": 46,
        "folder": "CV Projects 46 - Face Detection Second Approach",
        "title": "Face Detection (YOLO26 Approach)",
        "desc": "Face detection using YOLO26 model — modern replacement for Haarcascade approach.",
        "type": "yolo",
        "task": "detect",
        "libs": ["Ultralytics (YOLO26)", "OpenCV"],
        "input": "Image / Webcam",
        "output": "Detected faces with bounding boxes",
        "steps": ["Load image/webcam", "YOLO26 face detection", "Draw bounding boxes", "Display with confidence"],
    },
    {
        "num": 47,
        "folder": "CV Projects 47 -  Face Mask Detection",
        "title": "Face Mask Detection",
        "desc": "Detect whether a person is wearing a face mask using YOLO26 detection.",
        "type": "yolo",
        "task": "detect",
        "libs": ["Ultralytics (YOLO26)", "OpenCV"],
        "input": "Image / Webcam",
        "output": "Mask/No-mask labels on faces",
        "steps": ["Capture frame", "YOLO26 detection", "Classify mask status", "Draw boxes with labels", "Display result"],
    },
    {
        "num": 48,
        "folder": "CV Projects 48 - Face, Gender & Ethincity recognizer model",
        "title": "Face, Gender & Ethnicity Recognizer",
        "desc": "Multi-output CNN model for predicting age, gender, and ethnicity from facial images using YOLO26 + PyTorch.",
        "type": "yolo",
        "task": "detect",
        "libs": ["Ultralytics (YOLO26)", "PyTorch", "OpenCV"],
        "input": "Face images",
        "output": "Predicted attributes (age, gender, ethnicity)",
        "steps": ["Load face image", "Preprocess", "YOLO26 face detection", "Feature extraction", "Multi-output prediction", "Display results"],
    },
    {
        "num": 49,
        "folder": "CV Projects 49- Real Time TextDetection",
        "title": "Real-Time Text Detection",
        "desc": "Detect and recognize text in images using YOLO26 text detection and Tesseract OCR.",
        "type": "yolo",
        "task": "detect",
        "libs": ["Ultralytics (YOLO26)", "pytesseract", "OpenCV"],
        "input": "Image / Webcam",
        "output": "Detected text regions + recognized text",
        "steps": ["Capture frame", "YOLO26 text region detection", "Crop text regions", "OCR with Tesseract", "Display text overlay"],
    },
    {
        "num": 50,
        "folder": "CV Projects 50 - Reversing video using opencv",
        "title": "Reversing Video",
        "desc": "Reverse a video file frame-by-frame using OpenCV VideoCapture and VideoWriter.",
        "type": "opencv",
        "libs": ["OpenCV"],
        "input": "Video file",
        "output": "Reversed video",
        "steps": ["Load video", "Read all frames", "Reverse frame order", "Write reversed video", "Display result"],
    },
]


def _mermaid_pipeline(proj: dict) -> str:
    """Generate a Mermaid flowchart for the project pipeline."""
    steps = proj.get("steps", [])
    if not steps:
        return ""

    lines = ["```mermaid", "flowchart LR"]
    for i, step in enumerate(steps):
        node_id = chr(65 + i)  # A, B, C, ...
        if i == 0:
            lines.append(f"    {node_id}[{step}]")
        else:
            prev = chr(65 + i - 1)
            lines.append(f"    {prev} --> {node_id}[{step}]")
    lines.append("```")
    return "\n".join(lines)


def _file_listing(folder: Path) -> str:
    """Generate a file listing for the project folder."""
    if not folder.exists():
        return "_Folder not found._"

    entries = []
    for item in sorted(folder.iterdir()):
        if item.name in ("__pycache__", "project_meta.yaml", ".gitkeep"):
            continue
        if item.is_dir():
            entries.append(f"  {item.name}/")
        else:
            entries.append(f"  {item.name}")
    if not entries:
        return "_Empty folder._"

    return "```\n" + "\n".join(entries) + "\n```"


def generate_readme(proj: dict, force: bool = False) -> bool:
    """Generate README.md for one project. Returns True if written."""
    folder = ROOT / proj["folder"]
    readme = folder / "README.md"

    if readme.exists() and not force:
        return False

    num = proj["num"]
    title = proj["title"]
    desc = proj["desc"]
    ptype = proj["type"]
    libs = proj.get("libs", [])

    has_train = (folder / "train.py").exists()
    has_modern = (folder / "modern.py").exists()

    # Build README content
    lines: list[str] = []
    lines.append(f"# Project {num}: {title}\n")
    lines.append(f"{desc}\n")

    # Badges
    badges = []
    if ptype == "yolo":
        badges.append("![YOLO26](https://img.shields.io/badge/YOLO26-detect-blue)")
    badges.append("![OpenCV](https://img.shields.io/badge/OpenCV-4.12-green)")
    if has_train:
        badges.append("![Trainable](https://img.shields.io/badge/Trainable-yes-orange)")
    lines.append(" ".join(badges) + "\n")

    # Pipeline diagram
    lines.append("## Pipeline\n")
    lines.append(_mermaid_pipeline(proj) + "\n")

    # Quick Start
    lines.append("## Quick Start\n")
    lines.append("```bash")
    lines.append(f'cd "{proj["folder"]}"')
    if has_modern:
        lines.append(f"python -m core.runner {proj['folder'].split(' - ')[0].replace('CV Project ', 'p').replace('CV Projects ', 'p').replace('CV projects ', 'p').replace(' ', '')}")
    lines.append("```\n")

    if has_train:
        task = proj.get("task", "detect")
        default_model = {
            "detect": "yolo26n.pt",
            "pose": "yolo26n-pose.pt",
            "seg": "yolo26n-seg.pt",
            "cls": "yolo26n-cls.pt",
        }.get(task, "yolo26n.pt")
        lines.append("### Training\n")
        lines.append("```bash")
        lines.append(f"python train.py --model {default_model} --epochs 50 --device 0")
        lines.append("```\n")

    # Tech Stack
    lines.append("## Tech Stack\n")
    lines.append("| Component | Details |")
    lines.append("|---|---|")
    lines.append(f"| Type | {'Deep Learning (YOLO26)' if ptype == 'yolo' else 'Computer Vision (OpenCV)'} |")
    lines.append(f"| Input | {proj.get('input', 'Image / Webcam')} |")
    lines.append(f"| Output | {proj.get('output', 'Processed result')} |")
    lines.append(f"| Libraries | {', '.join(libs)} |")
    if ptype == "yolo":
        lines.append(f"| Task | {proj.get('task', 'detect')} |")
    lines.append("")

    # Files
    lines.append("## Files\n")
    lines.append(_file_listing(folder) + "\n")

    # Links
    lines.append("## See Also\n")
    lines.append("- [Root README](../../README.md)")
    lines.append("- [Model Registry](../../models/registry.py)")
    if ptype == "yolo":
        lines.append("- [YOLO Loader](../../utils/yolo.py)")
    if has_train:
        lines.append("- [Dataset Config](../../configs/datasets/)")
    lines.append("")

    # Write
    content = "\n".join(lines)
    readme.write_text(content, encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate per-project READMEs")
    parser.add_argument("--force", action="store_true", help="Overwrite existing READMEs")
    args = parser.parse_args()

    print(f"Generating README.md for {len(PROJECTS)} projects (force={args.force})")
    created = 0
    skipped = 0
    for proj in PROJECTS:
        folder = ROOT / proj["folder"]
        if not folder.exists():
            print(f"  SKIP (no folder): {proj['folder']}")
            skipped += 1
            continue
        written = generate_readme(proj, force=args.force)
        if written:
            print(f"  OK: P{proj['num']:02d} {proj['title']}")
            created += 1
        else:
            print(f"  SKIP (exists): P{proj['num']:02d}")
            skipped += 1

    print(f"\nDone: {created} created, {skipped} skipped")


if __name__ == "__main__":
    main()
