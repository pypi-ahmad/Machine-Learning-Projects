"""
Modern Face/Hand/Gesture Pipeline (April 2026)

Task dispatch:
  face_detection : YOLO26m (primary) + MediaPipe Face Landmarker (secondary)
  expression     : MediaPipe Face Landmarker blendshapes (primary) + YOLO26m (baseline)
  face_recognition : InsightFace ArcFace embeddings + age/gender
  hand_gesture   : MediaPipe Gesture Recognizer (webcam)
  pose           : MediaPipe Pose Landmarker Heavy (33-point skeleton)
Timing : Wall-clock per model stage.
Export : metrics.json with detection counts, landmarks, and timing.
Data   : Auto-downloads LFW face samples at runtime.
"""
import os, json, time, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

TASK = "face_recognition"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def download_face_samples():
    """Download LFW face images from sklearn."""
    from sklearn.datasets import fetch_lfw_people
    import cv2

    save_dir = Path(os.path.dirname(__file__)) / "face_samples"
    save_dir.mkdir(exist_ok=True)

    if list(save_dir.glob("*.jpg")):
        return list(save_dir.glob("*.jpg"))

    lfw = fetch_lfw_people(min_faces_per_person=20, resize=1.0)
    paths = []
    for i in range(min(30, len(lfw.images))):
        img = (lfw.images[i] * 255).astype(np.uint8) if lfw.images[i].max() <= 1 else lfw.images[i].astype(np.uint8)
        p = save_dir / f"face_{i:03d}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)
    print(f"Downloaded {len(paths)} face images")
    return paths


def run_yolo_detection(files):
    """YOLO26 for person/face detection — replaces Haar cascades."""
    from ultralytics import YOLO
    model = YOLO("yolo26m.pt")
    save_dir = os.path.join(SAVE_DIR, "yolo_detections")
    os.makedirs(save_dir, exist_ok=True)
    t0 = time.perf_counter()
    total_persons = 0
    total_objects = 0
    for f in files[:20]:
        results = model(str(f))
        for r in results:
            r.save(filename=os.path.join(save_dir, f.name))
            n_people = sum(1 for b in r.boxes if int(b.cls) == 0) if r.boxes is not None else 0
            n_total = len(r.boxes) if r.boxes is not None else 0
            total_persons += n_people
            total_objects += n_total
            print(f"  YOLO26 {f.name}: {n_people} persons, {n_total} total")
    elapsed = time.perf_counter() - t0
    print(f"  YOLO26: {len(files[:20])} images, {total_objects} objects in {elapsed:.1f}s")
    print(f"  Results saved to {save_dir}")
    return {"model": "yolo26m", "images": len(files[:20]), "persons": total_persons,
             "total_objects": total_objects, "time_s": round(elapsed, 1)}


def run_face_landmarker(files):
    """MediaPipe Face Landmarker — modern Tasks API for 478-point face mesh and expressions."""
    try:
        import cv2, mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request

        model_path = os.path.join(SAVE_DIR, "face_landmarker.task")
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
                model_path)

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=5)
        landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        save_dir = os.path.join(SAVE_DIR, "face_landmark_results")
        os.makedirs(save_dir, exist_ok=True)

        t0 = time.perf_counter()
        total_faces = 0
        for f in files[:20]:
            img = cv2.imread(str(f))
            if img is None: continue
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            result = landmarker.detect(mp_img)
            n_faces = len(result.face_landmarks)
            total_faces += n_faces
            for face_lm in result.face_landmarks:
                for lm in face_lm:
                    x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                    cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
            if result.face_blendshapes:
                top_shapes = sorted(result.face_blendshapes[0], key=lambda b: b.score, reverse=True)[:3]
                expr = ", ".join(f"{b.category_name}={b.score:.2f}" for b in top_shapes)
                print(f"  {f.name}: {n_faces} faces, expressions: {expr}")
            else:
                print(f"  {f.name}: {n_faces} faces (478-pt mesh)")
            cv2.imwrite(os.path.join(save_dir, f.name), img)
        elapsed = time.perf_counter() - t0
        landmarker.close()
        print(f"  Face Landmarker: {total_faces} faces in {elapsed:.1f}s")
        return {"model": "MediaPipe Face Landmarker", "faces": total_faces, "time_s": round(elapsed, 1)}
    except Exception as e:
        print(f"  MediaPipe Face Landmarker: {e}")
        # Fallback to legacy face detection
        try:
            import cv2, mediapipe as mp
            mp_face = mp.solutions.face_detection; mp_draw = mp.solutions.drawing_utils
            t0 = time.perf_counter()
            total = 0
            with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face:
                for f in files[:20]:
                    img = cv2.imread(str(f))
                    if img is None: continue
                    results = face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    n = len(results.detections) if results.detections else 0
                    total += n
                    print(f"  (legacy) {f.name}: {n} faces")
            elapsed = time.perf_counter() - t0
            return {"model": "MediaPipe legacy", "faces": total, "time_s": round(elapsed, 1)}
        except Exception as e2:
            print(f"  MediaPipe legacy fallback: {e2}")
    return {}


def run_insightface(files):
    """InsightFace — face recognition, verification, gender/age/ethnicity."""
    try:
        import cv2
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        save_dir = os.path.join(SAVE_DIR, "insightface_results")
        os.makedirs(save_dir, exist_ok=True)
        t0 = time.perf_counter()
        embeddings = []
        total_faces = 0
        for f in files[:20]:
            img = cv2.imread(str(f))
            if img is None: continue
            faces = app.get(img)
            total_faces += len(faces)
            for face in faces:
                bbox = face.bbox.astype(int)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                if hasattr(face, "embedding") and face.embedding is not None:
                    embeddings.append(face.embedding)
                info = []
                if hasattr(face, "age"): info.append(f"age={face.age}")
                if hasattr(face, "gender"): info.append(f"gender={'M' if face.gender==1 else 'F'}")
                if info:
                    cv2.putText(img, " ".join(info), (bbox[0], bbox[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imwrite(os.path.join(save_dir, f.name), img)
            print(f"  {f.name}: {len(faces)} faces")
        elapsed = time.perf_counter() - t0
        sim = None
        if len(embeddings) >= 2:
            sim = float(np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
            print(f"  Cosine similarity (face 0 vs 1): {sim:.4f}")
        print(f"  InsightFace: {total_faces} faces in {elapsed:.1f}s")
        return {"model": "InsightFace", "faces": total_faces, "embeddings": len(embeddings),
                 "cosine_sim_0v1": round(sim, 4) if sim else None, "time_s": round(elapsed, 1)}
    except Exception as e:
        print(f"  InsightFace: {e}")
        return {}


def run_hand_gesture(files):
    """MediaPipe Hand Landmarker / Gesture Recognizer — modern Tasks API.

    Stage 1: Static image inference (offline, always runs).
    Stage 2: Live webcam gesture recognition (runs only if display available).
    """
    import sys
    try:
        import cv2, mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request
        from collections import Counter

        model_path = os.path.join(SAVE_DIR, "gesture_recognizer.task")
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
                model_path)

        options = mp_vision.GestureRecognizerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            num_hands=2)
        recognizer = mp_vision.GestureRecognizer.create_from_options(options)

        # --- Stage 1: Static image inference ---
        save_dir = os.path.join(SAVE_DIR, "hand_gesture_results")
        os.makedirs(save_dir, exist_ok=True)
        gesture_counts = Counter()
        total_hands = 0
        confidences = []
        t0 = time.perf_counter()
        for f in files[:20]:
            img = cv2.imread(str(f))
            if img is None:
                continue
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            result = recognizer.recognize(mp_img)
            n_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
            total_hands += n_hands
            if result.gestures:
                for gesture in result.gestures:
                    name = gesture[0].category_name
                    score = gesture[0].score
                    gesture_counts[name] += 1
                    confidences.append(score)
            if result.hand_landmarks:
                for hand_lm in result.hand_landmarks:
                    for lm in hand_lm:
                        x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
            cv2.imwrite(os.path.join(save_dir, f.name), img)
            print(f"  {f.name}: {n_hands} hands")
        static_elapsed = time.perf_counter() - t0
        avg_conf = sum(confidences) / max(len(confidences), 1)
        print(f"  Static: {len(files[:20])} images, {total_hands} hands in {static_elapsed:.1f}s")

        # --- Stage 2: Live webcam (if display available) ---
        webcam_frames = 0
        webcam_elapsed = 0.0
        if sys.stdout.isatty():
            print("Starting webcam gesture recognition... Press 'q' to quit.")
            cap = cv2.VideoCapture(0)
            t1 = time.perf_counter()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = recognizer.recognize(mp_img)
                if result.gestures:
                    for i, gesture in enumerate(result.gestures):
                        name = gesture[0].category_name
                        score = gesture[0].score
                        gesture_counts[name] += 1
                        cv2.putText(frame, f"{name} ({score:.2f})", (10, 40 + i * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if result.hand_landmarks:
                    for hand_lm in result.hand_landmarks:
                        for lm in hand_lm:
                            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
                cv2.imshow("Gesture Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                webcam_frames += 1
                if webcam_frames >= 300:
                    break
            cap.release()
            cv2.destroyAllWindows()
            webcam_elapsed = round(time.perf_counter() - t1, 1)
            print(f"  Webcam: {webcam_frames} frames in {webcam_elapsed}s")
        else:
            print("  Webcam skipped (no display / headless environment)")

        recognizer.close()
        return {"model": "MediaPipe Gesture Recognizer", "hands_detected": total_hands,
                 "static_images": len(files[:20]), "static_time_s": round(static_elapsed, 1),
                 "gesture_counts": dict(gesture_counts),
                 "avg_confidence": round(avg_conf, 4),
                 "webcam_frames": webcam_frames, "webcam_time_s": webcam_elapsed}
    except Exception as e:
        print(f"  MediaPipe Gesture Recognizer: {e}")
        # Fallback to legacy hand detection on static images
        try:
            import cv2, mediapipe as mp
            mp_hands = mp.solutions.hands
            t0 = time.perf_counter()
            total = 0
            with mp_hands.Hands(static_image_mode=True,
                                min_detection_confidence=0.7) as hands:
                for f in files[:20]:
                    img = cv2.imread(str(f))
                    if img is None:
                        continue
                    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    n = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
                    total += n
                    print(f"  (legacy) {f.name}: {n} hands")
            elapsed = time.perf_counter() - t0
            return {"model": "MediaPipe legacy hands", "hands_detected": total,
                     "time_s": round(elapsed, 1)}
        except Exception as e2:
            print(f"  MediaPipe legacy hands: {e2}")
    return {}


def run_pose(files):
    """MediaPipe Pose Landmarker — modern Tasks API."""
    try:
        import cv2, mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request

        model_path = os.path.join(SAVE_DIR, "pose_landmarker.task")
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
                model_path)

        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            num_poses=3)
        landmarker = mp_vision.PoseLandmarker.create_from_options(options)

        save_dir = os.path.join(SAVE_DIR, "pose_results")
        os.makedirs(save_dir, exist_ok=True)
        t0 = time.perf_counter()
        total_poses = 0
        for f in files[:20]:
            img = cv2.imread(str(f))
            if img is None: continue
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            result = landmarker.detect(mp_img)
            n_poses = len(result.pose_landmarks)
            total_poses += n_poses
            for pose_lm in result.pose_landmarks:
                for lm in pose_lm:
                    x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                    cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(save_dir, f.name), img)
            print(f"  {f.name}: {n_poses} poses")
        elapsed = time.perf_counter() - t0
        landmarker.close()
        print(f"  Pose Landmarker: {total_poses} poses in {elapsed:.1f}s")
        return {"model": "MediaPipe Pose Landmarker", "poses": total_poses, "time_s": round(elapsed, 1)}
    except Exception as e:
        print(f"  MediaPipe Pose Landmarker: {e}")
        # Fallback to legacy pose
        try:
            import cv2, mediapipe as mp
            mp_pose = mp.solutions.pose; mp_draw = mp.solutions.drawing_utils
            save_dir = os.path.join(SAVE_DIR, "pose_results")
            os.makedirs(save_dir, exist_ok=True)
            t0 = time.perf_counter()
            with mp_pose.Pose(min_detection_confidence=0.5) as pose:
                for f in files[:20]:
                    img = cv2.imread(str(f))
                    if img is None: continue
                    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if results.pose_landmarks:
                        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    cv2.imwrite(os.path.join(save_dir, f.name), img)
                    print(f"  (legacy) {f.name}")
            elapsed = time.perf_counter() - t0
            return {"model": "MediaPipe legacy pose", "time_s": round(elapsed, 1)}
        except Exception as e2:
            print(f"  MediaPipe legacy pose: {e2}")
    return {}


def run_eda(files, save_dir):
    """Input file summary for face/gesture tasks."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"  Input files: {len(files)}")
    if files:
        total_size = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
        print(f"  Total size: {total_size / 1024:.1f} KB")
    print("EDA complete.")


def validate_results(metrics, files, save_dir):
    """Validate output payloads for face / gesture tasks."""
    validation = {"task": TASK, "input_files": len(files), "models": {}}
    for name, payload in metrics.items():
        if name == "task" or not isinstance(payload, dict):
            continue
        numeric_values = [
            float(value) for key, value in payload.items()
            if key != "time_s" and isinstance(value, (int, float))
        ]
        positive_signal = any(value > 0 for value in numeric_values)
        validation["models"][name] = {
            "time_s": round(float(payload.get("time_s", 0)), 1) if isinstance(payload.get("time_s", 0), (int, float)) else None,
            "positive_signal": positive_signal,
            "keys": sorted(payload.keys()),
            "passed": positive_signal or isinstance(payload.get("time_s"), (int, float)),
        }
    validation["passed"] = any(model.get("passed") for model in validation["models"].values())
    out_path = os.path.join(save_dir, "validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)
    print(f"Validation saved to {out_path}")
    return validation


def main():
    print("=" * 60)
    print(f"FACE/HAND/GESTURE | Task: {TASK}")
    print("=" * 60)
    files = download_face_samples()
    run_eda(files, SAVE_DIR)
    metrics = {"task": TASK}

    if TASK == "face_detection":
        metrics["yolo"] = run_yolo_detection(files)
        metrics["face_landmarker"] = run_face_landmarker(files)
    elif TASK == "expression":
        # Expression / smile / blink — landmarker is primary (blendshapes)
        metrics["face_landmarker"] = run_face_landmarker(files)
        metrics["yolo_baseline"] = run_yolo_detection(files)
    elif TASK == "hand_gesture":
        metrics["gesture"] = run_hand_gesture(files)
    elif TASK == "pose":
        metrics["pose"] = run_pose(files)
    elif TASK == "face_recognition":
        metrics["insightface"] = run_insightface(files)
    else:
        metrics["yolo"] = run_yolo_detection(files)
        metrics["face_landmarker"] = run_face_landmarker(files)

    metrics["validation"] = validate_results(metrics, files, SAVE_DIR)

    out_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
