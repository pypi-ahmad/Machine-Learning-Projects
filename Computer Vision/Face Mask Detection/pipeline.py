"""
Modern Face/Hand/Gesture Pipeline (April 2026)
Models: YOLO26 (face/person detection) + MediaPipe Face Landmarker (expressions/landmarks)
        + MediaPipe Hand Landmarker / Gesture Recognizer + InsightFace (recognition/verification)
Data: Auto-downloads LFW face samples at runtime
"""
import os, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

TASK = "face_detection"


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
    save_dir = os.path.join(os.path.dirname(__file__), "yolo_detections")
    os.makedirs(save_dir, exist_ok=True)
    for f in files[:20]:
        results = model(str(f))
        for r in results:
            r.save(filename=os.path.join(save_dir, f.name))
            n_people = sum(1 for b in r.boxes if int(b.cls) == 0) if r.boxes is not None else 0
            print(f"  ✓ YOLO26 {f.name}: {n_people} persons, {len(r.boxes) if r.boxes is not None else 0} total")
    print(f"YOLO26 results saved to {save_dir}")


def run_face_landmarker(files):
    """MediaPipe Face Landmarker — modern Tasks API for 478-point face mesh and expressions."""
    try:
        import cv2, mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request

        # Download face landmarker model
        model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
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

        save_dir = os.path.join(os.path.dirname(__file__), "face_landmark_results")
        os.makedirs(save_dir, exist_ok=True)

        for f in files[:20]:
            img = cv2.imread(str(f))
            if img is None: continue
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            result = landmarker.detect(mp_img)
            n_faces = len(result.face_landmarks)
            # Draw landmarks
            for face_lm in result.face_landmarks:
                for lm in face_lm:
                    x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                    cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
            # Report blendshapes (expressions)
            if result.face_blendshapes:
                top_shapes = sorted(result.face_blendshapes[0], key=lambda b: b.score, reverse=True)[:3]
                expr = ", ".join(f"{b.category_name}={b.score:.2f}" for b in top_shapes)
                print(f"  ✓ {f.name}: {n_faces} faces, expressions: {expr}")
            else:
                print(f"  ✓ {f.name}: {n_faces} faces (478-pt mesh)")
            cv2.imwrite(os.path.join(save_dir, f.name), img)
        landmarker.close()
        print(f"Face Landmarker results saved to {save_dir}")
    except Exception as e:
        print(f"✗ MediaPipe Face Landmarker: {e}")
        # Fallback to legacy face detection
        try:
            import cv2, mediapipe as mp
            mp_face = mp.solutions.face_detection; mp_draw = mp.solutions.drawing_utils
            with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face:
                for f in files[:20]:
                    img = cv2.imread(str(f))
                    if img is None: continue
                    results = face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    n = len(results.detections) if results.detections else 0
                    print(f"  ✓ (legacy) {f.name}: {n} faces")
        except Exception as e2:
            print(f"✗ MediaPipe legacy fallback: {e2}")


def run_insightface(files):
    """InsightFace — face recognition, verification, gender/age/ethnicity."""
    try:
        import cv2
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        save_dir = os.path.join(os.path.dirname(__file__), "insightface_results")
        os.makedirs(save_dir, exist_ok=True)
        embeddings = []
        for f in files[:20]:
            img = cv2.imread(str(f))
            if img is None: continue
            faces = app.get(img)
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
            print(f"  ✓ {f.name}: {len(faces)} faces")
        if len(embeddings) >= 2:
            sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
            print(f"  Cosine similarity (face 0 vs 1): {sim:.4f}")
        print(f"InsightFace results saved to {save_dir}")
    except Exception as e:
        print(f"✗ InsightFace: {e}")


def run_hand_gesture():
    """MediaPipe Hand Landmarker / Gesture Recognizer — modern Tasks API."""
    try:
        import cv2, mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request

        # Download gesture recognizer model
        model_path = os.path.join(os.path.dirname(__file__), "gesture_recognizer.task")
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
                model_path)

        options = mp_vision.GestureRecognizerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            num_hands=2)
        recognizer = mp_vision.GestureRecognizer.create_from_options(options)

        print("Starting webcam gesture recognition... Press 'q' to quit.")
        cap = cv2.VideoCapture(0)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = recognizer.recognize(mp_img)
            if result.gestures:
                for i, gesture in enumerate(result.gestures):
                    name = gesture[0].category_name
                    score = gesture[0].score
                    cv2.putText(frame, f"{name} ({score:.2f})", (10, 40 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if result.hand_landmarks:
                for hand_lm in result.hand_landmarks:
                    for lm in hand_lm:
                        x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
            cv2.imshow("Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
            frame_count += 1
            if frame_count >= 300: break  # auto-stop after ~10 sec
        cap.release(); cv2.destroyAllWindows()
        recognizer.close()
        print(f"✓ Gesture Recognizer processed {frame_count} frames")
    except Exception as e:
        print(f"✗ MediaPipe Gesture Recognizer: {e}")
        # Fallback to legacy hand detection
        try:
            import cv2, mediapipe as mp
            mp_hands = mp.solutions.hands; mp_draw = mp.solutions.drawing_utils
            print("(fallback) Starting webcam hand detection... Press 'q' to quit.")
            cap = cv2.VideoCapture(0)
            with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.multi_hand_landmarks:
                        for lm in results.multi_hand_landmarks:
                            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                    cv2.imshow("Hand Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"): break
            cap.release(); cv2.destroyAllWindows()
        except Exception as e2:
            print(f"✗ MediaPipe legacy hands: {e2}")


def run_pose(files):
    """MediaPipe Pose Landmarker — modern Tasks API."""
    try:
        import cv2, mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request

        model_path = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
                model_path)

        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            num_poses=3)
        landmarker = mp_vision.PoseLandmarker.create_from_options(options)

        save_dir = os.path.join(os.path.dirname(__file__), "pose_results")
        os.makedirs(save_dir, exist_ok=True)
        for f in files[:20]:
            img = cv2.imread(str(f))
            if img is None: continue
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            result = landmarker.detect(mp_img)
            n_poses = len(result.pose_landmarks)
            for pose_lm in result.pose_landmarks:
                for lm in pose_lm:
                    x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                    cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(save_dir, f.name), img)
            print(f"  ✓ {f.name}: {n_poses} poses")
        landmarker.close()
        print(f"Pose Landmarker results saved to {save_dir}")
    except Exception as e:
        print(f"✗ MediaPipe Pose Landmarker: {e}")
        # Fallback to legacy pose
        try:
            import cv2, mediapipe as mp
            mp_pose = mp.solutions.pose; mp_draw = mp.solutions.drawing_utils
            save_dir = os.path.join(os.path.dirname(__file__), "pose_results")
            os.makedirs(save_dir, exist_ok=True)
            with mp_pose.Pose(min_detection_confidence=0.5) as pose:
                for f in files[:20]:
                    img = cv2.imread(str(f))
                    if img is None: continue
                    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if results.pose_landmarks:
                        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    cv2.imwrite(os.path.join(save_dir, f.name), img)
                    print(f"  ✓ (legacy) {f.name}")
        except Exception as e2:
            print(f"✗ MediaPipe legacy pose: {e2}")


def main():
    print("=" * 60)
    print(f"FACE/HAND/GESTURE — {TASK}")
    print("=" * 60)
    files = download_face_samples()
    if TASK == "face_detection":
        run_yolo_detection(files)
        run_face_landmarker(files)
    elif TASK == "hand_gesture":
        run_hand_gesture()
    elif TASK == "pose":
        run_pose(files)
    elif TASK == "face_recognition":
        run_insightface(files)
    else:
        run_yolo_detection(files)
        run_face_landmarker(files)


if __name__ == "__main__":
    main()
