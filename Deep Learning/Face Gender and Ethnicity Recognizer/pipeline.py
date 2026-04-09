"""
Modern Face/Hand/Gesture Pipeline (April 2026)
Models: MediaPipe (face/hand/pose), InsightFace (recognition)
Data: Auto-downloads LFW face samples at runtime
"""
import os, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

TASK = "face_recognition"


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


def run_mediapipe_face(files):
    import cv2, mediapipe as mp
    mp_face = mp.solutions.face_detection
    mp_draw = mp.solutions.drawing_utils
    save_dir = os.path.join(os.path.dirname(__file__), "face_results")
    os.makedirs(save_dir, exist_ok=True)

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face:
        for f in files[:20]:
            img = cv2.imread(str(f))
            if img is None: continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face.process(rgb)
            n = 0
            if results.detections:
                n = len(results.detections)
                for det in results.detections: mp_draw.draw_detection(img, det)
            out_path = os.path.join(save_dir, f.name)
            cv2.imwrite(out_path, img)
            print(f"  ✓ {f.name}: {n} faces")
    print(f"Results saved to {save_dir}")


def run_insightface(files):
    try:
        import cv2
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        save_dir = os.path.join(os.path.dirname(__file__), "insightface_results")
        os.makedirs(save_dir, exist_ok=True)
        for f in files[:20]:
            img = cv2.imread(str(f))
            if img is None: continue
            faces = app.get(img)
            for face in faces:
                bbox = face.bbox.astype(int)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(save_dir, f.name), img)
            print(f"  ✓ {f.name}: {len(faces)} faces")
    except Exception as e:
        print(f"✗ InsightFace: {e}")


def run_hand_gesture():
    import cv2, mediapipe as mp
    mp_hands = mp.solutions.hands; mp_draw = mp.solutions.drawing_utils
    print("Starting webcam hand detection... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks: mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            cv2.imshow("Hand Gesture", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release(); cv2.destroyAllWindows()


def run_pose(files):
    import cv2, mediapipe as mp
    mp_pose = mp.solutions.pose; mp_draw = mp.solutions.drawing_utils
    save_dir = os.path.join(os.path.dirname(__file__), "pose_results"); os.makedirs(save_dir, exist_ok=True)
    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
        for f in files[:20]:
            img = cv2.imread(str(f))
            if img is None: continue
            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks: mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imwrite(os.path.join(save_dir, f.name), img)
            print(f"  ✓ {f.name}")


def main():
    print("=" * 60)
    print(f"FACE/HAND/GESTURE — {TASK}")
    print("=" * 60)
    files = download_face_samples()
    if TASK == "face_detection": run_mediapipe_face(files); run_insightface(files)
    elif TASK == "hand_gesture": run_hand_gesture()
    elif TASK == "pose": run_pose(files)
    elif TASK == "face_recognition": run_insightface(files)
    else: run_mediapipe_face(files)


if __name__ == "__main__":
    main()
