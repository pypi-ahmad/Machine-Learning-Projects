"""Face/Hand/Gesture template: MediaPipe + InsightFace — April 2026"""
import textwrap


def generate(project_path, config):
    task = config.get("task", "face_detection")

    return textwrap.dedent(f'''\
        """
        Modern Face/Hand/Gesture Pipeline (April 2026)
        Models: MediaPipe (face/hand/pose), InsightFace (recognition/analysis)
        """
        import os, warnings
        import numpy as np
        from pathlib import Path

        warnings.filterwarnings("ignore")

        TASK = "{task}"


        def find_images():
            data_dir = Path(os.path.dirname(__file__))
            exts = (".jpg", ".jpeg", ".png", ".bmp")
            files = []
            for ext in exts:
                files.extend(data_dir.rglob(f"*{{ext}}"))
            return files


        def run_mediapipe_face(files):
            """Face detection and mesh using MediaPipe."""
            import cv2
            import mediapipe as mp

            mp_face = mp.solutions.face_detection
            mp_draw = mp.solutions.drawing_utils
            save_dir = os.path.join(os.path.dirname(__file__), "face_results")
            os.makedirs(save_dir, exist_ok=True)

            with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face:
                for f in files[:20]:
                    img = cv2.imread(str(f))
                    if img is None:
                        continue
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = face.process(rgb)

                    n_faces = 0
                    if results.detections:
                        n_faces = len(results.detections)
                        for det in results.detections:
                            mp_draw.draw_detection(img, det)

                    out_path = os.path.join(save_dir, f.name)
                    cv2.imwrite(out_path, img)
                    print(f"  ✓ {{f.name}}: {{n_faces}} faces")

            print(f"Results saved to {{save_dir}}")


        def run_insightface(files):
            """Face recognition/analysis using InsightFace."""
            try:
                import cv2
                from insightface.app import FaceAnalysis

                app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                app.prepare(ctx_id=0, det_size=(640, 640))

                save_dir = os.path.join(os.path.dirname(__file__), "insightface_results")
                os.makedirs(save_dir, exist_ok=True)

                for f in files[:20]:
                    img = cv2.imread(str(f))
                    if img is None:
                        continue
                    faces = app.get(img)

                    for face in faces:
                        bbox = face.bbox.astype(int)
                        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        if hasattr(face, "age"):
                            cv2.putText(img, f"Age:{{face.age}} G:{{face.gender}}",
                                        (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                    out_path = os.path.join(save_dir, f.name)
                    cv2.imwrite(out_path, img)
                    print(f"  ✓ {{f.name}}: {{len(faces)}} faces")
            except Exception as e:
                print(f"✗ InsightFace: {{e}}")


        def run_hand_gesture():
            """Hand gesture detection using MediaPipe (webcam)."""
            import cv2
            import mediapipe as mp

            mp_hands = mp.solutions.hands
            mp_draw = mp.solutions.drawing_utils

            print("Starting webcam hand detection... Press 'q' to quit.")
            cap = cv2.VideoCapture(0)

            with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    if results.multi_hand_landmarks:
                        for hand_lm in results.multi_hand_landmarks:
                            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                    cv2.imshow("Hand Gesture Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            cap.release()
            cv2.destroyAllWindows()


        def run_pose_estimation(files):
            """Pose estimation using MediaPipe."""
            import cv2
            import mediapipe as mp

            mp_pose = mp.solutions.pose
            mp_draw = mp.solutions.drawing_utils

            save_dir = os.path.join(os.path.dirname(__file__), "pose_results")
            os.makedirs(save_dir, exist_ok=True)

            with mp_pose.Pose(min_detection_confidence=0.5) as pose:
                for f in files[:20]:
                    img = cv2.imread(str(f))
                    if img is None:
                        continue
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb)

                    if results.pose_landmarks:
                        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    out_path = os.path.join(save_dir, f.name)
                    cv2.imwrite(out_path, img)
                    print(f"  ✓ {{f.name}}: pose detected")


        def main():
            print("=" * 60)
            print("MODERN FACE/HAND/GESTURE PIPELINE")
            print("MediaPipe + InsightFace")
            print("=" * 60)

            files = find_images()
            print(f"Found {{len(files)}} image files")

            if TASK == "face_detection":
                run_mediapipe_face(files)
                run_insightface(files)
            elif TASK == "hand_gesture":
                run_hand_gesture()
            elif TASK == "pose":
                run_pose_estimation(files)
            elif TASK == "face_recognition":
                run_insightface(files)
            else:
                run_mediapipe_face(files)


        if __name__ == "__main__":
            main()
    ''')
