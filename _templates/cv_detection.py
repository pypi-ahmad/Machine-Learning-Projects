"""CV Object Detection template: YOLOv8/YOLO11 — April 2026"""
import textwrap


def generate(project_path, config):
    task = config.get("task", "detect")  # detect, track, segment

    return textwrap.dedent(f'''\
        """
        Modern CV Object Detection Pipeline (April 2026)
        Model: YOLO (Ultralytics) — detection, tracking, segmentation
        """
        import os, warnings
        from pathlib import Path

        warnings.filterwarnings("ignore")

        TASK = "{task}"


        def find_images():
            data_dir = Path(os.path.dirname(__file__))
            exts = (".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi")
            files = []
            for ext in exts:
                files.extend(data_dir.rglob(f"*{{ext}}"))
            return files


        def run_detection(files):
            from ultralytics import YOLO

            model = YOLO("yolo11n.pt")  # nano model, fast
            save_dir = os.path.join(os.path.dirname(__file__), "detections")
            os.makedirs(save_dir, exist_ok=True)

            for f in files[:20]:
                results = model(str(f))
                for r in results:
                    r.save(filename=os.path.join(save_dir, f.name))
                    boxes = r.boxes
                    if boxes is not None:
                        print(f"  ✓ {{f.name}}: {{len(boxes)}} objects detected")

            print(f"\\nResults saved to {{save_dir}}")
            return model


        def run_tracking(files):
            from ultralytics import YOLO

            model = YOLO("yolo11n.pt")
            video_files = [f for f in files if f.suffix in (".mp4", ".avi")]
            if not video_files:
                print("No video files found for tracking.")
                return

            for v in video_files[:3]:
                results = model.track(str(v), persist=True, save=True,
                                       project=os.path.dirname(__file__), name="tracking")
                print(f"  ✓ Tracked: {{v.name}}")


        def train_custom(data_yaml=None):
            """Fine-tune YOLO on custom dataset."""
            from ultralytics import YOLO

            data_dir = Path(os.path.dirname(__file__))
            if data_yaml is None:
                yaml_files = list(data_dir.glob("*.yaml")) + list(data_dir.glob("*.yml"))
                if yaml_files:
                    data_yaml = str(yaml_files[0])
                else:
                    print("No data.yaml found. Create one for custom training.")
                    return

            model = YOLO("yolo11n.pt")
            results = model.train(
                data=data_yaml,
                epochs=50,
                imgsz=640,
                batch=16,
                device=0,
                project=str(data_dir),
                name="yolo_train",
            )
            print(f"✓ Training complete. Best mAP: {{results.results_dict.get('metrics/mAP50(B)', 'N/A')}}")
            return model


        def main():
            print("=" * 60)
            print("MODERN CV DETECTION PIPELINE")
            print(f"Model: YOLO (Ultralytics) | Task: {{TASK}}")
            print("=" * 60)

            files = find_images()
            print(f"Found {{len(files)}} media files")

            if TASK == "detect":
                run_detection(files)
            elif TASK == "track":
                run_tracking(files)
            elif TASK == "train":
                train_custom()
            else:
                run_detection(files)


        if __name__ == "__main__":
            main()
    ''')
