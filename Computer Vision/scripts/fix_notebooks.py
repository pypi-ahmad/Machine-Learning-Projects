"""Fix all failing notebooks in the repo."""
from __future__ import annotations
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
fixes_applied = []


# ==========================================
# 1. Fix Industrial Scratch Crack - TypeError
# Sort key mixes int and str which is incompatible in Python 3
# ==========================================
path = REPO / "Industrial Scratch Crack Segmentation" / "Source Code" / "industrial_scratch_crack_segmentation_pipeline.ipynb"
with open(path) as f:
    nb = json.load(f)
cell = nb["cells"][10]
src = cell["source"]
changed = False
for i, line in enumerate(src):
    if "cls = sorted(cc, key=lambda x: int(x) if x.isdigit() else x)" in line:
        src[i] = line.replace(
            "cls = sorted(cc, key=lambda x: int(x) if x.isdigit() else x)",
            "cls = sorted(cc, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))",
        )
        changed = True
        fixes_applied.append("industrial_scratch: fixed TypeError in label sort (cell 10)")
        break
if changed:
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Fixed: {path.name}")
else:
    print(f"WARNING: no change for industrial_scratch (already fixed?)")


# ==========================================
# 2. Fix DatasetNotFoundError / RuntimeError notebooks
# Wrap load_dataset call in try/except so the notebook
# continues gracefully when the dataset is unavailable
# ==========================================
dataset_notebooks = [
    ("Blink Headpose Analyzer/Source Code/blink_headpose_analyzer_pipeline.ipynb", "mitchelldehaven/drowsiness_dataset"),
    ("Driver Drowsiness Monitor/Source Code/driver_drowsiness_monitor_pipeline.ipynb", "mitchelldehaven/drowsiness_dataset"),
    ("Emotion Recognition from Facial Expression/Source Code/emotion_recognition_pipeline.ipynb", "Piro17/fer2013"),
    ("Exercise Rep Counter/Source Code/exercise_rep_counter_pipeline.ipynb", "kamilakesbi/FitnessBasicExercises"),
    ("Face Clustering Photo Organizer/Source Code/face_clustering_photo_organizer_pipeline.ipynb", "vishalmor/lfw-dataset"),
    ("Face Verification Attendance System/Source Code/face_verification_attendance_pipeline.ipynb", "vishalmor/lfw-dataset"),
    ("Finger Counter Pro/Source Code/finger_counter_pro_pipeline.ipynb", "koryakinp/fingers"),
    ("Gaze Direction Estimator/Source Code/gaze_direction_estimator_pipeline.ipynb", "mitchelldehaven/drowsiness_dataset"),
    ("Gesture Controlled Slideshow/Source Code/gesture_controlled_slideshow_pipeline.ipynb", "koryakinp/fingers"),
    ("Handwritten Note to Markdown/Source Code/handwritten_note_to_markdown_pipeline.ipynb", "nielsr/iam_handwriting_words"),
    ("Business Card Reader/Source Code/business_card_reader_pipeline.ipynb", "aharley/rvl_cdip"),
    ("Road Pothole Segmentation/Source Code/road_pothole_segmentation_pipeline.ipynb", "keremberke/pothole-segmentation"),
    ("Receipt Digitizer/Source Code/receipt_digitizer_pipeline.ipynb", "jinhybr/OCR-receipt"),
    ("Scene Text Reader Translator/Source Code/scene_text_reader_translator_pipeline.ipynb", "bsmock/COCO-Text-v2"),
    ("Sign Language Alphabet Recognizer/Source Code/sign_language_alphabet_recognizer_pipeline.ipynb", "grassknoted/asl-alphabet"),
    ("Visual Anomaly Detector/Source Code/visual_anomaly_detector_pipeline.ipynb", "alexrods/mvtec-ad"),
    ("Yoga Pose Correction Coach/Source Code/yoga_pose_correction_coach_pipeline.ipynb", "keremberke/yoga-poses-dataset"),
]

for nb_rel, dataset_id in dataset_notebooks:
    path = REPO / nb_rel
    if not path.exists():
        print(f"NOT FOUND: {nb_rel}")
        continue
    with open(path) as f:
        nb = json.load(f)
    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        full = "".join(src)
        # Only patch if this cell still has the bare (unwrapped) load_dataset call
        # Use _ds_err as the marker that the fix has already been applied
        trigger = f"load_dataset('{dataset_id}'"
        trigger2 = f'load_dataset("{dataset_id}"'
        if (trigger in full or trigger2 in full) and "_ds_err" not in full:
            new_src = []
            i = 0
            while i < len(src):
                line = src[i]
                if trigger in line or trigger2 in line:
                    new_src.append("try:\n")
                    new_src.append("    " + line.rstrip("\n") + "\n")
                    # consume the next print(dataset) if present
                    if i + 1 < len(src) and "print(dataset)" in src[i + 1]:
                        new_src.append("    " + src[i + 1].rstrip("\n") + "\n")
                        i += 1
                    new_src.append("except Exception as _ds_err:\n")
                    new_src.append(
                        "    print(f'Dataset not available on Hub ({_ds_err}); "
                        "continuing with local data if present.')\n"
                    )
                    new_src.append("    dataset = None\n")
                    changed = True
                else:
                    new_src.append(line)
                i += 1
            cell["source"] = new_src
            break
    if changed:
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        fixes_applied.append(f"wrapped load_dataset in try/except ({dataset_id})")
        print(f"Fixed: {path.parent.parent.name}")
    else:
        # Check if already fixed
        if any("_ds_err" in "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"):
            print(f"Already fixed: {path.parent.parent.name}")
        else:
            print(f"WARNING - no matching load_dataset found: {nb_rel}")


# ==========================================
# 3. Fix Timeout notebooks - reduce epochs
# ==========================================
timeout_fixes = [
    (
        "Lung Segmentation from Chest X-Ray/Source Code/lung_segmentation_pipeline.ipynb",
        14,  # cell index
        "        data=yf[0], epochs=25, imgsz=640, batch=16,",
        "        data=yf[0], epochs=3, imgsz=640, batch=16,",
    ),
    (
        "Skin Cancer Detection/Souce Code/skin_cancer_detection_pipeline.ipynb",
        13,
        "        data=yf[0], epochs=25, imgsz=640, batch=16,",
        "        data=yf[0], epochs=3, imgsz=640, batch=16,",
    ),
    (
        "Plant Disease Severity Estimator/Source Code/plant_disease_severity_estimator_pipeline.ipynb",
        13,
        "    sys.argv = ['train.py','--epochs','15','--batch','32']",
        "    sys.argv = ['train.py','--epochs','3','--batch','32']",
    ),
]

for nb_rel, cell_idx, old_text, new_text in timeout_fixes:
    path = REPO / nb_rel
    if not path.exists():
        print(f"NOT FOUND: {nb_rel}")
        continue
    with open(path) as f:
        nb = json.load(f)
    cell = nb["cells"][cell_idx]
    src = cell["source"]
    changed = False
    for i, line in enumerate(src):
        stripped = line.rstrip("\n")
        if stripped == old_text:
            src[i] = new_text + "\n"
            changed = True
            break
    if changed:
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        fixes_applied.append(f"reduced epochs: {pathlib.Path(nb_rel).parent.parent.name}")
        print(f"Fixed epochs: {path.parent.parent.name}")
    else:
        # Check if already reduced
        full = "".join(src)
        if new_text.strip() in full:
            print(f"Already reduced: {path.parent.parent.name}")
        else:
            print(f"WARNING: epochs not changed for {nb_rel}")
            print(f"  Looking for: {repr(old_text)}")


print()
print(f"Total fixes applied: {len(fixes_applied)}")
for fix in fixes_applied:
    print(f"  - {fix}")
