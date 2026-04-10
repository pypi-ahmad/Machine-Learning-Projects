import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
from tensorflow.keras.models import model_from_json

_src = Path(__file__).resolve().parent


def main():
    root_dir = str(_src)
    # Load Face Detection Model
    face_cascade = cv2.CascadeClassifier(str(_src / "models" / "haarcascade_frontalface_default.xml"))
    # Load Anti-Spoofing Model graph
    json_file = open(str(_src / 'antispoofing_models' / 'antispoofing_model.json'),'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load antispoofing model weights 
    model.load_weights(str(_src / 'antispoofing_models' / 'antispoofing_model.h5'))
    print("Model loaded from disk")

    video = cv2.VideoCapture(0)
    while True:
        try:
            ret, frame = video.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in faces:  
                face = frame[y-5:y+h+5,x-5:x+w+5]
                resized_face = cv2.resize(face,(160,160))
                resized_face = resized_face.astype("float") / 255.0
                resized_face = np.expand_dims(resized_face, axis=0)
                preds = model.predict(resized_face)[0]
                print(preds)
                if preds> 0.5:
                    label = 'spoof'
                    cv2.putText(frame, label, (x,y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    cv2.rectangle(frame, (x, y), (x+w,y+h),
                        (0, 0, 255), 2)
                else:
                    label = 'real'
                    cv2.putText(frame, label, (x,y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    cv2.rectangle(frame, (x, y), (x+w,y+h),
                    (0, 255, 0), 2)
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        except Exception as e:
            pass
    video.release()        
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()