import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
from tensorflow.keras.models import model_from_json
from tqdm import tqdm
import time

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

    for img in tqdm(os.listdir(os.path.join(root_dir,'test'))):
        t1 = time.time()
        img_arr = cv2.imread(os.path.join(root_dir,'test',img))
        resized_face = cv2.resize(img_arr,(160,160))
        resized_face = resized_face.astype("float") / 255.0
        resized_face = np.expand_dims(resized_face, axis=0)
        preds = model.predict(resized_face)[0]
        if preds> 0.5:
            label = 'spoof'
            t2 = time.time()
            print( 'Time taken was {} seconds'.format( t2 - t1))
        else:
            label = 'real'
            t2 = time.time()
            print( 'Time taken was {} seconds'.format( t2 - t1))


if __name__ == "__main__":
    main()
        
