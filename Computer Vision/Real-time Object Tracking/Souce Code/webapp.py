"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
Migrated from YOLOv5 subprocess calls to ultralytics YOLO direct inference.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from PIL import Image
import torch
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os
import time
import warnings
from ultralytics import YOLO
from utils.paths import PathResolver
from utils.device import get_device

warnings.filterwarnings("ignore")

paths = PathResolver()
_src = Path(__file__).resolve().parent
_model_path = paths.models("realtime_object_tracking") / "best.pt"
_device = get_device()

app = Flask(__name__)
yolo_model = None  # loaded in __main__


def _ensure_model():
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO(str(_model_path))
    return yolo_model


def _run_inference(source_path: str) -> str:
    """Run YOLO inference and save results. Returns path to output directory."""
    model = _ensure_model()
    results = model(source_path, device=_device, save=True, project=str(_src / "runs" / "detect"))
    # Find latest output subfolder
    folder_path = str(_src / "runs" / "detect")
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    return os.path.join(folder_path, latest)



@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            if not f:
                return "Please upload a file."
            basepath = os.path.dirname(__file__)
            uploads_dir = os.path.join(basepath, 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            filepath = os.path.join(uploads_dir, secure_filename(f.filename))
            print("upload folder is ", filepath)
            f.save(filepath)

            predict_img.imgpath = secure_filename(f.filename)
            print("printing predict_img :::::: ", predict_img)

            file_extension = predict_img.imgpath.rsplit('.', 1)[1].lower()
            if file_extension in ('jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov', 'wmv'):
                _run_inference(filepath)

            folder_path = str(_src / 'runs' / 'detect')
            subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
            latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
                
            # get the image path of the detected objects video
            image_path = os.path.join(folder_path, latest_subfolder, predict_img.imgpath)
                
            # render the index.html template with the image_path variable as a parameter
            return render_template('index.html', image_path=image_path)
        else:
            return "Please upload a file."
    # return the index.html template when the HTTP GET method is used
    return render_template('index.html')



def get_frame():
    folder_path = str(_src / 'runs' / 'detect')
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    filename = predict_img.imgpath    
    image_path = folder_path+'/'+latest_subfolder+'/'+filename    
    video = cv2.VideoCapture(image_path)  # detected video path
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)   
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
        time.sleep(0.1)  #control the frame rate to display one frame every 100 milliseconds: 
# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/webcam_feed", methods=['GET'])
def webcam_feed():
    print("here")
    _run_inference('0')  # webcam source
    return "done"

#The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    folder_path = str(_src / 'runs' / 'detect')
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    directory = folder_path+'/'+latest_subfolder
    print("printing directory: ",directory)  
    filename = predict_img.imgpath
    file_extension = filename.rsplit('.', 1)[1].lower()
    #print("printing file extension from display function : ",file_extension)
    environ = request.environ
    if file_extension in ('jpg', 'jpeg', 'png'):
        return send_from_directory(directory, filename, environ)

    elif file_extension in ('mp4', 'avi', 'mov', 'wmv'):
        return render_template('video.html', filename=filename)

    else:
        return "Invalid file format"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLO models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    yolo_model = YOLO(str(_model_path))
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat

