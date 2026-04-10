#facial landmarking with OpenCV, dlib and python
#getting libraries
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

#construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor",
	default=str(_REPO_ROOT / "models" / "project_04" / "shape_predictor_68_face_landmarks.dat"),
	help = "path to facial landmark predictor")
ap.add_argument("-i", "--image", required = True, help = "path to input image")
args = vars(ap.parse_args())

#initialize dlib's face detector to create the facial landmark predictor

detector = dlib.get_frontal_face_detector()
if not Path(args["shape_predictor"]).exists():
	raise FileNotFoundError(f"[ERROR] shape predictor not found: {args['shape_predictor']}")
predictor = dlib.shape_predictor(args["shape_predictor"])

#loading the input image, resize and grayscaling it
image = cv2.imread(args["image"])
image = imutils.resize(image, width = 500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#detect faces in the grayscale image
rects = detector(gray, 1)

#loop to keep rolling
for(i, rect) in enumerate(rects):
	#determine the facial landmarks for the face region, then
	#convert the facial landmark (x, y) - coordinates to a NumPy array

	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	#convert dlib's rectangle to OpenCV style bounding box [(x, y, w, h)], then draw the face bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	#show the face number
	cv2.putText(image, "Face #{}".format(i+1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	#loop over the (x, y)-coordinates for rhe facial landmarks and draw them on the image

	for (x, y) in shape:
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

#show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()