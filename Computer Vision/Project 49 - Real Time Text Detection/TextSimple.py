import cv2
import pytesseract
import numpy as np
from PIL import ImageGrab
import time
import os
import shutil

# Use TESSERACT_CMD env var, or fall back to PATH lookup
_tesseract_cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd
image_val = cv2.imread('1.png')
image_val = cv2.cvtColor(image_val, cv2.COLOR_BGR2RGB)

#detecting Characters
himage_val, wimage_val,_ = image_val.shape
# this function creates imaginary boxes around each text and returns four values for each character
boxes = pytesseract.image_to_boxes(image_val)
for b in boxes.splitlines():
    print(b)
    ##splitting each unit of boxes and saving it in a list
    b = b.split(' ')
    print(b)
    # passing those boxes into variables after type conversion into int.
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    #creating a ractangle on screen
    cv2.rectangle(image_val, (x,himage_val- y), (w,himage_val- h), (50, 50, 255), 2)
    # putting text on rectangle
    cv2.putText(image_val,b[0],(x,himage_val- y+25),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)


cv2.imshow('image_val', image_val)
cv2.waitKey(0)
