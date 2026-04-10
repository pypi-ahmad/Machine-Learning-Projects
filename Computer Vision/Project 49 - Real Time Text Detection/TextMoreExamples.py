import cv2
import pytesseract
import numpy as np
from PIL import ImageGrab
import time
import os
import shutil

# download tesseract from this gitrepo - https://github.com/tesseract-ocr/tesseract 
# Set TESSERACT_CMD env var to point to your tesseract executable, or ensure it's on PATH
_tesseract_cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd
image_val = cv2.imread('1.png')
#coverting inverted color into true colors
image_val = cv2.cvtColor(image_val, cv2.COLOR_BGR2RGB) 
pytesseract
# Image to String# Detecting Characters
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
    cv2.rectangle(image_val, (x,himage_val- y), (w,himage_val- h), (50, 50, 255), 2)
    cv2.putText(image_val,b[0],(x,himage_val- y+25),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)

# uncomment the section which you want to apply for instence if you want to apply model to detect digits only the remove # symbol from line 51 to 60 if code is unchanged otherwise use you wisdom 
# Detecting Words
# #[   0          1           2           3           4          5         6       7       8        9        10       11 ]
# #['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height', 'conf', 'text']
# boxes = pytesseract.image_to_data(image_val)
# for a,b in enumerate(boxes.splitlines()):
#         print(b)
#         if a!=0:
#             b = b.split()
#             if len(b)==12:
#                 x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
#                 cv2.putText(image_val,b[11],(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)
#                 cv2.rectangle(image_val, (x,y), (x+w, y+h), (50, 50, 255), 2)


# Detecting ONLY Digits
# himage_val, wimage_val,_ = image_val.shape
# conf = r'--oem 3 --psm 6 outputbase digits'
# boxes = pytesseract.image_to_boxes(image_val,config=conf)
# for b in boxes.splitlines():
#     print(b)
#     b = b.split(' ')
#     print(b)
#     x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
#     cv2.rectangle(image_val, (x,himage_val- y), (w,himage_val- h), (50, 50, 255), 2)
#     cv2.putText(image_val,b[0],(x,himage_val- y+25),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)


# Webcam and Screen Capture Example
# cap = cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
# def captureScreen(bbox=(300,300,1500,1000)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr
# while True:
#     timer = cv2.getTickCount()
#     _,image_val = cap.read()
#     #image_val = captureScreen()
#     #DETECTING CHARACTERES
#     himage_val, wimage_val,_ = image_val.shape
#     boxes = pytesseract.image_to_boxes(image_val)
#     for b in boxes.splitlines():
#         #print(b)
#         b = b.split(' ')
#         #print(b)
#         x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
#         cv2.rectangle(image_val, (x,himage_val- y), (w,himage_val- h), (50, 50, 255), 2)
#         cv2.putText(image_val,b[0],(x,himage_val- y+25),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)
#     fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
#     #cv2.putText(image_val, str(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,230,20), 2);
#     cv2.imshow("Result",image_val)
#     cv2.waitKey(1)
#
#

cv2.imshow('image_val', image_val)
cv2.waitKey(0)
