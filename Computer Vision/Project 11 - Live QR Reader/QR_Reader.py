import cv2 as cv
import numpy as np
from pyzbar.pyzbar import decode
    
#img = cv.imread(args["image"])
cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
#taking  list of autorized people from files
with open('myDataFile.txt') as f:
    myDataList = f.read().splitlines()

while True:

    success, img = cap.read()
    for barcode in decode(img):
		#decoding barcode using pyzbar module
        myData = barcode.data.decode('utf-8')
        print(myData)

        if myData in myDataList:
            myOutput = 'Authorized'
            myColor = (0,255,0)
        else:
            myOutput = 'Un-Authorized'
            myColor = (0, 0, 255)
		#putting lines and text near QR-Code
        pts = np.array([barcode.polygon],np.int32)
        pts = pts.reshape((-1,1,2))
        cv.polylines(img,[pts],True,myColor,5)
        pts2 = barcode.rect
        cv.putText(img,myOutput,(pts2[0],pts2[1]),cv.FONT_HERSHEY_SIMPLEX,
                    0.9,myColor,2)

    cv.imshow('Result',img)
    cv.waitKey(1)