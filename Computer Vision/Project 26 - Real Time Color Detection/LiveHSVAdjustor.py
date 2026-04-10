import cv2
import numpy as np

widhth_val = 640
hieght_val = 480
cap = cv2.VideoCapture(0)
cap.set(3, widhth_val)
cap.set(4, hieght_val)

def empty(a):
    pass

cv2.namedWindow("Result")
cv2.resizeWindow("Result",640,240)
cv2.createTrackbar("Hue_Minimum","Result",0,179,empty)
cv2.createTrackbar("Hue_Maximum","Result",179,179,empty)
cv2.createTrackbar("Sat_Minimum","Result",0,255,empty)
cv2.createTrackbar("Sat_Maximum","Result",255,255,empty)
cv2.createTrackbar("Val_Min","Result",0,255,empty)
cv2.createTrackbar("Val_Max","Result",255,255,empty)

while True:

    _, img = cap.read()
    imgResult = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #converting bgr img into HSV
    h_min = cv2.getTrackbarPos("Hue_Minimum","Result")
    h_max = cv2.getTrackbarPos("Hue_Maximum", "Result")
    s_min = cv2.getTrackbarPos("Sat_Minimum", "Result")
    s_max = cv2.getTrackbarPos("Sat_Maximum", "Result")
    v_min = cv2.getTrackbarPos("Val_Min", "Result")
    v_max = cv2.getTrackbarPos("Val_Max", "Result")


    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgResult,lower,upper)
    result = cv2.bitwise_and(img,img, mask = mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hStack = np.hstack([img,mask,result])    
    cv2.imshow('Horizontal Stacking', hStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
