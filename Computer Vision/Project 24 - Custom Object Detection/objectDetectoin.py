import cv2
# PATH OF THE CASCADE FILES
path = 'haarcascades/haarcascade_frontalface_default.xml'  

obj = 'People'       # OBJECT NAME TO DISPLAY
width_val= 640                     # DISPLAY WIDTH
height_val = 480                  # DISPLAY HEIGHT
col_val= (255,0,255)

#LET'S START CAPTURING VIDEO FROM WEBCAMERA
cap = cv2.VideoCapture(0)
cap.set(3, width_val)
cap.set(4, height_val)

def empty(a):
    pass

# CREATING CONTROLLERS FOR THE FRAME
cv2.namedWindow("OUTPUT")
cv2.resizeWindow("OUTPUT",width_val,height_val+100)
cv2.createTrackbar("scaleValue","OUTPUT",400,1000,empty)
cv2.createTrackbar("Neighbour","OUTPUT",8,50,empty)
cv2.createTrackbar("Min_Area","OUTPUT",0,100000,empty)
cv2.createTrackbar("Brightness","OUTPUT",180,255,empty)

# LOAD THE CLASSIFIERS DOWNLOADED
cascade = cv2.CascadeClassifier(path)

while True:
    # Now we'll take values from the trackbars drawn in "OUTPUT" frame and setting values according to it 
    camBright = cv2.getTrackbarPos("Brightness", "OUTPUT")
    cap.set(10, camBright)
    
    # GET CAMERA IMAGE AND CONVERT TO GRAYscaleValue
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # DETECT THE OBJECT USING THE CASCADE
    scVal =1 + (cv2.getTrackbarPos("scaleValue", "OUTPUT") /1000)
    neigVal=cv2.getTrackbarPos("Neighbour", "OUTPUT")
    objects = cascade.detectMultiScale(gray,scVal, neigVal)
    
    # DISPLAY THE DETECTED OBJECTS
    for (x,y,w,h) in objects:
        area = w*h
        minArea = cv2.getTrackbarPos("Min_Area", "OUTPUT")
        if area >minArea:
            cv2.rectangle(img,(x,y),(x+w,y+h),col_val,3)
            cv2.putText(img,obj,(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,col_val,2)
            roi_col_val = img[y:y+h, x:x+w]

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
         break
