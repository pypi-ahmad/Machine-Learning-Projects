import cv2
import numpy as np

cap = cv2.VideoCapture(0)
path = 'test.png'

def empty(a):
    pass

# Creating Stack of images for presenting on screen
def stackImages(scale, imageArr):
    
    rows_num = len(imageArr)
    Col_num = len(imageArr[0])
    
    rowsAvailable = isinstance(imageArr[0], list)
    
    # Setting up height and width of frame    
    width = imageArr[0][0].shape[1]
    height = imageArr[0][0].shape[0]
    
    
    if rowsAvailable:
        for x in range(0, rows_num):
            for y in range(0, Col_num):
                if imageArr[x][y].shape[:2] == imageArr[0][0].shape[:2]:
                    imageArr[x][y] = cv2.resize(imageArr[x][y], (0, 0), None, scale, scale)
                else:
                    imageArr[x][y] = cv2.resize(imageArr[x][y], (imageArr[0][0].shape[1], imageArr[0][0].shape[0]),None, scale, scale)
                if len(imageArr[x][y].shape) == 2: imageArr[x][y] = cv2.cvtColor(imageArr[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows_num
        hor_con = [imageBlank] * rows_num
        for x in range(0, rows_num):
            hor[x] = np.hstack(imageArr[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows_num):
            if imageArr[x].shape[:2] == imageArr[0].shape[:2]:
                imageArr[x] = cv2.resize(imageArr[x], (0, 0), None, scale, scale)
            else:
                imageArr[x] = cv2.resize(imageArr[x], (imageArr[0].shape[1], imageArr[0].shape[0]), None, scale, scale)
            if len(imageArr[x].shape) == 2: imageArr[x] = cv2.cvtColor(imageArr[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imageArr)
        ver = hor
    return ver


cv2.namedWindow("controllers")
cv2.resizeWindow("controllers", 780, 240)
cv2.createTrackbar("Hue Minimum", "controllers", 0, 179, empty)
cv2.createTrackbar("Hue Maximum", "controllers", 179, 179, empty)
cv2.createTrackbar("Sat Minimum", "controllers", 0, 255, empty)
cv2.createTrackbar("Sat Maximum", "controllers", 255, 255, empty)
cv2.createTrackbar("Val Minimum", "controllers", 0, 255, empty)
cv2.createTrackbar("Val Maximum", "controllers", 255, 255, empty)

while True:
    # img = cv2.imread(path) #for image
    _, img = cap.read()  # for now we're capturing Video from primary webcamera
    
    # Convertiing default inverted colors into HSV form
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #Now geting the trackbar position and using it as value for respective variables
    horMin = cv2.getTrackbarPos("Hue Minimum", "controllers")
    
    horMax = cv2.getTrackbarPos("Hue Maximum", "controllers")
    
    satMin = cv2.getTrackbarPos("Sat Minimum", "controllers")
    
    satMax = cv2.getTrackbarPos("Sat Maximum", "controllers")
    
    valMin = cv2.getTrackbarPos("Val Minimum", "controllers")
    
    valMax = cv2.getTrackbarPos("Val Maximum", "controllers")
    
    print(f'{horMin},{horMax},{satMin},{satMax},{valMin},{valMax}')
    low = np.array([horMin, satMin, valMin])
    up = np.array([horMax, satMax, valMax])
    mask = cv2.inRange(imgHSV, low, up)
    imgRes = cv2.bitwise_and(img, img, mask=mask)

    imgStack = stackImages(0.6, ([img, imgHSV], [mask, imgRes]))
    cv2.imshow("Output of Stacked Images", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()