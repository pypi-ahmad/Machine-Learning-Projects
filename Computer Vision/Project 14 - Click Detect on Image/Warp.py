# This project deetcts the position of clicks on the image and finally provides a warp perspective after 4 clicks
# let's Starts By Importing required Libraries
import cv2
import numpy as np

circles = np.zeros((4,2),np.int)
counter = 0
#defining a function that determines and draws circle at click points
def mousePoints(event,x,y,flags,params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:

        circles[counter] = x,y
        counter = counter + 1
        print(circles)


#reading image stored in same folder
img = cv2.imread('cards.jpg')
while True:


    if counter == 4:
        width, height = 250,350
        pts1 = np.float32([circles[0],circles[1],circles[2],circles[3]])
        pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        imgOutput = cv2.warpPerspective(img,matrix,(width,height))
        cv2.imshow("Output Image ", imgOutput)


    for x in range (0,4):
        cv2.circle(img,(circles[x][0],circles[x][1]),3,(0,255,0),cv2.FILLED)

    cv2.imshow("Original Image ", img)
    cv2.setMouseCallback("Original Image ", mousePoints)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#destroy all windows
cv2.destroyAllWindows()