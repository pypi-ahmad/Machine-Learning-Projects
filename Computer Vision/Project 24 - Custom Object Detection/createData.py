import cv2
import os
import time


path = 'data/images' 
hue = 180
mod_val = 10  # SAVE EVERY ITH FRAME TO AVOID REPETITION
min_blur = 500  # SMALLER VALUE MEANS MORE BLURRINESS PRESENT
gray_img = False # IMAGES SAVED COLORED OR GRAY
save_data = True   # SAVE DATA FLAG
show_img = True  # IMAGE DISPLAY FLAG
img_widht = 180
img_height = 120


global countFolder
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10,hue)


count = 0
count_save =0

def save_dataFunc():
    global countFolder
    countFolder = 0
    while os.path.exists( path+ str(countFolder)):
        countFolder += 1
    os.makedirs(path + str(countFolder))

if save_data:save_dataFunc()


while True:

    success, img = cap.read()
    img = cv2.resize(img,(img_widht,img_height))
    if gray_img:img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if save_data:
        blur = cv2.Laplacian(img, cv2.CV_64F).var()
        if count % mod_val ==0 and blur > min_blur:
            nowTime = time.time()
            cv2.imwrite(path + str(countFolder) +
                    '/' + str(count_save)+"_"+ str(int(blur))+"_"+str(nowTime)+".png", img)
            count_save+=1
        count += 1

    if show_img:
        cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
