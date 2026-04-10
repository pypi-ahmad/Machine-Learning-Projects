import cv2

path = "road.jpg"
img  = cv2.imread(path)
print(img.shape)

width_val ,height_val = 900 , 900
# Using CV resize to resize
imgResize = cv2.resize(img,(width_val,height_val))
print(imgResize.shape)
# Cropping image
imgCropped = img[310:550,440:490]
imCropResize  = cv2.resize(imgCropped,(img.shape[1],img.shape[0]))
# getting Output on screen
cv2.imshow("Orignal",img)
cv2.imshow("First Resize",imgResize)
cv2.imshow("Second Resize",imgCropped)
cv2.imshow("Final Resize",imCropResize)
cv2.waitKey(0)
cv2.destroyAllWindows()
