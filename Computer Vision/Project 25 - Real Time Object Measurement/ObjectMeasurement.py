import cv2
import utlis

webcam = True
path = '1.jpg'
cap = cv2.VideoCapture(0)
cap.set(10,160)
cap.set(3,1920)
cap.set(4,1080)
scale_val = 3
width = 210 *scale_val
hieght= 297 *scale_val

while True:
    if webcam:_,img = cap.read()
    else: img = cv2.imread(path)

    imgContours , conts = utlis.getContours(img,minArea=50000,filter=4)
    if len(conts) != 0:
        biggest = conts[0][2]
        
        
        imgWarp = utlis.warpImg(img, biggest, width,hieght)
        
        imgContours2, conts2 = utlis.getContours(imgWarp,minArea=2000, filter=4,cThr=[50,50],draw = False)
        if len(conts2) != 0:
            for obj in conts2:
            # ordering contours using utils
                cv2.polylines(imgContours2,[obj[2]],True,(0,255,0),2)
                nPoints = utlis.reorder(obj[2])
                
                
                # Now Using our utils module to calculate distance
                nW = round((utlis.findDis(nPoints[0][0]//scale_val,nPoints[1][0]//scale_val)/10),1)
                nH = round((utlis.findDis(nPoints[0][0]//scale_val,nPoints[2][0]//scale_val)/10),1)
                
                # Drawing lines on screen
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),(255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),(255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,(255, 0, 255), 2)
                cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,(255, 0, 255), 2)
        cv2.imshow('A4', imgContours2)

    img = cv2.resize(img,(0,0),None,0.5,0.5)
    cv2.imshow('Original',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()