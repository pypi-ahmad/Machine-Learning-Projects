import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
from utils.paths import PathResolver

paths = PathResolver()
_models = paths.models("age_gender_recognition")
_src = Path(__file__).resolve().parent


def faceBox(faceNet,frame):
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    bboxs=[]
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)
    return frame, bboxs


faceProto = str(_src / "opencv_face_detector.pbtxt")
faceModel = str(_src / "opencv_face_detector_uint8.pb")

ageProto = str(_src / "age_deploy.prototxt")
ageModel = str(_models / "age_net.caffemodel")

genderProto = str(_src / "gender_deploy.prototxt")
genderModel = str(_models / "gender_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']


def main():
    faceNet=cv2.dnn.readNet(faceModel, faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)

    image=cv2.imread(str(_src / 'test1.jpg'))
    # get original image size
    h, w = image.shape[:2]

    # resize image with a factor of 0.5
    image = cv2.resize(image, (int(w*0.5), int(h*0.5)))

    padding=20

    image,bboxs=faceBox(faceNet,image)
    for bbox in bboxs:
        # face=image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        face = image[max(0,bbox[1]-padding):min(bbox[3]+padding,image.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, image.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPred=genderNet.forward()
        gender=genderList[genderPred[0].argmax()]

        ageNet.setInput(blob)
        agePred=ageNet.forward()
        age=ageList[agePred[0].argmax()]

        label="{},{}".format(gender,age)
        cv2.rectangle(image, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1) 
        cv2.putText(image, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Age-Gender",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
