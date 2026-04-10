import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import imutils
from utils.paths import PathResolver

paths = PathResolver()

def main():
    # initializing the HOG person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Load the video
    cap = cv2.VideoCapture(str(paths.data("pedestrian_detection") / "vid.mp4"))

    while cap.isOpened():
        ret, image = cap.read()
        if ret:
            image = imutils.resize(image, width=min(400, image.shape[1]))
            # detect all the regions that have pedestrians
            (regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)
            # draw the regions in the video
            for (x, y, w, h) in regions:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow("Image", image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
