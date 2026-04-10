import cv2 as cv
import numpy as np
import operator # local module
from keras.models import load_model
from keras.models import model_from_json
import SudokuSolver as sol # local Module

import os
if not os.path.exists("./digit_model.h5"):
    raise FileNotFoundError("[ERROR] digit_model.h5 not found in current directory")
classifier = load_model("./digit_model.h5")
marge = 4
case = 28 + 2 * marge
taille_grille = 9 * case
# capturing Screen
cp = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'XVID')
flag = 0
out = cv.VideoWriter('output.avi', fourcc, 30.0, (1080, 620))

# restraining Screen
while True:

    ret, frame = cp.read()
	# Grayscaling the input
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (7, 7), 0)
    thresh = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 9, 2)
	# detecting Contours
    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contour_grille = None
    maxArea = 0
	# making contour Grill
    for c in contours:
        area = cv.contourArea(c)
        if area > 25000:
            peri = cv.arcLength(c, True)
            polygone = cv.approxPolyDP(c, 0.01 * peri, True)
            if area > maxArea and len(polygone) == 4:
                contour_grille = polygone
                maxArea = area

    if contour_grille is not None:
        cv.drawContours(frame, [contour_grille], 0, (0, 255, 0), 2)
        points = np.vstack(contour_grille).squeeze()
        points = sorted(points, key=operator.itemgetter(1))
        if points[0][0] < points[1][0]:
            if points[3][0] < points[2][0]:
                pts1 = np.float32([points[0], points[1], points[3], points[2]])
            else:
                pts1 = np.float32([points[0], points[1], points[2], points[3]])
        else:
            if points[3][0] < points[2][0]:
                pts1 = np.float32([points[1], points[0], points[3], points[2]])
            else:
                pts1 = np.float32([points[1], points[0], points[2], points[3]])
        pts2 = np.float32([[0, 0], [taille_grille, 0], [0, taille_grille], [
                          taille_grille, taille_grille]])
		# Transforming Image
        M = cv.getPerspectiveTransform(pts1, pts2)
        grille = cv.warpPerspective(frame, M, (taille_grille, taille_grille))
        grille = cv.cvtColor(grille, cv.COLOR_BGR2GRAY)
        grille = cv.adaptiveThreshold(
            grille, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 3)
		# Casting Grills
        cv.imshow("grille", grille)
        if flag == 0:

            grille_txt = []
            for y in range(9):
                ligne = ""
                for x in range(9):
                    y2min = y * case + marge
                    y2max = (y + 1) * case - marge
                    x2min = x * case + marge
                    x2max = (x + 1) * case - marge
                    cv.imwrite("mat" + str(y) + str(x) + ".png",
                                grille[y2min:y2max, x2min:x2max])
                    img = grille[y2min:y2max, x2min:x2max]
                    x = img.reshape(1, 28, 28, 1)
                    if x.sum() > 10000:
                        prediction = np.argmax(classifier.predict(x), axis=1)
                        ligne += "{:d}".format(prediction[0])
                    else:
                        ligne += "{:d}".format(0)
                grille_txt.append(ligne)
            print(grille_txt)
            result = sol.sudoku(grille_txt)
        print("Resultat:", result)

        if result is not None:
            flag = 1
            fond = np.zeros(
                shape=(taille_grille, taille_grille, 3), dtype=np.float32)
            for y in range(len(result)):
                for x in range(len(result[y])):
                    if grille_txt[y][x] == "0":
                        cv.putText(fond, "{:d}".format(result[y][x]), ((
                            x) * case + marge + 3, (y + 1) * case - marge - 3), cv.FONT_HERSHEY_SCRIPT_COMPLEX, 0.9, (0, 0, 255), 1)
            M = cv.getPerspectiveTransform(pts2, pts1)
            h, w, c = frame.shape
            fondP = cv.warpPerspective(fond, M, (w, h))
            i2g = cv.cvtColor(fondP, cv.COLOR_BGR2GRAY)
            ret, mask = cv.threshold(i2g, 10, 255, cv.THRESH_BINARY)
            mask = mask.astype('uint8')
            mask_inv = cv.bitwise_not(mask)
            img1_bg = cv.bitwise_and(frame, frame, mask=mask_inv)
            img2_fg = cv.bitwise_and(fondP, fondP, mask=mask).astype('uint8')
            dst = cv.add(img1_bg, img2_fg)
            dst = cv.resize(dst, (1080, 620))
            cv.imshow("frame", dst)
            out.write(dst)

        else:
            frame = cv.resize(frame, (1080, 620))
            cv.imshow("frame", frame)
            out.write(frame)

    else:
        flag = 0
        frame = cv.resize(frame, (1080, 620))
        cv.imshow("frame", frame)
        out.write(frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break


out.release()
cp.release()
cv.destroyAllWindows()
