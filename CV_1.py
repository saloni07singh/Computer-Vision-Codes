import numpy as np
import cv2 as cv

img = cv.imread("vishnu.jpg")
if img is None:
    print("Image not found")
    exit()

cv.imshow("Image", img)

cols, rows = img.shape[:2]
abc = np.float32([[1, 0, 100],
                [0, 1, 200]])

m_translate = cv.warpAffine(img, abc, (cols, rows))

cv.imshow("Translated Image", m_translate)

A = cv.getRotationMatrix2D((cols/2, rows/2), 45, 1)

rotated = cv.warpAffine(img, A, (cols, rows))

cv.imshow("Rotated Image", rotated)

cv.waitKey(0)
cv.destroyAllWindows()