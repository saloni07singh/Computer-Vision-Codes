import cv2
import numpy as np
import glob

# Real-world coordinates of chessboard corners
objp = np.zeros((54,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []

# Read all chessboard images
images = glob.glob("images/*.jpg")
print("Images Found:", len(images))
print(images)
for file in images:

    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray,(9,6))

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

# Camera Calibration
ret, cameraMatrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("Camera Matrix")
print(cameraMatrix)

print("\nDistortion Coefficients")
print(distortion)