import cv2
import numpy as np

# Read image
img = cv2.imread("tiger.jpg")

if img is None:
    print("Error: Image not found!")
    exit()

# Get image dimensions
rows, cols = img.shape[:2]

# 1. Translation

# Shift image by (100 pixels right, 50 pixels down)

M_translate = np.float32([[1, 0, 100],
                          [0, 1, 50]])

translated = cv2.warpAffine(img, M_translate, (cols, rows))

# 2. Rotation

# Rotate image 45 degrees

M_rotate = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)

rotated = cv2.warpAffine(img, M_rotate, (cols, rows))

# 3. Scaling

scaled = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

# 4. Affine Transformation

pts1 = np.float32([[50,50],
                   [200,50],
                   [50,200]])

pts2 = np.float32([[10,100],
                   [200,50],
                   [100,250]])

M_affine = cv2.getAffineTransform(pts1, pts2)

affine = cv2.warpAffine(img, M_affine, (cols, rows))

# 5. Perspective Transformation

pts1 = np.float32([[50,50],
                   [300,50],
                   [50,300],
                   [300,300]])

pts2 = np.float32([[0,0],
                   [300,0],
                   [100,300],
                   [250,300]])

M_perspective = cv2.getPerspectiveTransform(pts1, pts2)

perspective = cv2.warpPerspective(img, M_perspective, (cols, rows))

# Display Results

cv2.imshow("Original", img)
cv2.imshow("Translated", translated)
cv2.imshow("Rotated", rotated)
cv2.imshow("Scaled", scaled)
cv2.imshow("Affine", affine)
cv2.imshow("Perspective", perspective)

cv2.waitKey(0)
cv2.destroyAllWindows()