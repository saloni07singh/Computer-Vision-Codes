import cv2
import numpy as np

# Read images
img1 = cv2.imread("image1.png")
img2 = cv2.imread("image2.png")

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect features using SIFT
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Match features
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Select good matches
good = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

# Get matching points
src_pts = []
dst_pts = []

for m in good:
    src_pts.append(kp1[m.queryIdx].pt)
    dst_pts.append(kp2[m.trainIdx].pt)

# Convert to required format


src_pts = np.float32(src_pts).reshape(-1, 1, 2)
dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

# RANSAC
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Keep only RANSAC inliers
matches_mask = mask.ravel().tolist()

result = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    good, None,
    matchesMask=matches_mask
)

cv2.imshow("RANSAC Feature Matching", result)
cv2.waitKey(0)
cv2.destroyAllWindows()