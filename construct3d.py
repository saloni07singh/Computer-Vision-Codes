import cv2
import numpy as np

# Read two images
img1 = cv2.imread("image1.png")
img2 = cv2.imread("image2.png")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and features
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print("Keypoints in Image 1:", len(kp1))
print("Keypoints in Image 2:", len(kp2))

# Match features
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Select good matches
good = []

for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

print("Good Matches:", len(good))

# Get matching points
points1 = np.float32(
    [kp1[m.queryIdx].pt for m in good]
)

points2 = np.float32(
    [kp2[m.trainIdx].pt for m in good]
)

# Camera matrices
K = np.array([
    [1000, 0, img1.shape[1] / 2],
    [0, 1000, img1.shape[0] / 2],
    [0, 0, 1]
], dtype=float)

print("points1 shape:", points1.shape)
print("points2 shape:", points2.shape)
print("points1 dtype:", points1.dtype)
print("points2 dtype:", points2.dtype)

# Find essential matrix
E, mask = cv2.findEssentialMat(
    points1, points2, K,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0
)

# Recover camera rotation and translation
_, R, t, mask = cv2.recoverPose(
    E, points1, points2, K
)

# First camera position
P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

# Second camera position
P2 = K @ np.hstack((R, t))

# Triangulate 3D points
points4D = cv2.triangulatePoints(
    P1, P2, points1, points2
)

# Convert from homogeneous coordinates
points3D = points4D[:3] / points4D[3]

print("3D Model Created")
print("Number of 3D Points:", points3D.shape[1])

# Display matching points
result = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    good[:50], None,
    flags=2
)

cv2.imshow("Feature Matching", result)
cv2.waitKey(0)
cv2.destroyAllWindows()