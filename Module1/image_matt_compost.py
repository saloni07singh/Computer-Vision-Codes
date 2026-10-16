import cv2
import numpy as np

# Read images
foreground = cv2.imread("person.jpg")
background = cv2.imread("background.jpg")

# Resize background
background = cv2.resize(
    background,
    (foreground.shape[1], foreground.shape[0])
)

# Create grayscale image
gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)

# Create foreground mask
_, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# Create inverse mask
mask_inv = cv2.bitwise_not(mask)

# Extract foreground
fg = cv2.bitwise_and(
    foreground,
    foreground,
    mask=mask
)

# Extract background area
bg = cv2.bitwise_and(
    background,
    background,
    mask=mask_inv
)

# Combine both
result = cv2.add(fg, bg)

cv2.imshow("Original", foreground)
cv2.imshow("Mask", mask)
cv2.imshow("Composited Image", result)

cv2.waitKey(0)
cv2.destroyAllWindows()