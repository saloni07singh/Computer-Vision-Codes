import cv2

# Read grayscale image
gray = cv2.imread("grey.jpeg", cv2.IMREAD_GRAYSCALE)

# Apply color map
color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

# Display
cv2.imshow("Grayscale Image", gray)
cv2.imshow("Colorized Image", color)

cv2.waitKey(0)
cv2.destroyAllWindows()