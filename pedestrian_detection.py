import cv2
# Create HOG Descriptor
hog = cv2.HOGDescriptor()
# Load pre-trained people detector
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# Read image
img = cv2.imread("pedest.jpg")
# Detect pedestrians
boxes, weights = hog.detectMultiScale(
    img,
    winStride=(8,8),
    padding=(8,8),
    scale=1.05
)
# Draw rectangles
for (x, y, w, h) in boxes:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
cv2.imshow("Pedestrian Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
