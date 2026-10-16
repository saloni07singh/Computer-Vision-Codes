import cv2
import easyocr
import numpy as np

# Create OCR reader
reader = easyocr.Reader(['en'])

# Read image
image = cv2.imread("text-detection-feature.jpg")

# Detect and recognize text
results = reader.readtext(image)

# Process results
for box, text, confidence in results:

    print("Text:", text)
    print("Confidence:", confidence)

    # Convert points to integers
    points = np.array(box, dtype=np.int32)

    # Draw box
    cv2.polylines(image, [points], True, (0, 255, 0), 2)

    # Display detected text
    x, y = points[0]
    cv2.putText(image, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2)

# Show result
cv2.imshow("Text Detection", image)

cv2.waitKey(0)
cv2.destroyAllWindows()